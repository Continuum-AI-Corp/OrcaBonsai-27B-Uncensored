"""Speculative decoding with the DFlash 2 drafter against the pack.

One round: the drafter proposes a block of 7 tokens after the pending token in a
single forward over ``[pending, mask x7]``, conditioned on the target's residual stream
at five layers; the target then runs one forward over ``[pending] + drafts`` (8 rows,
through the tile matmul in ``mma.py``), the longest prefix of drafts that matches the
target's own argmax is accepted, the target's prediction after it is a bonus token, and
the caches are rolled back to exactly the tokens accepted (attention caches are
trimmed; the linear-attention layers replay the accepted prefix through the fused kernel
from their recorded inputs, see ``fused.rollback``). Greedy only: the output is what plain greedy decoding would
produce, up to kernel rounding at near-ties.

Why it only pays sometimes. A round costs a draft (~40 ms) plus a verify (~115 ms on an
M1 Ultra) against a ~37 ms plain step, so it needs about 4 committed tokens per round
to break even. Code and structured text get 5-8; chat gets 2-3. ``Speculator.round``
reports each round's yield and ``run.py`` falls back to plain decoding for the rest of
the turn when the running average is below ``min_gain``.

Copy drafts. Before calling the drafter, a round looks the last four tokens up in the
conversation so far (:class:`NGramIndex`); if they occurred before and a full block of
tokens followed them, that block is the draft. It costs nothing to produce and, on
replies that re-emit code from the context, is accepted whole. The verify and
rollback are the same, so it is just as lossless as a drafter round.

Context across turns. The drafter's caches hold projected rows for every token the
target has consumed (positions must stay contiguous for RoPE), and the rows for the
tokens committed in the last round are carried as ``pending_ctx`` until the next draft
appends them. A turn that ends in plain decoding leaves the drafter without rows for
those tokens; the next prefill then rebuilds fresh drafter caches starting at the new
rows' position, and drafting recovers as rounds append more.
"""
from __future__ import annotations

import mlx.core as mx

from . import fused


class NGramIndex:
    """Where each n-gram of the conversation last occurred, for copy drafts.

    A coding reply mostly re-emits text that is already in the context (the file it
    was shown, the function it wrote a turn ago), so the continuation after the last
    n tokens is very often the continuation after their previous occurrence. That
    continuation is a free draft: no drafter forward, and long runs of it are
    accepted whole. n = 4 by default; shorter n-grams fire spuriously on prose and
    waste a round.
    """

    def __init__(self, n: int = 4):
        self.n = int(n)
        self.seq: list[int] = []
        self.last: dict[tuple, int] = {}     # position right after the latest occurrence
        self.prev: dict[tuple, int] = {}     # ... and after the one before it

    def extend(self, tokens):
        for t in tokens:
            self.seq.append(int(t))
            if len(self.seq) >= self.n:
                key = tuple(self.seq[-self.n:])
                if key in self.last:
                    self.prev[key] = self.last[key]
                self.last[key] = len(self.seq)

    def propose(self, cap: int) -> list[int]:
        """Tokens that followed the most recent earlier occurrence of the last n."""
        if len(self.seq) < self.n:
            return []
        key = tuple(self.seq[-self.n:])
        pos = self.last.get(key)
        if pos == len(self.seq):             # that is the key itself, at the end
            pos = self.prev.get(key)
        if pos is None:
            return []
        return self.seq[pos: pos + cap]


class Speculator:
    def __init__(self, model, drafter, stops, min_gain: float = 3.5, warmup: int = 10,
                 lookup_n: int = 4, lookup_min: int = 7):
        self.model = model
        self.lm = getattr(model, "language_model", model)
        self.drafter = drafter
        self.stops = set(stops)
        self.tap = list(drafter.config.target_layer_ids)
        self.bs = int(drafter.config.block_size)
        self.mask_id = int(drafter.config.mask_token_id)
        self.min_gain = float(min_gain)
        self.warmup = int(warmup)
        self.lookup_n = int(lookup_n)
        self.lookup_min = int(lookup_min)      # copy drafts shorter than this go to the drafter
        self.reset()

    def reset(self):
        self.dcache = None
        self.pending_ctx = None
        self.stale = True
        self.committed_lengths: list[int] = []
        self.recent: list[int] = []
        self.index = NGramIndex(self.lookup_n) if self.lookup_n else None
        self.lookup_rounds = 0
        self.lookup_tokens = 0

    def begin_turn(self):
        """A new reply gets a fresh chance at speculation whatever the last one did."""
        self.recent = []

    # -- target forwards -------------------------------------------------------------

    def prefill(self, ids, cache, start_pos: int, index: bool = True):
        """Run the target over ``ids`` (which its cache has not seen), capturing the
        drafter's context rows. Returns the last position's logits."""
        sink: list = []
        hn = self.lm.model(mx.array([list(ids)], dtype=mx.int32), cache=cache,
                           capture_layer_ids=self.tap, hidden_sink=sink)
        logits = self.lm.lm_head(hn[:, -1:])
        rows = mx.concatenate(sink, axis=-1)
        if index and self.index is not None:
            self.index.extend(ids)
        if self.dcache is None or self.stale:
            self.dcache = self.drafter.make_cache()
            for c in self.dcache:
                c.offset = int(start_pos)
            self.pending_ctx = rows
            self.stale = False
        else:
            self.pending_ctx = rows if self.pending_ctx is None else mx.concatenate(
                [self.pending_ctx, rows], axis=1)
        return logits[0, -1]

    def consume(self, token: int, cache):
        """Feed one token through the target (plain step), keeping the drafter's rows."""
        return self.prefill([token], cache, 0, index=False)   # already indexed as pending

    # -- one speculative round --------------------------------------------------------

    def round(self, pending: int, cache, limit: int = 8):
        """Draft, verify, accept, roll back.

        Returns ``(committed, consumed, stopped)``: ``committed`` are the new output
        tokens (accepted drafts plus the target's bonus token, cut at the first stop
        token and to at most ``limit`` tokens), ``consumed`` the tokens the target cache
        has now seen (``pending`` plus the accepted drafts), ``stopped`` whether a stop
        token was produced. The next pending token is ``committed[-1]`` unless stopped.
        """
        cap = self.bs - 1
        copied = []
        if self.index is not None:
            if not self.index.seq or self.index.seq[-1] != pending:
                self.index.extend([pending])
            copied = self.index.propose(cap)
        if len(copied) >= self.lookup_min:
            # free draft from the context; the drafter's context rows pile up in
            # pending_ctx until its next call appends them
            draft = mx.array(copied, dtype=mx.int32)
            from_lookup = True
        else:
            block = mx.array([[pending] + [self.mask_id] * cap], dtype=mx.int32)
            draft = self.drafter.select_block(block, self.pending_ctx, self.dcache,
                                              cap=cap, anchor_id=pending)[0]
            from_lookup = False
        verify_ids = mx.concatenate([mx.array([pending], dtype=draft.dtype), draft]).reshape(1, -1)
        cap = int(draft.shape[0])
        # The target forward over [pending] + drafts. Hidden states are captured for the
        # drafter; the linear-attention layers run the fused kernel with T=8 and record
        # their inputs so the rollback below can replay the accepted prefix.
        sink: list = []
        fused.RECORD = True
        try:
            hn = self.lm.model(verify_ids, cache=cache, capture_layer_ids=self.tap, hidden_sink=sink)
        finally:
            fused.RECORD = False
        logits = self.lm.lm_head(hn)
        tt = mx.argmax(logits[0], axis=-1)
        match = (draft == tt[:cap]).astype(mx.int32)
        n_arr = mx.cumprod(match).sum()
        mx.eval(n_arr, tt, draft)
        n = int(n_arr.item())
        drafts = draft.tolist()
        targets = tt.tolist()
        committed = (drafts[:n] + [targets[n]])[: max(1, limit)]

        stopped = False
        accepted = len(committed) - 1      # drafts consumed; the last committed token is pending
        for j, t in enumerate(committed):
            if t in self.stops:
                committed = committed[: j + 1]
                accepted = j          # the stop token itself stays unconsumed
                stopped = True
                break

        fused.rollback(self.model, cache, cap + 1, accepted + 1)
        rows = mx.concatenate(sink, axis=-1)[:, : accepted + 1]
        if from_lookup and self.pending_ctx is not None:
            self.pending_ctx = mx.concatenate([self.pending_ctx, rows], axis=1)
        else:
            self.pending_ctx = rows
        consumed = [pending] + drafts[:accepted]
        if self.index is not None:
            self.index.extend(committed)
        if from_lookup:
            self.lookup_rounds += 1
            self.lookup_tokens += len(committed)
        self.committed_lengths.append(len(committed))
        self.recent.append(len(committed))
        return committed, consumed, stopped

    def should_give_up(self) -> bool:
        """True when this reply's rounds so far average below ``min_gain`` tokens.

        The whole reply's mean, not a sliding window: acceptance swings between
        stretches of a reply (a code block drafts well, the prose around it does not),
        and a window gave up on replies that were paying overall.
        """
        k = self.recent
        if len(k) < self.warmup:
            return False
        return sum(k) / len(k) < self.min_gain

    def stats(self) -> str:
        k = self.committed_lengths
        if not k:
            return "no speculative rounds"
        out = f"{len(k)} rounds, {sum(k) / len(k):.2f} tokens/round"
        if self.lookup_rounds:
            out += (f", {self.lookup_rounds} copied from context at "
                    f"{self.lookup_tokens / self.lookup_rounds:.2f}")
        return out
