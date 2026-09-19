#!/usr/bin/env python3
"""Run Bonsai 2 27B with the refusal direction projected out at run time.

    python run.py --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit "your prompt"
    python run.py --pack ... --interactive
    python run.py --pack ... --alpha 0 "your prompt"     # ablation off, for comparison

The pack's weights are never modified. See bonsai_abliterate/ablation.py for why the
projection is applied at run time rather than baked into the weights.
"""
from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# MLX commits a Metal command buffer every 50 dispatches or 50 MB of distinct weight
# buffers on Ultra-class chips, and each boundary costs ~20 us of GPU idle plus a CPU
# completion handler. A decode step here is ~1,500 dispatches over 7.2 GB of weights,
# so the defaults cut it into well over a hundred buffers. Raising the limits measured
# 27.3 -> 29.1 tok/s on plain decoding on an M1 Ultra (mlx issue 4521 reports 5-8% on
# an M3 Ultra). Must be set before mlx is imported; an explicit value in the
# environment wins.
os.environ.setdefault("MLX_MAX_OPS_PER_BUFFER", "2000")
os.environ.setdefault("MLX_MAX_MB_PER_BUFFER", "2000")

import mlx.core as mx

from bonsai_abliterate.pack import eos_ids, load_pack, load_tokenizer, render_chat
from bonsai_abliterate.ablation import install, load_direction
from bonsai_abliterate import fused, mma, dflash, spec

DEFAULT_DIRECTION = Path(__file__).resolve().parent / "directions" / "refusal_dir.safetensors"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate from the Bonsai ternary pack with runtime refusal ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("prompts", nargs="*", help="prompts to answer; omit with --interactive")
    p.add_argument("--pack", required=True, help="path to the Ternary-Bonsai-2-27B-mlx-2bit directory")
    p.add_argument("--direction", default=str(DEFAULT_DIRECTION), help="refusal direction safetensors")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="projection strength; 1.0 matches a full weight edit, 0 disables it")
    p.add_argument("--max-new", type=int, default=4096,
                   help="generation budget in tokens; a reply that hits it is cut off "
                        "mid-sentence and flagged")
    p.add_argument("--temp", type=float, default=0.0, help="sampling temperature; 0 is greedy")
    p.add_argument("--top-p", type=float, default=0.95, help="nucleus cutoff when --temp > 0")
    p.add_argument("--thinking", action="store_true",
                   help="let the model reason at length before answering")
    p.add_argument("--layers", default="",
                   help="comma-separated layer indices to ablate; empty means all")
    p.add_argument("--draft", default="",
                   help="directory holding a DFlash 2 drafter (dflash/config.json and "
                        "dflash/model.safetensors); enables speculative decoding for greedy runs")
    p.add_argument("--spec-min-gain", type=float, default=3.5,
                   help="fall back to plain decoding for the rest of a reply once ten or more "
                        "rounds have averaged fewer tokens than this per round (a round costs "
                        "about 3.7 plain steps on an M1 Ultra)")
    p.add_argument("--lookup", action="store_true",
                   help="in speculative mode, draft by copying the continuation of the last "
                        "4-gram's earlier occurrence when a full block is available (measured "
                        "no gain over the drafter on this pack; see README)")
    p.add_argument("--no-fused", action="store_true",
                   help="run the pack runtime's own per-op decode path instead of the fused "
                        "linear-attention kernel (slower; useful for A/B checks)")
    p.add_argument("--interactive", action="store_true", help="read prompts from stdin in a loop")
    p.add_argument("--quiet", action="store_true", help="print only the replies")
    return p.parse_args(argv)


def sample(logits, temp, top_p):
    """Greedy when temp is 0, otherwise temperature + nucleus sampling."""
    if temp <= 0:
        return mx.argmax(logits)
    logits = logits / temp
    if not 0 < top_p < 1:
        return mx.random.categorical(logits)
    # Sample in descending-probability space, then map the choice back through the
    # sort order: simpler and cheaper than scattering a mask into vocab order.
    probs = mx.softmax(logits)
    order = mx.argsort(-probs)
    ordered = probs[order]
    keep = (mx.cumsum(ordered) - ordered) < top_p
    choice = mx.random.categorical(mx.log(mx.where(keep, ordered, 0.0) + 1e-30))
    return order[choice]


def generate(language_model, tok, ids, stops, max_new, temp, top_p, stream=True,
             cache=None):
    """Decode until a stop token or the budget.

    Returns ``(text, seconds, n, truncated, consumed)``. ``truncated`` is True when the
    budget ran out before a stop token; the caller should say so, because a reply cut
    off mid-sentence with no marker looks like the model broke when it only ran out of
    room. ``consumed`` is every token the cache has now seen: ``ids``, the reply, and
    the stop token when there was one (the lookahead below feeds it before the loop
    can see it). A caller that keeps ``cache`` for the next turn needs that list to
    know what the cache already holds.

    ``cache`` is the model's own cache list to continue from, with ``ids`` being only
    the tokens it has not seen yet; None starts fresh.
    """
    if cache is None:
        cache = language_model.make_cache() if hasattr(language_model, "make_cache") else None
    produced, t0 = [], time.time()

    def step(x):
        out = language_model(x, cache=cache)
        return sample(out.logits[0, -1].astype(mx.float32), temp, top_p).astype(mx.int32)

    # Decode is kernel-launch bound at batch 1, so keep the GPU fed: queue the next
    # token's forward pass before reading this token back. Waiting on .item() and then
    # building the next graph left the GPU idle between steps and cost ~15% here.
    # The final lookahead runs one step past the stop token; the cache is per call, so
    # that is harmless.
    token = step(mx.array([ids], dtype=mx.int32))
    mx.async_eval(token)
    truncated = True
    consumed = list(ids)
    for _ in range(max_new):
        following = step(token.reshape(1, 1))
        mx.async_eval(following)
        current = int(token.item())
        consumed.append(current)
        if current in stops:
            truncated = False
            break
        produced.append(current)
        if stream:
            sys.stdout.write(tok.decode([current]))
            sys.stdout.flush()
        token = following
    if stream:
        sys.stdout.write("\n")
    elapsed = time.time() - t0
    return tok.decode(produced), elapsed, len(produced), truncated, consumed


def generate_spec(speculator, tok, ids, stops, max_new, cache, start_pos, stream=True):
    """Speculative counterpart of generate(): same return tuple.

    ``ids`` are the tokens the cache has not seen; ``start_pos`` how many it has. Falls
    back to plain decoding for the rest of the reply when rounds stop paying, and to
    finish, feeds the last pending token so the cache holds every token of the reply
    (the same invariant generate() keeps), except a stop token, which stays unconsumed.
    """
    lm = speculator.lm
    t0 = time.time()
    speculator.begin_turn()
    pending = int(mx.argmax(speculator.prefill(ids, cache, start_pos)).item())
    produced, consumed = [], list(ids)
    truncated, stopped = True, False
    if pending in stops:
        stopped, truncated = True, False
    else:
        # the first token comes from the prefill; every later one from a round
        produced.append(pending)
        if stream:
            sys.stdout.write(tok.decode([pending]))
            sys.stdout.flush()
    while not stopped and len(produced) < max_new:
        if speculator.should_give_up():
            break
        committed, used, stopped = speculator.round(pending, cache, max_new - len(produced))
        consumed += used
        for t in committed:
            if t in stops:
                break
            produced.append(t)
            if stream:
                sys.stdout.write(tok.decode([t]))
                sys.stdout.flush()
        if stopped:
            truncated = False
            break
        pending = committed[-1]
    # ``pending`` is already in ``produced`` (it was the last committed token, or the
    # prefill's token) but the cache has not seen it yet.
    if not stopped and len(produced) < max_new:
        # rounds stopped paying: plain decoding from the pending token onwards. The
        # drafter misses the rows for these tokens, so its context is rebuilt next turn.
        speculator.stale = True
        _, _, _, truncated, rest = generate(lm, tok, [pending], stops, max_new - len(produced),
                                            0.0, 1.0, stream=stream, cache=cache)
        consumed += rest                      # [pending] + the plain reply (+ stop token)
        produced += [t for t in rest[1:] if t not in stops]
        return tok.decode(produced), time.time() - t0, len(produced), truncated, consumed
    if not stopped and pending not in stops:
        # budget reached: feed the pending token so the cache holds every reply token
        speculator.consume(pending, cache)
        consumed.append(pending)
    if stream:
        sys.stdout.write("\n")
    return tok.decode(produced), time.time() - t0, len(produced), truncated, consumed


def main(argv=None):
    args = parse_args(argv)
    if not args.prompts and not args.interactive:
        print("nothing to do: pass prompts or --interactive", file=sys.stderr)
        return 2

    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))

    log(f"[load] {args.pack}")
    t0 = time.time()
    model, config = load_pack(args.pack)
    tok = load_tokenizer(args.pack)
    stops = eos_ids(args.pack, config, tok)
    log(f"[load] {time.time()-t0:.1f}s, stop tokens {sorted(stops)}")

    if args.alpha != 0:
        direction, meta = load_direction(args.direction)
        layers = [int(x) for x in args.layers.split(",") if x.strip()] or None
        wrapped = install(model, config, direction, alpha=args.alpha, layers=layers)
        log(f"[ablate] alpha={args.alpha}, {len(wrapped)} residual writers wrapped")
    else:
        log("[ablate] disabled (alpha=0), running the pack unmodified")

    if not args.no_fused:
        log(f"[fused] {fused.install(model)} linear-attention layers decode through one kernel")

    language_model = model.language_model

    speculator = None
    if args.draft:
        if args.temp > 0:
            log("[spec] sampling requested; speculative decoding is greedy-only, disabled")
        else:
            drafter = dflash.load_drafter(args.draft, model)
            mma.install()
            mma.install_drafter(drafter)
            speculator = spec.Speculator(model, drafter, stops, min_gain=args.spec_min_gain,
                                         lookup_n=4 if args.lookup else 0)
            log(f"[spec] drafter loaded: block {speculator.bs}, taps {speculator.tap}")

    history: list[dict] = []
    # The cache carried between turns, and the token ids it has already consumed.
    # Re-tokenising the rendered history would not reproduce those ids: a model's own
    # output is not the canonical BPE tokenisation of its text (a 1220-token Chinese
    # reply re-tokenised to 1221 and diverged at the first word). So the next turn is
    # built as the tokens the cache has seen plus only the text the template appends
    # after them, which is what the model would have seen had the whole thing been
    # rendered at once. Prefill runs at ~80-100 tok/s here, so without this a 4K-token
    # history would cost the better part of a minute before each reply.
    session = {"ids": [], "cache": None}
    im_end = tok.token_to_id("<|im_end|>")

    def continuation(prompt: str):
        """Tokens to feed on top of the session cache for a new user turn, or None."""
        if session["cache"] is None:
            return None
        before = render_chat(args.pack, history, enable_thinking=args.thinking,
                             add_generation_prompt=False)
        after = render_chat(args.pack, history + [{"role": "user", "content": prompt}],
                            enable_thinking=args.thinking)
        cut = before.rfind("<|im_end|>")
        if cut < 0 or not after.startswith(before):
            return None
        delta = after[cut:]
        if session["ids"] and session["ids"][-1] == im_end:
            # The cache already holds the turn terminator (generation stopped on it).
            delta = delta[len("<|im_end|>"):]
        return tok.encode(delta, add_special_tokens=False).ids

    def answer(prompt: str, remember: bool) -> str:
        turns = history + [{"role": "user", "content": prompt}]
        new_ids = continuation(prompt) if remember else None
        if new_ids:
            cache, seen = session["cache"], len(session["ids"])
            log(f"\n>>> {prompt}\n[{seen + len(new_ids)} prompt tokens, {seen} cached, "
                f"{len(new_ids)} to prefill]")
        else:
            cache = language_model.make_cache()
            new_ids = tok.encode(render_chat(args.pack, turns, enable_thinking=args.thinking),
                                 add_special_tokens=False).ids
            session["ids"] = []
            log(f"\n>>> {prompt}\n[{len(new_ids)} prompt tokens]")

        if speculator is not None:
            if cache is None or not session["ids"]:
                speculator.reset()
            reply, elapsed, n, truncated, consumed = generate_spec(
                speculator, tok, new_ids, stops, args.max_new, cache, len(session["ids"]))
            log(f"[{n} tokens in {elapsed:.1f}s = {n/max(elapsed,1e-9):.2f} tok/s; "
                f"spec: {speculator.stats()}]")
        else:
            reply, elapsed, n, truncated, consumed = generate(
                language_model, tok, new_ids, stops, args.max_new, args.temp, args.top_p,
                cache=cache)
            log(f"[{n} tokens in {elapsed:.1f}s = {n/max(elapsed,1e-9):.2f} tok/s]")
        if truncated:
            # Always shown, even with --quiet: the cut-off reply is otherwise
            # indistinguishable from the model stopping on its own.
            print(f"[cut off at --max-new {args.max_new}; raise it for a complete reply]",
                  file=sys.stderr, flush=True)
        if remember:
            history.append({"role": "user", "content": prompt})
            history.append({"role": "assistant", "content": reply})
            session["ids"] = session["ids"] + consumed
            session["cache"] = cache
        return reply

    # Prompts given on the command line are independent questions; only --interactive
    # is a conversation.
    for prompt in args.prompts:
        answer(prompt, remember=False)

    if args.interactive:
        log("\n[chat] one message per line. /reset clears the history, Ctrl-D exits.")
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line in ("/reset", "/clear"):
                history.clear()
                session["ids"], session["cache"] = [], None
                log("[chat] history cleared")
                continue
            answer(line, remember=True)
            log(f"[chat] {len(history) // 2} turns in history")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
