#!/usr/bin/env python3
"""Run Bonsai 2 27B with the refusal direction projected out at run time.

    python run.py --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit "your prompt"
    python run.py --pack ... --interactive
    python run.py --pack ... --alpha 0 "your prompt"     # ablation off, for comparison

The pack's weights are never modified. See bonsai_abliterate/ablation.py for why the
projection is applied at run time rather than baked into the weights.
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mlx.core as mx

from bonsai_abliterate.pack import eos_ids, load_pack, load_tokenizer, render_chat
from bonsai_abliterate.ablation import install, load_direction

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

    language_model = model.language_model

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
