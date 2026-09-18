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
    p.add_argument("--max-new", type=int, default=256, help="maximum tokens to generate")
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


def generate(language_model, tok, ids, stops, max_new, temp, top_p, stream=True):
    cache = language_model.make_cache() if hasattr(language_model, "make_cache") else None
    produced, t0 = [], time.time()
    prompt = mx.array([ids], dtype=mx.int32)
    for _ in range(max_new):
        out = language_model(prompt, cache=cache)
        token = int(sample(out.logits[0, -1].astype(mx.float32), temp, top_p).item())
        if token in stops:
            break
        produced.append(token)
        if stream:
            sys.stdout.write(tok.decode([token]))
            sys.stdout.flush()
        prompt = mx.array([[token]], dtype=mx.int32)
    if stream:
        sys.stdout.write("\n")
    elapsed = time.time() - t0
    return tok.decode(produced), elapsed, len(produced)


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

    def answer(prompt: str):
        ids = tok.encode(render_chat(args.pack, prompt, enable_thinking=args.thinking),
                         add_special_tokens=False).ids
        log(f"\n>>> {prompt}\n[{len(ids)} prompt tokens]")
        _, elapsed, n = generate(language_model, tok, ids, stops,
                                 args.max_new, args.temp, args.top_p)
        log(f"[{n} tokens in {elapsed:.1f}s = {n/max(elapsed,1e-9):.2f} tok/s]")

    for prompt in args.prompts:
        answer(prompt)

    if args.interactive:
        log("\n[interactive] one prompt per line, Ctrl-D to exit")
        for line in sys.stdin:
            line = line.strip()
            if line:
                answer(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
