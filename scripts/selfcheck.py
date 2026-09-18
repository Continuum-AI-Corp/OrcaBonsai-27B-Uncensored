#!/usr/bin/env python3
"""Confirm the projection is actually in force on your copy of the pack.

Reports, for a few layers, how much of the residual stream still lies along the refusal
direction -- before and after the ablation is installed. A correct setup drives it to
roughly 1e-6 of the residual norm. A value that barely moves means the writers were not
wrapped (most often: the pack loaded through something other than its own runtime, or a
direction of the wrong dimensionality).

    python scripts/selfcheck.py --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit

Each forward pass costs minutes without Metal, so this checks one short prompt.
"""
from __future__ import annotations

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonsai_abliterate.pack import load_pack, load_tokenizer, render_chat
from bonsai_abliterate.ablation import install, load_direction, residual_components

DEFAULT_DIRECTION = Path(__file__).resolve().parents[1] / "directions" / "refusal_dir.safetensors"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--direction", default=str(DEFAULT_DIRECTION))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--prompt", default="What is the capital of France?")
    ap.add_argument("--layers", default="0,20,38,63")
    args = ap.parse_args()

    layer_ids = [int(x) for x in args.layers.split(",")]

    print(f"[load] {args.pack}", flush=True)
    model, config = load_pack(args.pack)
    tok = load_tokenizer(args.pack)
    direction, meta = load_direction(args.direction)

    hidden = config["text_config"]["hidden_size"]
    if direction.size != hidden:
        raise SystemExit(f"direction has {direction.size} dims, model hidden size is {hidden}")
    print(f"[dir] {args.direction}  {direction.size} dims  layer={meta.get('layer')}", flush=True)

    ids = tok.encode(render_chat(args.pack, args.prompt, enable_thinking=False),
                     add_special_tokens=False).ids
    print(f"[prompt] {len(ids)} tokens", flush=True)

    print("\nmeasuring before ablation (this takes a few minutes per pass)...", flush=True)
    before = residual_components(model, ids, direction, layer_ids)

    wrapped = install(model, config, direction, alpha=args.alpha)
    print(f"[ablate] wrapped {len(wrapped)} residual writers", flush=True)
    if len(wrapped) != 129:
        print(f"[warn] expected 129 writers on this architecture, wrapped {len(wrapped)}")

    print("measuring after ablation...", flush=True)
    after = residual_components(model, ids, direction, layer_ids)

    print(f"\n{'layer':>6} {'before':>12} {'after':>12} {'reduction':>12}")
    ok = True
    for layer, b, a in zip(layer_ids, before, after):
        factor = b / max(a, 1e-30)
        print(f"{layer:>6} {b:>12.6f} {a:>12.3e} {factor:>11.1e}x")
        if a > 1e-4:
            ok = False
    print("\nPASS: the direction is being projected out." if ok else
          "\nFAIL: residual component is still large; the writers are not wrapped.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
