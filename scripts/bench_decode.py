#!/usr/bin/env python3
"""Where a decode step's time goes on this machine, with and without the ablation.

    python scripts/bench_decode.py --pack /path/to/pack

Four measurements, all batch 1 unless stated, each reported in milliseconds:

  step        one full decode step (one token in, logits out), alpha=0 and alpha=1
  matmuls     the pack's 401 quantized matmuls alone, on random inputs of the right
              shape, launched together the way a real step launches them; and the same
              with the Hadamard input transform skipped, which is not a valid model but
              shows what the transform costs
  blocks      the matmuls split by block type: linear-attention, full-attention, MLP
  M scaling   the matmuls and the full step with M tokens in flight, which is the shape
              speculative decoding's verify pass and batched serving use

Why it exists: on an M1 Ultra the model decodes at ~24 tok/s with 800 GB/s of memory
bandwidth and 7.2 GB of weights, so it is not bandwidth-bound. The numbers here showed
that MLX's 2-bit matmul kernel is ALU-bound (the 4-bit kernel streams the same bytes at
nearly twice the rate) and that its cost grows almost linearly with M up to 16, which is
what makes speculative decoding unprofitable on this pack without a different kernel.
Rerun it after any change to the runtime path, or on a new machine, before believing an
estimate. Timings drift with GPU temperature; compare rows within one run.
"""
from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx

from bonsai_abliterate.pack import load_pack, load_tokenizer, render_chat
from bonsai_abliterate.ablation import install, load_direction

DEFAULT_DIRECTION = Path(__file__).resolve().parent.parent / "directions" / "refusal_dir.safetensors"


def ms(fn, n=20):
    mx.eval(fn())
    t0 = time.time()
    for _ in range(n):
        mx.eval(fn())
    return (time.time() - t0) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--direction", default=str(DEFAULT_DIRECTION))
    ap.add_argument("--prompt", default="Write an essay on the history of artificial intelligence.")
    args = ap.parse_args()

    print(f"[load] {args.pack}", flush=True)
    model, config = load_pack(args.pack)
    tok = load_tokenizer(args.pack)
    lm = model.language_model
    from runtime import Packed  # the pack's own module, importable after load_pack
    print(f"[gpu] {mx.device_info()['device_name']}")

    ids = tok.encode(render_chat(args.pack, args.prompt), add_special_tokens=False).ids

    def step_time(M=1):
        cache = lm.make_cache()
        mx.eval(lm(mx.array([ids], dtype=mx.int32), cache=cache).logits)
        x = mx.array([[ids[-1]] * M], dtype=mx.int32)
        return ms(lambda: lm(x, cache=cache).logits, n=10)

    print("\n== step")
    t0 = step_time()
    print(f"alpha=0: {t0:6.1f} ms/step  {1000/t0:5.1f} tok/s")
    direction, _ = load_direction(args.direction)
    install(model, config, direction, alpha=1.0)
    t1 = step_time()
    print(f"alpha=1: {t1:6.1f} ms/step  {1000/t1:5.1f} tok/s   (ablation costs {t1-t0:.1f} ms)")

    packed = [(n, m) for n, m in lm.named_modules() if isinstance(m, Packed) and not m.embedding]
    total_bytes = sum(m.weight.nbytes + m.scales.nbytes + m.biases.nbytes for _, m in packed)

    def matmuls(mods, M=1):
        xs = [mx.random.normal((1, M, m.weight.shape[1] * 16)).astype(mx.float16) for m in mods]
        mx.eval(*xs)
        return ms(lambda: [m(x) for m, x in zip(mods, xs)])

    print("\n== matmuls")
    all_mods = [m for _, m in packed]
    t = matmuls(all_mods)
    print(f"{len(all_mods)} matmuls, {total_bytes/1e9:.2f} GB: {t:6.1f} ms  ({total_bytes/t/1e6:4.0f} GB/s)")
    blocks = [m.block for m in all_mods]
    for m in all_mods:
        m.block = 0
    t_nohad = matmuls(all_mods)
    for m, b in zip(all_mods, blocks):
        m.block = b
    print(f"same, Hadamard skipped:       {t_nohad:6.1f} ms  ({total_bytes/t_nohad/1e6:4.0f} GB/s)  transform costs {t-t_nohad:.1f} ms")

    print("\n== blocks (matmuls only)")
    for label, key in (("linear_attn", ".linear_attn."), ("self_attn", ".self_attn."), ("mlp", ".mlp.")):
        mods = [m for n, m in packed if key in n]
        by = sum(m.weight.nbytes + m.scales.nbytes + m.biases.nbytes for m in mods)
        tb = matmuls(mods)
        print(f"{label:12s} {len(mods):3d} matmuls {by/1e9:.2f} GB: {tb:6.1f} ms  ({by/tb/1e6:4.0f} GB/s)")

    print("\n== M scaling (tokens per step)")
    print(f"{'M':>3s} {'matmuls ms':>11s} {'full step ms':>13s} {'ms/token':>9s}")
    for M in (1, 2, 4, 5, 8, 16, 32):
        tm = matmuls(all_mods, M)
        ts = step_time(M) if M <= 16 else float("nan")
        print(f"{M:3d} {tm:11.1f} {ts:13.1f} {ts/M:9.1f}")


if __name__ == "__main__":
    main()
