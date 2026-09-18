#!/usr/bin/env python3
"""Prototype: the pack's 2-bit matmul on simdgroup_matrix (8x8 mma), for M in 1..8.

    python scripts/proto_tern_mma.py --pack /path/to/pack

What it is for. Speculative decoding needs a "verify" matmul whose cost barely grows
with the number of tokens M in flight. MLX's quantized matmul on this pack costs
almost linearly in M up to 8 (per-row re-read and re-dequantise), so verifying 5
draft tokens costs 3.2 single steps and speculation cannot pay. This kernel computes
C^T = W . x^T with 8x8 matrix tiles: each lane dequantises its two adjacent codes of
one weight row (lane layout probed on the device: row = (l&7)>>1 + 4*(l>>4), col =
2*(l&1) + 4*((l>>3)&1)) through a 16-entry table into the A tile, x^T tiles are the B
operand, and per group of 128 the raw accumulator is scaled by the row's scale. The
cost is flat in M by construction.

What was measured on an M1 Ultra (microseconds per call, real weights, 32 calls
batched; "stock" is mx.quantized_matmul):

    shape          x  stock M=1  stock M=4  stock M=8   mma M=4  mma M=8
    17408x5120   128         83        273        534       227      221
    5120x17408    64         84        313        547       243      226
    10240x5120    48         82        179        346       197      145
    5120x6144     64         42        115        203       109      106
    all 401, ms          27.6       84.1      162.2      76.3     70.5

So the verify pass for 8 tokens costs 2.55x a plain step's matmuls instead of 5.9x,
and the kernel beats the stock one from M=4 on. It does NOT help M=1: on this GPU
simdgroup_multiply_accumulate runs on the ordinary FP32 ALUs (measured 3.48 TMAC/s,
about two thirds of the chip's FP32 peak; no matrix hardware before the M5 family),
so a 1-row problem padded to 8 rows does 8x the multiply-adds, and 89M weights x 8
MACs at 3.48 TMAC/s is 205 us, which is what the kernel measures at M=1 (220 us vs
83 us stock). Latency hiding (preloaded B tiles, two accumulators, split-K, simdgroup
count) moved nothing, as it should not when the bound is MAC throughput.

Implication for speculative decoding on this pack and machine: a block-8 verify step
costs roughly 2.5 plain steps plus the drafter, so it needs about 2.5 accepted tokens
per step to break even. Code and agent workloads with the published DFlash2 drafter
(94% acceptance reported on M4 Pro) would come out around 2x; chat-level acceptance
would come out slower than plain decoding, so any integration must be adaptive.
Errors against a float32 reference are at or below the stock kernel's.
"""
import sys, time
sys.path.insert(0, "/Users/seeker/code/OrcaBonsai-27B-Uncensored")
import mlx.core as mx

def lut():
    ents = []
    for b in range(16):
        c0, c1 = b & 3, (b >> 2) & 3
        v0 = 0.0 if c0 == 3 else float(c0 - 1); v1 = 0.0 if c1 == 3 else float(c1 - 1)
        ents.append(f"half2({v0:.1f}h, {v1:.1f}h)")
    return "constant half2 LUT[16] = {" + ", ".join(ents) + "};\n"

HEADER = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n" + lut()

SRC = r"""
    constexpr int KW = K / 16;
    constexpr int KG = K / 128;
    constexpr int GS = KG / SPLITK;          // groups per split
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const uint split = threadgroup_position_in_grid.y;
    const int n0 = (tg * SG + sgi) * 8;
    if (n0 >= N) return;
    const uint r = ((lane & 7) >> 1) + 4 * (lane >> 4);
    const uint c = 2 * (lane & 1) + 4 * ((lane >> 3) & 1);
    const device uint32_t* wrow = w + (n0 + r) * KW;
    const device half* srow = s + (n0 + r) * KG;
    float acc0 = 0.0f, acc1 = 0.0f;
    simdgroup_half8x8 A;
    thread auto& ae = A.thread_elements();
    for (int g = split * GS; g < (split + 1) * GS; g++) {
        simdgroup_float8x8 cg[NACC];
        for (int i = 0; i < NACC; i++) cg[i] = simdgroup_float8x8(0.0f);
        simdgroup_half8x8 Bt[PRE ? 16 : 1];
        if (PRE) {
            #pragma unroll
            for (int t = 0; t < 16; t++) simdgroup_load(Bt[t], xt, 8, ulong2(0, g * 128 + 8 * t));
        }
        #pragma unroll
        for (int wi = 0; wi < 8; wi++) {
            const uint32_t wv = wrow[g * 8 + wi];
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                const half2 v = LUT[extract_bits(wv, 16 * t + 2 * c, 4)];
                ae[0] = v.x; ae[1] = v.y;
                if (PRE) {
                    simdgroup_multiply_accumulate(cg[(2 * wi + t) % NACC], A, Bt[2 * wi + t], cg[(2 * wi + t) % NACC]);
                } else {
                    simdgroup_load(Bt[0], xt, 8, ulong2(0, g * 128 + (2 * wi + t) * 8));
                    simdgroup_multiply_accumulate(cg[(2 * wi + t) % NACC], A, Bt[0], cg[(2 * wi + t) % NACC]);
                }
            }
        }
        const float sc = float(srow[g]);
        for (int i = 0; i < NACC; i++) {
            thread auto& ce = cg[i].thread_elements();
            acc0 += sc * ce[0]; acc1 += sc * ce[1];
        }
    }
    device float* yo = y + split * (8 * N);
    yo[c * N + n0 + r] = acc0;
    yo[(c + 1) * N + n0 + r] = acc1;
"""

_k = None
def kern():
    global _k
    if _k is None:
        _k = mx.fast.metal_kernel(name="tern_mma2", input_names=["xt", "w", "s"], output_names=["y"],
                                  header=HEADER, source=SRC, ensure_row_contiguous=True)
    return _k

def tern_mma(x, w, s, sg=4, pre=1, nacc=2, splitk=1):
    M, K = x.shape; N = w.shape[0]
    xp = x if M == 8 else mx.concatenate([x, mx.zeros((8 - M, K), dtype=x.dtype)], 0)
    xt = mx.contiguous(xp.T)                      # (K, 8)
    tgs = (N // 8 + sg - 1) // sg
    y = kern()(inputs=[xt, w, s], template=[("K", K), ("N", N), ("SG", sg), ("PRE", pre), ("NACC", nacc), ("SPLITK", splitk)],
               grid=(tgs * sg * 32, splitk, 1), threadgroup=(sg * 32, 1, 1),
               output_shapes=[(splitk * 8, N)], output_dtypes=[mx.float32])[0]
    if splitk > 1:
        y = y.reshape(splitk, 8, N).sum(0)
    return y[:M].astype(x.dtype)

def bench(fn, n):
    for _ in range(3): mx.eval(fn())
    t0 = time.time(); mx.eval(fn()); return (time.time() - t0) / n

def main():
    from bonsai_abliterate.pack import load_pack
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--pack", required=True); args = ap.parse_args()
    model, config = load_pack(args.pack); lm = model.language_model
    mods = dict(lm.named_modules())
    want = ["model.layers.63.mlp.up_proj", "model.layers.63.mlp.down_proj", "model.layers.63.self_attn.o_proj"]
    REPS = 32
    configs = [dict(sg=4, pre=0, nacc=1, splitk=1), dict(sg=4, pre=1, nacc=1, splitk=1), dict(sg=4, pre=1, nacc=2, splitk=1),
               dict(sg=8, pre=1, nacc=2, splitk=1), dict(sg=2, pre=1, nacc=2, splitk=1),
               dict(sg=4, pre=1, nacc=2, splitk=2), dict(sg=4, pre=1, nacc=2, splitk=4)]
    for name in want:
        m = mods[name]; w, s, b = m.weight, m.scales, m.biases
        N, K = w.shape[0], w.shape[1] * 16
        wd = mx.dequantize(w, s, b, group_size=128, bits=2).astype(mx.float32)
        print(f"\n{name}: N={N} K={K}")
        for M in (1, 8):
            xs = [mx.random.normal((M, K)).astype(mx.float16) for _ in range(REPS)]; mx.eval(*xs)
            true = xs[0].astype(mx.float32) @ wd.T; sc = float(mx.max(mx.abs(true)).item())
            ts = bench(lambda: [mx.quantized_matmul(x, w, s, b, transpose=True, group_size=128, bits=2) for x in xs], REPS)
            print(f"  M={M}: stock {ts*1e6:5.0f} us")
            for cfg in configs:
                if (K // 128) % cfg["splitk"]: continue
                try:
                    out = tern_mma(xs[0], w, s, **cfg); mx.eval(out)
                    err = float(mx.max(mx.abs(out.astype(mx.float32) - true)).item())
                    tm = bench(lambda: [tern_mma(x, w, s, **cfg) for x in xs], REPS)
                    print(f"       {cfg}: {tm*1e6:5.0f} us  ratio {ts/tm:4.2f}  maxerr {err:.3g} (scale {sc:.3g})")
                except Exception as e:
                    print(f"       {cfg}: FAILED {str(e)[:300]}")

if __name__ == "__main__":
    main()
