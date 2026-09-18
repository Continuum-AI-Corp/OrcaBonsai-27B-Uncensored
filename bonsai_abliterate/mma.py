"""The pack's 2-bit matmul on ``simdgroup_matrix`` tiles, for 4 to 8 rows at once.

Why it exists. Speculative decoding verifies a block of draft tokens with one forward
pass over M = block tokens, and only pays if that pass costs about as much as a
single-token step. MLX's quantized matmul on this pack does not: its cost grows almost
linearly in M up to 8 (each row re-reads and re-dequantises the weights), so verifying
5 drafts costs 3.2 plain steps. This kernel computes ``C^T = W . x^T`` on 8x8 matrix
tiles, so the weights are dequantised once for all rows and the cost is flat in M.

Why it is only used for M >= 4. On the M1 family the matrix instruction runs on the
ordinary FP32 ALUs (3.48 TMAC/s measured, about two thirds of the chip's FP32 peak; no
matrix hardware before M5), so a 1-row problem padded to 8 rows does 8x the
multiply-adds and is 2.7x slower than the stock kernel. Measured over all 401 shapes on
an M1 Ultra: stock M=1 27.6 ms, stock M=8 162 ms, this kernel M=8 70.5 ms.

Kernel layout. Each lane of a simdgroup holds two adjacent columns of one row of the
8x8 A tile (layout probed on the device: row = (l&7)>>1 + 4*(l>>4), col = 2*(l&1) +
4*((l>>3)&1)), so one uint32 of the lane's weight row covers 16 k = two tiles. Codes
become values through a 16-entry table indexed by two codes at once: half2(c0-1, c1-1),
which is exact because the pack's group bias is minus its scale. x^T tiles (16 per
128-group, preloaded into registers) are the B operand; per group the raw accumulator
is scaled by the row's scale in float32. Weights are untouched.
"""
from __future__ import annotations

import mlx.core as mx

_LUT = "constant half2 LUT[16] = {" + ", ".join(
    f"half2({(0.0 if (b & 3) == 3 else float((b & 3) - 1)):.1f}h, "
    f"{(0.0 if ((b >> 2) & 3) == 3 else float(((b >> 2) & 3) - 1)):.1f}h)"
    for b in range(16)) + "};\n"

_HEADER = "#include <metal_simdgroup_matrix>\n#include <metal_simdgroup>\nusing namespace metal;\n" + _LUT

_SOURCE = r"""
    constexpr int KW = K / 16;
    constexpr int KG = K / 128;
    const uint lane = thread_index_in_simdgroup;
    const uint sgi = simdgroup_index_in_threadgroup;
    const uint tg = threadgroup_position_in_grid.x;
    const int n0 = (tg * SG + sgi) * 8;
    if (n0 >= N) return;
    const uint r = ((lane & 7) >> 1) + 4 * (lane >> 4);
    const uint c = 2 * (lane & 1) + 4 * ((lane >> 3) & 1);
    const device uint32_t* wrow = w + (n0 + r) * KW;
    const device half* srow = s + (n0 + r) * KG;
    float acc0 = 0.0f, acc1 = 0.0f;
    simdgroup_half8x8 A;
    thread auto& ae = A.thread_elements();
    for (int g = 0; g < KG; g++) {
        simdgroup_float8x8 cg[2];
        cg[0] = simdgroup_float8x8(0.0f);
        cg[1] = simdgroup_float8x8(0.0f);
        simdgroup_half8x8 Bt[16];
        #pragma unroll
        for (int t = 0; t < 16; t++) simdgroup_load(Bt[t], xt, 8, ulong2(0, g * 128 + 8 * t));
        #pragma unroll
        for (int wi = 0; wi < 8; wi++) {
            const uint32_t wv = wrow[g * 8 + wi];
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                const half2 v = LUT[extract_bits(wv, 16 * t + 2 * c, 4)];
                ae[0] = v.x; ae[1] = v.y;
                simdgroup_multiply_accumulate(cg[t], A, Bt[2 * wi + t], cg[t]);
            }
        }
        const float sc = float(srow[g]);
        thread auto& c0 = cg[0].thread_elements();
        thread auto& c1 = cg[1].thread_elements();
        acc0 += sc * (c0[0] + c1[0]);
        acc1 += sc * (c0[1] + c1[1]);
    }
    y[c * N + n0 + r] = acc0;
    y[(c + 1) * N + n0 + r] = acc1;
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bonsai_tern_mma", input_names=["xt", "w", "s"], output_names=["y"],
            header=_HEADER, source=_SOURCE, ensure_row_contiguous=True)
    return _kernel


def supports(w: mx.array, M: int) -> bool:
    N, KW = w.shape
    return 1 <= M <= 8 and N % 8 == 0 and (KW * 16) % 128 == 0


def tern_mma(x: mx.array, w: mx.array, s: mx.array, sg: int = 2) -> mx.array:
    """x: (M, K) with M <= 8, already in the pack's rotated input basis. Returns (M, N)."""
    M, K = x.shape
    N = w.shape[0]
    xp = x if M == 8 else mx.concatenate([x, mx.zeros((8 - M, K), dtype=x.dtype)], 0)
    xt = mx.contiguous(xp.T)
    tgs = (N // 8 + sg - 1) // sg
    y = _get_kernel()(
        inputs=[xt, w, s], template=[("K", K), ("N", N), ("SG", sg)],
        grid=(tgs * sg * 32, 1, 1), threadgroup=(sg * 32, 1, 1),
        output_shapes=[(8, N)], output_dtypes=[mx.float32])[0]
    return y[:M].astype(x.dtype)


_installed = False
_orig_call = None
MIN_ROWS = 4


def install() -> bool:
    """Route the pack's projections through the tile kernel when 4..8 rows are in flight.

    Patches the pack runtime's ``Packed.__call__`` (the module is importable once
    ``load_pack`` has run). Single-token decode and prefill keep the stock kernel, which
    is faster below 4 rows and from about 13 rows up.
    """
    global _installed, _orig_call
    if _installed:
        return True
    import runtime  # the pack's own module

    _orig_call = runtime.Packed.__call__

    def call(self, x):
        if self.embedding:
            return _orig_call(self, x)
        M = x.shape[-2] if x.ndim >= 2 else 1
        lead = x.shape[:-2]
        rows = 1
        for d in lead:
            rows *= d
        if rows != 1 or not (MIN_ROWS <= M <= 8) or not supports(self.weight, M):
            return _orig_call(self, x)
        if self.block:
            x = runtime.fwht(x, self.block, self.signs)
        K = x.shape[-1]
        dt = x.dtype
        x16 = x if dt == mx.float16 else x.astype(mx.float16)
        y = tern_mma(x16.reshape(M, K), self.weight, self.scales)
        return y.reshape(*lead, M, -1).astype(dt)

    runtime.Packed.__call__ = call
    _installed = True
    return True


def uninstall() -> None:
    global _installed
    if _installed:
        import runtime
        runtime.Packed.__call__ = _orig_call
        _installed = False
