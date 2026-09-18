"""Fused decode-step kernels for the pack's linear-attention layers.

Why: a decode step on this pack is bound by the number of kernel launches, not by
memory bandwidth (see ``scripts/bench_decode.py``). Each of the 48 gated-delta-net
layers spends ~200 us per token on the work *around* its matmuls: the depthwise conv
and its state shift, silu, the split into heads, two RMSNorms, the gate coefficients,
the delta-rule recurrence, and the gated RMSNorm on the way out. Those are a dozen
launches on tensors of a few thousand elements each, so every one of them costs its
launch latency and nothing else. This module replaces that chain with one Metal kernel
per layer: one threadgroup of 1024 threads per value head, each thread owning 16 entries
of that head's 128x128 recurrent state.

The kernel is applied only on the single-token, single-sequence, unmasked path that
decoding takes; prefill, batched and speculative paths fall through to the pack
runtime's own code. Weights are untouched: it reads the same conv weights, gate
parameters and norm weights the runtime reads.

Not fused, on purpose: the runtime's ``fwht`` (cast, sign multiply, hadamard
transform, cast) is four launches per projection, but a single-kernel replacement with
the Walsh-Hadamard butterfly in threadgroup memory measured slower than those four
(0.83 ms vs 0.65 ms per 48 calls); MLX's transform kernel is fast and the three
elementwise launches around it cost only a few microseconds each in a batched command
buffer. Also not done: stacking the projections that share an input (q/k/v, gate/up,
qkv/z) into one matmul. It cuts 401 launches to 257, and a synthetic dependent chain
of the matmuls measured 5 ms faster, but inside the model the MLP chain went from
18.7 to 18.4 ms and attention did not move: at these sizes the 2-bit matmul kernel
is throughput-bound, so one double-height matmul costs what two cost. The decode
step is mostly the serial sum of the matmuls' own times, not launch overhead around
them; the fusion above removes the part that is.

Numerics: the runtime's chain rounds to fp16 between its ops (conv output, the
normalised q and k, the recurrence output). The fused kernel keeps everything in
float32 until the final cast, so its result is not bit-identical to the runtime's,
only closer to the exact one; ``check`` reports the difference on real layers so the
gap is measured rather than assumed.
"""
from __future__ import annotations

import mlx.core as mx

_SOURCE = r"""
    // One threadgroup per value head. Layout of the conv channels: q (HK*DK), k (HK*DK),
    // v (HV*DV); the conv input is the 3 cached steps followed by the current one.
    constexpr int KD = HK * DK;            // key_dim
    constexpr int CD = 2 * KD + HV * DV;   // conv_dim
    constexpr int TAPS = 4;
    const uint h  = threadgroup_position_in_grid.x;
    const uint t  = thread_position_in_threadgroup.x;
    const uint hk = h / (HV / HK);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ln = thread_index_in_simdgroup;

    threadgroup float qs[DK], ks[DK], vs[DV], outs[DV];
    threadgroup float part_b[32], part_a[32];
    threadgroup float qn, kn, gg, bb, on;

    // 0. this head's b and a: dot products of x with two fp32 rows (in_proj_b, in_proj_a)
    {
        float pb = 0.0f, pa = 0.0f;
        for (uint i = t; i < HID; i += 1024) {
            float xv = float(x[i]);
            pb += xv * abw[h * HID + i];
            pa += xv * abw[(HV + h) * HID + i];
        }
        pb = simd_sum(pb); pa = simd_sum(pa);
        if (ln == 0) { part_b[sg] = pb; part_a[sg] = pa; }
    }

    // 1. depthwise conv + silu for this head's q, k and v channels
    if (t < DK + DK + DV) {
        int which = t < DK ? 0 : (t < 2 * DK ? 1 : 2);
        int i = which == 0 ? t : (which == 1 ? t - DK : t - 2 * DK);
        int c = which == 0 ? hk * DK + i : (which == 1 ? KD + hk * DK + i : 2 * KD + h * DV + i);
        float acc = 0.0f;
        for (int tap = 0; tap < TAPS - 1; tap++)
            acc += float(cstate[tap * CD + c]) * cw[tap * CD + c];
        acc += float(mixed[c]) * cw[(TAPS - 1) * CD + c];
        float sv = acc / (1.0f + exp(-acc));
        if (which == 0) qs[i] = sv; else if (which == 1) ks[i] = sv; else vs[i] = sv;
    }
    // conv state shift. v channels belong to this head; the q/k channels (2*KD of them)
    // are shared between the heads of a k group, so the first 2*KD/DV heads copy them.
    {
        int cv = 2 * KD + h * DV;
        if (t < DV) {
            for (int tap = 0; tap < TAPS - 2; tap++)
                cstate_out[tap * CD + cv + t] = cstate[(tap + 1) * CD + cv + t];
            cstate_out[(TAPS - 2) * CD + cv + t] = mixed[cv + t];
        }
        if (h < (2 * KD) / DV && t >= DV && t < 2 * DV) {
            int c = h * DV + (t - DV);
            for (int tap = 0; tap < TAPS - 2; tap++)
                cstate_out[tap * CD + c] = cstate[(tap + 1) * CD + c];
            cstate_out[(TAPS - 2) * CD + c] = mixed[c];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. rms scales for q and k (no weight), gate g and beta
    if (sg == 0) {
        float sq = 0.0f, sk = 0.0f;
        for (int i = 0; i < DK / 32; i++) {
            float a = qs[ln * (DK / 32) + i], b = ks[ln * (DK / 32) + i];
            sq += a * a; sk += b * b;
        }
        sq = simd_sum(sq); sk = simd_sum(sk);
        if (ln == 0) {
            const float inv_scale = rsqrt(float(DK));
            qn = rsqrt(sq / float(DK) + 1e-6f) * inv_scale * inv_scale;
            kn = rsqrt(sk / float(DK) + 1e-6f) * inv_scale;
        }
        float pb = simd_sum(part_b[ln]), pa = simd_sum(part_a[ln]);
        if (ln == 0) {
            float a_ = pa + dtb[h];
            float sp = a_ > 20.0f ? a_ : log1p(exp(a_));
            gg = exp(-exp(alog[h]) * sp);
            bb = 1.0f / (1.0f + exp(-pb));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. delta rule. thread t owns state[h, dv, c0 .. c0+CH) with dv = t/8, c0 = (t%8)*CH
    {
        constexpr int CH = DK / 8;
        const int dv = t >> 3, c0 = (t & 7) * CH;
        const device float* sp = state + (h * DV + dv) * DK + c0;
        device float* so = state_out + (h * DV + dv) * DK + c0;
        float st[CH], kk[CH];
        float kv = 0.0f;
        for (int i = 0; i < CH; i++) {
            st[i] = sp[i] * gg;
            kk[i] = ks[c0 + i] * kn;
            kv += st[i] * kk[i];
        }
        kv += simd_shuffle_xor(kv, 1); kv += simd_shuffle_xor(kv, 2); kv += simd_shuffle_xor(kv, 4);
        const float delta = (vs[dv] - kv) * bb;
        float o = 0.0f;
        for (int i = 0; i < CH; i++) {
            st[i] += kk[i] * delta;
            o += st[i] * qs[c0 + i] * qn;
            so[i] = st[i];
        }
        o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2); o += simd_shuffle_xor(o, 4);
        if ((t & 7) == 0) outs[dv] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. gated RMSNorm: rms over the head's DV outputs, weight nw, gate silu(z)
    if (sg == 0) {
        float ss = 0.0f;
        for (int i = 0; i < DV / 32; i++) { float a = outs[ln * (DV / 32) + i]; ss += a * a; }
        ss = simd_sum(ss);
        if (ln == 0) on = rsqrt(ss / float(DV) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < DV) {
        float xo = outs[t] * on * nw[t];
        float zz = float(z[h * DV + t]);
        y[h * DV + t] = half(zz / (1.0f + exp(-zz)) * xo);
    }
"""

_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = mx.fast.metal_kernel(
            name="bonsai_gdn_decode",
            input_names=["x", "abw", "mixed", "z", "cstate", "cw", "alog", "dtb", "state", "nw", "eps"],
            output_names=["y", "cstate_out", "state_out"],
            source=_SOURCE,
            ensure_row_contiguous=True,
        )
    return _kernel


def _prepare(m):
    """Per-layer constants the kernel reads, computed once and cached on the module.

    Stored under underscore names so MLX's parameter filter ignores them (the same
    convention ``ablation.Ablated`` uses).
    """
    if getattr(m, "_fused_consts", None) is None or m._fused_consts[0] != id(m.conv1d.weight):
        cw = m.conv1d.weight[:, :, 0].T.astype(mx.float32)          # (taps, conv_dim)
        alog = m.A_log.astype(mx.float32)
        dtb = m.dt_bias.astype(mx.float32)
        nw = m.norm.weight.astype(mx.float32)
        ab_w = mx.concatenate([m.in_proj_b.weight, m.in_proj_a.weight], 0).astype(mx.float32)
        eps = mx.array([float(m.layer_norm_epsilon)], dtype=mx.float32)
        mx.eval(cw, alog, dtb, nw, ab_w, eps)
        m._fused_consts = (id(m.conv1d.weight), cw, alog, dtb, nw, ab_w, eps)
    return m._fused_consts[1:]


def gdn_decode(m, x, cache):
    """One decode step of a ``Qwen3_5GatedDeltaNet`` layer: x is (1, 1, hidden)."""
    cw, alog, dtb, nw, ab_w, eps = _prepare(m)
    mixed = m.in_proj_qkv(x)
    z = m.in_proj_z(x)
    HV, HK, DK, DV = m.num_v_heads, m.num_k_heads, m.head_k_dim, m.head_v_dim
    cstate = cache[0]
    state = cache[1]
    if state is None:
        state = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    y, cstate_out, state_out = _get_kernel()(
        inputs=[x, ab_w, mixed, z, cstate, cw, alog, dtb, state, nw, eps],
        template=[("HV", HV), ("HK", HK), ("DK", DK), ("DV", DV), ("HID", x.shape[-1])],
        grid=(1024 * HV, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, 1, HV * DV), cstate.shape, (1, HV, DV, DK)],
        output_dtypes=[mixed.dtype, cstate.dtype, mx.float32],
    )
    cache[0] = cstate_out
    cache[1] = state_out
    if hasattr(cache, "advance"):
        cache.advance(1)
    return m.out_proj(y)


def _fast_path_ok(m, inputs, mask, cache, kwargs):
    return (
        cache is not None
        and cache[0] is not None
        and cache[1] is not None
        and mask is None
        and not kwargs.get("target_verify")
        and kwargs.get("gdn_sink") is None
        and inputs.ndim == 3
        and inputs.shape[0] == 1
        and inputs.shape[1] == 1
        and cache[0].shape[0] == 1
        and cache[0].shape[1] == m.conv_kernel_size - 1
        and cache[1].shape[0] == 1
        and m.head_k_dim % 32 == 0
        and m.head_v_dim % 32 == 0
        and 8 * m.head_v_dim == 1024
        and m.head_k_dim + m.head_k_dim + m.head_v_dim <= 1024
        and getattr(cache, "lengths", None) is None
        and m.conv_kernel_size == 4
    )


_fused_classes: dict[type, type] = {}


def _fused_class(base: type) -> type:
    """A subclass of the runtime's layer class whose __call__ takes the fast path.

    Python resolves ``obj()`` on the type, not the instance, so an instance attribute
    named ``__call__`` is ignored; swapping the instance's class is the way to patch
    one layer without touching the others or the runtime's module.
    """
    if base not in _fused_classes:
        def __call__(self, inputs, mask=None, cache=None, **kwargs):
            if _fast_path_ok(self, inputs, mask, cache, kwargs):
                return gdn_decode(self, inputs, cache)
            return base.__call__(self, inputs, mask, cache, **kwargs)

        _fused_classes[base] = type("Fused" + base.__name__, (base,), {"__call__": __call__, "_fused_base": base})
    return _fused_classes[base]


def install(model) -> int:
    """Route each linear-attention layer's single-token decode through the fused kernel.

    Returns the number of layers patched. Every other call shape (prefill, batches,
    masks, verification) still runs the runtime's own code.
    """
    language_model = getattr(model, "language_model", model)
    patched = 0
    for layer in language_model.model.layers:
        m = getattr(layer, "linear_attn", None)
        if m is None or hasattr(type(m), "_fused_base"):
            continue
        m.__class__ = _fused_class(type(m))
        patched += 1
    return patched


def uninstall(model) -> int:
    language_model = getattr(model, "language_model", model)
    n = 0
    for layer in language_model.model.layers:
        m = getattr(layer, "linear_attn", None)
        if m is not None and hasattr(type(m), "_fused_base"):
            m.__class__ = type(m)._fused_base
            n += 1
    return n


def check(model, token_ids, layers=None) -> list[dict]:
    """Compare the fused step against the runtime's own on real layers and real state.

    Prefills ``token_ids`` with the original path, then runs one decode token through
    both paths from the same cache and reports, per layer, the max abs difference of the
    layer output, the new recurrent state and the new conv state, alongside the output's
    scale. Uses the original path for the prefill so both start from identical state.
    """
    language_model = getattr(model, "language_model", model)
    was = uninstall(model)
    cache = language_model.make_cache()
    mx.eval(language_model(mx.array([list(token_ids)], dtype=mx.int32), cache=cache).logits)
    hidden = language_model.model.layers[0].input_layernorm.weight.shape[0]
    x = mx.random.normal((1, 1, hidden)).astype(mx.float16)
    mx.eval(x)
    reports = []
    for i, (layer, c) in enumerate(zip(language_model.model.layers, cache)):
        m = getattr(layer, "linear_attn", None)
        if m is None or (layers is not None and i not in layers):
            continue
        c_ref = type(c)(size=2); c_ref[0], c_ref[1] = c[0], c[1]
        c_new = type(c)(size=2); c_new[0], c_new[1] = c[0], c[1]
        ref = m(x, None, c_ref)
        out = gdn_decode(m, x, c_new)
        mx.eval(ref, out, c_ref[0], c_ref[1], c_new[0], c_new[1])
        f = lambda a, b: float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))).item())
        reports.append({
            "layer": i,
            "out_maxerr": f(ref, out),
            "out_scale": float(mx.max(mx.abs(ref.astype(mx.float32))).item()),
            "state_maxerr": f(c_ref[1], c_new[1]),
            "state_scale": float(mx.max(mx.abs(c_ref[1])).item()),
            "conv_maxerr": f(c_ref[0], c_new[0]),
        })
    if was:
        install(model)
    return reports
