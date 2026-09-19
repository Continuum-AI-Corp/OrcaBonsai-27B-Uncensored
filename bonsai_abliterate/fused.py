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
    // One threadgroup per value head; T tokens processed in sequence with the head's
    // recurrent state held in registers throughout. Conv channel layout: q (HK*DK),
    // k (HK*DK), v (HV*DV); the conv window is the TAPS-1 cached rows followed by the
    // T new rows of `mixed`.
    constexpr int KD = HK * DK;
    constexpr int CD = 2 * KD + HV * DV;
    constexpr int TAPS = 4;
    constexpr int CH = DK / 8;
    const uint h  = threadgroup_position_in_grid.x;
    const uint t  = thread_position_in_threadgroup.x;
    const uint hk = h / (HV / HK);
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ln = thread_index_in_simdgroup;

    threadgroup float qs[T * DK], ks[T * DK], vs[T * DV], outs[T * DV];
    threadgroup float part_b[32 * T], part_a[32 * T];
    threadgroup float gg[T], bb[T], qn[T], kn[T], on[T];

    // 0. gate coefficients for every token: b = <x_t, in_proj_b[h]>, a = <x_t, in_proj_a[h]>
    for (int tt = 0; tt < T; tt++) {
        float pb = 0.0f, pa = 0.0f;
        for (uint i = t; i < HID; i += 1024) {
            const float xv = float(x[tt * HID + i]);
            pb += xv * abw[h * HID + i];
            pa += xv * abw[(HV + h) * HID + i];
        }
        pb = simd_sum(pb); pa = simd_sum(pa);
        if (ln == 0) { part_b[sg * T + tt] = pb; part_a[sg * T + tt] = pa; }
    }
    // conv state shift: rows T..T+TAPS-2 of [cstate; mixed]. v channels belong to this
    // head; the q/k channels are shared per k group, so the first 2*KD/DV heads copy them.
    {
        const int cv = 2 * KD + h * DV;
        for (int tap = 0; tap < TAPS - 1; tap++) {
            const int j = T + tap;
            if (t < DV) {
                cstate_out[tap * CD + cv + t] = j < TAPS - 1 ? cstate[j * CD + cv + t] : mixed[(j - (TAPS - 1)) * CD + cv + t];
            }
            if (h < (2 * KD) / DV && t >= DV && t < 2 * DV) {
                const int c = h * DV + (t - DV);
                cstate_out[tap * CD + c] = j < TAPS - 1 ? cstate[j * CD + c] : mixed[(j - (TAPS - 1)) * CD + c];
            }
        }
    }
    // 1. depthwise conv + silu for every token over the window rows tt .. tt+TAPS-1
    if (t < DK + DK + DV) {
        const int which = t < DK ? 0 : (t < 2 * DK ? 1 : 2);
        const int i = which == 0 ? t : (which == 1 ? t - DK : t - 2 * DK);
        const int c = which == 0 ? hk * DK + i : (which == 1 ? KD + hk * DK + i : 2 * KD + h * DV + i);
        float w4[TAPS];
        for (int tap = 0; tap < TAPS; tap++) w4[tap] = cw[tap * CD + c];
        for (int tt = 0; tt < T; tt++) {
            float acc = 0.0f;
            for (int tap = 0; tap < TAPS; tap++) {
                const int j = tt + tap;
                const float v = j < TAPS - 1 ? float(cstate[j * CD + c]) : float(mixed[(j - (TAPS - 1)) * CD + c]);
                acc += v * w4[tap];
            }
            const float sv = acc / (1.0f + exp(-acc));
            if (which == 0) qs[tt * DK + i] = sv; else if (which == 1) ks[tt * DK + i] = sv; else vs[tt * DV + i] = sv;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 2. gate reduce, rms scales for q and k, per token
    if (sg == 0) {
        for (int tt = 0; tt < T; tt++) {
            const float pb = simd_sum(part_b[ln * T + tt]), pa = simd_sum(part_a[ln * T + tt]);
            float sq = 0.0f, sk = 0.0f;
            for (int i = 0; i < DK / 32; i++) {
                const float a = qs[tt * DK + ln * (DK / 32) + i], b = ks[tt * DK + ln * (DK / 32) + i];
                sq += a * a; sk += b * b;
            }
            sq = simd_sum(sq); sk = simd_sum(sk);
            if (ln == 0) {
                const float a_ = pa + dtb[h];
                const float sp = a_ > 20.0f ? a_ : log1p(exp(a_));
                gg[tt] = exp(-exp(alog[h]) * sp);
                bb[tt] = 1.0f / (1.0f + exp(-pb));
                const float inv_scale = rsqrt(float(DK));
                qn[tt] = rsqrt(sq / float(DK) + 1e-6f) * inv_scale * inv_scale;
                kn[tt] = rsqrt(sk / float(DK) + 1e-6f) * inv_scale;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 3. delta rule over the tokens, state in registers, no barriers
    const int dv = t >> 3, c0 = (t & 7) * CH;
    float st[CH];
    {
        const device float* sp = state + (h * DV + dv) * DK + c0;
        for (int i = 0; i < CH; i++) st[i] = sp[i];
    }
    for (int tt = 0; tt < T; tt++) {
        const float g_ = gg[tt], b_ = bb[tt], kn_ = kn[tt], qn_ = qn[tt];
        const threadgroup float* kt = ks + tt * DK + c0;
        const threadgroup float* qt = qs + tt * DK + c0;
        float kv = 0.0f;
        for (int i = 0; i < CH; i++) {
            st[i] *= g_;
            kv += st[i] * kt[i];
        }
        kv *= kn_;
        kv += simd_shuffle_xor(kv, 1); kv += simd_shuffle_xor(kv, 2); kv += simd_shuffle_xor(kv, 4);
        const float delta = (vs[tt * DV + dv] - kv) * b_ * kn_;
        float o = 0.0f;
        for (int i = 0; i < CH; i++) {
            st[i] += kt[i] * delta;
            o += st[i] * qt[i];
        }
        o *= qn_;
        o += simd_shuffle_xor(o, 1); o += simd_shuffle_xor(o, 2); o += simd_shuffle_xor(o, 4);
        if ((t & 7) == 0) outs[tt * DV + dv] = o;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // 4. gated RMSNorm scale per token
    if (sg == 0) {
        for (int tt = 0; tt < T; tt++) {
            float ss = 0.0f;
            for (int i = 0; i < DV / 32; i++) { const float a = outs[tt * DV + ln * (DV / 32) + i]; ss += a * a; }
            ss = simd_sum(ss);
            if (ln == 0) on[tt] = rsqrt(ss / float(DV) + eps[0]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < DV) {
        for (int tt = 0; tt < T; tt++) {
            const float xo = outs[tt * DV + t] * on[tt] * nw[t];
            const float zz = float(z[tt * HV * DV + h * DV + t]);
            y[tt * HV * DV + h * DV + t] = half(zz / (1.0f + exp(-zz)) * xo);
        }
    }
    {
        device float* so = state_out + (h * DV + dv) * DK + c0;
        for (int i = 0; i < CH; i++) so[i] = st[i];
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


RECORD = False   # when True, gdn_decode keeps its inputs on the module for rollback()


def gdn_decode(m, x, cache):
    """Decode T (1..8) tokens of a ``Qwen3_5GatedDeltaNet`` layer: x is (1, T, hidden)."""
    cw, alog, dtb, nw, ab_w, eps = _prepare(m)
    T = x.shape[1]
    mixed = m.in_proj_qkv(x)
    z = m.in_proj_z(x)
    HV, HK, DK, DV = m.num_v_heads, m.num_k_heads, m.head_k_dim, m.head_v_dim
    cstate = cache[0]
    state = cache[1]
    if state is None:
        state = mx.zeros((1, HV, DV, DK), dtype=mx.float32)
    y, cstate_out, state_out = _run_kernel(m, x, mixed, z, cstate, state, T, cw, alog, dtb, nw, ab_w, eps)
    if RECORD:
        m._fused_last = (x, mixed, z, cstate, state, T)
    cache[0] = cstate_out
    cache[1] = state_out
    if hasattr(cache, "advance"):
        cache.advance(T)
    return m.out_proj(y)


def _run_kernel(m, x, mixed, z, cstate, state, T, cw, alog, dtb, nw, ab_w, eps):
    HV, HK, DK, DV = m.num_v_heads, m.num_k_heads, m.head_k_dim, m.head_v_dim
    return _get_kernel()(
        inputs=[x, ab_w, mixed, z, cstate, cw, alog, dtb, state, nw, eps],
        template=[("HV", HV), ("HK", HK), ("DK", DK), ("DV", DV), ("HID", x.shape[-1]), ("T", T)],
        grid=(1024 * HV, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, T, HV * DV), cstate.shape, (1, HV, DV, DK)],
        output_dtypes=[mixed.dtype, cstate.dtype, mx.float32],
    )


def rollback(model, cache, fed: int, keep: int) -> None:
    """After a recorded forward of ``fed`` tokens, leave the caches as if only the first
    ``keep`` of them had been fed.

    Linear-attention layers are replayed from their recorded inputs and pre-forward
    state through the same kernel with T=keep (the state never leaves the kernel, so
    there is nothing to snapshot per position); attention caches are trimmed.
    """
    language_model = getattr(model, "language_model", model)
    for layer, c in zip(language_model.model.layers, cache):
        m = getattr(layer, "linear_attn", None)
        if m is None:
            if fed > keep and c is not None and c.is_trimmable():
                c.trim(fed - keep)
            continue
        rec = getattr(m, "_fused_last", None)
        if rec is None:
            continue
        x, mixed, z, cstate, state, T = rec
        if keep >= T:
            continue
        cw, alog, dtb, nw, ab_w, eps = _prepare(m)
        _, cstate_out, state_out = _run_kernel(m, x[:, :keep], mixed[:, :keep], z[:, :keep],
                                               cstate, state, keep, cw, alog, dtb, nw, ab_w, eps)
        c[0] = cstate_out
        c[1] = state_out
        if hasattr(c, "advance"):
            c.advance(keep - T)


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
        and 1 <= inputs.shape[1] <= 8
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
