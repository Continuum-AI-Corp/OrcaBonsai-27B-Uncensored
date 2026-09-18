"""Apply a refusal direction to a Prism Hadamard ternary pack at run time.

Ordinary abliteration is a permanent weight edit: every matrix that writes the residual
stream is orthogonalised against the refusal direction, ``W <- W - r (r^T W)``. That is
not possible on Bonsai 2 27B. Its weights are ternary -- the affine container stores
``scale = s`` and ``bias = -s``, so the 2-bit codes ``{0,1,2}`` decode to exactly
``{-s, 0, +s}`` -- while the orthogonalised matrix is dense and full precision. Storing
it back would mean re-quantizing to ternary, and the model's quality at 1.72 bits/weight
comes from quantization-aware training, not from the format: re-quantizing without that
training is what destroys it.

This module applies the same operator at the point of use instead:

    y <- y - alpha (y . r) r

on the output of every residual writer. That is algebraically identical to
orthogonalising the matrix which produced ``y``, but the packed weights stay
bit-identical, so **no quantization error is added at all**. It is also reversible, and
``alpha`` stays adjustable at run time -- neither of which a baked edit can offer.

Two properties of the pack matter when wiring this up:

* The Hadamard rotation is folded into the **input** dimension only
  (``hadamard.json``: ``axis=input-last-dimension``, block 1024). The vectors being
  projected are the *output* rows of the writers, and the embedding output is
  un-rotated by ``Packed`` itself. So the direction is an ordinary vector in the plain
  hidden basis and **this path needs no Hadamard handling**.
* The residual writers are ``mlp.down_proj`` (64), ``linear_attn.out_proj`` (48) and
  ``self_attn.o_proj`` (16), plus ``model.embed_tokens``: 129 modules in total. The
  architecture is 48 linear-attention and 16 full-attention layers, so wrapping only
  ``o_proj`` would miss three quarters of the attention writes.
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from mlx import nn

# Suffixes of the modules whose output is added to the residual stream.
RESIDUAL_WRITERS = ("mlp.down_proj", "self_attn.o_proj", "linear_attn.out_proj")
EMBEDDING_PATH = "model.embed_tokens"


_PROJECT_SOURCE = r"""
    // One threadgroup per row of y: dot with the direction in float32, then subtract.
    // 1024 threads, each owning ceil(D/1024) strided elements.
    threadgroup float partial[32];
    const uint row = threadgroup_position_in_grid.x;
    const uint t = thread_position_in_threadgroup.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint ln = thread_index_in_simdgroup;
    const device T* yr = y + row * D;
    device T* outr = out + row * D;
    float acc = 0.0f;
    for (uint i = t; i < D; i += 1024) acc += float(yr[i]) * d[i];
    acc = simd_sum(acc);
    if (ln == 0) partial[sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float v = partial[ln];
        v = simd_sum(v);
        if (ln == 0) partial[0] = v * alpha[0];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float c = partial[0];
    for (uint i = t; i < D; i += 1024) outr[i] = static_cast<T>(float(yr[i]) - c * d[i]);
"""

_project_kernel = None


def _get_project_kernel():
    global _project_kernel
    if _project_kernel is None:
        _project_kernel = mx.fast.metal_kernel(
            name="bonsai_project_out",
            input_names=["y", "d", "alpha"],
            output_names=["out"],
            source=_PROJECT_SOURCE,
            ensure_row_contiguous=True,
        )
    return _project_kernel


def project_out(y: mx.array, direction: mx.array, alpha: mx.array) -> mx.array:
    """``y - alpha (y . d) d`` along the last axis, as one Metal launch.

    The same arithmetic as the array-op version below (float32 throughout, cast at the
    end), in one kernel instead of six: on a launch-bound decode step the six cost
    ~3 ms per token across the 129 sites, the one costs a third of that.
    """
    D = y.shape[-1]
    rows = y.size // D
    out = _get_project_kernel()(
        inputs=[y.reshape(rows, D), direction, alpha],
        template=[("T", y.dtype), ("D", D)],
        grid=(rows * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(rows, D)],
        output_dtypes=[y.dtype],
    )[0]
    return out.reshape(y.shape)


class Ablated(nn.Module):
    """Wrap a module so its output loses the component along ``direction``.

    The constants are stored under underscore-prefixed names because MLX's parameter
    filter skips those: they must not show up in ``model.parameters()`` or in a later
    ``load_weights(strict=True)``.
    """

    def __init__(self, inner, direction: mx.array, alpha: float = 1.0):
        super().__init__()
        self.inner = inner
        d = direction.astype(mx.float32).reshape(-1)
        self._direction = d / mx.maximum(mx.linalg.norm(d), 1e-12)
        self._alpha = float(alpha)
        self._alpha_arr = mx.array([float(alpha)], dtype=mx.float32)

    def __call__(self, *args, **kwargs):
        y = self.inner(*args, **kwargs)
        if mx.default_device() == mx.gpu and mx.metal.is_available() and y.dtype in (mx.float16, mx.bfloat16, mx.float32):
            return project_out(y, self._direction, self._alpha_arr)
        yf = y.astype(mx.float32)
        component = mx.sum(yf * self._direction, axis=-1, keepdims=True)
        return (yf - self._alpha * component * self._direction).astype(y.dtype)


def load_direction(path: str | Path) -> tuple[mx.array, dict]:
    """Load a unit refusal direction and its metadata from a safetensors file."""
    arrays, meta = mx.load(str(path), return_metadata=True)
    d = arrays["direction"].astype(mx.float32).reshape(-1)
    return d / mx.maximum(mx.linalg.norm(d), 1e-12), meta


def install(
    model,
    config: dict,
    direction: mx.array,
    alpha: float = 1.0,
    layers=None,
    include_embedding: bool = True,
) -> list[str]:
    """Wrap the pack's residual writers in place. Returns the wrapped module paths.

    Call this **after** the pack's loader. Wrapping renames the module subtree
    (``down_proj`` becomes ``down_proj.inner``), which would break a strict weight load
    if done first.

    ``alpha``: 1.0 reproduces a full weight orthogonalisation; lower values trade
    ablation strength for capability retention; 0.0 is a no-op.

    ``layers``: restrict to these layer indices, or None for every layer.
    """
    language_model = getattr(model, "language_model", model)
    wanted = None if layers is None else set(layers)
    wrapped = []

    for record in config["modules"]:
        path = record["path"]
        if path == EMBEDDING_PATH:
            if not include_embedding:
                continue
        elif path.endswith(RESIDUAL_WRITERS):
            if wanted is not None and int(path.split(".")[2]) not in wanted:
                continue
        else:
            continue

        parts = path.split(".")
        parent = language_model
        for part in parts[:-1]:
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        inner = getattr(parent, parts[-1])
        if isinstance(inner, Ablated):
            raise ValueError(f"{path} is already ablated")
        setattr(parent, parts[-1], Ablated(inner, direction, alpha))
        wrapped.append(path)

    model.eval()
    return wrapped


def residual_components(model, token_ids, direction: mx.array, layer_ids) -> list[float]:
    """``|<h, d>| / ||h||`` of the last-token residual at each requested layer.

    Run it before and after :func:`install` to confirm the projection is in force.

    Measure block outputs, never the model's final output: the closing RMSNorm
    multiplies elementwise by a diagonal weight, which does not preserve orthogonality
    and reintroduces a component along the direction even after a perfect ablation.
    """
    language_model = getattr(model, "language_model", model)
    d = direction.astype(mx.float32).reshape(-1)
    d = d / mx.maximum(mx.linalg.norm(d), 1e-12)
    out = language_model(
        mx.array([list(token_ids)], dtype=mx.int32), capture_layer_ids=list(layer_ids)
    )
    mx.eval(out.hidden_states)
    fractions = []
    for h in out.hidden_states:
        v = h[0, -1].astype(mx.float32)
        fractions.append(
            float((mx.abs(mx.sum(v * d)) / mx.maximum(mx.linalg.norm(v), 1e-12)).item())
        )
    return fractions
