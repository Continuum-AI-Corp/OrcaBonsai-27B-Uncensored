# OrcaRouter Ternary Bonsai 2 27B Uncensored

**Runtime-uncensored Ternary Bonsai 2 27B — without modifying or re-quantizing the original weights by **[`OrcaRouter research team`](https://www.orcarouter.ai)**.**

This repository applies refusal-direction ablation to [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit) entirely **at runtime**.

The original Bonsai pack remains **bit-identical**.

* **27B parameters**
* **~1.72 bits/weight**
* **0 modified weights**
* **0 re-quantization**
* **0 additional weight quantization error**
* **Runtime-adjustable ablation strength**
* **129 residual intervention sites**
* **Apple Silicon / MLX**

You supply the original Ternary Bonsai 2 pack. This repository supplies the refusal direction and the runtime that applies it.

> **Keep the weights compressed. Change the behavior at inference.**

---

## How it works

Traditional abliteration modifies model weights by orthogonalizing matrices that write into the residual stream against a learned refusal direction `r`:

```text
W ← W - r(rᵀW)
```

For a conventional BF16 or FP16 model, the resulting matrix can simply be saved as a new checkpoint.

Ternary Bonsai 2 is different.

Its weights are aggressively quantized. The affine container stores:

```text
scale = s
bias  = -s
```

so the 2-bit codes:

```text
{0, 1, 2}
```

decode to:

```text
{-s, 0, +s}
```

The model's quality at approximately **1.72 bits per weight** is the result of quantization-aware training.

Orthogonalizing one of these matrices produces a dense, full-precision matrix. Saving that result back into the ternary representation would therefore require **re-quantization**.

And simply re-quantizing the edited weights does not reproduce the quantization-aware training process that produced the original model.

So we don't edit the weights.

### Runtime ablation

Instead, the equivalent projection is applied when each residual contribution is produced:

```text
y ← y - α · dot(y, r) · r
```

where:

```text
y = residual contribution
r = normalized refusal direction
α = intervention strength
```

At `alpha=1`, the component of each residual write parallel to the refusal direction is removed.

Conceptually:

```text
      Original Ternary Bonsai 2
              27B
               │
               │
        bit-identical weights
               │
               ▼
      ┌──────────────────┐
      │  packed matmul   │
      └────────┬─────────┘
               │
               │ y
               ▼
      ┌──────────────────┐
      │ Runtime Ablation │
      │                  │
      │ y ← y - α(y·r)r │
      └────────┬─────────┘
               │
               ▼
         residual stream
               │
               ▼
            output
```

The projection runs in float32.

The original ternary weights are never modified.

---

# Quickstart

## Apple Silicon

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
  "your prompt"
```

By default:

```text
alpha = 1
```

which enables the full runtime projection.

---

# Verify the ablation

Run:

```bash
python scripts/selfcheck.py \
  --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit
```

`selfcheck.py` measures how much of the residual stream lies along the refusal direction before and after intervention.

A correctly instrumented setup should drive the remaining component to approximately:

```text
~1e-6 of the residual norm
```

If the number barely changes, the residual writers were probably not wrapped correctly.

The self-check also verifies that all expected intervention sites are present.

Expected:

```text
129 residual writers
```

---

# Original vs. uncensored

Because the intervention happens entirely at runtime, the same model pack can be run with or without ablation.

### Original behavior

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
  --alpha 0 \
  "your prompt"
```

### Full runtime ablation

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
  --alpha 1 \
  "your prompt"
```

There is no second checkpoint.

```text
alpha = 0
     │
     └── Original model behavior

alpha = 1
     │
     └── Full refusal-direction projection
```

This also makes A/B testing straightforward because both configurations use the **same underlying weights**.

---

# Why not release modified weights?

Because modifying the weights defeats one of the most interesting properties of Bonsai.

A conventional abliteration performs:

```text
W' = W - r(rᵀW)
```

But `W'` is no longer ternary.

It contains arbitrary floating-point values.

To store it in the original pack, we would need something approximately equivalent to:

```text
ternary(
    W - r(rᵀW)
)
```

That introduces a new quantization step.

The original Bonsai model, however, achieved its compression through **quantization-aware training**, not through naïve post-training conversion of arbitrary dense matrices.

Instead we preserve:

```text
W
```

exactly and transform its output:

```text
y = Wx

y' = y - α(y·r)r
```

The stored model therefore remains untouched.

---

# 129 intervention sites

One important implementation detail is that **wrapping only `o_proj` is not enough**.

Ternary Bonsai 2 uses a hybrid architecture containing:

```text
48 linear-attention layers
16 full-attention layers
64 MLP blocks
```

Each of these can write into the residual stream.

The runtime therefore wraps:

| Residual writer        |   Count |
| ---------------------- | ------: |
| `mlp.down_proj`        |      64 |
| `linear_attn.out_proj` |      48 |
| `self_attn.o_proj`     |      16 |
| `model.embed_tokens`   |       1 |
| **Total**              | **129** |

In other words:

```text
64 + 48 + 16 + 1 = 129
```

Wrapping only:

```text
self_attn.o_proj
```

would intercept only 16 of these sites.

`run.py` prints the number of wrapped residual writers.

`selfcheck.py` warns when it does not detect the expected:

```text
129
```

---

# Do not touch the Hadamard transform

Another important detail is the basis in which the projection is applied.

The Bonsai pack keeps projections in a rotated basis on their **input dimension**.

Its runtime compensates for that rotation on the activation side.

The refusal projection in this repository operates on the **outputs** of those projections.

Those outputs are already in the normal hidden basis.

Likewise, the embedding output is un-rotated by the pack's own `Packed` implementation.

Therefore the refusal direction is simply an ordinary:

```text
5120-dimensional vector
```

No additional Hadamard rotation should be applied to the direction.

Doing so would project against the wrong basis.

---

# Use the model's bundled runtime

The pack must be loaded using **its own bundled runtime**.

An ordinary MLX loader may appear to load the model successfully while silently producing incorrect computation because it does not apply the activation transformations required by the stored weights.

This repository handles that through:

```text
bonsai_abliterate.pack
```

including the detail that:

```text
vision_artifact.load_vl_model
```

is the loader that accepts the:

```text
schema_version: 2
```

configuration used by these packs.

If outputs look completely wrong before ablation is even enabled, verify the pack-loading path first.

---

# Tuning `alpha`

The intervention does not have to be binary.

`alpha` controls how strongly the refusal direction is removed.

| `alpha` | Effect                                     |
| ------: | ------------------------------------------ |
|     `0` | Projection disabled; original behavior     |
|   `0.5` | Partial ablation                           |
|   `0.7` | Moderate ablation                          |
|   `0.9` | Strong ablation                            |
|   `1.0` | Full projection; default                   |
|  `>1.0` | Over-projection; may degrade model quality |

For example:

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
  --alpha 0.7 \
  "your prompt"
```

This makes the intervention a **runtime control parameter** rather than a permanent property of a checkpoint.

---

# Layer-selective ablation

The projection can also be restricted to specific layers.

For example:

```bash
python run.py \
  --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
  --layers 20,21,22,23 \
  "your prompt"
```

Layer selection provides another dimension for exploring the trade-off between behavioral intervention and capability retention.

Instead of producing many different checkpoints, experiments can vary:

```text
direction
×
alpha
×
layers
```

against the same immutable model pack.

---

# Direction files

The `directions/` directory contains:

| File                      | Purpose                                  |
| ------------------------- | ---------------------------------------- |
| `refusal_dir.safetensors` | MLX / Python                             |
| `refusal_dir_fp32.bin`    | Swift / C                                |
| `direction.json`          | Dimensions, application rule and caveats |

The Safetensors file stores the vector under:

```text
direction
```

The raw binary representation contains:

```text
5120 × little-endian Float32
```

with no header.

---

# Direction transfer caveat

The current refusal direction was estimated from the **BF16 base model** from which this Bonsai pack was trained.

The architecture and hidden basis are identical.

However, the degree to which the learned direction transfers through the model's quantization-aware training has **not yet been fully measured**.

This distinction matters.

The runtime can verify mathematically that it is removing the supplied direction from the residual stream.

That does not, by itself, prove that the transferred direction captures exactly the same behavioral feature in the QAT model.

Behavioral evaluation should therefore sweep:

```text
alpha
```

and, where useful:

```text
layers
```

before drawing conclusions about transfer quality.

---

# iOS / mlx-swift

A Swift reference implementation is included at:

```text
swift/RefusalAblation.swift
```

Load:

```text
directions/refusal_dir_fp32.bin
```

as:

```text
5120 little-endian Float32
```

and wrap each residual writer with:

```text
AblatedLinear
```

The pack's `PACK-RUNTIME.md` currently states that Swift support is layer-level only and that full-model loading still requires model integration.

Once the model integration exists, the refusal intervention itself is small.

Per residual write it adds approximately:

```text
1 × dot product
1 × AXPY
```

No alternative model weights need to be stored.

---

# Running on x86 Linux

The pack's quantized matrix multiplication currently has kernels for:

```text
Metal
CPU
```

but not CUDA through `mlx-cuda`.

Attempting to use the required operation on CUDA currently results in:

```text
QuantizedMatmul has no CUDA implementation
```

As a result, putting this particular MLX pack on an NVIDIA machine does not currently provide the expected GPU acceleration.

VRAM is not the primary constraint.

CPU inference works, but a forward pass for a 27B model can take minutes.

That makes the Linux CPU path useful for:

* implementation testing
* behavior verification
* reproducibility checks

rather than high-throughput serving.

Build:

```bash
docker build \
  -t orcarouter-ternary-bonsai-2-27b-uncensored \
  -f docker/Dockerfile .
```

Run:

```bash
PACK=/path/to/Ternary-Bonsai-2-27B-mlx-2bit \
docker/run.sh \
  python run.py \
    --pack /pack \
    --max-new 32 \
    "your prompt"
```

---

# Architecture

The key distinction of this release is that **uncensoring is a runtime property rather than a checkpoint property**.

```text
┌─────────────────────────────────────────────┐
│       Ternary Bonsai 2 · 27B               │
│                                             │
│       ~1.72 bits / weight                   │
│       original packed weights              │
└──────────────────┬──────────────────────────┘
                   │
                   │ weights remain
                   │ bit-identical
                   ▼
┌─────────────────────────────────────────────┐
│             Runtime Inference               │
│                                             │
│  residual writer                            │
│        │                                    │
│        ▼                                    │
│  y = writer(x)                              │
│        │                                    │
│        ▼                                    │
│  y' = y - α · dot(y,r) · r                  │
│        │                                    │
│        ▼                                    │
│  residual stream                            │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
                 output
```

### Properties

```text
Model parameters           27B
Effective weight size      ~1.72 bits/weight
Hidden dimension           5120

Residual intervention
sites                       129

Weight modification        None
Weight re-quantization     None
Additional weight
quantization error         None

Intervention strength      Runtime adjustable
Layer selection            Runtime adjustable
Original behavior          alpha=0
Full projection            alpha=1
```

---

# Why runtime intervention is useful

Permanent weight editing couples a behavioral modification to a particular checkpoint.

Runtime intervention separates the two.

```text
                 MODEL
                   │
            immutable weights
                   │
                   ▼
              INFERENCE
                   │
          ┌────────┴────────┐
          │                 │
      alpha = 0         alpha = 1
          │                 │
          ▼                 ▼
      original          ablated
      behavior          behavior
```

The same architecture can potentially support more than one learned direction without generating another copy of the underlying model.

Conceptually:

```text
Immutable Model
      │
      ▼
Runtime Intervention
      │
      ├── direction
      ├── strength
      └── layer scope
      │
      ▼
Inference
```

This repository currently implements **refusal-direction ablation**.

---

# What this release is — and isn't

This is **not a newly trained 27B model**.

It is a runtime behavioral intervention for the existing Ternary Bonsai 2 27B MLX pack.

The underlying model architecture, QAT training, ternary representation and original packed weights come from the upstream Bonsai release.

The contribution here is the runtime refusal-direction implementation, direction artifacts, residual-writer instrumentation and verification tooling.

That distinction is intentional:

> **The model stays immutable. Behavioral intervention happens at inference.**

---

# Responsible use

Removing a learned refusal direction can cause the model to respond to requests that the original model would decline.

The technique should therefore be treated as a research and inference-control mechanism, not as evidence that every resulting output is safe, correct or appropriate.

Deployments should apply their own access controls, policy enforcement and security boundaries appropriate to their use case.

---

# Credits

Built on:

**Ternary Bonsai 2 27B** by `prism-ml`

Runtime refusal-direction implementation and tooling by **[`OrcaRouter`](https://www.orcarouter.ai)**.

The original model pack is not redistributed or modified by this repository.

---

# License

Apache-2.0, matching the underlying model.
