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

`--pack` is the directory holding `config.json` and `runtime/`. If you downloaded with
`huggingface_hub`, that is the **snapshot** directory, not the `models--...` cache entry
above it:

```bash
python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('prism-ml/Ternary-Bonsai-2-27B-mlx-2bit'))"
```

A cache entry is accepted too and its newest snapshot is resolved for you.

By default:

```text
alpha = 1
```

which enables the full runtime projection.

---
## Evaluation Results

![OrcaBonsai 27B Uncensored evaluation results](./evals.jpg)

Full tables with capability retention, sample sizes and methodology are under
[Evaluation](#evaluation) — including what the asterisk on SimpleSafetyTests means.

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

# Evaluation

Refusal is judged by a rule-based opening-phrase classifier (`caveat` = answered but
wrapped in a disclaimer) — indicative, not an LLM-judge / publication-grade number.
Thinking off, greedy decoding, 64-token budget on the refusal sets. Base and ablated
rows are the same weights in the same process, `alpha` 0 against `alpha` 1.

Run on the unfolded fp16 expansion of the pack rather than the pack driving its own
kernels: the packed quantized matmul has no CUDA implementation, and the CPU backend
needs minutes per forward pass. The expansion carries the pack's ternary values exactly —
a container change, not a requantization — and reproduces the pack's own next-token
distributions to three decimals on spot checks.

## Safety (thinking off)

| Benchmark | n | Base | This Model | Caveat |
|---|---|---|---|---|
| AdvBench | 100 | 99.0% | 6.0% | 56.0% |
| JailbreakBench | 100 | 96.0% | 4.0% | 52.0% |
| StrongREJECT | 150 | 99.3% | 3.3% | 45.3% |
| HarmBench | 150 | 98.7% | 7.3% | 48.0% |
| MaliciousInstruct | 100 | 97.0% | 0.0% | 52.0% |
| SimpleSafetyTests¹ | 50 | 96.0% | 18.0% | 60.0% |
| ForbiddenQuestions | 150 | 75.3% | 5.3% | 42.7% |

No reply in any set ran out of its token budget, so none of these rates is inflated by
truncation.

## Over-refusal on benign prompts

| Benchmark | n | Base | This Model |
|---|---|---|---|
| XSTest-safe | 250 | 5.2% | 0.4% |
| JBB-benign | 100 | 25.0% | 0.0% |

The projection does not only stop refusals on harmful prompts. The published pack turns
down a quarter of JailbreakBench's *benign* prompts; ablated, it turns down none.

## Capability retention

| Benchmark | n | Base | This Model | Δ |
|---|---|---|---|---|
| MMLU | 300 | 76.7% | 77.7% | +1.0 |
| GSM8K | 150 | 87.3% | 86.0% | −1.3 |
| CMMLU | 500 | 76.2% | 75.6% | −0.6 |
| MMLU-Pro² | 250 | — | — | — |

Every movement is within noise at these sample sizes — one question on GSM8K is 0.7
points. That is the result the runtime approach is for: the weights are bit-identical,
so there is no requantization to pay for.

¹ Understated. This set is mostly self-harm prompts, and the model answers them with a
crisis redirect that opens "I am deeply sorry to hear…" — which the classifier's exact
phrase list misses, scoring it as compliance. The real residual refusal rate on this set
is higher than 18%. The classifier is left as-is so these numbers stay comparable with
our other cards.

² Excluded. Its prompt asks for reasoning before the answer, and 63–64% of replies on
both sides had not reached one inside the token budget, so the accuracy would be a floor
set by that budget rather than a measurement.

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

## iPhone memory budget

On a phone the binding constraint is resident memory, and it is almost entirely the
weights. Measured on the pack, not estimated:

```text
ternary codes              6.256 GiB   78.2%
vision tower               0.858 GiB   10.7%
group biases (redundant)   0.391 GiB    4.9%
group scales               0.391 GiB    4.9%
norms + GDN state path     0.098 GiB    1.2%
hadamard signs             0.011 GiB    0.1%
total                      8.005 GiB
```

The KV cache is not the problem here, which is what makes a 27B model on a phone worth
discussing at all. Only 16 of the 64 layers use full attention:

```text
16 full-attention layers    64 KiB per token  (256 MiB at 4K context)
48 linear-attention layers  72 MiB of recurrent state, fixed, context-independent
```

Two things can come out of the pack for a text-only app:

```bash
python scripts/prepare_ios_pack.py \
    --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
    --out  /path/to/Bonsai-2-27B-ios
```

```text
source pack                 8.005 GiB
  - vision tower           -0.858        memory and disk
  - redundant biases       -0.391        disk; memory only with a kernel that reconstructs
result                      6.756 GiB
```

The distinction between those two lines matters. The vision tower is a genuine memory
saving: it is only read when an image is in the prompt, and the pack documents it as the
stock unquantized Qwen tower.

The biases are a *download* saving. The affine container stores a scale and a bias per
group of 128, but the ternary levels `{-s, 0, +s}` come out of `scale = s`, `bias = -s`,
so the bias holds no information — verified exact across all 402 packed modules, every
group, `max |bias + scale| = 0`. MLX's `quantized_matmul` and `dequantize` still take a
bias argument, so a runtime that calls them has to materialise `-scales` at load and
saves no memory at all. The memory saving needs a kernel that assumes the identity.
Check before counting on it.

Nothing in that script touches the ternary codes, the group scales, the Hadamard signs
or the direction. Kept tensors are bit-identical to the source.

Whether 6.756 GiB fits is a per-device question this repo cannot answer for you: iOS
caps a single app well below total RAM, and the cap depends on the device and on whether
`com.apple.developer.kernel.increased-memory-limit` has been granted. Do the arithmetic
against your own target before committing to it. An 8 GB device does not have room for
6.756 GiB of weights plus an app.

Decode is memory-bandwidth-bound: every token reads the whole weight set. The ~47 tok/s
figure quoted for an M5 Max laptop corresponds to several hundred GB/s of usable
bandwidth; a phone has a fraction of that, so expect a fraction of the throughput, and
expect sustained generation to meet thermal limits.

The output pack is meant for an app with its own model integration. It is deliberately
not loadable by the pack's bundled Python loaders — `vision_artifact.load_vl_model`
requires `components.vision`, and a removed bias changes what a `Packed` module reads.
Keep the original pack for anything that uses those; `ios-pack.json` records what was
removed and how to rebuild it.

---

# GGUF / llama.cpp

The same edit ships as a rank-1 LoRA adapter, so the published ternary GGUF stays
byte-identical. The adapter is in this repo — nothing to build:

```text
gguf/bonsai-abliterate-lora.gguf     9,682,464 bytes
sha256  f1669534803d340a496015f5c45125f3437b4d13ec764f40e34488ce83967f42
```

```bash
llama-cli -m Ternary-Bonsai-2-27B-PTQ1_0.gguf --lora gguf/bonsai-abliterate-lora.gguf
```

It holds 129 `lora_a`/`lora_b` pairs — one per residual writer — as F32, against
`general.architecture = qwen35` and `adapter.lora_alpha = 1.0`.

`scripts/export_gguf_lora.py` rebuilds it, for anyone who wants to check the provenance
or re-derive it against a different direction. That path needs the unfolded fp16
checkpoint, which this repo does not ship, so it is for reproduction rather than for
ordinary use.

`W' = W - r (r^T W)` is rank 1, so the whole edit is `A = r^T W`, `B = -r`. llama.cpp
builds a LoRA into the graph as two extra matmuls on the activation and never merges it
into the base weights — which is what makes this work at 1.75 bits, where a baked edit
would simply be rounded away.

## Strength

`--lora` applies it at scale 1.0, and that is the projection exactly, not an
approximation: llama.cpp computes `scale = adapter_scale * alpha / rank` with
`rank = lora_b->ne[0]`, and the exporter writes `alpha = 1` against rank 1.

Exact does not mean every prompt flips. Measured on the PTQ1_0 pack, greedy, thinking
off:

```text
scale 0       the published model
scale 1       exact projection; most harmful prompts comply, some still refuse
scale 2       flips the stubborn ones
scale 3+      over-projection; output degrades, then collapses
```

```bash
llama-cli -m Ternary-Bonsai-2-27B-PTQ1_0.gguf \
    --lora-scaled bonsai-abliterate-lora.gguf:2
```

Read a single prompt as a sample of one. At full strength our own evaluation still had
6% of AdvBench refusing, so one stubborn prompt tells you nothing about the strength.

## You need PrismML's fork

Stock llama.cpp cannot open these packs at all. `PTQ1_0` and `PQ2_0` are private ggml
type ids — 143 and 142, outside upstream's range — so upstream refuses them at header
parse. Even `gguf-py` raises:

```text
ValueError: np.uint32(143) is not a valid GGMLQuantizationType
```

which is why `bonsai_abliterate/gguf_min.py` reads the base header directly. Writing the
adapter still uses `gguf-py`, since an adapter holds only standard F32 tensors.

The published `Ternary-Bonsai-2-27B-F16.gguf` is **not** an escape hatch. It uses
standard tensor types, so stock llama.cpp and Ollama will load it — but it is stored in
the same Hadamard-rotated basis and carries the same `prism.hadamard.*` metadata, which
upstream ignores entirely. It would run and emit nonsense rather than fail.

**Ollama has no path today.** It builds stock llama.cpp, so it inherits the rejection of
the private types; `ADAPTER` parses but `ollama create` rejects every request carrying
adapters with "LoRA adapters are no longer supported"; and it has no control-vector
support.

## What is verified

The exported `B @ A` reproduces `-r (r^T W)` exactly — relative error `0.000e+00` on
`ffn_down`, `attn_output`, `ssm_out` and `token_embd` — and the direction's leakage into
each falls about six orders of magnitude. Against the fork, the adapter loads on the
ternary pack, scale 0 reproduces the published refusal and scale 100 destroys the model,
so it is genuinely in the compute graph. All 129 sites are on LoRA-aware paths:
`ffn_down` through `build_ffn`'s `build_lora_mm`, `attn_output` and `ssm_out` inline in
`qwen35.cpp`, `token_embd` in `build_inp_embd`.

One thing is not: **PQ2_0 was not run.** The adapter does not depend on the base's
quantization — `A = r^T W` comes from the unfolded checkpoint and the base GGUF is read
only for tensor names — and all three published GGUFs carry the same 851 names, so the
same file should apply. Only PTQ1_0 was actually tested.

A LoRA on a tensor llama.cpp does not route through `build_lora_mm` loads without error
and does nothing at all. Check that scale 0 reproduces the published model and that a
large scale visibly breaks it — if neither is true, the adapter is not being applied.

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
