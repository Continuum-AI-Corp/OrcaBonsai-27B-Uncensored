# bonsai2-abliterate

Refusal-direction ablation for [`prism-ml/Ternary-Bonsai-2-27B-mlx-2bit`](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-mlx-2bit),
applied at run time. **The pack's weights are never modified** — they stay bit-identical,
so the ablation adds no quantization error at all.

You supply the pack. This repo supplies the direction and the code that applies it.

## Quickstart

On Apple Silicon:

```bash
pip install -r requirements.txt
python run.py --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit "your prompt"
```

Check it is actually working:

```bash
python scripts/selfcheck.py --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit
```

`selfcheck.py` reports how much of the residual stream still lies along the refusal
direction, before and after. A correct setup drives it to about `1e-6` of the residual
norm; a number that barely moves means the residual writers were not wrapped.

Compare against the untouched model with `--alpha 0`, which disables the projection:

```bash
python run.py --pack ... --alpha 0 "your prompt"     # original behaviour
python run.py --pack ... --alpha 1 "your prompt"     # ablated
```

## Why the weights are not edited

Abliteration normally orthogonalises every matrix that writes the residual stream
against the refusal direction, `W <- W - r (r^T W)`, and saves the result. That is not
available here.

Bonsai's weights are **ternary**: the affine container stores `scale = s` and
`bias = -s`, so the 2-bit codes `{0,1,2}` decode to exactly `{-s, 0, +s}`. The
orthogonalised matrix is dense and full precision, so storing it back would mean
re-quantizing to ternary — and the model's quality at 1.72 bits/weight is the product of
quantization-aware training, not of the format. Re-quantizing without that training is
what destroys the model.

Applying the same operator at the point of use is algebraically identical:

```
y <- y - alpha * dot(y, r) * r          on every residual write
```

Removing the `r` component from everything written into the residual stream is the same
as removing it from the stream itself. The packed weights are untouched, the projection
runs in float32, and `alpha` stays adjustable — none of which a baked edit offers.

## Two things to get right when integrating

- **Wrap every residual writer, not just `o_proj`.** This is a hybrid architecture: 48
  linear-attention layers and 16 full-attention layers. The writers are each layer's
  `mlp.down_proj` (64), `linear_attn.out_proj` (48), `self_attn.o_proj` (16), and
  `model.embed_tokens` — **129 sites**. Wrapping only `o_proj` misses three quarters of
  the attention writes. `run.py` prints the count; `selfcheck.py` warns if it is not 129.
- **Do not touch the Hadamard transform.** The pack keeps every projection in a rotated
  basis on its *input* dimension and its runtime compensates on the activation side. The
  vectors projected here are *outputs*, which are already in the plain hidden basis, and
  the embedding output is un-rotated by the pack's own `Packed` module. The direction is
  an ordinary 5120-d vector; no rotation belongs in this path.

Also worth knowing: the pack must be loaded through **its own bundled runtime**. An
ordinary MLX loader will appear to work and silently compute nonsense, because it does
not apply the activation transform the stored weights require. `bonsai_abliterate.pack`
handles this, including the detail that only `vision_artifact.load_vl_model` accepts the
`schema_version: 2` config these packs use.

## Tuning `alpha`

| `alpha` | effect |
|---|---|
| `0` | projection off; the pack's original behaviour |
| `1` | matches a full permanent weight orthogonalisation (default) |
| `0.5`–`0.9` | weaker ablation, more of the original behaviour retained |
| `> 1` | over-projection; can degrade fluency |

`--layers 20,21,...` restricts the projection to selected layers, which is another way
to trade strength for capability retention.

## iOS / mlx-swift

`swift/RefusalAblation.swift` is a reference implementation: load
`directions/refusal_dir_fp32.bin` (5120 little-endian Float32, no header) and wrap each
residual writer with `AblatedLinear`.

Note that the pack's `PACK-RUNTIME.md` states Swift support is layer-level only and
full-model loading still requires model integration. Since that has to be written
anyway, the projection costs one dot product and one axpy per residual write.

## What's in `directions/`

| File | For |
|---|---|
| `refusal_dir.safetensors` | MLX / Python; tensor key `direction` |
| `refusal_dir_fp32.bin` | Swift / C; 5120 little-endian Float32, no header |
| `direction.json` | dimensions, application rule, and caveats |

The direction was estimated on the bf16 base model this pack was trained from. The
architecture and hidden basis are identical, but **how well it transfers across the
quantization-aware training has not been measured yet** — if the effect is weaker than
expected, sweep `alpha` upward before concluding the direction is wrong.

## Running on x86 Linux

The pack's quantized matmul has kernels for Metal and CPU only. `mlx-cuda` has none
(`QuantizedMatmul has no CUDA implementation`), so an NVIDIA GPU does not help and VRAM
is not the constraint. A CPU forward pass of this 27B model takes minutes — fine for
checking behaviour, not for serving.

```bash
docker build -t bonsai2-abliterate -f docker/Dockerfile .
PACK=/path/to/Ternary-Bonsai-2-27B-mlx-2bit docker/run.sh \
    python run.py --pack /pack --max-new 32 "your prompt"
```

## License

Apache-2.0, matching the model.
