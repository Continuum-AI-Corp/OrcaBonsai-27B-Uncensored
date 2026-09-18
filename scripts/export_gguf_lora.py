#!/usr/bin/env python3
"""Export the abliteration as a rank-1 LoRA adapter for llama.cpp.

The published ternary GGUFs stay byte-identical. llama.cpp builds a LoRA into the graph
as two extra matmuls on the activation and never merges it into the base weights, so a
rank-1 adapter is an exact, reversible way to carry the edit:

    W' = W - r (r^T W)        is rank 1, so        A = r^T W,  B = -r,  Delta_W = B @ A

Load it with `--lora`, dial it with `--lora-scaled adapter.gguf:0.5`, drop it to return
to the published behaviour. Nothing is requantized, which matters here: baking this edit
into ternary weights is a no-op -- the edit is ~1.4% of ||W|| against a grid step of
1.7-2.4x a typical weight, so re-quantizing rounds it away and leaves 99.4% of the
direction in place.

Two things about this model would silently produce a broken adapter, and both were
settled by measurement rather than assumption:

* **The adapter is authored in the unfolded basis, with no Hadamard applied.** The base
  weights are Hadamard-folded, but the fork's `build_lora_mm` rotates the activation only
  for the base matmul and hands the LoRA branch the unrotated activation. So A comes
  straight from the unfolded fp16 checkpoint.
* **`ssm_out` needs no permutation.** Its input dimension was suspected to be V-head
  reordered relative to the checkpoint. Comparing the dequantized, unfolded GGUF tensor
  against the checkpoint gives 2.08e-04 under the identity and 1.38 under either
  permutation -- the same residual as the `ffn_down` control, which is just the
  checkpoint's fp16 storage. No reorder. `--check` re-runs that comparison on your files.

    python scripts/export_gguf_lora.py \
        --checkpoint /path/to/Bonsai-2-27B-unfolded-fp16 \
        --base-gguf  /path/to/Ternary-Bonsai-2-27B-PTQ1_0.gguf \
        --out        bonsai-abliterate-lora.gguf

Then:

    llama-cli -m Ternary-Bonsai-2-27B-PTQ1_0.gguf --lora bonsai-abliterate-lora.gguf

with PrismML's llama.cpp fork -- stock llama.cpp cannot open the base pack at all, since
its tensor type id is private to that fork.

## Strength

`--lora` applies the adapter at scale 1.0, which is the projection exactly: llama.cpp
computes `scale = adapter_scale * alpha / rank` with `rank = lora_b->ne[0]`, and this
writes `alpha = 1` against rank 1. It is the same operator as a full weight
orthogonalisation, not an approximation of one.

Exact does not mean every prompt flips. In our own evaluation of the same direction at
full strength, 6% of AdvBench prompts still refused, and a stubborn prompt looks
identical at scale 0 and scale 1 while flipping at scale 2. Read a single prompt as a
sample of one, not as a calibration.

Measured on the PTQ1_0 pack with the fork, greedy, thinking off:

    scale 0     the published model: "I cannot provide instructions on how to..."
    scale 1     exact projection; most harmful prompts comply, some still refuse
    scale 2     flips the stubborn ones
    scale >= 3  over-projection; output degrades, then collapses

So `--lora-scaled adapter.gguf:2` is the knob if you want a harder ablation, and 3 is
past the useful range. There is no equivalent of this dial in a baked edit.

## What was and was not verified

Verified here: the exported `B @ A` reproduces `-r (r^T W)` exactly, relative error
0.000e+00 on ffn_down, attn_output, ssm_out and token_embd, and the direction's leakage
into each falls about six orders of magnitude. Verified end to end against the fork: the
adapter loads on the ternary pack, scale 0 reproduces the published refusal and scale 100
destroys the model, so it is genuinely in the compute graph. Every target family is on a
LoRA-aware path -- `ffn_down` through `build_ffn`'s `build_lora_mm`, `attn_output` and
`ssm_out` inline in `qwen35.cpp`, `token_embd` in `build_inp_embd`.

Not verified: PQ2_0. The adapter does not depend on the base's quantization -- A comes
from the unfolded checkpoint and the base GGUF is read only for tensor names -- and all
three published GGUFs carry the same 851 names, so the same adapter should apply. Only
PTQ1_0 was actually run.
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bonsai_abliterate.gguf_min import Gguf

# GGUF residual-writer suffix -> how to find the same tensor in the HF checkpoint.
WRITERS = {
    "ffn_down.weight": "mlp.down_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}
EMBED_GGUF = "token_embd.weight"
EMBED_HF = "model.language_model.embed_tokens.weight"
DEFAULT_DIRECTION = Path(__file__).resolve().parents[1] / "directions" / "refusal_dir.safetensors"


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True,
                    help="unfolded fp16 HF checkpoint (the adapter is authored from this)")
    ap.add_argument("--base-gguf", required=True,
                    help="the ternary GGUF the adapter will be loaded against; read for "
                         "its exact tensor names and architecture")
    ap.add_argument("--direction", default=str(DEFAULT_DIRECTION))
    ap.add_argument("--out", default="bonsai-abliterate-lora.gguf")
    ap.add_argument("--no-embedding", action="store_true",
                    help="skip token_embd, leaving the 128 projection writers")
    ap.add_argument("--pack", default=None,
                    help="the MLX pack, needed only by --check (for its Hadamard sign "
                         "vectors and its PTQ1_0 transcoder)")
    ap.add_argument("--check", action="store_true",
                    help="verify the checkpoint really matches the base pack, and that "
                         "ssm_out needs no permutation, before writing anything")
    return ap.parse_args()


def load_direction(path):
    from safetensors.numpy import load_file

    d = load_file(path)["direction"].astype(np.float32).reshape(-1)
    return d / np.linalg.norm(d)


class Checkpoint:
    """Read one tensor at a time from a sharded HF checkpoint.

    Per tensor rather than per shard on purpose: the checkpoint carries a bf16 vision
    tower alongside the fp16 language model, and numpy has no bf16, so loading a whole
    shard raises "data type 'bfloat16' not understood" even when every tensor we want
    from it is fp16. It is also a great deal lighter on memory.
    """

    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def __getitem__(self, name):
        from safetensors import safe_open

        with safe_open(str(self.root / self.index[name]), framework="np") as f:
            return f.get_tensor(name).astype(np.float32)


def gguf_to_hf(name: str) -> str | None:
    if name == EMBED_GGUF:
        return EMBED_HF
    for suffix, hf_suffix in WRITERS.items():
        if name.endswith(suffix) and name.startswith("blk."):
            layer = name.split(".")[1]
            return f"model.language_model.layers.{layer}.{hf_suffix}"
    return None


def run_check(base: Gguf, ckpt: Checkpoint, pack_dir: str | None) -> None:
    """Confirm the checkpoint is the same weights as the pack, and settle ssm_out.

    Needs the MLX pack for its Hadamard sign vectors and its PTQ1_0 transcoder, so this
    is opt-in rather than part of every export.
    """
    if not pack_dir:
        raise SystemExit("--check needs --pack pointing at the MLX pack (for the sign "
                         "vectors and runtime/codec.py)")
    sys.path.insert(0, str(Path(pack_dir) / "runtime"))
    import mlx.core as mx
    from codec import transcode
    from runtime import fwht

    weights = mx.load(str(Path(pack_dir) / "model.safetensors"))
    signs = {}
    for k, v in weights.items():
        if k.endswith(".signs"):
            signs.setdefault(int(v.shape[0]), v)

    def unfolded(name):
        shape = base.shape_out_in(name)
        words, sc, bi = transcode(base.raw(name), shape, base.tensors[name]["type_name"])
        rot = mx.dequantize(mx.array(words), mx.array(sc), mx.array(bi),
                            group_size=128, bits=2).astype(mx.float32)
        return np.asarray(fwht(rot, 1024, signs[shape[1]], inverse=True))

    def rel(a, b):
        return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))

    print("[check] control -- ffn_down should match with no permutation")
    a = unfolded("blk.0.ffn_down.weight")
    b = ckpt["model.language_model.layers.0.mlp.down_proj.weight"]
    control = rel(a, b)
    print(f"[check]   identity rel = {control:.3e}")
    if control > 1e-2:
        raise SystemExit("the checkpoint does not match the base pack; wrong --checkpoint?")

    print("[check] ssm_out -- is its input dimension permuted?")
    a = unfolded("blk.0.ssm_out.weight")
    b = ckpt["model.language_model.layers.0.linear_attn.out_proj.weight"]
    nv, nk, hd = 48, 16, 128
    vperm = np.arange(nv * hd).reshape(nv // nk, nk, hd).transpose(1, 0, 2).reshape(-1)
    scores = {"identity": rel(a, b),
              "vperm": rel(a[:, vperm], b),
              "inverse vperm": rel(a[:, np.argsort(vperm)], b)}
    for k, v in scores.items():
        print(f"[check]   {k:<14} rel = {v:.3e}")
    best = min(scores, key=scores.get)
    if best != "identity":
        raise SystemExit(f"ssm_out appears to need the '{best}' permutation on this pack; "
                         "this exporter assumes identity -- stop and re-derive")
    print("[check] identity confirmed, no permutation needed")


def main():
    a = parse()
    base = Gguf(a.base_gguf)
    arch = base.kv.get("general.architecture")
    print(f"[base] {a.base_gguf}")
    print(f"[base] arch={arch}  tensors={len(base.tensors)}  gguf v{base.version}")
    if base.kv.get("prism.hadamard.version") is None:
        print("[base] WARNING: no prism.hadamard.* metadata -- is this really the folded pack?")

    ckpt = Checkpoint(a.checkpoint)
    r = load_direction(a.direction)
    print(f"[dir] {a.direction}  {r.size} dims")

    if a.check:
        run_check(base, ckpt, a.pack)

    targets = []
    for name in base.tensors:
        hf = gguf_to_hf(name)
        if hf is None:
            continue
        if name == EMBED_GGUF and a.no_embedding:
            continue
        targets.append((name, hf))
    targets.sort()

    by_kind = {}
    for name, _ in targets:
        kind = EMBED_GGUF if name == EMBED_GGUF else name.split(".", 2)[2]
        by_kind[kind] = by_kind.get(kind, 0) + 1
    print(f"[targets] {len(targets)} sites: {by_kind}")

    import gguf

    writer = gguf.GGUFWriter(a.out, arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora_alpha", 1.0)

    for name, hf in targets:
        W = ckpt[hf]
        if name == EMBED_GGUF:
            # The embedding is looked up, not multiplied: a row IS a residual vector, so
            # the edit is W - (W r) r^T. llama.cpp also stores this one's factors the
            # opposite way round from every other tensor.
            lora_a = (W @ r).reshape(-1, 1).astype(np.float32)   # (n_vocab, 1)
            lora_b = (-r).reshape(-1, 1).astype(np.float32)      # (n_embd, 1)
        else:
            # W is [out, in]; the projection removes r from the output side.
            lora_a = (r @ W).reshape(1, -1).astype(np.float32)   # (1, n_in)
            lora_b = (-r).reshape(-1, 1).astype(np.float32)      # (n_out, 1)
            if W.shape[0] != r.size:
                raise SystemExit(f"{hf} has {W.shape[0]} output rows, direction has "
                                 f"{r.size} -- not a residual writer?")
        writer.add_tensor(f"{name}.lora_a", lora_a)
        writer.add_tensor(f"{name}.lora_b", lora_b)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size = Path(a.out).stat().st_size
    print(f"[done] {a.out}  ({size/1e6:.1f} MB, {2*len(targets)} tensors)")
    print()
    print("Use it against the base pack with PrismML's llama.cpp fork:")
    print(f"  llama-cli -m {Path(a.base_gguf).name} --lora {Path(a.out).name}")
    print(f"  llama-cli -m {Path(a.base_gguf).name} --lora-scaled {Path(a.out).name}:2")
    print()
    print("Scale 1.0 is the projection exactly; scale 2 flips prompts that resist it, and")
    print("3 or more over-projects and degrades the output. Judge the strength on several")
    print("prompts -- at full strength some fraction still refuses, so one prompt tells you")
    print("very little. Scale 0 should reproduce the published model; if it does not, the")
    print("adapter is not what you think it is.")


if __name__ == "__main__":
    main()
