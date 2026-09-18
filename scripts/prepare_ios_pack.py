#!/usr/bin/env python3
"""Slim the pack down for bundling into an iOS app.

On a phone the binding constraint is resident memory, not disk. Two things can come out
of the pack, and they are not equally useful:

**The vision tower — 0.858 GiB, a real memory saving.** It is only needed when an image
is in the prompt. A text-only app never touches it, and the pack documents it as the
stock unquantized Qwen tower, so nothing about the language model changes.

**The group biases — 0.391 GiB, a *disk* saving.** The affine container stores a scale
and a bias per group of 128, but the ternary levels {-s, 0, +s} are reproduced by
scale = s and bias = -s, so the bias carries no information. This script verifies that
identity holds for every group before dropping anything. The caveat matters: MLX's
`quantized_matmul` and `dequantize` take a bias array, so a runtime that calls them has
to materialise `-scales` at load and saves no memory at all -- only download size. The
memory saving needs a kernel that assumes the identity, which the Prism ternary kernels
may already do; check before counting on it.

Nothing here touches the ternary codes, the group scales, the Hadamard signs or the
refusal direction. The intervention stays what it is: a runtime projection.

    python scripts/prepare_ios_pack.py --pack /path/to/Ternary-Bonsai-2-27B-mlx-2bit \
        --out /path/to/Bonsai-2-27B-ios

The result is meant for an app with its own model integration. It is deliberately *not*
loadable by the pack's bundled Python loaders: `vision_artifact.load_vl_model` requires
`components.vision`, and dropping the biases changes what a `Packed` module must read.
Keep the original pack for anything that uses those. `ios-pack.json` records exactly
what was removed and how to rebuild it.
"""
from __future__ import annotations

import sys
import json
import shutil
import argparse
from pathlib import Path

import mlx.core as mx

CARRY = ("config.json", "hadamard.json", "tokenizer.json", "tokenizer_config.json",
         "chat_template.jinja", "generation_config.json", "LICENSE", "NOTICE.txt")
GIB = 2 ** 30


def parse():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", required=True, help="source Ternary-Bonsai-2-27B-mlx-2bit")
    ap.add_argument("--out", required=True, help="destination directory")
    ap.add_argument("--keep-vision", action="store_true",
                    help="keep the 0.858 GiB vision tower (needed for image prompts)")
    ap.add_argument("--keep-biases", action="store_true",
                    help="keep the redundant biases, for a runtime that reads them directly")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty --out")
    return ap.parse_args()


def verify_bias_identity(weights, modules) -> None:
    """Refuse to drop the biases unless bias == -scale exactly, for every group."""
    worst, checked = 0.0, 0
    for path in modules:
        key = "language_model." + path
        s, b = weights[key + ".scales"], weights[key + ".biases"]
        d = mx.max(mx.abs(b.astype(mx.float32) + s.astype(mx.float32)))
        mx.eval(d)
        worst = max(worst, float(d.item()))
        checked += 1
    print(f"[verify] bias == -scale over {checked} modules: max |bias + scale| = {worst:.3e}")
    if worst != 0.0:
        raise SystemExit(
            f"the bias is not exactly -scale (max deviation {worst:.3e}), so dropping it "
            "would change the weights; re-run with --keep-biases"
        )


def main():
    a = parse()
    src, dst = Path(a.pack), Path(a.out)
    if dst.exists() and any(dst.iterdir()) and not a.force:
        raise SystemExit(f"{dst} is not empty; pass --force to overwrite")
    dst.mkdir(parents=True, exist_ok=True)

    config = json.loads((src / "config.json").read_text())
    modules = [rec["path"] for rec in config["modules"]]
    weights = mx.load(str(src / "model.safetensors"))
    print(f"[load] {len(weights)} tensors from {src}")

    if not a.keep_biases:
        verify_bias_identity(weights, modules)

    kept, dropped = {}, {"vision": 0, "biases": 0}
    for key, val in weights.items():
        nbytes = val.size * val.dtype.size
        if not a.keep_vision and not key.startswith("language_model."):
            dropped["vision"] += nbytes
            continue
        if not a.keep_biases and key.endswith(".biases"):
            dropped["biases"] += nbytes
            continue
        kept[key] = val

    total_src = sum(v.size * v.dtype.size for v in weights.values())
    total_dst = sum(v.size * v.dtype.size for v in kept.values())
    mx.save_safetensors(str(dst / "model.safetensors"), kept, metadata={"format": "mlx"})

    for name in CARRY:
        if (src / name).exists():
            shutil.copyfile(src / name, dst / name)

    # The config must stop advertising what is no longer there, or a loader will look for
    # a tower that was removed and fail late instead of early.
    if not a.keep_vision:
        config.setdefault("components", {})["vision"] = False
        config.pop("vision_config", None)
    config["ios_pack"] = {"biases_dropped": not a.keep_biases,
                          "vision_dropped": not a.keep_vision}
    (dst / "config.json").write_text(json.dumps(config, indent=2))

    (dst / "ios-pack.json").write_text(json.dumps({
        "source": str(src),
        "vision_tower_removed": not a.keep_vision,
        "biases_removed": not a.keep_biases,
        "reconstruct_biases": ("biases = -scales, elementwise, per packed module; the "
                              "identity was verified exact before removal"
                              if not a.keep_biases else None),
        "loader_note": ("not loadable by the pack's bundled Python loaders; intended for "
                        "an app with its own model integration"),
        "bytes": {"source": total_src, "result": total_dst,
                  "vision_removed": dropped["vision"], "biases_removed": dropped["biases"]},
    }, indent=2))

    print()
    print(f"{'source pack':<34}{total_src/GIB:7.3f} GiB")
    if dropped["vision"]:
        print(f"{'  - vision tower':<34}{-dropped['vision']/GIB:7.3f}      (memory and disk)")
    if dropped["biases"]:
        print(f"{'  - redundant biases':<34}{-dropped['biases']/GIB:7.3f}      (disk; memory "
              "only with a kernel that reconstructs)")
    print(f"{'result':<34}{total_dst/GIB:7.3f} GiB   -> {dst}")
    print()
    print("Resident memory on device is dominated by these weights. The KV cache is not "
          "the problem on this architecture:")
    t = config["text_config"]
    full = sum(1 for x in t["layer_types"] if x == "full_attention")
    per_tok = 2 * t["num_key_value_heads"] * t["head_dim"] * 2 * full
    print(f"  {full} full-attention layers  -> {per_tok/1024:.0f} KiB per token "
          f"({per_tok*4096/GIB*1024:.0f} MiB at 4K context)")
    lin = len(t["layer_types"]) - full
    state = t["linear_num_value_heads"] * t["linear_value_head_dim"] * t["linear_key_head_dim"] * 2
    print(f"  {lin} linear-attention layers -> {state*lin/GIB*1024:.0f} MiB of recurrent "
          "state, fixed, independent of context length")
    print()
    print("Check your own device's per-app limit before assuming this fits: iOS caps an "
          "app well below total RAM, and the cap depends on the device and on whether "
          "`com.apple.developer.kernel.increased-memory-limit` is granted.")


if __name__ == "__main__":
    main()
