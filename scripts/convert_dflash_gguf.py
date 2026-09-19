#!/usr/bin/env python3
"""Convert a DFlash 2 drafter from GGUF to the layout ``run.py --draft`` loads.

    python scripts/convert_dflash_gguf.py --gguf Bonsai-2-27B-DFlash2-Q8_0.gguf \
        --config config.json --out /path/to/drafter

Why: the only DFlash 2 head fine-tuned against the ternary Bonsai 2 target
(ProCreations/Ternary-Bonsai-2-27B-DFlash2, Apache-2.0) is published as a GGUF for a
patched llama.cpp. Its tensors are standard types (Q8_0 and F32), so stock gguf-py
reads it; this script dequantises them, renames them to the vendored drafter's
parameter names, and writes ``<out>/dflash/model_fp16.safetensors`` next to the
head's ``config.json``. ``dflash.load_drafter`` quantises that checkpoint on load the
same way the published 4-bit head is quantised (4-bit group-64 linears, 8-bit
codebooks), so the two heads compare at equal precision.

Measured against the unadapted head on an M1 Ultra with the ablation on, tokens
accepted per round: code 4.97 -> 5.73, edit 6.80 -> 7.07, prose 1.63 -> 1.73.
"""
from __future__ import annotations

import re
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx

_LAYER = {
    "attn_conv_base": "attention_conv.base_kernel",
    "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_conv_base": "mlp_conv.base_kernel",
    "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
_TOP = {
    "enc.output_norm.weight": "hidden_norm.weight",
    "output_norm.weight": "norm.weight",
    "fc.weight": "fc.weight",
    "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
    "selector_predecessor.weight": "candidate_selector.predecessor_codebook.weight",
    "selector_successor.weight": "candidate_selector.successor_codebook.weight",
}


def map_name(name: str) -> str:
    if name in _TOP:
        return _TOP[name]
    m = re.match(r"blk\.(\d+)\.(.*)", name)
    if m and m.group(2) in _LAYER:
        return f"layers.{m.group(1)}.{_LAYER[m.group(2)]}"
    raise KeyError(f"no mapping for tensor {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config", required=True, help="the head's config.json (dflash_config inside)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from gguf import GGUFReader, dequantize

    out = Path(args.out) / "dflash"
    out.mkdir(parents=True, exist_ok=True)
    reader = GGUFReader(args.gguf)
    tensors = {}
    for t in reader.tensors:
        arr = np.asarray(dequantize(t.data, t.tensor_type) if t.tensor_type.name != "F32" else t.data)
        arr = arr.reshape(tuple(int(x) for x in reversed(t.shape)))   # gguf lists ne0 first
        tensors[map_name(t.name)] = mx.array(arr.astype(np.float16))
    cfg = json.loads(Path(args.config).read_text())
    if "dflash_config" not in cfg:
        raise SystemExit("config.json has no dflash_config")
    shutil.copy(args.config, out / "config.json")
    mx.save_safetensors(str(out / "model_fp16.safetensors"), tensors,
                        metadata={"source": str(args.gguf), "note": "dequantised from GGUF; quantised on load"})
    print(f"wrote {len(tensors)} tensors to {out}")


if __name__ == "__main__":
    main()
