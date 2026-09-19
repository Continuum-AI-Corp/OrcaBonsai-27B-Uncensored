"""Loading the pack and rendering prompts through its chat template."""
from __future__ import annotations

import sys
from pathlib import Path


def resolve_pack(pack_dir: str | Path) -> Path:
    """Return the directory that actually holds the pack, and say so if it does not.

    `huggingface_hub` stores a download as `models--org--name/snapshots/<sha>/`, with
    `blobs/` and `refs/` beside it. Pointing at the `models--...` directory itself is the
    obvious thing to try and it lands one level too high: nothing is there, so
    `runtime/` does not exist, and the failure used to surface much later as
    `ModuleNotFoundError: No module named 'vision_artifact'` -- which says nothing about
    the real problem. Resolve that layout, and otherwise fail here with the reason.
    """
    pack_dir = Path(pack_dir).expanduser()
    if not pack_dir.exists():
        raise FileNotFoundError(f"no such directory: {pack_dir}")

    if not (pack_dir / "config.json").exists():
        snapshots = pack_dir / "snapshots"
        if snapshots.is_dir():
            # An HF cache entry. Take the newest snapshot that is actually a pack.
            candidates = sorted((d for d in snapshots.iterdir()
                                 if (d / "config.json").exists()),
                                key=lambda d: d.stat().st_mtime, reverse=True)
            if candidates:
                return candidates[0]
            raise FileNotFoundError(
                f"{pack_dir} is a huggingface cache entry but none of its snapshots "
                f"contain config.json -- the download may be incomplete")
        raise FileNotFoundError(
            f"{pack_dir} does not look like a pack: no config.json.\n"
            f"If you downloaded with huggingface_hub, pass the snapshot directory:\n"
            f"  python -c \"from huggingface_hub import snapshot_download; "
            f"print(snapshot_download('prism-ml/Ternary-Bonsai-2-27B-mlx-2bit'))\"")

    if not (pack_dir / "runtime" / "vision_artifact.py").exists():
        raise FileNotFoundError(
            f"{pack_dir} has a config.json but no runtime/vision_artifact.py. The pack "
            f"must be downloaded whole -- its own loader is the only one that applies "
            f"the Hadamard transform these weights require.")
    return pack_dir


def load_pack(pack_dir: str | Path, load_processor: bool = False):
    """Load a Prism Hadamard pack via its own bundled runtime.

    Returns ``(model, config)``, or ``(model, processor, config)`` when
    ``load_processor`` is set.

    The pack ships two loaders and only ``vision_artifact`` accepts the vision-carrying
    builds: ``artifact.load_model`` requires ``schema_version == 1`` and rejects the
    ``schema_version: 2`` config these packs use.

    Ordinary MLX loaders will appear to work and silently compute nonsense, because they
    do not apply the Hadamard activation transform the stored weights require.
    """
    pack_dir = resolve_pack(pack_dir)
    runtime = str(pack_dir / "runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    from vision_artifact import load_vl_model

    model, processor, config = load_vl_model(str(pack_dir), load_processor=load_processor)
    return (model, processor, config) if load_processor else (model, config)


def load_tokenizer(pack_dir: str | Path):
    """Load the pack's tokenizer without going through ``AutoTokenizer``.

    ``AutoTokenizer`` resolves classes through ``config.json``'s ``model_type``, which
    is deliberately ``prism_hadamard_qwen35`` here, so reading ``tokenizer.json``
    directly is both simpler and more reliable.

    Goes through ``resolve_pack`` like ``load_pack`` does, so an HF cache entry works
    here too instead of failing on a missing ``tokenizer.json`` after the model loaded.
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(resolve_pack(pack_dir) / "tokenizer.json"))


def render_chat(pack_dir: str | Path, conversation, enable_thinking: bool = False,
                add_generation_prompt: bool = True) -> str:
    """Render a conversation through the pack's own chat template.

    ``conversation`` is either a single user prompt or a list of
    ``{"role": ..., "content": ...}`` messages. Multi-turn works by re-rendering the
    whole history each turn, which is what the template expects: the assistant turns
    have to be inside it, or the model answers every question as if it were the first.

    ``enable_thinking=False`` emits a closed, empty think block so the reply is a direct
    answer. The model defaults to extended reasoning otherwise, which is usually not
    what you want when checking behaviour.

    ``add_generation_prompt=False`` renders the conversation as history only, ending
    after the last turn's ``<|im_end|>``; run.py uses that to find the exact text the
    template appends for a new turn without re-tokenising the earlier ones.
    """
    import jinja2

    messages = ([{"role": "user", "content": conversation}]
                if isinstance(conversation, str) else list(conversation))
    template = (resolve_pack(pack_dir) / "chat_template.jinja").read_text()
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    return env.from_string(template).render(
        messages=messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
    )


def eos_ids(pack_dir: str | Path, config: dict, tok) -> set[int]:
    """Stop tokens: the config's EOS plus the chat template's turn terminator."""
    ids = {config["text_config"].get("eos_token_id")}
    for name in ("<|im_end|>", "<|endoftext|>"):
        tid = tok.token_to_id(name)
        if tid is not None:
            ids.add(tid)
    ids.discard(None)
    return ids
