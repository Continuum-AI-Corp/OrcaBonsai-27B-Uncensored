"""Loading the pack and rendering prompts through its chat template."""
from __future__ import annotations

import sys
from pathlib import Path


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
    pack_dir = Path(pack_dir)
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
    """
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(pack_dir) / "tokenizer.json"))


def render_chat(pack_dir: str | Path, conversation, enable_thinking: bool = False) -> str:
    """Render a conversation through the pack's own chat template.

    ``conversation`` is either a single user prompt or a list of
    ``{"role": ..., "content": ...}`` messages. Multi-turn works by re-rendering the
    whole history each turn, which is what the template expects: the assistant turns
    have to be inside it, or the model answers every question as if it were the first.

    ``enable_thinking=False`` emits a closed, empty think block so the reply is a direct
    answer. The model defaults to extended reasoning otherwise, which is usually not
    what you want when checking behaviour.
    """
    import jinja2

    messages = ([{"role": "user", "content": conversation}]
                if isinstance(conversation, str) else list(conversation))
    template = (Path(pack_dir) / "chat_template.jinja").read_text()
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False)
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    return env.from_string(template).render(
        messages=messages,
        add_generation_prompt=True,
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
