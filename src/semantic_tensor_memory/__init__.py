"""Top-level package for the Semantic Tensor Memory project."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

from .app import build_app, main

__all__ = [
    "build_app",
    "main",
    "UniversalMemoryStore",
    "Modality",
    "create_universal_embedder",
    "embed_text",
    "TextEmbedder",
    "create_text_embedding",
    "embed_sentence",
]

_LAZY_IMPORTS: Dict[str, Tuple[str, str]] = {
    "UniversalMemoryStore": ("memory.universal_core", "UniversalMemoryStore"),
    "Modality": ("memory.universal_core", "Modality"),
    "create_universal_embedder": ("memory.universal_core", "create_universal_embedder"),
    "embed_text": ("memory.universal_core", "embed_text"),
    "TextEmbedder": ("memory.text_embedder", "TextEmbedder"),
    "create_text_embedding": ("memory.text_embedder", "create_text_embedding"),
    "embed_sentence": ("memory.text_embedder", "embed_sentence"),
}


def __getattr__(name: str) -> Any:
    """Dynamically import heavy optional dependencies on first access."""

    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    module_path, attribute_name = _LAZY_IMPORTS[name]
    try:
        module = importlib.import_module(f".{module_path}", __name__)
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in minimal envs
        raise ModuleNotFoundError(
            "semantic_tensor_memory optional dependencies are required to access "
            f"'{name}'. Install the full package extras to enable this functionality."
        ) from exc

    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover - debugging helper
    return sorted(set(__all__) | set(globals().keys()))
