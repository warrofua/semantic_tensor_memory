"""Top-level package for the Semantic Tensor Memory project."""

from .memory.universal_core import (
    UniversalMemoryStore,
    Modality,
    create_universal_embedder,
    embed_text,
)
from .memory.text_embedder import TextEmbedder, create_text_embedding, embed_sentence

__all__ = [
    "UniversalMemoryStore",
    "Modality",
    "create_universal_embedder",
    "embed_text",
    "TextEmbedder",
    "create_text_embedding",
    "embed_sentence",
]
