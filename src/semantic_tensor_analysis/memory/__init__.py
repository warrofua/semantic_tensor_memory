"""Semantic Tensor Memory Core Package

This package implements the core functionality for the Semantic Tensor Memory system,
including token embedding, storage management, and drift analysis.

Modules:
    embedder: BERT-based token embedding generation
    store: Tensor storage, serialization, and management
    drift: Semantic drift analysis and metrics
"""

from functools import lru_cache

from .text_embedder import embed_sentence, get_token_count
from .store import load, save, append, to_batch, flatten
from .drift import drift_series, token_drift, cosine, session_mean


@lru_cache(maxsize=1)
def get_text_embedder():
    """
    Return a shared TextEmbedder instance.

    Centralizes loading of large BERT/SBERT models so callers don't
    re-instantiate them repeatedly across the app.
    """
    from semantic_tensor_analysis.memory.text_embedder import TextEmbedder

    return TextEmbedder()


__all__ = [
    "embed_sentence",
    "get_token_count",
    "get_text_embedder",
    "load",
    "save",
    "append",
    "to_batch",
    "flatten",
    "drift_series",
    "token_drift",
    "cosine",
    "session_mean",
]
