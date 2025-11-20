"""Semantic Tensor Memory Core Package

This package implements the core functionality for the Semantic Tensor Memory system,
including token embedding, storage management, and drift analysis.

Modules:
    embedder: BERT-based token embedding generation
    store: Tensor storage, serialization, and management
    drift: Semantic drift analysis and metrics
"""

from .text_embedder import embed_sentence, get_token_count
from .store import load, save, append, to_batch, flatten
from .drift import drift_series, token_drift, cosine, session_mean

__all__ = [
    'embed_sentence',
    'get_token_count',
    'load',
    'save',
    'append',
    'to_batch',
    'flatten',
    'drift_series',
    'token_drift',
    'cosine',
    'session_mean'
]
