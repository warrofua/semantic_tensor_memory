"""
Compatibility wrapper for legacy callers.

All embedding logic now lives in ``text_embedder.TextEmbedder``; this module
delegates to a singleton instance to keep older imports working.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch

# Ensure tokenizer threads behave across environments
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@lru_cache(maxsize=1)
def _get_text_embedder():
    from semantic_tensor_analysis.memory.text_embedder import TextEmbedder

    return TextEmbedder()


@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """
    Return a single-vector embedding for ``text`` (legacy API).

    Uses TextEmbedder under the hood and returns the sequence embedding as
    a [1, embedding_dim] tensor to stay compatible with historical callers.
    """
    embedder = _get_text_embedder()
    embedding = embedder.process_raw_data(text, session_id="legacy_embed")
    seq = embedding.sequence_embedding
    return seq.unsqueeze(0) if seq.ndim == 1 else seq


def get_token_count(text: str) -> int:
    """
    Approximate token count using the same embedder as ``embed_sentence``.

    Falls back to 1 for stub/test mode.
    """
    embedder = _get_text_embedder()
    # In stub mode, event_embeddings is randomly shaped; guard for empty events
    events = embedder.extract_events(text)
    return max(len(events), 1)
