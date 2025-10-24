"""Model management and caching utilities."""

from __future__ import annotations

import gc
import os
from typing import Callable

import streamlit as st
import torch

try:  # pragma: no cover - optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - fallback path
    psutil = None


__all__ = [
    "get_memory_usage",
    "cleanup_memory",
    "get_cached_text_embedder",
    "get_cached_universal_store",
]


def get_memory_usage() -> float:
    """Return current process memory usage in megabytes."""
    try:
        if psutil is None:
            raise RuntimeError
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except Exception:
        return 0.0


def cleanup_memory() -> None:
    """Force memory cleanup and garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@st.cache_resource
def get_cached_text_embedder():
    """Load and cache the heavy text embedder model exactly once per session."""
    from memory.text_embedder import TextEmbedder

    st.info("🔄 Loading embedding models (this happens once per session)...")
    embedder = TextEmbedder()
    st.success("✅ Models loaded and cached!")
    return embedder


@st.cache_resource
def get_cached_universal_store():
    """Return a cached universal memory store instance."""
    from memory.universal_core import UniversalMemoryStore

    return UniversalMemoryStore()
