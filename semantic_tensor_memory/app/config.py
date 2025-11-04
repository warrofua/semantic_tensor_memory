"""Application configuration and environment setup."""

from __future__ import annotations

import os
import warnings
from typing import Optional

import streamlit as st
import torch


def setup_environment() -> None:
    """Configure global environment settings for Streamlit + PyTorch."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")
    os.environ.setdefault("TORCH_DISABLE_WARN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if hasattr(torch, "_C") and hasattr(torch._C, "_disable_jit_profiling"):
            torch._C._disable_jit_profiling()
    except Exception:
        # Best effort configuration only – Streamlit sessions should not crash if this fails.
        pass


def configure_page(
    *,
    page_title: str = "Universal Multimodal STM",
    page_icon: Optional[str] = "🌐",
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
) -> None:
    """Apply Streamlit page configuration."""
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )
