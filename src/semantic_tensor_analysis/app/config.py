"""Application configuration and environment setup."""

from __future__ import annotations

import os
from pathlib import Path
import warnings
from typing import Optional

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    torch = None  # type: ignore[assignment]

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    st = None  # type: ignore[assignment]
    _streamlit_import_error: Optional[ModuleNotFoundError] = ModuleNotFoundError(
        "streamlit is required to configure the application. Install optional dependencies "
        "with `pip install semantic-tensor-memory[app]` or add `streamlit` to your environment."
    )
else:
    _streamlit_import_error = None


def setup_environment() -> None:
    """Configure global environment settings for Streamlit + PyTorch."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    os.environ.setdefault("TORCH_USE_CUDA_DSA", "0")
    os.environ.setdefault("TORCH_DISABLE_WARN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if torch is not None:
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
    page_title: str = "Semantic Tensor Analysis",
    page_icon: Optional[str] = None,
    layout: str = "wide",
    initial_sidebar_state: str = "expanded",
) -> None:
    """Apply Streamlit page configuration."""
    if st is None:  # pragma: no cover - exercised in minimal test envs
        assert _streamlit_import_error is not None
        raise _streamlit_import_error

    icon = page_icon
    if icon is None:
        asset_path = Path(__file__).resolve().parent / "assets" / "3D_Trajectory_Arrow.png"
        icon = str(asset_path) if asset_path.exists() else "🌐"

    st.set_page_config(
        page_title=page_title,
        page_icon=icon,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
    )
