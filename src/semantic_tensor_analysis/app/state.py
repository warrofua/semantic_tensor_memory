"""Session state orchestration utilities."""

from __future__ import annotations

from typing import Set

import streamlit as st
import streamlit.components.v1 as components

from semantic_tensor_analysis.memory.universal_core import Modality
from semantic_tensor_analysis.streamlit.utils import initialize_session_state as _initialize_session_state

from .models import cleanup_memory, get_cached_universal_store, get_memory_usage

__all__ = [
    "initialize_universal_session_state",
    "collapse_sidebar_once_after_load",
]


def initialize_universal_session_state() -> None:
    """Initialize Streamlit session state with universal STM support."""
    _initialize_session_state()

    if "initial_memory" not in st.session_state:
        st.session_state.initial_memory = get_memory_usage()

    if "universal_store" not in st.session_state:
        st.session_state.universal_store = get_cached_universal_store()

    if "active_modalities" not in st.session_state:
        st.session_state.active_modalities = set()

    if "modality_sessions" not in st.session_state:
        st.session_state.modality_sessions = {modality: [] for modality in Modality}

    current_memory = get_memory_usage()
    if current_memory > st.session_state.initial_memory + 2000:
        st.warning(f"🧠 Memory usage high ({current_memory:.0f}MB). Running cleanup...")
        cleanup_memory()


def _collapse_sidebar_via_js() -> None:
    components.html(
        """
        <script>
        (function() {
          const doc = window.parent.document;
          const btn = doc.querySelector('[data-testid="stSidebarCollapseButton"]');
          const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
          if (btn && sidebar && sidebar.offsetWidth > 0) {
            btn.click();
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )


def collapse_sidebar_once_after_load() -> None:
    """Collapse the sidebar exactly once after data has been loaded."""
    if not st.session_state.get("sidebar_minimized_after_load", False):
        _collapse_sidebar_via_js()
        st.session_state["sidebar_minimized_after_load"] = True
