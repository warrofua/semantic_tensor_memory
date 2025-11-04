"""Sidebar layout helpers."""

from __future__ import annotations

import os

import streamlit as st

from .models import cleanup_memory

__all__ = ["render_simple_sidebar"]


def render_simple_sidebar() -> None:
    """Render a clean, minimal sidebar."""
    with st.sidebar:
        logo_path = "semantic_tensor_art_logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)

        dataset_info = st.session_state.get("dataset_info", {})
        if dataset_info.get("session_count", 0) > 0:
            filename = dataset_info.get("filename", "Unknown")
            session_count = dataset_info.get("session_count", 0)
            st.markdown(f"**📁 {filename}**")
            st.markdown(f"🔢 {session_count} sessions")

            if st.button("🔄 New Dataset"):
                st.session_state.memory = []
                st.session_state.meta = []
                st.session_state.dataset_info = {"session_count": 0}
                cleanup_memory()
                st.rerun()
        else:
            st.markdown("**No dataset loaded**")

        st.markdown("---")

        preferred_method = st.session_state.get("preferred_method")
        if preferred_method:
            st.markdown("**📐 Preferred Method:**")
            st.markdown(f"🎯 {preferred_method.upper()}")
            st.caption("Set in Dimensionality tab")
