"""Sidebar layout helpers."""

from __future__ import annotations

import streamlit as st

from .models import cleanup_memory
from semantic_tensor_analysis.storage.manager import StorageManager
from pathlib import Path

__all__ = ["render_simple_sidebar"]


def render_simple_sidebar() -> None:
    """Render a clean, minimal sidebar."""
    with st.sidebar:
        assets_dir = Path(__file__).resolve().parent / "assets"
        logo_path = assets_dir / "semantic_tensor_art_logo.png"
        if logo_path.exists():
            st.image(str(logo_path), width=200)

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

        st.markdown("---")

        with st.expander("🗂️ Storage (sessions)", expanded=False):
            manager = StorageManager()
            stats = manager.get_stats()
            st.write(f"Files: {stats.total_files}")
            st.write(f"Size: {stats.total_size_mb:.2f} MB")
            if stats.oldest_mtime and stats.newest_mtime:
                st.write(f"Age: {stats.oldest_mtime.date()} → {stats.newest_mtime.date()}")
            cleanup_days = st.number_input(
                "Cleanup sessions older than days",
                min_value=1,
                max_value=365,
                value=30,
                key="storage_cleanup_days",
            )
            mode = st.radio(
                "Cleanup mode",
                options=["Preview (no delete)", "Delete"],
                horizontal=True,
                key="storage_cleanup_mode",
            )
            if st.button("🧹 Run cleanup", key="storage_cleanup_apply"):
                dry_run = mode.startswith("Preview")
                removed, freed = manager.cleanup_old_sessions(int(cleanup_days), dry_run=dry_run)
                if dry_run:
                    st.info(f"Preview: would remove {removed} files (~{freed/1024/1024:.2f} MB)")
                else:
                    st.success(f"Removed {removed} files (~{freed/1024/1024:.2f} MB)")
