"""Temporal analysis tab leveraging multi-resolution conversation embeddings."""

from __future__ import annotations

import streamlit as st
import hashlib
from typing import List

from semantic_tensor_analysis.chat.history_analyzer import ChatMessage
from semantic_tensor_analysis.app.temporal_resolution_manager import (
    TemporalResolutionManager,
    TemporalResolution,
)
from semantic_tensor_analysis.app.temporal_visualizations import (
    create_temporal_semantic_flow,
    create_temporal_heatmap,
    render_temporal_visualizations_tab as render_temporal_viz_tab,
)


def _get_chat_messages() -> List[ChatMessage]:
    """Best-effort retrieval of chat messages from session state."""
    messages = st.session_state.get("chat_messages")
    if messages and all(isinstance(m, ChatMessage) for m in messages):
        return messages  # type: ignore[return-value]
    # Fallback: try to build from uploaded conversation logs if present
    raw = st.session_state.get("uploaded_chat_messages")
    if raw and all(getattr(m, "content", None) for m in raw):
        try:
            return [
                ChatMessage(
                    content=getattr(m, "content", ""),
                    role=getattr(m, "role", "user"),
                    timestamp=getattr(m, "timestamp", None),
                    conversation_id=getattr(m, "conversation_id", None),
                    message_id=getattr(m, "message_id", None),
                )
                for m in raw
            ]
        except Exception:
            return []
    return []


def _messages_cache_key(messages: List[ChatMessage]) -> str:
    """Deterministic hash so we only embed conversations once per data load."""
    hasher = hashlib.sha1()
    for msg in messages:
        hasher.update((msg.role or "").encode("utf-8"))
        hasher.update((msg.content or "").encode("utf-8"))
        if msg.message_id:
            hasher.update(str(msg.message_id).encode("utf-8"))
        if msg.timestamp:
            hasher.update(str(msg.timestamp).encode("utf-8"))
    return hasher.hexdigest()


def render_temporal_analysis_tab() -> None:
    """Render the temporal analysis tab."""
    st.header("⏱️ Temporal Analysis (Multi-Resolution)")

    messages = _get_chat_messages()
    if not messages:
        st.info("Upload AI conversation data to enable temporal analysis (turns → conversations → days).")
        return

    cache_key = _messages_cache_key(messages)
    manager: TemporalResolutionManager

    cached_manager = st.session_state.get("temporal_manager")
    cached_key = st.session_state.get("temporal_manager_cache_key")

    if cached_manager and cached_key == cache_key:
        manager = cached_manager
        results = st.session_state.get("temporal_manager_results")
        if not results:
            results = manager.process_conversation_messages(messages)
            st.session_state["temporal_manager_results"] = results
    else:
        manager = TemporalResolutionManager()
        results = manager.process_conversation_messages(messages)
        st.session_state["temporal_manager"] = manager
        st.session_state["temporal_manager_cache_key"] = cache_key
        st.session_state["temporal_manager_results"] = results

    st.success(
        f"Processed {results['total_messages']} messages across "
        f"{results['conversations_created']} conversations "
        f"spanning {results['days_covered']} days."
    )

    # Controls
    st.markdown("### 🔍 Select Temporal Resolution")
    resolution = st.radio(
        "Resolution",
        [TemporalResolution.TURN, TemporalResolution.CONVERSATION, TemporalResolution.DAY],
        format_func=lambda r: r.value.title(),
        horizontal=True,
    )

    st.markdown("### 🌊 Semantic Flow")
    flow_fig = create_temporal_semantic_flow(manager, resolution=resolution)
    if flow_fig:
        st.plotly_chart(flow_fig, use_container_width=True)

    st.markdown("### 🔥 Temporal Heatmap")
    heatmap = create_temporal_heatmap(manager, resolution=resolution)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)

    st.divider()
    st.markdown("### 🧭 Advanced Temporal Visualizations")
    render_temporal_viz_tab()


__all__ = ["render_temporal_analysis_tab"]
