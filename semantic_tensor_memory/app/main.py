"""Main entry point wiring for the Streamlit application."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import streamlit as st

from .config import configure_page, setup_environment
from .data_ingestion import render_upload_screen
from .sidebar import render_simple_sidebar
from .state import collapse_sidebar_once_after_load, initialize_universal_session_state
from .tabs import (
    render_enhanced_concept_analysis_tab,
    render_dimensionality_tab,
    render_explainability_dashboard,
    render_overview_dashboard,
    render_pattern_analysis_tab,
    render_semantic_evolution_tab,
)


@dataclass(frozen=True)
class AppComponents:
    """Lightweight structure describing the modular app surface."""

    configure_page: Callable[[], None]
    initialize_state: Callable[[], None]
    render_upload: Callable[[], None]
    tabs: Dict[str, Callable[[], None]]


def build_app(*, include_ai_insights: bool = True) -> AppComponents:
    """Return callables used to assemble the Streamlit app."""
    tabs: Dict[str, Callable[[], None]] = {
        "overview": render_overview_dashboard,
        "evolution": render_semantic_evolution_tab,
        "patterns": render_pattern_analysis_tab,
        "dimensionality": render_dimensionality_tab,
        "concepts": render_enhanced_concept_analysis_tab,
        "explain": render_explainability_dashboard,
    }

    if include_ai_insights:
        from chat_analysis import render_comprehensive_chat_analysis  # Lazy import

        tabs["ai_insights"] = render_comprehensive_chat_analysis
    else:
        tabs["ai_insights"] = lambda: None

    return AppComponents(
        configure_page=lambda: configure_page(),
        initialize_state=initialize_universal_session_state,
        render_upload=render_upload_screen,
        tabs=tabs,
    )


def main() -> None:
    """Run the Streamlit app."""
    setup_environment()
    configure_page()
    initialize_universal_session_state()

    has_data = len(st.session_state.get("memory", [])) > 0
    if not has_data:
        st.session_state["sidebar_minimized_after_load"] = False
        render_upload_screen()
        return

    collapse_sidebar_once_after_load()
    render_simple_sidebar()

    tab_labels = [
        "🏠 Overview",
        "🌊 Evolution",
        "🔍 Patterns",
        "📐 Dimensionality",
        "🧠 Concepts",
        "💡 Explain",
        "🤖 AI Insights",
    ]
    tabs = st.tabs(tab_labels)

    tab_renderers = build_app().tabs
    for label, tab_context in zip(
        [
            "overview",
            "evolution",
            "patterns",
            "dimensionality",
            "concepts",
            "explain",
            "ai_insights",
        ],
        tabs,
    ):
        with tab_context:
            tab_renderers[label]()


__all__ = ["main", "build_app"]
