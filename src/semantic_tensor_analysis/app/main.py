"""Main entry point wiring for the Streamlit application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, Optional

try:
    _skip_streamlit = bool(
        os.environ.get("STA_SKIP_STREAMLIT")
        or os.environ.get("PYTEST_CURRENT_TEST")
    )
    if _skip_streamlit:
        raise ModuleNotFoundError("Streamlit skipped in test mode")
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    st = None  # type: ignore[assignment]
    _streamlit_import_error: Optional[ModuleNotFoundError] = ModuleNotFoundError(
        "streamlit is required to run the application. Install optional dependencies "
        "with `pip install semantic-tensor-memory[app]` or add `streamlit` to your environment."
    )
else:
    _streamlit_import_error = None


@dataclass(frozen=True)
class AppComponents:
    """Lightweight structure describing the modular app surface."""

    configure_page: Callable[[], None]
    initialize_state: Callable[[], None]
    render_upload: Callable[[], None]
    tabs: Dict[str, Callable[[], None]]


def _missing_dependency_callable(error: ModuleNotFoundError) -> Callable[..., None]:
    """Return a callable that re-raises a helpful error about optional dependencies."""

    def _missing(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError(
            "semantic_tensor_analysis.app requires optional application dependencies ("
            "e.g. streamlit, numpy, pandas) to be installed."
        ) from error

    return _missing


def _build_placeholder_app(
    error: ModuleNotFoundError, *, include_ai_insights: bool
) -> AppComponents:
    """Construct an ``AppComponents`` instance with informative placeholders."""

    missing = _missing_dependency_callable(error)
    tabs: Dict[str, Callable[[], None]] = {
        "overview": missing,
        "evolution": missing,
        "patterns": missing,
        "dimensionality": missing,
        "concepts": missing,
        "explain": missing,
    }

    tabs["ai_insights"] = missing if include_ai_insights else (lambda: None)

    return AppComponents(
        configure_page=missing,
        initialize_state=missing,
        render_upload=missing,
        tabs=tabs,
    )


def build_app(*, include_ai_insights: bool = True) -> AppComponents:
    """Return callables used to assemble the Streamlit app."""

    try:
        config = import_module(".config", __name__)
        data_ingestion = import_module(".data_ingestion", __name__)
        state = import_module(".state", __name__)
        tabs_module = import_module(".tabs", __name__)
    except ModuleNotFoundError as exc:
        return _build_placeholder_app(exc, include_ai_insights=include_ai_insights)

    tabs: Dict[str, Callable[[], None]] = {
        "overview": tabs_module.render_overview_dashboard,
        "evolution": tabs_module.render_semantic_evolution_tab,
        "patterns": tabs_module.render_pattern_analysis_tab,
        "dimensionality": tabs_module.render_dimensionality_tab,
        "concepts": tabs_module.render_enhanced_concept_analysis_tab,
        "explain": tabs_module.render_explainability_dashboard,
    }

    if include_ai_insights:
        try:
            from ..chat.analysis import render_comprehensive_chat_analysis
        except ModuleNotFoundError as exc:
            tabs["ai_insights"] = _missing_dependency_callable(exc)
        else:
            tabs["ai_insights"] = render_comprehensive_chat_analysis
    else:
        tabs["ai_insights"] = lambda: None

    return AppComponents(
        configure_page=lambda: config.configure_page(),
        initialize_state=state.initialize_universal_session_state,
        render_upload=data_ingestion.render_upload_screen,
        tabs=tabs,
    )


def main() -> None:
    """Run the Streamlit app."""

    if st is None:  # pragma: no cover - exercised in minimal test envs
        assert _streamlit_import_error is not None
        raise _streamlit_import_error

    try:
        config = import_module(".config", __name__)
        data_ingestion = import_module(".data_ingestion", __name__)
        sidebar = import_module(".sidebar", __name__)
        state = import_module(".state", __name__)
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime safeguard
        raise ModuleNotFoundError(
            "semantic_tensor_analysis.app requires optional application dependencies "
            "(e.g. streamlit, numpy, pandas) to be installed."
        ) from exc

    config.setup_environment()
    config.configure_page()
    state.initialize_universal_session_state()

    has_data = len(st.session_state.get("memory", [])) > 0
    if not has_data:
        st.session_state["sidebar_minimized_after_load"] = False
        data_ingestion.render_upload_screen()
        return

    state.collapse_sidebar_once_after_load()
    sidebar.render_simple_sidebar()

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
