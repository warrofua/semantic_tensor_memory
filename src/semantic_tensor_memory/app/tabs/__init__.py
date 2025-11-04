"""Tab renderers for the Streamlit app (lazy imports)."""

from __future__ import annotations

from typing import Any

__all__ = [
    "render_enhanced_concept_analysis_tab",
    "render_dimensionality_tab",
    "render_semantic_evolution_tab",
    "render_explainability_dashboard",
    "render_overview_dashboard",
    "render_pattern_analysis_tab",
]


def render_overview_dashboard(*args: Any, **kwargs: Any) -> Any:
    from .overview import render_overview_dashboard as _impl

    return _impl(*args, **kwargs)


def render_semantic_evolution_tab(*args: Any, **kwargs: Any) -> Any:
    from .evolution import render_semantic_evolution_tab as _impl

    return _impl(*args, **kwargs)


def render_pattern_analysis_tab(*args: Any, **kwargs: Any) -> Any:
    from .patterns import render_pattern_analysis_tab as _impl

    return _impl(*args, **kwargs)


def render_dimensionality_tab(*args: Any, **kwargs: Any) -> Any:
    from .dimensionality import render_dimensionality_tab as _impl

    return _impl(*args, **kwargs)


def render_enhanced_concept_analysis_tab(*args: Any, **kwargs: Any) -> Any:
    from .concepts import render_enhanced_concept_analysis_tab as _impl

    return _impl(*args, **kwargs)


def render_explainability_dashboard(*args: Any, **kwargs: Any) -> Any:
    from .explain import render_explainability_dashboard as _impl

    return _impl(*args, **kwargs)
