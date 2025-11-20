"""Pytest configuration to make the ``src`` layout importable and lighten optional deps."""

import os
import sys
from pathlib import Path

# Skip Streamlit imports during tests to avoid heavy/GUI dependencies
os.environ.setdefault("STA_SKIP_STREAMLIT", "1")
os.environ.setdefault("STA_LIGHTWEIGHT_EMBEDDER", "1")

# Provide a lightweight Streamlit stub so modules importing streamlit won't fail during tests.
if os.environ.get("STA_SKIP_STREAMLIT"):
    import types
    import numpy as _np
    import importlib

    class _Stub:
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, _name):
            return self

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    st_stub = _Stub()
    sys.modules.setdefault("streamlit", st_stub)
    sys.modules.setdefault("streamlit.components", st_stub)
    sys.modules.setdefault("streamlit.components.v1", st_stub)

    # Minimal sklearn patch to satisfy import hooks in streamlit utils during tests
    try:
        import sklearn  # type: ignore
    except ImportError:
        sk_stub = types.ModuleType("sklearn")
        submods = {
            "sklearn.decomposition": types.ModuleType("sklearn.decomposition"),
            "sklearn.preprocessing": types.ModuleType("sklearn.preprocessing"),
            "sklearn.cluster": types.ModuleType("sklearn.cluster"),
            "sklearn.manifold": types.ModuleType("sklearn.manifold"),
            "sklearn.metrics.pairwise": types.ModuleType("sklearn.metrics.pairwise"),
        }
        for name, module in submods.items():
            sys.modules[name] = module
        sys.modules["sklearn"] = sk_stub

    # Patch numpy alias missing in newer versions for plotly express
    if not hasattr(_np, "byte"):
        _np.byte = _np.int8


def _ensure_src_on_path() -> None:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


_ensure_src_on_path()


def pytest_collection_modifyitems(items):
    """
    Skip heavy tests only when STA_MINIMAL_TESTS=1; otherwise run full suite.
    """
    import pytest
    if os.environ.get("STA_MINIMAL_TESTS") != "1":
        return

    skip_reasons = {
        "test_compute_drift_series_simple": "Torch math not reliable in sandbox",
        "test_cross_modal_analysis": "Fixture not available in minimal env",
        "test_concept_analysis": "Sklearn/plotly heavy dependencies unavailable",
        "test_similarity_matrix": "LLM/text embedding stack not available in sandbox",
        "test_universal_text_embedding": "LLM/text embedding stack not available in sandbox",
        "test_universal_memory_store": "LLM/text embedding stack not available in sandbox",
        "test_backward_compatibility": "LLM/text embedding stack not available in sandbox",
    }

    for item in items:
        reason = skip_reasons.get(item.name)
        if reason:
            item.add_marker(pytest.mark.skip(reason=reason))
