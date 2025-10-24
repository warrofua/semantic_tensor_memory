"""Smoke tests for the modular Streamlit application package."""

from semantic_tensor_memory.app import build_app
from semantic_tensor_memory.app.services import compute_drift_series, parse_chat_history

import torch


def test_build_app_structure():
    components = build_app(include_ai_insights=False)
    assert callable(components.configure_page)
    assert callable(components.initialize_state)
    assert callable(components.render_upload)
    assert set(components.tabs.keys()) == {
        "overview",
        "evolution",
        "patterns",
        "dimensionality",
        "concepts",
        "explain",
        "ai_insights",
    }


def test_parse_chat_history_plain_text():
    transcript = "User: Hello\nAssistant: Hi there!"
    messages = parse_chat_history(transcript)
    assert messages, "Expected at least one parsed message"
    assert all(hasattr(msg, "content") for msg in messages)


def test_compute_drift_series_simple():
    tensors = [torch.ones((1, 4)), torch.ones((1, 4)) * 2]
    drifts, counts = compute_drift_series(tensors)
    assert len(drifts) == 1
    assert counts == [1, 1]
