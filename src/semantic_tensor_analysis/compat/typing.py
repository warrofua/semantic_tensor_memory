"""
Compatibility helpers for typing behaviors across Python versions.

Altair 5.x expects TypedDict to accept ``closed=`` which is only available via
``typing_extensions`` on some Python versions. We monkeypatch early to keep
imports working in constrained environments (e.g., tests).
"""

from __future__ import annotations

import typing


def ensure_closed_typeddict_support() -> None:
    """Monkeypatch typing.TypedDict to the typing_extensions implementation when needed."""
    # Avoid work if the runtime already supports closed=True
    try:
        class _ClosedProbe(typing.TypedDict, closed=True):  # type: ignore[arg-type]
            pass
        return
    except TypeError:
        pass

    try:
        import typing_extensions as tix
    except Exception:
        return  # Silently skip; downstream import will error normally

    try:
        typing.TypedDict = tix.TypedDict  # type: ignore[assignment]
    except Exception:
        return
