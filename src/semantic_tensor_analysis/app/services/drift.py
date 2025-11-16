"""Semantic drift helper functions."""

from __future__ import annotations

from typing import Iterable, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    torch = None  # type: ignore[assignment]
    _torch_import_error = ModuleNotFoundError(
        "torch is required for drift analysis. Install the optional 'app' extras to "
        "enable this functionality."
    )
else:
    _torch_import_error = None

__all__ = ["compute_drift_series"]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - exercised in minimal test envs
        assert _torch_import_error is not None
        raise _torch_import_error


def _session_mean(tensor: "torch.Tensor") -> "torch.Tensor":
    _require_torch()
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor.mean(0)


def _cosine(a: "torch.Tensor", b: "torch.Tensor") -> float:
    _require_torch()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def compute_drift_series(tensors: Iterable["torch.Tensor"]) -> Tuple[list, list]:
    """Compute drift statistics from a collection of tensors."""
    _require_torch()
    tensor_list = list(tensors)
    if len(tensor_list) < 2:
        return [], []

    means = torch.stack([_session_mean(t) for t in tensor_list])
    token_counts = [t.shape[0] for t in tensor_list]
    drifts = [1 - _cosine(means[i], means[i - 1]) for i in range(1, len(means))]
    return drifts, token_counts
