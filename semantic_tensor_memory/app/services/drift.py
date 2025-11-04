"""Semantic drift helper functions."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch

__all__ = ["compute_drift_series"]


def _session_mean(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor.mean(0)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()


def compute_drift_series(tensors: Iterable[torch.Tensor]) -> Tuple[list, list]:
    """Compute drift statistics from a collection of tensors."""
    tensor_list = list(tensors)
    if len(tensor_list) < 2:
        return [], []

    means = torch.stack([_session_mean(t) for t in tensor_list])
    token_counts = [t.shape[0] for t in tensor_list]
    drifts = [1 - _cosine(means[i], means[i - 1]) for i in range(1, len(means))]
    return drifts, token_counts
