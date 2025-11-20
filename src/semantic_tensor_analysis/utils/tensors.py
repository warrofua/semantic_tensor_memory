"""Lightweight tensor helpers shared across analytics and visualization."""

from __future__ import annotations

from typing import Any

import torch


def to_cpu_numpy(tensor: Any):
    """
    Safely convert a tensor-like object to a NumPy array.

    - If ``tensor`` is a torch.Tensor, it is detached and moved to CPU first.
    - If it's already a NumPy array (or not a tensor), it is returned unchanged.
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

