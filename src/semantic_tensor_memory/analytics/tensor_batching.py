"""Utilities for batching ragged token tensors with masks.

Functions here standardize how we pad variable-length session tensors
and expose helpers to compute masked statistics and flattened views
without accidentally including padded rows.
"""

from typing import List, Tuple
import torch
import numpy as np


def pad_and_stack(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [tokens, dim] tensors to a batch with a boolean mask.

    Args:
        tensors: list of session tensors, each [num_tokens_i, embed_dim]

    Returns:
        batch: [batch_size, max_tokens, embed_dim]
        mask:  [batch_size, max_tokens] (True where valid token exists)
    """
    if not tensors:
        return torch.empty(0, 0, 0), torch.empty(0, 0, dtype=torch.bool)

    embed_dim = tensors[0].shape[1] if tensors[0].ndim == 2 else 0
    max_tokens = max(t.shape[0] for t in tensors) if tensors else 0
    batch_size = len(tensors)

    batch = torch.zeros(batch_size, max_tokens, embed_dim, dtype=tensors[0].dtype)
    mask = torch.zeros(batch_size, max_tokens, dtype=torch.bool)

    for i, t in enumerate(tensors):
        length = t.shape[0]
        if length == 0:
            continue
        batch[i, :length] = t
        mask[i, :length] = True

    return batch, mask


def masked_session_means(batch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute per-session mean embedding using a mask to ignore padding.

    Args:
        batch: [B, T, D]
        mask:  [B, T]

    Returns:
        means: [B, D]
    """
    if batch.numel() == 0:
        return torch.empty(0, 0, dtype=batch.dtype)

    # Avoid division by zero
    lengths = mask.sum(dim=1).clamp_min(1).unsqueeze(-1)  # [B,1]
    summed = (batch * mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
    means = summed / lengths
    return means


def flatten_with_mask(batch: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Flatten a masked batch to 2D points and return session/token indices.

    Args:
        batch: [B, T, D]
        mask:  [B, T]

    Returns:
        flat: [N, D] valid token embeddings
        session_ids: [N] session index per token
        token_ids: [N] token position within session
    """
    if batch.numel() == 0:
        return torch.empty(0, 0, dtype=batch.dtype), np.array([], dtype=int), np.array([], dtype=int)

    B, T, D = batch.shape
    # Ensure contiguous before reshaping
    batch = batch.contiguous()
    mask = mask.contiguous()
    valid_positions = mask.reshape(-1)
    flat_all = batch.reshape(B * T, D)
    flat = flat_all[valid_positions]

    # Build indices
    session_grid, token_grid = torch.meshgrid(
        torch.arange(B, dtype=torch.long), torch.arange(T, dtype=torch.long), indexing='ij'
    )
    session_ids = session_grid.contiguous().reshape(-1)[valid_positions].cpu().numpy()
    token_ids = token_grid.contiguous().reshape(-1)[valid_positions].cpu().numpy()
    return flat, session_ids, token_ids

