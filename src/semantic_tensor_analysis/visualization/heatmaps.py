"""semantic_tensor_analysis.visualization.heatmaps
=================================================

Matplotlib-based heatmap renderers for inspecting semantic drift at the session
and token level.  These helpers replace the assorted ``viz`` heatmap modules
and provide a single import location for Streamlit components and tests.
"""

from __future__ import annotations

from typing import List, Optional

import matplotlib.pyplot as plt
import torch
from rich import print


def heatmap(tensors: List[torch.Tensor]) -> None:
    """Plot a session-to-session drift heatmap for the provided embeddings."""

    if not tensors:
        print("[yellow]No tensors provided to heatmap()[/yellow]")
        return

    means = torch.stack([t.mean(0) for t in tensors])
    if torch.isnan(means).any() or torch.isinf(means).any():
        print("[yellow]Warning:[/yellow] NaN or Inf values in session means")
        means = torch.nan_to_num(means, nan=0.0, posinf=1.0, neginf=-1.0)

    means_norm = torch.nn.functional.normalize(means, p=2, dim=1)
    sims = torch.mm(means_norm, means_norm.t())
    dist = 1 - sims.numpy()

    token_counts = [t.shape[0] for t in tensors]

    plt.figure(figsize=(10, 8))
    plt.imshow(dist, cmap="magma")
    plt.colorbar(label="Cosine distance")
    for i, count in enumerate(token_counts):
        plt.text(i, i, f"{count}", ha="center", va="center", color="white")

    plt.title("Session-to-Session Semantic Drift")
    plt.xlabel("Session")
    plt.ylabel("Session")
    plt.show()


def token_heatmap(tensors: List[torch.Tensor], window: int = 3) -> None:
    """Plot a token-level drift heatmap for the ``window`` most recent sessions."""

    if len(tensors) < window or window <= 0:
        print("[yellow]Not enough sessions for token_heatmap()[/yellow]")
        return

    recent = tensors[-window:]
    if not all(t.numel() for t in recent):
        print("[yellow]One of the sessions is empty[/yellow]")
        return

    max_tokens = max(t.shape[0] for t in recent)
    embed_dim = recent[0].shape[1]
    padded = torch.zeros(window, max_tokens, embed_dim)
    for i, tensor in enumerate(recent):
        padded[i, : tensor.shape[0]] = tensor

    if torch.isnan(padded).any() or torch.isinf(padded).any():
        print("[yellow]Warning:[/yellow] NaN or Inf values in token embeddings")
        padded = torch.nan_to_num(padded, nan=0.0, posinf=1.0, neginf=-1.0)

    padded_norm = torch.nn.functional.normalize(padded, p=2, dim=2)
    sims = torch.bmm(padded_norm, padded_norm.transpose(1, 2))

    plt.figure(figsize=(12, 4))
    for i in range(window):
        plt.subplot(1, window, i + 1)
        plt.imshow(1 - sims[i].numpy(), cmap="magma")
        plt.colorbar(label="Distance")
        plt.title(f"Session -{window - i}")
        plt.xlabel("Token")
        plt.ylabel("Token")
    plt.tight_layout()
    plt.show()


def token_alignment_heatmap(
    tensors: List[torch.Tensor], i: int, j: int
) -> Optional[plt.Figure]:
    """Return a heatmap visualising token alignment distance between sessions."""

    if i < 0 or j < 0 or i >= len(tensors) or j >= len(tensors) or i == j:
        print("[red]Invalid indices for token alignment heatmap[/red]")
        return None

    session_a = tensors[i]
    session_b = tensors[j]
    if session_a.numel() == 0 or session_b.numel() == 0:
        print("[yellow]One of the sessions is empty[/yellow]")
        return None

    session_a = torch.nn.functional.normalize(session_a, p=2, dim=1)
    session_b = torch.nn.functional.normalize(session_b, p=2, dim=1)
    dist = 1 - torch.mm(session_a, session_b.t()).numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist, cmap="magma")
    fig.colorbar(im, ax=ax, label="Distance")
    ax.set_title(f"Token Alignment Distance\nSession {i + 1} vs Session {j + 1}")
    ax.set_xlabel(f"Session {j + 1} Tokens")
    ax.set_ylabel(f"Session {i + 1} Tokens")
    fig.tight_layout()
    return fig
