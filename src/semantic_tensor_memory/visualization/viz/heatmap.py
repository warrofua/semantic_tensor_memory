import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from rich import print
from semantic_tensor_memory.memory.sequence_drift import sequence_drift
import itertools

def heatmap(tensors: List[torch.Tensor]):
    """Create session-to-session similarity heatmap from ragged tensors.
    
    Args:
        tensors: List of session embeddings, where each tensor has shape [tokens, embed_dim]
    
    The heatmap shows cosine distances between session means, with token counts
    annotated on the diagonal. Darker colors indicate greater semantic drift.
    """
    # Compute session means
    means = torch.stack([t.mean(0) for t in tensors])
    
    # Check for numerical issues
    if torch.isnan(means).any() or torch.isinf(means).any():
        print("[yellow]Warning:[/yellow] NaN or Inf values in session means")
        means = torch.nan_to_num(means, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute cosine similarity matrix with numerical stability
    means_norm = torch.nn.functional.normalize(means, p=2, dim=1)
    sims = torch.mm(means_norm, means_norm.t())
    dist = 1 - sims.numpy()
    
    # Get token counts for annotation
    token_counts = [t.shape[0] for t in tensors]
    
    # Plot
    plt.figure(figsize=(10,8))
    plt.imshow(dist, cmap="magma")
    plt.colorbar(label="Cosine distance")
    
    # Add token count annotations
    for i in range(len(tensors)):
        plt.text(i, i, f"{token_counts[i]}", 
                ha='center', va='center', color='white')
    
    plt.title("Session-to-Session Semantic Drift")
    plt.xlabel("Session")
    plt.ylabel("Session")
    plt.show()

def token_heatmap(tensors: List[torch.Tensor], window: int = 3):
    """Create token-level drift heatmap for recent sessions.
    
    Args:
        tensors: List of session embeddings, where each tensor has shape [tokens, embed_dim]
        window: Number of most recent sessions to analyze (default: 3)
    
    The heatmap shows token-to-token similarities within each recent session,
    helping identify which tokens are drifting in meaning over time.
    """
    if len(tensors) < window:
        return
    
    # Get recent sessions
    recent = tensors[-window:]
    max_tokens = max(t.shape[0] for t in recent)
    
    # Create padded tensor for visualization
    padded = torch.zeros(window, max_tokens, recent[0].shape[1])
    for i, t in enumerate(recent):
        padded[i, :t.shape[0]] = t
    
    # Check for numerical issues
    if torch.isnan(padded).any() or torch.isinf(padded).any():
        print("[yellow]Warning:[/yellow] NaN or Inf values in token embeddings")
        padded = torch.nan_to_num(padded, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Compute token-to-token similarities with numerical stability
    padded_norm = torch.nn.functional.normalize(padded, p=2, dim=2)
    sims = torch.bmm(padded_norm, padded_norm.transpose(1, 2))
    
    # Plot
    plt.figure(figsize=(12,4))
    for i in range(window):
        plt.subplot(1, window, i+1)
        plt.imshow(1 - sims[i].numpy(), cmap="magma")
        plt.colorbar(label="Distance")
        plt.title(f"Session -{window-i}")
        plt.xlabel("Token")
        plt.ylabel("Token")
    plt.tight_layout()
    plt.show() 


def token_alignment_heatmap(tensors: List[torch.Tensor], i: int, j: int):
    """Create a matplotlib figure of token-aligned distance heatmap between two sessions.

    Returns a matplotlib Figure for embedding in Streamlit (use st.pyplot(fig)).
    """
    if i < 0 or j < 0 or i >= len(tensors) or j >= len(tensors) or i == j:
        print("[red]Invalid indices for token alignment heatmap[/red]")
        return None

    A = tensors[i]
    B = tensors[j]
    if A.numel() == 0 or B.numel() == 0:
        print("[yellow]One of the sessions is empty[/yellow]")
        return None

    # Normalize
    A_norm = torch.nn.functional.normalize(A, p=2, dim=1)
    B_norm = torch.nn.functional.normalize(B, p=2, dim=1)
    # Pairwise distances
    dist = 1 - torch.mm(A_norm, B_norm.t()).numpy()  # [Na, Nb]

    # Build figure
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(dist, cmap="magma")
    fig.colorbar(im, ax=ax, label="Distance")
    ax.set_title(f"Token Alignment Distance\nSession {i+1} vs Session {j+1}")
    ax.set_xlabel(f"Session {j+1} Tokens")
    ax.set_ylabel(f"Session {i+1} Tokens")
    fig.tight_layout()
    return fig