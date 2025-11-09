import torch, math
from typing import List, Tuple

def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute cosine similarity between two tensors."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()

def session_mean(mat: torch.Tensor) -> torch.Tensor:
    """Extract session representation -> [embed_dim].
    
    🧠 SEMANTIC UPGRADE: With CLS token embeddings, this just extracts
    the single semantic vector (no averaging needed).
    """
    if mat.shape[0] == 1:
        # CLS token approach: already have single semantic vector
        return mat.squeeze(0)
    else:
        # Fallback for legacy token sequences: use mean
        return mat.mean(0)

def drift_series(tensors: List[torch.Tensor]) -> Tuple[List[float], List[int]]:
    """Return drift metrics between consecutive sessions.
    
    Returns:
        Tuple of (drift_scores, token_counts)
        - drift_scores: List of cosine distances between consecutive session means
        - token_counts: List of token counts per session for context
    """
    means = torch.stack([session_mean(t) for t in tensors])
    token_counts = [t.shape[0] for t in tensors]
    drifts = [1 - cosine(means[i], means[i-1]) for i in range(1, len(means))]
    return drifts, token_counts

def token_drift(tensors: List[torch.Tensor], window: int = 3) -> List[Tuple[int, float]]:
    """Analyze drift at token level over a sliding window.
    
    Args:
        tensors: List of session embeddings
        window: Number of sessions to look back for drift analysis
    
    Returns:
        List of (token_idx, drift_score) tuples for tokens showing significant drift
    """
    if len(tensors) < window:
        return []
    
    # Get token embeddings for last window sessions
    recent = torch.cat(tensors[-window:])
    token_drifts = []
    
    # Compare each token with its position in previous sessions
    for i in range(recent.shape[0]):
        if i < window:  # Skip tokens from earlier sessions
            continue
        current = recent[i]
        prev = recent[i - window]
        drift = 1 - cosine(current, prev)
        if drift > 0.2:  # Threshold for significant drift
            token_drifts.append((i, drift))
    
    return sorted(token_drifts, key=lambda x: x[1], reverse=True) 
