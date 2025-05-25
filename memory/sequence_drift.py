import torch
import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment

def sequence_drift(tensors: List[torch.Tensor], max_length: int = 32) -> List[float]:
    """
    Calculate drift between sessions using full token sequences.
    
    This preserves semantic relationships by comparing token-to-token
    alignments instead of just session means.
    
    Args:
        tensors: List of session embeddings [tokens, embed_dim]
        max_length: Max tokens to consider (for computational efficiency)
    
    Returns:
        List of drift scores between consecutive sessions
    """
    if len(tensors) < 2:
        return []
    
    drift_scores = []
    
    for i in range(1, len(tensors)):
        prev_session = tensors[i-1][:max_length]  # [tokens_prev, embed_dim]
        curr_session = tensors[i][:max_length]    # [tokens_curr, embed_dim]
        
        # Compute pairwise token similarities
        prev_norm = torch.nn.functional.normalize(prev_session, p=2, dim=1)
        curr_norm = torch.nn.functional.normalize(curr_session, p=2, dim=1)
        
        # Similarity matrix: [tokens_prev, tokens_curr]
        similarity_matrix = torch.mm(prev_norm, curr_norm.t())
        distance_matrix = 1 - similarity_matrix.numpy()
        
        # Find optimal alignment between token sequences
        # This accounts for varying sequence lengths
        if distance_matrix.shape[0] > 0 and distance_matrix.shape[1] > 0:
            # Use Hungarian algorithm to find optimal token alignment
            row_indices, col_indices = linear_sum_assignment(distance_matrix)
            
            # Calculate drift as average distance of optimal alignment
            alignment_distances = distance_matrix[row_indices, col_indices]
            drift_score = np.mean(alignment_distances)
        else:
            drift_score = 1.0  # Maximum drift for empty sequences
        
        drift_scores.append(drift_score)
    
    return drift_scores

def token_importance_drift(tensors: List[torch.Tensor], top_k: int = 10) -> List[Tuple[int, float]]:
    """
    Identify which token positions show the most semantic drift.
    
    Args:
        tensors: List of session embeddings
        top_k: Number of top drifting token positions to return
    
    Returns:
        List of (session_index, drift_score) for most drifting tokens
    """
    if len(tensors) < 3:
        return []
    
    token_drifts = []
    
    # Compare each session with previous sessions
    for i in range(2, len(tensors)):
        curr_session = tensors[i]
        
        # Compare with sliding window of previous sessions
        window_sessions = tensors[max(0, i-3):i]
        
        for token_idx in range(min(curr_session.shape[0], 20)):  # Check first 20 tokens
            curr_token = curr_session[token_idx]
            
            # Calculate drift from previous similar positions
            position_drifts = []
            for prev_session in window_sessions:
                if token_idx < prev_session.shape[0]:
                    prev_token = prev_session[token_idx]
                    drift = 1 - cosine(curr_token.numpy(), prev_token.numpy())
                    position_drifts.append(drift)
            
            if position_drifts:
                avg_drift = np.mean(position_drifts)
                if avg_drift > 0.3:  # Significant drift threshold
                    token_drifts.append((i, avg_drift))
    
    # Return top drifting positions
    return sorted(token_drifts, key=lambda x: x[1], reverse=True)[:top_k]

def semantic_coherence_score(tensor: torch.Tensor) -> float:
    """
    Calculate how semantically coherent a session is internally.
    
    Higher scores indicate more coherent semantic content.
    Lower scores might indicate topic switching or confusion.
    """
    if tensor.shape[0] < 2:
        return 1.0
    
    # Normalize embeddings
    normalized = torch.nn.functional.normalize(tensor, p=2, dim=1)
    
    # Calculate pairwise similarities
    similarities = torch.mm(normalized, normalized.t())
    
    # Get upper triangle (excluding diagonal)
    n = similarities.shape[0]
    upper_triangle = similarities[torch.triu(torch.ones(n, n), diagonal=1) == 1]
    
    # Coherence is average similarity between all token pairs
    coherence = upper_triangle.mean().item()
    
    return coherence

def enhanced_drift_analysis(tensors: List[torch.Tensor], meta: List[dict]) -> dict:
    """
    Comprehensive drift analysis using full sequence information.
    
    Returns:
        Dictionary with multiple drift metrics and insights
    """
    results = {
        'sequence_drift': sequence_drift(tensors),
        'token_importance': token_importance_drift(tensors),
        'coherence_scores': [semantic_coherence_score(t) for t in tensors],
        'session_lengths': [t.shape[0] for t in tensors],
    }
    
    # Add interpretations
    if results['coherence_scores']:
        results['coherence_trend'] = 'increasing' if results['coherence_scores'][-1] > results['coherence_scores'][0] else 'decreasing'
        results['avg_coherence'] = np.mean(results['coherence_scores'])
    
    if results['sequence_drift']:
        results['avg_drift'] = np.mean(results['sequence_drift'])
        results['drift_trend'] = 'increasing' if len(results['sequence_drift']) > 1 and results['sequence_drift'][-1] > results['sequence_drift'][0] else 'stable'
    
    return results 