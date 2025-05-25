from transformers import AutoTokenizer, AutoModel
import torch
import os
from typing import Tuple

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
EMBED_DIM = model.config.hidden_size  # 768

# Fix threading issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """
    HYBRID STM APPROACH: Preserve tensor nature while adding semantic intelligence.
    
    Returns full token sequence embeddings (preserving STM's core innovation)
    while also computing CLS representation for efficient session-level analysis.
    
    Args:
        text: Input text
    
    Returns: 
        torch.Tensor: [tokens, embedding_dim] - PRESERVES token-level granularity
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    output = model(**inputs)
    
    # Return FULL token embeddings - preserves STM tensor nature
    embeddings = output.last_hidden_state.squeeze(0)  # [tokens, embed_dim]
    return embeddings

@torch.inference_mode()
def embed_sentence_with_summary(text: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    HYBRID: Full token embeddings + CLS summary for best of both worlds.
    
    Args:
        text: Input text
    
    Returns:
        Tuple of:
        - torch.Tensor: [tokens, embedding_dim] - Full token-level data (STM core)
        - torch.Tensor: [embedding_dim] - CLS session summary (for efficiency)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    output = model(**inputs)
    
    # Full token embeddings (STM's core innovation)
    token_embeddings = output.last_hidden_state.squeeze(0)  # [tokens, embed_dim]
    
    # CLS summary for efficient session-level analysis
    cls_summary = output.last_hidden_state[0, 0, :]  # [embed_dim]
    
    return token_embeddings, cls_summary

def get_session_summary(token_embeddings: torch.Tensor, method: str = "cls") -> torch.Tensor:
    """
    Extract session-level summary from token embeddings.
    
    Args:
        token_embeddings: [tokens, embed_dim] tensor
        method: "cls" (first token), "mean" (average), or "attention_weighted"
    
    Returns:
        torch.Tensor: [embed_dim] session summary
    """
    if method == "cls":
        return token_embeddings[0, :]  # CLS token
    elif method == "mean":
        return token_embeddings.mean(0)  # Average (old approach)
    elif method == "attention_weighted":
        # Simple attention weighting (could be enhanced)
        weights = torch.softmax(token_embeddings.norm(dim=1), dim=0)
        return (token_embeddings * weights.unsqueeze(1)).sum(0)
    else:
        raise ValueError(f"Unknown method: {method}")

def get_token_count(text: str) -> int:
    """Returns actual number of tokens (preserving STM's granular tracking)."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    return inputs['input_ids'].shape[1]

def analyze_token_level_drift(prev_tokens: torch.Tensor, curr_tokens: torch.Tensor) -> dict:
    """
    STM CORE FEATURE: Token-level drift analysis.
    
    This preserves STM's key innovation of granular semantic tracking.
    """
    from memory.sequence_drift import sequence_drift, token_importance_drift
    
    # Token-to-token alignment drift (STM's unique capability)
    sequence_drifts = sequence_drift([prev_tokens, curr_tokens])
    
    # Identify which tokens are drifting most
    importance_drifts = token_importance_drift([prev_tokens, curr_tokens])
    
    return {
        'sequence_drift': sequence_drifts[0] if sequence_drifts else 0.0,
        'token_importance': importance_drifts,
        'token_count_change': curr_tokens.shape[0] - prev_tokens.shape[0],
        'embedding_stability': torch.cosine_similarity(
            prev_tokens.mean(0), curr_tokens.mean(0), dim=0
        ).item()
    } 