from transformers import AutoTokenizer, AutoModel
import torch
import os

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
EMBED_DIM = model.config.hidden_size  # 768

# Set environment variable to prevent threading issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """
    Embed a sentence using BERT's [CLS] token representation.
    
    The [CLS] token is specifically trained to represent the entire sequence
    and preserves semantic relationships better than averaging all tokens.
    
    Args:
        text: Input text
    
    Returns: 
        torch.Tensor: [1, embedding_dim] - CLS token representation
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    output = model(**inputs)
    
    # Use [CLS] token (first token) instead of averaging all tokens
    cls_embedding = output.last_hidden_state[0, 0, :]  # [embed_dim]
    
    # Return as [1, embed_dim] to maintain compatibility with existing code
    return cls_embedding.unsqueeze(0)

@torch.inference_mode()
def embed_sentence_with_attention_pooling(text: str) -> torch.Tensor:
    """
    Advanced: Use attention-weighted pooling of token embeddings.
    
    This uses BERT's attention weights to create a weighted average,
    preserving important token relationships.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    output = model(**inputs, output_attentions=True)
    
    # Get last layer embeddings and attention weights
    embeddings = output.last_hidden_state.squeeze(0)  # [tokens, embed_dim]
    attention = output.attentions[-1].squeeze(0)  # [heads, tokens, tokens]
    
    # Use average attention to CLS token as importance weights
    cls_attention = attention[:, 0, :].mean(0)  # [tokens] - attention to CLS
    cls_attention = torch.softmax(cls_attention, dim=0)  # Normalize
    
    # Weighted sum of embeddings
    weighted_embedding = (embeddings * cls_attention.unsqueeze(1)).sum(0)  # [embed_dim]
    
    return weighted_embedding.unsqueeze(0)

def get_token_count(text: str) -> int:
    """Returns 1 since we produce one vector per text (for compatibility)."""
    return 1

def get_embedding_dimension() -> int:
    """Returns the embedding dimension."""
    return EMBED_DIM 