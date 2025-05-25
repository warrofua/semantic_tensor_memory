from sentence_transformers import SentenceTransformer
import torch
import os

# Choose model based on performance vs accuracy tradeoff
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"  # Best accuracy
# MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Best speed

# Initialize model
model = SentenceTransformer(MODEL_NAME)
EMBED_DIM = model.get_sentence_embedding_dimension()  # 768 for mpnet, 384 for MiniLM

# Set environment variable to prevent threading issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """
    Embed a sentence into a fixed-size semantic vector.
    
    Args:
        text: Input text (any length)
    
    Returns: 
        torch.Tensor: [1, embedding_dim] - single semantic vector per session
    """
    # Encode directly to semantic vector (handles long text automatically)
    embedding = model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    
    # Return as [1, embed_dim] to maintain compatibility with existing code
    return embedding.unsqueeze(0)

def embed_batch(texts: list) -> torch.Tensor:
    """
    Embed multiple texts efficiently.
    
    Args:
        texts: List of text strings
    
    Returns:
        torch.Tensor: [batch_size, embedding_dim]
    """
    embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
    return embeddings.unsqueeze(1) if embeddings.dim() == 1 else embeddings

def get_token_count(text: str) -> int:
    """Returns 1 since we produce one vector per text (for compatibility)."""
    return 1

def get_embedding_dimension() -> int:
    """Returns the embedding dimension."""
    return EMBED_DIM 