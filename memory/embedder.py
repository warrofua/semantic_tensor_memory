from transformers import AutoTokenizer, AutoModel
import torch
import os

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
EMBED_DIM = model.config.hidden_size  # 768

# Fix threading issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """
    Embed a sentence using BERT's [CLS] token representation.
    
    🧠 SEMANTIC UPGRADE: Uses [CLS] token instead of averaging all tokens.
    The [CLS] token is specifically trained to represent the entire sequence
    and preserves semantic relationships that averaging destroys.
    
    Args:
        text: Input text
    
    Returns: 
        torch.Tensor: [1, embedding_dim] - CLS token representation
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    output = model(**inputs)
    
    # Use [CLS] token (first token) instead of averaging all tokens
    # This preserves semantic composition: "I love cats" ≠ "Cats love me"
    cls_embedding = output.last_hidden_state[0, 0, :]  # [embed_dim]
    
    # Return as [1, embed_dim] to maintain compatibility with existing code
    return cls_embedding.unsqueeze(0)

def get_token_count(text: str) -> int:
    """Returns 1 since we now produce one semantic vector per session."""
    return 1
