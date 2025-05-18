from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
EMBED_DIM = model.config.hidden_size  # 768

@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """
    Embed a single sentence into [tokens, embed_dim] tensor.
    Returns: 
        torch.Tensor: [tokens, embedding_dim]
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    output = model(**inputs)
    embeddings = output.last_hidden_state.squeeze(0)  # [tokens, embed_dim]
    return embeddings

def get_token_count(text: str) -> int:
    """Returns number of tokens used by the model for this input."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    return inputs['input_ids'].shape[1]
