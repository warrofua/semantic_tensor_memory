import torch, os, json, time
from rich import print
from typing import List, Dict, Tuple

MEM_PATH = "memory_tensor.pt"
META_PATH = "memory_meta.json"

def normalize_meta_entry(entry: Dict, text: str = None, tokens: int = None) -> Dict:
    """Ensure metadata entry has all required fields with consistent naming."""
    normalized = {
        "ts": entry.get("ts", time.time()),
        "text": entry.get("text", text),
        "tokens": entry.get("tokens", entry.get("len", tokens))
    }
    # Remove any None values
    return {k: v for k, v in normalized.items() if v is not None}

def load() -> Tuple[List[torch.Tensor], List[Dict]]:
    """Load existing memory tensors and metadata if available."""
    if os.path.exists(MEM_PATH):
        data = torch.load(MEM_PATH)
        # Convert single tensor to list if needed
        if isinstance(data, torch.Tensor):
            tensors = [data[i] for i in range(data.shape[0])]
        else:
            tensors = data
            
        # Load and normalize metadata
        meta = json.load(open(META_PATH))
        meta = [normalize_meta_entry(entry) for entry in meta]
        
        # Backfill missing fields if possible
        for i, (tensor, entry) in enumerate(zip(tensors, meta)):
            if "tokens" not in entry:
                entry["tokens"] = tensor.shape[0]
            if "text" not in entry:
                entry["text"] = f"Session {i+1}"  # Placeholder if no text available
        
        print(f"[green]Loaded[/green] memory: {len(tensors)} sessions")
        return tensors, meta
    return [], []

def append(tensors: List[torch.Tensor], new: torch.Tensor, meta: List[Dict], meta_row: Dict) -> Tuple[List[torch.Tensor], List[Dict]]:
    """Append new embedding to ragged tensor list and save."""
    # Ensure new entry has all required fields
    meta_row = normalize_meta_entry(meta_row, tokens=new.shape[0])
    
    tensors.append(new)
    meta.append(meta_row)
    
    # Save both tensor and normalized metadata
    torch.save(tensors, MEM_PATH)
    json.dump(meta, open(META_PATH, "w"), indent=2)  # Pretty print JSON
    return tensors, meta

def to_batch(tensors: List[torch.Tensor], max_tokens: int = 32) -> torch.Tensor:
    """Convert ragged list to padded batch tensor for analysis."""
    padded = []
    for t in tensors:
        if t.shape[0] > max_tokens:
            padded.append(t[:max_tokens])
        else:
            pad_size = max_tokens - t.shape[0]
            padded.append(torch.cat([t, torch.zeros(pad_size, t.shape[1])]))
    return torch.stack(padded)  # [sessions, max_tokens, dim]

def flatten(tensors: List[torch.Tensor]) -> Tuple[torch.Tensor, List[int]]:
    """Flatten all tokens for PCA/drift analysis, return token counts for mapping back."""
    token_counts = [t.shape[0] for t in tensors]
    flat = torch.cat(tensors)
    return flat, token_counts

def save(tensors: List[torch.Tensor], meta: List[Dict]) -> None:
    """Save tensors and metadata to disk."""
    torch.save(tensors, MEM_PATH)
    json.dump(meta, open(META_PATH, "w"), indent=2) 
