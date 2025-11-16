from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Initialize models
BERT_MODEL = "bert-base-uncased"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Best semantic quality

bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
bert_model = AutoModel.from_pretrained(BERT_MODEL)
sbert_model = SentenceTransformer(SBERT_MODEL)

BERT_DIM = bert_model.config.hidden_size  # 768
SBERT_DIM = sbert_model.get_sentence_embedding_dimension()  # 768

# Fix threading issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

@dataclass
class DualEmbedding:
    """
    Dual-resolution semantic representation:
    - Token-level: Preserves STM's granular analysis capability
    - Sentence-level: Provides high-quality semantic comparison
    """
    token_embeddings: torch.Tensor      # [tokens, bert_dim] - STM core
    sentence_embedding: torch.Tensor    # [sbert_dim] - semantic intelligence
    text: str                          # Original text
    token_count: int                   # Number of tokens
    tokens: List[str]                  # Actual token strings (for debugging)

@torch.inference_mode()
def create_dual_embedding(text: str) -> DualEmbedding:
    """
    Create both token-level (BERT) and sentence-level (S-BERT) embeddings.
    
    This preserves STM's tensor nature while adding semantic intelligence:
    - Token embeddings: For granular drift analysis (STM's unique value)
    - Sentence embeddings: For high-quality semantic comparison
    
    Args:
        text: Input text
        
    Returns:
        DualEmbedding with both representations indexed to same content
    """
    # BERT: Token-level embeddings (STM core)
    bert_inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=False)
    bert_output = bert_model(**bert_inputs)
    token_embeddings = bert_output.last_hidden_state.squeeze(0)  # [tokens, 768]
    
    # S-BERT: Sentence-level semantic embedding  
    sentence_embedding = sbert_model.encode(text, convert_to_tensor=True, show_progress_bar=False)
    
    # Extract tokens for debugging/analysis
    tokens = bert_tokenizer.convert_ids_to_tokens(bert_inputs['input_ids'].squeeze(0))
    
    return DualEmbedding(
        token_embeddings=token_embeddings,
        sentence_embedding=sentence_embedding,
        text=text,
        token_count=len(tokens),
        tokens=tokens
    )

class DualMemoryStore:
    """
    Memory store that maintains both token and sentence-level representations.
    
    Enables multi-resolution analysis:
    - Token-level drift (STM's core innovation)
    - Sentence-level semantic similarity (enhanced quality)
    - Cross-resolution insights
    """
    
    def __init__(self):
        self.embeddings: List[DualEmbedding] = []
        self.metadata: List[Dict] = []
    
    def add_session(self, text: str, meta: Optional[Dict] = None) -> int:
        """Add a session with dual embeddings."""
        embedding = create_dual_embedding(text)
        self.embeddings.append(embedding)
        
        if meta is None:
            meta = {}
        meta.update({
            'text': text,
            'tokens': embedding.token_count,
            'session_id': len(self.embeddings) - 1
        })
        self.metadata.append(meta)
        
        return len(self.embeddings) - 1
    
    def get_token_tensors(self) -> List[torch.Tensor]:
        """Get token-level tensors for STM analysis."""
        return [emb.token_embeddings for emb in self.embeddings]
    
    def get_sentence_tensors(self) -> torch.Tensor:
        """Get sentence-level tensors for semantic analysis."""
        return torch.stack([emb.sentence_embedding for emb in self.embeddings])
    
    def analyze_multi_resolution_drift(self, session_a: int, session_b: int) -> Dict:
        """
        Compare sessions at both token and sentence levels.
        
        Returns comprehensive drift analysis combining STM's granular 
        tracking with high-quality semantic comparison.
        """
        emb_a = self.embeddings[session_a]
        emb_b = self.embeddings[session_b]
        
        # Token-level analysis (STM's core strength)
        from .sequence_drift import sequence_drift, semantic_coherence_score
        token_drift = sequence_drift([emb_a.token_embeddings, emb_b.token_embeddings])
        
        coherence_a = semantic_coherence_score(emb_a.token_embeddings)
        coherence_b = semantic_coherence_score(emb_b.token_embeddings)
        
        # Sentence-level analysis (semantic intelligence)
        sentence_similarity = torch.cosine_similarity(
            emb_a.sentence_embedding, emb_b.sentence_embedding, dim=0
        ).item()
        sentence_drift = 1 - sentence_similarity
        
        # Cross-resolution insights
        token_mean_similarity = torch.cosine_similarity(
            emb_a.token_embeddings.mean(0), emb_b.token_embeddings.mean(0), dim=0
        ).item()
        
        # Detect semantic vs structural changes
        semantic_vs_structural = abs(sentence_similarity - token_mean_similarity)
        
        return {
            'token_level': {
                'sequence_drift': token_drift[0] if token_drift else 0.0,
                'coherence_a': coherence_a,
                'coherence_b': coherence_b,
                'coherence_change': coherence_b - coherence_a,
                'token_count_change': emb_b.token_count - emb_a.token_count
            },
            'sentence_level': {
                'semantic_similarity': sentence_similarity,
                'semantic_drift': sentence_drift,
                'high_quality_comparison': True  # S-BERT advantage
            },
            'cross_resolution': {
                'token_mean_similarity': token_mean_similarity,
                'semantic_vs_structural_ratio': semantic_vs_structural,
                'interpretation': self._interpret_drift_pattern(
                    sentence_similarity, token_mean_similarity, semantic_vs_structural
                )
            }
        }
    
    def _interpret_drift_pattern(self, sent_sim: float, token_sim: float, diff: float) -> str:
        """Interpret the relationship between sentence and token-level changes."""
        if diff < 0.1:
            return "Consistent semantic and structural change"
        elif sent_sim > token_sim:
            return "Semantic meaning preserved despite structural changes"
        elif token_sim > sent_sim:
            return "Similar structure but semantic meaning shifted"
        else:
            return "Complex semantic-structural divergence detected"

def create_enhanced_categories(memory_store: DualMemoryStore) -> Dict:
    """
    Use both token and sentence embeddings for superior category discovery.
    
    Combines:
    - Token-level concept extraction (granular)
    - Sentence-level semantic clustering (high quality)
    """
    from semantic_tensor_analysis.visualization.viz.holistic_semantic_analysis import extract_all_concepts_globally
    
    # Extract concepts from token-level analysis (STM approach)
    token_tensors = memory_store.get_token_tensors()
    token_concepts = extract_all_concepts_globally(token_tensors, memory_store.metadata)
    
    # Use sentence embeddings for high-quality clustering
    sentence_embeddings = {}
    for concept in token_concepts.keys():
        # Use S-BERT for concept embeddings (better semantic quality)
        concept_emb = sbert_model.encode(concept, convert_to_tensor=True, show_progress_bar=False)
        sentence_embeddings[concept] = concept_emb
    
    return {
        'concepts': token_concepts,
        'embeddings': sentence_embeddings,
        'method': 'dual_resolution',
        'token_level_extraction': True,
        'sentence_level_clustering': True
    }

# Convenience functions for backward compatibility
@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """STM-compatible function: returns token embeddings."""
    return create_dual_embedding(text).token_embeddings

def get_token_count(text: str) -> int:
    """STM-compatible function: returns token count."""
    return create_dual_embedding(text).token_count 
