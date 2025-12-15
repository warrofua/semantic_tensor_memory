"""
Text Modality Implementation for Semantic Tensor Analysis

Transforms the dual BERT + S-BERT system into the universal modality interface.
Preserves all semantic intelligence while enabling multimodal extensibility.
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

from .universal_core import Modality, ModalityEmbedder, EventDescriptor, UniversalEmbedding

# Initialize models (carry over from dual_embedder.py)
BERT_MODEL = "bert-base-uncased"
SBERT_MODEL = "sentence-transformers/all-mpnet-base-v2"

class TextEmbedder(ModalityEmbedder):
    """
    Text modality implementation using dual BERT + S-BERT architecture.
    
    Transforms text into universal embeddings while preserving:
    - Token-level granularity (STM's core innovation) → Event-level granularity
    - Sentence-level semantics (S-BERT quality) → Sequence-level semantics
    - Multi-resolution analysis capabilities
    """
    
    def __init__(self):
        # Fix threading issues
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        self._use_stub = os.environ.get("STA_LIGHTWEIGHT_EMBEDDER") == "1"
        self._modality = Modality.TEXT

        if self._use_stub:
            # Lightweight, deterministic stub for tests/offline runs
            self.bert_tokenizer = None
            self.bert_model = None
            self.sbert_model = None
            self._embedding_dim = 16
        else:
            # Initialize models
            self.bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
            self.bert_model = AutoModel.from_pretrained(BERT_MODEL)
            self.sbert_model = SentenceTransformer(SBERT_MODEL)
            self._embedding_dim = self.bert_model.config.hidden_size  # 768
    
    @property
    def modality(self) -> Modality:
        return self._modality
    
    @property 
    def embedding_dimension(self) -> int:
        return self._embedding_dim
    
    def extract_events(self, raw_data: Any, **kwargs) -> List[EventDescriptor]:
        """
        Extract text events (tokens/phrases) from raw text.
        
        Transforms BERT tokenization into universal event descriptors,
        preserving STM's token-level granularity.
        """
        text = str(raw_data)
        
        if self._use_stub:
            tokens = text.split()
            inputs = {"input_ids": torch.tensor([[i for i, _ in enumerate(tokens)]])}
        else:
            # Tokenize with BERT (preserving original approach)
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True, padding=False)
            tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze(0))
        
        # Convert tokens to event descriptors
        events = []
        current_time = time.time()
        
        for i, token in enumerate(tokens):
            # Skip special tokens for cleaner event extraction
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
                
            # Create event descriptor for each token
            raw_id = inputs['input_ids'][0][i]
            token_id = int(raw_id) if isinstance(raw_id, (int, float)) else raw_id.item()
            event = EventDescriptor(
                event_type=f"text_token_{token.replace('##', '')}",  # Handle wordpieces
                confidence=1.0,  # High confidence for text tokens
                timestamp=current_time + (i * 0.001),  # Microsecond offsets for ordering
                duration=None,  # Instantaneous events
                location=f"position_{i}",  # Position in sequence
                metadata={
                    'token': token,
                    'token_id': token_id,
                    'position': i,
                    'is_wordpiece': token.startswith('##'),
                    'original_text': text
                }
            )
            events.append(event)
        
        return events
    
    def embed_events(self, events: List[EventDescriptor], **kwargs) -> UniversalEmbedding:
        """
        Create universal embedding from text events using dual BERT + S-BERT.
        
        Preserves the dual-resolution approach:
        - Event embeddings: BERT token embeddings (STM granularity)
        - Sequence embedding: S-BERT sentence embedding (semantic quality)
        """
        # Reconstruct text from events for embedding
        original_text = events[0].metadata['original_text'] if events else ""

        # Optional contextual signals (previous sessions, temporal position, domain hints)
        context_window = kwargs.get("context_window") or []
        temporal_position = kwargs.get("temporal_position") or kwargs.get("session_position")
        total_sessions = kwargs.get("total_sessions")
        domain_hint = kwargs.get("domain_hint") or kwargs.get("domain")
        session_timestamp = kwargs.get("session_timestamp")
        previous_timestamp = kwargs.get("previous_timestamp")

        context_prefix = self._build_context_prefix(
            context_window=context_window,
            temporal_position=temporal_position,
            total_sessions=total_sessions,
            domain_hint=domain_hint,
            session_timestamp=session_timestamp,
            previous_timestamp=previous_timestamp,
        )

        # Use contextualized text for sequence-level embedding; keep raw text for token granularity
        sequence_text = f"{context_prefix} {original_text}".strip() if context_prefix else original_text

        if self._use_stub:
            torch.manual_seed(abs(hash(sequence_text)) % (2**31))
            num_tokens = max(len(events), 1)
            event_embeddings = torch.randn(num_tokens, self._embedding_dim)
            sequence_embedding = torch.randn(self._embedding_dim)
            context_vector = torch.randn(self._embedding_dim) if context_prefix else None
        else:
            # BERT: Event-level embeddings (token granularity)
            bert_inputs = self.bert_tokenizer(original_text, return_tensors="pt", truncation=True, padding=False)
            
            with torch.inference_mode():
                bert_output = self.bert_model(**bert_inputs)
                event_embeddings = bert_output.last_hidden_state.squeeze(0)  # [tokens, 768]
                context_vector = (
                    self.sbert_model.encode(context_prefix, convert_to_tensor=True, show_progress_bar=False)
                    if context_prefix
                    else None
                )
                # S-BERT: Sequence-level embedding (semantic quality, contextualized)
                sequence_embedding = self.sbert_model.encode(
                    sequence_text, convert_to_tensor=True, show_progress_bar=False
                )

        if context_prefix and context_vector is not None:
            # Broadcast context into token-level space to make drift sensitive to evolving meaning
            event_embeddings = event_embeddings + context_vector.to(event_embeddings)
        
        # Calculate quality metrics
        event_coherence = 1.0 if self._use_stub else self._calculate_event_coherence(event_embeddings)
        sequence_coherence = torch.norm(sequence_embedding).item()
        extraction_confidence = 1.0  # High confidence for clean text
        
        # Create universal embedding
        universal_embedding = UniversalEmbedding(
            event_embeddings=event_embeddings,
            sequence_embedding=sequence_embedding,
            modality=self.modality,
            events=events,
            session_id="",  # Will be set by process_raw_data
            timestamp=0.0,  # Will be set by process_raw_data
            duration=0.0,   # Will be set by process_raw_data
            sensor_metadata={
                'bert_model': BERT_MODEL,
                'sbert_model': SBERT_MODEL,
                'text_length': len(original_text),
                'num_tokens': len(events)
            },
            processing_metadata={
                'context_applied': bool(context_prefix),
                'context_window_size': len(context_window),
                'temporal_position': temporal_position,
                'total_sessions': total_sessions,
                'domain_hint': domain_hint,
                'context_prefix': context_prefix if context_prefix else None,
            },
            event_coherence=event_coherence,
            sequence_coherence=sequence_coherence,
            extraction_confidence=extraction_confidence
        )
        
        return universal_embedding

    def _build_context_prefix(
        self,
        context_window: List[str],
        temporal_position: Optional[int],
        total_sessions: Optional[int],
        domain_hint: Optional[str],
        session_timestamp: Optional[float],
        previous_timestamp: Optional[float],
    ) -> str:
        """Assemble lightweight context string used to contextualize embeddings."""
        parts: List[str] = []

        if domain_hint:
            parts.append(f"[DOMAIN: {domain_hint}]")

        if temporal_position is not None:
            if total_sessions:
                parts.append(f"[SESSION {temporal_position}/{total_sessions}]")
            else:
                parts.append(f"[SESSION {temporal_position}]")

        if session_timestamp is not None and previous_timestamp is not None:
            delta = session_timestamp - previous_timestamp
            parts.append(f"[DELTA_T: {delta:.1f}s]")

        if context_window:
            trimmed = [self._truncate_context_text(text) for text in context_window[-3:]]
            merged_context = " || ".join(t for t in trimmed if t)
            if merged_context:
                parts.append(f"Previous context: {merged_context}")

        joined = " ".join(parts).strip()
        if not joined:
            return ""
        joined = " ".join(joined.split())  # normalize whitespace
        max_len = 400
        if len(joined) > max_len:
            joined = joined[: max_len - 3] + "..."
        return joined

    def _truncate_context_text(self, text: str, limit: int = 240) -> str:
        """Trim context text to avoid overly long prompts."""
        text = text.strip()
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."
    
    def _calculate_event_coherence(self, event_embeddings: torch.Tensor) -> float:
        """Calculate coherence between adjacent events (tokens)."""
        if event_embeddings.shape[0] < 2:
            return 1.0
        
        # Calculate average cosine similarity between adjacent tokens
        similarities = []
        for i in range(event_embeddings.shape[0] - 1):
            sim = torch.cosine_similarity(
                event_embeddings[i], event_embeddings[i + 1], dim=0
            ).item()
            similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def extract_semantic_events(self, raw_data: Any, **kwargs) -> List[EventDescriptor]:
        """
        Alternative event extraction that focuses on semantic units rather than tokens.
        
        Useful for higher-level semantic analysis while preserving granularity.
        """
        text = str(raw_data)
        
        # Extract semantic events (phrases, entities, concepts)
        events = []
        current_time = time.time()
        
        # Simple semantic segmentation (can be enhanced with NER, phrase extraction, etc.)
        sentences = re.split(r'[.!?]+', text)
        words = text.split()
        
        # Word-level events (more semantic than tokens)
        for i, word in enumerate(words):
            if word.strip():
                event = EventDescriptor(
                    event_type=f"semantic_word_{word.lower()}",
                    confidence=1.0,
                    timestamp=current_time + (i * 0.01),
                    duration=None,
                    location=f"word_position_{i}",
                    metadata={
                        'word': word,
                        'position': i,
                        'sentence_context': self._get_sentence_context(word, sentences),
                        'original_text': text
                    }
                )
                events.append(event)
        
        return events
    
    def _get_sentence_context(self, word: str, sentences: List[str]) -> str:
        """Find which sentence contains the word."""
        for sentence in sentences:
            if word in sentence:
                return sentence.strip()
        return ""

# Backward compatibility functions
def create_text_embedding(text: str) -> UniversalEmbedding:
    """Create universal embedding from text (compatible with existing code)."""
    embedder = TextEmbedder()
    return embedder.process_raw_data(text)

def get_text_events(text: str) -> List[EventDescriptor]:
    """Extract text events for analysis."""
    embedder = TextEmbedder()
    return embedder.extract_events(text)

# STM-compatible interface
@torch.inference_mode()
def embed_sentence(text: str) -> torch.Tensor:
    """
    STM-compatible function: returns a single sequence embedding as [1, dim].
    
    Legacy callers expect a 2D tensor; we provide the sequence embedding from
    TextEmbedder to match semantic behavior used across the app.
    """
    embedding = create_text_embedding(text)
    seq = embedding.sequence_embedding
    return seq.unsqueeze(0) if seq.ndim == 1 else seq

def get_token_count(text: str) -> int:
    """STM-compatible function: returns event count."""
    events = get_text_events(text)
    return len(events) 
