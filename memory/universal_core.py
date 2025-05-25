"""
Universal Multimodal Semantic Tensor Memory - Core Architecture

This module defines the foundational interfaces and data structures for 
universal multimodal semantic memory that works across text, vision, audio,
sensors, and any future modalities.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Tuple
import torch
from enum import Enum
import time
from pathlib import Path

class Modality(Enum):
    """Supported modalities for universal semantic memory."""
    TEXT = "text"
    VISION = "vision" 
    AUDIO = "audio"
    THERMAL = "thermal"
    MOTION = "motion"
    PRESSURE = "pressure"
    # Extensible for future modalities

@dataclass
class EventDescriptor:
    """
    Universal event descriptor that works across all modalities.
    
    Examples:
    - Text: ["person", "enters", "room"] 
    - Vision: ["person_detected", "door_opened", "motion_started"]
    - Audio: ["speech_detected", "door_slam", "footsteps"]
    - Thermal: ["temperature_spike", "heat_source_detected"]
    """
    event_type: str              # "person_enters", "door_opens", "temperature_spike"
    confidence: float            # 0.0 - 1.0 confidence in event detection
    timestamp: float             # Unix timestamp with microsecond precision
    duration: Optional[float]    # Event duration in seconds (if applicable)
    location: Optional[str]      # Spatial context (room, coordinates, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)  # Modality-specific data

@dataclass 
class UniversalEmbedding:
    """
    Universal embedding that preserves STM's dual-resolution concept
    across all modalities.
    
    This is the core data structure that enables semantic tensor memory
    for any input modality while maintaining:
    - Event-level granularity (STM's core innovation)
    - Sequence-level semantic understanding
    - Cross-modal compatibility
    """
    event_embeddings: torch.Tensor       # [events, dim] - Granular analysis (STM core)
    sequence_embedding: torch.Tensor     # [dim] - Holistic semantic representation
    modality: Modality                   # Source modality
    events: List[EventDescriptor]        # Structured event data
    session_id: str                      # Unique session identifier
    timestamp: float                     # Session timestamp
    duration: float                      # Session duration in seconds
    
    # Modality-specific metadata
    sensor_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Quality metrics
    event_coherence: Optional[float] = None      # Internal event consistency
    sequence_coherence: Optional[float] = None   # Overall semantic coherence
    extraction_confidence: Optional[float] = None  # Event extraction quality

class ModalityEmbedder(ABC):
    """
    Abstract interface for modality-specific embedding systems.
    
    Each modality (text, vision, audio, sensors) implements this interface
    to provide uniform embedding capabilities while preserving the 
    dual-resolution semantic tensor memory concept.
    """
    
    @property
    @abstractmethod
    def modality(self) -> Modality:
        """Return the modality this embedder handles."""
        pass
    
    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the embedding dimension for this modality."""
        pass
    
    @abstractmethod
    def extract_events(self, raw_data: Any, **kwargs) -> List[EventDescriptor]:
        """
        Extract structured events from raw sensor data.
        
        Args:
            raw_data: Modality-specific input (text, image, audio, sensor reading)
            **kwargs: Modality-specific parameters
            
        Returns:
            List of structured event descriptors
        """
        pass
    
    @abstractmethod
    def embed_events(self, events: List[EventDescriptor], **kwargs) -> UniversalEmbedding:
        """
        Create dual-resolution embeddings from event descriptors.
        
        Args:
            events: List of event descriptors from extract_events()
            **kwargs: Modality-specific embedding parameters
            
        Returns:
            UniversalEmbedding with both event-level and sequence-level representations
        """
        pass
    
    def process_raw_data(self, raw_data: Any, session_id: Optional[str] = None, **kwargs) -> UniversalEmbedding:
        """
        Complete pipeline: raw data → events → embeddings.
        
        This is the main entry point for processing any modality data.
        """
        if session_id is None:
            session_id = f"{self.modality.value}_{int(time.time() * 1000000)}"
        
        start_time = time.time()
        
        # Extract events from raw data
        events = self.extract_events(raw_data, **kwargs)
        
        # Create embeddings
        embedding = self.embed_events(events, **kwargs)
        
        # Update metadata
        processing_time = time.time() - start_time
        embedding.session_id = session_id
        embedding.timestamp = start_time
        embedding.duration = processing_time
        embedding.processing_metadata.update({
            'processing_time': processing_time,
            'num_events': len(events),
            'raw_data_type': type(raw_data).__name__
        })
        
        return embedding

class UniversalMemoryStore:
    """
    Universal memory store that handles multimodal semantic tensor memory.
    
    Manages storage, retrieval, and analysis of universal embeddings across
    all modalities while preserving STM's core tensor memory capabilities.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.embeddings: List[UniversalEmbedding] = []
        self.modality_counts: Dict[Modality, int] = {}
        self.storage_path = Path(storage_path) if storage_path else Path("data/universal")
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    def add_session(self, embedding: UniversalEmbedding) -> int:
        """Add a universal embedding session to memory."""
        session_index = len(self.embeddings)
        self.embeddings.append(embedding)
        
        # Update modality counts
        modality = embedding.modality
        self.modality_counts[modality] = self.modality_counts.get(modality, 0) + 1
        
        # Save to storage
        self._save_session(embedding, session_index)
        
        return session_index
    
    def get_sessions_by_modality(self, modality: Modality) -> List[Tuple[int, UniversalEmbedding]]:
        """Get all sessions for a specific modality."""
        return [(i, emb) for i, emb in enumerate(self.embeddings) 
                if emb.modality == modality]
    
    def get_event_tensors(self, modality: Optional[Modality] = None) -> List[torch.Tensor]:
        """Get event-level tensors (STM-compatible interface)."""
        if modality:
            return [emb.event_embeddings for emb in self.embeddings 
                    if emb.modality == modality]
        return [emb.event_embeddings for emb in self.embeddings]
    
    def get_sequence_tensors(self, modality: Optional[Modality] = None) -> torch.Tensor:
        """Get sequence-level tensors for semantic analysis."""
        if modality:
            sequences = [emb.sequence_embedding for emb in self.embeddings 
                        if emb.modality == modality]
        else:
            sequences = [emb.sequence_embedding for emb in self.embeddings]
        
        if not sequences:
            return torch.empty(0, 768)  # Return empty tensor with appropriate shape
        
        return torch.stack(sequences)
    
    def analyze_cross_modal_drift(self, session_a: int, session_b: int) -> Dict[str, Any]:
        """
        Analyze semantic drift between sessions, potentially across modalities.
        
        This extends STM's drift analysis to work across different input modalities,
        enabling novel cross-modal semantic analysis.
        """
        emb_a = self.embeddings[session_a]
        emb_b = self.embeddings[session_b]
        
        # Sequence-level similarity (cross-modal compatible)
        sequence_similarity = torch.cosine_similarity(
            emb_a.sequence_embedding, emb_b.sequence_embedding, dim=0
        ).item()
        
        # Event-level analysis (if same modality)
        event_analysis = {}
        if emb_a.modality == emb_b.modality:
            event_similarity = torch.cosine_similarity(
                emb_a.event_embeddings.mean(0), emb_b.event_embeddings.mean(0), dim=0
            ).item()
            event_analysis = {
                'event_similarity': event_similarity,
                'event_count_change': len(emb_b.events) - len(emb_a.events),
                'same_modality': True
            }
        else:
            event_analysis = {
                'same_modality': False,
                'cross_modal_analysis': True
            }
        
        return {
            'sequence_similarity': sequence_similarity,
            'sequence_drift': 1 - sequence_similarity,
            'modality_a': emb_a.modality.value,
            'modality_b': emb_b.modality.value,
            'event_analysis': event_analysis,
            'timestamp_gap': abs(emb_b.timestamp - emb_a.timestamp),
            'coherence_comparison': {
                'a': emb_a.sequence_coherence,
                'b': emb_b.sequence_coherence
            }
        }
    
    def _save_session(self, embedding: UniversalEmbedding, index: int):
        """Save session to persistent storage."""
        session_file = self.storage_path / f"session_{index:06d}_{embedding.modality.value}.pkl"
        torch.save(embedding, session_file)
    
    def load_from_storage(self):
        """Load existing sessions from storage."""
        if not self.storage_path.exists():
            return
        
        session_files = sorted(self.storage_path.glob("session_*.pkl"))
        for session_file in session_files:
            try:
                embedding = torch.load(session_file)
                self.embeddings.append(embedding)
                modality = embedding.modality
                self.modality_counts[modality] = self.modality_counts.get(modality, 0) + 1
            except Exception as e:
                print(f"Warning: Could not load {session_file}: {e}")

def create_universal_embedder(modality: str) -> ModalityEmbedder:
    """
    Factory function to create appropriate embedder for a modality.
    
    This enables easy extension to new modalities without changing core code.
    """
    modality_enum = Modality(modality.lower())
    
    if modality_enum == Modality.TEXT:
        from memory.text_embedder import TextEmbedder
        return TextEmbedder()
    elif modality_enum == Modality.VISION:
        from memory.vision_embedder import VisionEmbedder
        return VisionEmbedder()
    elif modality_enum == Modality.AUDIO:
        raise NotImplementedError(f"Audio modality not yet implemented - ready for development")
    else:
        raise NotImplementedError(f"Modality {modality} not yet implemented")

# Backward compatibility functions for existing STM code
def get_universal_memory_store() -> UniversalMemoryStore:
    """Get universal memory store instance."""
    return UniversalMemoryStore()

def embed_text(text: str) -> torch.Tensor:
    """STM-compatible text embedding function."""
    text_embedder = create_universal_embedder("text")
    universal_embedding = text_embedder.process_raw_data(text)
    return universal_embedding.event_embeddings 