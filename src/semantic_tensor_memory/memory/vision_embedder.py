"""
Vision Modality Implementation for Universal Multimodal STM

Demonstrates extension to visual inputs using CLIP for event extraction
and dual-resolution embedding that preserves STM's core concepts.
"""

import torch
from typing import List, Any, Dict, Optional
import time
from PIL import Image
import numpy as np

from .universal_core import (
    ModalityEmbedder, Modality, EventDescriptor, UniversalEmbedding
)

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git")

class VisionEmbedder(ModalityEmbedder):
    """
    Vision modality implementation using CLIP for dual-resolution analysis.
    
    Extends STM concepts to visual data:
    - Event-level: Visual objects, actions, scene changes
    - Sequence-level: Overall scene understanding and context
    """
    
    def __init__(self, model_name: str = "ViT-B/32"):
        if not CLIP_AVAILABLE:
            raise ImportError("CLIP is required for vision embedding. Install with: pip install git+https://github.com/openai/CLIP.git")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model_name = model_name
        
        self._modality = Modality.VISION
        self._embedding_dim = self.model.visual.output_dim  # 512 for ViT-B/32
        
        # Predefined visual event templates for object/action detection
        self.event_templates = [
            "a person",
            "a door", 
            "movement",
            "entering",
            "leaving",
            "sitting",
            "standing",
            "walking",
            "car",
            "animal",
            "food",
            "furniture",
            "electronics",
            "light change",
            "weather change",
            "crowd",
            "empty space"
        ]
    
    @property
    def modality(self) -> Modality:
        return self._modality
    
    @property
    def embedding_dimension(self) -> int:
        return self._embedding_dim
    
    def extract_events(self, raw_data: Any, **kwargs) -> List[EventDescriptor]:
        """
        Extract visual events from image using CLIP-based scene understanding.
        
        Transforms visual content into structured events while preserving
        STM's event-level granularity concept.
        """
        # Handle different input types
        if isinstance(raw_data, str):
            # File path
            image = Image.open(raw_data).convert('RGB')
        elif isinstance(raw_data, Image.Image):
            # PIL Image
            image = raw_data.convert('RGB')
        elif isinstance(raw_data, np.ndarray):
            # NumPy array
            image = Image.fromarray(raw_data).convert('RGB')
        else:
            raise ValueError(f"Unsupported image format: {type(raw_data)}")
        
        # Preprocess image for CLIP
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Extract events using CLIP text-image similarity
        events = []
        current_time = time.time()
        
        with torch.no_grad():
            # Encode image
            image_features = self.model.encode_image(image_tensor)
            
            # Test against event templates
            text_tokens = clip.tokenize(self.event_templates).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            
            # Calculate similarities
            similarities = torch.cosine_similarity(image_features, text_features)
            
            # Create events for detected concepts (above threshold)
            threshold = kwargs.get('detection_threshold', 0.15)
            
            for i, (template, similarity) in enumerate(zip(self.event_templates, similarities)):
                if similarity.item() > threshold:
                    event = EventDescriptor(
                        event_type=f"vision_detected_{template.replace(' ', '_')}",
                        confidence=float(similarity.item()),
                        timestamp=current_time + (i * 0.001),
                        duration=None,  # Instantaneous detection
                        location=f"scene_region_{i}",  # Could be enhanced with spatial detection
                        metadata={
                            'detected_concept': template,
                            'clip_similarity': float(similarity.item()),
                            'model_name': self.model_name,
                            'image_size': image.size,
                            'detection_rank': i
                        }
                    )
                    events.append(event)
        
        # If no events detected, create a general "visual_scene" event
        if not events:
            events.append(EventDescriptor(
                event_type="vision_general_scene",
                confidence=1.0,
                timestamp=current_time,
                duration=None,
                location="full_scene",
                metadata={
                    'detected_concept': "general visual content",
                    'image_size': image.size,
                    'model_name': self.model_name
                }
            ))
        
        return events
    
    def embed_events(self, events: List[EventDescriptor], **kwargs) -> UniversalEmbedding:
        """
        Create universal embedding from visual events using CLIP.
        
        Preserves dual-resolution approach:
        - Event embeddings: Individual concept embeddings
        - Sequence embedding: Overall scene embedding
        """
        target_device = torch.device("cpu")

        if not events:
            # Create empty embeddings for no events
            event_embeddings = torch.zeros(1, self._embedding_dim, device=target_device)
            sequence_embedding = torch.zeros(self._embedding_dim, device=target_device)
        else:
            # Get image from first event metadata
            # Note: In a real implementation, you'd want to store the processed image
            # For now, we'll create embeddings based on detected concepts

            # Event-level embeddings: Encode detected concepts
            event_texts = [event.metadata.get('detected_concept', event.event_type) for event in events]

            with torch.no_grad():
                # Encode each detected concept separately
                event_embeddings_list = []
                for text in event_texts:
                    text_tokens = clip.tokenize([text]).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    event_embeddings_list.append(text_features.squeeze(0))

                event_embeddings = torch.stack(event_embeddings_list).to(target_device)

                # Sequence-level embedding: Combined scene understanding
                # Create a description of the overall scene
                scene_description = f"A scene containing: {', '.join(event_texts[:5])}"  # Limit length
                scene_tokens = clip.tokenize([scene_description]).to(self.device)
                sequence_embedding = self.model.encode_text(scene_tokens).squeeze(0).to(target_device)

        if event_embeddings.device != target_device:
            event_embeddings = event_embeddings.to(target_device)
        if sequence_embedding.device != target_device:
            sequence_embedding = sequence_embedding.to(target_device)

        # Calculate quality metrics
        event_coherence = self._calculate_visual_coherence(events)
        sequence_coherence = torch.norm(sequence_embedding).item()
        extraction_confidence = self._calculate_extraction_confidence(events)
        
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
                'clip_model': self.model_name,
                'num_detected_concepts': len(events),
                'detection_threshold': 0.15,  # Could be parameterized
                'device': self.device
            },
            processing_metadata={},  # Will be updated by process_raw_data
            event_coherence=event_coherence,
            sequence_coherence=sequence_coherence,
            extraction_confidence=extraction_confidence
        )
        
        return universal_embedding
    
    def _calculate_visual_coherence(self, events: List[EventDescriptor]) -> float:
        """Calculate coherence of detected visual events."""
        if len(events) < 2:
            return 1.0
        
        # Use confidence scores as a proxy for coherence
        confidences = [event.confidence for event in events]
        return sum(confidences) / len(confidences)
    
    def _calculate_extraction_confidence(self, events: List[EventDescriptor]) -> float:
        """Calculate overall confidence in event extraction."""
        if not events:
            return 0.0
        
        # Average confidence weighted by detection strength
        total_confidence = sum(event.confidence for event in events)
        return min(total_confidence / len(events), 1.0)
    
    def extract_motion_events(self, image_sequence: List[Any], **kwargs) -> List[EventDescriptor]:
        """
        Extract motion events from a sequence of images.
        
        This demonstrates how the event extraction can be extended for
        temporal visual analysis (video processing).
        """
        events = []
        current_time = time.time()
        
        if len(image_sequence) < 2:
            return events
        
        # Simple motion detection by comparing consecutive frames
        for i in range(len(image_sequence) - 1):
            # Process consecutive images
            events_a = self.extract_events(image_sequence[i])
            events_b = self.extract_events(image_sequence[i + 1])
            
            # Compare detected concepts to infer motion
            concepts_a = set(e.metadata.get('detected_concept', '') for e in events_a)
            concepts_b = set(e.metadata.get('detected_concept', '') for e in events_b)
            
            # Detect changes
            new_concepts = concepts_b - concepts_a
            lost_concepts = concepts_a - concepts_b
            
            # Create motion events
            for concept in new_concepts:
                motion_event = EventDescriptor(
                    event_type=f"vision_motion_{concept.replace(' ', '_')}_appears",
                    confidence=0.8,
                    timestamp=current_time + i,
                    duration=1.0,  # Assume 1 second between frames
                    location=f"frame_{i}_to_{i+1}",
                    metadata={
                        'motion_type': 'appearance',
                        'concept': concept,
                        'frame_index': i
                    }
                )
                events.append(motion_event)
            
            for concept in lost_concepts:
                motion_event = EventDescriptor(
                    event_type=f"vision_motion_{concept.replace(' ', '_')}_disappears",
                    confidence=0.8,
                    timestamp=current_time + i,
                    duration=1.0,
                    location=f"frame_{i}_to_{i+1}",
                    metadata={
                        'motion_type': 'disappearance',
                        'concept': concept,
                        'frame_index': i
                    }
                )
                events.append(motion_event)
        
        return events

# Factory function for easy creation
def create_vision_embedding(image: Any) -> UniversalEmbedding:
    """Create universal embedding from image."""
    embedder = VisionEmbedder()
    return embedder.process_raw_data(image)

def get_visual_events(image: Any, threshold: float = 0.15) -> List[EventDescriptor]:
    """Extract visual events from image."""
    embedder = VisionEmbedder()
    return embedder.extract_events(image, detection_threshold=threshold) 