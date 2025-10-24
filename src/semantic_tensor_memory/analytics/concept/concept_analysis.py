#!/usr/bin/env python3
"""
Concept Analysis Module for Universal STM

Leverages existing S-BERT sequence embeddings to provide enhanced concept-level analysis
without modifying the core Universal STM architecture.
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from semantic_tensor_memory.memory.universal_core import UniversalMemoryStore, UniversalEmbedding, Modality

@dataclass
class ConceptCluster:
    """Represents a cluster of semantically similar sessions."""
    cluster_id: int
    session_indices: List[int]
    centroid: torch.Tensor
    coherence_score: float
    representative_text: str
    theme_keywords: List[str]

@dataclass
class ConceptDriftPattern:
    """Represents temporal drift patterns in concept evolution."""
    session_from: int
    session_to: int
    drift_magnitude: float
    drift_direction: str  # "diverging", "converging", "stable"
    concept_shift_keywords: List[str]
    temporal_gap: float

@dataclass
class ConceptEvolution:
    """Complete concept evolution analysis results."""
    total_sessions: int
    concept_clusters: List[ConceptCluster]
    drift_patterns: List[ConceptDriftPattern]
    concept_velocity: List[float]  # Rate of change over time
    major_shifts: List[int]  # Session indices of major concept changes
    concept_persistence: Dict[str, float]  # How long concepts persist

class ConceptAnalyzer:
    """
    Enhanced concept analysis using existing S-BERT sequence embeddings.
    
    Does NOT modify the Universal STM core - only analyzes existing data.
    """
    
    def __init__(self, store: UniversalMemoryStore):
        self.store = store
        self.sequence_embeddings = None
        self.session_metadata = None
        
    def _extract_sequence_data(self, modality: Optional[Modality] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """Extract S-BERT sequence embeddings and metadata from existing store."""
        if modality:
            sessions = self.store.get_sessions_by_modality(modality)
            embeddings = [emb for _, emb in sessions]
        else:
            embeddings = self.store.embeddings
            
        if not embeddings:
            raise ValueError("No embeddings found in store")
        
        # Extract S-BERT sequence embeddings (already computed!)
        sequence_tensors = torch.stack([emb.sequence_embedding for emb in embeddings])
        
        # Extract metadata for interpretation
        metadata = []
        for i, emb in enumerate(embeddings):
            # Get first few words for representation
            if emb.events:
                first_event = emb.events[0]
                text_sample = first_event.metadata.get('original_text', '')[:100]
            else:
                text_sample = f"Session {i}"
                
            metadata.append({
                'session_idx': i,
                'session_id': emb.session_id,
                'modality': emb.modality.value,
                'timestamp': emb.timestamp,
                'text_sample': text_sample,
                'num_events': len(emb.events),
                'sequence_coherence': emb.sequence_coherence,
                'event_coherence': emb.event_coherence
            })
            
        return sequence_tensors, metadata
    
    def analyze_concept_clusters(self, n_clusters: int = 5, modality: Optional[Modality] = None) -> List[ConceptCluster]:
        """
        Cluster sessions by concept similarity using existing S-BERT embeddings.
        """
        sequence_tensors, metadata = self._extract_sequence_data(modality)
        
        # Use existing S-BERT embeddings for clustering
        embeddings_np = sequence_tensors.cpu().numpy()
        
        # K-means clustering on concept space
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)
        
        clusters = []
        for cluster_id in range(n_clusters):
            # Find sessions in this cluster
            cluster_mask = cluster_labels == cluster_id
            session_indices = np.where(cluster_mask)[0].tolist()
            
            if not session_indices:
                continue
                
            # Get cluster centroid from K-means (ensure same device as sequence_tensors)
            centroid = torch.tensor(kmeans.cluster_centers_[cluster_id], device=sequence_tensors.device)
            
            # Calculate cluster coherence (average intra-cluster similarity)
            cluster_embeddings = embeddings_np[cluster_mask]
            if len(cluster_embeddings) > 1:
                similarities = cosine_similarity(cluster_embeddings)
                coherence_score = np.mean(similarities[np.triu_indices_from(similarities, k=1)])
            else:
                coherence_score = 1.0
            
            # Find most representative session (closest to centroid)
            distances = torch.norm(sequence_tensors[cluster_mask] - centroid, dim=1)
            rep_idx_in_cluster = torch.argmin(distances).item()
            rep_session_idx = session_indices[rep_idx_in_cluster]
            representative_text = metadata[rep_session_idx]['text_sample']
            
            # Extract theme keywords (simple approach - can be enhanced)
            theme_keywords = self._extract_cluster_themes(session_indices, metadata)
            
            clusters.append(ConceptCluster(
                cluster_id=cluster_id,
                session_indices=session_indices,
                centroid=centroid,
                coherence_score=float(coherence_score),
                representative_text=representative_text,
                theme_keywords=theme_keywords
            ))
            
        return sorted(clusters, key=lambda c: len(c.session_indices), reverse=True)
    
    def analyze_concept_drift_patterns(self, modality: Optional[Modality] = None) -> List[ConceptDriftPattern]:
        """
        Analyze temporal concept drift using existing cross-modal drift analysis.
        """
        sequence_tensors, metadata = self._extract_sequence_data(modality)
        
        if len(sequence_tensors) < 2:
            return []
        
        drift_patterns = []
        
        # Use existing cross-modal drift analysis function
        for i in range(len(self.store.embeddings) - 1):
            drift_analysis = self.store.analyze_cross_modal_drift(i, i + 1)
            
            # Classify drift direction
            similarity = drift_analysis['sequence_similarity']
            drift_magnitude = drift_analysis['sequence_drift']
            
            if drift_magnitude < 0.1:
                direction = "stable"
            elif drift_magnitude < 0.3:
                direction = "converging" if similarity > 0.7 else "diverging"
            else:
                direction = "diverging"
            
            # Simple keyword extraction for concept shifts
            shift_keywords = self._extract_shift_keywords(i, i + 1, metadata)
            
            drift_patterns.append(ConceptDriftPattern(
                session_from=i,
                session_to=i + 1,
                drift_magnitude=drift_magnitude,
                drift_direction=direction,
                concept_shift_keywords=shift_keywords,
                temporal_gap=drift_analysis['timestamp_gap']
            ))
            
        return drift_patterns
    
    def analyze_concept_velocity(self, modality: Optional[Modality] = None) -> List[float]:
        """
        Calculate the rate of concept change over time.
        """
        drift_patterns = self.analyze_concept_drift_patterns(modality)
        
        if not drift_patterns:
            return []
        
        # Velocity = drift magnitude / time gap (with minimum time to avoid division by zero)
        velocities = []
        for pattern in drift_patterns:
            time_gap = max(pattern.temporal_gap, 0.1)  # Minimum 0.1 seconds
            velocity = pattern.drift_magnitude / time_gap
            velocities.append(velocity)
            
        return velocities
    
    def identify_major_concept_shifts(self, threshold: float = 0.5, modality: Optional[Modality] = None) -> List[int]:
        """
        Identify sessions with major concept shifts using existing drift analysis.
        """
        drift_patterns = self.analyze_concept_drift_patterns(modality)
        
        major_shifts = []
        for pattern in drift_patterns:
            if pattern.drift_magnitude > threshold:
                major_shifts.append(pattern.session_to)
                
        return major_shifts
    
    def analyze_complete_concept_evolution(self, n_clusters: int = 5, modality: Optional[Modality] = None) -> ConceptEvolution:
        """
        Complete concept evolution analysis using existing S-BERT embeddings.
        """
        sequence_tensors, metadata = self._extract_sequence_data(modality)
        
        # Run all analyses
        clusters = self.analyze_concept_clusters(n_clusters, modality)
        drift_patterns = self.analyze_concept_drift_patterns(modality)
        velocities = self.analyze_concept_velocity(modality)
        major_shifts = self.identify_major_concept_shifts(modality=modality)
        
        # Simple concept persistence analysis
        concept_persistence = self._analyze_concept_persistence(clusters, len(sequence_tensors))
        
        return ConceptEvolution(
            total_sessions=len(sequence_tensors),
            concept_clusters=clusters,
            drift_patterns=drift_patterns,
            concept_velocity=velocities,
            major_shifts=major_shifts,
            concept_persistence=concept_persistence
        )
    
    def _extract_cluster_themes(self, session_indices: List[int], metadata: List[Dict]) -> List[str]:
        """Extract theme keywords from cluster sessions."""
        # Simple keyword extraction - combine text samples
        all_text = ""
        for idx in session_indices:
            if idx < len(metadata):
                all_text += " " + metadata[idx]['text_sample']
        
        # Basic keyword extraction (can be enhanced with NLP)
        import re
        words = re.findall(r'\b[a-zA-Z]{4,}\b', all_text.lower())
        word_freq = {}
        for word in words:
            if word not in ['that', 'this', 'with', 'from', 'they', 'have', 'been', 'will', 'would', 'could', 'should', 'session']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top 5 keywords
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, count in top_words]
    
    def _extract_shift_keywords(self, session_a: int, session_b: int, metadata: List[Dict]) -> List[str]:
        """Extract keywords representing concept shift between sessions."""
        if session_a >= len(metadata) or session_b >= len(metadata):
            return []
        
        text_a = metadata[session_a]['text_sample']
        text_b = metadata[session_b]['text_sample']
        
        # Simple difference keywords
        import re
        words_a = set(re.findall(r'\b[a-zA-Z]{4,}\b', text_a.lower()))
        words_b = set(re.findall(r'\b[a-zA-Z]{4,}\b', text_b.lower()))
        
        # Words that appear in B but not A (new concepts)
        new_words = words_b - words_a
        return list(new_words)[:3]  # Top 3 new concepts
    
    def _analyze_concept_persistence(self, clusters: List[ConceptCluster], total_sessions: int) -> Dict[str, float]:
        """Analyze how long different concepts persist."""
        persistence = {}
        
        for cluster in clusters:
            theme = cluster.theme_keywords[0] if cluster.theme_keywords else f"cluster_{cluster.cluster_id}"
            # Persistence = proportion of total sessions in this cluster
            persistence[theme] = len(cluster.session_indices) / total_sessions
            
        return persistence

# Utility functions for integration
def analyze_existing_store_concepts(store: UniversalMemoryStore, n_clusters: int = 5) -> ConceptEvolution:
    """
    Analyze concepts in an existing Universal Memory Store.
    
    This is the main entry point for concept analysis using existing S-BERT data.
    """
    analyzer = ConceptAnalyzer(store)
    return analyzer.analyze_complete_concept_evolution(n_clusters)

def get_concept_similarity_matrix(store: UniversalMemoryStore, modality: Optional[Modality] = None) -> np.ndarray:
    """
    Get session-to-session concept similarity matrix using existing S-BERT embeddings.
    """
    analyzer = ConceptAnalyzer(store)
    sequence_tensors, _ = analyzer._extract_sequence_data(modality)
    
    # Calculate cosine similarity matrix
    embeddings_np = sequence_tensors.cpu().numpy()
    similarity_matrix = cosine_similarity(embeddings_np)
    
    return similarity_matrix 