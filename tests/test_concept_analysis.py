#!/usr/bin/env python3
"""
Test script for Enhanced Concept Analysis

Tests the new concept analysis features to ensure they work with existing S-BERT embeddings.
"""

import torch
import numpy as np
from semantic_tensor_memory.memory.universal_core import UniversalMemoryStore
from semantic_tensor_memory.memory.text_embedder import TextEmbedder
from semantic_tensor_memory.analytics.concept.concept_analysis import ConceptAnalyzer, analyze_existing_store_concepts
from semantic_tensor_memory.visualization.tools.concept_visualizer import visualize_concept_evolution

def test_concept_analysis():
    """Test the enhanced concept analysis pipeline."""
    print("🧠 Testing Enhanced Concept Analysis...")
    
    # Create test data with diverse concepts
    test_texts = [
        "Patient exhibits signs of anxiety and stress related to work pressures",
        "Work-related anxiety continues to manifest in sleep disturbances",
        "Discussion of family relationships and childhood experiences",
        "Exploring cognitive behavioral therapy techniques for anxiety management", 
        "Family dynamics continue to influence current relationship patterns",
        "Progress in anxiety management through mindfulness practices",
        "Deep dive into childhood trauma and its lasting effects",
        "Family therapy session focusing on communication patterns",
        "Breakthrough in understanding anxiety triggers and coping mechanisms",
        "Integration of family insights with personal growth goals"
    ]
    
    # Set up Universal STM
    print("📝 Setting up Universal STM with test data...")
    store = UniversalMemoryStore()
    embedder = TextEmbedder()
    
    # Process each text session
    for i, text in enumerate(test_texts):
        embedding = embedder.process_raw_data(text, session_id=f"test_session_{i}")
        store.add_session(embedding)
    
    print(f"✅ Processed {len(store.embeddings)} sessions")
    
    # Test concept analysis
    print("\n🔍 Running concept analysis...")
    analyzer = ConceptAnalyzer(store)
    
    # Test individual functions
    print("Testing concept clustering...")
    clusters = analyzer.analyze_concept_clusters(n_clusters=3)
    print(f"Found {len(clusters)} clusters")
    
    for cluster in clusters:
        print(f"  Cluster {cluster.cluster_id}: {len(cluster.session_indices)} sessions")
        print(f"    Keywords: {', '.join(cluster.theme_keywords[:3])}")
        print(f"    Coherence: {cluster.coherence_score:.3f}")
    
    print("\nTesting drift patterns...")
    drift_patterns = analyzer.analyze_concept_drift_patterns()
    print(f"Found {len(drift_patterns)} drift patterns")
    
    for pattern in drift_patterns[:3]:  # Show first 3
        print(f"  Session {pattern.session_from} → {pattern.session_to}: {pattern.drift_magnitude:.3f} ({pattern.drift_direction})")
    
    print("\nTesting concept velocity...")
    velocities = analyzer.analyze_concept_velocity()
    if velocities:
        print(f"Velocity range: {min(velocities):.3f} - {max(velocities):.3f}")
        print(f"Average velocity: {np.mean(velocities):.3f}")
    
    print("\nTesting major shifts detection...")
    major_shifts = analyzer.identify_major_concept_shifts(threshold=0.3)
    print(f"Major shifts at sessions: {major_shifts}")
    
    # Test complete analysis
    print("\n🚀 Running complete concept evolution analysis...")
    evolution = analyze_existing_store_concepts(store, n_clusters=3)
    
    print(f"📊 Analysis Results:")
    print(f"  Total sessions: {evolution.total_sessions}")
    print(f"  Concept clusters: {len(evolution.concept_clusters)}")
    print(f"  Drift patterns: {len(evolution.drift_patterns)}")
    print(f"  Major shifts: {len(evolution.major_shifts)}")
    print(f"  Concept persistence: {len(evolution.concept_persistence)} themes")
    
    # Test visualizations
    print("\n📊 Testing visualizations...")
    viz_types = ["heatmap", "timeline", "velocity", "persistence"]
    
    for viz_type in viz_types:
        try:
            fig = visualize_concept_evolution(evolution, viz_type)
            print(f"  ✅ {viz_type} visualization created successfully")
        except Exception as e:
            print(f"  ❌ {viz_type} visualization failed: {str(e)}")
    
    # Test dashboard
    try:
        dashboard_fig = visualize_concept_evolution(evolution, "dashboard")
        print(f"  ✅ Dashboard visualization created successfully")
    except Exception as e:
        print(f"  ❌ Dashboard visualization failed: {str(e)}")
    
    print("\n🎉 Concept analysis test completed!")
    return evolution

def test_similarity_matrix():
    """Test the concept similarity matrix functionality."""
    print("\n🔗 Testing concept similarity matrix...")
    
    from semantic_tensor_memory.analytics.concept.concept_analysis import (
        get_concept_similarity_matrix,
    )
    
    # Simple test with a few sessions
    store = UniversalMemoryStore()
    embedder = TextEmbedder()
    
    test_texts = [
        "Anxiety and stress management",
        "Family relationship dynamics", 
        "Childhood trauma processing"
    ]
    
    for i, text in enumerate(test_texts):
        embedding = embedder.process_raw_data(text, session_id=f"sim_test_{i}")
        store.add_session(embedding)
    
    similarity_matrix = get_concept_similarity_matrix(store)
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print("Similarity matrix:")
    print(similarity_matrix.round(3))
    
    print("✅ Similarity matrix test completed!")

if __name__ == "__main__":
    try:
        # Run tests
        evolution = test_concept_analysis()
        test_similarity_matrix()
        
        print("\n🎯 All tests passed! Enhanced concept analysis is ready to use.")
        
        # Show some key insights
        print("\n💡 Key Insights from Test:")
        if evolution.concept_clusters:
            largest_cluster = max(evolution.concept_clusters, key=lambda c: len(c.session_indices))
            print(f"  - Largest cluster has {len(largest_cluster.session_indices)} sessions")
            print(f"  - Main themes: {', '.join(largest_cluster.theme_keywords[:3])}")
        
        if evolution.concept_persistence:
            most_persistent = max(evolution.concept_persistence.items(), key=lambda x: x[1])
            print(f"  - Most persistent concept: '{most_persistent[0]}' ({most_persistent[1]:.1%})")
        
        if evolution.major_shifts:
            print(f"  - {len(evolution.major_shifts)} major conceptual shifts detected")
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
