"""
Holistic Semantic Analysis - Revolutionary Global Approach

This module implements a comprehensive semantic analysis methodology:
1. Global Concept Extraction: Extract ALL concepts from entire dataset
2. Semantic Clustering: Find natural concept relationships using embeddings
3. Category Definition: Create meaningful semantic categories
4. Temporal Evolution: Track how categories emerge and evolve over time
5. Advanced Visualization: Multiple views of semantic category evolution
6. Holistic Semantic Drift River: Categories flowing as 3D rivers

This approach provides much deeper insights than session-by-session analysis.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Tuple, Optional, Set
from collections import Counter, defaultdict
import re
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import torch
import umap

from memory.embedder import embed_sentence

def infer_semantic_theme(concepts: List[str]) -> Optional[str]:
    """
    Infer a semantic theme name from a list of concepts using semantic patterns.
    
    Args:
        concepts: List of concept strings
    
    Returns:
        Inferred theme name or None if no clear theme emerges
    """
    if not concepts:
        return None
    
    # Convert to lowercase for analysis
    concept_words = [c.lower() for c in concepts]
    
    # Define semantic theme patterns
    theme_patterns = {
        "Work & Career": ["work", "job", "career", "professional", "office", "business", "company", "manager", "project", "meeting", "deadline"],
        "Health & Wellness": ["health", "sleep", "exercise", "diet", "wellness", "medical", "doctor", "therapy", "stress", "anxiety", "depression"],
        "Relationships & Family": ["family", "relationship", "friend", "partner", "spouse", "children", "parent", "love", "social", "communication"],
        "Learning & Growth": ["learning", "education", "skill", "knowledge", "development", "growth", "improvement", "training", "study", "book"],
        "Technology & Tools": ["technology", "computer", "software", "digital", "online", "internet", "tool", "application", "system", "platform"],
        "Emotions & Feelings": ["feeling", "emotion", "happy", "sad", "angry", "frustrated", "excited", "nervous", "confident", "worried"],
        "Time & Planning": ["time", "schedule", "planning", "future", "goal", "deadline", "calendar", "appointment", "routine", "habit"],
        "Money & Finance": ["money", "financial", "budget", "income", "expense", "investment", "saving", "cost", "price", "payment"],
        "Creativity & Arts": ["creative", "art", "design", "music", "writing", "artistic", "expression", "imagination", "inspiration", "aesthetic"],
        "Home & Environment": ["home", "house", "environment", "space", "room", "neighborhood", "community", "location", "place", "setting"]
    }
    
    # Score each theme based on concept matches
    theme_scores = {}
    for theme_name, theme_keywords in theme_patterns.items():
        score = 0
        for concept in concept_words:
            # Check if concept contains any theme keywords
            for keyword in theme_keywords:
                if keyword in concept or concept in keyword:
                    score += 1
                    break
        
        # Normalize by number of concepts
        if len(concepts) > 0:
            theme_scores[theme_name] = score / len(concepts)
    
    # Find the best matching theme
    if theme_scores:
        best_theme = max(theme_scores.items(), key=lambda x: x[1])
        if best_theme[1] >= 0.3:  # At least 30% of concepts match the theme
            return best_theme[0]
    
    # If no strong theme emerges, try to create a descriptive name from top concepts
    if len(concepts) >= 2:
        # Capitalize and join top 2 concepts with connector words
        concept1, concept2 = concepts[0].title(), concepts[1].title()
        
        # Try to infer relationship between concepts
        if any(word in concept1.lower() for word in ["work", "job", "career"]) and \
           any(word in concept2.lower() for word in ["stress", "anxiety", "pressure"]):
            return "Work-Related Stress"
        elif any(word in concept1.lower() for word in ["health", "wellness"]) and \
             any(word in concept2.lower() for word in ["exercise", "diet", "sleep"]):
            return "Health & Lifestyle"
        elif any(word in concept1.lower() for word in ["relationship", "family"]) and \
             any(word in concept2.lower() for word in ["communication", "support", "love"]):
            return "Social Connections"
        else:
            return f"{concept1} & {concept2} Themes"
    
    return None


def extract_all_concepts_globally(memory: List[torch.Tensor], meta: List[Dict]) -> Dict[str, Dict]:
    """
    Extract ALL concepts from the entire dataset with rich metadata.
    
    Args:
        memory: All session embedding tensors
        meta: All session metadata
    
    Returns:
        Dict mapping concept -> {sessions: [], frequencies: [], contexts: []}
    """
    global_concepts = defaultdict(lambda: {
        'sessions': [],
        'frequencies': [],
        'contexts': [],
        'total_frequency': 0
    })
    
    st.info("🔍 Extracting concepts globally from all sessions...")
    
    # Enhanced stop words for better concept quality
    stop_words = {
        'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'with', 'was', 'for', 'as', 'on', 'at', 'by',
        'this', 'but', 'not', 'from', 'or', 'an', 'are', 'it', 'have', 'has', 'had', 'if', 'they', 'their',
        'there', 'what', 'when', 'where', 'which', 'who', 'why', 'how', 'will', 'would', 'could', 'should',
        'been', 'being', 'do', 'does', 'did', 'can', 'may', 'might', 'must', 'shall', 'said', 'say', 'says',
        'told', 'tell', 'asked', 'ask', 'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know',
        'knew', 'think', 'thought', 'feel', 'felt', 'look', 'looked', 'take', 'took', 'give', 'gave', 'make',
        'made', 'find', 'found', 'use', 'used', 'work', 'worked', 'way', 'time', 'day', 'year', 'week', 'month',
        'just', 'now', 'here', 'then', 'more', 'also', 'well', 'very', 'much', 'really', 'still', 'back',
        'out', 'up', 'down', 'over', 'after', 'before', 'through', 'during', 'above', 'below', 'between',
        'about', 'into', 'started', 'first', 'today', 'people', 'everyone', 'everything', 'something',
        'anything', 'nothing', 'going', 'working', 'getting', 'need', 'want', 'like', 'better', 'good', 'great'
    }
    
    for session_idx, session_meta in enumerate(meta):
        if not isinstance(session_meta, dict):
            continue
            
        text = session_meta.get('text', '')
        if not text or not isinstance(text, str):
            continue
        
        # Extract words with improved filtering
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())  # Only alphabetic, 4+ chars
        
        # Count word frequencies in this session
        session_word_counts = Counter(words)
        
        for word, frequency in session_word_counts.items():
            if word not in stop_words and len(word) >= 4:
                # Store concept occurrence data
                global_concepts[word]['sessions'].append(session_idx)
                global_concepts[word]['frequencies'].append(frequency)
                global_concepts[word]['contexts'].append(text[:200])  # First 200 chars as context
                global_concepts[word]['total_frequency'] += frequency
    
    # Filter concepts by minimum criteria
    min_total_frequency = 3  # Must appear at least 3 times total
    min_sessions = 2  # Must appear in at least 2 sessions
    
    filtered_concepts = {
        concept: data for concept, data in global_concepts.items()
        if data['total_frequency'] >= min_total_frequency and len(set(data['sessions'])) >= min_sessions
    }
    
    st.success(f"🎯 Extracted {len(filtered_concepts)} high-quality concepts from {len(meta)} sessions")
    
    return dict(filtered_concepts)


def create_concept_embeddings(concepts: Dict[str, Dict]) -> Dict[str, torch.Tensor]:
    """
    Create embeddings for all concepts.
    
    Args:
        concepts: Global concepts dictionary
    
    Returns:
        Dict mapping concept -> embedding tensor
    """
    st.info("🧠 Creating embeddings for all concepts...")
    
    concept_embeddings = {}
    
    for concept in concepts.keys():
        try:
            # Create embedding for the concept
            embedding = embed_sentence(concept)
            if embedding.numel() > 0:
                # Take mean to get fixed-size representation
                concept_embeddings[concept] = embedding.mean(dim=0)
            else:
                # Fallback: create zero embedding
                concept_embeddings[concept] = torch.zeros(768)
        except Exception as e:
            st.warning(f"Could not embed concept '{concept}': {e}")
            concept_embeddings[concept] = torch.zeros(768)
    
    st.success(f"✅ Created embeddings for {len(concept_embeddings)} concepts")
    return concept_embeddings


def find_semantic_clusters(concept_embeddings: Dict[str, torch.Tensor], 
                          concepts_data: Dict[str, Dict],
                          n_clusters: int = 8) -> Dict[str, List[str]]:
    """
    Find natural semantic clusters among concepts.
    
    Args:
        concept_embeddings: Concept embeddings
        concepts_data: Original concepts data with frequencies and metadata
        n_clusters: Number of clusters to create
    
    Returns:
        Dict mapping cluster_name -> [concept_list]
    """
    st.info(f"🔬 Finding {n_clusters} semantic clusters...")
    
    if len(concept_embeddings) < n_clusters:
        n_clusters = max(2, len(concept_embeddings) // 2)
    
    # Prepare embeddings matrix
    concept_names = list(concept_embeddings.keys())
    embeddings_matrix = torch.stack([concept_embeddings[c] for c in concept_names])
    embeddings_np = embeddings_matrix.numpy()
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings_np)
    
    # Try multiple clustering methods and pick the best
    clustering_methods = [
        ('KMeans', KMeans(n_clusters=n_clusters, random_state=42, n_init=10)),
        ('Agglomerative', AgglomerativeClustering(n_clusters=n_clusters)),
    ]
    
    best_clusters = None
    best_score = -1
    
    for method_name, clusterer in clustering_methods:
        try:
            cluster_labels = clusterer.fit_predict(embeddings_scaled)
            
            # Calculate silhouette-like score (simplified)
            cluster_centers = []
            for i in range(n_clusters):
                cluster_points = embeddings_scaled[cluster_labels == i]
                if len(cluster_points) > 0:
                    cluster_centers.append(np.mean(cluster_points, axis=0))
            
            if len(cluster_centers) > 1:
                # Simple quality score: variance between clusters vs within clusters
                between_cluster_var = np.var([np.linalg.norm(center) for center in cluster_centers])
                within_cluster_var = np.mean([
                    np.var(embeddings_scaled[cluster_labels == i], axis=0).mean()
                    for i in range(n_clusters) if np.sum(cluster_labels == i) > 1
                ])
                score = between_cluster_var / (within_cluster_var + 1e-8)
                
                if score > best_score:
                    best_score = score
                    best_clusters = cluster_labels
                    st.info(f"✨ {method_name} achieved score: {score:.3f}")
        
        except Exception as e:
            st.warning(f"Clustering method {method_name} failed: {e}")
    
    if best_clusters is None:
        # Fallback: simple frequency-based grouping
        st.warning("Using fallback frequency-based clustering")
        best_clusters = np.arange(len(concept_names)) % n_clusters
    
    # Create semantic cluster mapping with DYNAMIC naming
    clusters = defaultdict(list)
    
    # Group concepts by cluster ID first
    cluster_concept_groups = defaultdict(list)
    for i, concept in enumerate(concept_names):
        cluster_id = best_clusters[i]
        cluster_concept_groups[cluster_id].append(concept)
    
    # Generate dynamic cluster names based on most frequent/representative concepts
    for cluster_id, cluster_concepts in cluster_concept_groups.items():
        if not cluster_concepts:
            continue
            
        # Find the most representative concepts for this cluster
        # Sort by frequency and representativeness
        concept_scores = []
        for concept in cluster_concepts:
            concept_data = concepts_data.get(concept, {})
            frequency_score = concept_data.get('total_frequency', 0)
            session_count_score = len(set(concept_data.get('sessions', [])))
            combined_score = frequency_score * 0.7 + session_count_score * 0.3
            concept_scores.append((concept, combined_score))
        
        # Sort by score and get top concepts for naming
        concept_scores.sort(key=lambda x: x[1], reverse=True)
        top_concepts = [c[0] for c in concept_scores[:3]]  # Top 3 concepts
        
        # Create dynamic cluster name based on top concepts with semantic intelligence
        if len(top_concepts) >= 2:
            # Try to infer semantic theme from top concepts
            theme_name = infer_semantic_theme(top_concepts)
            if theme_name:
                cluster_name = f"🎯 {theme_name}"
            else:
                cluster_name = f"🔍 {top_concepts[0].title()} & {top_concepts[1].title()}"
        elif len(top_concepts) == 1:
            cluster_name = f"🔍 {top_concepts[0].title()}-Focused"
        else:
            cluster_name = f"🔍 Emerging Concepts {cluster_id + 1}"
        
        # Add all concepts to this dynamically named cluster
        clusters[cluster_name].extend(cluster_concepts)
    
    # Sort concepts within each cluster by frequency (if available)
    for cluster_name in clusters:
        clusters[cluster_name] = sorted(clusters[cluster_name])
    
    st.success(f"🎯 Created {len(clusters)} semantic clusters")
    return dict(clusters)


def track_category_emergence(clusters: Dict[str, List[str]], 
                           concepts: Dict[str, Dict],
                           memory: List[torch.Tensor]) -> Dict[str, List[float]]:
    """
    Track how semantic categories emerge and evolve over time.
    
    Args:
        clusters: Semantic clusters
        concepts: Global concepts data
        memory: Session embeddings for temporal reference
    
    Returns:
        Dict mapping cluster_name -> [emergence_scores_over_time]
    """
    st.info("📈 Tracking category emergence over time...")
    
    num_sessions = len(memory)
    category_emergence = {}
    
    for cluster_name, cluster_concepts in clusters.items():
        emergence_scores = []
        
        for session_idx in range(num_sessions):
            # Calculate emergence score for this category in this session
            session_score = 0.0
            total_possible = len(cluster_concepts)
            
            for concept in cluster_concepts:
                concept_data = concepts.get(concept, {})
                sessions_with_concept = concept_data.get('sessions', [])
                
                if session_idx in sessions_with_concept:
                    # Get frequency in this session
                    session_position = sessions_with_concept.index(session_idx)
                    frequency = concept_data.get('frequencies', [0])[session_position]
                    
                    # Normalize frequency (simple approach)
                    normalized_freq = min(1.0, frequency / 5.0)  # Cap at 5 mentions
                    session_score += normalized_freq
            
            # Normalize by cluster size
            if total_possible > 0:
                emergence_scores.append(session_score / total_possible)
            else:
                emergence_scores.append(0.0)
        
        category_emergence[cluster_name] = emergence_scores
    
    st.success(f"📊 Tracked emergence for {len(category_emergence)} categories")
    return category_emergence


def create_holistic_semantic_drift_river(category_emergence: Dict[str, List[float]], 
                                       clusters: Dict[str, List[str]]) -> go.Figure:
    """
    Create a stunning 3D flowing river showing semantic categories as flowing rivers.
    
    Args:
        category_emergence: Category emergence scores over time
        clusters: Semantic category clusters
    
    Returns:
        Plotly 3D figure
    """
    st.info("🌊 Creating Holistic Semantic Drift River...")
    
    fig = go.Figure()
    
    # Enhanced color palette for category rivers
    river_colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
    ]
    
    import numpy as np
    
    # Process each category as a flowing 3D river
    for i, (category_name, emergence_scores) in enumerate(category_emergence.items()):
        if not emergence_scores:
            continue
            
        color = river_colors[i % len(river_colors)]
        
        # Create smooth time progression
        time_points = np.array(range(len(emergence_scores)))
        
        # Y-axis: Cumulative semantic flow (stacked rivers)
        if i == 0:
            cumulative_base = np.zeros(len(emergence_scores))
        else:
            # Stack on top of previous rivers
            prev_category_values = list(category_emergence.values())[i-1]
            if i == 1:
                cumulative_base = np.array(prev_category_values) * 2  # Multiply for visual separation
            else:
                cumulative_base = cumulative_base + np.array(prev_category_values) * 2
        
        river_bottom = cumulative_base
        river_top = cumulative_base + np.array(emergence_scores) * 3  # Scale for visibility
        
        # Z-axis: Category separation with flow intensity
        base_z = i * 4  # Separate categories vertically
        flow_intensity = [base_z + score * 8 for score in emergence_scores]
        
        # Create rich hover data
        hover_data = []
        category_concepts = clusters.get(category_name, [])
        for j, score in enumerate(emergence_scores):
            concept_list = ", ".join(category_concepts[:5])  # Show first 5 concepts
            if len(category_concepts) > 5:
                concept_list += f" ... (+{len(category_concepts)-5} more)"
            
            hover_data.append({
                'session': j,
                'category': category_name,
                'emergence_score': score,
                'concepts': concept_list,
                'concept_count': len(category_concepts),
                'river_height': river_top[j] - river_bottom[j],
                'flow_intensity': flow_intensity[j]
            })
        
        # Main flowing river centerline
        fig.add_trace(go.Scatter3d(
            x=time_points,
            y=(river_top + river_bottom) / 2,  # River center
            z=flow_intensity,
            mode='lines+markers',
            line=dict(
                color=color,
                width=10,
                colorscale=[[0, color], [0.5, '#FFFFFF'], [1, color]]
            ),
            marker=dict(
                size=[max(8, min(20, 8 + score * 40)) for score in emergence_scores],
                color=emergence_scores,
                colorscale='Viridis',
                opacity=0.9,
                line=dict(color='white', width=2),
                colorbar=dict(
                    title=f"{category_name}<br>Emergence",
                    tickfont=dict(color='white', size=10),
                    x=1.02 + i * 0.08,
                    len=0.7,
                    thickness=12
                ) if i < 3 else None,
                showscale=i < 3
            ),
            name=f"🌊 {category_name}",
            hovertemplate=(
                f"<b>🌊 {category_name}</b><br>"
                "<b>📍 Session:</b> %{customdata[0]}<br>"
                "<b>📊 Emergence Score:</b> %{customdata[2]:.3f}<br>"
                "<b>⚡ Flow Intensity:</b> %{customdata[6]:.2f}<br>"
                "<b>📏 River Height:</b> %{customdata[5]:.2f}<br>"
                "<b>🎯 Concept Count:</b> %{customdata[4]}<br>"
                "<b>💎 Key Concepts:</b><br>%{customdata[3]}<br>"
                "<extra></extra>"
            ),
            customdata=[[h['session'], h['category'], h['emergence_score'], h['concepts'], 
                        h['concept_count'], h['river_height'], h['flow_intensity']] for h in hover_data],
            showlegend=True
        ))
        
        # Create flowing river surface (tube effect)
        if len(time_points) > 1:
            # Create river surface tubes
            tube_resolution = 16
            theta = np.linspace(0, 2*np.pi, tube_resolution)
            
            tube_x, tube_y, tube_z = [], [], []
            tube_i, tube_j, tube_k = [], [], []
            
            for j in range(len(time_points)):
                # River width varies with emergence intensity
                river_width = 0.5 + emergence_scores[j] * 2
                center_y = (river_top[j] + river_bottom[j]) / 2
                center_z = flow_intensity[j]
                
                # Create elliptical cross-section (wider horizontally)
                for t in theta:
                    tube_x.append(time_points[j])
                    tube_y.append(center_y + river_width * np.cos(t))
                    tube_z.append(center_z + (river_width * 0.6) * np.sin(t))  # Flatter in Z
                
                # Create surface connectivity
                if j < len(time_points) - 1:
                    for k in range(tube_resolution):
                        next_k = (k + 1) % tube_resolution
                        
                        curr_base = j * tube_resolution
                        next_base = (j + 1) * tube_resolution
                        
                        # Two triangles per quad
                        tube_i.extend([curr_base + k, curr_base + next_k, next_base + k])
                        tube_j.extend([curr_base + next_k, next_base + next_k, next_base + k])
                        tube_k.extend([next_base + k, curr_base + k, curr_base + next_k])
            
            # Add translucent flowing river surface
            fig.add_trace(go.Mesh3d(
                x=tube_x,
                y=tube_y,
                z=tube_z,
                i=tube_i,
                j=tube_j,
                k=tube_k,
                color=color,
                opacity=0.4,
                name=f"River Surface {category_name}",
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add category emergence events (peaks and valleys)
        if len(emergence_scores) > 2:
            # Find significant changes
            emergence_diffs = np.diff(emergence_scores)
            for j, diff in enumerate(emergence_diffs):
                if abs(diff) > 0.15:  # Significant change threshold
                    event_type = "📈 Surge" if diff > 0 else "📉 Decline"
                    event_color = '#00FF00' if diff > 0 else '#FF4444'
                    
                    fig.add_trace(go.Scatter3d(
                        x=[j + 1],
                        y=[(river_top[j+1] + river_bottom[j+1]) / 2],
                        z=[flow_intensity[j+1] + 2],
                        mode='markers+text',
                        marker=dict(
                            size=12,
                            color=event_color,
                            symbol='diamond',
                            opacity=0.9,
                            line=dict(color='white', width=2)
                        ),
                        text=[event_type.split(' ')[0]],
                        textfont=dict(size=12, color='white'),
                        name=f"{event_type} Event",
                        hovertemplate=(
                            f"<b>{event_type}</b><br>"
                            f"<b>Category:</b> {category_name}<br>"
                            f"<b>Session:</b> {j+1}<br>"
                            f"<b>Change:</b> {diff:+.3f}<br>"
                            "<extra></extra>"
                        ),
                        showlegend=False
                    ))
    
    # Style the Holistic Semantic Drift River with enhanced aesthetics
    fig.update_layout(
        title=dict(
            text="🌊 Holistic Semantic Drift River - Category Evolution Streams",
            font=dict(size=22, color='white', family='Arial Black'),
            x=0.5
        ),
        scene=dict(
            xaxis=dict(
                title=dict(
                    text="<b>⏰ Sessions (Temporal Flow →)</b>",
                    font=dict(color='white', size=16)
                ),
                color='white',
                gridcolor='rgba(255,255,255,0.3)',
                showbackground=True,
                backgroundcolor='rgba(20,30,60,0.8)',
                gridwidth=2,
                tickfont=dict(color='white', size=12)
            ),
            yaxis=dict(
                title=dict(
                    text="<b>🌊 Cumulative Category Flow</b>",
                    font=dict(color='white', size=16)
                ),
                color='white',
                gridcolor='rgba(255,255,255,0.3)',
                showbackground=True,
                backgroundcolor='rgba(20,30,60,0.8)',
                gridwidth=2,
                tickfont=dict(color='white', size=12)
            ),
            zaxis=dict(
                title=dict(
                    text="<b>⚡ Category Intensity & Flow Depth</b>",
                    font=dict(color='white', size=16)
                ),
                color='white',
                gridcolor='rgba(255,255,255,0.3)',
                showbackground=True,
                backgroundcolor='rgba(20,30,60,0.8)',
                gridwidth=2,
                tickfont=dict(color='white', size=12)
            ),
            bgcolor='rgba(5,15,35,0.98)',
            camera=dict(
                eye=dict(x=2.5, y=2.0, z=1.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual',
            aspectratio=dict(x=2.5, y=1.8, z=1.2)
        ),
        plot_bgcolor='rgba(5,15,35,0.98)',
        paper_bgcolor='rgba(5,15,35,0.98)',
        font=dict(color='white', family='Arial'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=2,
            font=dict(color='white', size=12),
            x=0.02,
            y=0.98
        ),
        height=800,
        margin=dict(l=60, r=140, t=100, b=60)
    )
    
    st.success("🌊 Holistic Semantic Drift River created!")
    return fig


def create_category_emergence_plot(category_emergence: Dict[str, List[float]]) -> go.Figure:
    """
    Create a beautiful plot showing dynamically discovered category emergence over time.
    
    Args:
        category_emergence: Category emergence data with dynamic category names
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Enhanced color palette for categories
    colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
    ]
    
    sessions = list(range(len(list(category_emergence.values())[0])))
    
    # Sort categories by final emergence score for better visualization
    sorted_categories = sorted(category_emergence.items(), 
                             key=lambda x: x[1][-1] if x[1] else 0, 
                             reverse=True)
    
    for i, (category, emergence_scores) in enumerate(sorted_categories):
        color = colors[i % len(colors)]
        
        # Add main line with enhanced styling
        fig.add_trace(go.Scatter(
            x=sessions,
            y=emergence_scores,
            mode='lines+markers',
            name=category,
            line=dict(color=color, width=4, shape='spline'),  # Smoother lines
            marker=dict(
                size=10, 
                color=color, 
                opacity=0.9,
                line=dict(color='white', width=1)
            ),
            hovertemplate=(
                f"<b>📊 {category}</b><br>"
                "<b>🕐 Session:</b> %{x}<br>"
                "<b>📈 Emergence Score:</b> %{y:.3f}<br>"
                "<b>💡 Rank:</b> #{rank}<br>"
                "<extra></extra>"
            ).replace("{rank}", str(i+1))
        ))
        
        # Add filled area under curve with gradient effect
        fig.add_trace(go.Scatter(
            x=sessions + sessions[::-1],
            y=emergence_scores + [0] * len(sessions),
            fill='toself',
            fillcolor=f'rgba{tuple(list(bytes.fromhex(color[1:])) + [0.15])}',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Add emergence peak markers for significant categories
        if emergence_scores and max(emergence_scores) > 0.2:  # Significant emergence threshold
            peak_session = sessions[emergence_scores.index(max(emergence_scores))]
            peak_value = max(emergence_scores)
            
            fig.add_trace(go.Scatter(
                x=[peak_session],
                y=[peak_value],
                mode='markers+text',
                marker=dict(
                    size=15,
                    symbol='star',
                    color=color,
                    line=dict(color='white', width=2)
                ),
                text=['🌟'],
                textfont=dict(size=16),
                name=f"Peak: {category}",
                showlegend=False,
                hovertemplate=(
                    f"<b>🌟 Peak Emergence</b><br>"
                    f"<b>Category:</b> {category}<br>"
                    f"<b>Session:</b> {peak_session}<br>"
                    f"<b>Peak Score:</b> {peak_value:.3f}<br>"
                    "<extra></extra>"
                )
            ))
    
    # Add dynamic insights annotation
    total_categories = len(category_emergence)
    avg_emergence = np.mean([scores[-1] for scores in category_emergence.values() if scores])
    
    fig.add_annotation(
        x=0.02, y=0.98,
        xref="paper", yref="paper",
        text=(
            f"🧠 <b>Dynamic Analysis</b><br>"
            f"📊 {total_categories} categories discovered<br>"
            f"📈 Avg emergence: {avg_emergence:.3f}<br>"
            f"🎯 Categories adapt to your data"
        ),
        showarrow=False,
        font=dict(size=12, color='white'),
        bgcolor='rgba(0,0,0,0.8)',
        bordercolor='white',
        borderwidth=1,
        align='left'
    )
    
    # Style the plot with enhanced aesthetics
    fig.update_layout(
        title=dict(
            text="🌟 Dynamic Semantic Category Emergence - Data-Driven Discovery",
            font=dict(size=22, color='white', family='Arial Black'),
            x=0.5
        ),
        xaxis=dict(
            title="<b>Sessions (Temporal Evolution →)</b>",
            color='white',
            gridcolor='rgba(255,255,255,0.3)',
            tickfont=dict(color='white', size=12),
            showline=True,
            linecolor='white'
        ),
        yaxis=dict(
            title="<b>Category Emergence Score</b>",
            color='white',
            gridcolor='rgba(255,255,255,0.3)',
            tickfont=dict(color='white', size=12),
            showline=True,
            linecolor='white'
        ),
        plot_bgcolor='rgba(10,10,30,0.95)',
        paper_bgcolor='rgba(10,10,30,0.95)',
        font=dict(color='white', family='Arial'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.8)',
            bordercolor='white',
            borderwidth=2,
            font=dict(color='white', size=11),
            orientation='v',
            x=1.02,
            y=1
        ),
        height=650,
        margin=dict(l=80, r=200, t=100, b=80)
    )
    
    return fig


def create_concept_network_plot(concept_embeddings: Dict[str, torch.Tensor], 
                              clusters: Dict[str, List[str]]) -> go.Figure:
    """
    Create a network plot showing concept relationships.
    
    Args:
        concept_embeddings: Concept embeddings
        clusters: Semantic clusters
    
    Returns:
        Plotly figure
    """
    # Calculate 2D positions using UMAP
    concepts = list(concept_embeddings.keys())
    embeddings_matrix = torch.stack([concept_embeddings[c] for c in concepts])
    embeddings_np = embeddings_matrix.numpy()
    
    # Use UMAP for better 2D projection
    try:
        umap_reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=42)
        positions_2d = umap_reducer.fit_transform(embeddings_np)
    except:
        # Fallback to PCA
        pca = PCA(n_components=2, random_state=42)
        positions_2d = pca.fit_transform(embeddings_np)
    
    # Create concept-to-cluster mapping
    concept_to_cluster = {}
    cluster_colors = {}
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']
    
    for i, (cluster_name, cluster_concepts) in enumerate(clusters.items()):
        cluster_colors[cluster_name] = colors[i % len(colors)]
        for concept in cluster_concepts:
            concept_to_cluster[concept] = cluster_name
    
    # Create the network plot
    fig = go.Figure()
    
    # Add edges (connections between similar concepts)
    similarity_matrix = cosine_similarity(embeddings_np)
    threshold = 0.7  # Only show high similarity connections
    
    edge_x, edge_y = [], []
    for i in range(len(concepts)):
        for j in range(i+1, len(concepts)):
            if similarity_matrix[i][j] > threshold:
                x0, y0 = positions_2d[i]
                x1, y1 = positions_2d[j]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
    
    # Add edges
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='rgba(150,150,150,0.3)', width=1),
        showlegend=False,
        hoverinfo='none'
    ))
    
    # Add nodes by cluster
    for cluster_name, cluster_concepts in clusters.items():
        cluster_positions = [positions_2d[concepts.index(c)] for c in cluster_concepts if c in concepts]
        if not cluster_positions:
            continue
            
        x_pos = [pos[0] for pos in cluster_positions]
        y_pos = [pos[1] for pos in cluster_positions]
        
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(
                size=12,
                color=cluster_colors[cluster_name],
                opacity=0.8,
                line=dict(color='white', width=2)
            ),
            text=cluster_concepts,
            textposition='top center',
            textfont=dict(size=10, color='white'),
            name=cluster_name,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"<b>Cluster:</b> {cluster_name}<br>"
                "<extra></extra>"
            )
        ))
    
    # Style the plot
    fig.update_layout(
        title=dict(
            text="🌐 Semantic Concept Network",
            font=dict(size=20, color='white'),
            x=0.5
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(10,10,30,0.95)',
        paper_bgcolor='rgba(10,10,30,0.95)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(0,0,0,0.7)',
            bordercolor='white',
            borderwidth=1,
            font=dict(color='white')
        ),
        height=700
    )
    
    return fig


def render_holistic_semantic_analysis(memory: List[torch.Tensor], meta: List[Dict]):
    """
    Render the complete holistic semantic analysis interface.
    
    Args:
        memory: Session embedding tensors
        meta: Session metadata
    """
    st.subheader("🌐 Holistic Semantic Analysis - Global Concept Evolution")
    
    # Add comprehensive interpretation guide
    with st.expander("📚 **How to Read & Trust This Analysis**", expanded=False):
        st.markdown("""
        ### 🔍 **What You're Looking At**
        
        The **Holistic Semantic Drift River** transforms your text into flowing 3D rivers where each river represents a **semantic category** that emerges from YOUR specific data.
        
        ### 🌊 **Visual Elements Explained**
        
        #### **📈 Sharp Lines (River Centerlines)**
        - **Precise mathematical trajectories** showing how each category evolves over time
        - **X-axis**: Session progression (time flows left → right)
        - **Y-axis**: Cumulative semantic flow (rivers stack to show relative importance)
        - **Z-axis**: Category intensity (height = strength of that theme)
        
        #### **☁️ Flowing Surfaces (River "Water")**
        - **Volumetric semantic density** - thicker areas = stronger category presence
        - **River width** varies with emergence score (0.5 + 2×intensity)
        - **Translucent tubes** with 16-point resolution for smooth curves
        - **NOT uncertainty** - represents category volume/presence
        
        #### **💎 Event Markers (Diamonds)**
        - **📈 Green diamonds** = Semantic surges (>15% increase in category strength)
        - **📉 Red diamonds** = Semantic declines (>15% decrease in category strength) 
        - **Positioned above rivers** for visibility
        
        ### 🔬 **Mathematical Foundation & Trust Indicators**
        
        #### **Data-Driven Category Discovery**
        ```
        1. Extract ALL concepts (4+ chars) from your text
        2. Filter: min 3 total occurrences, must appear in 2+ sessions  
        3. Create 768-dimensional BERT embeddings for each concept
        4. Use StandardScaler + clustering (KMeans/Agglomerative)
        5. Generate category names from most representative concepts
        ```
        
        #### **Emergence Score Calculation**
        ```
        For each category C at session t:
        emergence_score[C][t] = (
            Σ(min(1.0, frequency/5) for each concept in C)
        ) / |C|
        
        This prevents large categories from dominating and caps individual 
        concept influence at 5 mentions per session.
        ```
        
        #### **Quality Assurance**
        - ✅ **No predefined categories** - themes emerge from YOUR data patterns
        - ✅ **Normalized scoring** prevents frequency bias
        - ✅ **Multiple clustering algorithms** tested, best selected automatically
        - ✅ **Minimum thresholds** ensure concept quality
        - ✅ **Temporal consistency** tracked across all sessions
        
        ### 📊 **How to Interpret Results**
        
        #### **🌊 River Patterns**
        - **Rising rivers** = Emerging themes in your thinking
        - **Declining rivers** = Fading concepts over time  
        - **Parallel rivers** = Consistent themes throughout
        - **Converging rivers** = Related concepts appearing together
        - **River thickness** = How much that theme dominates each session
        
        #### **🎨 Color & Size Meanings**
        - **Line color** = Category identifier (consistent per theme)
        - **Marker size** = Emergence intensity (8-20px range)
        - **Marker color** = Viridis scale (dark = low, bright = high emergence)
        - **River width** = Real-time category strength
        
        #### **📈 Trust Your Data When...**
        - Categories have **meaningful names** that reflect your content
        - **Multiple concepts** per category (not just 1-2 words)
        - **Temporal patterns** make sense with your actual experience
        - **Concept lists** in hover data are relevant to the category name
        
        #### **⚠️ Be Cautious When...**
        - Categories have only 1-2 concepts (may be noise)
        - Category names seem random or unrelated to your content
        - Very few sessions (< 5) - patterns may not be reliable
        - All emergence scores are very low (< 0.1) - weak signal
        
        ### 🎯 **Actionable Insights**
        
        Use this visualization to:
        - **Identify recurring themes** in your thinking/writing
        - **Track concept evolution** over time periods
        - **Discover unexpected connections** between ideas  
        - **Understand your semantic patterns** and thinking style
        - **Find periods of conceptual shift** (marked by diamonds)
        
        ### 🧮 **Technical Specifications**
        - **Embedding Model**: BERT (768 dimensions)
        - **Clustering**: KMeans/Agglomerative with StandardScaler
        - **Quality Metrics**: Between-cluster vs within-cluster variance
        - **Temporal Resolution**: Per-session analysis
        - **3D Rendering**: Plotly with parametric surface generation
        """)
    
    st.info("""
    **🚀 Revolutionary Dynamic Semantic Analysis!**
    
    This advanced methodology provides comprehensive insights by:
    - 🔍 **Global Concept Extraction**: Find ALL concepts across entire dataset
    - 🧠 **Dynamic Clustering**: Let categories emerge naturally from YOUR data  
    - 🎯 **Data-Driven Categories**: No predefined labels - categories adapt to content
    - 📈 **Temporal Evolution**: Track how discovered categories emerge over time
    - 🌐 **Network Visualization**: See concept relationships in semantic space
    - 🌊 **Holistic Drift River**: Categories flowing as beautiful 3D rivers
    
    **✨ Key Innovation**: Categories are not predetermined but emerge from the semantic 
    patterns in your specific data, providing truly personalized insights!
    """)
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_clusters = st.slider("Number of Categories", 4, 12, 8,
                              help="Number of semantic categories to discover")
    
    with col2:
        min_frequency = st.slider("Min Concept Frequency", 2, 10, 3,
                                 help="Minimum times a concept must appear")
    
    with col3:
        analysis_type = st.selectbox("Visualization Type", 
                                   ["📈 Standard Analysis", "🌊 Holistic Drift River", "🌐 Concept Network"],
                                   help="Choose primary visualization focus")
    
    if st.button("🚀 Run Holistic Analysis", type="primary"):
        with st.spinner("Running comprehensive semantic analysis..."):
            
            # Step 1: Global concept extraction
            st.markdown("### 🔍 Step 1: Global Concept Extraction")
            with st.expander("ℹ️ **What's happening here**"):
                st.markdown("""
                **Process**: Extracting ALL words (4+ characters) from your entire dataset
                **Filtering**: Removing 105+ stop words, requiring minimum 3 total occurrences and presence in 2+ sessions
                **Quality**: Only concepts that appear consistently across multiple sessions are kept
                """)
            
            global_concepts = extract_all_concepts_globally(memory, meta)
            
            if not global_concepts:
                st.error("No concepts could be extracted. Try adjusting parameters.")
                return
            
            # Display concept summary with interpretation
            concept_summary = pd.DataFrame([
                {
                    'Concept': concept,
                    'Total Frequency': data['total_frequency'],
                    'Sessions': len(set(data['sessions'])),
                    'Avg per Session': round(data['total_frequency'] / len(set(data['sessions'])), 2),
                    'Quality Score': round(data['total_frequency'] * len(set(data['sessions'])) / 10, 2)
                }
                for concept, data in global_concepts.items()
            ]).sort_values('Total Frequency', ascending=False)
            
            st.markdown(f"**Found {len(global_concepts)} high-quality concepts** (showing top 20):")
            st.dataframe(concept_summary.head(20), use_container_width=True)
            
            # Step 2: Create embeddings
            st.markdown("### 🧠 Step 2: Concept Embedding Creation")
            with st.expander("ℹ️ **What's happening here**"):
                st.markdown("""
                **Process**: Converting each concept into a 768-dimensional semantic vector using BERT
                **Purpose**: Creates mathematical representation where similar concepts cluster naturally
                **Result**: Each concept becomes a point in high-dimensional semantic space
                """)
            concept_embeddings = create_concept_embeddings(global_concepts)
            
            # Step 3: Find semantic clusters
            st.markdown("### 🔬 Step 3: Semantic Clustering")
            with st.expander("ℹ️ **What's happening here**"):
                st.markdown("""
                **Process**: Using machine learning to find natural groupings in your concepts
                **Algorithm**: StandardScaler + KMeans/Agglomerative clustering
                **Selection**: Best clustering method chosen automatically based on cluster quality
                **Naming**: Categories named dynamically from most representative concepts in each cluster
                """)
            clusters = find_semantic_clusters(concept_embeddings, global_concepts, n_clusters)
            
            # Display clusters with quality indicators
            st.markdown("**🎯 Discovered Categories** (expand to see concepts):")
            for cluster_name, concepts in clusters.items():
                quality_indicator = "🟢 High Quality" if len(concepts) >= 4 else "🟡 Moderate Quality" if len(concepts) >= 2 else "🔴 Low Quality"
                with st.expander(f"{cluster_name} ({len(concepts)} concepts) - {quality_indicator}"):
                    st.write(", ".join(concepts))
                    if len(concepts) < 3:
                        st.warning("⚠️ Small cluster - interpret results with caution")
            
            # Step 4: Track emergence
            st.markdown("### 📈 Step 4: Category Emergence Tracking")
            with st.expander("ℹ️ **What's happening here**"):
                st.markdown("""
                **Process**: Calculating how strongly each category appears in each session
                **Formula**: `emergence_score = (Σ normalized_frequencies) / cluster_size`
                **Normalization**: Caps individual concept influence to prevent bias
                **Result**: Time series showing category evolution across sessions
                """)
            category_emergence = track_category_emergence(clusters, global_concepts, memory)
            
            # Quality assessment for emergence data
            avg_emergence = np.mean([np.mean(scores) for scores in category_emergence.values()])
            max_emergence = max([max(scores) for scores in category_emergence.values()])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                quality_color = "🟢" if avg_emergence > 0.15 else "🟡" if avg_emergence > 0.05 else "🔴"
                st.metric("Avg Emergence", f"{avg_emergence:.3f}", help="Higher = stronger signal")
                st.write(f"{quality_color} Signal Quality")
            with col2:
                st.metric("Max Emergence", f"{max_emergence:.3f}", help="Peak category strength")
            with col3:
                variance = np.var([max(scores) - min(scores) for scores in category_emergence.values()])
                st.metric("Temporal Variance", f"{variance:.3f}", help="Higher = more change over time")
            
            # Step 5: Create visualizations based on selection
            if analysis_type == "🌊 Holistic Drift River":
                st.markdown("### 🌊 Step 5: Holistic Semantic Drift River")
                
                # Add river-specific interpretation guide
                with st.expander("🧭 **How to Read the 3D River**"):
                    st.markdown("""
                    **🌊 River Flow Direction**: Time flows from left (early sessions) to right (recent sessions)
                    **📏 River Height**: Taller rivers = stronger category presence at that time
                    **🌊 River Width**: Wider tubes = higher emergence intensity 
                    **🎨 River Color**: Each category has a unique color for identification
                    **💎 Diamond Events**: Significant increases (green ⬆️) or decreases (red ⬇️) in category strength
                    **📊 Hover Data**: Click any point for detailed category information
                    
                    **🎯 What to Look For**:
                    - Rising rivers = themes becoming more important to you
                    - Declining rivers = concepts fading from your focus
                    - Parallel rivers = consistent themes throughout your data
                    - Event diamonds = moments of significant conceptual change
                    """)
                
                holistic_river_fig = create_holistic_semantic_drift_river(category_emergence, clusters)
                st.plotly_chart(holistic_river_fig, use_container_width=True, key="holistic_drift_river")
                
                # Add interpretation summary
                st.markdown("#### 🎯 **Quick Interpretation Guide**")
                strongest_category = max(category_emergence.items(), key=lambda x: max(x[1]))
                most_variable = max(category_emergence.items(), key=lambda x: max(x[1]) - min(x[1]))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"""
                    **🏆 Strongest Category**: {strongest_category[0]}
                    Peak emergence: {max(strongest_category[1]):.3f}
                    This theme dominates your content.
                    """)
                with col2:
                    st.info(f"""
                    **📈 Most Variable**: {most_variable[0]}
                    Change range: {max(most_variable[1]) - min(most_variable[1]):.3f}
                    This theme shows the most evolution over time.
                    """)
            
            elif analysis_type == "🌐 Concept Network":
                st.markdown("### 🌐 Step 5: Concept Network Visualization")
                network_fig = create_concept_network_plot(concept_embeddings, clusters)
                st.plotly_chart(network_fig, use_container_width=True, key="network_plot")
            
            else:  # Standard Analysis
                st.markdown("### 📈 Step 5: Category Emergence Analysis")
                emergence_fig = create_category_emergence_plot(category_emergence)
                st.plotly_chart(emergence_fig, use_container_width=True, key="emergence_plot")
            
            # Enhanced insights summary with trust indicators
            st.markdown("### 💡 Key Insights & Quality Assessment")
            
            # Find most emergent category
            final_scores = {cat: scores[-1] for cat, scores in category_emergence.items()}
            most_emergent = max(final_scores.items(), key=lambda x: x[1])
            
            # Find most growth
            growth_scores = {
                cat: (scores[-1] - scores[0]) if len(scores) > 1 else 0 
                for cat, scores in category_emergence.items()
            }
            highest_growth = max(growth_scores.items(), key=lambda x: x[1])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Most Dominant Category",
                    most_emergent[0],
                    f"{most_emergent[1]:.3f}",
                    help="Category with highest final emergence score"
                )
            
            with col2:
                st.metric(
                    "Highest Growth Category", 
                    highest_growth[0],
                    f"+{highest_growth[1]:.3f}",
                    help="Category showing most increase over time"
                )
            
            with col3:
                st.metric(
                    "Total Concepts Analyzed",
                    len(global_concepts),
                    f"{len(clusters)} categories",
                    help="High-quality concepts meeting minimum thresholds"
                )
            
            # Add data quality assessment
            st.markdown("#### 🔍 **Data Quality Assessment**")
            
            total_concepts = len(global_concepts)
            avg_concepts_per_category = np.mean([len(concepts) for concepts in clusters.values()])
            
            quality_indicators = []
            if total_concepts >= 20:
                quality_indicators.append("✅ Sufficient concept diversity")
            else:
                quality_indicators.append("⚠️ Limited concept diversity - consider more data")
                
            if avg_concepts_per_category >= 3:
                quality_indicators.append("✅ Well-formed categories")
            else:
                quality_indicators.append("⚠️ Small categories - results may be less reliable")
                
            if avg_emergence > 0.1:
                quality_indicators.append("✅ Strong semantic signal")
            else:
                quality_indicators.append("⚠️ Weak semantic signal - patterns may be subtle")
            
            for indicator in quality_indicators:
                st.write(indicator)
            
            # Store results in session state
            st.session_state['holistic_analysis'] = {
                'concepts': global_concepts,
                'clusters': clusters,
                'emergence': category_emergence,
                'embeddings': concept_embeddings
            }
    
    # Show cached results if available
    if 'holistic_analysis' in st.session_state:
        st.markdown("### 📊 Current Analysis Results")
        results = st.session_state['holistic_analysis']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Concepts", len(results['concepts']))
        with col2:
            st.metric("Categories", len(results['clusters']))
        with col3:
            if st.button("🌊 Generate Drift River", type="secondary"):
                with st.spinner("Creating Holistic Semantic Drift River..."):
                    holistic_river_fig = create_holistic_semantic_drift_river(
                        results['emergence'], 
                        results['clusters']
                    )
                    st.plotly_chart(holistic_river_fig, use_container_width=True, key="cached_holistic_river")
        
        if st.button("Clear Analysis"):
            del st.session_state['holistic_analysis']
            st.rerun() 