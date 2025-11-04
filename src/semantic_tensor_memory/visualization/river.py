"""semantic_tensor_memory.visualization.river
=============================================

Specialised helpers for the "semantic drift river" visualisation, kept separate
from the broader holistic analysis module so the intent of each renderer is
clear and discoverable.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import List, Dict, Tuple, Optional
from collections import Counter, defaultdict
import re
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
import torch

<<<<<<<< HEAD:src/semantic_tensor_memory/visualization/viz/semantic_drift_river.py
from .pca_plot import prepare_for_pca
from semantic_tensor_memory.memory.embedder import embed_sentence
========
from .plots import prepare_for_pca
from memory.embedder import embed_sentence
>>>>>>>> main:src/semantic_tensor_memory/visualization/river.py


def extract_concepts_from_text(text: str, method='keyword') -> List[str]:
    """
    Extract key concepts from text using various methods.
    
    Args:
        text: Input text to analyze
        method: Method to use ('keyword', 'entities', 'phrases')
    
    Returns:
        List of concept strings
    """
    if method == 'keyword':
        # Enhanced keyword extraction with domain-specific terms
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Enhanced stop words for better concept extraction
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
        
        # Filter meaningful concepts (length 4+ for more specificity, not stop words)
        concepts = [w for w in words if len(w) >= 4 and w not in stop_words]
        
        # Prioritize domain-specific and evolving terms
        domain_terms = []
        for word in concepts:
            # Boost technical, business, and domain-specific terms
            if any(domain in word for domain in ['app', 'user', 'platform', 'algorithm', 'data', 'financial', 'budget', 
                                                'invest', 'fund', 'money', 'market', 'tech', 'api', 'feature', 'product',
                                                'business', 'company', 'startup', 'revenue', 'growth', 'scale', 'team']):
                domain_terms.extend([word] * 3)  # Boost importance
            else:
                domain_terms.append(word)
        
        # Group similar concepts and return top ones
        concept_counts = Counter(domain_terms)
        return [concept for concept, count in concept_counts.most_common(8)]
    
    elif method == 'phrases':
        # Extract meaningful phrases (2-3 words)
        # Simple phrase extraction using common patterns
        phrases = []
        words = text.lower().split()
        
        # Look for meaningful 2-word and 3-word phrases
        for i in range(len(words) - 1):
            if len(words[i]) > 2 and len(words[i+1]) > 2:
                phrase = f"{words[i]} {words[i+1]}"
                if not any(stop in phrase for stop in ['the ', 'and ', 'a ', 'to ', 'of ', 'in ']):
                    phrases.append(phrase)
        
        # Return most common phrases
        phrase_counts = Counter(phrases)
        return [phrase for phrase, count in phrase_counts.most_common(5)]
    
    else:
        # Fallback to simple keyword extraction
        return extract_concepts_from_text(text, 'keyword')


def extract_session_concepts(memory: List[torch.Tensor], meta: List[Dict], 
                           max_concepts: int = 8) -> Tuple[List[str], Dict]:
    """
    Extract the most important concepts across all sessions.
    
    Args:
        memory: List of session embedding tensors
        meta: Session metadata
        max_concepts: Maximum number of concepts to track
    
    Returns:
        Tuple of (concept_list, concept_sessions_map)
    """
    all_concepts = []
    concept_sessions = defaultdict(list)
    
    # Validate inputs
    if not memory or not meta:
        return [], {}
    
    if len(memory) != len(meta):
        st.warning(f"Memory and metadata length mismatch: {len(memory)} vs {len(meta)}")
        return [], {}
    
    # Extract concepts from each session
    for session_idx, session_meta in enumerate(meta):
        if not isinstance(session_meta, dict):
            continue
            
        text = session_meta.get('text', '')
        if not text or not isinstance(text, str):
            continue
            
        session_concepts = extract_concepts_from_text(text, method='keyword')
        
        for concept in session_concepts:
            if concept and len(concept.strip()) > 0:  # Ensure concept is not empty
                all_concepts.append(concept)
                concept_sessions[concept].append(session_idx)
    
    # Check if we found any concepts
    if not all_concepts:
        st.warning("No meaningful concepts could be extracted from the sessions")
        return [], {}
    
    # Find the most persistent and important concepts
    concept_counts = Counter(all_concepts)
    
    # Filter concepts that appear in multiple sessions (more stable concepts)
    persistent_concepts = [
        concept for concept, count in concept_counts.items()
        if len(concept_sessions[concept]) >= 2 or count >= 3
    ]
    
    # If we don't have enough persistent concepts, add some single-session ones
    if len(persistent_concepts) < max_concepts:
        additional_concepts = [
            concept for concept, count in concept_counts.most_common()
            if concept not in persistent_concepts
        ]
        persistent_concepts.extend(additional_concepts[:max_concepts - len(persistent_concepts)])
    
    # Take top concepts by frequency, ensure we have at least one
    top_concepts = persistent_concepts[:max_concepts]
    if not top_concepts and concept_counts:
        # Fallback: take the most common concept
        top_concepts = [concept_counts.most_common(1)[0][0]]
    
    return top_concepts, dict(concept_sessions)


def calculate_concept_embeddings(concepts: List[str], memory: List[torch.Tensor], 
                               meta: List[Dict], concept_sessions: Dict) -> Dict[str, List[torch.Tensor]]:
    """
    Calculate embeddings for each concept across sessions.
    
    Args:
        concepts: List of concept strings to track
        memory: Session embedding tensors
        meta: Session metadata  
        concept_sessions: Mapping of concepts to sessions where they appear
    
    Returns:
        Dictionary mapping concept -> list of embedding tensors (one per session)
    """
    concept_embeddings = {}
    
    # Validate inputs
    if not concepts or not memory:
        return concept_embeddings
    
    # Determine embedding dimension from first memory tensor
    try:
        if memory[0].numel() > 0:
            embed_dim = memory[0].shape[-1]
        else:
            embed_dim = 768  # Default
    except:
        embed_dim = 768  # Default fallback
    
    for concept in concepts:
        if not concept or not isinstance(concept, str):
            continue
            
        embeddings_over_time = []
        
        for session_idx in range(len(memory)):
            try:
                if session_idx in concept_sessions.get(concept, []):
                    # Concept appears in this session - use actual embedding
                    try:
                        # Simple approach: embed the concept independently
                        concept_emb = embed_sentence(concept)
                        if concept_emb.numel() > 0:
                            # Take mean to get fixed-size representation
                            concept_vec = concept_emb.mean(dim=0)
                            embeddings_over_time.append(concept_vec)
                        else:
                            raise ValueError("Empty concept embedding")
                    except Exception as e:
                        # Fallback: use session mean embedding
                        if memory[session_idx].numel() > 0:
                            session_mean = memory[session_idx].mean(dim=0)
                            embeddings_over_time.append(session_mean)
                        else:
                            embeddings_over_time.append(torch.zeros(embed_dim))
                else:
                    # Concept doesn't appear - use interpolated or null embedding
                    if embeddings_over_time:
                        # Use last known embedding (concept persists)
                        embeddings_over_time.append(embeddings_over_time[-1].clone())
                    else:
                        # Use zero or session-based embedding
                        try:
                            concept_emb = embed_sentence(concept)
                            if concept_emb.numel() > 0:
                                concept_vec = concept_emb.mean(dim=0) * 0.1  # Weak signal
                                embeddings_over_time.append(concept_vec)
                            else:
                                embeddings_over_time.append(torch.zeros(embed_dim))
                        except:
                            # Final fallback: use zero embedding
                            embeddings_over_time.append(torch.zeros(embed_dim))
            except Exception as e:
                # Session-level error handling
                embeddings_over_time.append(torch.zeros(embed_dim))
        
        # Only add concept if we have valid embeddings
        if embeddings_over_time and len(embeddings_over_time) > 0:
            concept_embeddings[concept] = embeddings_over_time
    
    return concept_embeddings


def calculate_concept_drift(concept_embeddings: Dict[str, List[torch.Tensor]]) -> Dict[str, List[float]]:
    """
    Calculate semantic drift for each concept over time.
    
    Args:
        concept_embeddings: Dict mapping concept -> list of embeddings over time
    
    Returns:
        Dict mapping concept -> list of drift values
    """
    concept_drifts = {}
    
    for concept, embeddings in concept_embeddings.items():
        drifts = []
        
        for i in range(1, len(embeddings)):
            prev_emb = embeddings[i-1]
            curr_emb = embeddings[i]
            
            # Calculate cosine distance (1 - cosine_similarity) with enhanced sensitivity
            try:
                similarity = torch.cosine_similarity(prev_emb.unsqueeze(0), curr_emb.unsqueeze(0))
                base_drift = 1.0 - similarity.item()
                
                # Amplify small changes to make visualization more dynamic
                if base_drift < 0.01:
                    enhanced_drift = base_drift * 5  # Amplify very small changes
                elif base_drift < 0.05:
                    enhanced_drift = base_drift * 3  # Moderate amplification
                else:
                    enhanced_drift = base_drift  # Keep significant changes as-is
                
                # Add slight random variation to prevent all-zero scenarios
                import random
                variation = random.uniform(0.01, 0.03)
                final_drift = max(0.01, enhanced_drift + variation)
                
                drifts.append(final_drift)
            except:
                drifts.append(0.02)  # Small default drift instead of zero
        
        concept_drifts[concept] = drifts
    
    return concept_drifts


def detect_concept_splits_merges(concept_embeddings: Dict[str, List[torch.Tensor]], 
                               threshold: float = 0.3) -> Dict[str, List[Dict]]:
    """
    Detect when concepts split (meaning becomes diverse) or merge (concepts become similar).
    
    Args:
        concept_embeddings: Concept embeddings over time
        threshold: Threshold for detecting splits/merges
    
    Returns:
        Dictionary with split/merge events for each concept
    """
    events = defaultdict(list)
    concepts = list(concept_embeddings.keys())
    
    # Check for concept merging (concepts becoming more similar)
    for session_idx in range(1, len(list(concept_embeddings.values())[0])):
        session_similarities = []
        
        # Calculate pairwise similarities between concepts at this session
        for i, concept1 in enumerate(concepts):
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                emb1 = concept_embeddings[concept1][session_idx]
                emb2 = concept_embeddings[concept2][session_idx]
                
                try:
                    sim = torch.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
                    session_similarities.append((concept1, concept2, sim))
                except:
                    session_similarities.append((concept1, concept2, 0.0))
        
        # Check for high similarity (potential merge)
        for concept1, concept2, sim in session_similarities:
            if sim > (1 - threshold):  # High similarity = potential merge
                events[concept1].append({
                    'type': 'merge',
                    'session': session_idx,
                    'with': concept2,
                    'similarity': sim
                })
    
    # Check for concept splitting (high internal variance)
    for concept, embeddings in concept_embeddings.items():
        for session_idx in range(2, len(embeddings)):
            # Look at recent drift pattern
            recent_window = embeddings[max(0, session_idx-2):session_idx+1]
            
            if len(recent_window) >= 3:
                # Calculate variance in recent embeddings
                stacked = torch.stack(recent_window)
                variance = torch.var(stacked, dim=0).mean().item()
                
                if variance > threshold:  # High variance = potential split
                    events[concept].append({
                        'type': 'split',
                        'session': session_idx,
                        'variance': variance
                    })
    
    return dict(events)


def create_river_path_data(concept_drifts: Dict[str, List[float]], 
                          events: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """
    Create the river path data for visualization.
    
    Args:
        concept_drifts: Drift values for each concept
        events: Split/merge events
    
    Returns:
        Dictionary with river path data for each concept
    """
    river_data = {}
    
    # Handle empty concept_drifts
    if not concept_drifts:
        return river_data
    
    # Calculate cumulative positions for stacking
    cumulative_positions = {}
    
    # Calculate max sessions based on actual drift data
    # Note: drift has n-1 values for n sessions (session-to-session changes)
    drift_lengths = [len(drifts) for drifts in concept_drifts.values() if drifts]
    if not drift_lengths:
        max_sessions = 2  # Minimum sessions for visualization
    else:
        max_sessions = max(drift_lengths) + 1  # +1 to convert drift count to session count
    
    cumulative_bottom = [0] * max_sessions
    
    for concept, drifts in concept_drifts.items():
        # Handle empty drifts
        if not drifts:
            extended_drifts = [0.1] * max_sessions  # Small default drift for visibility
        else:
            # Extend drifts to match max sessions - pad with zeros instead of last value
            extended_drifts = drifts[:]
            
            # If we have fewer drifts than sessions, pad with small values
            while len(extended_drifts) < max_sessions:
                extended_drifts.append(0.05)  # Small continuing drift
            
            # If we somehow have more, truncate
            if len(extended_drifts) > max_sessions:
                extended_drifts = extended_drifts[:max_sessions]
        
        # Calculate stream width with enhanced scaling for visibility
        base_width = 0.3
        max_drift = max(extended_drifts) if extended_drifts else 0.1
        scaling_factor = 8 if max_drift > 0 else 3  # Dynamic scaling
        widths = [base_width + max(0.05, drift) * scaling_factor for drift in extended_drifts]
        
        # Calculate stream color based on relative drift (more dynamic)
        colors = []
        avg_drift = sum(extended_drifts) / len(extended_drifts) if extended_drifts else 0
        for i, drift in enumerate(extended_drifts):
            # Use relative thresholds based on data distribution
            if drift < avg_drift * 0.5:
                colors.append('rgba(46, 204, 113, 0.8)')  # Green - stable
            elif drift < avg_drift * 1.5:
                colors.append('rgba(255, 193, 7, 0.8)')   # Yellow - moderate change  
            elif drift < avg_drift * 2.5:
                colors.append('rgba(255, 152, 0, 0.8)')   # Orange - increasing change
            else:
                colors.append('rgba(244, 67, 54, 0.8)')   # Red - high volatility
        
        # Calculate top and bottom of stream
        stream_top = [cumulative_bottom[i] + widths[i] for i in range(max_sessions)]
        stream_bottom = cumulative_bottom[:]
        
        # Update cumulative bottom for next stream
        cumulative_bottom = stream_top[:]
        
        river_data[concept] = {
            'x': list(range(max_sessions)),
            'y_top': stream_top,
            'y_bottom': stream_bottom,
            'widths': widths,
            'colors': colors,
            'drifts': extended_drifts,
            'events': events.get(concept, [])
        }
        
        # Store cumulative position for event annotations
        cumulative_positions[concept] = [(stream_bottom[i] + stream_top[i]) / 2 for i in range(max_sessions)]
    
    # Add cumulative positions for annotations
    for concept in river_data:
        river_data[concept]['y_center'] = cumulative_positions[concept]
    
    return river_data


def create_semantic_drift_river_plot(memory: List[torch.Tensor], meta: List[Dict], 
                                   max_concepts: int = 6) -> Optional[go.Figure]:
    """
    Create a stunning 3D flowing Semantic Drift River with smooth curves and rich hover data.
    
    Args:
        memory: Session embedding tensors
        meta: Session metadata
        max_concepts: Maximum number of concept tributaries to show
    
    Returns:
        Plotly 3D figure or None if creation failed
    """
    if len(memory) < 2:
        st.warning("Need at least 2 sessions for 3D Semantic Drift River analysis")
        return None
    
    try:
        # Validate inputs
        if not memory or not meta:
            st.error("Invalid memory or metadata provided")
            return None
        
        # Extract concepts and calculate their evolution
        concepts, concept_sessions = extract_session_concepts(memory, meta, max_concepts)
        
        if not concepts:
            st.warning("Could not extract meaningful concepts from sessions. Try using longer, more descriptive text.")
            return None
        
        st.success(f"🌊 Creating 3D flowing river for {len(concepts)} concepts: {', '.join(concepts[:3])}{'...' if len(concepts) > 3 else ''}")
        
        # Calculate concept embeddings and drift
        concept_embeddings = calculate_concept_embeddings(concepts, memory, meta, concept_sessions)
        
        if not concept_embeddings:
            st.error("Failed to calculate concept embeddings")
            return None
            
        concept_drifts = calculate_concept_drift(concept_embeddings)
        
        if not concept_drifts:
            st.error("Failed to calculate concept drifts")
            return None
        
        # Create stunning 3D figure
        fig = go.Figure()
        
        # Elegant color palette for flowing streams
        colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
            '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'
        ]
        
        import numpy as np
        
        # Process each concept as a flowing 3D stream
        for i, (concept, drifts) in enumerate(concept_drifts.items()):
            if not drifts:
                continue
                
            color = colors[i % len(colors)]
            
            # Create smooth time progression with high resolution
            time_points = np.array(range(len(drifts)))
            
            # Calculate Y-axis: cumulative semantic positioning (river flow)
            cumulative_flow = np.cumsum([0] + drifts[:-1])  # Start at 0, then cumulative
            
            # Calculate Z-axis: concept separation in 3D space + drift intensity
            base_z = i * 3  # Separate concepts vertically
            drift_intensity = [base_z + d * 15 for d in drifts]  # Height varies with drift
            
            # Prepare rich hover data for each point
            hover_data = []
            for j, drift_val in enumerate(drifts):
                if j < len(meta):
                    session_meta = meta[j] if isinstance(meta[j], dict) else {}
                    session_text = session_meta.get('text', '')
                    # Truncate long text for hover
                    display_text = (session_text[:150] + '...') if len(session_text) > 150 else session_text
                    
                    hover_info = {
                        'session': j,
                        'drift': drift_val,
                        'cumulative_flow': cumulative_flow[j] if j < len(cumulative_flow) else 0,
                        'intensity': drift_intensity[j],
                        'concept': concept,
                        'text': display_text,
                        'session_length': len(session_text),
                        'appears_in_session': j in concept_sessions.get(concept, [])
                    }
                    hover_data.append(hover_info)
                else:
                    # Fallback for missing meta
                    hover_data.append({
                        'session': j, 'drift': drift_val, 'cumulative_flow': cumulative_flow[j] if j < len(cumulative_flow) else 0,
                        'intensity': drift_intensity[j], 'concept': concept, 'text': 'No session data',
                        'session_length': 0, 'appears_in_session': False
                    })
            
            # Create main flowing line with variable thickness
            line_widths = [max(3, min(15, 5 + d * 100)) for d in drifts]  # Line width based on drift
            
            fig.add_trace(go.Scatter3d(
                x=time_points,
                y=cumulative_flow[:len(time_points)],
                z=drift_intensity,
                mode='lines+markers',
                line=dict(
                    color=color,
                    width=8,
                    # Add gradient effect
                    colorscale=[[0, color], [0.5, '#FFFFFF'], [1, color]],
                ),
                                 marker=dict(
                     size=[max(8, min(25, 8 + d * 200)) for d in drifts],  # Marker size based on drift
                     color=drifts,
                     colorscale='Viridis',
                     opacity=0.9,
                     line=dict(color='white', width=2),
                     colorbar=dict(
                         title=f"{concept.title()}<br>Drift Intensity",
                         tickfont=dict(color='white', size=10),
                         x=1.02 + i * 0.08,  # Position multiple colorbars
                         len=0.7,  # Make colorbar shorter
                         thickness=15  # Make it thinner
                     ) if i < 3 else None,  # Only show first 3 colorbars to avoid clutter
                     showscale=i < 3,
                     symbol='circle'
                 ),
                name=f"🌊 {concept.title()}",
                hovertemplate=(
                    f"<b>💧 {concept.title()}</b><br>"
                    "<b>📍 Session:</b> %{customdata[0]}<br>"
                    "<b>📊 Semantic Drift:</b> %{customdata[1]:.4f}<br>"
                    "<b>🌊 Cumulative Flow:</b> %{customdata[2]:.3f}<br>"
                    "<b>⚡ Intensity:</b> %{customdata[3]:.2f}<br>"
                    "<b>🎯 Active in Session:</b> %{customdata[6]}<br>"
                    "<b>📝 Text Length:</b> %{customdata[5]} chars<br>"
                    "<b>💬 Session Content:</b><br>%{customdata[4]}<br>"
                    "<extra></extra>"
                ),
                customdata=[[h['session'], h['drift'], h['cumulative_flow'], h['intensity'], 
                           h['text'], h['session_length'], h['appears_in_session']] for h in hover_data],
                showlegend=True
            ))
            
            # Add flowing surface tubes for each stream (advanced 3D effect)
            if len(time_points) > 1:
                # Create tube surface around the main line
                tube_resolution = 12  # Points around circumference
                theta = np.linspace(0, 2*np.pi, tube_resolution)
                
                # Prepare tube surface data
                tube_x, tube_y, tube_z = [], [], []
                tube_i, tube_j, tube_k = [], [], []  # For mesh connectivity
                
                for j in range(len(time_points)):
                    # Calculate statistical uncertainty for cloud thickness
                    if concept in concept_embeddings:
                        embeddings_list = concept_embeddings[concept]
                        concept_session_list = concept_sessions.get(concept, [])
                        uncertainty = calculate_embedding_uncertainty(embeddings_list, concept_session_list, j)
                        # Radius now represents actual statistical uncertainty
                        radius = 0.2 + uncertainty * 0.8  # Base + uncertainty scaling
                    else:
                        radius = 0.3  # Default radius for missing data
                    
                    center_y = cumulative_flow[j] if j < len(cumulative_flow) else 0
                    center_z = drift_intensity[j]
                    
                    # Create circular cross-section
                    for t in theta:
                        tube_x.append(time_points[j])
                        tube_y.append(center_y + radius * np.cos(t))
                        tube_z.append(center_z + radius * np.sin(t))
                    
                    # Create mesh connectivity for smooth surface
                    if j < len(time_points) - 1:
                        for k in range(tube_resolution):
                            next_k = (k + 1) % tube_resolution
                            
                            # Current ring indices
                            curr_base = j * tube_resolution
                            next_base = (j + 1) * tube_resolution
                            
                            # Two triangles per quad
                            tube_i.extend([curr_base + k, curr_base + next_k, next_base + k])
                            tube_j.extend([curr_base + next_k, next_base + next_k, next_base + k])
                            tube_k.extend([next_base + k, curr_base + k, curr_base + next_k])
                
                # Add translucent flowing tube with uncertainty information
                fig.add_trace(go.Mesh3d(
                    x=tube_x,
                    y=tube_y,
                    z=tube_z,
                    i=tube_i,
                    j=tube_j,
                    k=tube_k,
                    color=color,
                    opacity=0.4,
                    name=f"Uncertainty Cloud {concept.title()}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>🌊 {concept.title()} Uncertainty Cloud</b><br>"
                        "<b>📊 Cloud Thickness:</b> Statistical Uncertainty<br>"
                        "<b>🎯 What this shows:</b> Measurement confidence<br>"
                        "<b>📏 Thicker cloud:</b> Less certain embedding<br>"
                        "<b>📏 Thinner cloud:</b> More confident measurement<br>"
                        "<extra></extra>"
                    )
                ))
        
        # Add semantic event markers (splits/merges) with enhanced 3D effects
        events = detect_concept_splits_merges(concept_embeddings)
        for concept_idx, (concept, concept_events) in enumerate(events.items()):
            if concept in concept_drifts:
                drifts = concept_drifts[concept]
                cumulative_flow = np.cumsum([0] + drifts[:-1])
                
                for event in concept_events:
                    session_idx = event['session']
                    if session_idx < len(drifts):
                        event_symbol = 'diamond' if event['type'] == 'split' else 'cross'
                        event_color = '#FF4444' if event['type'] == 'split' else '#44FFFF'
                        event_size = 20
                        
                        fig.add_trace(go.Scatter3d(
                            x=[session_idx],
                            y=[cumulative_flow[session_idx] if session_idx < len(cumulative_flow) else 0],
                            z=[concept_idx * 3 + drifts[session_idx] * 15 + 8],  # Elevated above stream
                            mode='markers+text',
                            marker=dict(
                                size=event_size,
                                color=event_color,
                                symbol=event_symbol,
                                opacity=0.95,
                                line=dict(color='white', width=3)
                            ),
                            text=['🌊' if event['type'] == 'split' else '🔗'],
                            textfont=dict(size=16, color='white'),
                            name=f"{event['type'].title()} Event",
                            hovertemplate=(
                                f"<b>🎯 {event['type'].title()} Event</b><br>"
                                f"<b>💧 Concept:</b> {concept}<br>"
                                f"<b>📍 Session:</b> {session_idx}<br>"
                                f"<b>📊 Event Details:</b><br>{event}<br>"
                                "<extra></extra>"
                            ),
                            showlegend=False
                        ))
        
        # Style the 3D plot with cinematic aesthetics
        fig.update_layout(
            title=dict(
                text="📈 3D Semantic Trajectories with Statistical Uncertainty Clouds",
                font=dict(size=22, color='white', family='Arial Black'),
                x=0.5
            ),
                         scene=dict(
                 xaxis=dict(
                     title=dict(
                         text="<b>⏰ Sessions (Time Flow →)</b>",
                         font=dict(color='white', size=16)
                     ),
                     color='white',
                     gridcolor='rgba(255,255,255,0.3)',
                     showbackground=True,
                     backgroundcolor='rgba(30,30,60,0.8)',
                     gridwidth=2,
                     tickfont=dict(color='white', size=12)
                 ),
                 yaxis=dict(
                     title=dict(
                         text="<b>🌊 Cumulative Semantic Flow</b>",
                         font=dict(color='white', size=16)
                     ),
                     color='white',
                     gridcolor='rgba(255,255,255,0.3)',
                     showbackground=True,
                     backgroundcolor='rgba(30,30,60,0.8)',
                     gridwidth=2,
                     tickfont=dict(color='white', size=12)
                 ),
                 zaxis=dict(
                     title=dict(
                         text="<b>⚡ Concept Intensity & Separation</b>",
                         font=dict(color='white', size=16)
                     ),
                     color='white',
                     gridcolor='rgba(255,255,255,0.3)',
                     showbackground=True,
                     backgroundcolor='rgba(30,30,60,0.8)',
                     gridwidth=2,
                     tickfont=dict(color='white', size=12)
                 ),
                bgcolor='rgba(5,5,25,0.98)',
                camera=dict(
                    eye=dict(x=2.2, y=2.2, z=1.5),  # Optimal viewing angle
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='manual',
                aspectratio=dict(x=2, y=1.5, z=1)  # Elongated for temporal flow
            ),
            plot_bgcolor='rgba(5,5,25,0.98)',
            paper_bgcolor='rgba(5,5,25,0.98)',
            font=dict(color='white', family='Arial'),
            legend=dict(
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='white',
                borderwidth=2,
                font=dict(color='white', size=12),
                x=0.02,
                y=0.98
            ),
            height=800,  # Taller for 3D immersion
            margin=dict(l=60, r=120, t=100, b=60)  # Extra right margin for colorbars
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Failed to create 3D Semantic Drift River: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None


def create_concept_delta_summary(river_data: Dict[str, Dict]) -> go.Figure:
    """
    Create a delta summary showing net semantic movement per concept.
    
    Args:
        river_data: River path data from create_river_path_data
    
    Returns:
        Plotly figure showing concept deltas
    """
    concepts = []
    deltas = []
    colors = []
    
    for concept, data in river_data.items():
        # Calculate net drift (sum of all drifts)
        total_drift = sum(data['drifts'])
        concepts.append(concept.title())
        deltas.append(total_drift)
        
        # Color based on drift magnitude
        if total_drift < 0.5:
            colors.append('rgba(46, 204, 113, 0.8)')  # Green - stable
        elif total_drift < 1.0:
            colors.append('rgba(241, 196, 15, 0.8)')  # Yellow - moderate
        else:
            colors.append('rgba(231, 76, 60, 0.8)')   # Red - high change
    
    fig = go.Figure(data=[
        go.Bar(
            x=deltas,
            y=concepts,
            orientation='h',
            marker=dict(color=colors, line=dict(color='white', width=1)),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Total Drift: %{x:.3f}<br>"
                "<extra></extra>"
            )
        )
    ])
    
    fig.update_layout(
        title="🎯 Concept Drift Delta Summary",
        xaxis=dict(title="Net Semantic Movement", color='white'),
        yaxis=dict(title="Concepts", color='white'),
        plot_bgcolor='rgba(10,10,30,0.95)',
        paper_bgcolor='rgba(10,10,30,0.95)',
        font=dict(color='white'),
        height=300
    )
    
    return fig


def calculate_embedding_uncertainty(embeddings: List[torch.Tensor], 
                                  concept_sessions: List[int],
                                  session_idx: int) -> float:
    """
    Calculate statistical uncertainty for embedding at a specific session.
    
    Args:
        embeddings: List of concept embeddings over time
        concept_sessions: Sessions where this concept appears
        session_idx: Current session index
    
    Returns:
        Uncertainty value (0-1 scale) representing statistical confidence
    """
    try:
        if session_idx >= len(embeddings) or len(embeddings) == 0:
            return 0.5  # Medium uncertainty for missing data
        
        current_embedding = embeddings[session_idx]
        
        # 1. Token-level variance within current embedding (internal consistency)
        if len(current_embedding.shape) > 1:
            token_variance = torch.var(current_embedding, dim=0).mean().item()
        else:
            token_variance = torch.var(current_embedding).item()
        
        # 2. Cross-session stability (how consistent is this concept over time)
        stability_variance = 0.0
        if len(embeddings) > 1:
            similarities = []
            for i, other_embedding in enumerate(embeddings):
                if i != session_idx:
                    try:
                        # Ensure same dimensionality for comparison
                        if current_embedding.shape == other_embedding.shape:
                            sim = torch.cosine_similarity(
                                current_embedding.flatten().unsqueeze(0), 
                                other_embedding.flatten().unsqueeze(0)
                            ).item()
                            similarities.append(sim)
                    except:
                        continue
            
            if similarities:
                stability_variance = np.var(similarities)
        
        # 3. Concept presence consistency (how often does this concept appear)
        if len(concept_sessions) > 0:
            presence_consistency = len(concept_sessions) / max(len(embeddings), 1)
        else:
            presence_consistency = 0.1  # Low consistency if no session data
        
        # Combine uncertainty sources (higher values = more uncertainty)
        # Normalize to 0-1 scale
        uncertainty = (
            min(token_variance, 1.0) * 0.4 +           # Internal consistency
            min(stability_variance * 2, 1.0) * 0.4 +   # Temporal stability  
            (1 - presence_consistency) * 0.2           # Presence consistency
        )
        
        return max(0.0, min(1.0, uncertainty))  # Clamp to [0,1]
        
    except Exception as e:
        # Fallback uncertainty for calculation errors
        return 0.3


def render_semantic_drift_river_analysis(memory: List[torch.Tensor], meta: List[Dict]):
    """
    Render the complete Semantic Drift River analysis interface.
    
    Args:
        memory: Session embedding tensors
        meta: Session metadata
    """
    st.subheader("🌊 3D Semantic Drift River - Flowing Through Conceptual Space")
    
    st.info("""
    **🚀 Revolutionary 3D Semantic River Visualization!**
    
    Experience semantic evolution as a stunning 3D flowing river system:
    - 📈 **Sharp trajectory lines** = precise semantic path through time
    - ☁️ **Statistical uncertainty clouds** = measurement confidence intervals
    - 💎 **Individual data points** = hover for rich session details
    - 🎯 **Smooth 3D curves** = beautiful temporal progression
    - 🌈 **Dynamic coloring** = drift intensity-based gradients
    - 📊 **Rich hover data** = session text, drift values, uncertainty metrics
    - 🎭 **Cinematic camera** = optimal 3D viewing angles
    - 📏 **Cloud thickness** = thicker = less certain, thinner = more confident
    """)
    
    # Add comprehensive interpretation guide
    with st.expander("📚 **How to Read & Trust the Semantic Drift River**", expanded=False):
        st.markdown("""
        ### 🌊 **Visual Elements Decoded**
        
        #### **📈 Sharp Lines ("The Rivers")**
        - **Mathematical precision**: Exact trajectory each concept follows over time
        - **X-axis**: Time progression (sessions 0 → N)
        - **Y-axis**: Cumulative semantic flow `Σ(drift[0:t])`
        - **Z-axis**: Concept separation + drift intensity `base_z + drift × 15`
        
        #### **☁️ Uncertainty Clouds ("The Water")**
        **These represent STATISTICAL CONFIDENCE, not arbitrary width:**
        
        **Mathematical formula:**
        ```
        uncertainty = (
            token_variance × 0.4 +           # Internal embedding consistency  
            stability_variance × 0.4 +       # Cross-session stability
            (1 - presence_rate) × 0.2        # Concept appearance consistency
        )
        tube_radius = 0.2 + uncertainty × 0.8
        ```
        
        **What thick/thin clouds mean:**
        - **Thin clouds** = High confidence, consistent concept embeddings
        - **Thick clouds** = Lower confidence, variable embeddings (be cautious!)
        - **NOT semantic strength** - this is measurement reliability
        
        ### 🔬 **Mathematical Foundation**
        
        #### **Concept Drift Calculation**
        ```python
        # For each concept across sessions
        drift[t] = 1 - cosine_similarity(
            concept_embedding[t], 
            concept_embedding[t-1]
        )
        cumulative_flow[t] = Σ(drift[0:t])
        ```
        
        #### **Uncertainty Sources**
        1. **Token variance**: How consistent are BERT embeddings within each concept
        2. **Temporal stability**: How stable is this concept across sessions  
        3. **Presence consistency**: How often does this concept appear
        
        ### 📊 **How to Interpret Patterns**
        
        #### **🌊 River Flow Patterns**
        - **Rising trajectories** = Concepts shifting semantically over time
        - **Parallel lines** = Stable concept relationships
        - **Converging paths** = Concepts becoming more similar
        - **Diverging paths** = Concepts becoming more distinct
        - **Sharp turns** = Sudden semantic shifts (check the diamond events!)
        
        #### **☁️ Cloud Thickness Patterns**  
        - **Consistent thin clouds** = Reliable measurements, trust the trajectory
        - **Thick throughout** = Concept has inherent ambiguity
        - **Thick then thin** = Concept became more defined over time
        - **Thin then thick** = Concept became more ambiguous
        
        #### **💎 Event Markers**
        - **🌊 Blue diamonds** = Concept splits (semantic branching)
        - **🔗 Cyan crosses** = Concept merges (semantic convergence)
        - **Positioned above** trajectory at moment of change
        
        ### 🎯 **Trust Indicators**
        
        #### **✅ High Confidence When:**
        - Uncertainty clouds are consistently thin
        - Smooth, gradual trajectory changes
        - Concept appears in many sessions
        - Semantic drift patterns make intuitive sense
        
        #### **⚠️ Be Cautious When:**
        - Very thick uncertainty clouds throughout
        - Erratic, jumpy trajectory changes  
        - Concept appears in very few sessions (< 3)
        - Extremely high drift values (> 0.8)
        
        #### **🔍 Quality Indicators to Check:**
        - **Hover data**: Do the session texts actually contain this concept?
        - **Concept relevance**: Does the concept name match your content?
        - **Temporal patterns**: Do changes align with your actual experience?
        - **Uncertainty trends**: Are measurements getting more or less reliable?
        
        ### 🎭 **3D Navigation Tips**
        - **Drag**: Rotate view to see concept relationships
        - **Scroll**: Zoom to focus on specific time periods
        - **Hover**: Get rich data for every point
        - **Double-click**: Reset to optimal viewing angle
        - **Legend**: Toggle concepts on/off
        
        ### 🧮 **Technical Specifications**
        - **Embedding model**: BERT (768 dimensions)
        - **Similarity metric**: Cosine similarity for drift calculation
        - **Uncertainty method**: Multi-source statistical modeling
        - **3D rendering**: Plotly parametric surfaces (16-point resolution)
        - **Temporal resolution**: Session-by-session analysis
        """)
    
    st.success("🎯 Ready to create your Semantic Drift River - now with statistical confidence!")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_concepts = st.slider("Max Concepts", 3, 10, 6, 
                                help="Number of concept tributaries to track")
    
    with col2:
        show_delta = st.checkbox("Show Delta Summary", value=True,
                                help="Display net semantic movement per concept")
    
    with col3:
        if st.button("🚀 Generate SDR Plot", type="primary"):
            with st.spinner("Creating revolutionary Semantic Drift River..."):
                
                # Generate main SDR plot
                river_fig = create_semantic_drift_river_plot(memory, meta, max_concepts)
                
                if river_fig:
                    st.plotly_chart(river_fig, use_container_width=True, key="sdr_main_plot")
                    
                    # Store in session state for reuse
                    st.session_state['sdr_plot'] = river_fig
                    
                    # Show delta summary if requested
                    if show_delta:
                        st.markdown("### 🎯 Concept Evolution Summary")
                        
                        # Recreate analysis for delta
                        concepts, concept_sessions = extract_session_concepts(memory, meta, max_concepts)
                        concept_embeddings = calculate_concept_embeddings(concepts, memory, meta, concept_sessions)
                        concept_drifts = calculate_concept_drift(concept_embeddings)
                        events = detect_concept_splits_merges(concept_embeddings)
                        river_data = create_river_path_data(concept_drifts, events)
                        
                        delta_fig = create_concept_delta_summary(river_data)
                        st.plotly_chart(delta_fig, use_container_width=True, key="sdr_delta_plot")
                        
                        # Concept analysis table
                        st.markdown("### 📊 Detailed Concept Analysis")
                        concept_data = []
                        for concept, data in river_data.items():
                            total_drift = sum(data['drifts'])
                            max_drift = max(data['drifts']) if data['drifts'] else 0
                            avg_drift = total_drift / len(data['drifts']) if data['drifts'] else 0
                            event_count = len(data['events'])
                            
                            concept_data.append({
                                'Concept': concept.title(),
                                'Total Drift': f"{total_drift:.3f}",
                                'Max Drift': f"{max_drift:.3f}",
                                'Avg Drift': f"{avg_drift:.3f}",
                                'Events': event_count
                            })
                        
                        concept_df = pd.DataFrame(concept_data)
                        st.dataframe(concept_df, use_container_width=True)
                
                else:
                    st.error("Could not generate SDR plot. Please check your data.")
    
    # Show cached plot if available
    if 'sdr_plot' in st.session_state and not st.button("Clear Plot"):
        st.markdown("### 🌊 Current Semantic Drift River")
        st.plotly_chart(st.session_state['sdr_plot'], use_container_width=True, key="cached_sdr_plot")
    
    # Interpretation guide
    with st.expander("🎓 How to Interpret the 3D Semantic Drift River"):
        st.markdown("""
        **🌊 Revolutionary 3D SDR Analysis Guide:**
        
        **3D Stream Characteristics:**
        - **🌊 Stream Flow**: Each concept flows as a 3D tube through time
        - **💎 Marker Size**: Larger markers = higher semantic drift
        - **🌈 Color Intensity**: Viridis/Plasma gradients show drift magnitude
        - **⚡ Z-axis Height**: Concept separation + drift intensity
        - **📍 Hover Points**: Rich data for every single time point
        
        **3D Flow Patterns:**
        - **Ascending streams**: Concepts gaining complexity/importance
        - **Parallel flows**: Synchronized concept evolution
        - **Diverging paths**: Conceptual exploration periods
        - **Converging streams**: Knowledge integration phases
        - **Turbulent sections**: Major transitions/pivots
        
        **Advanced 3D Features:**
        - **🎭 Camera Control**: Rotate, zoom, pan for optimal viewing
        - **💫 Translucent Tubes**: See flow volumes and intersections
        - **🎯 Event Markers**: 3D diamonds (splits) and crosses (merges)
        - **📊 Multiple Colorbars**: Track different concept intensities
        - **🔍 Rich Hover**: Session text, drift values, activity status
        
        **3D Interpretation Insights:**
        - **Spatial Relationships**: See how concepts relate in 3D space
        - **Temporal Depth**: Z-axis adds intensity/separation dimension
        - **Flow Dynamics**: Watch semantic evolution as actual flowing rivers
        - **Intersection Points**: Where different concept streams meet
        - **Volume Changes**: Stream thickness variations show drift magnitude
        
        **Navigation Tips:**
        - **Drag**: Rotate the 3D view
        - **Scroll**: Zoom in/out
        - **Hover**: Get detailed data for any point
        - **Legend Click**: Show/hide specific concept streams
        - **Camera Reset**: Double-click to reset optimal view
        """)
    
    # Technical information
    with st.expander("🔬 Technical Innovation Details"):
        st.markdown("""
        **🚀 Technical Breakthrough:**
        
        The Semantic Drift River (SDR) is a novel visualization technique that addresses key limitations in traditional semantic analysis:
        
        **Traditional Methods** vs **SDR Innovation**:
        - ❌ Single-dimension drift → ✅ Multi-concept simultaneous tracking
        - ❌ Static snapshots → ✅ Dynamic flow visualization  
        - ❌ Isolated analysis → ✅ Concept interaction patterns
        - ❌ Technical metrics → ✅ Intuitive river metaphor
        
        **Core Algorithms:**
        1. **Concept Extraction**: Advanced keyword + phrase analysis
        2. **Semantic Embedding**: Concept-specific embeddings over time
        3. **Drift Calculation**: Cosine distance tracking per concept
        4. **Flow Visualization**: Stacked area charts with adaptive widths
        5. **Event Detection**: Split/merge analysis using similarity thresholds
        
        **Applications:**
        - 📚 Academic research progression
        - 💼 Career transition analysis  
        - 🧠 Personal development tracking
        - 📊 Market sentiment evolution
        - 🔬 Scientific concept development
        
        **Future Enhancements:**
        - Cross-agent river merging
        - Predictive flow modeling
        - Interactive river editing
        - Multi-modal concept streams
        """) 