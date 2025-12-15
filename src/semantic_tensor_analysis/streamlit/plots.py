"""Streamlit-specific plotting functions for Semantic Tensor Memory.

This module contains Plotly and Altair plotting functions specifically designed 
for the Streamlit interface, including interactive plots, animations, and multi-view dashboards.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
from plotly.subplots import make_subplots
from semantic_tensor_analysis.compat.typing import ensure_closed_typeddict_support

ensure_closed_typeddict_support()
import altair as alt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from collections import Counter
import re
from datetime import datetime
import io
import base64
import time
from semantic_tensor_analysis.utils.tensors import to_cpu_numpy


def plot_drift_plotly(drifts, token_counts):
    """Create interactive drift plot using Plotly with secondary y-axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add drift scores (line, left y-axis)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(drifts))),
            y=drifts,
            name="Drift Score",
            line=dict(color='blue')
        ),
        secondary_y=False,
    )
    
    # Add token counts (bar, right y-axis)
    fig.add_trace(
        go.Bar(
            x=list(range(len(token_counts))),
            y=token_counts,
            name="Token Count",
            opacity=0.3,
            marker_color='gray'
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_layout(
        title="Session Drift Analysis",
        xaxis_title="Session",
        hovermode="x unified",
        showlegend=True
    )
    fig.update_yaxes(title_text="Drift Score", secondary_y=False)
    fig.update_yaxes(title_text="Token Count", secondary_y=True)
    
    return fig


def plot_heatmap_plotly(tensors):
    """Create interactive heatmap using Plotly with padding/mask-aware session vectors."""
    from semantic_tensor_analysis.analytics.tensor_batching import pad_and_stack, masked_session_means
    # Pad and compute masked means for stability
    batch, mask = pad_and_stack(tensors)
    means = masked_session_means(batch, mask)  # [B, D]

    # Compute cosine similarity matrix
    means_norm = torch.nn.functional.normalize(means, p=2, dim=1) if means.numel() else means
    sims = torch.mm(means_norm, means_norm.t()) if means.numel() else torch.zeros(0, 0)
    dist = 1 - to_cpu_numpy(sims) if sims.numel() else np.zeros((0, 0))

    # Get token counts for annotation
    token_counts = [t.shape[0] for t in tensors]

    # Create heatmap with a cool-to-warm color scale
    fig = go.Figure(data=go.Heatmap(
        z=dist,
        colorscale='RdYlBu',
        reversescale=True,
        text=[[f"{count}" for count in token_counts] for _ in range(len(token_counts))] if len(token_counts) > 0 else None,
        texttemplate="%{text}" if len(token_counts) > 0 else None,
        textfont={"size": 10},
        colorbar=dict(title="Cosine distance")
    ))

    fig.update_layout(
        title="Session-to-Session Semantic Drift",
        xaxis_title="Session",
        yaxis_title="Session"
    )

    return fig


def create_pca_visualization(results, meta_slice, is_3d=False):
    """Create PCA visualization from results with proper statistical annotations."""
    if results is None:
        return None
        
    reduced = results['reduced']
    session_ids = results['session_ids']
    pca = results['pca']
    
    # Create DataFrame for plotting
    if is_3d:
        df = pd.DataFrame({
            'PCA1': reduced[:, 0],
            'PCA2': reduced[:, 1],
            'PCA3': reduced[:, 2],
            'Session': [f"Session {j+1}" for j in session_ids],
            'SessionIdx': session_ids,
            'Text': [meta_slice[j]['text'] for j in session_ids]
        })
    else:
        df = pd.DataFrame({
            'PCA1': reduced[:, 0],
            'PCA2': reduced[:, 1],
            'Session': [f"Session {j+1}" for j in session_ids],
            'SessionIdx': session_ids,
            'Text': [meta_slice[j]['text'] for j in session_ids]
        })
    
    # Calculate session means for axis labeling
    unique_sessions = sorted(set(session_ids))
    session_means_pca = []
    session_metadata_for_labels = []
    
    for session_idx in unique_sessions:
        session_mask = np.array(session_ids) == session_idx
        session_pca_mean = reduced[session_mask].mean(axis=0)
        session_means_pca.append(session_pca_mean)
        session_metadata_for_labels.append({
            'session_idx': session_idx,
            'text': meta_slice[session_idx]['text'],
            'session_name': f'Session {session_idx+1}',
            'token_count': meta_slice[session_idx].get('tokens', len(meta_slice[session_idx]['text']))
        })
    
    session_means_pca = np.array(session_means_pca)
    
    # Create visualization
    if is_3d:
        fig = px.scatter_3d(
            df,
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color='SessionIdx',
            color_continuous_scale='RdYlBu',
            range_color=[df['SessionIdx'].min(), df['SessionIdx'].max()],
            hover_data=['Session', 'Text'],
            title=f"3D Semantic Drift Map (Explains {results['cumulative_variance'][-1]:.1%} of variance)"
        )
        fig.update_traces(marker=dict(size=4, reversescale=True))
        
        # Generate dynamic axis labels
        try:
            from semantic_tensor_analysis.streamlit.utils import generate_dynamic_axis_labels
            axis_labels = generate_dynamic_axis_labels(
                session_means_pca, 
                session_metadata_for_labels, 
                pca
            )
        except Exception as e:
            st.warning(f"Could not generate dynamic axis labels: {e}")
            axis_labels = [
                f"Semantic Dimension 1 (Explains {results['explained_variance_ratio'][0]:.1%} of variance)",
                f"Semantic Dimension 2 (Explains {results['explained_variance_ratio'][1]:.1%} of variance)", 
                f"Semantic Dimension 3 (Explains {results['explained_variance_ratio'][2]:.1%} of variance)"
            ]
        
        fig.update_layout(
            scene = dict(
                xaxis_title=f'<b>{axis_labels[0]}</b><br><span style="font-size:12px; color:gray">Explains {results["explained_variance_ratio"][0]:.1%} of variance</span>',
                yaxis_title=f'<b>{axis_labels[1]}</b><br><span style="font-size:12px; color:gray">Explains {results["explained_variance_ratio"][1]:.1%} of variance</span>',
                zaxis_title=f'<b>{axis_labels[2]}</b><br><span style="font-size:12px; color:gray">Explains {results["explained_variance_ratio"][2]:.1%} of variance</span>',
            ),
            coloraxis_colorbar=dict(title="Session"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    else:
        fig = px.scatter(
            df,
            x='PCA1',
            y='PCA2',
            color='SessionIdx',
            color_continuous_scale='RdYlBu',
            range_color=[df['SessionIdx'].min(), df['SessionIdx'].max()],
            hover_data=['Session', 'Text'],
            title=f"Semantic Drift Map (Explains {results['cumulative_variance'][-1]:.1%} of variance)"
        )
        fig.update_traces(marker=dict(reversescale=True))
        
        # Generate dynamic axis labels
        try:
            from semantic_tensor_analysis.streamlit.utils import generate_dynamic_axis_labels
            axis_labels = generate_dynamic_axis_labels(
                session_means_pca, 
                session_metadata_for_labels, 
                pca
            )
        except Exception as e:
            st.warning(f"Could not generate dynamic axis labels: {e}")
            axis_labels = [
                f"Semantic Dimension 1 (Explains {results['explained_variance_ratio'][0]:.1%} of variance)",
                f"Semantic Dimension 2 (Explains {results['explained_variance_ratio'][1]:.1%} of variance)"
            ]
        
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            coloraxis_colorbar=dict(title="Session"),
            xaxis_title=f'{axis_labels[0]}',
            yaxis_title=f'{axis_labels[1]}'
        )
    
    return fig


def create_animated_pca_trajectory(results, meta_slice, animation_speed=500):
    """Create animated PCA visualization showing semantic trajectory evolution over time."""
    if results is None:
        return None
        
    reduced = results['reduced']
    session_ids = results['session_ids']
    
    # Create session-level aggregations for trajectory
    unique_sessions = sorted(set(session_ids))
    trajectory_data = []
    
    for i, session_idx in enumerate(unique_sessions):
        session_mask = np.array(session_ids) == session_idx
        session_points = reduced[session_mask]
        
        # Calculate session centroid
        centroid = session_points.mean(axis=0)
        
        # Calculate session spread (variance)
        spread = np.std(session_points, axis=0)
        
        trajectory_data.append({
            'Session': session_idx + 1,
            'PCA1': centroid[0],
            'PCA2': centroid[1],
            'PCA3': centroid[2] if reduced.shape[1] > 2 else 0,
            'Spread1': spread[0],
            'Spread2': spread[1],
            'Spread3': spread[2] if reduced.shape[1] > 2 else 0,
            'Text': meta_slice[session_idx]['text'][:100] + "...",
            'SessionOrder': i,
            'CumulativeVariance': results['cumulative_variance'][-1] if i == len(unique_sessions)-1 else results['cumulative_variance'][min(i, len(results['cumulative_variance'])-1)]
        })
    
    df_trajectory = pd.DataFrame(trajectory_data)
    
    # Create animated trajectory plot
    if reduced.shape[1] >= 3:
        # 3D animated trajectory
        fig = px.scatter_3d(
            df_trajectory,
            x='PCA1', y='PCA2', z='PCA3',
            animation_frame='SessionOrder',
            size='Session',
            color='Session',
            color_continuous_scale='RdYlBu',
            hover_data=['Text'],
            title=f"Animated Semantic Trajectory (3D) - {results['cumulative_variance'][-1]:.1%} Variance"
        )
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=df_trajectory['PCA1'],
            y=df_trajectory['PCA2'], 
            z=df_trajectory['PCA3'],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=4, color='red'),
            name='Trajectory Path',
            showlegend=True
        ))
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 ({results['explained_variance_ratio'][0]:.1%})",
                yaxis_title=f"PC2 ({results['explained_variance_ratio'][1]:.1%})",
                zaxis_title=f"PC3 ({results['explained_variance_ratio'][2]:.1%})"
            )
        )
    else:
        # 2D animated trajectory
        fig = px.scatter(
            df_trajectory,
            x='PCA1', y='PCA2',
            animation_frame='SessionOrder',
            size='Session',
            color='Session',
            color_continuous_scale='RdYlBu',
            hover_data=['Text'],
            title=f"Animated Semantic Trajectory (2D) - {results['cumulative_variance'][-1]:.1%} Variance"
        )
        
        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=df_trajectory['PCA1'],
            y=df_trajectory['PCA2'],
            mode='lines+markers',
            line=dict(color='red', width=3),
            marker=dict(size=6, color='red'),
            name='Trajectory Path',
            showlegend=True
        ))
        
        fig.update_layout(
            xaxis_title=f"PC1 ({results['explained_variance_ratio'][0]:.1%})",
            yaxis_title=f"PC2 ({results['explained_variance_ratio'][1]:.1%})"
        )
    
    # Configure animation
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': animation_speed, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 200}
                    }]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '🔄 Reset',
                    'method': 'animate',
                    'args': [[df_trajectory['SessionOrder'].iloc[0]], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }]
    )
    
    return fig


def create_temporal_heatmap(results, meta_slice, window_size=5):
    """Create temporal heatmap showing semantic similarity evolution in sliding windows."""
    if results is None or len(results['session_ids']) < window_size:
        return None
        
    reduced = results['reduced']
    session_ids = results['session_ids']
    unique_sessions = sorted(set(session_ids))
    
    # Calculate session centroids
    session_centroids = []
    for session_idx in unique_sessions:
        session_mask = np.array(session_ids) == session_idx
        centroid = reduced[session_mask].mean(axis=0)
        session_centroids.append(centroid)
    
    session_centroids = np.array(session_centroids)
    
    # Create sliding window similarity matrices
    window_data = []
    for start_idx in range(len(unique_sessions) - window_size + 1):
        end_idx = start_idx + window_size
        window_centroids = session_centroids[start_idx:end_idx]
        
        # Calculate cosine similarity matrix for this window
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(window_centroids)
        
        # Store data for animation
        for i in range(window_size):
            for j in range(window_size):
                window_data.append({
                    'Window': start_idx,
                    'Session_i': unique_sessions[start_idx + i] + 1,
                    'Session_j': unique_sessions[start_idx + j] + 1,
                    'Similarity': sim_matrix[i, j],
                    'WindowStart': start_idx + 1,
                    'WindowEnd': end_idx,
                    'i': i,
                    'j': j
                })
    
    if not window_data:
        return None
        
    df_temporal = pd.DataFrame(window_data)
    
    # Create animated heatmap
    fig = px.imshow(
        df_temporal.pivot_table(
            values='Similarity', 
            index='Session_i', 
            columns='Session_j', 
            aggfunc='first'
        ),
        animation_frame=df_temporal['Window'].unique()[0] if len(df_temporal['Window'].unique()) == 1 else None,
        color_continuous_scale='RdYlBu',
        aspect='auto',
        title=f"Temporal Semantic Similarity (Window Size: {window_size})"
    )
    
    if len(df_temporal['Window'].unique()) > 1:
        # Create frames for animation
        frames = []
        for window_idx in sorted(df_temporal['Window'].unique()):
            window_df = df_temporal[df_temporal['Window'] == window_idx]
            if not window_df.empty:
                frame_matrix = window_df.pivot_table(
                    values='Similarity',
                    index='Session_i', 
                    columns='Session_j',
                    aggfunc='first'
                ).fillna(0)
                
                frames.append(go.Frame(
                    data=[go.Heatmap(
                        z=frame_matrix.values,
                        x=frame_matrix.columns,
                        y=frame_matrix.index,
                        colorscale='RdYlBu',
                        reversescale=True
                    )],
                    name=str(window_idx)
                ))
        
        fig.frames = frames
        
        # Add animation controls
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 800, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }]
        )
    
    fig.update_layout(
        xaxis_title="Session",
        yaxis_title="Session",
        coloraxis_colorbar=dict(title="Cosine Similarity")
    )
    
    return fig


def create_variance_evolution_animation(results):
    """Create animated plot showing how explained variance evolves with components."""
    if results is None:
        return None
        
    explained_variance = results['explained_variance_ratio']
    cumulative_variance = results['cumulative_variance']
    
    # Create data for animation
    animation_data = []
    for i in range(1, len(explained_variance) + 1):
        for j in range(i):
            animation_data.append({
                'Component': j + 1,
                'Individual_Variance': explained_variance[j],
                'Cumulative_Variance': cumulative_variance[i-1],
                'Frame': i,
                'ComponentsShown': i
            })
    
    df_variance = pd.DataFrame(animation_data)
    
    # Create animated bar chart
    fig = px.bar(
        df_variance,
        x='Component',
        y='Individual_Variance',
        animation_frame='Frame',
        title="Principal Component Variance Explanation (Animated)",
        labels={
            'Individual_Variance': 'Explained Variance Ratio',
            'Component': 'Principal Component'
        }
    )
    
    # Add cumulative variance line
    fig.add_trace(go.Scatter(
        x=list(range(1, len(cumulative_variance) + 1)),
        y=cumulative_variance,
        mode='lines+markers',
        name='Cumulative Variance',
        line=dict(color='red', width=3),
        marker=dict(size=8, color='red')
    ))
    
    fig.update_layout(
        xaxis_title="Principal Component",
        yaxis_title="Explained Variance Ratio",
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'buttons': [
                {
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': 600, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 200}
                    }]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }]
    )
    
    return fig


def create_pca_timeline_animation(memory, meta, animation_speed=800, is_3d=False):
    """Create animation showing how PCA space evolves as sessions are added progressively."""
    if len(memory) < 3:
        return None
    
    from semantic_tensor_analysis.streamlit.utils import robust_pca_pipeline
    
    # Add progress indicator and limit processing for performance
    total_sessions = len(memory)
    
    # For performance, limit to processing every 2-3 sessions for large datasets
    if total_sessions > 15:
        session_steps = list(range(2, min(8, total_sessions), 1)) + list(range(8, total_sessions + 1, 2))
    else:
        session_steps = list(range(2, total_sessions + 1))
    
    # Always include the final session
    if session_steps[-1] != total_sessions:
        session_steps.append(total_sessions)
    
    # Generate PCA results for each progressive subset
    timeline_data = []
    
    with st.spinner(f"Generating PCA timeline for {len(session_steps)} frames..."):
        for i, n_sessions in enumerate(session_steps):
            memory_subset = memory[:n_sessions]
            meta_subset = meta[:n_sessions]
            
            # Calculate PCA for this subset
            results = robust_pca_pipeline(
                memory_subset, 
                meta_subset, 
                n_components=3 if is_3d else 2,
                method='auto'
            )
            
            if results:
                reduced = results['reduced']
                session_ids = results['session_ids']
                
                # Create frame data
                frame_data = []
                for j, session_id in enumerate(session_ids):
                    # Convert boolean IsNew to numeric size values
                    is_newest = session_id == n_sessions - 1
                    marker_size = 12 if is_newest else 6  # Larger for newest session
                    
                    frame_data.append({
                        'Session': session_id + 1,
                        'PCA1': reduced[j, 0],
                        'PCA2': reduced[j, 1],
                        'PCA3': reduced[j, 2] if is_3d else 0,
                        'Frame': i,  # Use frame index instead of n_sessions for consistency
                        'SessionsIncluded': n_sessions,
                        'Text': meta_subset[session_id]['text'][:100] + "...",
                        'ExplainedVar': results['cumulative_variance'][-1],
                        'Quality': results['quality_assessment'],
                        'MarkerSize': marker_size,  # Use numeric size values
                        'IsNewest': 'Yes' if is_newest else 'No'  # String for hover
                    })
                
                timeline_data.extend(frame_data)
    
    if not timeline_data:
        return None
    
    df_timeline = pd.DataFrame(timeline_data)
    
    # Create animated plot
    if is_3d:
        fig = px.scatter_3d(
            df_timeline,
            x='PCA1', y='PCA2', z='PCA3',
            animation_frame='Frame',
            color='Session',
            size='MarkerSize',  # Use numeric MarkerSize instead of boolean IsNew
            color_continuous_scale='RdYlBu',
            hover_data=['Text', 'ExplainedVar', 'Quality', 'IsNewest', 'SessionsIncluded'],
            title="PCA Space Evolution Over Time (3D)"
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title=f"PC1 (Semantic Dimension)",
                yaxis_title=f"PC2 (Semantic Dimension)",
                zaxis_title=f"PC3 (Semantic Dimension)"
            )
        )
    else:
        fig = px.scatter(
            df_timeline,
            x='PCA1', y='PCA2',
            animation_frame='Frame',
            color='Session',
            size='MarkerSize',  # Use numeric MarkerSize instead of boolean IsNew
            color_continuous_scale='RdYlBu',
            hover_data=['Text', 'ExplainedVar', 'Quality', 'IsNewest', 'SessionsIncluded'],
            title="PCA Space Evolution Over Time (2D)"
        )
        
        fig.update_layout(
            xaxis_title=f"PC1 (Semantic Dimension)",
            yaxis_title=f"PC2 (Semantic Dimension)"
        )
    
    # Configure animation with custom controls
    fig.update_layout(
        updatemenus=[{
            'type': 'buttons',
            'showactive': False,
            'x': 0.1,
            'y': 0,
            'buttons': [
                {
                    'label': '▶️ Play Timeline',
                    'method': 'animate',
                    'args': [None, {
                        'frame': {'duration': animation_speed, 'redraw': True},
                        'fromcurrent': True,
                        'transition': {'duration': 300}
                    }]
                },
                {
                    'label': '⏸️ Pause',
                    'method': 'animate',
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': False},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '🔄 Reset',
                    'method': 'animate',
                    'args': [[df_timeline['Frame'].min()], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                },
                {
                    'label': '⏭️ Final',
                    'method': 'animate',
                    'args': [[df_timeline['Frame'].max()], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }]
                }
            ]
        }],
        annotations=[{
            'text': f'Timeline shows {len(session_steps)} key points in PCA evolution',
            'showarrow': False,
            'xref': 'paper', 'yref': 'paper',
            'x': 0.5, 'y': -0.1,
            'xanchor': 'center', 'yanchor': 'top',
            'font': {'size': 12, 'color': 'gray'}
        }]
    )
    
    return fig


def create_4d_semantic_space_visualization(memory, meta):
    """Create 4D PCA visualization with semantic space connections and 4th dimension mapping."""
    if len(memory) < 2:
        return None
    
    from semantic_tensor_analysis.streamlit.utils import robust_pca_pipeline
    
    # Calculate 4D PCA for full dataset
    results = robust_pca_pipeline(
        memory, 
        meta, 
        n_components=4,  # Get 4 components for true 4D visualization
        method='auto'
    )
    
    if not results:
        return None
    
    reduced = results['reduced']
    session_ids = results['session_ids']
    
    # Calculate session-level data for tunneling
    unique_sessions = sorted(set(session_ids))
    session_centroids = []
    session_metadata = []
    
    for session_idx in unique_sessions:
        session_mask = np.array(session_ids) == session_idx
        session_points = reduced[session_mask]
        
        # Calculate session centroid in 4D space
        centroid = session_points.mean(axis=0)
        session_centroids.append(centroid)
        
        session_metadata.append({
            'session_idx': session_idx,
            'session_name': f'Session {session_idx + 1}',
            'text': meta[session_idx]['text'][:100] + "...",
            'variance': np.std(session_points, axis=0).mean(),  # Average variance across dimensions
            'token_count': meta[session_idx].get('tokens', len(meta[session_idx]['text']))
        })
    
    session_centroids = np.array(session_centroids)
    
    if len(session_centroids) < 2:
        return None
    
    # Create the 4D tunneling visualization
    fig = go.Figure()
    
    # 1. Add session points (3D coordinates + 4th dimension controls visual properties)
    x_coords = session_centroids[:, 0]
    y_coords = session_centroids[:, 1] 
    z_coords = session_centroids[:, 2]
    w_coords = session_centroids[:, 3]  # 4th dimension
    
    # Normalize 4th dimension for visual effects
    w_normalized = (w_coords - w_coords.min()) / (w_coords.max() - w_coords.min()) if w_coords.max() != w_coords.min() else np.ones_like(w_coords)
    
    # Create hover text
    hover_texts = []
    for i, meta_info in enumerate(session_metadata):
        hover_text = [
            f"<b>{meta_info['session_name']}</b>",
            f"PC1: {x_coords[i]:.3f}",
            f"PC2: {y_coords[i]:.3f}", 
            f"PC3: {z_coords[i]:.3f}",
            f"PC4: {w_coords[i]:.3f}",
            f"4D Intensity: {w_normalized[i]:.3f}",
            f"Session Variance: {meta_info['variance']:.3f}",
            f"Tokens: {meta_info['token_count']}",
            f"Content: {meta_info['text']}"
        ]
        hover_texts.append("<br>".join(hover_text))
    
    # 2. Add tunneling connections between consecutive sessions
    for i in range(len(session_centroids) - 1):
        start_point = session_centroids[i]
        end_point = session_centroids[i + 1]
        
        # Create tunnel segments
        n_segments = 20  # Number of segments in tunnel
        t_values = np.linspace(0, 1, n_segments)
        
        tunnel_x = []
        tunnel_y = []
        tunnel_z = []
        tunnel_colors = []
        tunnel_sizes = []
        
        for t in t_values:
            # Linear interpolation between points
            tunnel_point = start_point * (1 - t) + end_point * t
            tunnel_x.append(tunnel_point[0])
            tunnel_y.append(tunnel_point[1])
            tunnel_z.append(tunnel_point[2])
            
            # Use 4th dimension to control tunnel properties
            w_interp = start_point[3] * (1 - t) + end_point[3] * t
            w_norm_interp = (w_interp - w_coords.min()) / (w_coords.max() - w_coords.min()) if w_coords.max() != w_coords.min() else 0.5
            
            tunnel_colors.append(w_norm_interp)
            tunnel_sizes.append(3 + 7 * w_norm_interp)  # Size varies from 3 to 10 based on 4th dimension
        
        # Add tunnel as a line with varying properties
        fig.add_trace(go.Scatter3d(
            x=tunnel_x,
            y=tunnel_y,
            z=tunnel_z,
            mode='lines+markers',
            line=dict(
                color=tunnel_colors,
                colorscale='Viridis',
                width=8,
                colorbar=dict(title="4th Dimension<br>Intensity") if i == 0 else None
            ),
            marker=dict(
                size=tunnel_sizes,
                color=tunnel_colors,
                colorscale='Viridis',
                opacity=0.6
            ),
            name=f'Tunnel {i+1}→{i+2}',
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # 3. Add main session points (larger, more prominent)
    fig.add_trace(go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers+text',
        marker=dict(
            size=15 + 10 * w_normalized,  # Size based on 4th dimension
            color=w_normalized,
            colorscale='RdYlBu',
            opacity=0.9,
            line=dict(width=3, color='white'),
            colorbar=dict(
                title="4th Dimension (PC4)",
                x=1.02
            )
        ),
        text=[f"S{i+1}" for i in range(len(unique_sessions))],
        textposition='top center',
        textfont=dict(size=12, color='black'),
        hovertext=hover_texts,
        hoverinfo='text',
        name='Session Centers'
    ))
    
    # 4. Add directional flow indicators (arrows)
    for i in range(len(session_centroids) - 1):
        start_point = session_centroids[i]
        end_point = session_centroids[i + 1]
        
        # Calculate arrow position (3/4 along the tunnel)
        arrow_pos = start_point * 0.25 + end_point * 0.75
        direction = end_point - start_point
        direction_norm = direction / np.linalg.norm(direction[:3])  # Normalize 3D direction
        
        # Add arrow
        fig.add_trace(go.Cone(
            x=[arrow_pos[0]],
            y=[arrow_pos[1]],
            z=[arrow_pos[2]],
            u=[direction_norm[0]],
            v=[direction_norm[1]],
            w=[direction_norm[2]],
            sizemode="scaled",
            sizeref=0.3,
            anchor="tail",
            colorscale="Reds",
            showscale=False,
            opacity=0.7,
            name=f'Flow {i+1}→{i+2}',
            showlegend=False
        ))
    
    # Generate dynamic axis labels
    try:
        from semantic_tensor_analysis.streamlit.utils import generate_dynamic_axis_labels
        axis_labels = generate_dynamic_axis_labels(
            session_centroids, 
            session_metadata, 
            results['pca']
        )
    except:
        axis_labels = [
            f"PC1 ({results['explained_variance_ratio'][0]:.1%})",
            f"PC2 ({results['explained_variance_ratio'][1]:.1%})",
            f"PC3 ({results['explained_variance_ratio'][2]:.1%})"
        ]
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"4D Semantic Space Visualization<br><sub>3D space + 4th dimension controls visual properties | Total variance: {results['cumulative_variance'][-1]:.1%}</sub>",
            font=dict(size=18),
            x=0.5
        ),
        scene=dict(
            xaxis_title=f"<b>{axis_labels[0]}</b>",
            yaxis_title=f"<b>{axis_labels[1]}</b>",
            zaxis_title=f"<b>{axis_labels[2]}</b>",
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.8),
                center=dict(x=0, y=0, z=0)
            ),
            bgcolor='rgba(240,240,240,0.1)',
            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
            yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1),
            zaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=1)
        ),
        margin=dict(l=0, r=50, b=0, t=80),
        height=700,
        showlegend=True,
        legend=dict(
            x=0,
            y=1,
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='gray',
            borderwidth=1
        )
    )
    
    return fig


def create_liminal_tunnel_visualization(memory, meta, tunnel_segments=25, tunnel_radius_base=0.15):
    """Create an optimized 3D tube tunnel through liminal space combining PCA and t-SNE."""
    if len(memory) < 3:
        return None
    
    # PERFORMANCE OPTIMIZATION: Add early timeout check
    start_time = time.time()
    max_computation_time = 30  # 30 second timeout
    
    try:
        from semantic_tensor_analysis.streamlit.utils import robust_pca_pipeline
        from semantic_tensor_analysis.visualization.viz.pca_plot import prepare_for_pca
        from sklearn.manifold import TSNE
        from scipy.interpolate import interp1d
        
        # OPTIMIZATION 1: Reduced tunnel complexity
        tunnel_segments = min(tunnel_segments, 20)  # Cap at 20 segments
        
        # Prepare data with size limits
        flat, session_ids, token_ids = prepare_for_pca(memory)
        
        # OPTIMIZATION 2: Limit data size for performance
        max_data_points = 2000  # Dramatically reduced from unlimited
        if len(flat) > max_data_points:
            indices = np.linspace(0, len(flat)-1, max_data_points, dtype=int)
            flat = flat[indices]
            session_ids = [session_ids[i] for i in indices]
            token_ids = [token_ids[i] for i in indices]
        
        # Calculate session centroids for temporal path
        unique_sessions = sorted(set(session_ids))
        session_centroids_high_dim = []
        session_metadata = []
        
        for session_idx in unique_sessions:
            session_mask = np.array(session_ids) == session_idx
            session_points = flat[session_mask]
            if len(session_points) > 0:
                centroid = session_points.mean(axis=0)
                session_centroids_high_dim.append(centroid)
                
                session_metadata.append({
                    'session_idx': session_idx,
                    'session_name': f'Session {session_idx + 1}',
                    'text': meta[session_idx]['text'][:100] + "...",
                    'token_count': session_points.shape[0],
                    'variance': np.std(session_points, axis=0).mean()
                })
        
        session_centroids_high_dim = np.array(session_centroids_high_dim)
        
        # Check timeout
        if time.time() - start_time > max_computation_time:
            st.warning("⚠️ Computation timeout - using simplified visualization")
            return create_simple_tunnel_fallback(session_centroids_high_dim, session_metadata)
        
        # OPTIMIZATION 3: Skip t-SNE for small datasets or use faster alternative
        if len(session_centroids_high_dim) <= 8:
            st.info("🚀 Using PCA-only mode for optimal performance")
            # Use PCA only for small datasets
            pca_results = robust_pca_pipeline(memory, meta, n_components=3, method='pca')
            if not pca_results:
                return None
            
            pca_session_centroids = []
            for session_idx in unique_sessions:
                session_mask = np.array(pca_results['session_ids']) == session_idx
                if np.any(session_mask):
                    pca_centroid = pca_results['reduced'][session_mask].mean(axis=0)
                    pca_session_centroids.append(pca_centroid)
            
            hybrid_coords = np.array(pca_session_centroids)
            
        else:
            # OPTIMIZATION 4: Heavily constrained t-SNE
            st.info("🔄 Running fast hybrid PCA+t-SNE analysis...")
            
            # 1. PCA for global structure 
            pca_results = robust_pca_pipeline(memory, meta, n_components=3, method='pca')
            if not pca_results:
                return None
            
            pca_session_centroids = []
            for session_idx in unique_sessions:
                session_mask = np.array(pca_results['session_ids']) == session_idx
                if np.any(session_mask):
                    pca_centroid = pca_results['reduced'][session_mask].mean(axis=0)
                    pca_session_centroids.append(pca_centroid)
            
            pca_session_centroids = np.array(pca_session_centroids)
            
            # 2. FAST t-SNE with aggressive constraints
            max_samples_tsne = min(100, len(session_centroids_high_dim))  # Drastically reduced
            if len(session_centroids_high_dim) > max_samples_tsne:
                tsne_indices = np.linspace(0, len(session_centroids_high_dim)-1, max_samples_tsne, dtype=int)
                tsne_data = session_centroids_high_dim[tsne_indices]
            else:
                tsne_data = session_centroids_high_dim
            
            # Check timeout before t-SNE
            if time.time() - start_time > max_computation_time * 0.7:
                st.warning("⚠️ Skipping t-SNE due to time constraints - using PCA only")
                hybrid_coords = pca_session_centroids
            else:
                try:
                    # ULTRA-FAST t-SNE parameters
                    with st.spinner("⚡ Fast t-SNE computation..."):
                        tsne = TSNE(
                            n_components=3, 
                            random_state=42, 
                            perplexity=min(5, len(tsne_data)//3),  # Much smaller perplexity
                            max_iter=250,  # Reduced iterations
                            learning_rate=500,  # Faster learning
                            early_exaggeration=4  # Faster convergence
                        )
                        tsne_result = tsne.fit_transform(tsne_data)
                    
                    # 3. Quick hybrid combination
                    alpha = 0.8  # More weight to faster PCA
                    
                    # Simple normalization
                    pca_norm = pca_session_centroids / np.std(pca_session_centroids)
                    tsne_norm = tsne_result / np.std(tsne_result)
                    
                    # Direct interpolation if needed
                    if len(tsne_norm) != len(pca_norm):
                        from scipy.interpolate import interp1d
                        if len(tsne_norm) > 1:
                            t_old = np.linspace(0, 1, len(tsne_norm))
                            t_new = np.linspace(0, 1, len(pca_norm))
                            tsne_interp = np.column_stack([
                                interp1d(t_old, tsne_norm[:, 0], kind='linear')(t_new),
                                interp1d(t_old, tsne_norm[:, 1], kind='linear')(t_new),
                                interp1d(t_old, tsne_norm[:, 2], kind='linear')(t_new)
                            ])
                            hybrid_coords = alpha * pca_norm + (1 - alpha) * tsne_interp
                        else:
                            hybrid_coords = pca_norm
                    else:
                        hybrid_coords = alpha * pca_norm + (1 - alpha) * tsne_norm
                        
                except Exception as e:
                    st.warning(f"⚠️ t-SNE failed ({str(e)}) - using PCA only")
                    hybrid_coords = pca_session_centroids
        
        # Check final timeout
        if time.time() - start_time > max_computation_time:
            st.warning("⚠️ Computation timeout - using simplified tunnel")
            return create_simple_tunnel_fallback(hybrid_coords, session_metadata)
        
        # OPTIMIZATION 5: Simplified tunnel creation
        return create_optimized_tunnel_mesh(hybrid_coords, session_metadata, tunnel_segments, tunnel_radius_base)
        
    except Exception as e:
        st.error(f"❌ Liminal tunnel creation failed: {str(e)}")
        return None

def create_simple_tunnel_fallback(coords, metadata):
    """Create a simple line-based tunnel for when full mesh fails."""
    fig = go.Figure()
    
    # Simple line path
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1], 
        z=coords[:, 2] if coords.shape[1] > 2 else np.zeros(len(coords)),
        mode='lines+markers',
        line=dict(color='purple', width=8),
        marker=dict(size=10, color='yellow'),
        name='Simplified Tunnel Path',
        hovertemplate="<b>Session %{text}</b><br>Simplified tunnel path<extra></extra>",
        text=[f"S{i+1}" for i in range(len(coords))]
    ))
    
    fig.update_layout(
        title="⚡ Simplified Tunnel (Performance Mode)",
        scene=dict(bgcolor='rgba(5,5,15,0.95)'),
        height=600
    )
    
    return fig

def create_optimized_tunnel_mesh(hybrid_coords, session_metadata, tunnel_segments, tunnel_radius_base):
    """Create optimized tunnel mesh with reduced complexity."""
    if len(hybrid_coords) < 3:
        return create_simple_tunnel_fallback(hybrid_coords, session_metadata)
    
    # OPTIMIZATION 6: Linear interpolation instead of splines
    t_original = np.linspace(0, 1, len(hybrid_coords))
    t_smooth = np.linspace(0, 1, tunnel_segments)
    
    # Simple linear interpolation (faster than cubic splines)
    tunnel_x = np.interp(t_smooth, t_original, hybrid_coords[:, 0])
    tunnel_y = np.interp(t_smooth, t_original, hybrid_coords[:, 1])
    tunnel_z = np.interp(t_smooth, t_original, hybrid_coords[:, 2])
    
    # OPTIMIZATION 7: Simplified tube with fewer radial segments
    n_radial = 8  # Reduced from 16
    theta = np.linspace(0, 2*np.pi, n_radial, endpoint=False)
    
    tube_x, tube_y, tube_z = [], [], []
    tube_colors = []
    
    for i in range(len(tunnel_x)):
        # Simple perpendicular vectors (less computation)
        progress = i / (len(tunnel_x) - 1)
        radius = tunnel_radius_base * (1 + 0.2 * progress)  # Simple radius variation
        
        # Basic circular cross-section in XY plane
        for angle in theta:
            tube_x.append(tunnel_x[i] + radius * np.cos(angle))
            tube_y.append(tunnel_y[i] + radius * np.sin(angle))
            tube_z.append(tunnel_z[i])
            tube_colors.append(progress)
    
    # OPTIMIZATION 8: No animation frames - static visualization
    fig = go.Figure()
    
    # Main tunnel path
    fig.add_trace(go.Scatter3d(
        x=tunnel_x,
        y=tunnel_y,
        z=tunnel_z,
        mode='lines',
        line=dict(color='purple', width=6),
        name='Tunnel Centerline',
        hovertemplate="<b>Tunnel Path</b><br>Progress: %{customdata:.1%}<extra></extra>",
        customdata=[i/len(tunnel_x) for i in range(len(tunnel_x))]
    ))
    
    # Session markers
    fig.add_trace(go.Scatter3d(
        x=hybrid_coords[:, 0],
        y=hybrid_coords[:, 1],
        z=hybrid_coords[:, 2],
        mode='markers+text',
        marker=dict(size=12, color='yellow', symbol='diamond'),
        text=[f"S{i+1}" for i in range(len(hybrid_coords))],
        textposition='top center',
        name='Session Anchors',
        hovertemplate="<b>%{text}</b><br>%{customdata}<extra></extra>",
        customdata=[m['text'] for m in session_metadata]
    ))
    
    fig.update_layout(
        title="⚡ Optimized Liminal Tunnel (Fast Mode)",
        scene=dict(
            bgcolor='rgba(5,5,15,0.95)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        height=600
    )
    
    return fig


def plot_enhanced_ridgeline_altair(memory, meta, show_trends=True, highlight_changes=True):
    """Create enhanced ridgeline plot with improved readability and interpretation aids."""
    from semantic_tensor_analysis.visualization.viz.pca_plot import prepare_for_pca
    
    try:
        # Prepare data for dimensionality reduction
        flat, session_ids, token_ids = prepare_for_pca(memory)
        
        # Robust data validation
        if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
            return None, None
        flat = (flat - np.mean(flat, axis=0)) / np.std(flat, axis=0)
        if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
            return None, None
        
        # Calculate multiple dimensionality reduction techniques
        pca = PCA(n_components=3)
        pca_reduced = pca.fit_transform(flat)
        
        # Calculate t-SNE components (with smaller dataset for speed)
        n_samples = min(1000, flat.shape[0])
        if flat.shape[0] > n_samples:
            indices = np.random.choice(flat.shape[0], n_samples, replace=False)
            tsne_flat = flat[indices]
            tsne_session_ids = session_ids[indices]
        else:
            tsne_flat = flat
            tsne_session_ids = session_ids
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_flat)//4))
        tsne_reduced = tsne.fit_transform(tsne_flat)
        
        # Generate session embeddings for meaningful labeling
        session_embeddings = []
        session_metadata = []
        
        for idx, tensor in enumerate(memory):
            session_emb = to_cpu_numpy(tensor.mean(0))
            session_embeddings.append(session_emb)
            session_text = meta[idx].get('text', f'Session {idx+1}')
            session_metadata.append({
                'session_idx': idx,
                'text': session_text,
                'session_name': f'Session {idx+1}',
                'token_count': tensor.shape[0]
            })
        
        session_embeddings = np.array(session_embeddings)
        
        # Apply PCA to session embeddings for labeling
        pca_for_labels = PCA(n_components=3)
        session_pca = pca_for_labels.fit_transform(session_embeddings)
        
        # Generate meaningful labels
        from semantic_tensor_analysis.streamlit.utils import generate_dynamic_axis_labels
        try:
            pca_labels = generate_dynamic_axis_labels(
                session_pca,
                session_metadata, 
                pca_for_labels
            )
        except:
            pca_labels = [
                f"Semantic Dimension 1 ({pca.explained_variance_ratio_[0]:.1%})",
                f"Semantic Dimension 2 ({pca.explained_variance_ratio_[1]:.1%})",
                f"Semantic Dimension 3 ({pca.explained_variance_ratio_[2]:.1%})"
            ]
        
        # Calculate session-wise statistics for enhanced analysis
        ridgeline_data = []
        unique_sessions = sorted(np.unique(session_ids))
        
        # Define features with more descriptive information
        features = [
            {
                'name': pca_labels[0],
                'values': pca_reduced[:, 0], 
                'session_ids': session_ids,
                'type': 'Primary Semantic',
                'variance': pca.explained_variance_ratio_[0],
                'order': 0
            },
            {
                'name': pca_labels[1],
                'values': pca_reduced[:, 1], 
                'session_ids': session_ids,
                'type': 'Secondary Semantic',
                'variance': pca.explained_variance_ratio_[1],
                'order': 1
            },
            {
                'name': pca_labels[2],
                'values': pca_reduced[:, 2], 
                'session_ids': session_ids,
                'type': 'Tertiary Semantic',
                'variance': pca.explained_variance_ratio_[2],
                'order': 2
            },
            {
                'name': "Non-linear Pattern A",
                'values': tsne_reduced[:, 0], 
                'session_ids': tsne_session_ids,
                'type': 'Non-linear',
                'variance': 0,  # t-SNE doesn't have explained variance
                'order': 3
            },
            {
                'name': "Non-linear Pattern B",
                'values': tsne_reduced[:, 1], 
                'session_ids': tsne_session_ids,
                'type': 'Non-linear',
                'variance': 0,
                'order': 4
            }
        ]
        
        # Calculate session statistics and trends
        session_stats = {}
        for session_idx in unique_sessions:
            session_stats[session_idx] = {
                'session_name': f'Session {session_idx + 1}',
                'text_preview': meta[session_idx]['text'][:100] + "...",
                'token_count': meta[session_idx].get('tokens', len(meta[session_idx]['text']))
            }
        
        # Generate enhanced ridgeline data with statistical insights
        for feature in features:
            feature_name = feature['name']
            feature_values = feature['values']
            feature_session_ids = feature['session_ids']
            feature_type = feature['type']
            feature_variance = feature['variance']
            feature_order = feature['order']
            
            # Calculate session-wise trends
            session_means = []
            session_sessions = []
            
            for session_idx in unique_sessions:
                session_mask = feature_session_ids == session_idx
                if np.sum(session_mask) > 0:
                    session_values = feature_values[session_mask]
                    session_mean = np.mean(session_values)
                    session_std = np.std(session_values)
                    
                    session_means.append(session_mean)
                    session_sessions.append(session_idx)
                    
                    # Create distribution points for visualization
                    n_points = 50  # More points for smoother curves
                    
                    # Adaptive scaling based on data range
                    if session_std > 0:
                        # Use 2.5 standard deviations for better distribution visualization
                        value_range = np.linspace(-2.5*session_std, 2.5*session_std, n_points)
                        max_density = 1.0 / (session_std * np.sqrt(2 * np.pi))  # Proper Gaussian normalization
                    else:
                        value_range = np.linspace(-0.1, 0.1, n_points)  # Small range for zero variance
                        max_density = 1.0
                    
                    for i, offset in enumerate(value_range):
                        if session_std > 0:
                            # Proper Gaussian density calculation
                            density = np.exp(-0.5 * (offset/session_std)**2) / (session_std * np.sqrt(2 * np.pi))
                            # Normalize density to 0-1 range for better visualization
                            normalized_density = density / max_density
                        else:
                            normalized_density = 1.0 if abs(offset) < 0.05 else 0.0
                        
                        ridgeline_data.append({
                            'Session': session_idx + 1,
                            'Feature': feature_name,
                            'Value': session_mean + offset,
                            'Density': normalized_density,
                            'FeatureOrder': feature_order,
                            'FeatureType': feature_type,
                            'FeatureVariance': f"{feature_variance:.1%}" if feature_variance > 0 else "N/A",
                            'SessionMean': session_mean,
                            'SessionStd': session_std,
                            'SessionName': session_stats[session_idx]['session_name'],
                            'TextPreview': session_stats[session_idx]['text_preview'],
                            'TokenCount': session_stats[session_idx]['token_count'],
                            'DistributionPoint': i
                        })
            
            # Calculate trend information if requested
            if show_trends and len(session_means) > 3:
                # Simple linear trend
                x = np.array(session_sessions)
                y = np.array(session_means)
                
                # Calculate correlation coefficient for trend strength
                if len(x) > 1:
                    correlation = np.corrcoef(x, y)[0, 1] if not np.isnan(np.corrcoef(x, y)[0, 1]) else 0
                    
                    # Add trend information to the data
                    for i, session_idx in enumerate(session_sessions):
                        # Find all records for this session and feature
                        session_feature_mask = [
                            (d['Session'] == session_idx + 1) and (d['Feature'] == feature_name) 
                            for d in ridgeline_data
                        ]
                        
                        for j, is_match in enumerate(session_feature_mask):
                            if is_match:
                                ridgeline_data[j]['TrendCorrelation'] = correlation
                                ridgeline_data[j]['TrendDirection'] = 'Increasing' if correlation > 0.3 else 'Decreasing' if correlation < -0.3 else 'Stable'
        
        if len(ridgeline_data) == 0:
            return None, None
            
        df_ridge = pd.DataFrame(ridgeline_data)
        
        # Calculate adaptive Y-axis scaling based on data
        max_density = df_ridge['Density'].max() if not df_ridge.empty else 1.0
        value_range = df_ridge['Value'].max() - df_ridge['Value'].min() if not df_ridge.empty else 1.0
        
        # Adaptive height calculation based on data characteristics
        adaptive_height = max(100, min(200, 120 + value_range * 10))  # Dynamic height between 100-200
        adaptive_y_range = [0, max(80, adaptive_height * 0.8)]  # Adaptive Y range
        
        # Create enhanced Altair ridgeline plot with better visibility
        base = alt.Chart(df_ridge).add_params(
            alt.selection_interval(bind='scales', encodings=['x'])
        )
        
        # Main ridgeline visualization with improved styling
        ridgeline = base.mark_area(
            orient='vertical',
            opacity=0.85,  # Increased opacity for better visibility
            stroke='#ffffff',  # White stroke for contrast
            strokeWidth=1.5,
            interpolate='cardinal'
        ).encode(
            alt.X('Session:O', 
                  title='Session Number',
                  axis=alt.Axis(
                      labelAngle=0, 
                      labelFontSize=14,
                      titleFontSize=16,
                      labelColor='#333333',
                      titleColor='#000000',
                      gridColor='#cccccc'
                  )),
            alt.Y('Value:Q',
                  title='',
                  scale=alt.Scale(range=adaptive_y_range),  # Adaptive Y range
                  axis=alt.Axis(labels=False, ticks=False, grid=False)),
            alt.Row('Feature:N',
                    title='Semantic Features (Hover for Details)',
                    sort=alt.SortField('FeatureOrder', order='ascending'),
                    header=alt.Header(
                        labelAngle=0, 
                        labelAlign='left', 
                        labelPadding=25, 
                        titleOrient='left',
                        labelFontSize=15,
                        labelFontWeight='bold',
                        labelColor='#000000',
                        titleColor='#000000'
                    )),
            alt.Color('FeatureType:N',
                      scale=alt.Scale(
                          domain=['Primary Semantic', 'Secondary Semantic', 'Tertiary Semantic', 'Non-linear'], 
                          range=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Brighter, more contrasting colors
                      ),
                      legend=alt.Legend(
                          title="Feature Type",
                          titleFontSize=16,
                          labelFontSize=14,
                          orient='top-right',
                          titleColor='#000000',
                          labelColor='#000000',
                          fillColor='#f8f9fa',
                          strokeColor='#dee2e6',
                          padding=10
                      )),
            alt.Opacity('Density:Q',
                        scale=alt.Scale(range=[0.3, 1.0]),  # Higher minimum opacity
                        legend=None),
            tooltip=[
                alt.Tooltip('SessionName:N', title='Session'),
                alt.Tooltip('Feature:N', title='Semantic Feature'),
                alt.Tooltip('FeatureType:N', title='Type'),
                alt.Tooltip('FeatureVariance:N', title='Explained Variance'),
                alt.Tooltip('SessionMean:Q', title='Session Mean', format='.3f'),
                alt.Tooltip('SessionStd:Q', title='Session Std Dev', format='.3f'),
                alt.Tooltip('TrendDirection:N', title='Trend'),
                alt.Tooltip('TokenCount:Q', title='Token Count'),
                alt.Tooltip('TextPreview:N', title='Content Preview')
            ]
        ).resolve_scale(
            y='independent'
        ).properties(
            width=700,  # Increased width
            height=adaptive_height,  # Adaptive height
            title=alt.TitleParams(
                text=[
                    "Enhanced Semantic Feature Evolution Across Sessions",
                    "Interactive visualization with content-based labels and trend analysis"
                ],
                anchor='start',
                fontSize=18,
                fontWeight='bold',
                subtitleFontSize=14,
                subtitleColor='#666666',
                color='#000000'
            )
        ).configure_view(
            strokeWidth=0,
            fill='#ffffff'  # White background
        ).configure_axis(
            gridColor='#e0e0e0',
            domainColor='#333333'
        ).configure(
            background='#ffffff',
            padding={"left": 20, "top": 20, "right": 20, "bottom": 20}
        )
        
        # Add annotations for significant changes if requested
        if highlight_changes:
            # Calculate significant changes (sessions with high variance or trend changes)
            significant_sessions = []
            for session in unique_sessions:
                session_data = df_ridge[df_ridge['Session'] == session + 1]
                if not session_data.empty:
                    mean_std = session_data['SessionStd'].mean()
                    if mean_std > session_data['SessionStd'].quantile(0.8):  # Top 20% in variability
                        # Use adaptive positioning based on the Y range
                        annotation_y = session_data['Value'].max() + (adaptive_y_range[1] * 0.1)  # 10% above max
                        significant_sessions.append({
                            'Session': session + 1,
                            'Annotation': '⚡',
                            'Y': annotation_y
                        })
            
            if significant_sessions:
                annotations_df = pd.DataFrame(significant_sessions)
                
                annotations = alt.Chart(annotations_df).mark_text(
                    align='center',
                    baseline='middle',
                    fontSize=18,  # Slightly larger
                    fontWeight='bold',
                    color='#ff2222'  # Even brighter red for visibility
                ).encode(
                    x=alt.X('Session:O'),
                    y=alt.Y('Y:Q', scale=alt.Scale(range=adaptive_y_range)),  # Use same scale
                    text='Annotation:N'
                )
                
                ridgeline = ridgeline + annotations
        
        # Get scaling information for user feedback
        scaling_info = get_ridgeline_scaling_info(df_ridge, adaptive_height, adaptive_y_range)
        
        return ridgeline, scaling_info
        
    except Exception as e:
        st.error(f"Error creating enhanced ridgeline plot: {e}")
        return None, None


def get_ridgeline_scaling_info(df_ridge, adaptive_height, adaptive_y_range):
    """Get scaling information for the ridgeline plot."""
    if df_ridge.empty:
        return None
    
    return {
        'sessions': df_ridge['Session'].nunique(),
        'features': df_ridge['Feature'].nunique(),
        'value_range': f"{df_ridge['Value'].min():.2f} to {df_ridge['Value'].max():.2f}",
        'adaptive_height': adaptive_height,
        'y_range': f"{adaptive_y_range[0]} to {adaptive_y_range[1]:.0f}",
        'max_density': f"{df_ridge['Density'].max():.3f}"
    }


def plot_ridgeline_plotly(memory, meta, show_trends=True, highlight_changes=True):
    """Create ridgeline plot using Plotly for better visibility and control."""
    from semantic_tensor_analysis.visualization.viz.pca_plot import prepare_for_pca
    
    try:
        # Prepare data for dimensionality reduction
        flat, session_ids, token_ids = prepare_for_pca(memory)
        
        # Robust data validation
        if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
            return None
        flat = (flat - np.mean(flat, axis=0)) / np.std(flat, axis=0)
        if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
            return None
        
        # Calculate PCA components
        pca = PCA(n_components=3)
        pca_reduced = pca.fit_transform(flat)
        
        # Calculate t-SNE components (with smaller dataset for speed)
        n_samples = min(1000, flat.shape[0])
        if flat.shape[0] > n_samples:
            indices = np.random.choice(flat.shape[0], n_samples, replace=False)
            tsne_flat = flat[indices]
            tsne_session_ids = session_ids[indices]
        else:
            tsne_flat = flat
            tsne_session_ids = session_ids
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_flat)//4))
        tsne_reduced = tsne.fit_transform(tsne_flat)
        
        # Generate session embeddings for meaningful labeling
        session_embeddings = []
        session_metadata = []
        
        for idx, tensor in enumerate(memory):
            session_emb = to_cpu_numpy(tensor.mean(0))
            session_embeddings.append(session_emb)
            session_text = meta[idx].get('text', f'Session {idx+1}')
            session_metadata.append({
                'session_idx': idx,
                'text': session_text,
                'session_name': f'Session {idx+1}',
                'token_count': tensor.shape[0]
            })
        
        session_embeddings = np.array(session_embeddings)
        
        # Apply PCA to session embeddings for labeling
        pca_for_labels = PCA(n_components=3)
        session_pca = pca_for_labels.fit_transform(session_embeddings)
        
        # Generate meaningful labels
        from semantic_tensor_analysis.streamlit.utils import generate_dynamic_axis_labels
        try:
            pca_labels = generate_dynamic_axis_labels(
                session_pca,
                session_metadata, 
                pca_for_labels
            )
        except:
            pca_labels = [
                f"Semantic Dimension 1 ({pca.explained_variance_ratio_[0]:.1%})",
                f"Semantic Dimension 2 ({pca.explained_variance_ratio_[1]:.1%})",
                f"Semantic Dimension 3 ({pca.explained_variance_ratio_[2]:.1%})"
            ]
        
        # Calculate session-wise statistics for each feature
        unique_sessions = sorted(np.unique(session_ids))
        
        # Define features with more descriptive information
        features = [
            {
                'name': pca_labels[0],
                'values': pca_reduced[:, 0], 
                'session_ids': session_ids,
                'type': 'Primary Semantic',
                'variance': pca.explained_variance_ratio_[0],
                'color': '#1f77b4'
            },
            {
                'name': pca_labels[1],
                'values': pca_reduced[:, 1], 
                'session_ids': session_ids,
                'type': 'Secondary Semantic',
                'variance': pca.explained_variance_ratio_[1],
                'color': '#ff7f0e'
            },
            {
                'name': pca_labels[2],
                'values': pca_reduced[:, 2], 
                'session_ids': session_ids,
                'type': 'Tertiary Semantic',
                'variance': pca.explained_variance_ratio_[2],
                'color': '#2ca02c'
            },
            {
                'name': "Non-linear Pattern A",
                'values': tsne_reduced[:, 0], 
                'session_ids': tsne_session_ids,
                'type': 'Non-linear',
                'variance': 0,
                'color': '#d62728'
            },
            {
                'name': "Non-linear Pattern B",
                'values': tsne_reduced[:, 1], 
                'session_ids': tsne_session_ids,
                'type': 'Non-linear',
                'variance': 0,
                'color': '#9467bd'
            }
        ]
        
        # Create Plotly figure with subplots
        from plotly.subplots import make_subplots
        fig = make_subplots(
            rows=len(features), 
            cols=1,
            subplot_titles=[f"{f['name']} ({f['type']})" for f in features],
            vertical_spacing=0.08,
            shared_xaxes=True
        )
        
        # Calculate trends and generate ridgeline data for each feature
        for feature_idx, feature in enumerate(features):
            feature_name = feature['name']
            feature_values = feature['values']
            feature_session_ids = feature['session_ids']
            feature_color = feature['color']
            
            # Calculate session-wise statistics
            session_means = []
            session_stds = []
            session_sessions = []
            
            for session_idx in unique_sessions:
                session_mask = feature_session_ids == session_idx
                if np.sum(session_mask) > 0:
                    session_values = feature_values[session_mask]
                    session_mean = np.mean(session_values)
                    session_std = np.std(session_values)
                    
                    session_means.append(session_mean)
                    session_stds.append(session_std)
                    session_sessions.append(session_idx + 1)
            
            # Create ridgeline curves for each session
            for i, (session, mean, std) in enumerate(zip(session_sessions, session_means, session_stds)):
                if std > 0:
                    # Create smooth curve
                    x_curve = np.linspace(mean - 3*std, mean + 3*std, 100)
                    y_curve = np.exp(-0.5 * ((x_curve - mean) / std) ** 2)
                    
                    # Normalize and scale for visibility
                    y_curve = y_curve / np.max(y_curve) * 0.8  # Scale to fit nicely
                    
                    # Add curve
                    fig.add_trace(
                        go.Scatter(
                            x=x_curve,
                            y=y_curve + i * 0.9,  # Offset each session
                            mode='lines',
                            fill='tonexty' if i > 0 else 'tozeroy',
                            fillcolor=f"rgba({int(feature_color[1:3], 16)}, {int(feature_color[3:5], 16)}, {int(feature_color[5:7], 16)}, 0.6)",
                            line=dict(color=feature_color, width=2),
                            name=f"Session {session}",
                            showlegend=False,
                            hovertemplate=(
                                f"<b>Session {session}</b><br>"
                                f"Feature: {feature_name}<br>"
                                f"Mean: {mean:.3f}<br>"
                                f"Std: {std:.3f}<br>"
                                f"Type: {feature['type']}<br>"
                                f"Variance: {feature['variance']:.1%}" if feature['variance'] > 0 else "N/A"
                                "<extra></extra>"
                            )
                        ),
                        row=feature_idx + 1,
                        col=1
                    )
                    
                    # Add session labels
                    fig.add_annotation(
                        x=mean,
                        y=i * 0.9 + 0.4,
                        text=str(session),
                        showarrow=False,
                        font=dict(size=10, color="black"),
                        row=feature_idx + 1,
                        col=1
                    )
                    
                    # Add lightning bolt for high variability sessions
                    if highlight_changes and std > np.percentile(session_stds, 80):
                        fig.add_annotation(
                            x=mean,
                            y=i * 0.9 + 0.7,
                            text="⚡",
                            showarrow=False,
                            font=dict(size=16, color="red"),
                            row=feature_idx + 1,
                            col=1
                        )
            
            # Calculate and show trend if requested
            if show_trends and len(session_means) > 3:
                correlation = np.corrcoef(session_sessions, session_means)[0, 1] if len(session_sessions) > 1 else 0
                if not np.isnan(correlation):
                    trend_direction = 'Increasing' if correlation > 0.3 else 'Decreasing' if correlation < -0.3 else 'Stable'
                    trend_line_color = 'green' if correlation > 0.3 else 'red' if correlation < -0.3 else 'gray'
                    
                    # Add trend line
                    fig.add_trace(
                        go.Scatter(
                            x=[min(session_sessions), max(session_sessions)],
                            y=[0, len(session_sessions) * 0.9],
                            mode='lines',
                            line=dict(color=trend_line_color, width=2, dash='dash'),
                            name=f"Trend: {trend_direction}",
                            showlegend=True if feature_idx == 0 else False,
                            hovertemplate=f"Trend: {trend_direction}<br>Correlation: {correlation:.3f}<extra></extra>"
                        ),
                        row=feature_idx + 1,
                        col=1
                    )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Enhanced Semantic Feature Evolution Across Sessions<br><sub>Plotly-based ridgeline with improved visibility</sub>",
                font=dict(size=20),
                x=0.5
            ),
            height=200 * len(features),  # Dynamic height based on number of features
            showlegend=True,
            hovermode='closest',
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update all y-axes to hide ticks and labels
        for i in range(len(features)):
            fig.update_yaxes(
                showticklabels=False,
                showgrid=False,
                zeroline=False,
                row=i + 1,
                col=1
            )
        
        # Update x-axis for bottom subplot only
        fig.update_xaxes(
            title_text="Feature Values",
            showgrid=True,
            gridcolor='lightgray',
            row=len(features),
            col=1
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating Plotly ridgeline plot: {e}")
        return None


def plot_ridgeline_altair(memory, meta):
    """Wrapper function to maintain compatibility - calls enhanced version."""
    return plot_enhanced_ridgeline_altair(memory, meta) 
