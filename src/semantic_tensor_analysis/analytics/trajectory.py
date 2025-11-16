"""Semantic trajectory analysis for Semantic Tensor Memory.

This module contains functions for analyzing semantic trajectories, velocity,
acceleration, and creating both 3D Plotly and multi-view Altair visualizations.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import altair as alt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from semantic_tensor_analysis.streamlit.utils import generate_dynamic_axis_labels


def calculate_semantic_trajectory_data(memory, meta):
    """Calculate semantic trajectory data including velocity and acceleration."""
    # Calculate session embeddings (mean of all tokens per session)
    session_embeddings = []
    session_metadata = []
    
    for idx, tensor in enumerate(memory):
        session_emb = tensor.mean(0).numpy()  # Mean embedding for the session
        session_embeddings.append(session_emb)
        session_text = meta[idx].get('text', f'Session {idx+1}')
        session_metadata.append({
            'session_idx': idx,
            'text': session_text,
            'session_name': f'Session {idx+1}',
            'token_count': tensor.shape[0]
        })
    
    session_embeddings = np.array(session_embeddings)
    
    # Robust data validation
    if np.isnan(session_embeddings).any() or np.isinf(session_embeddings).any():
        return None
    
    # PCA reduction to 3D using improved pipeline
    scaler = StandardScaler()
    session_embeddings_scaled = scaler.fit_transform(session_embeddings)
    
    # Apply PCA
    pca = PCA(n_components=3, random_state=42)
    embeddings_3d = pca.fit_transform(session_embeddings_scaled)
    
    # Check PCA quality
    explained_var_ratio = pca.explained_variance_ratio_
    cumulative_var = np.sum(explained_var_ratio)
    
    if cumulative_var < 0.5:
        st.info(f"ℹ️ PCA explains {cumulative_var:.1%} of variance in session-level data. This is normal for high-dimensional semantic data.")
    
    # Generate dynamic axis labels
    axis_labels = generate_dynamic_axis_labels(embeddings_3d, session_metadata, pca)
    
    # Calculate semantic velocity (rate of change between sessions)
    semantic_velocity = np.zeros(len(embeddings_3d))
    for i in range(1, len(embeddings_3d)):
        semantic_velocity[i] = np.linalg.norm(embeddings_3d[i] - embeddings_3d[i-1])
    
    # Calculate semantic acceleration (change in velocity)
    semantic_acceleration = np.zeros(len(semantic_velocity))
    for i in range(1, len(semantic_velocity)):
        semantic_acceleration[i] = abs(semantic_velocity[i] - semantic_velocity[i-1])
    
    # Identify significant semantic shifts (high velocity)
    velocity_threshold = np.mean(semantic_velocity) + np.std(semantic_velocity)
    significant_shifts = semantic_velocity > velocity_threshold
    
    return {
        'embeddings_3d': embeddings_3d,
        'session_metadata': session_metadata,
        'semantic_velocity': semantic_velocity,
        'semantic_acceleration': semantic_acceleration,
        'significant_shifts': significant_shifts,
        'axis_labels': axis_labels,
        'explained_var_ratio': explained_var_ratio,
        'cumulative_var': cumulative_var
    }


def create_3d_trajectory_plot(trajectory_data):
    """Create 3D trajectory plot using Plotly."""
    embeddings_3d = trajectory_data['embeddings_3d']
    session_metadata = trajectory_data['session_metadata']
    semantic_velocity = trajectory_data['semantic_velocity']
    semantic_acceleration = trajectory_data['semantic_acceleration']
    significant_shifts = trajectory_data['significant_shifts']
    axis_labels = trajectory_data['axis_labels']
    
    # Create colors based on semantic velocity
    colors = []
    for i, vel in enumerate(semantic_velocity):
        if significant_shifts[i]:
            colors.append('red')  # High semantic change
        elif vel > np.mean(semantic_velocity):
            colors.append('orange')  # Medium semantic change
        else:
            colors.append('blue')  # Low semantic change
    
    # Create hover text with semantic insights
    hover_text = []
    for i, meta in enumerate(session_metadata):
        text_preview = meta['text'][:100] + ('...' if len(meta['text']) > 100 else '')
        hover_info = [
            f"<b>{meta['session_name']}</b>",
            f"Semantic Velocity: {semantic_velocity[i]:.3f}",
            f"Semantic Acceleration: {semantic_acceleration[i]:.3f}",
            f"Token Count: {meta['token_count']}",
            f"Text: {text_preview}"
        ]
        hover_text.append("<br>".join(hover_info))
    
    # Create 3D trajectory plot
    fig = go.Figure()
    
    # Add trajectory line connecting sessions
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='lines',
        line=dict(width=3, color='gray'),
        name='Semantic Path',
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # Add session points
    fig.add_trace(go.Scatter3d(
        x=embeddings_3d[:, 0],
        y=embeddings_3d[:, 1],
        z=embeddings_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=8,
            color=colors,
            opacity=0.8,
            line=dict(width=2, color='white')
        ),
        text=[f"S{i+1}" for i in range(len(embeddings_3d))],
        textposition='top center',
        hovertext=hover_text,
        hoverinfo='text',
        name='Sessions'
    ))
    
    # Highlight significant semantic shifts with arrows
    for i in range(1, len(embeddings_3d)):
        if significant_shifts[i]:
            # Add arrow showing direction of significant change
            start = embeddings_3d[i-1]
            end = embeddings_3d[i]
            fig.add_trace(go.Scatter3d(
                x=[start[0], end[0]],
                y=[start[1], end[1]],
                z=[start[2], end[2]],
                mode='lines',
                line=dict(width=6, color='red'),
                name=f'Semantic Shift {i}→{i+1}',
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Update layout with dynamic labels
    fig.update_layout(
        title='Semantic Trajectory: Evolution of Meaning Across Sessions',
        scene=dict(
            xaxis_title=f'<b>{axis_labels[0]}</b><br><span style="font-size:12px; color:gray">Primary semantic axis</span>',
            yaxis_title=f'<b>{axis_labels[1]}</b><br><span style="font-size:12px; color:gray">Secondary semantic axis</span>',
            zaxis_title=f'<b>{axis_labels[2]}</b><br><span style="font-size:12px; color:gray">Tertiary semantic axis</span>',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=600
    )
    
    return fig


def create_altair_semantic_trajectory(embeddings_3d, session_metadata, semantic_velocity, semantic_acceleration, significant_shifts, axis_labels):
    """Create coordinated multi-view semantic trajectory visualization using Altair."""
    try:
        # Prepare data for Altair - ensure all values are JSON serializable
        trajectory_data = []
        for i, meta in enumerate(session_metadata):
            shift_type = "High" if significant_shifts[i] else ("Medium" if semantic_velocity[i] > np.mean(semantic_velocity) else "Low")
            trajectory_data.append({
                'Session': int(i + 1),
                'SessionName': str(meta['session_name']),
                'Dim1': float(embeddings_3d[i, 0]),
                'Dim2': float(embeddings_3d[i, 1]), 
                'Dim3': float(embeddings_3d[i, 2]),
                'Velocity': float(semantic_velocity[i]),
                'Acceleration': float(semantic_acceleration[i]),
                'ShiftType': str(shift_type),
                'Tokens': int(meta['token_count']),
                'Text': str(meta['text'][:100] + ('...' if len(meta['text']) > 100 else ''))
            })
        
        df = pd.DataFrame(trajectory_data)
        
        # Check if data is valid
        if df.empty or len(df) < 2:
            return None
            
        # Simple color scale
        color_scale = alt.Scale(domain=['Low', 'Medium', 'High'], 
                               range=['blue', 'orange', 'red'])
        
        # Create individual charts - each with its own data copy to avoid reference issues
        
        # 1. Time series for dimension 1
        dim1_chart = alt.Chart(df.copy()).mark_line(point=True).encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Dim1:Q', title=axis_labels[0]),
            color=alt.Color('ShiftType:N', scale=color_scale),
            tooltip=['SessionName:N', 'Dim1:Q', 'Velocity:Q']
        ).properties(width=180, height=120, title=f"{axis_labels[0]}")
        
        # 2. Time series for dimension 2
        dim2_chart = alt.Chart(df.copy()).mark_line(point=True).encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Dim2:Q', title=axis_labels[1]),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Dim2:Q', 'Velocity:Q']
        ).properties(width=180, height=120, title=f"{axis_labels[1]}")
        
        # 3. Time series for dimension 3
        dim3_chart = alt.Chart(df.copy()).mark_line(point=True).encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Dim3:Q', title=axis_labels[2]),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Dim3:Q', 'Velocity:Q']
        ).properties(width=180, height=120, title=f"{axis_labels[2]}")
        
        # 4. Scatter plot XY
        scatter_xy = alt.Chart(df.copy()).mark_circle(size=80).encode(
            x=alt.X('Dim1:Q', title=axis_labels[0]),
            y=alt.Y('Dim2:Q', title=axis_labels[1]),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Dim1:Q', 'Dim2:Q', 'Velocity:Q']
        ).properties(width=180, height=180, title="Semantic Space (XY)")
        
        # 5. Velocity chart
        velocity_chart = alt.Chart(df.copy()).mark_bar().encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Velocity:Q', title='Semantic Velocity'),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Velocity:Q']
        ).properties(width=180, height=120, title="Semantic Velocity")
        
        # 6. Acceleration chart
        acceleration_chart = alt.Chart(df.copy()).mark_bar().encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Acceleration:Q', title='Semantic Acceleration'),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Acceleration:Q']
        ).properties(width=180, height=120, title="Semantic Acceleration")
        
        # Create simple layout without complex concatenation
        charts = [
            dim1_chart,
            dim2_chart, 
            dim3_chart,
            scatter_xy,
            velocity_chart,
            acceleration_chart
        ]
        
        return charts  # Return list of charts instead of complex concatenation
        
    except Exception as e:
        st.error(f"Error creating Altair dashboard: {e}")
        return None


def display_trajectory_analysis_table(trajectory_data):
    """Display session-by-session analysis table."""
    session_metadata = trajectory_data['session_metadata']
    semantic_velocity = trajectory_data['semantic_velocity']
    semantic_acceleration = trajectory_data['semantic_acceleration']
    significant_shifts = trajectory_data['significant_shifts']
    
    trajectory_table_data = []
    for i, meta in enumerate(session_metadata):
        shift_type = "🔴 High" if significant_shifts[i] else ("🟠 Medium" if semantic_velocity[i] > np.mean(semantic_velocity) else "🔵 Low")
        trajectory_table_data.append({
            "Session": meta['session_name'],
            "Semantic Velocity": f"{semantic_velocity[i]:.3f}",
            "Semantic Acceleration": f"{semantic_acceleration[i]:.3f}",
            "Change Level": shift_type,
            "Tokens": meta['token_count']
        })
    
    return pd.DataFrame(trajectory_table_data)


def get_trajectory_context_for_chat(trajectory_data):
    """Prepare trajectory context for LLM analysis."""
    session_metadata = trajectory_data['session_metadata']
    semantic_velocity = trajectory_data['semantic_velocity']
    significant_shifts = trajectory_data['significant_shifts']
    axis_labels = trajectory_data['axis_labels']
    explained_var_ratio = trajectory_data['explained_var_ratio']
    
    significant_sessions = [i+1 for i, shift in enumerate(significant_shifts) if shift]
    
    return {
        'avg_velocity': f"{np.mean(semantic_velocity):.3f}",
        'max_shift': f"{np.max(semantic_velocity):.3f}",
        'significant_shifts': significant_sessions,
        'total_sessions': len(session_metadata),
        'explained_variance': ', '.join([f'{v:.2%}' for v in explained_var_ratio]),
        'axis_labels': axis_labels
    } 
