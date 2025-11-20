#!/usr/bin/env python3
"""
Temporal Multi-Resolution Visualizations

Enhanced visualizations that work with the multi-resolution temporal system.
These visualizations can zoom between turn-level, conversation-level, and day-level views.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import torch
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional

from semantic_tensor_analysis.app.temporal_resolution_manager import TemporalResolution


def create_temporal_semantic_flow(temporal_manager, resolution: TemporalResolution = TemporalResolution.TURN):
    """Create an interactive temporal semantic flow visualization."""
    
    resolution_data = temporal_manager.zoom_to_resolution(resolution)
    
    if not resolution_data['embeddings']:
        st.warning("No data available for this resolution")
        return None
    
    embeddings = torch.stack(resolution_data['embeddings'])
    metadata = resolution_data['metadata']
    
    # Apply PCA for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    coords_2d = pca.fit_transform(embeddings.numpy())
    
    # Create timeline data
    timeline_data = []
    for i, (coords, meta) in enumerate(zip(coords_2d, metadata)):
        
        if resolution == TemporalResolution.TURN:
            hover_text = f"Turn {meta.get('turn_index', i)}<br>"
            hover_text += f"Conversation: {meta.get('conversation_id', 'unknown')}<br>"
            hover_text += f"Text: {meta.get('user_text', '')[:100]}..."
            timestamp = meta.get('timestamp')
            size = 8
            color = meta.get('turn_index', i)
        
        elif resolution == TemporalResolution.CONVERSATION:
            hover_text = f"Conversation {meta.get('session_id', i)}<br>"
            hover_text += f"Turns: {meta.get('total_turns', 0)}<br>"
            hover_text += f"Duration: {meta.get('duration_minutes', 0):.1f} min"
            timestamp = meta.get('start_time')
            size = min(max(meta.get('total_turns', 1) * 2, 10), 30)
            color = meta.get('total_turns', 1)
        
        elif resolution == TemporalResolution.DAY:
            hover_text = f"Date: {meta.get('date', 'unknown')}<br>"
            hover_text += f"Conversations: {meta.get('total_conversations', 0)}<br>"
            hover_text += f"Messages: {meta.get('total_messages', 0)}<br>"
            hover_text += f"Diversity: {meta.get('semantic_diversity_score', 0):.3f}"
            timestamp = meta.get('date')
            size = min(max(meta.get('total_conversations', 1) * 3, 15), 50)
            color = meta.get('semantic_diversity_score', 0)
        
        timeline_data.append({
            'index': i,
            'x': coords[0],
            'y': coords[1], 
            'timestamp': timestamp,
            'hover_text': hover_text,
            'size': size,
            'color': color,
            'meta': meta
        })
    
    df = pd.DataFrame(timeline_data)
    
    # Create the plot
    fig = go.Figure()
    
    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='lines',
        line=dict(color='rgba(100,100,100,0.3)', width=2),
        name='Semantic Trajectory',
        hoverinfo='skip'
    ))
    
    # Add points
    fig.add_trace(go.Scatter(
        x=df['x'],
        y=df['y'],
        mode='markers',
        marker=dict(
            size=df['size'],
            color=df['color'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=f"{resolution.value.title()} Index"),
            opacity=0.8
        ),
        text=df['hover_text'],
        hovertemplate='%{text}<extra></extra>',
        name=f'{resolution.value.title()} Points'
    ))
    
    # Add arrow annotations showing direction
    if len(df) > 1:
        # Add arrows every few points to show direction
        arrow_indices = range(0, len(df)-1, max(1, len(df)//10))
        for i in arrow_indices:
            if i + 1 < len(df):
                fig.add_annotation(
                    x=df.iloc[i]['x'],
                    y=df.iloc[i]['y'],
                    ax=df.iloc[i+1]['x'],
                    ay=df.iloc[i+1]['y'],
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="rgba(255,100,100,0.6)",
                    showarrow=True
                )
    
    fig.update_layout(
        title=f"🌊 Semantic Flow - {resolution.value.title()} Level",
        xaxis_title="Semantic Dimension 1",
        yaxis_title="Semantic Dimension 2", 
        width=800,
        height=600,
        showlegend=False
    )
    
    return fig


def create_temporal_heatmap(temporal_manager, resolution: TemporalResolution = TemporalResolution.DAY):
    """Create a temporal heatmap showing semantic activity."""
    
    if resolution not in [TemporalResolution.DAY, TemporalResolution.CONVERSATION]:
        st.warning("Temporal heatmap only available for Day and Conversation levels")
        return None
    
    resolution_data = temporal_manager.zoom_to_resolution(resolution)
    metadata = resolution_data['metadata']
    
    if not metadata:
        st.warning("No temporal data available")
        return None
    
    # Prepare heatmap data
    if resolution == TemporalResolution.DAY:
        # Day-level: Show semantic diversity by date
        heatmap_data = []
        for meta in metadata:
            date_obj = meta.get('date')
            if date_obj:
                heatmap_data.append({
                    'date': date_obj,
                    'day_of_week': date_obj.strftime('%A'),
                    'day': date_obj.day,
                    'month': date_obj.month,
                    'year': date_obj.year,
                    'conversations': meta.get('total_conversations', 0),
                    'messages': meta.get('total_messages', 0),
                    'diversity': meta.get('semantic_diversity_score', 0),
                    'peak_hours': len(meta.get('peak_activity_hours', []))
                })
        
        if not heatmap_data:
            st.warning("No valid date information found")
            return None
        
        df = pd.DataFrame(heatmap_data)
        
        # Create calendar heatmap
        fig = px.density_heatmap(
            df,
            x='day',
            y='month',
            z='diversity',
            title="📅 Daily Semantic Diversity Calendar",
            labels={'diversity': 'Semantic Diversity', 'day': 'Day of Month', 'month': 'Month'},
            color_continuous_scale='Viridis'
        )
        
    elif resolution == TemporalResolution.CONVERSATION:
        # Conversation-level: Show activity patterns by hour and day
        heatmap_data = []
        for meta in metadata:
            start_time = meta.get('start_time')
            if start_time and isinstance(start_time, datetime):
                heatmap_data.append({
                    'hour': start_time.hour,
                    'day_of_week': start_time.strftime('%A'),
                    'duration': meta.get('duration_minutes', 0),
                    'turns': meta.get('total_turns', 0),
                    'messages': meta.get('user_message_count', 0) + meta.get('assistant_message_count', 0)
                })
        
        if not heatmap_data:
            st.warning("No valid timestamp information found")
            return None
        
        df = pd.DataFrame(heatmap_data)
        
        # Group by hour and day of week
        activity_matrix = df.groupby(['day_of_week', 'hour']).agg({
            'duration': 'sum',
            'turns': 'sum',
            'messages': 'sum'
        }).reset_index()
        
        # Pivot for heatmap
        activity_pivot = activity_matrix.pivot(index='day_of_week', columns='hour', values='messages').fillna(0)
        
        # Reorder days
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_pivot = activity_pivot.reindex([d for d in day_order if d in activity_pivot.index])
        
        fig = px.imshow(
            activity_pivot,
            title="🕐 Conversation Activity Heatmap (Messages per Hour)",
            labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Total Messages'},
            color_continuous_scale='Viridis',
            aspect="auto"
        )
    
    return fig


def create_resolution_comparison_dashboard(temporal_manager):
    """Create a comprehensive dashboard comparing all resolutions."""
    
    st.subheader("🔍 Multi-Resolution Comparison Dashboard")
    
    # Get data for all resolutions
    turn_data = temporal_manager.zoom_to_resolution(TemporalResolution.TURN)
    conv_data = temporal_manager.zoom_to_resolution(TemporalResolution.CONVERSATION)
    day_data = temporal_manager.zoom_to_resolution(TemporalResolution.DAY)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔬 Turn Level")
        st.metric("Total Turns", len(turn_data['embeddings']))
        if turn_data['metadata']:
            avg_turn_length = np.mean([len(m.get('user_text', '')) for m in turn_data['metadata']])
            st.metric("Avg Turn Length", f"{avg_turn_length:.0f} chars")
    
    with col2:
        st.markdown("### 💬 Conversation Level")
        st.metric("Total Conversations", len(conv_data['embeddings']))
        if conv_data['metadata']:
            avg_conv_turns = np.mean([m.get('total_turns', 0) for m in conv_data['metadata']])
            st.metric("Avg Turns/Conv", f"{avg_conv_turns:.1f}")
    
    with col3:
        st.markdown("### 📅 Day Level")
        st.metric("Total Days", len(day_data['embeddings']))
        if day_data['metadata']:
            avg_daily_convs = np.mean([m.get('total_conversations', 0) for m in day_data['metadata']])
            st.metric("Avg Convs/Day", f"{avg_daily_convs:.1f}")
    
    # Resolution comparison plots
    st.subheader("📊 Resolution Comparison Visualizations")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Temporal flow comparison
        selected_flow_res = st.selectbox(
            "Semantic Flow Resolution",
            options=["Turn Level", "Conversation Level", "Day Level"],
            key="flow_resolution"
        )
        
        resolution_map = {
            "Turn Level": TemporalResolution.TURN,
            "Conversation Level": TemporalResolution.CONVERSATION,
            "Day Level": TemporalResolution.DAY
        }
        
        flow_fig = create_temporal_semantic_flow(temporal_manager, resolution_map[selected_flow_res])
        if flow_fig:
            st.plotly_chart(flow_fig, use_container_width=True)
    
    with viz_col2:
        # Temporal heatmap
        selected_heatmap_res = st.selectbox(
            "Activity Heatmap Resolution",
            options=["Day Level", "Conversation Level"],
            key="heatmap_resolution"
        )
        
        heatmap_resolution = TemporalResolution.DAY if selected_heatmap_res == "Day Level" else TemporalResolution.CONVERSATION
        heatmap_fig = create_temporal_heatmap(temporal_manager, heatmap_resolution)
        if heatmap_fig:
            st.plotly_chart(heatmap_fig, use_container_width=True)


def create_semantic_evolution_timeline(temporal_manager, focus_date: Optional[date] = None):
    """Create a timeline showing semantic evolution within a specific time period."""
    
    st.subheader("📈 Semantic Evolution Timeline")
    
    if focus_date:
        # Focus on specific date - show conversation and turn evolution
        st.info(f"📅 Focused on: {focus_date}")
        
        # Get conversations for this date
        conv_data = temporal_manager.zoom_to_resolution(TemporalResolution.CONVERSATION, focus_date=focus_date)
        
        if not conv_data['embeddings']:
            st.warning(f"No conversations found for {focus_date}")
            return
        
        # Create timeline of conversations within the day
        timeline_data = []
        for i, (embedding, meta) in enumerate(zip(conv_data['embeddings'], conv_data['metadata'])):
            start_time = meta.get('start_time')
            timeline_data.append({
                'conversation_id': meta.get('session_id', f'conv_{i}'),
                'start_time': start_time,
                'duration': meta.get('duration_minutes', 0),
                'turns': meta.get('total_turns', 0),
                'messages': meta.get('user_message_count', 0) + meta.get('assistant_message_count', 0),
                'embedding': embedding
            })
        
        # Sort by start time
        timeline_data.sort(key=lambda x: x['start_time'] or datetime.min)
        
        # Calculate semantic drift between consecutive conversations
        if len(timeline_data) > 1:
            drifts = []
            for i in range(len(timeline_data) - 1):
                emb1 = timeline_data[i]['embedding']
                emb2 = timeline_data[i + 1]['embedding']
                
                similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
                drift = 1 - similarity
                drifts.append(drift)
            
            # Create drift timeline
            fig = go.Figure()
            
            # Add drift line
            x_labels = [f"Conv {i+1}" for i in range(len(drifts))]
            fig.add_trace(go.Scatter(
                x=x_labels,
                y=drifts,
                mode='lines+markers',
                name='Semantic Drift',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"🌊 Semantic Drift Between Conversations - {focus_date}",
                xaxis_title="Conversation Transitions",
                yaxis_title="Semantic Drift",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show conversation details
            st.subheader("💬 Conversation Details")
            for i, conv in enumerate(timeline_data):
                with st.expander(f"Conversation {i+1}: {conv['start_time'].strftime('%H:%M') if conv['start_time'] else 'Unknown time'}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", f"{conv['duration']:.1f} min")
                    with col2:
                        st.metric("Turns", conv['turns'])
                    with col3:
                        st.metric("Messages", conv['messages'])
    
    else:
        # Show overall daily evolution
        day_data = temporal_manager.zoom_to_resolution(TemporalResolution.DAY)
        
        if len(day_data['embeddings']) < 2:
            st.warning("Need at least 2 days for evolution timeline")
            return
        
        # Calculate daily semantic drift
        daily_drifts = []
        dates = []
        
        for i in range(len(day_data['embeddings']) - 1):
            emb1 = day_data['embeddings'][i]
            emb2 = day_data['embeddings'][i + 1]
            
            similarity = torch.cosine_similarity(emb1, emb2, dim=0).item()
            drift = 1 - similarity
            daily_drifts.append(drift)
            dates.append(day_data['metadata'][i]['date'])
        
        # Create daily evolution plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=daily_drifts,
            mode='lines+markers',
            name='Daily Semantic Drift',
            line=dict(color='blue', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="📈 Daily Semantic Evolution",
            xaxis_title="Date",
            yaxis_title="Semantic Drift",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show daily statistics
        st.subheader("📊 Daily Statistics")
        daily_stats = []
        for meta in day_data['metadata']:
            daily_stats.append({
                'Date': meta['date'],
                'Conversations': meta['total_conversations'],
                'Total Messages': meta['total_messages'],
                'Semantic Diversity': f"{meta['semantic_diversity_score']:.3f}",
                'Peak Hours': ', '.join(map(str, meta.get('peak_activity_hours', [])))
            })
        
        stats_df = pd.DataFrame(daily_stats)
        st.dataframe(stats_df, use_container_width=True)


def render_temporal_visualizations_tab():
    """Render the temporal visualizations tab with multi-resolution support."""
    
    if 'temporal_manager' not in st.session_state:
        st.warning("🔍 Multi-resolution temporal data not available. Upload a conversation file to enable temporal visualizations.")
        return
    
    temporal_manager = st.session_state.temporal_manager
    
    # Visualization selector
    viz_options = [
        "🔍 Resolution Comparison Dashboard",
        "🌊 Semantic Flow",
        "🔥 Activity Heatmap",
        "📈 Evolution Timeline"
    ]
    
    selected_viz = st.selectbox("Choose Visualization", viz_options)
    
    if selected_viz == "🔍 Resolution Comparison Dashboard":
        create_resolution_comparison_dashboard(temporal_manager)
    
    elif selected_viz == "🌊 Semantic Flow":
        col1, col2 = st.columns([3, 1])
        with col2:
            resolution_options = {
                "Turn Level": TemporalResolution.TURN,
                "Conversation Level": TemporalResolution.CONVERSATION,
                "Day Level": TemporalResolution.DAY
            }
            selected_res = st.selectbox("Resolution", list(resolution_options.keys()))
        
        with col1:
            fig = create_temporal_semantic_flow(temporal_manager, resolution_options[selected_res])
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif selected_viz == "🔥 Activity Heatmap":
        col1, col2 = st.columns([3, 1])
        with col2:
            heatmap_options = ["Day Level", "Conversation Level"]
            selected_heatmap = st.selectbox("Heatmap Type", heatmap_options)
        
        with col1:
            resolution = TemporalResolution.DAY if selected_heatmap == "Day Level" else TemporalResolution.CONVERSATION
            fig = create_temporal_heatmap(temporal_manager, resolution)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    elif selected_viz == "📈 Evolution Timeline":
        # Date picker for focused analysis
        nav_options = temporal_manager.get_temporal_navigation_options()
        
        col1, col2 = st.columns([3, 1])
        with col2:
            focus_options = ["All Days"] + [str(d) for d in nav_options.get('available_dates', [])]
            selected_focus = st.selectbox("Focus Date", focus_options)
            focus_date = None if selected_focus == "All Days" else datetime.strptime(selected_focus, "%Y-%m-%d").date()
        
        with col1:
            create_semantic_evolution_timeline(temporal_manager, focus_date)
