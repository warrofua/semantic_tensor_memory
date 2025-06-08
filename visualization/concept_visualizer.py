#!/usr/bin/env python3
"""
Enhanced Concept Visualizer for Universal STM

Provides meaningful visualizations of concept evolution using existing S-BERT embeddings.
Replaces misleading PCA plots with interpretable concept-focused visualizations.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import networkx as nx

from analysis.concept_analysis import ConceptEvolution, ConceptCluster, ConceptDriftPattern

class ConceptVisualizer:
    """
    Enhanced visualizations for concept analysis that focus on interpretable representations
    rather than dimensionality reduction artifacts.
    """
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def create_concept_cluster_heatmap(self, evolution: ConceptEvolution) -> go.Figure:
        """
        Create a heatmap showing concept cluster membership and coherence.
        More meaningful than PCA projections.
        """
        clusters = evolution.concept_clusters
        
        # Create cluster membership matrix
        cluster_matrix = []
        cluster_labels = []
        
        for cluster in clusters:
            # Create row for this cluster showing session membership
            row = [0] * evolution.total_sessions
            for session_idx in cluster.session_indices:
                row[session_idx] = cluster.coherence_score
            
            cluster_matrix.append(row)
            
            # Create label with theme keywords
            theme_str = ", ".join(cluster.theme_keywords[:3])
            cluster_labels.append(f"Cluster {cluster.cluster_id}: {theme_str}")
        
        fig = go.Figure(data=go.Heatmap(
            z=cluster_matrix,
            x=[f"Session {i+1}" for i in range(evolution.total_sessions)],
            y=cluster_labels,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Coherence Score"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Concept Cluster Membership & Coherence",
            xaxis_title="Sessions",
            yaxis_title="Concept Clusters",
            height=400 + len(clusters) * 40,
            font=dict(size=12)
        )
        
        return fig
    
    def create_drift_timeline(self, evolution: ConceptEvolution) -> go.Figure:
        """
        Timeline showing concept drift magnitude over sessions.
        Much more interpretable than spatial projections.
        """
        drift_patterns = evolution.drift_patterns
        
        if not drift_patterns:
            return go.Figure().add_annotation(text="No drift patterns to visualize")
        
        sessions = [p.session_to for p in drift_patterns]
        drift_magnitudes = [p.drift_magnitude for p in drift_patterns]
        directions = [p.drift_direction for p in drift_patterns]
        
        # Color by drift direction
        colors = {'stable': 'green', 'converging': 'blue', 'diverging': 'red'}
        point_colors = [colors.get(d, 'gray') for d in directions]
        
        fig = go.Figure()
        
        # Main drift line
        fig.add_trace(go.Scatter(
            x=sessions,
            y=drift_magnitudes,
            mode='lines+markers',
            name='Concept Drift',
            line=dict(width=2, color='darkblue'),
            marker=dict(
                size=10,
                color=point_colors,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>Session %{x}</b><br>' +
                         'Drift: %{y:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Add major shifts
        if evolution.major_shifts:
            fig.add_trace(go.Scatter(
                x=evolution.major_shifts,
                y=[drift_magnitudes[s-2] if s-2 < len(drift_magnitudes) else 0.5 for s in evolution.major_shifts],
                mode='markers',
                name='Major Shifts',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='white')
                ),
                hovertemplate='<b>Major Shift</b><br>' +
                             'Session: %{x}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title="Concept Drift Evolution Over Time",
            xaxis_title="Session Number",
            yaxis_title="Drift Magnitude",
            height=400,
            hovermode='closest',
            showlegend=True
        )
        
        return fig
    
    def create_concept_velocity_chart(self, evolution: ConceptEvolution) -> go.Figure:
        """
        Show rate of concept change over time (concept velocity).
        """
        if not evolution.concept_velocity:
            return go.Figure().add_annotation(text="No velocity data to visualize")
        
        sessions = list(range(1, len(evolution.concept_velocity) + 1))
        velocities = evolution.concept_velocity
        
        # Create velocity chart with moving average
        fig = go.Figure()
        
        # Raw velocity
        fig.add_trace(go.Scatter(
            x=sessions,
            y=velocities,
            mode='lines+markers',
            name='Concept Velocity',
            line=dict(width=1, color='lightblue'),
            marker=dict(size=6),
            opacity=0.7
        ))
        
        # Moving average (if enough data)
        if len(velocities) > 3:
            window = min(5, len(velocities) // 3)
            moving_avg = pd.Series(velocities).rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
            
            fig.add_trace(go.Scatter(
                x=sessions,
                y=moving_avg,
                mode='lines',
                name=f'Moving Average ({window} sessions)',
                line=dict(width=3, color='darkblue')
            ))
        
        fig.update_layout(
            title="Concept Change Velocity",
            xaxis_title="Session Transition",
            yaxis_title="Velocity (Drift/Time)",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_concept_persistence_pie(self, evolution: ConceptEvolution) -> go.Figure:
        """
        Show concept persistence as a pie chart.
        """
        if not evolution.concept_persistence:
            return go.Figure().add_annotation(text="No persistence data to visualize")
        
        themes = list(evolution.concept_persistence.keys())
        persistence_values = list(evolution.concept_persistence.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=themes,
            values=persistence_values,
            hole=0.3,
            hovertemplate='<b>%{label}</b><br>' +
                         'Persistence: %{value:.1%}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title="Concept Persistence Distribution",
            height=500,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.05)
        )
        
        return fig
    
    def create_concept_network_graph(self, evolution: ConceptEvolution, similarity_threshold: float = 0.3) -> go.Figure:
        """
        Create a network graph showing relationships between concept clusters.
        """
        clusters = evolution.concept_clusters
        
        if len(clusters) < 2:
            return go.Figure().add_annotation(text="Need at least 2 clusters for network visualization")
        
        # Calculate cluster similarities based on shared sessions
        G = nx.Graph()
        
        # Add nodes (clusters)
        for cluster in clusters:
            theme_str = ", ".join(cluster.theme_keywords[:2])
            G.add_node(cluster.cluster_id, 
                      label=f"C{cluster.cluster_id}: {theme_str}",
                      size=len(cluster.session_indices),
                      coherence=cluster.coherence_score)
        
        # Add edges based on session overlap
        for i, cluster_a in enumerate(clusters):
            for cluster_b in clusters[i+1:]:
                # Calculate Jaccard similarity of session sets
                set_a = set(cluster_a.session_indices)
                set_b = set(cluster_b.session_indices)
                
                if set_a and set_b:  # Avoid division by zero
                    intersection = len(set_a & set_b)
                    union = len(set_a | set_b)
                    similarity = intersection / union if union > 0 else 0
                    
                    if similarity > similarity_threshold:
                        G.add_edge(cluster_a.cluster_id, cluster_b.cluster_id, weight=similarity)
        
        # Create layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Extract node and edge information
        node_x = [pos[node][0] for node in G.nodes()]
        node_y = [pos[node][1] for node in G.nodes()]
        node_text = [G.nodes[node]['label'] for node in G.nodes()]
        node_sizes = [G.nodes[node]['size'] * 3 for node in G.nodes()]
        node_colors = [G.nodes[node]['coherence'] for node in G.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = G.edges[edge]['weight']
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight*10, color='gray'),
                opacity=0.5,
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                colorbar=dict(title="Coherence"),
                line=dict(width=2, color='white')
            ),
            text=[f"C{i}" for i in G.nodes()],
            textposition="middle center",
            hovertext=node_text,
            hoverinfo='text',
            name='Concept Clusters'
        ))
        
        fig.update_layout(
            title="Concept Cluster Relationship Network",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Node size = cluster size, Color = coherence, Edges = similarity",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12)
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig
    
    def create_comprehensive_dashboard(self, evolution: ConceptEvolution) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple concept visualizations.
        """
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Concept Drift Timeline", "Concept Velocity",
                "Cluster Membership Heatmap", "Concept Persistence",
                "Drift Patterns Distribution", "Quality Metrics"
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"colspan": 2}, None],
                [{"type": "bar"}, {"type": "bar"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # 1. Drift timeline (top left)
        if evolution.drift_patterns:
            sessions = [p.session_to for p in evolution.drift_patterns]
            drift_magnitudes = [p.drift_magnitude for p in evolution.drift_patterns]
            
            fig.add_trace(
                go.Scatter(x=sessions, y=drift_magnitudes, mode='lines+markers', name='Drift'),
                row=1, col=1
            )
        
        # 2. Velocity chart (top right)
        if evolution.concept_velocity:
            velocity_sessions = list(range(1, len(evolution.concept_velocity) + 1))
            fig.add_trace(
                go.Scatter(x=velocity_sessions, y=evolution.concept_velocity, mode='lines+markers', name='Velocity'),
                row=1, col=2
            )
        
        # 3. Cluster heatmap (middle, full width)
        if evolution.concept_clusters:
            cluster_matrix = []
            cluster_labels = []
            
            for cluster in evolution.concept_clusters:
                row = [0] * evolution.total_sessions
                for session_idx in cluster.session_indices:
                    row[session_idx] = 1
                cluster_matrix.append(row)
                theme_str = ", ".join(cluster.theme_keywords[:2])
                cluster_labels.append(f"C{cluster.cluster_id}: {theme_str}")
            
            fig.add_trace(
                go.Heatmap(
                    z=cluster_matrix,
                    y=cluster_labels,
                    x=[f"S{i+1}" for i in range(evolution.total_sessions)],
                    colorscale='Blues',
                    showscale=False
                ),
                row=2, col=1
            )
        
        # 4. Drift direction distribution (bottom left)
        if evolution.drift_patterns:
            directions = [p.drift_direction for p in evolution.drift_patterns]
            direction_counts = pd.Series(directions).value_counts()
            
            fig.add_trace(
                go.Bar(x=direction_counts.index, y=direction_counts.values, name='Direction Count'),
                row=3, col=1
            )
        
        # 5. Quality metrics (bottom right)
        if evolution.concept_clusters:
            coherence_scores = [c.coherence_score for c in evolution.concept_clusters]
            cluster_sizes = [len(c.session_indices) for c in evolution.concept_clusters]
            
            fig.add_trace(
                go.Bar(x=[f"C{c.cluster_id}" for c in evolution.concept_clusters], 
                      y=coherence_scores, name='Coherence'),
                row=3, col=2
            )
        
        fig.update_layout(
            height=1000,
            title_text="Concept Evolution Dashboard",
            showlegend=False
        )
        
        return fig

# Utility function for easy integration
def visualize_concept_evolution(evolution: ConceptEvolution, chart_type: str = "dashboard") -> go.Figure:
    """
    Main entry point for concept visualization.
    
    Args:
        evolution: ConceptEvolution analysis results
        chart_type: Type of chart to create ("dashboard", "heatmap", "timeline", "velocity", "network", "persistence")
    """
    visualizer = ConceptVisualizer()
    
    if chart_type == "dashboard":
        return visualizer.create_comprehensive_dashboard(evolution)
    elif chart_type == "heatmap":
        return visualizer.create_concept_cluster_heatmap(evolution)
    elif chart_type == "timeline":
        return visualizer.create_drift_timeline(evolution)
    elif chart_type == "velocity":
        return visualizer.create_concept_velocity_chart(evolution)
    elif chart_type == "network":
        return visualizer.create_concept_network_graph(evolution)
    elif chart_type == "persistence":
        return visualizer.create_concept_persistence_pie(evolution)
    else:
        raise ValueError(f"Unknown chart type: {chart_type}") 