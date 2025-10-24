"""semantic_tensor_memory.visualization.concepts
==================================================

Concept-oriented analysis and visualization helpers for the Semantic Tensor
Memory project.  This module centralises the concept clustering analytics and
related Plotly renderers so the rest of the codebase can import a single,
well-documented source of truth instead of relying on multiple legacy modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from memory.universal_core import Modality, UniversalMemoryStore


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
    concept_velocity: List[float]
    major_shifts: List[int]
    concept_persistence: Dict[str, float]


class ConceptAnalyzer:
    """Concept-level analytics driven by the Universal STM memory store."""

    def __init__(self, store: UniversalMemoryStore):
        self.store = store
        self.sequence_embeddings = None
        self.session_metadata = None

    def _extract_sequence_data(
        self, modality: Optional[Modality] = None
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """Return stacked sequence embeddings and lightweight session metadata."""

        if modality:
            sessions = self.store.get_sessions_by_modality(modality)
            embeddings = [emb for _, emb in sessions]
        else:
            embeddings = self.store.embeddings

        if not embeddings:
            raise ValueError("No embeddings found in store")

        sequence_tensors = torch.stack([emb.sequence_embedding for emb in embeddings])

        metadata: List[Dict[str, Any]] = []
        for i, emb in enumerate(embeddings):
            if emb.events:
                first_event = emb.events[0]
                text_sample = first_event.metadata.get("original_text", "")[:100]
            else:
                text_sample = f"Session {i}"

            metadata.append(
                {
                    "session_idx": i,
                    "session_id": emb.session_id,
                    "modality": emb.modality.value,
                    "timestamp": emb.timestamp,
                    "text_sample": text_sample,
                    "num_events": len(emb.events),
                    "sequence_coherence": emb.sequence_coherence,
                    "event_coherence": emb.event_coherence,
                }
            )

        return sequence_tensors, metadata

    def analyze_concept_clusters(
        self, n_clusters: int = 5, modality: Optional[Modality] = None
    ) -> List[ConceptCluster]:
        """Cluster sessions by concept similarity using S-BERT embeddings."""

        sequence_tensors, metadata = self._extract_sequence_data(modality)
        embeddings_np = sequence_tensors.cpu().numpy()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings_np)

        clusters: List[ConceptCluster] = []
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            session_indices = np.where(cluster_mask)[0].tolist()
            if not session_indices:
                continue

            centroid = torch.tensor(
                kmeans.cluster_centers_[cluster_id], device=sequence_tensors.device
            )

            cluster_embeddings = embeddings_np[cluster_mask]
            if len(cluster_embeddings) > 1:
                similarities = cosine_similarity(cluster_embeddings)
                coherence_score = float(
                    np.mean(similarities[np.triu_indices_from(similarities, k=1)])
                )
            else:
                coherence_score = 1.0

            distances = torch.norm(sequence_tensors[cluster_mask] - centroid, dim=1)
            rep_idx_in_cluster = torch.argmin(distances).item()
            rep_session_idx = session_indices[rep_idx_in_cluster]
            representative_text = metadata[rep_session_idx]["text_sample"]

            theme_keywords = self._extract_cluster_themes(session_indices, metadata)

            clusters.append(
                ConceptCluster(
                    cluster_id=cluster_id,
                    session_indices=session_indices,
                    centroid=centroid,
                    coherence_score=coherence_score,
                    representative_text=representative_text,
                    theme_keywords=theme_keywords,
                )
            )

        return sorted(clusters, key=lambda c: len(c.session_indices), reverse=True)

    def analyze_concept_drift_patterns(
        self, modality: Optional[Modality] = None
    ) -> List[ConceptDriftPattern]:
        """Infer temporal concept drift using the store's drift analysis helpers."""

        sequence_tensors, metadata = self._extract_sequence_data(modality)
        if len(sequence_tensors) < 2:
            return []

        drift_patterns: List[ConceptDriftPattern] = []
        for i in range(len(self.store.embeddings) - 1):
            drift_analysis = self.store.analyze_cross_modal_drift(i, i + 1)
            similarity = drift_analysis["sequence_similarity"]
            drift_magnitude = drift_analysis["sequence_drift"]

            if drift_magnitude < 0.1:
                direction = "stable"
            elif drift_magnitude < 0.3:
                direction = "converging" if similarity > 0.7 else "diverging"
            else:
                direction = "diverging"

            shift_keywords = self._extract_shift_keywords(i, i + 1, metadata)

            drift_patterns.append(
                ConceptDriftPattern(
                    session_from=i,
                    session_to=i + 1,
                    drift_magnitude=drift_magnitude,
                    drift_direction=direction,
                    concept_shift_keywords=shift_keywords,
                    temporal_gap=drift_analysis["timestamp_gap"],
                )
            )

        return drift_patterns

    def analyze_concept_velocity(
        self, modality: Optional[Modality] = None
    ) -> List[float]:
        """Calculate the rate of concept change over time."""

        drift_patterns = self.analyze_concept_drift_patterns(modality)
        velocities: List[float] = []
        for pattern in drift_patterns:
            time_gap = max(pattern.temporal_gap, 0.1)
            velocities.append(pattern.drift_magnitude / time_gap)
        return velocities

    def identify_major_concept_shifts(
        self, threshold: float = 0.5, modality: Optional[Modality] = None
    ) -> List[int]:
        """Return session indices where the drift magnitude crosses ``threshold``."""

        drift_patterns = self.analyze_concept_drift_patterns(modality)
        return [pattern.session_to for pattern in drift_patterns if pattern.drift_magnitude > threshold]

    def analyze_complete_concept_evolution(
        self, n_clusters: int = 5, modality: Optional[Modality] = None
    ) -> ConceptEvolution:
        """Run the full concept evolution pipeline and return structured results."""

        sequence_tensors, _ = self._extract_sequence_data(modality)
        clusters = self.analyze_concept_clusters(n_clusters, modality)
        drift_patterns = self.analyze_concept_drift_patterns(modality)
        velocities = self.analyze_concept_velocity(modality)
        major_shifts = self.identify_major_concept_shifts(modality=modality)
        concept_persistence = self._analyze_concept_persistence(
            clusters, len(sequence_tensors)
        )

        return ConceptEvolution(
            total_sessions=len(sequence_tensors),
            concept_clusters=clusters,
            drift_patterns=drift_patterns,
            concept_velocity=velocities,
            major_shifts=major_shifts,
            concept_persistence=concept_persistence,
        )

    def _extract_cluster_themes(
        self, session_indices: List[int], metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Extract simple keyword themes from the provided session metadata."""

        all_text = " ".join(
            metadata[idx]["text_sample"] for idx in session_indices if idx < len(metadata)
        )
        words = re.findall(r"\b[a-zA-Z]{4,}\b", all_text.lower())  # type: ignore[name-defined]
        word_freq: Dict[str, int] = {}
        for word in words:
            if word not in {
                "that",
                "this",
                "with",
                "from",
                "they",
                "have",
                "been",
                "will",
                "would",
                "could",
                "should",
                "session",
            }:
                word_freq[word] = word_freq.get(word, 0) + 1
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        return [word for word, _ in top_words]

    def _extract_shift_keywords(
        self, session_a: int, session_b: int, metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """Return keywords that appear in ``session_b`` but not in ``session_a``."""

        if session_a >= len(metadata) or session_b >= len(metadata):
            return []
        text_a = metadata[session_a]["text_sample"]
        text_b = metadata[session_b]["text_sample"]
        words_a = set(re.findall(r"\b[a-zA-Z]{4,}\b", text_a.lower()))  # type: ignore[name-defined]
        words_b = set(re.findall(r"\b[a-zA-Z]{4,}\b", text_b.lower()))  # type: ignore[name-defined]
        return list(words_b - words_a)[:3]

    def _analyze_concept_persistence(
        self, clusters: List[ConceptCluster], total_sessions: int
    ) -> Dict[str, float]:
        """Compute how long different concept themes persist across sessions."""

        persistence: Dict[str, float] = {}
        for cluster in clusters:
            theme = (
                cluster.theme_keywords[0]
                if cluster.theme_keywords
                else f"cluster_{cluster.cluster_id}"
            )
            persistence[theme] = len(cluster.session_indices) / max(total_sessions, 1)
        return persistence


def analyze_existing_store_concepts(
    store: UniversalMemoryStore, n_clusters: int = 5
) -> ConceptEvolution:
    """Analyze concepts stored in ``store`` and return aggregated evolution data."""

    analyzer = ConceptAnalyzer(store)
    return analyzer.analyze_complete_concept_evolution(n_clusters)


def get_concept_similarity_matrix(
    store: UniversalMemoryStore, modality: Optional[Modality] = None
) -> np.ndarray:
    """Return the session-to-session cosine similarity matrix for the store."""

    analyzer = ConceptAnalyzer(store)
    sequence_tensors, _ = analyzer._extract_sequence_data(modality)
    embeddings_np = sequence_tensors.cpu().numpy()
    return cosine_similarity(embeddings_np)


class ConceptVisualizer:
    """Plotly helpers that render :class:`ConceptEvolution` outputs."""

    def __init__(self) -> None:
        self.color_palette = px.colors.qualitative.Set3

    def create_concept_cluster_heatmap(self, evolution: ConceptEvolution) -> go.Figure:
        clusters = evolution.concept_clusters
        cluster_matrix: List[List[float]] = []
        cluster_labels: List[str] = []

        for cluster in clusters:
            row = [0.0] * evolution.total_sessions
            for session_idx in cluster.session_indices:
                row[session_idx] = cluster.coherence_score
            cluster_matrix.append(row)
            theme_str = ", ".join(cluster.theme_keywords[:3])
            cluster_labels.append(f"Cluster {cluster.cluster_id}: {theme_str}")

        fig = go.Figure(
            data=go.Heatmap(
                z=cluster_matrix,
                x=[f"Session {i + 1}" for i in range(evolution.total_sessions)],
                y=cluster_labels,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Coherence Score"),
                hoverongaps=False,
            )
        )
        fig.update_layout(
            title="Concept Cluster Membership & Coherence",
            xaxis_title="Sessions",
            yaxis_title="Concept Clusters",
            height=400 + len(clusters) * 40,
            font=dict(size=12),
        )
        return fig

    def create_drift_timeline(self, evolution: ConceptEvolution) -> go.Figure:
        drift_patterns = evolution.drift_patterns
        if not drift_patterns:
            fig = go.Figure()
            fig.add_annotation(text="No drift patterns to visualize")
            return fig

        sessions = [pattern.session_to for pattern in drift_patterns]
        drift_magnitudes = [pattern.drift_magnitude for pattern in drift_patterns]
        directions = [pattern.drift_direction for pattern in drift_patterns]
        colors = {"stable": "green", "converging": "blue", "diverging": "red"}
        point_colors = [colors.get(direction, "gray") for direction in directions]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sessions,
                y=drift_magnitudes,
                mode="lines+markers",
                name="Concept Drift",
                line=dict(width=2, color="darkblue"),
                marker=dict(size=10, color=point_colors, line=dict(width=2, color="white")),
                hovertemplate="<b>Session %{x}</b><br>Drift: %{y:.3f}<extra></extra>",
            )
        )

        if evolution.major_shifts:
            highlight_y = [
                drift_magnitudes[min(idx - 2, len(drift_magnitudes) - 1)]
                if idx - 2 >= 0
                else 0.5
                for idx in evolution.major_shifts
            ]
            fig.add_trace(
                go.Scatter(
                    x=evolution.major_shifts,
                    y=highlight_y,
                    mode="markers",
                    name="Major Shifts",
                    marker=dict(size=15, color="red", symbol="star", line=dict(width=2, color="white")),
                    hovertemplate="<b>Major Shift</b><br>Session: %{x}<extra></extra>",
                )
            )

        fig.update_layout(
            title="Concept Drift Evolution Over Time",
            xaxis_title="Session Number",
            yaxis_title="Drift Magnitude",
            height=400,
            hovermode="closest",
            showlegend=True,
        )
        return fig

    def create_concept_velocity_chart(self, evolution: ConceptEvolution) -> go.Figure:
        if not evolution.concept_velocity:
            fig = go.Figure()
            fig.add_annotation(text="No velocity data to visualize")
            return fig

        sessions = list(range(1, len(evolution.concept_velocity) + 1))
        velocities = evolution.concept_velocity
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sessions,
                y=velocities,
                mode="lines+markers",
                name="Concept Velocity",
                line=dict(width=1, color="lightblue"),
                marker=dict(size=6),
                opacity=0.7,
            )
        )

        if len(velocities) > 3:
            window = min(5, max(2, len(velocities) // 3))
            moving_avg = (
                pd.Series(velocities)
                .rolling(window=window, center=True)
                .mean()
                .bfill()
                .ffill()
            )
            fig.add_trace(
                go.Scatter(
                    x=sessions,
                    y=moving_avg,
                    mode="lines",
                    name=f"Moving Average ({window} sessions)",
                    line=dict(width=3, color="darkblue"),
                )
            )

        fig.update_layout(
            title="Concept Change Velocity",
            xaxis_title="Session Transition",
            yaxis_title="Velocity (Drift/Time)",
            height=400,
            hovermode="x unified",
        )
        return fig

    def create_concept_persistence_pie(self, evolution: ConceptEvolution) -> go.Figure:
        if not evolution.concept_persistence:
            fig = go.Figure()
            fig.add_annotation(text="No persistence data to visualize")
            return fig

        themes = list(evolution.concept_persistence.keys())
        persistence_values = list(evolution.concept_persistence.values())
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=themes,
                    values=persistence_values,
                    hole=0.3,
                    hovertemplate="<b>%{label}</b><br>Persistence: %{value:.1%}<extra></extra>",
                )
            ]
        )
        fig.update_layout(title="Concept Persistence Overview", height=400)
        return fig

    def create_comprehensive_dashboard(self, evolution: ConceptEvolution) -> go.Figure:
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=(
                "Concept Drift Timeline",
                "Concept Velocity",
                "Cluster Membership Heatmap",
                "Concept Persistence",
                "Drift Patterns Distribution",
                "Quality Metrics",
            ),
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"colspan": 2}, None],
                [{"type": "bar"}, {"type": "bar"}],
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1,
        )

        if evolution.drift_patterns:
            sessions = [pattern.session_to for pattern in evolution.drift_patterns]
            drift_magnitudes = [pattern.drift_magnitude for pattern in evolution.drift_patterns]
            fig.add_trace(
                go.Scatter(x=sessions, y=drift_magnitudes, mode="lines+markers", name="Drift"),
                row=1,
                col=1,
            )

        if evolution.concept_velocity:
            velocity_sessions = list(range(1, len(evolution.concept_velocity) + 1))
            fig.add_trace(
                go.Scatter(
                    x=velocity_sessions,
                    y=evolution.concept_velocity,
                    mode="lines+markers",
                    name="Velocity",
                ),
                row=1,
                col=2,
            )

        if evolution.concept_clusters:
            cluster_matrix: List[List[int]] = []
            cluster_labels: List[str] = []
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
                    x=[f"S{i + 1}" for i in range(evolution.total_sessions)],
                    colorscale="Blues",
                    showscale=False,
                ),
                row=2,
                col=1,
            )

        if evolution.drift_patterns:
            directions = [pattern.drift_direction for pattern in evolution.drift_patterns]
            direction_counts = pd.Series(directions).value_counts()
            fig.add_trace(
                go.Bar(x=direction_counts.index, y=direction_counts.values, name="Direction Count"),
                row=3,
                col=1,
            )

        if evolution.concept_clusters:
            coherence_scores = [cluster.coherence_score for cluster in evolution.concept_clusters]
            fig.add_trace(
                go.Bar(
                    x=[f"C{cluster.cluster_id}" for cluster in evolution.concept_clusters],
                    y=coherence_scores,
                    name="Coherence",
                ),
                row=3,
                col=2,
            )

        fig.update_layout(height=1000, title_text="Concept Evolution Dashboard", showlegend=False)
        return fig

    def create_concept_network_graph(
        self, evolution: ConceptEvolution, similarity_threshold: float = 0.3
    ) -> go.Figure:
        clusters = evolution.concept_clusters
        if len(clusters) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 clusters for network visualization")
            return fig

        graph = nx.Graph()
        for cluster in clusters:
            theme_str = ", ".join(cluster.theme_keywords[:2])
            graph.add_node(
                cluster.cluster_id,
                label=f"C{cluster.cluster_id}: {theme_str}",
                size=len(cluster.session_indices),
                coherence=cluster.coherence_score,
            )

        for i, cluster_a in enumerate(clusters):
            for cluster_b in clusters[i + 1 :]:
                set_a = set(cluster_a.session_indices)
                set_b = set(cluster_b.session_indices)
                if not set_a or not set_b:
                    continue
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                similarity = intersection / union if union else 0.0
                if similarity > similarity_threshold:
                    graph.add_edge(cluster_a.cluster_id, cluster_b.cluster_id, weight=similarity)

        pos = nx.spring_layout(graph, seed=42)

        fig = go.Figure()
        for edge in graph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = graph.edges[edge]["weight"]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode="lines",
                    line=dict(width=weight * 10, color="gray"),
                    opacity=0.5,
                    showlegend=False,
                    hoverinfo="none",
                )
            )

        node_x = [pos[node][0] for node in graph.nodes()]
        node_y = [pos[node][1] for node in graph.nodes()]
        node_sizes = [graph.nodes[node]["size"] * 3 for node in graph.nodes()]
        node_colors = [graph.nodes[node]["coherence"] for node in graph.nodes()]
        node_labels = [graph.nodes[node]["label"] for node in graph.nodes()]

        fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers+text",
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale="Viridis",
                    colorbar=dict(title="Coherence"),
                    line=dict(width=2, color="white"),
                ),
                text=[f"C{node}" for node in graph.nodes()],
                textposition="middle center",
                hovertext=node_labels,
                hoverinfo="text",
                name="Concept Clusters",
            )
        )

        fig.update_layout(
            title="Concept Cluster Relationship Network",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Node size = cluster size, Color = coherence, Edges = similarity",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(size=12),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
        )
        return fig

    def create_concept_network(self, evolution: ConceptEvolution) -> go.Figure:
        """Backward compatible alias for :meth:`create_concept_network_graph`."""

        return self.create_concept_network_graph(evolution)

    def create_concept_timeline(self, evolution: ConceptEvolution) -> go.Figure:
        sessions = list(range(1, evolution.total_sessions + 1))
        cluster_presence: Dict[str, List[int]] = {}
        for cluster in evolution.concept_clusters:
            theme = cluster.theme_keywords[0] if cluster.theme_keywords else f"Cluster {cluster.cluster_id}"
            presence = [0] * evolution.total_sessions
            for idx in cluster.session_indices:
                presence[idx] = 1
            cluster_presence[theme] = presence

        fig = make_subplots(rows=len(cluster_presence), cols=1, shared_xaxes=True)
        for row, (theme, presence) in enumerate(cluster_presence.items(), start=1):
            fig.add_trace(
                go.Bar(x=sessions, y=presence, name=theme, marker_color=self.color_palette[row % len(self.color_palette)]),
                row=row,
                col=1,
            )
        fig.update_layout(
            title="Concept Presence Timeline",
            xaxis_title="Session",
            height=200 * max(len(cluster_presence), 1),
            showlegend=False,
        )
        return fig

    def create_drift_summary(self, evolution: ConceptEvolution) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for pattern in evolution.drift_patterns:
            records.append(
                {
                    "From Session": pattern.session_from,
                    "To Session": pattern.session_to,
                    "Drift Magnitude": pattern.drift_magnitude,
                    "Direction": pattern.drift_direction,
                    "Shift Keywords": ", ".join(pattern.concept_shift_keywords),
                    "Temporal Gap": pattern.temporal_gap,
                }
        )
        return pd.DataFrame(records)


# ``ConceptAnalyzer`` relies on regex helpers; import after class definition to keep the
# public surface clean and avoid polluting module scope with unused names.
import re  # noqa: E402  (import after class definition for readability)


def visualize_concept_evolution(
    evolution: ConceptEvolution, chart_type: str = "dashboard"
) -> go.Figure:
    """Return a Plotly figure for the requested concept evolution chart type."""

    visualizer = ConceptVisualizer()
    if chart_type == "dashboard":
        return visualizer.create_comprehensive_dashboard(evolution)
    if chart_type == "heatmap":
        return visualizer.create_concept_cluster_heatmap(evolution)
    if chart_type == "timeline":
        return visualizer.create_drift_timeline(evolution)
    if chart_type == "velocity":
        return visualizer.create_concept_velocity_chart(evolution)
    if chart_type == "network":
        return visualizer.create_concept_network_graph(evolution)
    if chart_type == "persistence":
        return visualizer.create_concept_persistence_pie(evolution)

    raise ValueError(f"Unknown chart type: {chart_type}")
