"""Enhanced concept analysis tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from viz.heatmap import token_alignment_heatmap

__all__ = ["render_enhanced_concept_analysis_tab"]


def render_enhanced_concept_analysis_tab() -> None:
    """Render enhanced concept analysis using existing S-BERT embeddings."""
    st.header("🧠 Enhanced Concept Analysis")
    st.markdown("**Leveraging existing S-BERT sequence embeddings for concept-level analysis**")

    if not st.session_state.get("memory") or len(st.session_state.memory) < 2:
        st.info("📁 Need at least 2 sessions for concept analysis")
        st.markdown(
            """
        **What you'll get with Enhanced Concept Analysis:**
        - **🎯 Concept Clustering**: Group sessions by semantic similarity using S-BERT embeddings
        - **📈 Drift Timeline**: Track how concepts evolve over time
        - **⚡ Velocity Analysis**: Measure rate of concept change
        - **🌐 Network Visualization**: See relationships between concept clusters
        - **🥧 Persistence Analysis**: Identify long-lasting vs transient concepts
        - **📊 Comprehensive Dashboard**: All analyses in one view

        Upload your data to begin concept-level analysis!
        """
        )
        return

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        n_clusters = st.slider(
            "Number of concept clusters", 2, min(8, len(st.session_state.memory)), 5
        )
    with col2:
        analysis_scope = st.selectbox(
            "Analysis Scope",
            ["All Sessions", "Recent 50", "Recent 100", "First 50"],
        )
    with col3:
        st.write("")
        if st.button("🔍 Analyze Concepts", type="primary", key="concept_analyze_btn"):
            st.session_state.run_concept_analysis = True

    if st.session_state.get("run_concept_analysis", False):
        with st.spinner("🧠 Analyzing concept evolution using S-BERT embeddings..."):
            try:
                from analysis.concept_analysis import ConceptAnalyzer
                from memory.text_embedder import TextEmbedder
                from memory.universal_core import UniversalMemoryStore

                universal_store = UniversalMemoryStore()
                text_embedder = TextEmbedder()

                memory_data = st.session_state.memory
                if analysis_scope == "Recent 50" and len(memory_data) > 50:
                    memory_data = memory_data[-50:]
                elif analysis_scope == "Recent 100" and len(memory_data) > 100:
                    memory_data = memory_data[-100:]
                elif analysis_scope == "First 50":
                    memory_data = memory_data[:50]

                progress_bar = st.progress(0)
                for i, session_text in enumerate(memory_data):
                    embedding = text_embedder.process_raw_data(
                        session_text,
                        session_id=f"session_{i}",
                    )
                    universal_store.add_session(embedding)
                    progress_bar.progress((i + 1) / len(memory_data))
                progress_bar.empty()

                analyzer = ConceptAnalyzer(universal_store)
                concept_evolution = analyzer.analyze_complete_concept_evolution(n_clusters)

                st.session_state.concept_evolution = concept_evolution
                st.session_state.concept_store = universal_store
                st.session_state.run_concept_analysis = False

                st.success(
                    f"✅ Analyzed {concept_evolution.total_sessions} sessions with {len(concept_evolution.concept_clusters)} concept clusters"
                )
            except Exception as exc:  # pragma: no cover - heavy analysis path
                st.error(f"Error in concept analysis: {str(exc)}")
                st.session_state.run_concept_analysis = False
                return

    if not hasattr(st.session_state, "concept_evolution"):
        return

    concept_evolution = st.session_state.concept_evolution
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Sessions", concept_evolution.total_sessions)
    with col2:
        st.metric("🎯 Clusters", len(concept_evolution.concept_clusters))
    with col3:
        major_shifts = len(concept_evolution.major_shifts)
        st.metric("🚀 Major Shifts", major_shifts)
    with col4:
        if concept_evolution.drift_patterns:
            avg_drift = np.mean([p.drift_magnitude for p in concept_evolution.drift_patterns])
            st.metric("📈 Avg Drift", f"{avg_drift:.3f}")
        else:
            st.metric("📈 Avg Drift", "N/A")

    st.subheader("📊 Concept Evolution Visualizations")
    viz_choice = st.selectbox(
        "Choose Visualization",
        ["📈 Dashboard", "🔥 Cluster Heatmap", "📊 Drift Timeline", "⚡ Velocity Chart", "🌐 Network Graph", "🥧 Persistence"],
        key="concept_viz_choice",
    )

    try:
        from visualization.concept_visualizer import visualize_concept_evolution

        chart_type_map = {
            "📈 Dashboard": "dashboard",
            "🔥 Cluster Heatmap": "heatmap",
            "📊 Drift Timeline": "timeline",
            "⚡ Velocity Chart": "velocity",
            "🌐 Network Graph": "network",
            "🥧 Persistence": "persistence",
        }

        chart_type = chart_type_map[viz_choice]
        fig = visualize_concept_evolution(concept_evolution, chart_type)
        st.plotly_chart(fig, use_container_width=True)

        if viz_choice == "🔥 Cluster Heatmap":
            st.subheader("🎯 Cluster Details")
            for cluster in concept_evolution.concept_clusters[:5]:
                with st.expander(
                    f"Cluster {cluster.cluster_id}: {', '.join(cluster.theme_keywords[:3])}"
                ):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Sessions:** {len(cluster.session_indices)}")
                        st.write(f"**Coherence:** {cluster.coherence_score:.3f}")
                    with col2:
                        st.write(f"**Keywords:** {', '.join(cluster.theme_keywords)}")
                        st.write(f"**Sample:** {cluster.representative_text[:150]}...")
                    with st.expander("🔎 Token Alignment (top exemplars)"):
                        if len(cluster.session_indices) >= 2:
                            a = cluster.session_indices[0]
                            b = cluster.session_indices[1]
                            st.caption(f"Aligning Session {a+1} and Session {b+1}")
                            if st.button(
                                f"Show alignment for Cluster {cluster.cluster_id}",
                                key=f"cluster_align_{cluster.cluster_id}",
                            ):
                                try:
                                    token_alignment_heatmap(st.session_state.memory, a, b)
                                    st.caption("Close the Matplotlib window to continue.")
                                except Exception as exc:  # pragma: no cover - visualization path
                                    st.error(f"Alignment failed: {exc}")
        elif viz_choice == "📊 Drift Timeline" and concept_evolution.drift_patterns:
            st.subheader("📋 Drift Pattern Summary")
            drift_df = pd.DataFrame(
                [
                    {
                        "From": p.session_from + 1,
                        "To": p.session_to + 1,
                        "Drift": f"{p.drift_magnitude:.3f}",
                        "Direction": p.drift_direction,
                        "New Concepts": ", ".join(p.concept_shift_keywords[:2]),
                    }
                    for p in concept_evolution.drift_patterns[:10]
                ]
            )
            st.dataframe(drift_df, use_container_width=True)
        elif viz_choice == "⚡ Velocity Chart" and concept_evolution.concept_velocity:
            max_velocity_idx = np.argmax(concept_evolution.concept_velocity)
            st.info(
                f"🚀 **Peak velocity** at transition {max_velocity_idx + 1}: {concept_evolution.concept_velocity[max_velocity_idx]:.3f}"
            )
        elif viz_choice == "🥧 Persistence" and concept_evolution.concept_persistence:
            st.subheader("🏆 Top Persistent Concepts")
            persistence_df = pd.DataFrame(
                [
                    {"Concept": concept, "Persistence": f"{persistence:.1%}"}
                    for concept, persistence in sorted(
                        concept_evolution.concept_persistence.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:5]
                ]
            )
            st.dataframe(persistence_df, use_container_width=True)
    except Exception as exc:  # pragma: no cover - visualization path
        st.error(f"Visualization error: {str(exc)}")

    st.markdown("### 💡 About Enhanced Concept Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            **🎯 Concept Clustering**
            - Groups sessions by semantic similarity
            - Uses existing S-BERT sequence embeddings
            - Identifies coherent themes and topics
            - Shows which sessions belong together

            **📈 Temporal Drift Analysis**
            - Tracks concept evolution over time
            - Measures drift magnitude and direction
            - Identifies major conceptual shifts
            - Shows stability vs. change patterns
            """
        )
    with col2:
        st.markdown(
            """
            **⚡ Velocity & Network Analysis**
            - Measures rate of concept change
            - Maps relationships between clusters
            - Shows concept persistence over time
            - Provides comprehensive dashboards

            **🌟 Key Advantages**
            - Leverages existing S-BERT embeddings
            - No misleading PCA projections
            - Interpretable concept-focused views
            - Real semantic understanding
            """
        )
    st.info("🚀 **Ready to analyze?** Choose your settings above and click **'Analyze Concepts'** to begin!")
