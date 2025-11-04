"""Overview dashboard tab."""

from __future__ import annotations

import numpy as np
import streamlit as st

from semantic_tensor_memory.analytics import ConceptAnalyzer
from semantic_tensor_memory.streamlit.plots import plot_heatmap_plotly

from ..models import get_cached_text_embedder, get_cached_universal_store

__all__ = ["render_overview_dashboard"]


def render_overview_dashboard() -> None:
    """Render an enhanced overview dashboard with concept-level insights."""
    dataset_info = st.session_state.get("dataset_info", {})
    filename = dataset_info.get("filename", "Unknown Dataset")
    session_count = dataset_info.get("session_count", 0)
    source_type = dataset_info.get("source", "unknown")

    icon = "🤖" if source_type == "ai_conversation" else "📊"
    source_label = (
        "AI Conversation Analysis" if source_type == "ai_conversation" else "Semantic Analysis Overview"
    )

    st.markdown(
        f"""
    <div style="text-align: center; padding: 1rem 0;">
        <h1>{icon} {filename}</h1>
        <h3 style=\"color: #666;\">{source_label}</h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if "performance_metrics" in dataset_info:
        performance = dataset_info["performance_metrics"]
        processing_time = dataset_info.get("processing_time", 0)
        memory_usage = dataset_info.get("memory_usage", 0)
        processing_strategy = dataset_info.get("processing_strategy", "full_processing")
        complexity_score = dataset_info.get("complexity_score", 0)

        with st.expander("📊 **Processing Performance Summary**", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "⚡ Processing Speed",
                    f"{performance['sessions_per_second']:.1f}/s",
                    help="Sessions processed per second",
                )
            with col2:
                st.metric(
                    "🧠 Memory Efficiency",
                    f"{performance['memory_efficiency']:.1f}",
                    help="Sessions per MB of memory used",
                )
            with col3:
                st.metric(
                    "✅ Success Rate",
                    f"{performance['success_rate']:.1%}",
                    help="Percentage of sessions successfully processed",
                )
            with col4:
                st.metric(
                    "🎯 Quality Score",
                    f"{performance['estimated_quality']:.1%}",
                    help="Overall processing quality estimate",
                )

            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**⏱️ Total Time**: {processing_time:.1f}s")
                st.markdown(f"**🔧 Strategy**: {processing_strategy.replace('_', ' ').title()}")
            with col2:
                st.markdown(f"**💾 Memory Usage**: {memory_usage:.0f}MB")
                st.markdown(f"**📊 Dataset Complexity**: {complexity_score:.2f}/1.0")
            with col3:
                quality = performance["estimated_quality"]
                if quality > 0.8:
                    quality_status = "🟢 Excellent"
                elif quality > 0.6:
                    quality_status = "🟡 Good"
                else:
                    quality_status = "🔴 Needs Review"
                st.markdown(f"**Quality Status**: {quality_status}")
                if processing_strategy == "progressive_sampling":
                    st.markdown("**🎯 Used Smart Sampling** for large dataset")
                elif processing_strategy == "smart_batching":
                    st.markdown("**🔄 Used Adaptive Batching** for efficiency")
                else:
                    st.markdown("**✅ Full Processing** completed")

    if source_type == "ai_conversation":
        st.info("🤖 **AI Conversation Detected**: Analyzing your message evolution across AI interactions")

    if session_count < 2:
        st.info("📊 Upload at least 2 sessions to see meaningful semantic analysis.")
        return

    st.markdown("### 🚀 Quick Semantic Insights")

    if st.button("🔍 Generate Quick Analysis", type="primary", key="quick_analysis_btn"):
        with st.spinner("🧠 Analyzing your semantic patterns..."):
            try:

                if "quick_analysis_store" not in st.session_state:
                    st.session_state.quick_analysis_store = get_cached_universal_store()
                    st.session_state.quick_analysis_embedder = get_cached_text_embedder()
                else:
                    st.session_state.quick_analysis_store = get_cached_universal_store()

                universal_store = st.session_state.quick_analysis_store
                text_embedder = st.session_state.quick_analysis_embedder

                sample_size = min(20, len(st.session_state.memory))
                memory_sample = st.session_state.memory[-sample_size:]

                for i, session_text in enumerate(memory_sample):
                    if i < len(st.session_state.meta):
                        meta_text = st.session_state.meta[i].get("text", str(session_text))
                    else:
                        meta_text = str(session_text)

                    embedding = text_embedder.process_raw_data(
                        meta_text,
                        session_id=f"overview_session_{i}",
                    )
                    universal_store.add_session(embedding)

                analyzer = ConceptAnalyzer(universal_store)
                concept_evolution = analyzer.analyze_complete_concept_evolution(n_clusters=3)
                st.session_state.overview_concept_analysis = concept_evolution
            except Exception as exc:  # pragma: no cover - UI feedback path
                st.error(f"Quick analysis failed: {str(exc)}")

    if hasattr(st.session_state, "overview_concept_analysis"):
        concept_evolution = st.session_state.overview_concept_analysis

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📊 Sessions Analyzed", concept_evolution.total_sessions)
        with col2:
            st.metric("🎯 Main Concepts", len(concept_evolution.concept_clusters))
        with col3:
            major_shifts = len(concept_evolution.major_shifts)
            st.metric("🚀 Major Shifts", major_shifts)
        with col4:
            if concept_evolution.drift_patterns:
                avg_drift = np.mean([p.drift_magnitude for p in concept_evolution.drift_patterns])
                st.metric("📈 Avg Drift", f"{avg_drift:.3f}")
            else:
                st.metric("📈 Avg Drift", "N/A")

        st.markdown("### 💡 Your Main Concepts")
        if concept_evolution.concept_clusters:
            for i, cluster in enumerate(concept_evolution.concept_clusters[:3]):
                with st.expander(
                    f"🎯 Concept {i+1}: {', '.join(cluster.theme_keywords[:3])}",
                    expanded=i == 0,
                ):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.write(f"**Sessions:** {len(cluster.session_indices)}")
                        st.write(f"**Main themes:** {', '.join(cluster.theme_keywords[:5])}")
                        st.write(f"**Representative text:** _{cluster.representative_text[:200]}..._")
                    with col2:
                        st.metric("Coherence", f"{cluster.coherence_score:.3f}")
                        if concept_evolution.concept_persistence:
                            main_theme = (
                                cluster.theme_keywords[0]
                                if cluster.theme_keywords
                                else f"cluster_{cluster.cluster_id}"
                            )
                            persistence = concept_evolution.concept_persistence.get(main_theme, 0)
                            st.metric("Persistence", f"{persistence:.1%}")

        st.markdown("### 📊 Concept Evolution Timeline")
        try:
            from semantic_tensor_memory.visualization.tools.concept_visualizer import (
                visualize_concept_evolution,
            )

            fig = visualize_concept_evolution(concept_evolution, "timeline")
            st.plotly_chart(fig, use_container_width=True, key="overview_concept_timeline")
        except Exception as exc:  # pragma: no cover - visualization path
            st.error(f"Could not generate timeline: {str(exc)}")

        st.markdown("### 🎯 Recommended Next Steps")
        insights = []
        if major_shifts > 2:
            insights.append(
                "🚀 **High conceptual evolution** - Explore the 'Concepts' tab to understand major shifts"
            )
        if concept_evolution.concept_clusters and len(
            concept_evolution.concept_clusters[0].session_indices
        ) > len(concept_evolution.concept_clusters) * 0.6:
            insights.append(
                "🎯 **Dominant theme detected** - Check 'Patterns' tab for deeper thematic analysis"
            )
        if concept_evolution.drift_patterns and any(
            p.drift_magnitude > 0.5 for p in concept_evolution.drift_patterns
        ):
            insights.append("🌊 **Strong semantic shifts** - Use 'Evolution' tab to track temporal changes")

        if insights:
            for insight in insights:
                st.markdown(f"- {insight}")
        else:
            st.markdown("- 📈 **Stable semantic patterns** - Your content shows consistent themes over time")
            st.markdown("- 🔍 **Explore deeper** - Use the 'Concepts' tab for detailed cluster analysis")

        st.markdown("---")
        st.markdown("### 🧭 Explore Your Data")
        exploration_col1, exploration_col2, exploration_col3 = st.columns(3)
        with exploration_col1:
            if st.button("🧠 Deep Concept Analysis", key="goto_concepts"):
                st.info("👉 Navigate to the **'🧠 Concepts'** tab for comprehensive concept analysis!")
        with exploration_col2:
            if st.button("🌊 Evolution Patterns", key="goto_evolution"):
                st.info("👉 Check the **'🌊 Evolution'** tab to track semantic changes over time!")
        with exploration_col3:
            if st.button("🔍 Pattern Discovery", key="goto_patterns"):
                st.info("👉 Visit the **'🔍 Patterns'** tab for advanced pattern analysis!")
    else:
        st.markdown(
            """
        ### 🎯 What You Can Discover

        **Click 'Generate Quick Analysis' above to get:**
        - 🧠 **Main concepts** in your data
        - 📈 **Semantic evolution** patterns
        - 🚀 **Major conceptual shifts**
        - 💡 **Personalized insights** and recommendations

        ### 📊 Your Dataset
        """
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Sessions", session_count)
        with col2:
            total_tokens = dataset_info.get("total_tokens", 0)
            st.metric("🔤 Total Tokens", f"{total_tokens:,}")
        with col3:
            avg_tokens = total_tokens / max(session_count, 1)
            st.metric("📏 Avg Length", f"{avg_tokens:.0f} tokens")

        st.markdown(
            """
        ### 🚀 Ready to Analyze?

        Your semantic tensor memory system is loaded and ready! Here's what each tab offers:

        - **🧠 Concepts**: Advanced concept clustering and evolution analysis
        - **🌊 Evolution**: Track semantic changes over time
        - **🔍 Patterns**: Discover hidden patterns and relationships
        - **📐 Dimensionality**: Compare different analysis methods
        - **🤖 AI Insights**: Get AI-powered analysis and interpretations

        **Start with the quick analysis above, then dive deeper into any tab!**
        """
        )

    if session_count >= 2 and st.session_state.get("memory"):
        with st.expander("🔥 Mini Similarity Heatmap (sampled)", expanded=False):
            max_n = min(30, session_count)
            n = st.slider("Sessions (first N)", 2, max_n, min(10, max_n))
            try:
                fig = plot_heatmap_plotly(st.session_state.memory[:n])
                st.plotly_chart(fig, use_container_width=True, key="overview_mini_heatmap")
                st.caption("Go to 🔍 Patterns → Similarity Heatmap for full view and token alignment.")
            except Exception as exc:  # pragma: no cover - visualization path
                st.error(f"Mini heatmap failed: {exc}")
