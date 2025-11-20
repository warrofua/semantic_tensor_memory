"""Explainability dashboard tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from semantic_tensor_analysis.explainability.engine import ExplainabilityEngine

__all__ = ["render_explainability_dashboard"]


def render_explainability_dashboard() -> None:
    """Render the explainability dashboard with analysis explanations."""
    st.header("💡 Understanding Your Analysis Results")

    dataset_info = st.session_state.get("dataset_info", {})
    has_performance_data = "performance_metrics" in dataset_info
    has_concept_analysis = hasattr(st.session_state, "concept_evolution")
    has_memory_data = len(st.session_state.get("memory", [])) > 0

    if not has_memory_data:
        st.info("📊 Upload data to see explanations and quality assessments.")
        return

    try:
        explainability_engine = ExplainabilityEngine()

        st.subheader("🎯 Processing Quality Assessment")
        if has_performance_data:
            performance = dataset_info["performance_metrics"]
            derived = False
        else:
            # Attempt a lightweight fallback so the tab doesn't feel empty
            session_count = dataset_info.get("session_count", len(st.session_state.get("memory", [])))
            processing_time = dataset_info.get("processing_time", 0) or 0
            sessions_per_second = (session_count / processing_time) if processing_time else 0.0
            performance = {
                "sessions_per_second": sessions_per_second,
                "memory_efficiency": float(session_count) if session_count else 0.0,
                "success_rate": 1.0 if session_count else 0.0,
                "estimated_quality": 0.5 if session_count else 0.0,
            }
            derived = True

        if performance["sessions_per_second"] or performance["success_rate"]:
            quality_explanation = explainability_engine.explain_processing_quality(
                success_rate=performance["success_rate"],
                memory_efficiency=performance["memory_efficiency"],
                processing_speed=performance["sessions_per_second"],
                estimated_quality=performance["estimated_quality"],
            )

            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown("### 📊 What These Numbers Mean")
                st.markdown(f"**Quality Score: {performance['estimated_quality']:.1%}**")
                st.markdown(quality_explanation.what_it_means)
                st.markdown("### 🔍 Why These Results?")
                st.markdown(quality_explanation.why_these_results)
                if quality_explanation.what_to_do_next:
                    st.markdown("### 💡 What You Can Do")
                    for rec in quality_explanation.what_to_do_next:
                        st.markdown(f"- {rec}")
            with col2:
                quality_score = performance["estimated_quality"]
                if quality_score > 0.8:
                    st.success("🟢 **Excellent Quality**")
                    st.markdown("Your data processed beautifully!")
                elif quality_score > 0.6:
                    st.warning("🟡 **Good Quality**")
                    st.markdown("Solid results with room for improvement.")
                else:
                    st.error("🔴 **Needs Attention**")
                    st.markdown("Several issues detected.")
                st.metric("Success Rate", f"{performance['success_rate']:.1%}")
                st.metric("Memory Efficiency", f"{performance['memory_efficiency']:.1f}")
                st.metric("Processing Speed", f"{performance['sessions_per_second']:.1f}/s")
            if derived:
                st.info("ℹ️ Showing inferred quality (no recorded processing metrics). Re-run processing for full assessment.")
        else:
            st.info("🔄 Process data to see quality assessment")

        st.markdown("---")
        st.subheader("📊 Dataset Complexity Analysis")
        if has_performance_data:
            complexity_score = dataset_info.get("complexity_score", 0)
            processing_strategy = dataset_info.get("processing_strategy", "full_processing")
            complexity_explanation = explainability_engine.explain_complexity_score(
                complexity_score, processing_strategy
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Complexity Score", f"{complexity_score:.2f}/1.0")
                if complexity_score > 0.7:
                    st.markdown("🔴 **High Complexity**")
                    st.markdown("Large, diverse dataset")
                elif complexity_score > 0.4:
                    st.markdown("🟡 **Medium Complexity**")
                    st.markdown("Moderate size and diversity")
                else:
                    st.markdown("🟢 **Low Complexity**")
                    st.markdown("Small, focused dataset")
            with col2:
                st.markdown("**Processing Strategy**")
                strategy_display = processing_strategy.replace("_", " ").title()
                st.markdown(f"🎯 {strategy_display}")
                if processing_strategy == "progressive_sampling":
                    st.markdown("Applied intelligent sampling")
                elif processing_strategy == "smart_batching":
                    st.markdown("Used optimized batching")
                else:
                    st.markdown("Full processing applied")
            with col3:
                st.markdown("**Recommendation**")
                st.markdown(complexity_explanation.what_it_means)

        st.markdown("---")
        st.subheader("🧠 Analysis Results Explainer")

        if has_concept_analysis:
            concept_evolution = st.session_state.concept_evolution
            st.markdown("### 🎯 Concept Analysis Results")
            n_clusters = len(concept_evolution.concept_clusters)
            n_sessions = concept_evolution.total_sessions
            cluster_explanation = explainability_engine.explain_clustering_results(
                n_clusters=n_clusters,
                n_sessions=n_sessions,
                cluster_quality_score=0.7,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Clustering Quality")
                st.markdown(cluster_explanation.what_it_means)
                st.markdown(cluster_explanation.why_these_results)
            with col2:
                st.markdown("#### 💡 What This Means")
                for rec in cluster_explanation.what_to_do_next[:3]:
                    st.markdown(f"- {rec}")

        st.subheader("📚 Token & Session Insights")
        try:
            tokens_per_session = [t.shape[0] for t in st.session_state.memory]
            avg_tokens = np.mean(tokens_per_session) if tokens_per_session else 0
            max_tokens = np.max(tokens_per_session) if tokens_per_session else 0
            padded_ratio = 0.0
            if tokens_per_session and max_tokens > 0:
                total_slots = len(tokens_per_session) * max_tokens
                total_real = sum(tokens_per_session)
                padded_ratio = 1 - (total_real / total_slots)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg tokens/session", f"{avg_tokens:.0f}")
            with col2:
                st.metric("Max tokens", f"{max_tokens}")
            with col3:
                st.metric("Padding ratio", f"{padded_ratio:.1%}")
            st.caption(
                "We use padding + masks to compute stable session means and token-level PCA without biasing distances."
            )
        except Exception:  # pragma: no cover - optional stats
            pass

        st.subheader("❓ Frequently Asked Questions")
        with st.expander("🤔 **Why is my success rate low?**"):
            st.markdown(
                """
                **Common Reasons:**
                - **Noisy Data**: Text contains many non-readable characters or formatting
                - **Empty Sessions**: Some rows have no meaningful text content
                - **Encoding Issues**: File encoding problems (try UTF-8)
                - **Large File Size**: Memory constraints causing processing failures

                **Solutions:**
                - Clean your data before upload
                - Use intelligent sampling for large datasets
                - Check file encoding and format
                """
            )
        with st.expander("📊 **How do I improve clustering quality?**"):
            st.markdown(
                """
                **For Better Clusters:**
                - **More Data**: At least 20-50 sessions for meaningful clusters
                - **Consistent Topics**: Data should have some thematic coherence
                - **Sufficient Text**: Each session should have meaningful content
                - **Optimal Cluster Number**: Try different cluster numbers (3-15 typically work well)

                **Red Flags:**
                - Too many tiny clusters → Reduce cluster count
                - One giant cluster → Increase cluster count or check data diversity
                """
            )
        with st.expander("🧠 **What does high concept drift mean?**"):
            st.markdown(
                """
                **High Drift Indicates:**
                - **Learning Journey**: You're exploring new topics over time
                - **Project Evolution**: Your focus is shifting as you progress
                - **Conversation Dynamics**: AI conversations evolving in complexity

                **This is Often GOOD:**
                - Shows intellectual growth
                - Indicates active exploration
                - Demonstrates learning progression

                **Only Concerning If:**
                - You expected consistency but see chaos
                - The drift doesn't match your memory of the conversations
                """
            )
        with st.expander("⚡ **My processing is slow - what can I do?**"):
            st.markdown(
                """
                **Speed Optimization:**
                - **Use Sampling**: For datasets >1000 sessions, try intelligent sampling
                - **Smaller Batches**: Process in smaller chunks
                - **Close Other Apps**: Free up system memory
                - **Check File Size**: Very large files may need preprocessing

                **Memory Optimization:**
                - **Progressive Analysis**: Build complexity gradually
                - **Quality Over Quantity**: Sometimes less data gives better insights
                - **System Resources**: 16GB+ RAM recommended for large datasets
                """
            )

        st.subheader("🚀 Performance Recommendations")
        if has_performance_data:
            performance = dataset_info["performance_metrics"]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🎯 For Your Next Upload")
                if performance["success_rate"] < 0.9:
                    st.markdown("- 🧹 **Clean your data** - Remove empty rows and formatting issues")
                if performance["memory_efficiency"] < 2.0:
                    st.markdown("- 📊 **Use sampling** - Try intelligent sampling for large datasets")
                if performance["sessions_per_second"] < 1.0:
                    st.markdown("- ⚡ **Optimize processing** - Close other applications to free memory")
                st.markdown("- 📈 **Gradual complexity** - Start with small datasets, build up")
            with col2:
                st.markdown("### 💡 Analysis Strategy")
                dataset_size = dataset_info.get("session_count", 0)
                if dataset_size > 500:
                    st.markdown("- 🎯 **Use progressive sampling** for initial exploration")
                    st.markdown("- 🔄 **Smart batching** for full analysis")
                elif dataset_size > 100:
                    st.markdown("- 📊 **Full processing** with quality monitoring")
                    st.markdown("- 🧠 **Focus on concept evolution** analysis")
                else:
                    st.markdown("- ✅ **Perfect size** for comprehensive analysis")
                    st.markdown("- 🌟 **Try all analysis methods** to compare")
    except Exception as exc:  # pragma: no cover - explanation engine path
        st.error(f"Error initializing explainability engine: {str(exc)}")
        st.info("💡 The explainability features require the analysis modules to be properly loaded.")
