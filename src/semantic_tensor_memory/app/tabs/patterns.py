"""Pattern analysis tab."""

from __future__ import annotations

import numpy as np
import streamlit as st

from streamlit_plots import (
    create_4d_semantic_space_visualization,
    create_animated_pca_trajectory,
    create_liminal_tunnel_visualization,
    create_pca_timeline_animation,
    create_temporal_heatmap,
    create_variance_evolution_animation,
    plot_enhanced_ridgeline_altair,
    plot_heatmap_plotly,
    robust_pca_pipeline,
)
from viz.holistic_semantic_analysis import render_holistic_semantic_analysis
from viz.semantic_drift_river import render_semantic_drift_river_analysis
from viz.heatmap import token_alignment_heatmap

__all__ = ["render_pattern_analysis_tab"]


def render_pattern_analysis_tab() -> None:
    """Pattern discovery with multiple visualization techniques."""
    st.header("🔍 Pattern Analysis")

    if len(st.session_state.memory) <= 1:
        st.warning("Need ≥2 sessions for pattern analysis.")
        return

    analysis_type = st.radio(
        "Choose analysis type:",
        [
            "🌐 Holistic Semantic Analysis (REVOLUTIONARY!)",
            "🌊 Semantic Drift River (3D)",
            "📊 Ridgeline (Feature Evolution)",
            "🔥 Similarity Heatmap",
            "🎬 Animated Patterns",
        ],
        horizontal=True,
    )

    if analysis_type.startswith("🌐"):
        render_holistic_semantic_analysis(st.session_state.memory, st.session_state.meta)
        return
    if analysis_type.startswith("🌊"):
        render_semantic_drift_river_analysis(st.session_state.memory, st.session_state.meta)
        return
    if analysis_type.startswith("📊"):
        result = plot_enhanced_ridgeline_altair(
            st.session_state.memory,
            st.session_state.meta,
            show_trends=True,
            highlight_changes=True,
        )
        if result and len(result) == 2:
            fig, scaling_info = result
            if fig:
                st.altair_chart(fig, use_container_width=True)
                with st.expander("📈 Scaling Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Features", scaling_info.get("features", "N/A"))
                    with col2:
                        st.metric("Value Range", scaling_info.get("value_range", "N/A"))
                    with col3:
                        st.metric(
                            "Chart Height",
                            f"{scaling_info.get('adaptive_height', 'N/A')}px",
                        )
        return
    if analysis_type.startswith("🔥"):
        fig = plot_heatmap_plotly(st.session_state.memory)
        st.plotly_chart(fig, use_container_width=True, key="pattern_heatmap")
        with st.expander("🔎 Token Alignment Heatmap (pairwise)"):
            if len(st.session_state.memory) >= 2:
                col_i, col_j = st.columns(2)
                with col_i:
                    i = st.number_input(
                        "Session i",
                        min_value=1,
                        max_value=len(st.session_state.memory),
                        value=1,
                    )
                with col_j:
                    j = st.number_input(
                        "Session j",
                        min_value=1,
                        max_value=len(st.session_state.memory),
                        value=2,
                    )
                if st.button("Show token alignment", key="show_token_alignment"):
                    try:
                        fig_align = token_alignment_heatmap(
                            st.session_state.memory, int(i) - 1, int(j) - 1
                        )
                        if fig_align is not None:
                            st.pyplot(fig_align, use_container_width=True)
                    except Exception as exc:  # pragma: no cover - visualization path
                        st.error(f"Token alignment heatmap failed: {exc}")
        return

    st.subheader("🎬 Animated Semantic Evolution")
    animation_type = st.radio(
        "Choose animation type:",
        [
            "🎯 Trajectory Evolution",
            "📈 PCA Over Time",
            "📊 Variance Build-up",
            "🌊 Liminal Tunnel",
            "🌌 4D Semantic Space",
            "🔥 Temporal Similarity (sliding window)",
        ],
        horizontal=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        speed = st.selectbox("Speed", [300, 500, 800], index=1)
    with col2:
        include_3d = st.checkbox("3D View", value=True)

    results = robust_pca_pipeline(
        st.session_state.memory,
        st.session_state.meta,
        n_components=3 if include_3d else 2,
        method="auto",
    )

    preferred_method = st.session_state.get("preferred_method")
    if preferred_method:
        st.info(
            f"🎯 Note: Your preferred method is {preferred_method.upper()}, but animations currently use PCA for temporal consistency."
        )

    if not results:
        st.error("Unable to compute PCA pipeline for animations.")
        return

    if animation_type.startswith("🎯"):
        trajectory_fig = create_animated_pca_trajectory(results, st.session_state.meta, speed)
        if trajectory_fig:
            st.plotly_chart(trajectory_fig, use_container_width=True, key="pattern_animated_trajectory")
            with st.expander("🧠 Axis Explainer (LLM)"):
                try:
                    from viz.semantic_analysis import analyze_pca_patterns

                    reduced = results["reduced"]
                    session_ids = np.array(results["session_ids"])
                    pca1 = reduced[:, 0]
                    idxs = np.argsort(pca1)
                    sample_idxs = (
                        np.concatenate([idxs[:3], idxs[-3:]]) if len(idxs) >= 6 else idxs
                    )
                    texts = [
                        st.session_state.meta[session_ids[i]].get("text", "")
                        for i in sample_idxs
                    ]
                    scores = [float(pca1[i]) for i in sample_idxs]
                    if st.button("Explain axes", key="explain_axes_traj"):
                        summary = analyze_pca_patterns(texts, scores)
                        st.write(summary)
                except Exception as exc:  # pragma: no cover - optional feature
                    st.caption(f"Axis explainer unavailable: {exc}")
        return

    if animation_type.startswith("📈"):
        st.info("🎥 Watch how the PCA space evolves as sessions are progressively added")
        timeline_fig = create_pca_timeline_animation(
            st.session_state.memory,
            st.session_state.meta,
            animation_speed=speed,
            is_3d=include_3d,
        )
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True, key="pattern_pca_timeline")
            st.markdown(
                """
                **💡 Animation Tips:**
                - 🔴 **Larger dots** = newest session in each frame
                - 🌈 **Color** = temporal progression (blue → red)
                - 📊 **Space changes** = how axes reorient with new data
                - ⏭️ **Use Final** button to jump to complete dataset
                """
            )
        return

    if animation_type.startswith("📊"):
        variance_fig = create_variance_evolution_animation(results)
        if variance_fig:
            st.plotly_chart(variance_fig, use_container_width=True, key="pattern_variance_evolution")
        return

    if animation_type.startswith("🌊"):
        st.info("🌊 Liminal Tunnel: PCA + t-SNE hybrid flowing through 3D temporal space")
        tunnel_fig = create_liminal_tunnel_visualization(
            st.session_state.memory,
            st.session_state.meta,
        )
        if tunnel_fig:
            st.plotly_chart(tunnel_fig, use_container_width=True, key="pattern_liminal_tunnel")
            st.markdown(
                """
                **🌊 Liminal Tunnel Guide:**
                - **🚇 Tunnel spine** = Smooth temporal path through hybrid PCA-t-SNE space
                - **💎 Diamond anchors** = Actual session positions in hybrid space
                - **⚪ Flow particles** = Temporal progression indicators
                - **🌌 Liminal aesthetics** = Dark space with ethereal colors
                - **📐 Hybrid dimensions** = Global PCA structure + local t-SNE patterns
                """
            )
        else:
            st.error("Could not generate liminal tunnel visualization")
        return

    if animation_type.startswith("🌌"):
        st.info("🌌 4D Semantic Space: Pure PCA with 4th dimension controlling visual properties")
        tunnel_fig = create_4d_semantic_space_visualization(
            st.session_state.memory,
            st.session_state.meta,
        )
        if tunnel_fig:
            st.plotly_chart(tunnel_fig, use_container_width=True, key="pattern_4d_semantic_space")
            st.markdown(
                """
                **🌌 4D Semantic Space Guide:**
                - **🎯 Rotate & zoom** to explore 4D space from different angles
                - **🌈 Colors** represent the 4th semantic dimension (PC4)
                - **📏 Sizes** vary with 4th dimension intensity
                - **🔗 Connections** show semantic tunnels between sessions
                - **➡️ Arrows** show temporal flow direction
                """
            )
        else:
            st.error("Could not generate 4D semantic space visualization")
        return

    window = st.slider(
        "Window size",
        min_value=3,
        max_value=min(12, len(st.session_state.memory)),
        value=5,
    )
    temp_fig = create_temporal_heatmap(results, st.session_state.meta, window_size=window)
    if temp_fig:
        st.plotly_chart(temp_fig, use_container_width=True, key="pattern_temporal_similarity")
