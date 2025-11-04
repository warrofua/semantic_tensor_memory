"""Dimensionality analysis tab."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from alternative_dimensionality import compare_dimensionality_methods, create_alternative_visualization
from streamlit_plots import (
    create_4d_semantic_space_visualization,
    create_liminal_tunnel_visualization,
    create_pca_timeline_animation,
    robust_pca_pipeline,
)

__all__ = ["render_dimensionality_tab"]


def render_dimensionality_tab() -> None:
    """Dimensionality reduction analysis and comparison."""
    st.header("📐 Dimensionality Analysis")

    if len(st.session_state.memory) <= 1:
        st.warning("Need ≥2 sessions for dimensionality analysis.")
        return

    analysis_mode = st.radio(
        "Analysis mode:",
        ["🎯 Enhanced PCA", "🔬 Method Comparison", "🌊 Liminal Tunnel Visualization"],
        horizontal=True,
    )

    if analysis_mode.startswith("🎯"):
        preferred_method = st.session_state.get("preferred_method")
        if preferred_method:
            st.info(f"🎯 Your preferred method: **{preferred_method.upper()}** (from method comparison)")
            if preferred_method == "umap" and "method_results" in st.session_state:
                use_preferred = st.checkbox(
                    "Use UMAP instead of PCA",
                    value=False,
                    help="Use your preferred UMAP method for this analysis",
                )
            else:
                use_preferred = False
                if preferred_method != "pca":
                    st.warning(
                        f"⚠️ {preferred_method.upper()} not available for interactive timeline analysis. Using PCA."
                    )
        else:
            use_preferred = False

        col1, col2 = st.columns([3, 1])
        with col1:
            max_session = len(st.session_state.memory)
            timeline_idx = st.slider(
                "Sessions to include:",
                2,
                max_session,
                max_session,
                help="Adjust to see how patterns develop over time",
            )
        with col2:
            is_3d = st.checkbox("3D View", value=False)

        st.markdown("### 🎬 Animation Options")
        col1, col2, col3 = st.columns(3)
        with col1:
            show_timeline_animation = st.checkbox(
                "📈 PCA Over Time Animation",
                value=False,
                help="See how PCA space evolves as sessions are added",
            )
        with col2:
            if show_timeline_animation:
                animation_speed = st.selectbox(
                    "Animation Speed", [500, 800, 1200], index=1, help="Milliseconds per frame"
                )
            else:
                animation_speed = 800
        with col3:
            if show_timeline_animation:
                use_3d_animation = st.checkbox(
                    "3D Animation", value=is_3d, help="Use 3D view for timeline animation"
                )
            else:
                use_3d_animation = is_3d

        st.markdown("### 🔬 Point Granularity")
        granularity = st.radio(
            "Show points as:", ["Session means", "Tokens (sampled)"], index=0, horizontal=True
        )

        if use_preferred and "method_results" in st.session_state:
            method_results = st.session_state["method_results"]
            if method_results.get("umap_results"):
                fig = create_alternative_visualization(
                    method_results["umap_results"],
                    st.session_state.meta,
                    "UMAP",
                )
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="dimensionality_enhanced_umap")
                    st.caption("🎯 Using UMAP (your preferred method)")

                    umap_results = method_results["umap_results"]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Method", "UMAP")
                    with col2:
                        trust_score = method_results.get("results", {}).get("UMAP", {}).get(
                            "trust_score", "N/A"
                        )
                        if isinstance(trust_score, (int, float)):
                            trust_display = f"{trust_score:.3f}"
                        else:
                            trust_display = trust_score
                        st.metric("Trust Score", trust_display)
                    with col3:
                        st.metric("Samples", f"{umap_results.get('n_samples', 'N/A'):,}")
                else:
                    st.error("UMAP results not available. Falling back to PCA.")
                    use_preferred = False
        if not use_preferred:
            memory_slice = st.session_state.memory[:timeline_idx]
            meta_slice = st.session_state.meta[:timeline_idx]

            if show_timeline_animation:
                st.subheader("🎬 PCA Space Evolution Over Time")
                st.info("🎥 Watch how the semantic space develops as sessions are progressively added")

                timeline_fig = create_pca_timeline_animation(
                    st.session_state.memory[:timeline_idx],
                    st.session_state.meta[:timeline_idx],
                    animation_speed=animation_speed,
                    is_3d=use_3d_animation,
                )

                if timeline_fig:
                    st.plotly_chart(timeline_fig, use_container_width=True, key="dimensionality_pca_timeline")
                    with st.expander("🎭 How to Interpret the Animation"):
                        st.markdown(
                            """
                            **🎬 Animation Guide:**
                            - **▶️ Play Timeline**: Watch sessions being added chronologically
                            - **Larger dots**: Newest session in each frame
                            - **Color progression**: Blue → Red shows temporal order
                            - **Space shifts**: How PCA axes reorient as data grows
                            - **Quality changes**: Explained variance evolving over time

                            **🔍 What to Look For:**
                            - **Stable patterns**: Consistent positioning across frames
                            - **Sudden shifts**: New sessions dramatically changing the space
                            - **Convergence**: Space stabilizing as more data is added
                            - **Outliers**: Sessions that reshape the entire analysis
                            """
                        )
            else:
                results = robust_pca_pipeline(
                    memory_slice,
                    meta_slice,
                    n_components=3 if is_3d else 2,
                    method="auto",
                )

                if results:
                    if granularity.startswith("Tokens"):
                        try:
                            reduced = results["reduced"]
                            session_ids = np.array(results["session_ids"])
                            max_points = 2000
                            if reduced.shape[0] > max_points:
                                idx = np.linspace(0, reduced.shape[0] - 1, max_points, dtype=int)
                                reduced = reduced[idx]
                                session_ids = session_ids[idx]
                            df = pd.DataFrame(
                                {
                                    "PC1": reduced[:, 0],
                                    "PC2": reduced[:, 1],
                                    "SessionIdx": session_ids,
                                }
                            )
                            if is_3d and reduced.shape[1] > 2:
                                df["PC3"] = reduced[:, 2]
                                st.scatter_chart(df.rename(columns={"SessionIdx": "color"}))
                            else:
                                st.scatter_chart(df.rename(columns={"SessionIdx": "color"}))
                        except Exception as exc:  # pragma: no cover - fallback view
                            st.caption(f"Token-level visualization unavailable: {exc}")
                    else:
                        st.write("PCA results ready – explore via other tabs and analyses.")

    elif analysis_mode.startswith("🔬"):
        st.subheader("🔬 Method Comparison")
        with st.spinner("Comparing PCA, t-SNE, and UMAP..."):
            comparison_results = compare_dimensionality_methods(
                st.session_state.memory,
                st.session_state.meta,
            )
        st.session_state["comparison_results"] = comparison_results

        best_method = comparison_results.get("best_method", "PCA")
        st.success(f"🎯 Recommended method: **{best_method}**")

        if best_method == "UMAP" and comparison_results.get("umap_results"):
            fig = create_alternative_visualization(
                comparison_results["umap_results"],
                st.session_state.meta,
                "UMAP",
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True, key="dimensionality_method_comparison")

        st.markdown("### 🎯 Apply Recommendation")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"✅ Use {best_method} as Default", type="primary"):
                st.session_state["preferred_method"] = best_method.lower()
                st.session_state["method_results"] = comparison_results
                st.success(f"✅ {best_method} is now your default dimensionality reduction method!")
                st.info("💡 Other visualizations in the app will now use this method when possible.")
                st.rerun()
        with col2:
            if st.button("🔄 Reset to Auto"):
                st.session_state.pop("preferred_method", None)
                st.session_state.pop("method_results", None)
                st.success("🔄 Reset to automatic method selection.")
                st.rerun()

        with st.expander("📊 Detailed Results"):
            results = comparison_results.get("results", {})
            for method, data in results.items():
                st.markdown(f"**{method}:**")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            st.write(f"  - {key}: {value:.3f}")

        if "comparison_results" in st.session_state:
            st.markdown("### Previous Comparison Results")
            results = st.session_state["comparison_results"].get("results", {})
            df_results = pd.DataFrame(results).T
            st.dataframe(df_results, use_container_width=True)

    else:
        st.subheader("🌊 Liminal Tunnel Visualization")
        st.info("🚇 Journey through hybrid PCA-t-SNE space with temporal tunneling effects")

        tunnel_type = st.radio(
            "Choose tunnel type:",
            ["🌊 Liminal Tunnel (PCA + t-SNE)", "🌌 4D Semantic Space (Pure PCA)"],
            horizontal=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Generate Tunnel", type="primary"):
                if tunnel_type.startswith("🌊"):
                    with st.spinner("Creating liminal tunnel through PCA-t-SNE hybrid space..."):
                        tunnel_fig = create_liminal_tunnel_visualization(
                            st.session_state.memory, st.session_state.meta
                        )
                        tunnel_key = "liminal_tunnel_visualization"
                else:
                    with st.spinner("Creating 4D semantic space visualization..."):
                        tunnel_fig = create_4d_semantic_space_visualization(
                            st.session_state.memory, st.session_state.meta
                        )
                        tunnel_key = "4d_semantic_space_visualization"

                if tunnel_fig:
                    st.session_state[tunnel_key] = tunnel_fig
                    st.success(f"✅ {tunnel_type.split(' ')[1]} visualization created!")
                else:
                    st.error(f"❌ Could not create {tunnel_type.lower()} visualization")
        with col2:
            active_visualizations = []
            if "liminal_tunnel_visualization" in st.session_state:
                active_visualizations.append("Liminal Tunnel")
            if "4d_semantic_space_visualization" in st.session_state:
                active_visualizations.append("4D Semantic Space")

            if active_visualizations:
                clear_option = st.selectbox("Clear visualization:", ["None"] + active_visualizations)
                if st.button("🔄 Clear Selected"):
                    if clear_option == "Liminal Tunnel":
                        st.session_state.pop("liminal_tunnel_visualization", None)
                    elif clear_option == "4D Semantic Space":
                        st.session_state.pop("4d_semantic_space_visualization", None)
                    if clear_option != "None":
                        st.success(f"🗑️ {clear_option} visualization cleared")
                        st.rerun()

        if "liminal_tunnel_visualization" in st.session_state:
            st.markdown("### 🌊 Liminal Tunnel")
            st.plotly_chart(
                st.session_state["liminal_tunnel_visualization"],
                use_container_width=True,
                key="dimensionality_liminal_tunnel",
            )
            with st.expander("🌊 How to Interpret the Liminal Tunnel"):
                st.markdown(
                    """
                    **🌊 Liminal Tunnel Features:**
                    - **🚇 Tunnel Spine**: Smooth spline path through hybrid PCA-t-SNE space
                    - **💎 Session Anchors**: Diamond markers at actual session positions
                    - **⚪ Flow Particles**: White particles showing temporal progression
                    - **🌌 Tunnel Surface**: Semi-transparent surface creating tunnel effect
                    - **🎨 Liminal Aesthetics**: Dark ethereal space with plasma colors

                    **🔍 What to Look For:**
                    - **Tunnel curvature**: How PCA global structure + t-SNE local patterns combine
                    - **Session positioning**: Where sessions anchor in the hybrid space
                    - **Color progression**: Temporal flow from purple to yellow
                    - **Tunnel width**: Varies based on session variance and characteristics
                    - **Smooth transitions**: Spline interpolation creating fluid movement

                    **🎯 Interactive Features:**
                    - **Rotate & Zoom**: Explore the liminal space from different angles
                    - **Hover**: Get session details and coordinates
                    - **Flow visualization**: Follow particles along temporal path
                    """
                )

        if "4d_semantic_space_visualization" in st.session_state:
            st.markdown("### 🌌 4D Semantic Space")
            st.plotly_chart(
                st.session_state["4d_semantic_space_visualization"],
                use_container_width=True,
                key="dimensionality_4d_semantic_space",
            )
            with st.expander("🌌 How to Interpret the 4D Semantic Space"):
                st.markdown(
                    """
                    **🌌 4D Semantic Space Features:**
                    - **📍 Session Centers**: Large spheres representing session centroids
                    - **🔗 Semantic Tunnels**: Connect consecutive sessions through 4D space
                    - **🌈 4th Dimension Colors**: PC4 values control visual properties
                    - **📏 Variable Sizes**: Marker and tunnel sizes based on PC4 intensity
                    - **➡️ Flow Arrows**: Cone arrows showing temporal direction

                    **🔍 What to Look For:**
                    - **Color intensity**: High PC4 values = complex semantic patterns
                    - **Size variations**: Large elements = high 4th dimension activity
                    - **Tunnel paths**: How sessions connect through 4D semantic space
                    - **Arrow flow**: Temporal progression through the space
                    - **Clustering**: Periods of similar 4D semantic positioning

                    **🎯 Interactive Features:**
                    - **4D Exploration**: Rotate to see different 4D perspectives
                    - **Hover Details**: Get complete 4D coordinates and session info
                    - **Legend Control**: Toggle elements on/off
                    """
                )

        if not (
            "liminal_tunnel_visualization" in st.session_state
            or "4d_semantic_space_visualization" in st.session_state
        ):
            st.markdown("### 💡 About Tunnel Visualizations")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    """
                    **🌊 Liminal Tunnel**

                    The **Liminal Tunnel** creates a hybrid visualization combining:
                    - **PCA global structure** (70% weight)
                    - **t-SNE local relationships** (30% weight)
                    - **Smooth spline interpolation** for temporal flow
                    - **Ethereal liminal aesthetics** for immersive experience

                    **Perfect for:**
                    - Exploring temporal semantic evolution
                    - Seeing both global and local patterns
                    - Understanding transitional phases
                    - Immersive data exploration
                    """
                )
            with col2:
                st.markdown(
                    """
                    **🌌 4D Semantic Space**

                    The **4D Semantic Space** provides:
                    - **Pure 4D PCA analysis** with all components
                    - **4th dimension visual mapping** to properties
                    - **Semantic tunnels** connecting sessions
                    - **Temporal flow indicators** with arrows

                    **Perfect for:**
                    - Understanding high-dimensional relationships
                    - Exploring 4th semantic dimension effects
                    - Analyzing complex semantic patterns
                    - Mathematical precision in visualization
                    """
                )
            st.info("🚀 Choose a tunnel type and click **'Generate Tunnel'** to begin your journey!")
