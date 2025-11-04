"""Semantic evolution tab."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from semantic_tensor_memory.analytics import (
    calculate_semantic_trajectory_data,
    create_3d_trajectory_plot,
    display_trajectory_analysis_table,
)
from semantic_tensor_memory.memory.sequence_drift import (
    semantic_coherence_score,
    token_importance_drift,
)
from semantic_tensor_memory.streamlit.plots import plot_drift_plotly
from semantic_tensor_memory.visualization import token_alignment_heatmap

from ..services import compute_drift_series

__all__ = ["render_semantic_evolution_tab"]


def render_semantic_evolution_tab() -> None:
    """Combined semantic evolution analysis (drift + trajectory)."""
    st.header("🌊 Semantic Evolution")

    if len(st.session_state.memory) <= 1:
        st.warning("Need ≥2 sessions for evolution analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📈 Drift Analysis")
        drifts, counts = compute_drift_series(st.session_state.memory)
        if drifts:
            fig = plot_drift_plotly(drifts, counts)
            st.plotly_chart(fig, use_container_width=True, key="evolution_drift_plot")
        else:
            st.info("Not enough data to compute drift series yet.")

    with col2:
        st.subheader("🎯 3D Trajectory")
        trajectory_data = calculate_semantic_trajectory_data(
            st.session_state.memory, st.session_state.meta
        )
        if trajectory_data:
            fig = create_3d_trajectory_plot(trajectory_data)
            st.plotly_chart(fig, use_container_width=True, key="evolution_3d_trajectory")
        else:
            st.info("Trajectory analysis requires richer embeddings.")

    st.subheader("📊 Session-by-Session Analysis")
    if trajectory_data:
        table_data = display_trajectory_analysis_table(trajectory_data)
        st.dataframe(table_data, use_container_width=True)

    with st.expander("🔎 Token Drift Alignment (consecutive sessions)"):
        if len(st.session_state.memory) >= 2:
            max_pair = len(st.session_state.memory) - 1
            pair_idx = st.slider("Step (i → i+1)", 1, max_pair, 1)
            if st.button("Show alignment", key="evolution_token_alignment"):
                try:
                    fig_align = token_alignment_heatmap(
                        st.session_state.memory, pair_idx - 1, pair_idx
                    )
                    if fig_align is not None:
                        st.pyplot(fig_align, use_container_width=True)
                except Exception as exc:  # pragma: no cover - visualization path
                    st.error(f"Token alignment failed: {exc}")

    with st.expander("📌 Token Importance Drift & Coherence"):
        try:
            top_k = st.slider("Top drifting tokens (K)", 5, 20, 10)
            top = token_importance_drift(st.session_state.memory, top_k=top_k)
            if top:
                df_top = pd.DataFrame(top, columns=["SessionIndex", "DriftScore"])
                df_top["SessionIndex"] = df_top["SessionIndex"].astype(int) + 1
                st.dataframe(df_top, use_container_width=True)
            coherences = [semantic_coherence_score(t) for t in st.session_state.memory]
            st.line_chart(pd.DataFrame({"Coherence": coherences}))
        except Exception as exc:  # pragma: no cover - optional analysis
            st.caption(f"Drift/coherence summary unavailable: {exc}")
