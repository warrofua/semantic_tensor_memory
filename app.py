"""
Semantic Tensor Memory Analysis Application

A clean, intuitive Streamlit application for analyzing semantic evolution in text data.
Redesigned for optimal user experience and workflow.
"""

import streamlit as st
import time
import pandas as pd
import csv
import io
import os
import warnings
import numpy as np

# Fix PyTorch/Streamlit compatibility issues
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set environment variables to prevent PyTorch/Streamlit conflicts
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['TORCH_DISABLE_WARN'] = '1'

# Prevent PyTorch from interfering with Streamlit's file watcher
try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if hasattr(torch, '_C') and hasattr(torch._C, '_disable_jit_profiling'):
        torch._C._disable_jit_profiling()
except Exception:
    pass

# Import our modules
from memory.embedder import embed_sentence
from memory.store import save, append
from memory.drift import drift_series
from streamlit_utils import initialize_session_state, robust_pca_pipeline
from streamlit_plots import (
    plot_drift_plotly, 
    plot_heatmap_plotly, 
    create_pca_visualization,
    create_animated_pca_trajectory,
    create_temporal_heatmap,
    create_variance_evolution_animation,
    plot_ridgeline_plotly,
    plot_enhanced_ridgeline_altair,
    plot_ridgeline_altair,
    create_pca_timeline_animation,
    create_4d_semantic_space_visualization,
    create_liminal_tunnel_visualization
)
from viz.semantic_drift_river import render_semantic_drift_river_analysis
from viz.holistic_semantic_analysis import render_holistic_semantic_analysis
from chat_analysis import render_comprehensive_chat_analysis
from semantic_trajectory import (
    calculate_semantic_trajectory_data,
    create_3d_trajectory_plot,
    display_trajectory_analysis_table
)
from alternative_dimensionality import compare_dimensionality_methods, create_alternative_visualization
from viz.pca_summary import generate_narrative_summary

# Set page config
st.set_page_config(
    page_title="Semantic Tensor Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
initialize_session_state()


def handle_csv_import(uploaded_file):
    """Handle CSV file upload and import with clean feedback."""
    try:
        memory = []
        meta = []
        csv_data = uploaded_file.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(csv_data))
        
        # Process with progress
        rows = list(reader)
        if not rows:
            st.error("❌ The CSV file appears to be empty")
            return False
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        valid_sessions = 0
        for i, row in enumerate(rows):
            progress_bar.progress((i + 1) / len(rows))
            status_text.text(f"Processing row {i + 1} of {len(rows)}...")
            
            text = row.get('text', '').strip()
            if text:
                try:
                    emb = embed_sentence(text)
                    meta_row = dict(row)
                    meta_row['tokens'] = emb.shape[0]
                    memory.append(emb)
                    meta.append(meta_row)
                    valid_sessions += 1
                except Exception:
                    continue  # Skip problematic rows silently
        
        progress_bar.empty()
        status_text.empty()
        
        if memory:
            # Update session state
            st.session_state.memory = memory
            st.session_state.meta = meta
            
            # Update dataset info
            st.session_state.dataset_info = {
                'source': 'csv_import',
                'filename': uploaded_file.name,
                'upload_timestamp': time.time(),
                'session_count': len(memory),
                'total_tokens': sum(m.shape[0] for m in memory)
            }
            
            # Save the data
            save(memory, meta)
            
            st.success(f"✅ Successfully loaded **{uploaded_file.name}** with {len(memory)} sessions")
            return True
        else:
            st.error(f"❌ No valid text data found in {uploaded_file.name}")
            return False
        
    except Exception as e:
        st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
        return False


def render_upload_screen():
    """Render the clean upload interface for new users."""
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1>🧠 Semantic Tensor Memory</h1>
        <h3 style="color: #666;">Analyze how meaning evolves across text sessions</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 📁 Upload Your Data")
        st.markdown("Upload a CSV file with a **'text'** column containing your sessions, documents, or journal entries.")
        
        uploaded_file = st.file_uploader(
            "Choose CSV file",
            type=['csv'],
            help="CSV should have a 'text' column with your text data"
        )
        
        if uploaded_file is not None:
            if handle_csv_import(uploaded_file):
                st.rerun()

        st.markdown("---")
        
        # Example datasets
        st.markdown("### 🎯 Try Example Datasets")
        
        if st.button("📚 Load Demo Dataset", type="primary"):
            # Check if demo dataset exists
            if os.path.exists("demo_dataset.csv"):
                with open("demo_dataset.csv", "rb") as f:
                    if handle_csv_import(type('MockFile', (), {
                        'name': 'demo_dataset.csv',
                        'read': lambda: f.read()
                    })()):
                        st.rerun()
            else:
                st.error("Demo dataset not found. Please upload your own CSV file.")
        
        st.markdown("""
        <div style="background: #f0f2f6; padding: 1rem; border-radius: 0.5rem; margin-top: 1rem;">
        <strong>💡 What can you analyze?</strong>
        <ul>
        <li>📝 Journal entries over time</li>
        <li>📚 Document evolution in a project</li>
        <li>💬 Chat conversations or interviews</li>
        <li>📊 Survey responses across periods</li>
        <li>🎓 Learning journey documentation</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)


def render_simple_sidebar():
    """Render a clean, minimal sidebar."""
    with st.sidebar:
        st.image("semantic_tensor_art_logo.png", width=200)
        
        # Dataset status (compact)
        dataset_info = st.session_state.get('dataset_info', {})
        if dataset_info.get('session_count', 0) > 0:
            filename = dataset_info.get('filename', 'Unknown')
            session_count = dataset_info.get('session_count', 0)
            st.markdown(f"**📁 {filename}**")
            st.markdown(f"🔢 {session_count} sessions")
            
            # Quick actions
            if st.button("🔄 New Dataset"):
                st.session_state.memory = []
                st.session_state.meta = []
                st.session_state.dataset_info = {'session_count': 0}
                st.rerun()
        else:
            st.markdown("**No dataset loaded**")
        
        st.markdown("---")
        
        # Show preferred dimensionality method
        preferred_method = st.session_state.get('preferred_method')
        if preferred_method:
            st.markdown("---")
            st.markdown("**📐 Preferred Method:**")
            st.markdown(f"🎯 {preferred_method.upper()}")
            st.caption("Set in Dimensionality tab")


def render_overview_dashboard():
    """Render the main overview dashboard with key insights."""
    dataset_info = st.session_state.get('dataset_info', {})
    filename = dataset_info.get('filename', 'Unknown Dataset')
    session_count = dataset_info.get('session_count', 0)
    
    # Header
    st.markdown(f"# 🧠 {filename}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Sessions", session_count)
    
    with col2:
        total_tokens = dataset_info.get('total_tokens', 0)
        st.metric("🔤 Total Tokens", f"{total_tokens:,}")
    
    with col3:
        avg_tokens = total_tokens / max(session_count, 1)
        st.metric("📏 Avg Length", f"{avg_tokens:.0f}")
    
    with col4:
        # Quick semantic shift calculation
        if len(st.session_state.memory) > 1:
            try:
                drifts, _ = drift_series(st.session_state.memory)
                avg_drift = np.mean(drifts)
                st.metric("🌊 Avg Drift", f"{avg_drift:.3f}")
            except:
                st.metric("🌊 Avg Drift", "N/A")
        else:
            st.metric("🌊 Avg Drift", "N/A")
    
    st.markdown("---")
    
    # Main visualizations (2 columns)
    if len(st.session_state.memory) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Semantic Evolution")
            try:
                drifts, counts = drift_series(st.session_state.memory)
                fig = plot_drift_plotly(drifts, counts)
                st.plotly_chart(fig, use_container_width=True, key="overview_drift_plot")
            except Exception as e:
                st.error(f"Could not generate drift plot: {e}")
        
        with col2:
            st.subheader("🗺️ Semantic Space")
            try:
                # Check if user has a preferred method from comparison
                if 'preferred_method' in st.session_state and 'method_results' in st.session_state:
                    preferred_method = st.session_state['preferred_method']
                    method_results = st.session_state['method_results']
                    
                    if preferred_method == 'umap' and method_results.get('umap_results'):
                        fig = create_alternative_visualization(
                            method_results['umap_results'], 
                            st.session_state.meta, 
                            "UMAP"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="overview_umap_plot")
                            st.caption("🎯 Using UMAP (your preferred method)")
                        else:
                            # Fallback to PCA
                            results = robust_pca_pipeline(
                                st.session_state.memory, 
                                st.session_state.meta, 
                                n_components=2,
                                method='auto'
                            )
                            if results:
                                fig = create_pca_visualization(results, st.session_state.meta, is_3d=False)
                                st.plotly_chart(fig, use_container_width=True, key="overview_pca_plot")
                                st.caption("📊 Using PCA (fallback)")
                    else:
                        # For t-SNE or other methods, fallback to PCA for now
                        results = robust_pca_pipeline(
                            st.session_state.memory, 
                            st.session_state.meta, 
                            n_components=2,
                            method='auto'
                        )
                        if results:
                            fig = create_pca_visualization(results, st.session_state.meta, is_3d=False)
                            st.plotly_chart(fig, use_container_width=True, key="overview_pca_plot")
                            st.caption(f"📊 Using PCA (preferred {preferred_method.upper()} not available in 2D overview)")
                else:
                    # Default PCA visualization
                    results = robust_pca_pipeline(
                        st.session_state.memory, 
                        st.session_state.meta, 
                        n_components=2,
                        method='auto'
                    )
                    if results:
                        fig = create_pca_visualization(results, st.session_state.meta, is_3d=False)
                        st.plotly_chart(fig, use_container_width=True, key="overview_pca_plot")
                        st.caption("📊 Using PCA (default)")
                    else:
                        st.error("Could not generate PCA visualization")
            except Exception as e:
                st.error(f"Could not generate plot: {e}")
        
        # Quick insights
        st.subheader("🔍 Quick Insights")
        try:
            # Generate a quick narrative summary
            if 'results' in locals() and results:
                summary = generate_narrative_summary(
                    results['reduced'], 
                    results['session_ids'], 
                    results['token_ids'], 
                    st.session_state.meta
                )
                st.info(summary)
            else:
                st.info("💡 Your dataset is ready for analysis. Use the tabs above to explore semantic patterns, evolution, and get AI-powered insights.")
        except:
            st.info("💡 Your dataset is ready for analysis. Use the tabs above to explore semantic patterns, evolution, and get AI-powered insights.")
    
    else:
        st.info("📊 Upload more sessions to see semantic evolution analysis.")


def render_semantic_evolution_tab():
    """Combined semantic evolution analysis (drift + trajectory)."""
    st.header("🌊 Semantic Evolution")
    
    if len(st.session_state.memory) > 1:
        # Evolution over time
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Drift Analysis")
            drifts, counts = drift_series(st.session_state.memory)
            fig = plot_drift_plotly(drifts, counts)
            st.plotly_chart(fig, use_container_width=True, key="evolution_drift_plot")
        
        with col2:
            st.subheader("🎯 3D Trajectory")
            trajectory_data = calculate_semantic_trajectory_data(
                st.session_state.memory, 
                st.session_state.meta
            )
            if trajectory_data:
                fig = create_3d_trajectory_plot(trajectory_data)
                st.plotly_chart(fig, use_container_width=True, key="evolution_3d_trajectory")
        
        # Detailed analysis table
        st.subheader("📊 Session-by-Session Analysis")
        if trajectory_data:
            table_data = display_trajectory_analysis_table(trajectory_data)
            st.dataframe(table_data, use_container_width=True)
    
    else:
        st.warning("Need ≥2 sessions for evolution analysis.")


def render_pattern_analysis_tab():
    """Pattern discovery with multiple visualization techniques."""
    st.header("🔍 Pattern Analysis")
    
    if len(st.session_state.memory) > 1:
        # Analysis type selector
        analysis_type = st.radio(
            "Choose analysis type:",
            ["🌐 Holistic Semantic Analysis (REVOLUTIONARY!)", "🌊 Semantic Drift River (3D)", "📊 Ridgeline (Feature Evolution)", "🔥 Similarity Heatmap", "🎬 Animated Patterns"],
            horizontal=True
        )
        
        if analysis_type.startswith("🌐"):
            # Revolutionary Holistic Semantic Analysis
            render_holistic_semantic_analysis(st.session_state.memory, st.session_state.meta)
        
        elif analysis_type.startswith("🌊"):
            # 3D Semantic Drift River
            render_semantic_drift_river_analysis(st.session_state.memory, st.session_state.meta)
        
        elif analysis_type.startswith("📊"):
            # Enhanced ridgeline
            result = plot_enhanced_ridgeline_altair(
                st.session_state.memory, 
                st.session_state.meta,
                show_trends=True,
                highlight_changes=True
            )
            
            if result and len(result) == 2:
                fig, scaling_info = result
                if fig:
                    st.altair_chart(fig, use_container_width=True)
                    with st.expander("📈 Scaling Details"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Features", scaling_info.get('features', 'N/A'))
                        with col2:
                            st.metric("Value Range", scaling_info.get('value_range', 'N/A'))
                        with col3:
                            st.metric("Chart Height", f"{scaling_info.get('adaptive_height', 'N/A')}px")
        
        elif analysis_type.startswith("🔥"):
            # Heatmap
            fig = plot_heatmap_plotly(st.session_state.memory)
            st.plotly_chart(fig, use_container_width=True, key="pattern_heatmap")
        
        elif analysis_type.startswith("🎬"):
            # Animated analysis
            st.subheader("🎬 Animated Semantic Evolution")
            
            # Animation type selector
            animation_type = st.radio(
                "Choose animation type:",
                ["🎯 Trajectory Evolution", "📈 PCA Over Time", "📊 Variance Build-up", "🌊 Liminal Tunnel", "🌌 4D Semantic Space"],
                horizontal=True
            )
            
            # Quick animation settings
            col1, col2 = st.columns(2)
            with col1:
                speed = st.selectbox("Speed", [300, 500, 800], index=1)
            with col2:
                include_3d = st.checkbox("3D View", value=True)
            
            # Enhanced PCA for animation
            results = robust_pca_pipeline(
                st.session_state.memory, 
                st.session_state.meta, 
                n_components=3 if include_3d else 2,
                method='auto'
            )
            
            # Check if user has a preferred method and can use it
            preferred_method = st.session_state.get('preferred_method')
            if preferred_method:
                st.info(f"🎯 Note: Your preferred method is {preferred_method.upper()}, but animations currently use PCA for temporal consistency.")
            
            if results:
                if animation_type.startswith("🎯"):
                    # Animated trajectory
                    trajectory_fig = create_animated_pca_trajectory(results, st.session_state.meta, speed)
                    if trajectory_fig:
                        st.plotly_chart(trajectory_fig, use_container_width=True, key="pattern_animated_trajectory")
                
                elif animation_type.startswith("📈"):
                    # PCA over time animation
                    st.info("🎥 Watch how the PCA space evolves as sessions are progressively added")
                    timeline_fig = create_pca_timeline_animation(
                        st.session_state.memory,
                        st.session_state.meta,
                        animation_speed=speed,
                        is_3d=include_3d
                    )
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True, key="pattern_pca_timeline")
                        
                        # Quick tips
                        st.markdown("""
                        **💡 Animation Tips:**
                        - 🔴 **Larger dots** = newest session in each frame
                        - 🌈 **Color** = temporal progression (blue → red)
                        - 📊 **Space changes** = how axes reorient with new data
                        - ⏭️ **Use Final** button to jump to complete dataset
                        """)
                
                elif animation_type.startswith("📊"):
                    # Variance evolution
                    variance_fig = create_variance_evolution_animation(results)
                    if variance_fig:
                        st.plotly_chart(variance_fig, use_container_width=True, key="pattern_variance_evolution")
                
                elif animation_type.startswith("🌊"):
                    # Liminal Tunnel visualization - PCA + t-SNE hybrid through temporal space
                    st.info("🌊 Liminal Tunnel: PCA + t-SNE hybrid flowing through 3D temporal space")
                    tunnel_fig = create_liminal_tunnel_visualization(
                        st.session_state.memory,
                        st.session_state.meta
                    )
                    if tunnel_fig:
                        st.plotly_chart(tunnel_fig, use_container_width=True, key="pattern_liminal_tunnel")
                        
                        # Liminal tunnel interpretation
                        st.markdown("""
                        **🌊 Liminal Tunnel Guide:**
                        - **🚇 Tunnel spine** = Smooth temporal path through hybrid PCA-t-SNE space
                        - **💎 Diamond anchors** = Actual session positions in hybrid space
                        - **⚪ Flow particles** = Temporal progression indicators
                        - **🌌 Liminal aesthetics** = Dark space with ethereal colors
                        - **📐 Hybrid dimensions** = Global PCA structure + local t-SNE patterns
                        """)
                    else:
                        st.error("Could not generate liminal tunnel visualization")
                
                elif animation_type.startswith("🌌"):
                    # 4D Semantic Space visualization
                    st.info("🌌 4D Semantic Space: Pure PCA with 4th dimension controlling visual properties")
                    tunnel_fig = create_4d_semantic_space_visualization(
                        st.session_state.memory,
                        st.session_state.meta
                    )
                    if tunnel_fig:
                        st.plotly_chart(tunnel_fig, use_container_width=True, key="pattern_4d_semantic_space")
                        
                        # 4D semantic space interpretation
                        st.markdown("""
                        **🌌 4D Semantic Space Guide:**
                        - **🎯 Rotate & zoom** to explore 4D space from different angles
                        - **🌈 Colors** represent the 4th semantic dimension (PC4)
                        - **📏 Sizes** vary with 4th dimension intensity
                        - **🔗 Connections** show semantic tunnels between sessions
                        - **➡️ Arrows** show temporal flow direction
                        """)
                    else:
                        st.error("Could not generate 4D semantic space visualization")
    
    else:
        st.warning("Need ≥2 sessions for pattern analysis.")


def render_dimensionality_tab():
    """Dimensionality reduction analysis and comparison."""
    st.header("📐 Dimensionality Analysis")
    
    if len(st.session_state.memory) > 1:
        # Analysis options
        analysis_mode = st.radio(
            "Analysis mode:",
            ["🎯 Enhanced PCA", "🔬 Method Comparison", "🌊 Liminal Tunnel Visualization"],
            horizontal=True
        )
        
        if analysis_mode.startswith("🎯"):
            # Enhanced PCA with timeline
            st.subheader("Enhanced PCA Analysis")
            
            # Show current method preference
            preferred_method = st.session_state.get('preferred_method')
            if preferred_method:
                st.info(f"🎯 Your preferred method: **{preferred_method.upper()}** (from method comparison)")
                
                # Option to use preferred method if available
                if preferred_method == 'umap' and 'method_results' in st.session_state:
                    use_preferred = st.checkbox("Use UMAP instead of PCA", value=False, 
                                              help="Use your preferred UMAP method for this analysis")
                else:
                    use_preferred = False
                    if preferred_method != 'pca':
                        st.warning(f"⚠️ {preferred_method.upper()} not available for interactive timeline analysis. Using PCA.")
            else:
                use_preferred = False
            
            col1, col2 = st.columns([3, 1])
            with col1:
                max_session = len(st.session_state.memory)
                timeline_idx = st.slider(
                    "Sessions to include:",
                    2, max_session, max_session,
                    help="Adjust to see how patterns develop over time"
                )
            with col2:
                is_3d = st.checkbox("3D View", value=False)
            
            # Add animation controls
            st.markdown("### 🎬 Animation Options")
            col1, col2, col3 = st.columns(3)
            with col1:
                show_timeline_animation = st.checkbox("📈 PCA Over Time Animation", 
                                                    value=False,
                                                    help="See how PCA space evolves as sessions are added")
            with col2:
                if show_timeline_animation:
                    animation_speed = st.selectbox("Animation Speed", [500, 800, 1200], index=1,
                                                  help="Milliseconds per frame")
                else:
                    animation_speed = 800
            with col3:
                if show_timeline_animation:
                    use_3d_animation = st.checkbox("3D Animation", value=is_3d,
                                                  help="Use 3D view for timeline animation")
                else:
                    use_3d_animation = is_3d
            
            # Run analysis with preferred method if available and selected
            if use_preferred and 'method_results' in st.session_state:
                # Use cached UMAP results
                method_results = st.session_state['method_results']
                if method_results.get('umap_results'):
                    fig = create_alternative_visualization(
                        method_results['umap_results'], 
                        st.session_state.meta, 
                        "UMAP"
                    )
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key="dimensionality_enhanced_umap")
                        st.caption("🎯 Using UMAP (your preferred method)")
                        
                        # Show UMAP metrics
                        umap_results = method_results['umap_results']
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Method", "UMAP")
                        with col2:
                            trust_score = method_results.get('results', {}).get('UMAP', {}).get('trust_score', 'N/A')
                            st.metric("Trust Score", f"{trust_score:.3f}" if isinstance(trust_score, (int, float)) else trust_score)
                        with col3:
                            st.metric("Samples", f"{umap_results.get('n_samples', 'N/A'):,}")
                else:
                    st.error("UMAP results not available. Falling back to PCA.")
                    use_preferred = False
            
            if not use_preferred:
                # Standard PCA analysis
                memory_slice = st.session_state.memory[:timeline_idx]
                meta_slice = st.session_state.meta[:timeline_idx]
                
                if show_timeline_animation:
                    # Show PCA over time animation
                    st.subheader("🎬 PCA Space Evolution Over Time")
                    st.info("🎥 Watch how the semantic space develops as sessions are progressively added")
                    
                    timeline_fig = create_pca_timeline_animation(
                        st.session_state.memory[:timeline_idx],
                        st.session_state.meta[:timeline_idx],
                        animation_speed=animation_speed,
                        is_3d=use_3d_animation
                    )
                    
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True, key="dimensionality_pca_timeline")
                        
                        # Animation interpretation guide
                        with st.expander("🎭 How to Interpret the Animation"):
                            st.markdown("""
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
                            """)
                else:
                    # Standard static analysis
                    results = robust_pca_pipeline(
                        memory_slice, meta_slice, 
                        n_components=3 if is_3d else 2,
                        method='auto'
                    )
                    
                    if results:
                        # Visualization
                        fig = create_pca_visualization(results, meta_slice, is_3d=is_3d)
                        st.plotly_chart(fig, use_container_width=True, key="dimensionality_enhanced_pca")
                        
                        # Quality metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Explained Variance", f"{results['cumulative_variance'][-1]:.1%}")
                        with col2:
                            st.metric("Quality", results['quality_assessment'].title())
                        with col3:
                            st.metric("Method", results['method_used'])
                        with col4:
                            st.metric("Features", f"{results['n_features']:,}")
        
        elif analysis_mode.startswith("🔬"):
            # Method comparison
            st.subheader("Method Comparison")
            
            if st.button("🚀 Compare Methods", type="primary"):
                with st.spinner("Comparing PCA, UMAP, and t-SNE..."):
                    comparison_results = compare_dimensionality_methods(
                        st.session_state.memory, 
                        st.session_state.meta
                    )
                
                if comparison_results:
                    st.session_state['comparison_results'] = comparison_results
                    
                    # Show best method
                    best_method = comparison_results['best_method']
                    st.success(f"🏆 Best method: **{best_method}**")
                    
                    # Visualization of best method
                    if best_method == "UMAP" and comparison_results.get('umap_results'):
                        fig = create_alternative_visualization(
                            comparison_results['umap_results'], 
                            st.session_state.meta, 
                            "UMAP"
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True, key="dimensionality_method_comparison")
                    
                    # Add actionable buttons
                    st.markdown("### 🎯 Apply Recommendation")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button(f"✅ Use {best_method} as Default", type="primary"):
                            st.session_state['preferred_method'] = best_method.lower()
                            st.session_state['method_results'] = comparison_results
                            st.success(f"✅ {best_method} is now your default dimensionality reduction method!")
                            st.info("💡 Other visualizations in the app will now use this method when possible.")
                            st.rerun()
                    
                    with col2:
                        if st.button("🔄 Reset to Auto"):
                            if 'preferred_method' in st.session_state:
                                del st.session_state['preferred_method']
                            if 'method_results' in st.session_state:
                                del st.session_state['method_results']
                            st.success("🔄 Reset to automatic method selection.")
                            st.rerun()
                    
                    # Results summary
                    with st.expander("📊 Detailed Results"):
                        results = comparison_results.get('results', {})
                        for method, data in results.items():
                            st.markdown(f"**{method}:**")
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"  - {key}: {value:.3f}")
            
            # Show current method preference
            if 'preferred_method' in st.session_state:
                preferred = st.session_state['preferred_method'].upper()
                st.info(f"🎯 **Current default method**: {preferred}")
                
                # Show when this method is being used
                st.markdown("**This method is automatically used in:**")
                st.markdown("- 🏠 Overview dashboard visualizations")
                st.markdown("- 🌊 Evolution tab PCA plots")
                st.markdown("- 🔍 Pattern analysis animations")
            
            # Show stored results
            if 'comparison_results' in st.session_state:
                st.markdown("### Previous Comparison Results")
                results = st.session_state['comparison_results'].get('results', {})
                df_results = pd.DataFrame(results).T
                st.dataframe(df_results, use_container_width=True)
        
        elif analysis_mode.startswith("🌊"):
            # Liminal Tunnel Visualization
            st.subheader("🌊 Liminal Tunnel Visualization")
            st.info("🚇 Journey through hybrid PCA-t-SNE space with temporal tunneling effects")
            
            # Tunnel type selector
            tunnel_type = st.radio(
                "Choose tunnel type:",
                ["🌊 Liminal Tunnel (PCA + t-SNE)", "🌌 4D Semantic Space (Pure PCA)"],
                horizontal=True
            )
            
            # Tunnel visualization controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Generate Tunnel", type="primary"):
                    if tunnel_type.startswith("🌊"):
                        with st.spinner("Creating liminal tunnel through PCA-t-SNE hybrid space..."):
                            tunnel_fig = create_liminal_tunnel_visualization(
                                st.session_state.memory,
                                st.session_state.meta
                            )
                            tunnel_key = 'liminal_tunnel_visualization'
                    else:
                        with st.spinner("Creating 4D semantic space visualization..."):
                            tunnel_fig = create_4d_semantic_space_visualization(
                                st.session_state.memory,
                                st.session_state.meta
                            )
                            tunnel_key = '4d_semantic_space_visualization'
                    
                    if tunnel_fig:
                        st.session_state[tunnel_key] = tunnel_fig
                        st.success(f"✅ {tunnel_type.split(' ')[1]} visualization created!")
                    else:
                        st.error(f"❌ Could not create {tunnel_type.lower()} visualization")
            
            with col2:
                # Clear button for active visualizations
                active_visualizations = []
                if 'liminal_tunnel_visualization' in st.session_state:
                    active_visualizations.append('Liminal Tunnel')
                if '4d_semantic_space_visualization' in st.session_state:
                    active_visualizations.append('4D Semantic Space')
                
                if active_visualizations:
                    clear_option = st.selectbox("Clear visualization:", ["None"] + active_visualizations)
                    if st.button("🔄 Clear Selected"):
                        if clear_option == "Liminal Tunnel":
                            del st.session_state['liminal_tunnel_visualization']
                        elif clear_option == "4D Semantic Space":
                            del st.session_state['4d_semantic_space_visualization']
                        if clear_option != "None":
                            st.success(f"🗑️ {clear_option} visualization cleared")
                            st.rerun()
            
            # Show stored visualizations
            if 'liminal_tunnel_visualization' in st.session_state:
                st.markdown("### 🌊 Liminal Tunnel")
                st.plotly_chart(
                    st.session_state['liminal_tunnel_visualization'], 
                    use_container_width=True, 
                    key="dimensionality_liminal_tunnel"
                )
                
                # Liminal tunnel interpretation guide
                with st.expander("🌊 How to Interpret the Liminal Tunnel"):
                    st.markdown("""
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
                    """)
            
            if '4d_semantic_space_visualization' in st.session_state:
                st.markdown("### 🌌 4D Semantic Space")
                st.plotly_chart(
                    st.session_state['4d_semantic_space_visualization'], 
                    use_container_width=True, 
                    key="dimensionality_4d_semantic_space"
                )
                
                # 4D semantic space interpretation guide
                with st.expander("🌌 How to Interpret the 4D Semantic Space"):
                    st.markdown("""
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
                    """)
            
            # Show information if no visualizations are active
            if not ('liminal_tunnel_visualization' in st.session_state or '4d_semantic_space_visualization' in st.session_state):
                st.markdown("### 💡 About Tunnel Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
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
                    """)
                
                with col2:
                    st.markdown("""
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
                    """)
                
                st.info("🚀 Choose a tunnel type and click **'Generate Tunnel'** to begin your journey!")
    
        else:
            st.warning("Need ≥2 sessions for dimensionality analysis.")


def main():
    """Main application with clean, intuitive interface."""
    # Clean up any rerun flags
    if st.session_state.get("csv_imported", False):
        st.session_state["csv_imported"] = False
    
    # Check if we have data
    has_data = len(st.session_state.get('memory', [])) > 0
    
    if not has_data:
        # Show upload screen for new users
        render_upload_screen()
    else:
        # Show main app interface
        render_simple_sidebar()
        
        # Main tabs (simplified to 4 focused areas)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Overview", 
            "🌊 Evolution", 
            "🔍 Patterns", 
            "📐 Dimensionality",
            "🤖 AI Insights"
        ])
        
        with tab1:
            render_overview_dashboard()
        
        with tab2:
            render_semantic_evolution_tab()
        
        with tab3:
            render_pattern_analysis_tab()
        
        with tab4:
            render_dimensionality_tab()
        
        with tab5:
            st.header("🤖 AI-Powered Analysis")
            render_comprehensive_chat_analysis()


if __name__ == "__main__":
    main() 