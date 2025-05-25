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
    plot_ridgeline_altair
)
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
        
        # Model selection (compact)
        model_options = {
            "Qwen3": "qwen3:latest",
            "Mistral": "mistral:latest"
        }
        selected_model_label = st.selectbox(
            "AI Model:",
            list(model_options.keys()),
            key="model_selection"
        )
        st.session_state["selected_model"] = model_options[selected_model_label]


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
                # Quick PCA visualization
                results = robust_pca_pipeline(
                    st.session_state.memory, 
                    st.session_state.meta, 
                    n_components=2,
                    method='auto'
                )
                if results:
                    fig = create_pca_visualization(results, st.session_state.meta, is_3d=False)
                    st.plotly_chart(fig, use_container_width=True, key="overview_pca_plot")
                else:
                    st.error("Could not generate PCA visualization")
            except Exception as e:
                st.error(f"Could not generate PCA plot: {e}")
        
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
            ["📊 Ridgeline (Feature Evolution)", "🔥 Similarity Heatmap", "🎬 Animated Patterns"],
            horizontal=True
        )
        
        if analysis_type.startswith("📊"):
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
            
            if results:
                # Animated trajectory
                trajectory_fig = create_animated_pca_trajectory(results, st.session_state.meta, speed)
                if trajectory_fig:
                    st.plotly_chart(trajectory_fig, use_container_width=True, key="pattern_animated_trajectory")
                    
                # Variance evolution
                variance_fig = create_variance_evolution_animation(results)
                if variance_fig:
                    st.plotly_chart(variance_fig, use_container_width=True, key="pattern_variance_evolution")
    
        else:
            st.warning("Need ≥2 sessions for pattern analysis.")


def render_dimensionality_tab():
    """Dimensionality reduction analysis and comparison."""
    st.header("📐 Dimensionality Analysis")
    
    if len(st.session_state.memory) > 1:
        # Analysis options
        analysis_mode = st.radio(
            "Analysis mode:",
            ["🎯 Enhanced PCA", "🔬 Method Comparison"],
            horizontal=True
        )
        
        if analysis_mode.startswith("🎯"):
            # Enhanced PCA with timeline
            st.subheader("Enhanced PCA Analysis")
            
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
            
            # Run PCA
            memory_slice = st.session_state.memory[:timeline_idx]
            meta_slice = st.session_state.meta[:timeline_idx]
            
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
                    
                    # Results summary
                    with st.expander("📊 Detailed Results"):
                        results = comparison_results.get('results', {})
                        for method, data in results.items():
                            st.markdown(f"**{method}:**")
                            if isinstance(data, dict):
                                for key, value in data.items():
                                    if isinstance(value, (int, float)):
                                        st.write(f"  - {key}: {value:.3f}")
            
            # Show stored results
            if 'comparison_results' in st.session_state:
                st.markdown("### Previous Comparison Results")
                results = st.session_state['comparison_results'].get('results', {})
                df_results = pd.DataFrame(results).T
                st.dataframe(df_results, use_container_width=True)
    
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