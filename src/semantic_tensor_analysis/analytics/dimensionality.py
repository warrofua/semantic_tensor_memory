"""Alternative dimensionality reduction methods for Semantic Tensor Memory.

This module provides UMAP, t-SNE, and other advanced dimensionality reduction
techniques as alternatives to PCA for better handling of high-dimensional semantic data.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler


def try_import_umap():
    """Try to import UMAP, with fallback handling."""
    try:
        import umap
        return umap
    except ImportError:
        st.warning("UMAP not installed. Run `pip install umap-learn` for better dimensionality reduction.")
        return None


def umap_analysis(memory_slice, meta_slice, n_components=2, n_neighbors=15, min_dist=0.1):
    """
    Apply UMAP dimensionality reduction for better handling of semantic embeddings.
    
    Args:
        memory_slice: List of tensors for sessions
        meta_slice: Metadata for sessions
        n_components: Number of UMAP components
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
    
    Returns:
        dict containing UMAP results or None if processing failed
    """
    umap_lib = try_import_umap()
    if umap_lib is None:
        return None
    
    try:
        from semantic_tensor_analysis.visualization.viz.pca_plot import prepare_for_pca
        
        # Prepare data
        flat, session_ids, token_ids = prepare_for_pca(memory_slice)
        
        # Data validation
        if flat.shape[0] < n_components:
            st.error(f"Insufficient data points for UMAP analysis.")
            return None
        
        # Robust preprocessing
        scaler = RobustScaler()
        flat_scaled = scaler.fit_transform(flat)
        
        # Apply UMAP
        with st.spinner("Running UMAP analysis..."):
            reducer = umap_lib.UMAP(
                n_components=n_components,
                n_neighbors=min(n_neighbors, flat_scaled.shape[0] - 1),
                min_dist=min_dist,
                random_state=42,
                metric='cosine'  # Good for semantic embeddings
            )
            reduced = reducer.fit_transform(flat_scaled)
        
        # Calculate trust score (how well local structure is preserved)
        try:
            from sklearn.neighbors import NearestNeighbors
            # Find local neighborhoods in original space
            nn_orig = NearestNeighbors(n_neighbors=min(10, flat_scaled.shape[0]//2))
            nn_orig.fit(flat_scaled)
            orig_neighbors = nn_orig.kneighbors(flat_scaled, return_distance=False)
            
            # Find local neighborhoods in reduced space
            nn_reduced = NearestNeighbors(n_neighbors=min(10, reduced.shape[0]//2))
            nn_reduced.fit(reduced)
            reduced_neighbors = nn_reduced.kneighbors(reduced, return_distance=False)
            
            # Calculate neighborhood preservation
            trust_scores = []
            for i in range(len(orig_neighbors)):
                orig_set = set(orig_neighbors[i])
                reduced_set = set(reduced_neighbors[i])
                trust_scores.append(len(orig_set.intersection(reduced_set)) / len(orig_set))
            
            avg_trust = np.mean(trust_scores)
        except Exception:
            avg_trust = 0.0
        
        results = {
            'reduced': reduced,
            'reducer': reducer,
            'session_ids': session_ids,
            'token_ids': token_ids,
            'method_used': 'UMAP',
            'n_samples': flat_scaled.shape[0],
            'n_features': flat_scaled.shape[1],
            'trust_score': avg_trust,
            'scaler': scaler,
            'parameters': {
                'n_neighbors': reducer.n_neighbors,
                'min_dist': reducer.min_dist,
                'metric': reducer.metric
            }
        }
        
        return results
        
    except Exception as e:
        st.error(f"UMAP analysis failed: {e}")
        return None


def enhanced_tsne_analysis(memory_slice, meta_slice, n_components=2, perplexity=30, learning_rate=200):
    """
    Apply enhanced t-SNE with better parameters for semantic embeddings.
    
    Args:
        memory_slice: List of tensors for sessions
        meta_slice: Metadata for sessions
        n_components: Number of t-SNE components
        perplexity: Perplexity parameter for t-SNE
        learning_rate: Learning rate for t-SNE
    
    Returns:
        dict containing t-SNE results or None if processing failed
    """
    try:
        from semantic_tensor_analysis.visualization.viz.pca_plot import prepare_for_pca
        
        # Prepare data
        flat, session_ids, token_ids = prepare_for_pca(memory_slice)
        
        # Data validation
        if flat.shape[0] < 4:
            st.error(f"Insufficient data points for t-SNE analysis.")
            return None
        
        # Robust preprocessing
        scaler = RobustScaler()
        flat_scaled = scaler.fit_transform(flat)
        
        # Optimize t-SNE parameters based on data size
        n_samples = flat_scaled.shape[0]
        optimal_perplexity = min(perplexity, max(5, n_samples // 4))
        
        # Apply t-SNE
        with st.spinner("Running enhanced t-SNE analysis..."):
            reducer = TSNE(
                n_components=n_components,
                perplexity=optimal_perplexity,
                learning_rate=learning_rate,
                n_iter=1000,
                random_state=42,
                metric='cosine',
                init='pca'  # Better initialization
            )
            reduced = reducer.fit_transform(flat_scaled)
        
        # Calculate KL divergence (lower is better)
        kl_divergence = reducer.kl_divergence_
        
        results = {
            'reduced': reduced,
            'reducer': reducer,
            'session_ids': session_ids,
            'token_ids': token_ids,
            'method_used': 'Enhanced t-SNE',
            'n_samples': flat_scaled.shape[0],
            'n_features': flat_scaled.shape[1],
            'kl_divergence': kl_divergence,
            'scaler': scaler,
            'parameters': {
                'perplexity': optimal_perplexity,
                'learning_rate': learning_rate,
                'n_iter': reducer.n_iter
            }
        }
        
        return results
        
    except Exception as e:
        st.error(f"Enhanced t-SNE analysis failed: {e}")
        return None


def create_alternative_visualization(results, meta_slice, method_name="Alternative Method"):
    """Create visualization for alternative dimensionality reduction results."""
    if results is None:
        return None
    
    reduced = results['reduced']
    session_ids = results['session_ids']
    
    # Create DataFrame for plotting
    if reduced.shape[1] >= 2:
        df = pd.DataFrame({
            'Dim1': reduced[:, 0],
            'Dim2': reduced[:, 1],
            'Session': [f"Session {j+1}" for j in session_ids],
            'SessionIdx': session_ids,
            'Text': [meta_slice[j]['text'] for j in session_ids]
        })
        
        if reduced.shape[1] >= 3:
            df['Dim3'] = reduced[:, 2]
            
            # 3D plot
            fig = px.scatter_3d(
                df,
                x='Dim1',
                y='Dim2',
                z='Dim3',
                color='SessionIdx',
                color_continuous_scale='RdYlBu',
                hover_data=['Session', 'Text'],
                title=f"3D {method_name} Semantic Analysis"
            )
            fig.update_traces(marker=dict(size=4, reversescale=True))
        else:
            # 2D plot
            fig = px.scatter(
                df,
                x='Dim1',
                y='Dim2',
                color='SessionIdx',
                color_continuous_scale='RdYlBu',
                hover_data=['Session', 'Text'],
                title=f"2D {method_name} Semantic Analysis"
            )
            fig.update_traces(marker=dict(reversescale=True))
        
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            coloraxis_colorbar=dict(title="Session")
        )
        
        return fig
    
    return None


def compare_dimensionality_methods(memory_slice, meta_slice):
    """
    Compare multiple dimensionality reduction methods and recommend the best one.
    
    Args:
        memory_slice: List of tensors for sessions
        meta_slice: Metadata for sessions
    
    Returns:
        dict containing comparison results and recommendations
    """
    st.subheader("🔬 Dimensionality Reduction Method Comparison")
    
    results = {}
    
    # Run multiple methods
    with st.columns(3)[0]:
        st.write("**Testing PCA...**")
        from semantic_tensor_analysis.streamlit.utils import robust_pca_pipeline
        pca_results = robust_pca_pipeline(memory_slice, meta_slice, n_components=2, method='auto')
        if pca_results:
            results['PCA'] = {
                'explained_variance': pca_results['cumulative_variance'][-1],
                'quality_score': pca_results.get('quality_score', 0),
                'method_used': pca_results.get('method_used', 'PCA'),
                'condition_number': pca_results.get('condition_number', float('inf'))
            }
    
    with st.columns(3)[1]:
        st.write("**Testing UMAP...**")
        umap_results = umap_analysis(memory_slice, meta_slice, n_components=2)
        if umap_results:
            results['UMAP'] = {
                'trust_score': umap_results.get('trust_score', 0),
                'n_neighbors': umap_results['parameters']['n_neighbors'],
                'min_dist': umap_results['parameters']['min_dist']
            }
    
    with st.columns(3)[2]:
        st.write("**Testing t-SNE...**")
        tsne_results = enhanced_tsne_analysis(memory_slice, meta_slice, n_components=2)
        if tsne_results:
            results['t-SNE'] = {
                'kl_divergence': tsne_results.get('kl_divergence', float('inf')),
                'perplexity': tsne_results['parameters']['perplexity']
            }
    
    # Generate recommendations
    recommendations = []
    
    if 'PCA' in results:
        pca_score = results['PCA']['explained_variance']
        if pca_score > 0.7:
            recommendations.append("✅ **PCA** works excellently for your data")
        elif pca_score > 0.3:
            recommendations.append("⚠️ **PCA** has moderate performance")
        else:
            recommendations.append("❌ **PCA** explains little variance - try alternatives")
    
    if 'UMAP' in results:
        trust_score = results['UMAP']['trust_score']
        if trust_score > 0.7:
            recommendations.append("✅ **UMAP** preserves local structure excellently")
        elif trust_score > 0.5:
            recommendations.append("⚠️ **UMAP** has moderate structure preservation")
        else:
            recommendations.append("❌ **UMAP** may be distorting relationships")
    
    if 't-SNE' in results:
        kl_div = results['t-SNE']['kl_divergence']
        if kl_div < 1.0:
            recommendations.append("✅ **t-SNE** converged to good solution")
        elif kl_div < 2.0:
            recommendations.append("⚠️ **t-SNE** has moderate convergence")
        else:
            recommendations.append("❌ **t-SNE** may need parameter tuning")
    
    # Display results
    if results:
        st.markdown("### 📊 Comparison Results")
        comparison_df = pd.DataFrame.from_dict(results, orient='index')
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("### 💡 Recommendations")
        for rec in recommendations:
            st.markdown(f"- {rec}")
        
        # Overall recommendation
        best_method = "PCA"  # Default
        if 'UMAP' in results and results['UMAP']['trust_score'] > 0.6:
            best_method = "UMAP"
        elif 'PCA' in results and results['PCA']['explained_variance'] > 0.5:
            best_method = "PCA"
        elif 't-SNE' in results:
            best_method = "t-SNE"
        
        st.success(f"🎯 **Recommended method for your data**: {best_method}")
        
        return {
            'results': results,
            'recommendations': recommendations,
            'best_method': best_method,
            'pca_results': pca_results if 'pca_results' in locals() else None,
            'umap_results': umap_results if 'umap_results' in locals() else None,
            'tsne_results': tsne_results if 'tsne_results' in locals() else None
        }
    
    return None 
