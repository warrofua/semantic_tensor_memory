"""Streamlit utility functions for Semantic Tensor Memory.

This module contains utility functions and helpers for the Streamlit interface,
including session management, data validation, and analysis helpers.
"""

import streamlit as st
import time
import torch
import numpy as np
from memory.embedder import embed_sentence
from memory.store import append
import re
from collections import Counter
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import TruncatedSVD
import warnings
import requests


def initialize_session_state():
    """Initialize session state variables."""
    if 'memory' not in st.session_state:
        from memory.store import load
        st.session_state.memory, st.session_state.meta = load()

    # Initialize chat STM in session state
    if 'chat_memory' not in st.session_state:
        st.session_state.chat_memory = []
        st.session_state.chat_meta = []
    
    # Initialize dataset tracking information
    if 'dataset_info' not in st.session_state:
        st.session_state.dataset_info = {
            'source': 'default_load',  # 'default_load', 'csv_import', 'manual_entry'
            'filename': None,
            'upload_timestamp': None,
            'session_count': len(st.session_state.memory) if 'memory' in st.session_state else 0,
            'total_tokens': sum(m.shape[0] for m in st.session_state.memory) if 'memory' in st.session_state else 0
        }
    
    # Update session count if it's changed
    if 'memory' in st.session_state:
        current_count = len(st.session_state.memory)
        if st.session_state.dataset_info['session_count'] != current_count:
            st.session_state.dataset_info['session_count'] = current_count
            st.session_state.dataset_info['total_tokens'] = sum(m.shape[0] for m in st.session_state.memory)


def add_chat_message(role, text):
    """Add a chat message to the chat memory."""
    emb = embed_sentence(text)
    meta_row = {
        "ts": time.time(),
        "role": role,
        "text": text,
        "tokens": emb.shape[0]
    }
    st.session_state.chat_memory.append(emb)
    st.session_state.chat_meta.append(meta_row)


def is_ollama_model_available(model_name):
    """Check if an Ollama model is available locally."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            return any(model_name in m.get("name", "") for m in tags)
    except Exception:
        pass
    return False


def remove_highly_correlated_features(X, threshold=0.95):
    """Remove highly correlated features to reduce multicollinearity."""
    try:
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Find pairs of highly correlated features
        upper_tri = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        high_corr_pairs = np.where((np.abs(corr_matrix) > threshold) & upper_tri)
        
        # Remove one feature from each highly correlated pair
        features_to_remove = set()
        for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
            # Remove the feature with higher index (arbitrary choice)
            features_to_remove.add(max(i, j))
        
        # Keep features that are not highly correlated
        features_to_keep = [i for i in range(X.shape[1]) if i not in features_to_remove]
        
        if len(features_to_remove) > 0:
            st.info(f"Removed {len(features_to_remove)} highly correlated features (correlation > {threshold})")
            return X[:, features_to_keep], features_to_keep
        
        return X, list(range(X.shape[1]))
    except Exception as e:
        st.warning(f"Could not remove correlated features: {e}")
        return X, list(range(X.shape[1]))


def remove_low_variance_features(X, threshold=1e-6):
    """Remove features with very low variance."""
    try:
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)
        
        removed_count = X.shape[1] - X_selected.shape[1]
        if removed_count > 0:
            st.info(f"Removed {removed_count} low-variance features (variance < {threshold})")
        
        return X_selected, selector.get_support()
    except Exception as e:
        st.warning(f"Could not remove low-variance features: {e}")
        return X, np.ones(X.shape[1], dtype=bool)


def robust_outlier_detection(X, method='iqr', factor=1.5):
    """Detect and handle outliers in the data."""
    try:
        if method == 'iqr':
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Clip outliers instead of removing them to preserve sample size
            X_clipped = np.clip(X, lower_bound, upper_bound)
            
            outlier_count = np.sum(np.any((X < lower_bound) | (X > upper_bound), axis=1))
            if outlier_count > 0:
                st.info(f"Clipped {outlier_count} outlier samples using IQR method")
            
            return X_clipped
        
        return X
    except Exception as e:
        st.warning(f"Could not detect outliers: {e}")
        return X


def robust_pca_pipeline(memory_slice, meta_slice, n_components=2, return_scaler=False, 
                       use_incremental=False, variance_threshold=0.5, method='auto'):
    """
    Enhanced robust PCA pipeline with better preprocessing and multiple methods.
    
    Args:
        memory_slice: List of tensors for sessions
        meta_slice: Metadata for sessions
        n_components: Number of PCA components (2 or 3)
        return_scaler: Whether to return the fitted scaler
        use_incremental: Use IncrementalPCA for large datasets
        variance_threshold: Minimum acceptable explained variance
        method: 'pca', 'svd', or 'auto' for automatic selection
    
    Returns:
        dict containing analysis results or None if processing failed
    """
    from tensor_batching import pad_and_stack, flatten_with_mask
    
    try:
        # Step 1: Prepare data with pad + mask and flatten
        batch_tensor, mask_tensor = pad_and_stack(memory_slice)
        flat_t, session_ids_t, token_ids_t = flatten_with_mask(batch_tensor, mask_tensor)
        flat = flat_t.numpy()
        session_ids = session_ids_t
        token_ids = token_ids_t

        # Mask diagnostics
        total_slots = int(mask_tensor.numel())
        valid_slots = int(mask_tensor.sum().item())
        masked_slots = total_slots - valid_slots
        mask_ratio = masked_slots / max(1, total_slots)
        
        # Step 2: Enhanced data validation
        if flat.shape[0] < n_components:
            st.error(f"Insufficient data points ({flat.shape[0]}) for {n_components}-component analysis. Need at least {n_components} points.")
            return None
            
        if flat.shape[1] < n_components:
            st.error(f"Insufficient dimensions ({flat.shape[1]}) for {n_components}-component analysis. Need at least {n_components} dimensions.")
            return None
        
        # Step 3: Enhanced data cleaning
        if np.isnan(flat).any() or np.isinf(flat).any():
            st.error("Data contains NaN or Inf values after preprocessing. Cannot perform analysis.")
            return None
        
        # Step 4: Outlier detection and handling
        flat = robust_outlier_detection(flat, method='iqr', factor=2.0)
        
        # Step 5: Remove low-variance features
        flat, variance_mask = remove_low_variance_features(flat, threshold=1e-8)
        
        # Step 6: Remove highly correlated features to reduce multicollinearity
        flat, correlation_mask = remove_highly_correlated_features(flat, threshold=0.98)
        
        # Step 7: Enhanced standardization using RobustScaler
        try:
            # Try RobustScaler first (less sensitive to outliers)
            scaler = RobustScaler()
            flat_scaled = scaler.fit_transform(flat)
            scaler_type = "RobustScaler"
        except Exception:
            # Fallback to StandardScaler
            scaler = StandardScaler()
            flat_scaled = scaler.fit_transform(flat)
            scaler_type = "StandardScaler"
        
        # Step 8: Check for remaining multicollinearity
        try:
            condition_number = np.linalg.cond(np.cov(flat_scaled.T))
        except Exception:
            condition_number = float('inf')
        
        # Step 9: Choose appropriate dimensionality reduction method
        if method == 'auto':
            if condition_number > 1e12 or flat_scaled.shape[0] > 10000:
                method = 'svd'  # More stable for ill-conditioned data
            elif flat_scaled.shape[0] > 5000 and use_incremental:
                method = 'incremental_pca'
            else:
                method = 'pca'
        
        # Step 10: Apply dimensionality reduction
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            
            if method == 'svd':
                # Use TruncatedSVD which is more numerically stable
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(flat_scaled)
                explained_var_ratio = reducer.explained_variance_ratio_
                components = reducer.components_
                method_used = "TruncatedSVD"
                
            elif method == 'incremental_pca':
                # Use IncrementalPCA for large datasets
                reducer = IncrementalPCA(n_components=n_components, batch_size=min(200, flat_scaled.shape[0]//5))
                reduced = reducer.fit_transform(flat_scaled)
                explained_var_ratio = reducer.explained_variance_ratio_
                components = reducer.components_
                method_used = "IncrementalPCA"
                
            else:
                # Standard PCA
                reducer = PCA(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(flat_scaled)
                explained_var_ratio = reducer.explained_variance_ratio_
                components = reducer.components_
                method_used = "StandardPCA"
        
        # Step 11: Validate results
        if np.isnan(reduced).any() or np.isinf(reduced).any():
            st.error(f"{method_used} transformation produced NaN or Inf values.")
            return None
        
        # Step 12: Calculate metrics
        cumulative_var = np.cumsum(explained_var_ratio)
        
        # Step 13: Adaptive component suggestion
        if cumulative_var[-1] < variance_threshold:
            # Suggest optimal number of components
            if method != 'svd':  # SVD already uses all components efficiently
                temp_pca = PCA(random_state=42)
                temp_pca.fit(flat_scaled)
                
                # Find number of components needed for desired variance
                temp_cumvar = np.cumsum(temp_pca.explained_variance_ratio_)
                optimal_components = np.argmax(temp_cumvar >= variance_threshold) + 1
                
                if optimal_components <= 10:  # Reasonable upper limit
                    st.info(f"💡 **Tip**: Current analysis explains {cumulative_var[-1]:.1%} of variance. "
                           f"Consider using {optimal_components} components to explain {variance_threshold:.0%} of variance.")
        
        # Step 14: Quality assessment
        quality_score = cumulative_var[-1] * (1 - min(1.0, condition_number / 1e12))
        if quality_score > 0.6:
            quality = "excellent"
        elif quality_score > 0.4:
            quality = "good"
        elif quality_score > 0.2:
            quality = "fair"
        else:
            quality = "concerning"
        
        # Step 15: Prepare comprehensive results
        results = {
            'reduced': reduced,
            'pca': reducer,  # Keep for compatibility
            'reducer': reducer,
            'session_ids': session_ids,
            'token_ids': token_ids,
            'explained_variance_ratio': explained_var_ratio,
            'cumulative_variance': cumulative_var,
            'n_samples': flat_scaled.shape[0],
            'n_features': flat_scaled.shape[1],
            'original_features': flat.shape[1] if 'flat' in locals() else flat_scaled.shape[1],
            'condition_number': condition_number,
            'method_used': method_used,
            'scaler_type': scaler_type,
            'quality_score': quality_score,
            'quality_assessment': quality,
            'components': components
        }
        
        if return_scaler:
            results['scaler'] = scaler
        
        # Step 16: Display improvement suggestions and mask diagnostics
        with st.expander("🔧 Analysis Improvements Applied", expanded=False):
            st.markdown(f"""
            **Preprocessing Enhancements:**
            - ✅ **Outlier handling**: IQR-based clipping applied
            - ✅ **Feature selection**: Removed low-variance and highly correlated features
            - ✅ **Robust scaling**: Used {scaler_type} for better standardization
            - ✅ **Method selection**: Used {method_used} for optimal stability
            
            **Quality Metrics:**
            - **Explained variance**: {cumulative_var[-1]:.1%}
            - **Condition number**: {condition_number:.2e}
            - **Quality score**: {quality_score:.2f}/1.0 ({quality})
            - **Features processed**: {results['original_features']} → {results['n_features']}
            """)
            st.markdown(f"""
            **Masking Diagnostics:**
            - **Total slots**: {total_slots}
            - **Valid (unmasked)**: {valid_slots}
            - **Masked**: {masked_slots} ({mask_ratio:.1%})
            - **Samples used**: {flat_scaled.shape[0]}
            """)
            
            if condition_number > 1e10:
                st.warning("⚠️ High condition number detected. Consider using more aggressive feature selection or regularization.")
            
            if cumulative_var[-1] < 0.3:
                st.info("💡 Low explained variance is common with high-dimensional semantic embeddings. Consider using UMAP or autoencoders for better dimensionality reduction.")
        
        # Attach mask stats
        results['mask_stats'] = {
            'total_slots': total_slots,
            'valid_slots': valid_slots,
            'masked_slots': masked_slots,
            'mask_ratio': mask_ratio,
            'samples_used': int(flat_scaled.shape[0])
        }

        return results
        
    except Exception as e:
        st.error(f"Enhanced PCA pipeline failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


def extract_meaningful_words(text, min_length=3):
    """Extract meaningful words from text, filtering out common words."""
    # Convert to lowercase and extract words
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'a', 'to', 'of', 'in', 'is', 'that', 'with', 'was', 'for', 'as', 'on', 'at', 'by', 
        'this', 'but', 'not', 'from', 'or', 'an', 'are', 'it', 'have', 'has', 'had', 'if', 'they', 'their', 
        'there', 'what', 'when', 'where', 'which', 'who', 'why', 'how', 'will', 'would', 'could', 'should',
        'been', 'being', 'do', 'does', 'did', 'can', 'may', 'might', 'must', 'shall', 'said', 'say', 'says',
        'told', 'tell', 'asked', 'ask', 'get', 'got', 'go', 'went', 'come', 'came', 'see', 'saw', 'know',
        'knew', 'think', 'thought', 'feel', 'felt', 'look', 'looked', 'take', 'took', 'give', 'gave', 'make',
        'made', 'find', 'found', 'use', 'used', 'work', 'worked', 'way', 'time', 'day', 'year', 'week', 'month'
    }
    
    # Filter and return meaningful words
    meaningful_words = [w for w in words if len(w) >= min_length and w not in stop_words]
    return meaningful_words


def get_dominant_themes(texts, top_n=3):
    """Get the most common themes from a list of texts."""
    all_words = []
    for text in texts:
        words = extract_meaningful_words(text)
        all_words.extend(words)
    
    if not all_words:
        return []
    
    # Count word frequencies
    word_counts = Counter(all_words)
    # Get top words, but avoid very short or very long ones
    filtered_words = [(word, count) for word, count in word_counts.items() 
                     if 3 <= len(word) <= 12 and count > 1]
    
    if not filtered_words:
        # Fallback to all words if filtering is too strict
        filtered_words = list(word_counts.items())
    
    return [word for word, _ in sorted(filtered_words, key=lambda x: x[1], reverse=True)[:top_n]]


def generate_dynamic_axis_labels(embeddings_3d, session_metadata, pca):
    """Generate meaningful axis labels by analyzing extreme sessions for each PCA component."""
    axis_labels = []
    
    for axis in range(min(3, embeddings_3d.shape[1])):
        # Get sessions at the extremes of this axis
        axis_values = embeddings_3d[:, axis]
        
        # Find indices of sessions with highest and lowest values
        n_extreme = max(1, len(session_metadata) // 10)  # Take top/bottom 10% or at least 1
        high_indices = np.argsort(axis_values)[-n_extreme:]
        low_indices = np.argsort(axis_values)[:n_extreme]
        
        # Extract texts from extreme sessions
        high_texts = [session_metadata[i]['text'] for i in high_indices]
        low_texts = [session_metadata[i]['text'] for i in low_indices]
        
        # Get dominant themes for each end
        high_themes = get_dominant_themes(high_texts, top_n=2)
        low_themes = get_dominant_themes(low_texts, top_n=2)
        
        # Create descriptive label
        if high_themes and low_themes:
            high_label = high_themes[0].title() if high_themes else "High"
            low_label = low_themes[0].title() if low_themes else "Low"
            
            # Make sure the labels are different and meaningful
            if high_label.lower() == low_label.lower():
                if len(high_themes) > 1:
                    high_label = high_themes[1].title()
                elif len(low_themes) > 1:
                    low_label = low_themes[1].title()
                else:
                    high_label = f"Axis {axis+1}+"
                    low_label = f"Axis {axis+1}-"
            
            axis_labels.append(f"{low_label} ← → {high_label}")
        else:
            # Fallback label
            axis_labels.append(f"Semantic Dimension {axis+1}")
    
    return axis_labels


def collect_comprehensive_analysis_data():
    """
    Collect all analysis data from all tabs for comprehensive behavioral analysis.
    
    Returns:
        dict: Comprehensive analysis context for the behavioral chat system
    """
    analysis_data = {
        'total_sessions': len(st.session_state.memory) if 'memory' in st.session_state else 0,
        'drift_analysis': {},
        'pca_2d_analysis': {},
        'pca_3d_analysis': {},
        'heatmap_analysis': {},
        'semantic_trajectory': {},
        'ridgeline_analysis': {},
        'session_texts': [],
        'dates': []
    }
    
    if len(st.session_state.memory) == 0:
        return analysis_data
    
    # Collect session texts for context
    if 'meta' in st.session_state:
        analysis_data['session_texts'] = [meta.get('text', '') for meta in st.session_state.meta]
        # Attempt to collect dates if present
        try:
            analysis_data['dates'] = [meta.get('date') for meta in st.session_state.meta if 'date' in meta]
        except Exception:
            analysis_data['dates'] = []
    
    # Drift Analysis Data
    if len(st.session_state.memory) > 1:
        try:
            from memory.drift import drift_series
            drifts, counts = drift_series(st.session_state.memory)
            analysis_data['drift_analysis'] = {
                'drift_scores': [float(d) for d in drifts],
                'token_counts': [int(c) for c in counts],
                'avg_drift': float(np.mean(drifts)),
                'max_drift': float(np.max(drifts)),
                'drift_trend': 'increasing' if drifts[-1] > drifts[0] else 'decreasing'
            }
        except Exception:
            analysis_data['drift_analysis'] = {'error': 'Could not compute drift analysis'}
    
    # Enhanced PCA Analysis Data (2D and 3D) with improved pipeline
    try:
        memory_slice = st.session_state.memory
        meta_slice = st.session_state.meta
        
        # 2D PCA with enhanced pipeline (ragged -> flat handled in semantic_tensor_memory.visualization)
        results_2d = robust_pca_pipeline(memory_slice, meta_slice, n_components=2, method='auto')
        if results_2d:
            analysis_data['pca_2d_analysis'] = {
                'explained_variance': [float(v) for v in results_2d['explained_variance_ratio']],
                'cumulative_variance': float(results_2d['cumulative_variance'][-1]),
                'condition_number': float(results_2d['condition_number']),
                'n_samples': int(results_2d['n_samples']),
                'n_features': int(results_2d['n_features']),
                'quality_assessment': results_2d['quality_assessment'],
                'method_used': results_2d['method_used'],
                'quality_score': float(results_2d['quality_score'])
            }
        
        # 3D PCA with enhanced pipeline
        results_3d = robust_pca_pipeline(memory_slice, meta_slice, n_components=3, method='auto')
        if results_3d:
            analysis_data['pca_3d_analysis'] = {
                'explained_variance': [float(v) for v in results_3d['explained_variance_ratio']],
                'cumulative_variance': float(results_3d['cumulative_variance'][-1]),
                'condition_number': float(results_3d['condition_number']),
                'n_samples': int(results_3d['n_samples']),
                'n_features': int(results_3d['n_features']),
                'quality_assessment': results_3d['quality_assessment'],
                'method_used': results_3d['method_used'],
                'quality_score': float(results_3d['quality_score'])
            }
    except Exception:
        analysis_data['pca_2d_analysis'] = {'error': 'Could not compute 2D PCA analysis'}
        analysis_data['pca_3d_analysis'] = {'error': 'Could not compute 3D PCA analysis'}
    
    # Semantic Trajectory Data
    if len(st.session_state.memory) > 1:
        try:
            # Calculate session embeddings
            session_embeddings = []
            for tensor in st.session_state.memory:
                session_emb = tensor.mean(0).numpy()
                session_embeddings.append(session_emb)
            session_embeddings = np.array(session_embeddings)
            
            # Calculate semantic velocity and acceleration
            semantic_velocity = np.zeros(len(session_embeddings))
            for i in range(1, len(session_embeddings)):
                semantic_velocity[i] = np.linalg.norm(session_embeddings[i] - session_embeddings[i-1])
            
            semantic_acceleration = np.zeros(len(semantic_velocity))
            for i in range(1, len(semantic_velocity)):
                semantic_acceleration[i] = abs(semantic_velocity[i] - semantic_velocity[i-1])
            
            velocity_threshold = np.mean(semantic_velocity) + np.std(semantic_velocity)
            significant_shifts = semantic_velocity > velocity_threshold
            
            analysis_data['semantic_trajectory'] = {
                'avg_velocity': float(np.mean(semantic_velocity)),
                'max_velocity': float(np.max(semantic_velocity)),
                'significant_shifts': [int(i+1) for i, shift in enumerate(significant_shifts) if shift],
                'total_significant_shifts': int(np.sum(significant_shifts)),
                'velocity_trend': 'increasing' if semantic_velocity[-1] > semantic_velocity[1] else 'decreasing'
            }
        except Exception:
            analysis_data['semantic_trajectory'] = {'error': 'Could not compute semantic trajectory analysis'}
    
    # Heatmap Analysis
    if len(st.session_state.memory) > 1:
        try:
            means = torch.stack([t.mean(0) for t in st.session_state.memory])
            means_norm = torch.nn.functional.normalize(means, p=2, dim=1)
            sims = torch.mm(means_norm, means_norm.t())
            dist = 1 - sims.numpy()
            
            analysis_data['heatmap_analysis'] = {
                'avg_distance': float(np.mean(dist[np.triu_indices_from(dist, k=1)])),
                'max_distance': float(np.max(dist)),
                'min_distance': float(np.min(dist[np.triu_indices_from(dist, k=1)])),
                'session_similarity_pattern': 'high_variance' if np.std(dist) > 0.2 else 'stable'
            }
        except Exception:
            analysis_data['heatmap_analysis'] = {'error': 'Could not compute heatmap analysis'}
    
    # Ridgeline Analysis
    analysis_data['ridgeline_analysis'] = {
        'description': 'Evolution of semantic features (PCA and t-SNE) across sessions',
        'feature_types': ['PCA-based linear patterns', 't-SNE-based non-linear patterns'],
        'dynamic_labeling': 'Content-based semantic axis interpretation applied'
    }
    
    return analysis_data 