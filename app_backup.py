import streamlit as st
import time, torch
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from memory.embedder import embed_sentence
from memory.store import load, append, save
from memory.drift import drift_series, token_drift
import os
import requests
from transformers import AutoTokenizer
from plotly.subplots import make_subplots
import altair as alt

# Fix PyTorch/Streamlit compatibility issues
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Set environment variables to prevent PyTorch/Streamlit conflicts
os.environ['TORCH_USE_CUDA_DSA'] = '0'
os.environ['TORCH_DISABLE_WARN'] = '1'

# Prevent PyTorch from interfering with Streamlit's file watcher
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    # Disable PyTorch's dynamic class registration that conflicts with Streamlit
    if hasattr(torch, '_C') and hasattr(torch._C, '_disable_jit_profiling'):
        torch._C._disable_jit_profiling()
except Exception:
    pass

# Set page config
st.set_page_config(
    page_title="Semantic Tensor Memory",
    page_icon="semantic_tensor_art_logo.png",
    layout="wide"
)

# Initialize session state
if 'memory' not in st.session_state:
    st.session_state.memory, st.session_state.meta = load()

# Initialize chat STM in session state
if 'chat_memory' not in st.session_state:
    st.session_state.chat_memory = []
    st.session_state.chat_meta = []

def add_chat_message(role, text):
    emb = embed_sentence(text)
    meta_row = {
        "ts": time.time(),
        "role": role,
        "text": text,
        "tokens": emb.shape[0]
    }
    st.session_state.chat_memory.append(emb)
    st.session_state.chat_meta.append(meta_row)

def plot_drift_plotly(drifts, token_counts):
    """Create interactive drift plot using Plotly with secondary y-axis."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add drift scores (line, left y-axis)
    fig.add_trace(
        go.Scatter(
            x=list(range(len(drifts))),
            y=drifts,
            name="Drift Score",
            line=dict(color='blue')
        ),
        secondary_y=False,
    )
    # Add token counts (bar, right y-axis)
    fig.add_trace(
        go.Bar(
            x=list(range(len(token_counts))),
            y=token_counts,
            name="Token Count",
            opacity=0.3,
            marker_color='gray'
        ),
        secondary_y=True,
    )
    # Update layout
    fig.update_layout(
        title="Session Drift Analysis",
        xaxis_title="Session",
        hovermode="x unified",
        showlegend=True
    )
    fig.update_yaxes(title_text="Drift Score", secondary_y=False)
    fig.update_yaxes(title_text="Token Count", secondary_y=True)
    return fig

def plot_pca_plotly(reduced, session_ids, meta):
    """Create interactive PCA plot using Plotly with a cool-to-warm color scale."""
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'PCA1': reduced[:, 0],
        'PCA2': reduced[:, 1],
        'Session': [f"Session {i+1}" for i in session_ids],
        'Text': [meta[i]['text'] for i in session_ids]
    })
    
    # Assign a numeric session index for color mapping
    df['SessionIdx'] = session_ids
    
    # Create scatter plot with cool-to-warm color scale
    fig = px.scatter(
        df,
        x='PCA1',
        y='PCA2',
        color='SessionIdx',
        color_continuous_scale='RdYlBu',  # Cool-to-warm color scale
        range_color=[df['SessionIdx'].min(), df['SessionIdx'].max()],
        hover_data=['Session', 'Text'],
        title="Semantic Drift Map"
    )
    
    # Reverse the color scale so early sessions are blue, later are red
    fig.update_traces(marker=dict(reversescale=True))
    fig.update_layout(
        hovermode="closest",
        showlegend=False,
        coloraxis_colorbar=dict(title="Session")
    )
    
    return fig

def plot_heatmap_plotly(tensors):
    """Create interactive heatmap using Plotly with a cool-to-warm color scale."""
    # Compute session means
    means = torch.stack([t.mean(0) for t in tensors])
    
    # Compute cosine similarity matrix
    means_norm = torch.nn.functional.normalize(means, p=2, dim=1)
    sims = torch.mm(means_norm, means_norm.t())
    dist = 1 - sims.numpy()
    
    # Get token counts for annotation
    token_counts = [t.shape[0] for t in tensors]
    
    # Create heatmap with a cool-to-warm color scale
    fig = go.Figure(data=go.Heatmap(
        z=dist,
        colorscale='RdYlBu',  # Cool-to-warm color scale
        reversescale=True,    # So low values are blue, high values are red
        text=[[f"{count}" for count in token_counts] for _ in range(len(token_counts))],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(title="Cosine distance")
    ))
    
    # Update layout
    fig.update_layout(
        title="Session-to-Session Semantic Drift",
        xaxis_title="Session",
        yaxis_title="Session"
    )
    
    return fig

def plot_ridgeline_plotly(memory, meta):
    """Create ridgeline plot showing evolution of semantic features across sessions."""
    from viz.pca_plot import prepare_for_pca
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    try:
        # Prepare data for dimensionality reduction
        flat, session_ids, token_ids = prepare_for_pca(memory)
        
        # Robust data validation
        if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
            return None
        flat = (flat - np.mean(flat, axis=0)) / np.std(flat, axis=0)
        if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
            return None
        
        # Calculate PCA components
        pca = PCA(n_components=2)
        pca_reduced = pca.fit_transform(flat)
        
        # Calculate t-SNE components (with smaller dataset for speed)
        n_samples = min(1000, flat.shape[0])  # Limit samples for t-SNE speed
        if flat.shape[0] > n_samples:
            indices = np.random.choice(flat.shape[0], n_samples, replace=False)
            tsne_flat = flat[indices]
            tsne_session_ids = session_ids[indices]
        else:
            tsne_flat = flat
            tsne_session_ids = session_ids
            
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_flat)//4))
        tsne_reduced = tsne.fit_transform(tsne_flat)
        
        # Generate dynamic labels for PCA components
        # Create session embeddings for labeling (mean of all tokens per session)
        session_embeddings = []
        session_metadata = []
        
        for idx, tensor in enumerate(memory):
            session_emb = tensor.mean(0).numpy()
            session_embeddings.append(session_emb)
            session_text = meta[idx].get('text', f'Session {idx+1}')
            session_metadata.append({
                'session_idx': idx,
                'text': session_text,
                'session_name': f'Session {idx+1}',
                'token_count': tensor.shape[0]
            })
        
        session_embeddings = np.array(session_embeddings)
        
        # Apply PCA to session embeddings for labeling
        pca_for_labels = PCA(n_components=2)
        session_pca = pca_for_labels.fit_transform(session_embeddings)
        
        # Generate meaningful labels
        pca_labels = generate_dynamic_axis_labels(
            np.column_stack([session_pca, np.zeros(len(session_pca))]),  # Add dummy 3rd dimension
            session_metadata, 
            pca_for_labels
        )
        
        # For t-SNE, use simpler descriptive labels since it's non-linear
        tsne_labels = ["Non-linear Pattern A", "Non-linear Pattern B"]
        
        # Calculate session-wise statistics for each feature
        ridgeline_data = []
        unique_sessions = sorted(np.unique(session_ids))
        
        features = [
            (pca_labels[0], pca_reduced[:, 0], session_ids),
            (pca_labels[1], pca_reduced[:, 1], session_ids),
            (tsne_labels[0], tsne_reduced[:, 0], tsne_session_ids),
            (tsne_labels[1], tsne_reduced[:, 1], tsne_session_ids)
        ]
        
        for feature_idx, (feature_name, feature_values, feature_session_ids) in enumerate(features):
            for session_idx in unique_sessions:
                # Get values for this session and feature
                session_mask = feature_session_ids == session_idx
                if np.sum(session_mask) > 0:
                    session_values = feature_values[session_mask]
                    
                    # Calculate statistics
                    mean_val = np.mean(session_values)
                    std_val = np.std(session_values)
                    
                    # Create multiple points around the mean to simulate distribution
                    # This creates the "curve" effect across sessions
                    for offset in np.linspace(-2*std_val, 2*std_val, 20):
                        ridgeline_data.append({
                            'Session': session_idx + 1,
                            'Feature': feature_name,
                            'Value': mean_val + offset,
                            'Density': np.exp(-0.5 * (offset/std_val)**2) if std_val > 0 else 1.0,
                            'FeatureOrder': feature_idx,
                            'FeatureType': 'PCA-based' if feature_idx < 2 else 't-SNE-based'
                        })
        
        if len(ridgeline_data) == 0:
            return None
            
        df_ridge = pd.DataFrame(ridgeline_data)
        
        # Create Altair ridgeline plot
        base = alt.Chart(df_ridge).add_params(
            alt.selection_interval(bind='scales', encodings=['x'])
        )
        
        ridgeline = base.mark_area(
            orient='vertical',
            opacity=0.6,
            stroke='white',
            strokeWidth=1,
            interpolate='cardinal'
        ).encode(
            alt.X('Session:O', 
                  title='Session Number',
                  axis=alt.Axis(labelAngle=0)),
            alt.Y('Value:Q',
                  title='',
                  scale=alt.Scale(range=[0, 40]),
                  axis=alt.Axis(labels=False, ticks=False, grid=False)),
            alt.Row('Feature:N',
                    title='Semantic Features',
                    sort=alt.SortField('FeatureOrder', order='ascending'),
                    header=alt.Header(labelAngle=0, labelAlign='left', labelPadding=15, titleOrient='left')),
            alt.Color('FeatureType:N',
                      scale=alt.Scale(domain=['PCA-based', 't-SNE-based'], 
                                    range=['#1f77b4', '#ff7f0e']),
                      legend=alt.Legend(title="Feature Type")),
            alt.Opacity('Density:Q',
                        scale=alt.Scale(range=[0.1, 0.8]),
                        legend=None),
            tooltip=['Session:O', 'Feature:N', 'Value:Q', 'FeatureType:N']
        ).resolve_scale(
            y='independent'
        ).properties(
            width=500,
            height=80,
            title=alt.TitleParams(
                text="Semantic Feature Evolution Across Sessions (Content-Based Labels)",
                anchor='start',
                fontSize=16,
                fontWeight='bold'
            )
        )
        
        return ridgeline
        
    except Exception as e:
        st.error(f"Error creating ridgeline plot: {e}")
        return None

def is_ollama_model_available(model_name):
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code == 200:
            tags = resp.json().get("models", [])
            return any(model_name in m.get("name", "") for m in tags)
    except Exception:
        pass
    return False

def render_chat_analysis_panel(context=None, tab_id=None):
    selected_model = st.session_state["selected_model"]
    chat_key = f"chat_history_{tab_id}" if tab_id else "chat_history"
    
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    
    chat_history = st.session_state[chat_key]
    
    # Check for streaming state
    streaming_key = f'streaming_{tab_id}' if tab_id else 'streaming'
    is_streaming = st.session_state.get(streaming_key, False)
    
    # Display existing chat history first (before new streaming)
    for role, msg in chat_history:
        with st.chat_message(role):
            st.markdown(msg)
    
    # Show buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        explain_clicked = st.button("Explain", key=f"explain_btn_{tab_id}", disabled=is_streaming)
    
    with col2:
        if is_streaming:
            pause_clicked = st.button("Pause", key=f"pause_btn_{tab_id}")
            if pause_clicked:
                st.session_state[streaming_key] = False
                st.rerun()
    
    # Handle Explain button click
    if explain_clicked:
        st.session_state[streaming_key] = True

    if not is_ollama_model_available(selected_model):
        st.warning(
            f"The model '{selected_model}' is not available in your local Ollama. "
            f"To download it, run this command in your terminal:\n\n"
            f"```sh\nollama pull {selected_model}\n```"
        )
            st.session_state[streaming_key] = False
            return
        
        # Create prompt based on context
        if context:
            prompt_parts = ["Analyze this semantic tensor memory data:"]
            for key, value in context.items():
                prompt_parts.append(f"{key}: {value}")
            prompt_parts.append("\nPlease provide actionable recommendations for next steps or interventions.")
            prompt = "\n".join(prompt_parts)
        else:
            prompt = "Please analyze the current semantic tensor memory data and provide actionable recommendations."
        
        def stream_ollama_response(prompt_text):
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": selected_model,
                "prompt": prompt_text,
                "stream": True
            }
            try:
            with requests.post(url, json=payload, stream=True, timeout=180) as r:
                for line in r.iter_lines():
                        if not st.session_state.get(streaming_key, False):
                            break
                    if line:
                        try:
                            import json as _json
                            data = _json.loads(line.decode("utf-8"))
                            yield data.get("response", "")
                        except Exception:
                            continue
            except Exception as e:
                yield f"Error connecting to Ollama: {e}"
        
        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed = ""
            for token in stream_ollama_response(prompt):
                if not st.session_state.get(streaming_key, False):
                    break
                streamed += token
                placeholder.markdown(streamed)
            
            # Finalize the response
        st.session_state[chat_key].append(("assistant", streamed))
        add_chat_message("assistant", streamed)
            st.session_state[streaming_key] = False
    
    # User input
    user_input = st.chat_input("Ask a follow-up about the analysis or data...", key=f"chat_input_{chat_key}")
    if user_input and not is_streaming:
        st.session_state[chat_key].append(("user", user_input))
        add_chat_message("user", user_input)
        st.rerun()

def generate_dynamic_axis_labels(embeddings_3d, session_metadata, pca):
    """Generate meaningful axis labels by analyzing extreme sessions for each PCA component."""
    import re
    from collections import Counter
    
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
    
    axis_labels = []
    
    for axis in range(3):
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

def create_altair_semantic_trajectory(embeddings_3d, session_metadata, semantic_velocity, semantic_acceleration, significant_shifts, axis_labels):
    """Create coordinated multi-view semantic trajectory visualization using Altair."""
    import altair as alt
    
    try:
        # Prepare data for Altair - ensure all values are JSON serializable
        trajectory_data = []
        for i, meta in enumerate(session_metadata):
            shift_type = "High" if significant_shifts[i] else ("Medium" if semantic_velocity[i] > np.mean(semantic_velocity) else "Low")
            trajectory_data.append({
                'Session': int(i + 1),
                'SessionName': str(meta['session_name']),
                'Dim1': float(embeddings_3d[i, 0]),
                'Dim2': float(embeddings_3d[i, 1]), 
                'Dim3': float(embeddings_3d[i, 2]),
                'Velocity': float(semantic_velocity[i]),
                'Acceleration': float(semantic_acceleration[i]),
                'ShiftType': str(shift_type),
                'Tokens': int(meta['token_count']),
                'Text': str(meta['text'][:100] + ('...' if len(meta['text']) > 100 else ''))
            })
        
        df = pd.DataFrame(trajectory_data)
        
        # Check if data is valid
        if df.empty or len(df) < 2:
            return None
            
        # Simple color scale
        color_scale = alt.Scale(domain=['Low', 'Medium', 'High'], 
                               range=['blue', 'orange', 'red'])
        
        # Create individual charts - each with its own data copy to avoid reference issues
        
        # 1. Time series for dimension 1
        dim1_chart = alt.Chart(df.copy()).mark_line(point=True).encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Dim1:Q', title=axis_labels[0]),
            color=alt.Color('ShiftType:N', scale=color_scale),
            tooltip=['SessionName:N', 'Dim1:Q', 'Velocity:Q']
        ).properties(width=180, height=120, title=f"{axis_labels[0]}")
        
        # 2. Time series for dimension 2
        dim2_chart = alt.Chart(df.copy()).mark_line(point=True).encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Dim2:Q', title=axis_labels[1]),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Dim2:Q', 'Velocity:Q']
        ).properties(width=180, height=120, title=f"{axis_labels[1]}")
        
        # 3. Time series for dimension 3
        dim3_chart = alt.Chart(df.copy()).mark_line(point=True).encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Dim3:Q', title=axis_labels[2]),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Dim3:Q', 'Velocity:Q']
        ).properties(width=180, height=120, title=f"{axis_labels[2]}")
        
        # 4. Scatter plot XY
        scatter_xy = alt.Chart(df.copy()).mark_circle(size=80).encode(
            x=alt.X('Dim1:Q', title=axis_labels[0]),
            y=alt.Y('Dim2:Q', title=axis_labels[1]),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Dim1:Q', 'Dim2:Q', 'Velocity:Q']
        ).properties(width=180, height=180, title="Semantic Space (XY)")
        
        # 5. Velocity chart
        velocity_chart = alt.Chart(df.copy()).mark_bar().encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Velocity:Q', title='Semantic Velocity'),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Velocity:Q']
        ).properties(width=180, height=120, title="Semantic Velocity")
        
        # 6. Acceleration chart
        acceleration_chart = alt.Chart(df.copy()).mark_bar().encode(
            x=alt.X('Session:O', title='Session'),
            y=alt.Y('Acceleration:Q', title='Semantic Acceleration'),
            color=alt.Color('ShiftType:N', scale=color_scale, legend=None),
            tooltip=['SessionName:N', 'Acceleration:Q']
        ).properties(width=180, height=120, title="Semantic Acceleration")
        
        # Create simple layout without complex concatenation
        charts = [
            dim1_chart,
            dim2_chart, 
            dim3_chart,
            scatter_xy,
            velocity_chart,
            acceleration_chart
        ]
        
        return charts  # Return list of charts instead of complex concatenation
        
    except Exception as e:
        st.error(f"Error creating Altair dashboard: {e}")
        return None

def robust_pca_pipeline(memory_slice, meta_slice, n_components=2, return_scaler=False):
    """
    Robust PCA pipeline following statistical best practices.
    
    Args:
        memory_slice: List of tensors for sessions
        meta_slice: Metadata for sessions
        n_components: Number of PCA components (2 or 3)
        return_scaler: Whether to return the fitted scaler for inverse transforms
    
    Returns:
        dict containing PCA results or None if processing failed
    """
    from viz.pca_plot import prepare_for_pca
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import warnings
    
    try:
        # Step 1: Prepare data with robust preprocessing
        flat, session_ids, token_ids = prepare_for_pca(memory_slice)
        
        # Step 2: Additional statistical validation
        if flat.shape[0] < n_components:
            st.error(f"Insufficient data points ({flat.shape[0]}) for {n_components}-component PCA. Need at least {n_components} points.")
            return None
            
        if flat.shape[1] < n_components:
            st.error(f"Insufficient dimensions ({flat.shape[1]}) for {n_components}-component PCA. Need at least {n_components} dimensions.")
            return None
        
        # Step 3: Check data quality after preprocessing
        if np.isnan(flat).any() or np.isinf(flat).any():
            st.error("Data contains NaN or Inf values after preprocessing. Cannot perform PCA.")
            return None
            
        # Check for extreme values that could indicate numerical issues
        if np.abs(flat).max() > 1e6:
            st.warning("Data contains very large values. This may indicate numerical instability.")
            
        # Step 4: Apply standardization (already done in prepare_for_pca, but verify)
        data_mean = np.mean(flat, axis=0)
        data_std = np.std(flat, axis=0)
        
        # Check if data is already standardized (mean ≈ 0, std ≈ 1)
        if not (np.abs(data_mean).max() < 1e-10 and np.abs(data_std - 1).max() < 1e-10):
            # Re-standardize if needed
            scaler = StandardScaler()
            flat = scaler.fit_transform(flat)
        else:
            scaler = None
        
        # Step 5: Check for multicollinearity (optional warning)
        condition_number = np.linalg.cond(np.cov(flat.T))
        if condition_number > 1e12:
            st.warning(f"High condition number ({condition_number:.2e}) indicates potential multicollinearity.")
        
        # Step 6: Perform PCA with proper random state for reproducibility
        pca = PCA(n_components=n_components, random_state=42)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            reduced = pca.fit_transform(flat)
        
        # Step 7: Validate PCA results
        if np.isnan(reduced).any() or np.isinf(reduced).any():
            st.error("PCA transformation produced NaN or Inf values.")
            return None
        
        # Step 8: Check explained variance
        explained_var_ratio = pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var_ratio)
        
        # Warning if explained variance is very low
        if cumulative_var[-1] < 0.5:
            st.warning(f"PCA explains only {cumulative_var[-1]:.1%} of variance. Consider using more components or different dimensionality reduction.")
        
        # Step 9: Prepare results
        results = {
            'reduced': reduced,
            'pca': pca,
            'session_ids': session_ids,
            'token_ids': token_ids,
            'explained_variance_ratio': explained_var_ratio,
            'cumulative_variance': cumulative_var,
            'n_samples': flat.shape[0],
            'n_features': flat.shape[1],
            'condition_number': condition_number
        }
        
        if return_scaler and scaler is not None:
            results['scaler'] = scaler
            
        return results
        
    except Exception as e:
        st.error(f"PCA pipeline failed: {e}")
        return None

def create_pca_visualization(results, meta_slice, is_3d=False):
    """
    Create PCA visualization from results with proper statistical annotations.
    
    Args:
        results: Output from robust_pca_pipeline
        meta_slice: Session metadata
        is_3d: Whether to create 3D plot
    
    Returns:
        Plotly figure object
    """
    if results is None:
        return None
        
    reduced = results['reduced']
    session_ids = results['session_ids']
    pca = results['pca']
    
    # Create DataFrame for plotting
    if is_3d:
        df = pd.DataFrame({
            'PCA1': reduced[:, 0],
            'PCA2': reduced[:, 1],
            'PCA3': reduced[:, 2],
            'Session': [f"Session {j+1}" for j in session_ids],
            'SessionIdx': session_ids,
            'Text': [meta_slice[j]['text'] for j in session_ids]
        })
    else:
        df = pd.DataFrame({
            'PCA1': reduced[:, 0],
            'PCA2': reduced[:, 1],
            'Session': [f"Session {j+1}" for j in session_ids],
            'SessionIdx': session_ids,
            'Text': [meta_slice[j]['text'] for j in session_ids]
        })
    
    # For axis labeling, we need session-level data, not token-level
    # Calculate session means in the PCA space for proper axis labeling
    unique_sessions = sorted(set(session_ids))
    session_means_pca = []
    session_metadata_for_labels = []
    
    for session_idx in unique_sessions:
        session_mask = np.array(session_ids) == session_idx
        session_pca_mean = reduced[session_mask].mean(axis=0)
        session_means_pca.append(session_pca_mean)
        session_metadata_for_labels.append({
            'session_idx': session_idx,
            'text': meta_slice[session_idx]['text'],
            'session_name': f'Session {session_idx+1}',
            'token_count': meta_slice[session_idx].get('tokens', len(meta_slice[session_idx]['text']))
        })
    
    session_means_pca = np.array(session_means_pca)
    
    # Create visualization
    if is_3d:
        fig = px.scatter_3d(
            df,
            x='PCA1',
            y='PCA2',
            z='PCA3',
            color='SessionIdx',
            color_continuous_scale='RdYlBu',
            range_color=[df['SessionIdx'].min(), df['SessionIdx'].max()],
            hover_data=['Session', 'Text'],
            title=f"3D Semantic Drift Map (Explains {results['cumulative_variance'][-1]:.1%} of variance)"
        )
        fig.update_traces(marker=dict(size=4, reversescale=True))
        
        # Generate dynamic axis labels using session-level data
        try:
            axis_labels = generate_dynamic_axis_labels(
                session_means_pca, 
                session_metadata_for_labels, 
                pca
            )
        except Exception as e:
            # Fallback to generic labels if dynamic labeling fails
            st.warning(f"Could not generate dynamic axis labels: {e}")
            axis_labels = [
                f"Semantic Dimension 1 (Explains {results['explained_variance_ratio'][0]:.1%} of variance)",
                f"Semantic Dimension 2 (Explains {results['explained_variance_ratio'][1]:.1%} of variance)", 
                f"Semantic Dimension 3 (Explains {results['explained_variance_ratio'][2]:.1%} of variance)"
            ]
        
        fig.update_layout(
            scene = dict(
                xaxis_title=f'<b>{axis_labels[0]}</b><br><span style="font-size:12px; color:gray">Explains {results["explained_variance_ratio"][0]:.1%} of variance</span>',
                yaxis_title=f'<b>{axis_labels[1]}</b><br><span style="font-size:12px; color:gray">Explains {results["explained_variance_ratio"][1]:.1%} of variance</span>',
                zaxis_title=f'<b>{axis_labels[2]}</b><br><span style="font-size:12px; color:gray">Explains {results["explained_variance_ratio"][2]:.1%} of variance</span>',
            ),
            coloraxis_colorbar=dict(title="Session"),
            margin=dict(l=0, r=0, b=0, t=40)
        )
    else:
        fig = px.scatter(
            df,
            x='PCA1',
            y='PCA2',
            color='SessionIdx',
            color_continuous_scale='RdYlBu',
            range_color=[df['SessionIdx'].min(), df['SessionIdx'].max()],
            hover_data=['Session', 'Text'],
            title=f"Semantic Drift Map (Explains {results['cumulative_variance'][-1]:.1%} of variance)"
        )
        fig.update_traces(marker=dict(reversescale=True))
        
        # Generate dynamic axis labels using session-level data
        try:
            axis_labels = generate_dynamic_axis_labels(
                session_means_pca, 
                session_metadata_for_labels, 
                pca
            )
        except Exception as e:
            # Fallback to generic labels if dynamic labeling fails
            st.warning(f"Could not generate dynamic axis labels: {e}")
            axis_labels = [
                f"Semantic Dimension 1 (Explains {results['explained_variance_ratio'][0]:.1%} of variance)",
                f"Semantic Dimension 2 (Explains {results['explained_variance_ratio'][1]:.1%} of variance)"
            ]
        
        fig.update_layout(
            hovermode="closest",
            showlegend=False,
            coloraxis_colorbar=dict(title="Session"),
            xaxis_title=f'{axis_labels[0]}',
            yaxis_title=f'{axis_labels[1]}'
        )
    
    # Add feature labels at extremes (simplified for now)
    extremes = []
    cols = ['PCA1', 'PCA2', 'PCA3'] if is_3d else ['PCA1', 'PCA2']
    for axis, col in enumerate(cols):
        max_idx = df[col].idxmax()
        min_idx = df[col].idxmin()
        for idx, label in zip([max_idx, min_idx], [f"{col} +", f"{col} -"]):
            text_excerpt = ' '.join(df.loc[idx, 'Text'].split()[:6]) + ('...' if len(df.loc[idx, 'Text'].split()) > 6 else '')
            extreme_data = {
                'PCA1': float(df.loc[idx, 'PCA1']),
                'PCA2': float(df.loc[idx, 'PCA2']),
                'label': f"{label}: {text_excerpt}"
            }
            if is_3d:
                extreme_data['PCA3'] = float(df.loc[idx, 'PCA3'])
            extremes.append(extreme_data)
    
    # Add extreme point markers
    for ex in extremes:
        if is_3d:
            fig.add_trace(go.Scatter3d(
                x=[ex['PCA1']], y=[ex['PCA2']], z=[ex['PCA3']],
                mode='markers+text',
                marker=dict(size=10, color='black', symbol='diamond'),
                text=[ex['label']],
                textposition='top center',
                showlegend=False
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[ex['PCA1']], y=[ex['PCA2']],
                mode='markers+text',
                marker=dict(size=10, color='black', symbol='diamond'),
                text=[ex['label']],
                textposition='top center',
                showlegend=False
            ))
    
    return fig

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
        'session_texts': []
    }
    
    if len(st.session_state.memory) == 0:
        return analysis_data
    
    # Collect session texts for context
    if 'meta' in st.session_state:
        analysis_data['session_texts'] = [meta.get('text', '') for meta in st.session_state.meta]
    
    # Drift Analysis Data
    if len(st.session_state.memory) > 1:
        try:
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
    
    # PCA Analysis Data (2D and 3D)
    try:
        memory_slice = st.session_state.memory
        meta_slice = st.session_state.meta
        
        # 2D PCA
        results_2d = robust_pca_pipeline(memory_slice, meta_slice, n_components=2)
        if results_2d:
            analysis_data['pca_2d_analysis'] = {
                'explained_variance': [float(v) for v in results_2d['explained_variance_ratio']],
                'cumulative_variance': float(results_2d['cumulative_variance'][-1]),
                'condition_number': float(results_2d['condition_number']),
                'n_samples': int(results_2d['n_samples']),
                'n_features': int(results_2d['n_features']),
                'quality_assessment': 'excellent' if results_2d['cumulative_variance'][-1] > 0.7 else 'good' if results_2d['cumulative_variance'][-1] > 0.5 else 'concerning'
            }
        
        # 3D PCA
        results_3d = robust_pca_pipeline(memory_slice, meta_slice, n_components=3)
        if results_3d:
            analysis_data['pca_3d_analysis'] = {
                'explained_variance': [float(v) for v in results_3d['explained_variance_ratio']],
                'cumulative_variance': float(results_3d['cumulative_variance'][-1]),
                'condition_number': float(results_3d['condition_number']),
                'n_samples': int(results_3d['n_samples']),
                'n_features': int(results_3d['n_features'])
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

def render_comprehensive_chat_analysis():
    """
    Render the comprehensive chat analysis panel with all collected data.
    """
    st.header("🧠 Comprehensive Behavioral Analysis & Chat")
    
    # Collect all analysis data
    analysis_data = collect_comprehensive_analysis_data()
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", analysis_data['total_sessions'])
    with col2:
        if 'avg_drift' in analysis_data['drift_analysis']:
            st.metric("Avg Semantic Drift", f"{analysis_data['drift_analysis']['avg_drift']:.3f}")
        else:
            st.metric("Avg Semantic Drift", "N/A")
    with col3:
        if 'cumulative_variance' in analysis_data['pca_2d_analysis']:
            st.metric("PCA Variance Explained", f"{analysis_data['pca_2d_analysis']['cumulative_variance']:.1%}")
        else:
            st.metric("PCA Variance Explained", "N/A")
    with col4:
        if 'total_significant_shifts' in analysis_data['semantic_trajectory']:
            st.metric("Significant Shifts", analysis_data['semantic_trajectory']['total_significant_shifts'])
        else:
            st.metric("Significant Shifts", "N/A")
    
    # Show detailed analysis in expander
    with st.expander("📊 Detailed Analysis Data", expanded=False):
        st.json(analysis_data)
    
    # Render chat panel with comprehensive context
    render_chat_analysis_panel(context=analysis_data, tab_id="comprehensive")

def main():
    # Prevent rerun loop after CSV import
    if st.session_state.get("csv_imported", False):
        st.session_state["csv_imported"] = False
        st.stop()
    # Adjust the layout to move it down by about 1/4 inch
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem !important;
            margin-top: 0rem !important;
        }
        html, body {
            padding-top: 0 !important;
            margin-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.sidebar.image("semantic_tensor_art_logo.png", width=285)
    st.title("Semantic Tensor Memory Analysis")

    # Model selection dropdown (shared globally, only once)
    model_options = {
        "Qwen3 (Ollama)": "qwen3:latest",
        "Mistral (Ollama)": "mistral:latest"
    }
    selected_model_label = st.selectbox(
        "Choose LLM model for analysis:",
        list(model_options.keys()),
        key="behavior_model_global"
    )
    selected_model = model_options[selected_model_label]
    st.session_state["selected_model_label"] = selected_model_label
    st.session_state["selected_model"] = selected_model

    # Sidebar for input
    st.sidebar.header("Input")
    input_text = st.sidebar.text_area("Enter new session text:", height=150)
    
    if st.sidebar.button("Add Session"):
        if input_text.strip():
            emb = embed_sentence(input_text)
            meta_row = {
                "ts": time.time(),
                "text": input_text,
                "tokens": emb.shape[0]
            }
            st.session_state.memory, st.session_state.meta = append(
                st.session_state.memory, emb, st.session_state.meta, meta_row
            )
            st.sidebar.success(f"Stored session {len(st.session_state.meta)} with {emb.shape[0]} tokens.")
            st.rerun()
        else:
            st.sidebar.warning("Please enter some text.")
    
    # File uploader for CSV
    uploaded_file = st.sidebar.file_uploader("Import sessions from CSV", type=['csv'], key="uploaded_file")
    if uploaded_file is not None:
        import csv
        import io
        memory = []
        meta = []
        csv_data = uploaded_file.read().decode('utf-8')
        reader = csv.DictReader(io.StringIO(csv_data))
        for row in reader:
            text = row.get('text', '').strip()
            if not text:
                continue
            emb = embed_sentence(text)
            meta_row = dict(row)
            meta_row['tokens'] = emb.shape[0]
            memory.append(emb)
            meta.append(meta_row)
        if memory:
            st.session_state.memory = memory
            st.session_state.meta = meta
            save(memory, meta)
            st.sidebar.success(f"Imported and saved {len(memory)} sessions.")
            st.session_state["csv_imported"] = True
            st.rerun()
    
    # Main content area
    if len(st.session_state.memory) > 0:
        # Create tabs for different visualizations
        tab_names = ["Drift Analysis", "PCA Map", "3D PCA Map", "Heatmap", "Semantic Trajectory", "Ridgeline Plot", "Clinical Analysis & Chat"]
        tab_objects = st.tabs(tab_names)
        for i, tab in enumerate(tab_objects):
            with tab:
                if i == 0:
                    st.header("Drift Analysis")
                    if len(st.session_state.memory) > 1:
                        drifts, counts = drift_series(st.session_state.memory)
                        fig = plot_drift_plotly(drifts, counts)
                        st.plotly_chart(fig, use_container_width=True)
                        drift_data = []
                        for j, (d, c) in enumerate(zip(drifts, counts)):
                            drift_data.append({
                                "Session": f"{j+1} → {j+2}",
                                "Drift Score": f"{d:.3f}",
                                "Token Count": c
                            })
                        st.table(pd.DataFrame(drift_data))
                    else:
                        st.warning("Need ≥2 sessions to plot drift.")
                elif i == 1:
                    st.header("PCA Map")
                    explained_variance_str = ""
                    summary = ""
                    if len(st.session_state.memory) > 1:
                        max_session = len(st.session_state.memory)
                        if 'pca2d_played' not in st.session_state or not st.session_state.pca2d_played:
                            st.session_state.pca2d_played = True
                            for j in range(2, max_session + 1):
                                st.session_state['timeline2d'] = j
                                time.sleep(0.08)
                                st.rerun()
                        timeline_idx = st.slider(
                            "Timeline: Show up to session",
                            2, max_session,
                            st.session_state.get('timeline2d', max_session),
                            1,
                            key="timeline2d"
                        )
                        memory_slice = st.session_state.memory[:timeline_idx]
                        meta_slice = st.session_state.meta[:timeline_idx]
                        results = robust_pca_pipeline(memory_slice, meta_slice)
                        if results:
                            fig = create_pca_visualization(results, meta_slice)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display PCA diagnostics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Explained Variance", f"{results['cumulative_variance'][-1]:.1%}")
                            with col2:
                                st.metric("Data Points", f"{results['n_samples']:,}")
                            with col3:
                                st.metric("Features", f"{results['n_features']:,}")
                            
                            explained_variance_str = ', '.join([f'{v:.2%}' for v in results['explained_variance_ratio']])
                            st.write(f"**Component-wise explained variance:** {explained_variance_str}")
                            
                            # Show condition number if problematic
                            if results['condition_number'] > 1e6:
                                st.warning(f"⚠️ High condition number: {results['condition_number']:.2e} (may indicate multicollinearity)")
                            
                            # Add PCA quality assessment
                            with st.expander("📊 PCA Quality Assessment", expanded=False):
                                st.markdown("""
                                **Understanding PCA Quality:**
                                - **Explained Variance**: Higher is better (>70% excellent, >50% good, <30% concerning)
                                - **Condition Number**: Lower is better (<1e6 good, >1e12 problematic)
                                - **Data Points**: More data generally improves reliability
                                
                                **Best Practices Applied:**
                                ✅ Robust data preprocessing (NaN/Inf handling)  
                                ✅ Standardization (zero mean, unit variance)  
                                ✅ Multicollinearity detection  
                                ✅ Reproducible results (fixed random seed)  
                                ✅ Statistical validation of inputs and outputs  
                                """)
                                
                                st.write(f"**Condition Number:** {results['condition_number']:.2e}")
                                st.write(f"**Total Variance Captured:** {results['cumulative_variance'][-1]:.1%}")
                                st.write(f"**Samples per Feature Ratio:** {results['n_samples']/results['n_features']:.1f}")
                            
                            from viz.pca_summary import generate_narrative_summary
                            summary = generate_narrative_summary(results['reduced'], results['session_ids'], results['token_ids'], meta_slice)
                            st.subheader("Narrative Summary")
                            st.info(summary)
                            # Store PCA results in session state for use in clinical tab
                            st.session_state.pca_results = results
                            st.session_state.pca_results['meta'] = meta_slice
                            explained_variance_str = ', '.join([f'{v:.2%}' for v in results['explained_variance_ratio']])
                            summary = summary
                        else:
                            st.error("PCA analysis failed.")
                            explained_variance_str = "N/A"
                            summary = "PCA analysis failed"
                    else:
                        st.warning("Need ≥2 sessions for PCA analysis.")
                        explained_variance_str = "N/A"
                        summary = "Insufficient data for PCA analysis"
                elif i == 2:
                    st.header("3D PCA Map")
                    explained_variance_str = ""
                    summary = ""
                    if len(st.session_state.memory) > 1:
                        max_session = len(st.session_state.memory)
                        if 'pca3d_played' not in st.session_state or not st.session_state.pca3d_played:
                            st.session_state.pca3d_played = True
                            for j in range(2, max_session + 1):
                                st.session_state['timeline3d'] = j
                            time.sleep(0.08)
                            st.rerun()
                        timeline_idx = st.slider(
                            "Timeline: Show up to session",
                            2, max_session,
                            st.session_state.get('timeline3d', max_session),
                            1,
                            key="timeline3d"
                        )
                        memory_slice = st.session_state.memory[:timeline_idx]
                        meta_slice = st.session_state.meta[:timeline_idx]
                        results = robust_pca_pipeline(memory_slice, meta_slice, n_components=3)
                        if results:
                            fig = create_pca_visualization(results, meta_slice, is_3d=True)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display 3D PCA diagnostics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Explained Variance", f"{results['cumulative_variance'][-1]:.1%}")
                            with col2:
                                st.metric("Data Points", f"{results['n_samples']:,}")
                            with col3:
                                st.metric("Features", f"{results['n_features']:,}")
                            with col4:
                                st.metric("Components", "3")
                            
                            explained_variance_str = ', '.join([f'{v:.2%}' for v in results['explained_variance_ratio']])
                            st.write(f"**Component-wise explained variance:** {explained_variance_str}")
                            
                            # Show condition number if problematic
                            if results['condition_number'] > 1e6:
                                st.warning(f"⚠️ High condition number: {results['condition_number']:.2e} (may indicate multicollinearity)")
                            
                            # Add PCA quality assessment
                            with st.expander("📊 PCA Quality Assessment", expanded=False):
                                st.markdown("""
                                **Understanding PCA Quality:**
                                - **Explained Variance**: Higher is better (>70% excellent, >50% good, <30% concerning)
                                - **Condition Number**: Lower is better (<1e6 good, >1e12 problematic)
                                - **Data Points**: More data generally improves reliability
                                
                                **Best Practices Applied:**
                                ✅ Robust data preprocessing (NaN/Inf handling)  
                                ✅ Standardization (zero mean, unit variance)  
                                ✅ Multicollinearity detection  
                                ✅ Reproducible results (fixed random seed)  
                                ✅ Statistical validation of inputs and outputs  
                                """)
                                
                                st.write(f"**Condition Number:** {results['condition_number']:.2e}")
                                st.write(f"**Total Variance Captured:** {results['cumulative_variance'][-1]:.1%}")
                                st.write(f"**Samples per Feature Ratio:** {results['n_samples']/results['n_features']:.1f}")
                            
                            from viz.pca_summary import generate_narrative_summary
                            summary = generate_narrative_summary(results['reduced'], results['session_ids'], results['token_ids'], meta_slice)
                            st.subheader("Narrative Summary")
                            st.info(summary)
                            explained_variance_str = ', '.join([f'{v:.2%}' for v in results['explained_variance_ratio']])
                            summary = summary
                        else:
                            st.error("3D PCA analysis failed.")
                            explained_variance_str = "N/A"
                            summary = "3D PCA analysis failed"
                    else:
                        st.warning("Need ≥2 sessions for 3D PCA analysis.")
                        explained_variance_str = "N/A"
                        summary = "Insufficient data for 3D PCA analysis"
                elif i == 3:
                    st.header("Session Heatmap")
                    if len(st.session_state.memory) > 1:
                        fig = plot_heatmap_plotly(st.session_state.memory)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need ≥2 sessions for heatmap analysis.")
                elif i == 4:
                    st.header("Semantic Trajectory")
                    
                    # Add visualization type selector
                    viz_type = st.radio(
                        "Choose visualization type:",
                        ["3D Interactive (Plotly)", "Multi-View Dashboard (Altair)"],
                        horizontal=True,
                        help="3D view shows spatial relationships, Dashboard view shows coordinated analysis"
                    )
                    
                    if len(st.session_state.memory) > 1:
                        # Calculate session embeddings (mean of all tokens per session)
                        session_embeddings = []
                        session_metadata = []
                        
                        for idx, tensor in enumerate(st.session_state.memory):
                            session_emb = tensor.mean(0).numpy()  # Mean embedding for the session
                            session_embeddings.append(session_emb)
                            session_text = st.session_state.meta[idx].get('text', f'Session {idx+1}')
                            session_metadata.append({
                                'session_idx': idx,
                                'text': session_text,
                                'session_name': f'Session {idx+1}',
                                'token_count': tensor.shape[0]
                            })
                        
                        session_embeddings = np.array(session_embeddings)
                        
                    # Robust data validation
                        if np.isnan(session_embeddings).any() or np.isinf(session_embeddings).any():
                            st.error("Session embeddings contain NaN or Inf values. Cannot plot trajectory.")
                    else:
                            # PCA reduction to 3D using improved pipeline
                        from sklearn.decomposition import PCA
                            from sklearn.preprocessing import StandardScaler
                            
                            # Standardize session embeddings
                            scaler = StandardScaler()
                            session_embeddings_scaled = scaler.fit_transform(session_embeddings)
                            
                            # Apply PCA
                            pca = PCA(n_components=3, random_state=42)
                            embeddings_3d = pca.fit_transform(session_embeddings_scaled)
                            
                            # Check PCA quality
                            explained_var_ratio = pca.explained_variance_ratio_
                            cumulative_var = np.sum(explained_var_ratio)
                            
                            if cumulative_var < 0.5:
                                st.info(f"ℹ️ PCA explains {cumulative_var:.1%} of variance in session-level data. This is normal for high-dimensional semantic data.")
                            
                            # Generate dynamic axis labels
                            axis_labels = generate_dynamic_axis_labels(embeddings_3d, session_metadata, pca)
                            
                            # Calculate semantic velocity (rate of change between sessions)
                            semantic_velocity = np.zeros(len(embeddings_3d))
                            for i in range(1, len(embeddings_3d)):
                                semantic_velocity[i] = np.linalg.norm(embeddings_3d[i] - embeddings_3d[i-1])
                            
                            # Calculate semantic acceleration (change in velocity)
                            semantic_acceleration = np.zeros(len(semantic_velocity))
                            for i in range(1, len(semantic_velocity)):
                                semantic_acceleration[i] = abs(semantic_velocity[i] - semantic_velocity[i-1])
                            
                            # Identify significant semantic shifts (high velocity)
                            velocity_threshold = np.mean(semantic_velocity) + np.std(semantic_velocity)
                            significant_shifts = semantic_velocity > velocity_threshold
                            
                            if viz_type == "3D Interactive (Plotly)":
                                # Original 3D Plotly visualization
                                # Create colors based on semantic velocity
                                colors = []
                                for i, vel in enumerate(semantic_velocity):
                                    if significant_shifts[i]:
                                        colors.append('red')  # High semantic change
                                    elif vel > np.mean(semantic_velocity):
                                        colors.append('orange')  # Medium semantic change
                                    else:
                                        colors.append('blue')  # Low semantic change
                                
                                # Create hover text with semantic insights
                                hover_text = []
                                for i, meta in enumerate(session_metadata):
                                    text_preview = meta['text'][:100] + ('...' if len(meta['text']) > 100 else '')
                                    hover_info = [
                                        f"<b>{meta['session_name']}</b>",
                                        f"Semantic Velocity: {semantic_velocity[i]:.3f}",
                                        f"Semantic Acceleration: {semantic_acceleration[i]:.3f}",
                                        f"Token Count: {meta['token_count']}",
                                        f"Text: {text_preview}"
                                    ]
                                    hover_text.append("<br>".join(hover_info))
                                
                                # Create 3D trajectory plot
                                fig = go.Figure()
                                
                                # Add trajectory line connecting sessions
                                fig.add_trace(go.Scatter3d(
                                    x=embeddings_3d[:, 0],
                                    y=embeddings_3d[:, 1],
                                    z=embeddings_3d[:, 2],
                                    mode='lines',
                                    line=dict(width=3, color='gray'),
                                    name='Semantic Path',
                                    showlegend=False,
                                    hoverinfo='skip'
                                ))
                                
                                # Add session points
                                fig.add_trace(go.Scatter3d(
                                    x=embeddings_3d[:, 0],
                                    y=embeddings_3d[:, 1],
                                    z=embeddings_3d[:, 2],
                                    mode='markers+text',
                                    marker=dict(
                                        size=8,
                                        color=colors,
                                        opacity=0.8,
                                        line=dict(width=2, color='white')
                                    ),
                                    text=[f"S{i+1}" for i in range(len(embeddings_3d))],
                                    textposition='top center',
                                    hovertext=hover_text,
                                    hoverinfo='text',
                                    name='Sessions'
                                ))
                                
                                # Highlight significant semantic shifts with arrows
                                for i in range(1, len(embeddings_3d)):
                                    if significant_shifts[i]:
                                        # Add arrow showing direction of significant change
                                        start = embeddings_3d[i-1]
                                        end = embeddings_3d[i]
                                        fig.add_trace(go.Scatter3d(
                                            x=[start[0], end[0]],
                                            y=[start[1], end[1]],
                                            z=[start[2], end[2]],
                                            mode='lines',
                                            line=dict(width=6, color='red'),
                                            name=f'Semantic Shift {i}→{i+1}',
                                            showlegend=False,
                                            hoverinfo='skip'
                                        ))
                                
                                # Update layout with dynamic labels
                        fig.update_layout(
                                    title='Semantic Trajectory: Evolution of Meaning Across Sessions',
                                    scene=dict(
                                        xaxis_title=f'<b>{axis_labels[0]}</b><br><span style="font-size:12px; color:gray">Primary semantic axis</span>',
                                        yaxis_title=f'<b>{axis_labels[1]}</b><br><span style="font-size:12px; color:gray">Secondary semantic axis</span>',
                                        zaxis_title=f'<b>{axis_labels[2]}</b><br><span style="font-size:12px; color:gray">Tertiary semantic axis</span>',
                                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                                    ),
                                    margin=dict(l=0, r=0, b=0, t=40),
                                    height=600
                                )
                                
                        st.plotly_chart(fig, use_container_width=True)
                                
                            else:  # Altair Dashboard
                                # Create Altair multi-view dashboard
                                st.write("Creating Altair dashboard...")  # Debug info
                                charts = create_altair_semantic_trajectory(
                                    embeddings_3d, session_metadata, semantic_velocity, 
                                    semantic_acceleration, significant_shifts, axis_labels
                                )
                                
                                if charts:
                                    st.write("Dashboard created successfully, displaying...")  # Debug info
                                    try:
                                        # Display charts in an organized layout
                                        st.subheader("Semantic Dimensions Over Time")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.altair_chart(charts[0], use_container_width=True)  # Dim1
                                        with col2:
                                            st.altair_chart(charts[1], use_container_width=True)  # Dim2
                                        with col3:
                                            st.altair_chart(charts[2], use_container_width=True)  # Dim3
                                        
                                        st.subheader("Semantic Space & Dynamics")
                                        col1, col2, col3 = st.columns(3)
                                        with col1:
                                            st.altair_chart(charts[3], use_container_width=True)  # Scatter XY
                                        with col2:
                                            st.altair_chart(charts[4], use_container_width=True)  # Velocity
                                        with col3:
                                            st.altair_chart(charts[5], use_container_width=True)  # Acceleration
                                        
                                        st.success("Dashboard displayed successfully!")  # Debug info
                                    except Exception as e:
                                        st.error(f"Error displaying Altair chart: {e}")
                                        st.write("Falling back to simple charts...")
                                        
                                        # Fallback: Create simple individual charts
                                        trajectory_data = []
                                        for i, meta in enumerate(session_metadata):
                                            shift_type = "High" if significant_shifts[i] else ("Medium" if semantic_velocity[i] > np.mean(semantic_velocity) else "Low")
                                            trajectory_data.append({
                                                'Session': i + 1,
                                                'Dim1': float(embeddings_3d[i, 0]),
                                                'Velocity': float(semantic_velocity[i]),
                                                'ShiftType': shift_type
                                            })
                                        
                                        df_simple = pd.DataFrame(trajectory_data)
                                        
                                        # Simple fallback chart
                                        simple_chart = alt.Chart(df_simple).mark_line(point=True).encode(
                                            x=alt.X('Session:O'),
                                            y=alt.Y('Dim1:Q'),
                                            color=alt.Color('ShiftType:N')
                                        ).properties(width=400, height=200, title="Semantic Dimension 1 Over Time")
                                        
                                        st.altair_chart(simple_chart, use_container_width=True)
            else:
                                    st.error("Failed to create Altair dashboard - charts list is empty")
                                
                                st.markdown("""
                                **Dashboard Interaction Guide:**
                                - **Top row**: Time series of each semantic dimension
                                - **Middle row**: 2D projections of semantic space
                                - **Bottom row**: Velocity and acceleration patterns over time
                                - **Color coding**: Blue (low change), Orange (medium), Red (high semantic shifts)
                                """)
                                
                            # Display axis interpretation (common to both visualizations)
                            st.subheader("Semantic Axis Interpretation")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.info(f"**X-Axis**: {axis_labels[0]}")
                            with col2:
                                st.info(f"**Y-Axis**: {axis_labels[1]}")
                            with col3:
                                st.info(f"**Z-Axis**: {axis_labels[2]}")
                            
                            # Display semantic analysis metrics (common to both)
                            st.subheader("Semantic Dynamics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Average Semantic Velocity", f"{np.mean(semantic_velocity):.3f}")
                            with col2:
                                st.metric("Max Semantic Shift", f"{np.max(semantic_velocity):.3f}")
                            with col3:
                                st.metric("Significant Shifts", f"{np.sum(significant_shifts)}")
                            
                            # Show session-by-session analysis (common to both)
                            st.subheader("Session Analysis")
                            trajectory_data = []
                            for i, meta in enumerate(session_metadata):
                                shift_type = "🔴 High" if significant_shifts[i] else ("🟠 Medium" if semantic_velocity[i] > np.mean(semantic_velocity) else "🔵 Low")
                                trajectory_data.append({
                                    "Session": meta['session_name'],
                                    "Semantic Velocity": f"{semantic_velocity[i]:.3f}",
                                    "Semantic Acceleration": f"{semantic_acceleration[i]:.3f}",
                                    "Change Level": shift_type,
                                    "Tokens": meta['token_count']
                                })
                            
                            st.dataframe(pd.DataFrame(trajectory_data), use_container_width=True)
                            
                            st.markdown("""
                            **Semantic Trajectory Interpretation:**
                            - **Gray line**: Path through semantic space over time
                            - **Blue dots**: Low semantic change between sessions
                            - **Orange dots**: Medium semantic change
                            - **Red dots**: High semantic change (significant shifts)
                            - **Red thick lines**: Major semantic transitions
                            - **Semantic Velocity**: How much meaning changes between sessions
                            - **Semantic Acceleration**: How the rate of change evolves
                            """)
                            
                            # Prepare context for LLM analysis
                            significant_sessions = [i+1 for i, shift in enumerate(significant_shifts) if shift]
                            trajectory_context = {
                                'avg_velocity': f"{np.mean(semantic_velocity):.3f}",
                                'max_shift': f"{np.max(semantic_velocity):.3f}",
                                'significant_shifts': significant_sessions,
                                'total_sessions': len(session_metadata),
                                'explained_variance': ', '.join([f'{v:.2%}' for v in explained_var_ratio]),
                                'axis_labels': axis_labels,
                                'visualization_type': viz_type
                            }
                    else:
                        st.warning("Need ≥2 sessions for semantic trajectory analysis.")
                        trajectory_context = {'message': 'Insufficient data for semantic trajectory analysis'}
                    
                    render_chat_analysis_panel(context=trajectory_context, tab_id="trajectory")
                elif i == 5:
                    st.header("Ridgeline Plot")
                    if len(st.session_state.memory) > 1:
                        fig = plot_ridgeline_plotly(st.session_state.memory, st.session_state.meta)
                        if fig is not None:
                            st.altair_chart(fig, use_container_width=True)
                            st.markdown("""
                            **Ridgeline Plot Interpretation:**
                            - **X-axis**: Session numbers (chronological progression)
                            - **Y-axis**: Different semantic features with content-based labels
                            - **Feature Labels**: Dynamically generated based on content analysis
                              - **PCA-based features** (blue): Linear semantic patterns with meaningful labels (e.g., "Clinical ← → Personal")
                              - **t-SNE-based features** (orange): Non-linear semantic patterns capturing complex relationships
                            - **Curves**: Show how each semantic dimension evolves across sessions
                            - **Curve height**: Indicates the variance/spread of values in that session
                            - **Curve shifts**: Show semantic drift over time in interpretable dimensions
                            - **Dynamic Labels**: Generated by analyzing content at semantic extremes to provide meaningful axis interpretations
                            """)
                        else:
                            st.error("Unable to generate ridgeline plot. Need sufficient data with valid embeddings.")
                    else:
                        st.warning("Need ≥2 sessions for ridgeline plot.")
                    render_chat_analysis_panel(context={'plot_type': 'ridgeline', 'description': 'Evolution of semantic features (PCA and t-SNE) across sessions'}, tab_id="ridgeline")
                elif i == 6:
                    st.header("Clinical Analysis & Chat")
                    render_comprehensive_chat_analysis()
    else:
        st.info("No sessions available. Add some sessions using the sidebar input or import a CSV file.")

if __name__ == "__main__":
    main() 