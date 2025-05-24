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
    
    # Display chat history
    for role, msg in chat_history:
        with st.chat_message(role):
            st.markdown(msg)
    
    # User input
    user_input = st.chat_input("Ask a follow-up about the analysis or data...", key=f"chat_input_{chat_key}")
    if user_input and not is_streaming:
        st.session_state[chat_key].append(("user", user_input))
        add_chat_message("user", user_input)
        st.rerun()

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
        tab_names = ["Drift Analysis", "PCA Map", "3D PCA Map", "Heatmap", "Token Trajectory (3D)", "Clinical Analysis & Chat"]
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
                        render_chat_analysis_panel(context={'drifts': drifts[:5], 'drifts_full': drifts}, tab_id="drift")
                    else:
                        st.warning("Need ≥2 sessions to plot drift.")
                        render_chat_analysis_panel(context={'drifts': [], 'drifts_full': []}, tab_id="drift")
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
                        from viz.pca_plot import prepare_for_pca
                        import warnings
                        import traceback
                        try:
                            flat, session_ids, token_ids = prepare_for_pca(memory_slice)
                            # Robust data validation
                            if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
                                st.error("Data contains NaN, Inf, or extremely large values. Please check your input data.")
                                return
                            flat = (flat - np.mean(flat, axis=0)) / np.std(flat, axis=0)
                            if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
                                st.error("Data contains NaN, Inf, or extremely large values after normalization. Please check your input data.")
                                return
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=2)
                            reduced = pca.fit_transform(flat)
                            # Create DataFrame for plotting
                            df = pd.DataFrame({
                                'PCA1': reduced[:, 0],
                                'PCA2': reduced[:, 1],
                                'Session': [f"Session {j+1}" for j in session_ids],
                                'SessionIdx': session_ids,
                                'Text': [meta_slice[j]['text'] for j in session_ids]
                            })
                            fig = px.scatter(
                                df,
                                x='PCA1',
                                y='PCA2',
                                color='SessionIdx',
                                color_continuous_scale='RdYlBu',
                                range_color=[df['SessionIdx'].min(), df['SessionIdx'].max()],
                                hover_data=['Session', 'Text'],
                                title="Semantic Drift Map"
                            )
                            fig.update_traces(marker=dict(reversescale=True))
                            fig.update_layout(
                                hovermode="closest",
                                showlegend=False,
                                coloraxis_colorbar=dict(title="Session")
                            )
                            # Feature labels for 2D
                            extremes = []
                            for axis, col in enumerate(['PCA1', 'PCA2']):
                                max_idx = df[col].idxmax()
                                min_idx = df[col].idxmin()
                                for idx, label in zip([max_idx, min_idx], [f"{col} +", f"{col} -"]):
                                    text_excerpt = ' '.join(df.loc[idx, 'Text'].split()[:6]) + ('...' if len(df.loc[idx, 'Text'].split()) > 6 else '')
                                    extremes.append({
                                        'PCA1': float(df.loc[idx, 'PCA1']),
                                        'PCA2': float(df.loc[idx, 'PCA2']),
                                        'label': f"{label}: {text_excerpt}"
                                    })
                            for ex in extremes:
                                fig.add_trace(go.Scatter(
                                    x=[ex['PCA1']], y=[ex['PCA2']],
                                    mode='markers+text',
                                    marker=dict(size=10, color='black', symbol='diamond'),
                                    text=[ex['label']],
                                    textposition='top center',
                                    showlegend=False
                                ))
                            st.plotly_chart(fig, use_container_width=True)
                            explained_variance_str = ', '.join([f'{v:.2%}' for v in pca.explained_variance_ratio_])
                            st.write(f"PCA explained variance: {explained_variance_str}")
                            from viz.pca_summary import generate_narrative_summary
                            summary = generate_narrative_summary(reduced, session_ids, token_ids, meta_slice)
                            st.subheader("Narrative Summary")
                            st.info(summary)
                            # Store PCA results in session state for use in clinical tab
                            st.session_state.pca_results = {
                                'reduced': reduced,
                                'session_ids': session_ids,
                                'token_ids': token_ids,
                                'meta': meta_slice
                            }
                        except Exception as e:
                            st.error(f"PCA failed: {e}")
                            explained_variance_str = "N/A"
                            summary = "PCA analysis failed"
                    else:
                        st.warning("Need ≥2 sessions for PCA analysis.")
                        explained_variance_str = "N/A"
                        summary = "Insufficient data for PCA analysis"
                    render_chat_analysis_panel(context={'explained_variance': explained_variance_str, 'narrative_summary': summary}, tab_id="pca2d")
                elif i == 2:
                    st.header("3D PCA Map")
                    explained_variance_str = ""
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
                        from viz.pca_plot import prepare_for_pca
                        import warnings
                        import traceback
                        try:
                            flat, session_ids, token_ids = prepare_for_pca(memory_slice)
                            # Robust data validation
                            if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
                                st.error("Data contains NaN, Inf, or extremely large values. Please check your input data.")
                                return
                            flat = (flat - np.mean(flat, axis=0)) / np.std(flat, axis=0)
                            if np.isnan(flat).any() or np.isinf(flat).any() or np.abs(flat).max() > 1e6:
                                st.error("Data contains NaN, Inf, or extremely large values after normalization. Please check your input data.")
                                return
                            from sklearn.decomposition import PCA
                            pca = PCA(n_components=3)
                            reduced = pca.fit_transform(flat)
                            # Create DataFrame for plotting
                            df = pd.DataFrame({
                                'PCA1': reduced[:, 0],
                                'PCA2': reduced[:, 1],
                                'PCA3': reduced[:, 2],
                                'Session': [f"Session {j+1}" for j in session_ids],
                                'SessionIdx': session_ids,
                                'Text': [meta_slice[j]['text'] for j in session_ids]
                            })
                            # Find axis extremes for feature labeling
                            extremes = []
                            for axis, col in enumerate(['PCA1', 'PCA2', 'PCA3']):
                                max_idx = df[col].idxmax()
                                min_idx = df[col].idxmin()
                                for idx, label in zip([max_idx, min_idx], [f"{col} +", f"{col} -"]):
                                    text_excerpt = ' '.join(df.loc[idx, 'Text'].split()[:6]) + ('...' if len(df.loc[idx, 'Text'].split()) > 6 else '')
                                    extremes.append({
                                        'PCA1': float(df.loc[idx, 'PCA1']),
                                        'PCA2': float(df.loc[idx, 'PCA2']),
                                        'PCA3': float(df.loc[idx, 'PCA3']),
                                        'label': f"{label}: {text_excerpt}"
                                    })
                            fig = px.scatter_3d(
                                df,
                                x='PCA1',
                                y='PCA2',
                                z='PCA3',
                                color='SessionIdx',
                                color_continuous_scale='RdYlBu',
                                range_color=[df['SessionIdx'].min(), df['SessionIdx'].max()],
                                hover_data=['Session', 'Text'],
                                title="3D Semantic Drift Map"
                            )
                            fig.update_traces(marker=dict(size=4, reversescale=True))
                            fig.update_layout(
                                scene = dict(
                                    xaxis_title='<b>PCA-1</b><br><span style="font-size:12px; color:gray">Semantic Dimension 1</span>',
                                    yaxis_title='<b>PCA-2</b><br><span style="font-size:12px; color:gray">Semantic Dimension 2</span>',
                                    zaxis_title='<b>PCA-3</b><br><span style="font-size:12px; color:gray">Semantic Dimension 3</span>',
                                ),
                                coloraxis_colorbar=dict(title="Session"),
                                margin=dict(l=0, r=0, b=0, t=40)
                            )
                            # Add feature labels at axis extremes
                            for ex in extremes:
                                fig.add_trace(go.Scatter3d(
                                    x=[ex['PCA1']], y=[ex['PCA2']], z=[ex['PCA3']],
                                    mode='markers+text',
                                    marker=dict(size=10, color='black', symbol='diamond'),
                                    text=[ex['label']],
                                    textposition='top center',
                                    showlegend=False
                                ))
                            st.plotly_chart(fig, use_container_width=True)
                            explained_variance_str = ', '.join([f'{v:.2%}' for v in pca.explained_variance_ratio_])
                            st.write(f"PCA explained variance: {explained_variance_str}")
                        except Exception as e:
                            st.error(f"3D PCA failed: {e}")
                            explained_variance_str = "N/A"
                    else:
                        st.warning("Need ≥2 sessions for 3D PCA analysis.")
                        explained_variance_str = "N/A"
                    render_chat_analysis_panel(context={'explained_variance': explained_variance_str}, tab_id="pca3d")
                elif i == 3:
                    st.header("Session Heatmap")
                    if len(st.session_state.memory) > 1:
                        fig = plot_heatmap_plotly(st.session_state.memory)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Need ≥2 sessions for heatmap analysis.")
                    render_chat_analysis_panel(context={'notable_distances': "No notable distances found"}, tab_id="heatmap")
                elif i == 4:
                    st.header("Token Trajectory (3D)")
                    if len(st.session_state.memory) > 0:
                        session_idx = st.selectbox(
                            "Select session to visualize:",
                            options=list(range(len(st.session_state.memory))),
                            format_func=lambda j: f"Session {j+1}"
                        )
                        tensor = st.session_state.memory[session_idx]
                        embeddings = tensor.numpy()
                        session_text = st.session_state.meta[session_idx]['text'] if 'text' in st.session_state.meta[session_idx] else ''
                        if embeddings.shape[0] < 2:
                            st.warning("Session has fewer than 2 tokens; cannot plot trajectory.")
                        else:
                            # Robust data validation
                            if np.isnan(embeddings).any() or np.isinf(embeddings).any() or np.abs(embeddings).max() > 1e6:
                                st.error("Token embeddings contain NaN, Inf, or extremely large values. Cannot plot trajectory.")
                            else:
                                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                                tokens = tokenizer.tokenize(session_text)
                                if len(tokens) > embeddings.shape[0]:
                                    tokens = tokens[:embeddings.shape[0]]
                                elif len(tokens) < embeddings.shape[0]:
                                    tokens += [''] * (embeddings.shape[0] - len(tokens))
                                from sklearn.decomposition import PCA
                                pca = PCA(n_components=3)
                                embeddings_3d = pca.fit_transform(embeddings)
                                # Compute drift between consecutive tokens in 3D
                                drift = np.linalg.norm(np.diff(embeddings_3d, axis=0), axis=1)
                                drift = np.insert(drift, 0, 0)  # First token has no previous
                                # Highlight tokens with high drift (mean + std)
                                high_drift = drift > (drift.mean() + drift.std())
                                colors = np.where(high_drift, 'red', 'blue')
                                hover_text = [f"{tok}<br>Drift: {d:.3f}" for tok, d in zip(tokens, drift)]
                                fig = go.Figure(data=[go.Scatter3d(
                                    x=embeddings_3d[:,0],
                                    y=embeddings_3d[:,1],
                                    z=embeddings_3d[:,2],
                                    mode='lines+markers',
                                    marker=dict(size=6, color=colors, opacity=0.8),
                                    line=dict(width=2),
                                    text=hover_text,
                                    hoverinfo='text'
                                )])
                                fig.update_layout(
                                    title=f'Token Trajectory in 3D Embedding Space (Session {session_idx+1})',
                                    scene = dict(
                                        xaxis_title='PCA-1',
                                        yaxis_title='PCA-2',
                                        zaxis_title='PCA-3',
                                    ),
                                    margin=dict(l=0, r=0, b=0, t=40)
                                )
                                st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No sessions available to visualize.")
                    render_chat_analysis_panel(context={'high_drift_tokens': "No high-drift tokens found"}, tab_id="trajectory")
                elif i == 5:
                    st.header("Clinical Analysis & Chat")
                    render_chat_analysis_panel(tab_id="clinical")
    else:
        st.info("No sessions available. Add some sessions using the sidebar input or import a CSV file.")

if __name__ == "__main__":
    main() 