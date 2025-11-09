# Semantic Tensor Memory

A structured, interpretable memory system for tracking meaning across time, tokens, and context. This implementation provides a working prototype of the Semantic Tensor Memory (STM) system described in the accompanying paper/TeX write-up.

---

## 🚀 Quick Start

1. **(Optional) Create and activate a venv:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    - Additional system requirement: For the CLI CSV import feature, ensure your Python installation includes `tkinter` (standard on most desktop Python distributions).

3. **Start the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    - On first load, the sidebar opens to let you upload a CSV. After upload, the sidebar stays minimized for more canvas space.
    - Try with `ultimate_demo_dataset.csv` or `aba_therapy_dataset.csv` in the repo root.

4. **Interactive CLI demo (optional):**
    ```bash
    python demo.py
    ```
    - Type sentences to build memory, `import` to load a CSV (requires `tkinter`), `plot` for PCA/heatmap, `drift` for metrics, `tokens` for token-level drift, `exit` to quit.

---

## 🗂️ Project Structure

- `app.py`: Streamlit web application (tabs: Overview, Evolution, Patterns, Dimensionality, Concepts, Explain, AI Insights)
- `src/semantic_tensor_memory/streamlit/utils.py`: Data loading, PCA pipeline (mask-aware), session state and prompt helpers
- `src/semantic_tensor_memory/streamlit/plots.py`: Streamlit-specific plotting helpers (Plotly/Altair/inline Matplotlib)
- `src/semantic_tensor_memory/chat/analysis.py`: LLM prompts and analysis (Ollama), domain-aware insights with time-scale inference
- `src/semantic_tensor_memory/memory/`: Core memory implementation
  - `universal_core.py`: Universal STM types and `UniversalMemoryStore` (dynamic dims, ragged sequences)
  - `text_embedder.py`: Dual-resolution text embeddings (token-level BERT + sentence-level S-BERT)
  - `embedder.py` / `embedder_sbert.py` / `embedder_hybrid.py`: Embedding backends
  - `drift.py` / `sequence_drift.py`: Drift metrics, token alignment (Hungarian), token-importance drift
  - `store.py`: Storage utilities
- `src/semantic_tensor_memory/analytics/tensor_batching.py`: Ragged tensor batching utilities (`pad_and_stack`, `masked_session_means`, `flatten_with_mask`)
- `src/semantic_tensor_memory/visualization/viz/`: Visualization tools
  - `heatmap.py`: Similarity heatmaps, token alignment heatmap (returns Matplotlib Figure)
  - `pca_plot.py`, `pca_summary.py`, `semantic_analysis.py`, `holistic_semantic_analysis.py`
- `src/semantic_tensor_memory/visualization/tools/`: Additional concept visualizers
- `src/semantic_tensor_memory/demos/`: CLI demos and dataset helpers
- `ultimate_demo_dataset.csv`: Rich demo dataset
- `aba_therapy_dataset.csv`: ABA therapy dataset (and extended version for same client)
- `archive/`: Historical docs (safe to remove if not needed)
- `pyproject.toml`: Python package metadata and dependencies

---

## ✨ Features

- Dual-resolution embeddings (token-level BERT + sentence-level S-BERT)
- Ragged tensors with padding and masks for batch ops (`tensor_batching.py`)
- Mask-aware PCA pipeline with diagnostics; token or session-mean granularity
- Token alignment heatmaps (Hungarian alignment) and token-importance drift
- Temporal drift visuals: PCA trajectories, similarity heatmaps, temporal heatmaps
- Concept analysis: clustering, evolution, exemplar alignment in clusters
- Dimensionality tab with LLM Axis Explainer
- Explain tab with AI explanations (`what_it_means`, `why_these_results`, `what_to_do_next`)
- AI Insights: domain-aware prompt that infers appropriate time scale (days/weeks/months/quarters) from dataset span
- Streamlit UX: first-load expanded sidebar, minimized after upload; inline Matplotlib (no blocking windows)
- Datasets: `ultimate_demo_dataset.csv`, `aba_therapy_dataset.csv`
## 📦 Datasets

- `ultimate_demo_dataset.csv`: High-quality demo with clear trajectories and richer, longer texts.
- `aba_therapy_dataset.csv`: ABA-specific schema/content; extended to a larger set for the same client.

Upload either via the Streamlit sidebar to explore the full suite of analyses.

Expected columns (typical): `session_id`, `date`, `title` (optional), `text`.

---

## 💡 Extensions & Ideas

- Drift alerts
- Sentence search
- HTML dashboard
- LLM explanation layer
- Clinical applications

---

## ⚠️ Notes

- The `venv/` directory is excluded from git and should not be committed.
- For LLM-powered summaries, ensure Ollama is installed and running with a supported model (e.g., `qwen3:latest`).
- The Streamlit app renders Matplotlib figures inline; no external windows will block interaction.
- Key dependencies: `torch`, `transformers`, `scikit-learn`, `plotly`, `streamlit`, `pandas`, `numpy`, `rich`, `requests`. The CLI CSV import feature requires `tkinter`.

---

## 📄 Citation
If you use this codebase or ideas in your research, please cite the accompanying paper or link to this repository.

---

## 📄 Documentation Alignment: Paper/TeX vs. Codebase

This section maps the `semantic-tensor-memory.tex` write-up (and associated PDF) to the codebase. It documents feature completeness and correspondence.

### Overview

- The paper/TeX describes the motivation, architecture, algorithms, applications, and limitations of STM.
- The codebase implements STM with ragged tensor handling, dual-resolution embeddings, token alignment, and domain-aware LLM interpretation.

### Feature Correspondence Table

| Area                | Paper Coverage | Codebase Coverage | Notes                                                      |
|---------------------|----------------|-------------------|------------------------------------------------------------|
| STM Architecture    | Yes            | Yes               | Aligned; dynamic dims and ragged sequences implemented.    |
| Data Import         | Yes            | Yes               | CSV upload in Streamlit; CLI import with tkinter.          |
| Visualization       | Yes            | Yes               | PCA, heatmaps, token alignment, token trajectories.        |
| LLM Integration     | Yes            | Yes               | Axis Explainer; domain-aware insights with time scale.     |
| Applications        | Yes            | Yes               | ABA and general datasets provided.                         |
| Example Analysis    | Yes            | Yes               | Demo datasets included.                                    |
| Limitations/Future  | Yes            | Partial           | Multimodal audio, alerts, streaming, storage optimizations.|
| UI/CLI Details      | Brief          | Yes               | More detail in codebase/README than in paper.              |
| Figures             | Yes            | Yes               | All figures rendered inline in app; assets can be saved.   |

### Summary

- All major features and analyses described in the paper are implemented.
- The code includes practical details (CLI commands, Streamlit UI) beyond the paper.
- Remaining roadmap items: audio modality, drift alerts/governance, streaming ingestion, storage efficiency, and expanded tests/CI.

---

## 🧩 How tensors flow (ragged → padded + mask)

- Token embeddings are variable-length per session (ragged).
- `tensor_batching.pad_and_stack` pads to `max_tokens` and returns `(batch, mask)`.
- Downstream ops (PCA, heatmaps) use masks to ignore padding:
  - `masked_session_means(batch, mask)` for session granularity
  - `flatten_with_mask(batch, mask)` for token granularity with `(session_ids, token_ids)`

## 🔗 Token alignment & drift

- Consecutive/session-pair alignment via Hungarian algorithm (in `sequence_drift.py`).
- Visualize with `viz.heatmap.token_alignment_heatmap` (returns a Matplotlib Figure; rendered inline in Streamlit).

## 🧠 AI prompts

- Prompts in `chat_analysis.py` infer domain and an appropriate time scale (days/weeks/months/quarters) from the dataset date span.
- Explain tab uses `AnalysisExplanation` fields: `what_it_means`, `why_these_results`, `what_to_do_next`.

---

## 🛠️ Troubleshooting

- Port 8501 in use: `lsof -ti:8501 | xargs -r kill -9`
- Ollama not running: install/start Ollama and pull a model (e.g., `qwen3:latest`).
- PyTorch view/reshape error: the PCA pipeline uses `.reshape(...)` and contiguous tensors in `tensor_batching.py`.
- `pytest` not found: install via `pip install pytest` or use the app directly.
