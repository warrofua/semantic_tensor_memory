# Semantic Tensor Analysis

A **temporal semantic evolution analysis framework** for tracking how meaning changes across time, tokens, and context. STA provides ready-made analysis workflows, drift metrics, and specialized visualizations for understanding semantic trajectories through sessions of text.

## What is STA?

**Semantic Tensor Analysis** is a toolkit for analyzing how meaning evolves across temporal sequences:

- ✅ **Analysis framework** for temporal semantic evolution
- ✅ **Visualization toolkit** for meaning drift across sessions
- ✅ **Research platform** for studying how concepts evolve over time
- ✅ **Domain-specific workflows** for clinical progress tracking, learning journeys, conversation analysis, and research note evolution

### Core Innovation: Dual-Resolution Temporal Analysis

STA tracks semantic meaning at two resolutions simultaneously:
- **Token-level granularity** (via BERT tokenization) - track individual concept drift
- **Sequence-level semantics** (via Sentence-BERT) - track holistic meaning evolution

This enables both fine-grained token alignment analysis and high-level semantic trajectory computation.

---

## 💡 Concrete Example: What Can STA Tell You?

**Scenario: ABA Therapy Progress Tracking**

You have 30 therapy session notes for a patient over 6 months:

```python
# Load sessions
store = UniversalMemoryStore()
for note in therapy_notes:
    store.add_session(note)

# Run analysis
```

**STA automatically reveals:**

1. **Semantic Trajectory** (`evolution_tab`)
   - "Patient meaning shifted from cluster 'behavioral challenges' (weeks 1-8) to 'skill acquisition' (weeks 9-20) to 'generalization' (weeks 21-30)"
   - Velocity graph shows rapid progress in weeks 12-15, plateau in weeks 22-26

2. **Token-Level Drift** (`token_alignment_heatmap`)
   - Words that appeared/disappeared: "tantrum" (high early, faded), "independence" (emerged week 10), "peer interaction" (emerged week 18)
   - Optimal alignment shows which specific concepts persisted vs. transformed

3. **Concept Evolution** (`concepts_tab`)
   - KMeans identifies 4 semantic clusters: "regulation struggles", "skill building", "social engagement", "mastery"
   - Transition graph shows patient moved through clusters sequentially with brief regression in week 23

4. **Inflection Points** (`trajectory_computation`)
   - Week 12: Acceleration spike (breakthrough moment)
   - Week 23: Temporary deceleration (regression or plateau)
   - Week 28: Final acceleration (consolidation phase)

5. **PCA Narrative** (`dimensionality_tab` + LLM)
   - "PC1 (43% variance) represents 'independence vs. support needs'"
   - "PC2 (28% variance) represents 'emotional regulation vs. dysregulation'"
   - "Patient trajectory: moved positively along PC1 while PC2 oscillated, then stabilized"

6. **Domain-Aware Insights** (`AI_insights_tab`)
   - "Based on 6-month span, this represents a typical ABA intensive phase"
   - "The regression in week 23 aligns with expected variance in skill acquisition"
   - "Recommend: Continue current approach, monitor for sustained generalization"

**All of this from just uploading a CSV.** No custom code, no manual analysis.

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
- `src/semantic_tensor_analysis/streamlit/utils.py`: Data loading, PCA pipeline (mask-aware), session state and prompt helpers
- `src/semantic_tensor_analysis/streamlit/plots.py`: Streamlit-specific plotting helpers (Plotly/Altair/inline Matplotlib)
- `src/semantic_tensor_analysis/chat/analysis.py`: LLM prompts and analysis (Ollama), domain-aware insights with time-scale inference
- `src/semantic_tensor_analysis/memory/`: Core memory implementation
  - `universal_core.py`: Universal STA types and `UniversalMemoryStore` (dynamic dims, ragged sequences)
  - `text_embedder.py`: Dual-resolution text embeddings (token-level BERT + sentence-level S-BERT)
  - `embedder.py` / `embedder_sbert.py` / `embedder_hybrid.py`: Embedding backends
  - `drift.py` / `sequence_drift.py`: Drift metrics, token alignment (Hungarian), token-importance drift
  - `store.py`: Storage utilities
- `src/semantic_tensor_analysis/analytics/tensor_batching.py`: Ragged tensor batching utilities (`pad_and_stack`, `masked_session_means`, `flatten_with_mask`)
- `src/semantic_tensor_analysis/visualization/viz/`: Visualization tools
  - `heatmap.py`: Similarity heatmaps, token alignment heatmap (returns Matplotlib Figure)
  - `pca_plot.py`, `pca_summary.py`, `semantic_analysis.py`, `holistic_semantic_analysis.py`
- `src/semantic_tensor_analysis/visualization/tools/`: Additional concept visualizers
- `src/semantic_tensor_analysis/demos/`: CLI demos and dataset helpers
- `ultimate_demo_dataset.csv`: Rich demo dataset
- `aba_therapy_dataset.csv`: ABA therapy dataset (and extended version for same client)
- `archive/`: Historical docs (safe to remove if not needed)
- `pyproject.toml`: Python package metadata and dependencies

---

## ✨ Features: What Makes STA Unique

### 🎯 Temporal Semantic Analysis

**1. Ragged Tensor Operations for Variable-Length Sessions**
- Handle sessions with 100-500 tokens each using masked batch operations
- `pad_and_stack()`: Convert variable-length sequences to padded tensors with masks
- `masked_session_means()`: Compute statistics ignoring padding
- Critical for session-level analysis while preserving token-level detail

**2. Token-Level Drift Analysis via Optimal Alignment**
- Hungarian algorithm for optimal token-to-token matching across sessions
- Identify which specific concepts/words changed between sessions
- `token_importance_drift()`: Quantify which tokens drifted most
- Token alignment heatmaps showing cross-session concept mapping

**3. Semantic Trajectory Computation**
- **Velocity**: Rate of meaning change between consecutive sessions
- **Acceleration**: When semantic change accelerated or decelerated
- **Inflection point detection**: Identify when meaning shifted rapidly
- 3D trajectory visualization through semantic space over time

**4. Global Temporal Operations**
- **PCA across all sessions**: Define semantic axes based on complete dataset
- **Concept clustering**: Group sessions by semantic similarity (KMeans)
- **Cluster transition tracking**: When did meaning move from theme A to theme B?
- **Temporal patterns**: Identify cycles, trends, and evolution patterns

### 📊 Specialized Visualizations (~40% of Codebase)

- **Semantic Drift River**: 3D "river" visualization of meaning flow over time
- **PCA Timeline Animations**: Watch semantic position evolve across sessions
- **Temporal Heatmaps**: Session-to-session similarity matrices
- **Token Alignment Plots**: Visualize optimal token matching
- **Concept Evolution Graphs**: How themes emerged, peaked, and faded
- **Ridgeline Plots**: Distribution evolution across time
- **Liminal Tunnel Visualization**: Immersive drift tunnel
- **4D Semantic Space**: Multi-dimensional semantic trajectories

### 🧠 Domain-Adaptive AI Analysis

- **Dual-resolution embeddings**: Token-level (BERT) + sentence-level (S-BERT) simultaneously
- **Domain inference**: Automatically detect ABA therapy, clinical notes, learning journeys, research
- **Time-scale awareness**: Adaptive analysis for daily, weekly, monthly, or quarterly patterns
- **LLM-powered insights**: Via Ollama integration for semantic narrative generation
- **Axis interpretation**: Explain what PCA dimensions mean in domain-specific terms

### 🛠️ Ready-Made Analysis Workflows

- **Clinical Progress Tracking**: ABA therapy, patient notes, treatment evolution
- **Learning Journey Mapping**: How understanding evolves through course materials
- **Research Evolution**: Track how hypotheses and questions change over time
- **Conversation Analysis**: Understand topic drift in dialogue
- **Content Versioning**: Analyze how writing/ideas change across drafts

### 📦 Production-Ready Features

- Multi-format data ingestion (CSV, JSON, TXT)
- Persistent storage with CPU/GPU portability
- Session state management for interactive exploration
- Performance optimization with adaptive processing
- Comprehensive test suite (6 test modules, 13k+ lines)
- Streamlit web UI with 7 analysis tabs
- CLI demo for rapid iteration

---

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
- The codebase implements STA with ragged tensor handling, dual-resolution embeddings, token alignment, and domain-aware LLM interpretation.

### Feature Correspondence Table

| Area                | Paper Coverage | Codebase Coverage | Notes                                                      |
|---------------------|----------------|-------------------|------------------------------------------------------------|
| STA Architecture    | Yes            | Yes               | Aligned; dynamic dims and ragged sequences implemented.    |
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

## 🧩 Technical Architecture: Why Sessions, Not Individual Vectors?

### The Session-Based Approach

STA operates on **sessions** (temporal snapshots containing variable-length sequences), not individual vectors:

```python
# A session is a variable-length sequence
session = UniversalEmbedding(
    event_embeddings=[token_1_emb, token_2_emb, ..., token_n_emb],  # n varies per session
    sequence_embedding=session_mean,  # Holistic meaning
    events=[EventDescriptor(...), ...]  # Token metadata
)

# Sessions vary in length:
session_1: [100 tokens × 768 dims]
session_2: [237 tokens × 768 dims]
session_3: [89 tokens × 768 dims]
```

This enables **dual-resolution analysis**: zoom into token-level details or analyze session-level trends.

### Ragged Tensor Operations with Masking

The key innovation for handling variable-length sessions:

```python
from semantic_tensor_analysis.analytics.tensor_batching import (
    pad_and_stack,
    masked_session_means,
    flatten_with_mask
)

# Convert ragged sequences to batched tensor
sessions_tensor, mask = pad_and_stack(sessions)
# Shape: [3, 237, 768]  (padded to max length = 237)
# Mask: [3, 237] boolean  (False = padding, ignore in computation)

# Compute session-level statistics (ignoring padding)
session_means = masked_session_means(sessions_tensor, mask)
# Shape: [3, 768] - one mean per session

# Flatten to token level with provenance tracking
flat_tokens, session_ids, token_ids = flatten_with_mask(sessions_tensor, mask)
# flat_tokens: [426, 768]  (100 + 237 + 89 tokens total)
# session_ids: [426]  (which session each token came from)
# token_ids: [426]  (position within session)
```

**Why this matters:**
- Padding doesn't corrupt statistics (masked operations)
- Can analyze at session OR token granularity seamlessly
- Enables optimal token alignment across sessions (Hungarian algorithm)
- PCA can operate on all tokens while preserving session boundaries

### Flow: Raw Data → Analysis → Visualization

```
CSV/JSON/TXT
    ↓
Text Embedding (dual-resolution)
    ├→ Token embeddings [n_tokens, 768] via BERT
    └→ Sequence embedding [768] via Sentence-BERT
    ↓
Session Creation (UniversalEmbedding)
    ↓
Storage (UniversalMemoryStore)
    ↓
Ragged Tensor Batching (pad_and_stack)
    ↓
Global Analysis
    ├→ PCA across all sessions/tokens
    ├→ Concept clustering (KMeans)
    ├→ Token alignment (Hungarian)
    └→ Drift computation (cosine distance)
    ↓
Visualization
    ├→ Temporal trajectories (velocity, acceleration)
    ├→ Heatmaps (session similarity, token alignment)
    ├→ 3D semantic space (PCA projection)
    └→ Concept evolution graphs
    ↓
Optional: LLM narrative generation (Ollama)
```

**The key insight:** Operations are **across sessions** (temporal), not **within a database** (spatial).

---

## 🤔 FAQ: Why Not Just Use...?

### "Why not just use a Jupyter notebook with sklearn?"

**You can!** STA essentially packages what you'd build in a research notebook into a reusable framework:

**Without STA:**
```python
# You'd need to implement:
- Dual BERT + S-BERT embedding pipeline
- Ragged tensor padding and masking logic
- Hungarian algorithm for token alignment
- Drift velocity/acceleration computation
- 10+ specialized visualization functions
- Domain-adaptive prompts for LLM analysis
- Streamlit UI for interactive exploration
```

**With STA:**
```python
# Just load your data
store = UniversalMemoryStore()
for session in sessions:
    store.add_session(session)

# Everything else is ready to use
```

**STA saves you from re-implementing this infrastructure for every temporal semantic analysis project.**

### "Why not use LangSmith or W&B for tracking?"

**Great tools, different purposes:**

| Feature | LangSmith | W&B | STA |
|---------|-----------|-----|-----|
| **Conversation tracking** | ✅ Excellent | ❌ | ✅ |
| **Metric dashboards** | ✅ | ✅ Excellent | ✅ |
| **Semantic drift analysis** | ❌ | ❌ | ✅ Token + session level |
| **Token alignment** | ❌ | ❌ | ✅ Hungarian algorithm |
| **Trajectory computation** | ❌ | ❌ | ✅ Velocity, acceleration |
| **Domain-specific workflows** | ❌ | ❌ | ✅ Clinical, learning, research |

**Use LangSmith/W&B for production monitoring. Use STA for deep temporal semantic analysis.**

### "Why not just compute cosine similarity between embeddings?"

Simple similarity misses **temporal patterns**:

```python
# Simple approach: pairwise similarity
similarity(session_1, session_2)  # → 0.87
similarity(session_2, session_3)  # → 0.82

# STA approach: temporal dynamics
velocity = compute_drift_velocity([session_1, session_2, session_3])
# → [0.13, 0.18]  (change is accelerating)

inflection_points = detect_rapid_shifts(velocity)
# → [session_5, session_12]  (when meaning changed rapidly)

token_drift = token_importance_drift(session_1, session_3)
# → ["anxiety": high drift, "coping": low drift]  (which concepts changed)
```

**STA provides the calculus of semantic change, not just static snapshots.**

### "Why session-based instead of continuous streaming?"

**Session-based is intentional** for certain domains:

- **Clinical notes**: Each therapy session is a natural boundary
- **Learning journeys**: Each lesson/assignment is discrete
- **Research evolution**: Each draft/experiment is a snapshot
- **Meeting summaries**: Each meeting is a unit of analysis

**Future work**: STA could support streaming by defining windows, but sessions align with how many domains naturally structure temporal data.

---

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
