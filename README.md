# Semantic Tensor Analysis
[![CI](https://github.com/warrofua/semantic-tensor-analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/warrofua/semantic-tensor-analysis/actions/workflows/tests.yml)

Semantic Tensor Analysis (STA) sits on top of your embeddings or vector store and gives you temporal drift, trajectories, and token-level alignment—no custom infra required. It keeps token-level detail alongside session-level summaries so you can inspect drift without losing context.

## What is STA?

- ✅ Token- and session-level embeddings (BERT + Sentence-BERT) in one pipeline
- ✅ Drift metrics and clustering for ordered text sessions (CSV/JSON/TXT)
- ✅ Visual explanations (PCA, heatmaps, trajectories) tailored to time-ordered data
- ✅ Domain-adaptive summaries for learning logs, project journals, research notes, and conversations
- ✅ Context-aware embeddings (optional prior-session and timestamp signals) with configurable token drift caps

### Who is this for?
- Researchers tracking concept drift over time
- Builders and analysts working with time-ordered text (journals, standups, changelogs, feedback)
- Anyone with time-stamped text who wants more than cosine similarity

### Why STA (what’s different)
- Dual-resolution memory: token-level (BERT) + sequence-level (SBERT) stored together for Hungarian token alignment, token drift heatmaps, and session trajectories without re-embedding.
- Contextual embedding: optional rolling context window + timestamps become lightweight prefixes to stabilize SBERT sequence embeddings while keeping raw token granularity intact.
- Ragged, mask-aware analytics: pad/stack/flatten utilities consistently handle variable-length sessions across PCA, clustering, trajectories—no silent truncation.
- Temporal semantics first: velocity/acceleration of meaning, inflection-point cues, and multi-view trajectories for ordered text (not just static similarity).
- Concept evolution with alignment: session clustering + transition graphs plus token alignment to show *what* moved and *how*.
- Vision grounding for charts: server-side Plotly→PNG snapshots fed to local vision GGUF (llama.cpp); graceful fallback to text-only if vision isn’t available.
- Storage hygiene: built-in storage stats/cleanup (sidebar + CLI), CPU-portable persistence.
- Grounded LLM context: prompts reuse analysis context (clusters, PCA axes, drift) instead of generic summaries.
- Browser-safe ingestion: large CSV/JSON uploads auto-sample to stay within a conservative browser memory budget; configure context window and token-alignment cap from the sidebar.

### Core approach: dual resolution

STA tracks meaning at two resolutions:
- **Token-level** (BERT): follow individual concept drift
- **Session-level** (Sentence-BERT): follow overall semantic movement

Both are kept so you can align tokens while also inspecting higher-level trajectories.

---

## 💡 Concrete Example: What Can STA Tell You?

**Scenario: Project Journal / Sprint Notes**

You have 30 weekly sprint notes over ~6 months:

```python
# Load sessions
store = UniversalMemoryStore()
for note in sprint_notes:
    store.add_session(note)

# Run analysis
```

**STA automatically reveals:**

1. **Semantic Trajectory** (`evolution_tab`)
   - "Themes shifted from 'scoping & uncertainty' (weeks 1–8) to 'implementation & delivery' (weeks 9–20) to 'stabilization & maintenance' (weeks 21–30)"
   - Velocity graph shows rapid change around weeks 12–15, plateau around weeks 22–26

2. **Token-Level Drift** (`token_alignment_heatmap`)
   - Words that appeared/disappeared: "prototype" (high early, faded), "deadline" (emerged week 10), "refactor" (emerged week 18)
   - Optimal alignment shows which specific concepts persisted vs. transformed

3. **Concept Evolution** (`concepts_tab`)
   - KMeans identifies clusters like "planning", "delivery", "quality", "launch"
   - Transition graph shows the narrative moved through clusters with a brief regression in week 23

4. **Inflection Points** (`trajectory_computation`)
   - Week 12: Acceleration spike (breakthrough moment)
   - Week 23: Temporary deceleration (regression or plateau)
   - Week 28: Final acceleration (consolidation phase)

5. **PCA Narrative** (`dimensionality_tab` + LLM)
   - "PC1 (43% variance) represents 'exploration vs. execution'"
   - "PC2 (28% variance) represents 'quality vs. speed'"
   - "Trajectory: moved toward execution while oscillating on quality, then stabilized"

6. **Narrative Insights** (`AI_insights_tab`)
   - "The mid-project shift aligns with moving from ideation to delivery"
   - "The regression around week 23 matches a scope change or external constraint"
   - "Recommend: clarify milestones, watch for recurring blockers, and validate outcomes"

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
    - Try with `ultimate_demo_dataset.csv` or `finance_notes_expanded.csv` for a bigger stress test.

4. **Interactive CLI demo (optional):**
    ```bash
    python demo.py
    ```
    - Type sentences to build memory, `import` to load a CSV (requires `tkinter`), `plot` for PCA/heatmap, `drift` for metrics, `tokens` for token-level drift, `exit` to quit.

---

## 🗂️ Project Structure

- `app.py`: Streamlit web app (tabs: Overview, Evolution, Patterns, Dimensionality, Concepts, Explain, AI Insights); wires sidebar chat and loaders.
- `src/semantic_tensor_analysis/app/`: App modules (`main.py`, `tabs/`, `sidebar.py`, `sidebar_chat.py`, `temporal_visualizations.py`, assets/config).
- `src/semantic_tensor_analysis/memory/`: Core STM types (`universal_core.py`), text embedder (`text_embedder.py`), drift (`drift.py`, `sequence_drift.py`), storage (`store.py`), legacy shim modules that forward to `archive/legacy_embedders/` when explicitly enabled.
- `src/semantic_tensor_analysis/storage/`: Storage manager/stats/cleanup utilities (`manager.py`).
- `src/semantic_tensor_analysis/streamlit/`: Streamlit helpers (`utils.py`, `plots.py`) used across tabs.
- `src/semantic_tensor_analysis/analytics/`: Tensor batching, dimensionality, trajectories, and concept analytics.
- `src/semantic_tensor_analysis/visualization/`: Plotting backends (`viz/`, `tools/`, Streamlit-facing `plots.py`).
- `src/semantic_tensor_analysis/chat/`: LLM integration (`llama_cpp_analyzer.py`, `unified_analyzer.py`, insights in `analysis.py`, history parsing).
- `src/semantic_tensor_analysis/demos/`: CLI demos.
- `archive/legacy_embedders/`: Archived embedders kept for compatibility only.
- `data/`: Demo CSVs (`ultimate_demo_dataset.csv`, `demo_dataset.csv`, `finance_notes.csv`).
- `tests/`: Test suite.
- `pyproject.toml`: Package metadata/dependencies.

---

## 📘 Examples

- `examples/finance_narrative.ipynb`: Embed narrative CSV → run concept clustering → inspect clusters.

Open in Jupyter/VS Code and run locally; both use the STA API (no Streamlit).

---

## Capabilities

### Temporal semantic analysis

- Mask-aware batching for variable-length sessions (`pad_and_stack`, `masked_session_means`)
- Token-level drift with Hungarian alignment and token importance drift
- Trajectories with velocity/acceleration to spot rapid semantic shifts
- PCA + clustering over ordered sessions for broad patterns and transitions

### Visualizations

- PCA timelines and 3D trajectories
- Similarity and token-alignment heatmaps
- Concept evolution and transition graphs
- Ridgeline/distribution views
- Trajectory tunnel (experimental) for long-run drift

### LLM-assisted insights

- Token + sentence embeddings kept together for downstream prompts
- Domain-adaptive summaries (learning, research, projects, conversations)
- Axis interpretation for PCA dimensions

### Workflows

- Learning/journey mapping
- Research note evolution
- Conversation/topic drift
- Draft/version comparison

### Practicalities

- CSV/JSON/TXT ingestion
- Persistent storage (CPU-portable)
- Session state management in Streamlit
- Test suite coverage across embedding, storage, and viz
- CLI demo for fast iteration
- Sidebar controls let you set the rolling embedding context window (for SBERT prefixes) and a max token cap for Hungarian token drift (set to 0 to disable truncation).

---

## 📦 Datasets

- `ultimate_demo_dataset.csv`: High-quality demo with clear trajectories and richer, longer texts.
- `demo_dataset.csv`: General-purpose narrative journey dataset.
- `finance_notes.csv`, `finance_notes_expanded.csv`: Finance-themed notes (expanded version is a bigger stress test).

Upload any via the Streamlit sidebar to explore the full suite of analyses.

Expected columns (typical): `session_id`, `date`, `title` (optional), `text`.

---

## 🤖 LLM Backend Setup

STA uses **llama.cpp** as the default backend (sidebar auto-configured to `http://localhost:8080`, model `local`). Ollama UI is deprecated.

**Advantages**: Faster inference, lower memory footprint, vision support with the right GGUF.

1. **Install llama-cpp-python:**
   ```bash
   pip install llama-cpp-python
   ```

2. **Download a GGUF model:**
   - Vision (Apple M4/16GB): `Qwen/Qwen3-VL-4B-Instruct-GGUF` (e.g., Q4_0 or Q4_K_M).
   - Text-only: 4–8B Q4/Q5 GGUFs (Mistral-7B, Llama-3-8B, Qwen2-7B, Phi-3-Mini) work well.

3. **Run `llama-server`:**
   ```bash
   ./server -m /path/to/model.gguf -c 4096 --host 0.0.0.0 --port 8080
   ```

4. **In the app:**
   - Sidebar auto-uses llama.cpp at `http://localhost:8080` with model `local`.
   - Vision snapshot button will leverage a vision-capable GGUF if provided.
   - For vision models (e.g., Qwen3-VL), start llama-server with both model and projector, e.g.:
     ```bash
     llama-server \
       -m /path/to/Qwen3VL-8B-Instruct-Q4_K_M.gguf \
       --mmproj /path/to/mmproj-Qwen3-VL-8B-Instruct-Q8_0.gguf \
       --port 8080 --ctx-size 5000
     ```

### No LLM (Optional)

You can use STA without any LLM backend. The core analysis and visualizations work independently. You'll just miss the AI-generated narrative insights.

---

## 💡 Extensions & Ideas

- Drift alerts
- Sentence search
- HTML dashboard
- Enhanced multimodal support
- Product/ops applications

---

## ⚠️ Notes

- The `venv/` directory is excluded from git and should not be committed.
- **LLM Integration**: STA supports two LLM backends:
  - **llama.cpp** (recommended): Use local GGUF models for faster, memory-efficient inference
  - **Ollama**: Traditional Ollama server with model management
- The Streamlit app renders Matplotlib figures inline; no external windows will block interaction.
- Key dependencies: `torch`, `transformers`, `scikit-learn`, `plotly`, `streamlit`, `pandas`, `numpy`, `rich`, `requests`, `llama-cpp-python`.
- **tkinter** (for file browser): Usually pre-installed with Python. On Linux, install with `sudo apt-get install python3-tk` if needed.
- **Storage:** Session files are stored under `data/universal/`. Check sidebar storage stats and use the cleanup expander to prune old sessions; CLI available via `python -m semantic_tensor_analysis.storage.manager --stats` and cleanup options.
- Python 3.14+: TypedDict `closed=` quirks are patched at import so Altair and other typing consumers continue to load.

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
| Applications        | Yes            | Yes               | Project, finance, and general datasets provided.          |
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
| **Domain-specific workflows** | ❌ | ❌ | ✅ Projects, learning, research |

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
# → ["deadline": high drift, "refactor": low drift]  (which concepts changed)
```

**STA provides the calculus of semantic change, not just static snapshots.**

### "Why session-based instead of continuous streaming?"

**Session-based is intentional** for certain domains:

- **Project updates**: Each sprint/meeting note is a natural boundary
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
