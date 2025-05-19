# Semantic Tensor Memory

A structured, interpretable memory system for tracking meaning across time, tokens, and context. This implementation provides a working prototype of the Semantic Tensor Memory (STM) system described in the accompanying paper.

---

## 🚀 Quick Start

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2. **Run the interactive demo:**
    ```bash
    python demo.py
    ```

3. **Try these CLI commands:**
    - Type sentences to build semantic memory interactively.
    - Type `import` to import session notes from a CSV file (popup file picker).
    - Type `plot` to visualize semantic drift (PCA, heatmap, and LLM clinical summary).
    - Type `drift` to see drift metrics between sessions.
    - Type `tokens` to see token-level drift.
    - Type `exit` to quit.

---

## 🗂️ Project Structure

- `memory/`: Core memory implementation
  - `embedder.py`: Token embedding using BERT
  - `store.py`: Tensor storage, management, and save/load utilities
  - `drift.py`: Semantic drift analysis utilities
- `viz/`: Visualization and analysis tools
  - `pca_plot.py`: 2D drift visualization, PCA, and LLM clinical summary
  - `heatmap.py`: Session similarity heatmaps
  - `pca_summary.py`: Narrative and keyword PCA axis summaries
  - `semantic_analysis.py`: LLM-powered clinical/narrative analysis (Ollama integration)
- `demo.py`: Interactive CLI demo (with CSV import and file picker)
- `requirements.txt`: Python dependencies

---

## ✨ Features

- Token-level semantic tracking
- Temporal drift visualization (PCA, heatmap)
- Session similarity and drift metrics
- Persistent memory storage
- CSV import with file picker
- LLM-powered clinical and narrative summaries (Ollama integration)
- Rich CLI for interactive exploration

---

## 💡 Extensions & Ideas

- Drift alerts
- Sentence search
- HTML dashboard
- LLM explanation layer
- Clinical applications

---

## ⚠️ Notes

- The `venv/` directory is excluded from git and should **not** be committed.
- For LLM-powered summaries, ensure [Ollama](https://ollama.com/) is installed and running with a supported model (e.g., `qwen3:latest`).
- For best results, run the demo from a standard terminal (not inside VS Code or over SSH) to enable the file picker popup.

---

## 📄 Citation
If you use this codebase or ideas in your research, please cite the accompanying paper or link to this repository. 