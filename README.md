# Semantic Tensor Memory

A structured memory system for tracking meaning across time, tokens, and context. This implementation provides a working prototype of the Semantic Tensor Memory (STM) system described in the paper.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the demo:
```bash
python demo.py
```

3. Type sentences to build semantic memory, or type 'plot' to visualize drift.

## Project Structure

- `memory/`: Core memory implementation
  - `embedder.py`: Token embedding using BERT
  - `store.py`: Tensor storage and management
  - `drift.py`: Semantic drift analysis utilities
- `viz/`: Visualization tools
  - `pca_plot.py`: 2D drift visualization
  - `heatmap.py`: Session similarity heatmaps
- `demo.py`: Interactive CLI demo

## Features

- Token-level semantic tracking
- Temporal drift visualization
- Session similarity analysis
- Persistent memory storage

## Extensions

See the paper for potential extensions including:
- Drift alerts
- Sentence search
- HTML dashboard
- LLM explanation layer
- Clinical applications 