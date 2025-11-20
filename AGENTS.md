# AGENTS.md - AI Assistant Project Guide

**Last Updated:** 2025-01-19
**Project:** Semantic Tensor Analysis (STA)
**Status:** Production-ready (with enhancement opportunities)

---

## 🎯 Project Mission

**Semantic Tensor Analysis** is a temporal semantic evolution analysis framework that tracks how meaning changes across time, tokens, and context. It provides researchers, clinicians, and analysts with powerful tools to understand semantic trajectories through sessions of text.

### Core Innovation: Dual-Resolution Temporal Analysis
- **Token-level granularity** (BERT) - tracks individual concept drift
- **Sequence-level semantics** (Sentence-BERT) - tracks holistic meaning evolution

---

## 📁 Project Structure Overview

```
semantic_tensor_analysis/
├── app.py                          # Main Streamlit entry point (3,700+ lines)
├── pyproject.toml                  # Package metadata and dependencies
├── README.md                       # Comprehensive user documentation
├── AGENTS.md                       # This file - AI assistant guide
│
├── src/semantic_tensor_analysis/   # Main source (~20K lines)
│   ├── memory/                     # Core memory & embedding system
│   │   ├── universal_core.py       # Universal multimodal architecture
│   │   ├── text_embedder.py        # Text modality (BERT + S-BERT)
│   │   ├── vision_embedder.py      # Vision modality (stub)
│   │   ├── drift.py                # Session-level drift analysis
│   │   └── sequence_drift.py       # Token-level drift (Hungarian)
│   │
│   ├── analytics/                  # Analysis & computation
│   │   ├── tensor_batching.py      # Ragged tensor operations
│   │   ├── trajectory.py           # Semantic velocity/acceleration
│   │   ├── dimensionality.py       # PCA and reduction
│   │   └── concept/                # Concept analysis & clustering
│   │
│   ├── visualization/              # Extensive viz toolkit (~40% codebase)
│   │   ├── viz/                    # Core visualizations
│   │   ├── tools/                  # Visualization tools
│   │   └── plots.py                # Streamlit-specific plotting
│   │
│   ├── app/                        # Modular application structure
│   │   ├── main.py                 # App entry & wiring
│   │   ├── tabs/                   # Individual tab implementations
│   │   └── services/               # Service layer
│   │
│   ├── chat/                       # LLM analysis (Ollama)
│   │   ├── analysis.py             # Domain-aware insights
│   │   └── history_analyzer.py     # Chat history parsing
│   │
│   ├── explainability/             # Explainability engine
│   └── demos/                      # CLI demos & utilities
│
├── tests/                          # Test suite (needs attention)
├── data/                           # Datasets & storage
│   ├── universal/                  # Universal memory storage (~20GB)
│   └── *.csv                       # Demo datasets
│
├── docs/                           # Documentation
│   └── semantic-tensor-memory.tex  # LaTeX paper
│
└── archive/                        # Historical documentation
```

---

## 🧠 Core Concepts & Architecture

### 1. Universal Embedding System

```python
# Dual-resolution embedding
UniversalEmbedding:
    event_embeddings: [tokens, 768]  # BERT - token-level
    sequence_embedding: [768]        # S-BERT - session-level
    modality: Modality               # TEXT, VISION, AUDIO, etc.
    events: List[EventDescriptor]    # Structured event data
```

**Key Innovation:** Sessions can have variable lengths (100-500 tokens), handled via ragged tensor operations with masking.

### 2. Ragged Tensor Operations

```python
# Variable-length sessions
session_1: [100 tokens, 768 dims]
session_2: [237 tokens, 768 dims]
session_3: [89 tokens, 768 dims]

# Batching with masking
batch, mask = pad_and_stack([s1, s2, s3])
# batch: [3, 237, 768] (padded to max)
# mask: [3, 237] (True where valid)

# Session-level analysis (masked)
session_means = masked_session_means(batch, mask)
# [3, 768] - padding ignored

# Token-level analysis (flattened)
flat_tokens, session_ids = flatten_with_mask(batch, mask)
# [426, 768] (100+237+89 valid tokens)
```

### 3. Analysis Pipeline

```
Raw Data (CSV/JSON/TXT)
    ↓
Text Embedding (dual-resolution)
    ├→ Token embeddings [n_tokens, 768] via BERT
    └→ Sequence embedding [768] via S-BERT
    ↓
Session Creation (UniversalEmbedding)
    ↓
Storage (UniversalMemoryStore)
    ↓
Ragged Tensor Batching (pad_and_stack)
    ↓
Global Analysis
    ├→ PCA across sessions/tokens
    ├→ Concept clustering (KMeans)
    ├→ Token alignment (Hungarian)
    └→ Drift computation (cosine distance)
    ↓
Visualization
    ├→ Temporal trajectories
    ├→ Heatmaps
    ├→ 3D semantic space
    └→ Concept evolution
    ↓
Optional: LLM narrative (Ollama)
```

---

## 🔍 Current State Assessment

### ✅ What Works Well

1. **Core Functionality**
   - ✅ Dual-resolution embeddings (BERT + S-BERT)
   - ✅ Universal multimodal architecture
   - ✅ Ragged tensor batching with masking
   - ✅ Token-level drift (Hungarian alignment)
   - ✅ Session-level drift analysis
   - ✅ Semantic trajectory (velocity, acceleration)
   - ✅ PCA and dimensionality reduction
   - ✅ Concept clustering and evolution

2. **Visualization**
   - ✅ Session similarity heatmaps
   - ✅ Token alignment heatmaps
   - ✅ PCA scatter plots and trajectories
   - ✅ 3D semantic drift visualization
   - ✅ Streamlit integration (7 tabs)
   - ✅ Interactive visualizations (Plotly, Altair)

3. **Data Handling**
   - ✅ CSV import (multiple formats)
   - ✅ JSON conversation parsing
   - ✅ TXT conversation detection
   - ✅ Persistent storage (CPU-portable)
   - ✅ Adaptive data processing

4. **LLM Integration**
   - ✅ Ollama integration
   - ✅ Domain inference (ABA, clinical, learning, research)
   - ✅ Time-scale detection
   - ✅ Content-focused insights

### ⚠️ Known Issues

1. **Testing**
   - ⚠️ Test suite has import errors (Streamlit conflicts)
   - ⚠️ Limited test coverage (5 tests collected vs 20K+ LOC)
   - ⚠️ No CI/CD pipeline

2. **Storage**
   - ⚠️ `data/universal/` is 20GB with 25K+ session files
   - ⚠️ No automatic cleanup utilities
   - ⚠️ `memory_tensor.pt` and `memory_meta.json` in project root (should be .gitignored)

3. **Documentation**
   - ⚠️ Some functions lack comprehensive docstrings
   - ⚠️ No formal API reference docs
   - ⚠️ Missing tutorial notebooks

4. **Code Quality**
   - ⚠️ Code duplication in embedding modules (embedder.py, dual_embedder.py, embedder_cls.py, etc.)
   - ⚠️ Legacy code that may need consolidation
   - ⚠️ Incomplete multimodal support (vision/audio stubbed)

5. **Dependencies**
   - ⚠️ Heavy dependencies (~840MB models on first run)
   - ⚠️ No llama.cpp integration (only Ollama currently)
   - ⚠️ No offline mode documented

---

## 🎯 Enhancement Roadmap

### Phase 1: Cleanup & Stabilization (HIGH PRIORITY)

1. **Fix Test Suite**
   - Resolve Streamlit import conflicts
   - Increase test coverage to >60%
   - Add integration tests for full workflows
   - Set up basic CI/CD

2. **Storage Management**
   - Implement session cleanup utilities
   - Add storage size monitoring
   - Document storage requirements
   - Consider compression strategies

3. **Code Consolidation**
   - Remove duplicate embedding code
   - Consolidate legacy modules
   - Add .gitignore entries for memory files
   - Type hints throughout

4. **Documentation**
   - Add missing docstrings
   - Create API reference
   - Add architecture diagrams
   - Create troubleshooting guide

### Phase 2: Feature Enhancements (MEDIUM PRIORITY)

5. **llama.cpp Integration** ⭐ REQUESTED
   - Add `llama-cpp-python` dependency
   - Create LlamaCppAnalyzer class
   - Integration with existing chat/analysis.py
   - Support for local GGUF models
   - Fallback to Ollama if unavailable

6. **Performance Optimization**
   - Profile memory usage on large datasets
   - Optimize PCA for large session counts
   - Add progress indicators
   - Batch processing for very large datasets

7. **Export & Reporting**
   - Export analysis results (PDF, HTML)
   - Drift alerts/notifications
   - Sentence-level search
   - Multi-dataset comparison

### Phase 3: Advanced Features (LOW PRIORITY)

8. **Multimodal Support**
   - Complete vision embedder
   - Add audio modality
   - Cross-modal analysis examples

9. **Deployment**
   - Docker containerization
   - Cloud deployment guides
   - Standalone executables
   - REST API server mode

---

## 🛠️ Working with This Codebase

### Key Files to Understand

1. **Entry Points**
   - `app.py` - Main Streamlit application (start here for UI)
   - `src/semantic_tensor_analysis/demos/demo.py` - CLI demo
   - `src/semantic_tensor_analysis/__init__.py` - Public API

2. **Core Architecture**
   - `memory/universal_core.py` - Universal embedding system
   - `memory/text_embedder.py` - Text modality implementation
   - `analytics/tensor_batching.py` - Ragged tensor handling
   - `analytics/trajectory.py` - Semantic dynamics

3. **Analysis & Visualization**
   - `streamlit/utils.py` - PCA pipeline, data loading
   - `visualization/viz/` - Core visualization functions
   - `chat/analysis.py` - LLM-powered insights

### Common Tasks

#### Adding a New Analysis Feature

1. Implement core logic in `src/semantic_tensor_analysis/analytics/`
2. Add visualization in `src/semantic_tensor_analysis/visualization/viz/`
3. Create Streamlit wrapper in `src/semantic_tensor_analysis/streamlit/plots.py`
4. Add tab in `app.py` or extend existing tab
5. Write tests in `tests/`
6. Update README with example

#### Adding a New Modality

1. Create embedder in `memory/` (inherit from `ModalityEmbedder`)
2. Implement `extract_events()`, `embed_events()`, `embed_sequence()`
3. Register in `create_universal_embedder()` factory
4. Add tests in `tests/test_universal_system.py`
5. Create demo dataset
6. Document in README

#### Debugging Test Failures

Common issues:
- **Streamlit import conflicts**: Tests try to import Streamlit modules
- **CUDA/device issues**: Ensure `.cpu()` calls in storage
- **Ragged tensor shape mismatches**: Check mask handling
- **Missing models**: First run downloads BERT/S-BERT models

### Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e .

# Run tests
pytest tests/ -v

# Run Streamlit app
streamlit run app.py

# Run CLI demo
python -m semantic_tensor_analysis.demos.demo
```

### Code Style Guidelines

1. **Docstrings**: Use Google-style docstrings
2. **Type Hints**: Add type hints to all function signatures
3. **Naming**:
   - Classes: `PascalCase`
   - Functions/variables: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
4. **Imports**: Group by stdlib, third-party, local
5. **Line Length**: Max 100 characters
6. **Error Handling**: Use specific exceptions, provide context

---

## 🧪 Testing Strategy

### Current Test Coverage

```
tests/
├── conftest.py                     # Fixtures (600+ lines)
├── test_universal_system.py        # Universal STM tests
├── test_dual_embedding.py          # Dual embedding tests
├── test_concept_analysis.py        # Concept analysis tests
├── test_semantic_upgrade.py        # Semantic upgrade tests
└── test_embedding_comparison.py    # Embedding comparison
```

**Current Issues:**
- Only 5 tests collected
- 2 import errors (Streamlit conflicts)
- Limited coverage relative to codebase size

### Recommended Test Additions

1. **Unit Tests**
   - Ragged tensor operations
   - Drift computation
   - Token alignment
   - PCA pipeline

2. **Integration Tests**
   - End-to-end data flow (CSV → analysis → viz)
   - Multi-session analysis
   - Storage persistence

3. **Performance Tests**
   - Large dataset handling (1000+ sessions)
   - Memory profiling
   - Computation benchmarks

---

## 🔌 Integration Points

### Current Integrations

1. **Ollama** (`chat/analysis.py`)
   - Local LLM inference
   - Domain-aware prompts
   - Streaming responses

2. **Streamlit** (`app.py`, `streamlit/`)
   - Interactive web UI
   - Session state management
   - File upload handling

3. **scikit-learn** (throughout)
   - PCA, KMeans, cosine similarity
   - Standard ML operations

4. **Plotly/Altair** (`visualization/`)
   - Interactive visualizations
   - 3D plots, animations

### Planned Integrations

1. **llama.cpp** ⭐ NEXT
   - Local GGUF model support
   - Faster inference
   - Lower memory footprint

2. **LangSmith/W&B** (future)
   - Export analysis results
   - Integration with ML platforms

3. **REST API** (future)
   - Server mode for remote access
   - Programmatic analysis

---

## 🚨 Critical Considerations

### When Modifying Core Architecture

1. **Ragged Tensor Compatibility**
   - Always use `pad_and_stack()` for batching
   - Respect masks in all operations
   - Don't assume fixed session lengths

2. **Storage Portability**
   - Always save tensors to CPU: `.cpu()` before pickling
   - Use `map_location='cpu'` when loading
   - Test cross-platform (CPU/GPU)

3. **Memory Management**
   - Be mindful of session accumulation
   - Use `torch.no_grad()` for inference
   - Clear cache with `gc.collect()` when needed

4. **Backward Compatibility**
   - Legacy storage format support in `store.py`
   - Gradual deprecation, not breaking changes
   - Migration utilities for old data

### Security & Privacy

- **No data transmission**: All processing is local
- **No telemetry**: No analytics or tracking
- **Data ownership**: Users control all stored data
- **Sensitive data**: Be mindful in clinical/personal contexts

---

## 📚 Additional Resources

### Documentation

- `README.md` - User-facing documentation
- `docs/semantic-tensor-memory.tex` - Academic paper
- `archive/` - Historical design documents

### Demo Datasets

- `data/ultimate_demo_dataset.csv` - Rich demo (60 sessions)
- `data/aba_therapy_dataset.csv` - Clinical progress (80 sessions)
- `data/demo_dataset.csv` - General demo (184 sessions)

### External Resources

- BERT: `bert-base-uncased` (Hugging Face)
- S-BERT: `all-mpnet-base-v2` (Sentence Transformers)
- Ollama: Local LLM runtime (ollama.ai)

---

## 💡 Tips for AI Assistants

### Understanding User Intent

Common requests and how to handle them:

1. **"Add a new feature"**
   - Ask: Where in pipeline? (data ingestion, analysis, viz, UI)
   - Check: Does it fit existing architecture?
   - Plan: Break into subtasks (core logic → tests → UI → docs)

2. **"Fix a bug"**
   - Reproduce: Try to recreate the issue
   - Isolate: Which component? (memory, analytics, viz, app)
   - Test: Verify fix doesn't break related functionality

3. **"Improve performance"**
   - Profile: Where's the bottleneck?
   - Optimize: Target specific slow operations
   - Measure: Benchmark before/after

4. **"Add documentation"**
   - Audience: Users or developers?
   - Scope: API reference, tutorials, or architecture?
   - Format: Docstrings, README, or separate docs?

### Project-Specific Patterns

1. **Data Flow**: Always CSV/JSON → Embedder → UniversalEmbedding → Store → Analysis → Viz
2. **Error Handling**: Graceful degradation, clear error messages
3. **UI Philosophy**: Sidebar for controls, main area for visualizations
4. **Modularity**: Keep analysis logic separate from visualization separate from UI

### Useful Search Patterns

```bash
# Find all embedding-related code
rg "embed" --type py

# Find all visualization functions
rg "def.*plot|def.*visualize" src/semantic_tensor_analysis/visualization/

# Find all Streamlit tabs
ls src/semantic_tensor_analysis/app/tabs/

# Find test files
find tests/ -name "test_*.py"
```

---

## 🎯 Current Focus: llama.cpp Integration

### Objective
Add llama.cpp support as an alternative to Ollama for local LLM inference.

### Requirements
1. Add `llama-cpp-python` to dependencies
2. Create `LlamaCppAnalyzer` class in `chat/`
3. Support GGUF model loading
4. Integrate with existing `analysis.py` workflow
5. Provide fallback to Ollama if llama.cpp unavailable
6. Update documentation

### Implementation Strategy
1. Research llama-cpp-python API
2. Create wrapper class with same interface as Ollama
3. Add model path configuration
4. Test with common GGUF models
5. Update UI to allow model selection
6. Document usage in README

---

## 📝 Change Log

### 2025-01-19 - Initial AGENTS.md Creation
- Created comprehensive guide for AI assistants
- Documented current state and architecture
- Established enhancement roadmap
- Prioritized llama.cpp integration

---

## 🤝 Contributing Guidelines (for AI Assistants)

### Before Making Changes

1. **Understand the request**: Ask clarifying questions
2. **Review relevant code**: Read existing implementations
3. **Check tests**: Understand test coverage
4. **Plan the work**: Break into manageable tasks

### Making Changes

1. **Follow patterns**: Match existing code style
2. **Preserve compatibility**: Don't break existing features
3. **Add tests**: Test new functionality
4. **Update docs**: Keep documentation current

### After Changes

1. **Verify functionality**: Test changes thoroughly
2. **Check side effects**: Ensure no regressions
3. **Document work**: Update AGENTS.md if architecture changed
4. **Suggest next steps**: What should happen next?

---

**Remember:** This is a mature, well-architected project. Respect the existing patterns and architecture. When in doubt, ask the user before making significant changes.
