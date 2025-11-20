# Semantic Tensor Analysis - Improvement Plan

**Date Created:** 2025-01-19
**Project Status:** Production-ready with enhancement opportunities
**Codebase Size:** ~20,000 lines
**Current Version:** 0.1.0

---

## Executive Summary

Semantic Tensor Analysis is a mature, well-architected temporal semantic analysis framework. The codebase is functional and production-ready for text-based analysis. This plan outlines targeted improvements to enhance testing, performance, usability, and feature completeness.

### Audit Findings

**Strengths:**
- ✅ Solid core architecture (dual-resolution embeddings, ragged tensor handling)
- ✅ Comprehensive visualization suite (~40% of codebase)
- ✅ Good documentation (README, inline comments)
- ✅ Clean separation of concerns (memory, analytics, visualization, UI)

**Issues Identified:**
- ⚠️ Test suite has Streamlit import conflicts (2 test files fail to import)
- ⚠️ Limited test coverage (5 tests vs 20K+ LOC)
- ⚠️ 20GB storage footprint with 25K+ session files (cleanup needed)
- ⚠️ Duplicate embedder code (5 embedder files, 446 total lines)
- ⚠️ No llama.cpp integration (only Ollama currently supported)
- ⚠️ Incomplete multimodal support (vision/audio stubbed)

---

## Phase 1: Cleanup & Stabilization (WEEKS 1-2)

**Goal:** Fix critical issues, improve code quality, ensure project health

### 1.1 Fix Test Suite Import Errors (Priority: CRITICAL)

**Problem:**
```
tests/test_concept_analysis.py - Streamlit import fails
tests/test_universal_system.py - May have similar issues
```

**Root Cause:** Tests import from `semantic_tensor_analysis.__init__.py` which imports `app.main`, which imports Streamlit. Streamlit's initialization interferes with pytest.

**Solution:**
1. Refactor `__init__.py` to use lazy imports for app-related code
2. Create separate test fixtures that don't require Streamlit
3. Add pytest configuration to handle Streamlit gracefully
4. Mock Streamlit components in tests where necessary

**Implementation Steps:**
- [ ] Modify `src/semantic_tensor_analysis/__init__.py` to make app imports optional
- [ ] Update test imports to avoid triggering Streamlit initialization
- [ ] Add `conftest.py` fixtures for mocking Streamlit
- [ ] Verify all tests can be collected: `pytest --collect-only`
- [ ] Run full test suite: `pytest tests/ -v`

**Success Criteria:**
- All test files can be imported without errors
- At least 5 tests pass (current baseline)
- No Streamlit-related import errors

**Files to Modify:**
- `src/semantic_tensor_analysis/__init__.py` (already has lazy imports, extend them)
- `tests/conftest.py` (add Streamlit mocking)
- `tests/test_concept_analysis.py` (update imports)

---

### 1.2 Increase Test Coverage (Priority: HIGH)

**Current State:** 5 tests collected, ~20K LOC → 0.025% coverage

**Target:** Achieve 60% coverage for core modules within 2 weeks

**Test Priorities:**

**Core Memory System (CRITICAL):**
- [ ] `memory/universal_core.py`: Test UniversalEmbedding, EventDescriptor, UniversalMemoryStore
- [ ] `memory/text_embedder.py`: Test TextEmbedder, dual embeddings, token extraction
- [ ] `memory/drift.py`: Test drift_series calculation
- [ ] `memory/sequence_drift.py`: Test Hungarian alignment, token importance drift

**Analytics System (HIGH):**
- [ ] `analytics/tensor_batching.py`: Test pad_and_stack, masked_session_means, flatten_with_mask
- [ ] `analytics/trajectory.py`: Test velocity, acceleration, inflection point detection
- [ ] `analytics/dimensionality.py`: Test PCA pipeline
- [ ] `analytics/concept/concept_analysis.py`: Test KMeans clustering

**Integration Tests (MEDIUM):**
- [ ] End-to-end: CSV → embedding → analysis → storage
- [ ] Session persistence: Save and load from disk
- [ ] Large dataset handling: 1000+ sessions

**Test Implementation Plan:**
```bash
# Create new test files
tests/test_tensor_batching.py          # Ragged tensor operations
tests/test_trajectory.py               # Semantic dynamics
tests/test_storage_persistence.py      # Storage operations
tests/test_integration_e2e.py          # Full pipeline

# Enhance existing tests
tests/test_universal_system.py         # Add more coverage
tests/test_dual_embedding.py           # Add edge cases
```

**Success Criteria:**
- 30+ unit tests covering core functions
- 5+ integration tests covering full workflows
- pytest coverage report showing >60% for src/semantic_tensor_analysis/

---

### 1.3 Consolidate Duplicate Embedder Code (Priority: MEDIUM)

**Current State:**
```
embedder.py          (40 lines)   - Legacy BERT [CLS] token
embedder_sbert.py    (52 lines)   - Legacy S-BERT only
embedder_cls.py      (66 lines)   - Legacy BERT CLS variant
embedder_hybrid.py   (107 lines)  - Legacy hybrid approach
embedder_config.py   (181 lines)  - Configuration/factory
dual_embedder.py     (exists)     - Legacy dual system
text_embedder.py     (primary)    - Current production implementation
```

**Active Usage:**
- `text_embedder.py` - PRIMARY (used in app, demos, tests)
- `embedder.py` - SECONDARY (used in some visualizations and streamlit utils)
- Others - LEGACY (not actively imported)

**Consolidation Plan:**

**Step 1: Identify all imports**
```bash
# Already done - key findings:
# - text_embedder.py is primary (app.py, demos/, tests/)
# - embedder.py is used in streamlit/utils.py and visualization/
# - Others are not actively imported
```

**Step 2: Migration strategy**
- [ ] Update `streamlit/utils.py` to use `text_embedder.py` instead of `embedder.py`
- [ ] Update `visualization/` modules to use `text_embedder.py`
- [ ] Create compatibility shims in `embedder.py` that delegate to `text_embedder.py`
- [ ] Add deprecation warnings to legacy embedder files
- [ ] Move legacy files to `archive/legacy_embedders/`

**Step 3: Testing**
- [ ] Verify all imports resolve correctly
- [ ] Run full test suite
- [ ] Test Streamlit app functionality
- [ ] Test CLI demo

**Success Criteria:**
- Only `text_embedder.py` contains embedding logic
- `embedder.py` becomes a thin compatibility wrapper
- All other embedder files moved to archive
- No functionality broken

**Files to Modify:**
- `src/semantic_tensor_analysis/streamlit/utils.py`
- `src/semantic_tensor_analysis/visualization/viz/semantic_drift_river.py`
- `src/semantic_tensor_analysis/visualization/viz/holistic_semantic_analysis.py`
- `src/semantic_tensor_analysis/visualization/river.py`
- `src/semantic_tensor_analysis/visualization/holistic.py`

---

### 1.4 Storage Management & Cleanup (Priority: MEDIUM)

**Problem:**
- `data/universal/` directory: 20GB, 25,569 session files
- No automatic cleanup mechanism
- Can grow unbounded over time

**Solution: Storage Management Utilities**

**Create `src/semantic_tensor_analysis/storage/manager.py`:**

```python
class StorageManager:
    """Manage session storage lifecycle."""

    def get_storage_stats() -> Dict[str, Any]:
        """Return storage size, file count, oldest/newest sessions."""

    def cleanup_old_sessions(days: int = 30) -> int:
        """Remove sessions older than N days."""

    def archive_sessions(session_ids: List[str], archive_path: Path) -> None:
        """Move sessions to archive directory."""

    def optimize_storage() -> None:
        """Compress old sessions, remove duplicates."""

    def export_sessions(session_ids: List[str], format: str) -> Path:
        """Export sessions to CSV/JSON for backup."""
```

**Implementation Tasks:**
- [ ] Create `src/semantic_tensor_analysis/storage/` package
- [ ] Implement `StorageManager` class
- [ ] Add CLI command: `python -m semantic_tensor_analysis.storage --cleanup --days=30`
- [ ] Add Streamlit UI in sidebar: "Storage Management" expander
- [ ] Add tests for storage operations

**Features:**
- Display storage stats (size, count, date range)
- One-click cleanup of old sessions
- Export/archive functionality
- Compression for old sessions

**Success Criteria:**
- Storage size visible in Streamlit UI
- Cleanup reduces storage by expected amount
- No data loss during cleanup
- Archived sessions can be restored

---

### 1.5 Documentation Improvements (Priority: MEDIUM)

**Current State:**
- ✅ Excellent README (446 lines, comprehensive)
- ✅ Good inline comments
- ✅ LaTeX paper in docs/
- ⚠️ Some functions lack docstrings
- ⚠️ No API reference documentation
- ⚠️ No tutorial notebooks

**Documentation Tasks:**

**Add Missing Docstrings:**
- [ ] Audit all public functions for docstrings
- [ ] Add Google-style docstrings to functions missing them
- [ ] Document all function parameters and return types
- [ ] Add usage examples to complex functions

**Create API Reference:**
- [ ] Use Sphinx or mkdocs to generate API docs
- [ ] Organize by module (memory, analytics, visualization)
- [ ] Include code examples
- [ ] Publish to GitHub Pages

**Create Tutorial Notebooks:**
- [ ] `tutorials/01_quick_start.ipynb` - Basic usage
- [ ] `tutorials/02_custom_analysis.ipynb` - Extending STA
- [ ] `tutorials/03_clinical_workflow.ipynb` - ABA therapy example
- [ ] `tutorials/04_advanced_visualization.ipynb` - Custom viz

**Success Criteria:**
- 100% of public functions have docstrings
- API reference published online
- 4+ tutorial notebooks in repository
- Contribution guide for developers

---

## Phase 2: Feature Enhancements (WEEKS 3-4)

**Goal:** Add requested features, improve performance, enhance usability

### 2.1 llama.cpp Integration (Priority: CRITICAL - REQUESTED)

**Current State:**
- Only Ollama supported for LLM analysis
- Ollama requires separate installation and model management

**Goal:**
- Add llama.cpp as alternative LLM backend
- Support local GGUF models
- Provide fallback chain: llama.cpp → Ollama → No LLM

**Implementation Plan:**

**Step 1: Add Dependencies**
```toml
# pyproject.toml
dependencies = [
    # ... existing dependencies ...
    "llama-cpp-python>=0.2.0",  # Add with proper version constraints
]
```

**Step 2: Create LlamaCpp Analyzer**

Create `src/semantic_tensor_analysis/chat/llama_cpp_analyzer.py`:

```python
from typing import Iterator, Optional
from pathlib import Path
import llama_cpp

class LlamaCppAnalyzer:
    """LLM analysis using llama.cpp for local GGUF models."""

    def __init__(self, model_path: str, n_ctx: int = 4096):
        """Initialize llama.cpp model."""
        self.llm = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=4,
        )

    def stream_response(self, prompt: str, max_tokens: int = 1024) -> Iterator[str]:
        """Stream response from llama.cpp model."""
        for output in self.llm(
            prompt,
            max_tokens=max_tokens,
            stream=True,
            temperature=0.7,
        ):
            yield output['choices'][0]['text']

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Generate complete response (non-streaming)."""
        response = self.llm(prompt, max_tokens=max_tokens, temperature=0.7)
        return response['choices'][0]['text']
```

**Step 3: Integrate with Existing Analysis**

Modify `src/semantic_tensor_analysis/chat/analysis.py`:

```python
from typing import Optional, Union
from .llama_cpp_analyzer import LlamaCppAnalyzer

class UnifiedLLMAnalyzer:
    """Unified interface for LLM analysis with multiple backends."""

    def __init__(self,
                 backend: str = "auto",  # "llama_cpp", "ollama", "auto"
                 llama_cpp_model_path: Optional[str] = None,
                 ollama_model: str = "qwen3:latest"):
        """Initialize LLM with specified backend."""
        self.backend = self._select_backend(backend, llama_cpp_model_path)

    def _select_backend(self, backend: str, llama_path: Optional[str]) -> str:
        """Auto-select best available backend."""
        if backend == "auto":
            # Try llama.cpp first if model path provided
            if llama_path and self._llama_cpp_available():
                return "llama_cpp"
            # Fall back to Ollama
            elif self._ollama_available():
                return "ollama"
            # No LLM available
            else:
                return "none"
        return backend

    def stream_response(self, prompt: str) -> Iterator[str]:
        """Stream response using configured backend."""
        if self.backend == "llama_cpp":
            yield from self.llama_cpp_analyzer.stream_response(prompt)
        elif self.backend == "ollama":
            yield from self.ollama_stream_response(prompt)
        else:
            yield "LLM analysis not available. Install llama.cpp or Ollama."
```

**Step 4: Update Streamlit UI**

Add model selection in sidebar:

```python
# app.py or app/sidebar.py
with st.sidebar:
    st.subheader("LLM Configuration")

    llm_backend = st.selectbox(
        "LLM Backend",
        ["Auto", "llama.cpp", "Ollama", "None"],
        help="Select LLM backend for analysis"
    )

    if llm_backend == "llama.cpp":
        model_path = st.text_input(
            "Model Path",
            placeholder="/path/to/model.gguf",
            help="Path to GGUF model file"
        )
```

**Step 5: Documentation**

- [ ] Update README with llama.cpp installation instructions
- [ ] Add model download guide (Hugging Face → GGUF)
- [ ] Document configuration options
- [ ] Add troubleshooting section

**Implementation Tasks:**
- [ ] Add `llama-cpp-python` to pyproject.toml
- [ ] Create `chat/llama_cpp_analyzer.py`
- [ ] Create `chat/unified_analyzer.py` (abstraction layer)
- [ ] Update `chat/analysis.py` to use unified analyzer
- [ ] Add Streamlit UI for model selection
- [ ] Test with popular GGUF models (Mistral, Llama, Qwen)
- [ ] Add integration tests
- [ ] Update documentation

**Success Criteria:**
- llama.cpp models can be loaded and queried
- Streaming responses work correctly
- Fallback to Ollama works if llama.cpp unavailable
- UI allows easy model selection
- Performance is acceptable (similar to Ollama)

**Recommended Test Models:**
- Mistral-7B-Instruct GGUF (~4GB)
- Llama-3-8B-Instruct GGUF (~4.5GB)
- Qwen2-7B-Instruct GGUF (~4GB)

---

### 2.2 Performance Optimization (Priority: HIGH)

**Current Bottlenecks:**
1. PCA computation on large datasets (1000+ sessions)
2. Token alignment for long sessions (500+ tokens)
3. Initial embedding computation (first load)
4. Visualization rendering for large heatmaps

**Optimization Tasks:**

**PCA Optimization:**
- [ ] Implement incremental PCA for large datasets
- [ ] Add caching for PCA results
- [ ] Use approximate PCA (randomized SVD) for >500 sessions
- [ ] Add progress indicators for long operations

**Embedding Optimization:**
- [ ] Batch process embeddings (10 sessions at a time)
- [ ] Add embedding cache to avoid re-computing
- [ ] Use fp16 for BERT inference (2x speed, minimal quality loss)
- [ ] Parallelize session processing where possible

**Visualization Optimization:**
- [ ] Downsample heatmaps for >100 sessions
- [ ] Use WebGL for large Plotly visualizations
- [ ] Lazy-load visualizations (render on tab switch)
- [ ] Add "Quick View" mode with reduced quality

**Implementation:**
```python
# analytics/dimensionality.py
from sklearn.decomposition import IncrementalPCA

def optimized_pca_pipeline(sessions, n_components=3):
    """PCA with automatic optimization for large datasets."""
    if len(sessions) > 500:
        # Use incremental PCA for large datasets
        return incremental_pca(sessions, n_components)
    else:
        # Use standard PCA for small datasets
        return standard_pca(sessions, n_components)
```

**Success Criteria:**
- PCA on 1000 sessions completes in <10 seconds
- Embedding 100 sessions completes in <30 seconds
- Heatmaps render smoothly for 200+ sessions
- Progress indicators show during long operations

---

### 2.3 Export & Reporting (Priority: MEDIUM)

**Feature:** Export analysis results to various formats

**Export Formats:**
1. **PDF Report** - Publication-ready analysis report
2. **HTML Dashboard** - Interactive standalone HTML
3. **CSV Export** - Raw data for external analysis
4. **JSON Export** - Machine-readable results

**Implementation:**

Create `src/semantic_tensor_analysis/export/reporter.py`:

```python
class AnalysisReporter:
    """Generate analysis reports in multiple formats."""

    def export_pdf(self, analysis_results: Dict, output_path: Path) -> None:
        """Generate PDF report with visualizations."""

    def export_html(self, analysis_results: Dict, output_path: Path) -> None:
        """Generate interactive HTML dashboard."""

    def export_csv(self, sessions: List, output_path: Path) -> None:
        """Export session data and metrics to CSV."""

    def export_json(self, analysis_results: Dict, output_path: Path) -> None:
        """Export complete analysis as JSON."""
```

**Streamlit Integration:**
```python
# Add to sidebar
with st.sidebar:
    st.subheader("Export Analysis")

    export_format = st.selectbox("Format", ["PDF", "HTML", "CSV", "JSON"])

    if st.button("Export"):
        file_data = generate_export(st.session_state.analysis, export_format)
        st.download_button("Download", file_data, file_name=f"analysis.{export_format.lower()}")
```

**Success Criteria:**
- All 4 export formats implemented
- Exports include all visualizations
- PDF is publication-ready
- HTML is fully interactive

---

## Phase 3: Advanced Features (WEEKS 5-6)

**Goal:** Complete multimodal support, add advanced features

### 3.1 Complete Vision Modality (Priority: MEDIUM)

**Current State:** Vision embedder is stubbed but not implemented

**Implementation:**

```python
# memory/vision_embedder.py
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

class VisionEmbedder(ModalityEmbedder):
    """Vision modality using CLIP."""

    def __init__(self):
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def process_raw_data(self, image_path: str) -> UniversalEmbedding:
        """Process image to UniversalEmbedding."""
        image = Image.open(image_path)

        # Extract visual features
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.get_image_features(**inputs)

        # Create UniversalEmbedding
        return UniversalEmbedding(
            event_embeddings=outputs,  # [1, 512]
            sequence_embedding=outputs.mean(dim=0),
            modality=Modality.VISION,
            events=[EventDescriptor(event_type="image", ...)],
            ...
        )
```

**Tasks:**
- [ ] Implement full VisionEmbedder with CLIP
- [ ] Add support for image sequences (videos)
- [ ] Create vision-specific visualizations
- [ ] Add demo dataset with images
- [ ] Write tests

---

### 3.2 Drift Alerts & Monitoring (Priority: LOW)

**Feature:** Automatic alerts when semantic drift exceeds thresholds

**Implementation:**

```python
class DriftMonitor:
    """Monitor semantic drift and trigger alerts."""

    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds
        self.alerts = []

    def check_drift(self, sessions: List[UniversalEmbedding]) -> List[Alert]:
        """Check drift and generate alerts."""
        drift_score = calculate_drift(sessions[-2], sessions[-1])

        if drift_score > self.thresholds['critical']:
            return Alert(level="CRITICAL", message="Severe drift detected")
```

**Integration:**
- Email notifications
- Webhook support
- Streamlit notifications
- Slack integration

---

## Phase 4: Polish & Production (WEEK 7)

### 4.1 Docker Containerization

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install -e .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

**Tasks:**
- [ ] Create Dockerfile
- [ ] Create docker-compose.yml
- [ ] Add .dockerignore
- [ ] Test container build
- [ ] Publish to Docker Hub

---

### 4.2 CI/CD Pipeline

Create `.github/workflows/test.yml`:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -e .[dev]
      - name: Run tests
        run: pytest tests/ -v --cov=src/
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

**Tasks:**
- [ ] Create GitHub Actions workflow
- [ ] Add pre-commit hooks
- [ ] Set up code coverage reporting
- [ ] Add linting (flake8, black, mypy)

---

## Success Metrics

### Phase 1 (Weeks 1-2)
- ✅ All tests pass (5+ → 30+ tests)
- ✅ Test coverage >60% for core modules
- ✅ Zero duplicate embedder code
- ✅ Storage management utilities working
- ✅ API documentation published

### Phase 2 (Weeks 3-4)
- ✅ llama.cpp integration complete and tested
- ✅ PCA performance improved by 2-3x
- ✅ Export functionality (4 formats) working
- ✅ Performance benchmarks documented

### Phase 3 (Weeks 5-6)
- ✅ Vision modality functional
- ✅ Drift monitoring implemented
- ✅ Advanced features documented

### Phase 4 (Week 7)
- ✅ Docker container published
- ✅ CI/CD pipeline running
- ✅ All documentation complete
- ✅ Project ready for v0.2.0 release

---

## Timeline Summary

```
Week 1-2: Cleanup & Stabilization
├─ Fix tests
├─ Increase coverage
├─ Consolidate embedders
├─ Storage management
└─ Documentation

Week 3-4: Feature Enhancements
├─ llama.cpp integration  ⭐ PRIORITY
├─ Performance optimization
└─ Export & reporting

Week 5-6: Advanced Features
├─ Vision modality
├─ Drift monitoring
└─ Advanced visualizations

Week 7: Polish & Production
├─ Docker
├─ CI/CD
└─ Release v0.2.0
```

---

## Resource Requirements

**Development Time:** ~140 hours (7 weeks × 20 hours/week)

**Infrastructure:**
- GitHub Actions (free tier sufficient)
- Docker Hub (free tier)
- GitHub Pages for docs (free)

**Models/Data:**
- llama.cpp GGUF models (~4-8GB)
- CLIP model for vision (~1GB)
- Test datasets (provided)

---

## Risk Assessment

### Low Risk
- Test suite fixes (well-understood problem)
- Code consolidation (mechanical refactoring)
- Documentation improvements (straightforward)

### Medium Risk
- llama.cpp integration (new dependency, compatibility concerns)
- Performance optimization (may require architectural changes)
- Vision modality (new domain, requires validation)

### High Risk
- Storage cleanup (potential data loss if buggy)
- Breaking changes to API (backward compatibility)

**Mitigation Strategies:**
- Comprehensive testing before merging
- Feature flags for experimental features
- Backup mechanisms for storage operations
- Versioned API with deprecation warnings

---

## Conclusion

This improvement plan transforms Semantic Tensor Analysis from a production-ready research tool into a polished, enterprise-grade framework. The phased approach ensures stability while adding valuable features.

**Immediate Next Steps:**
1. ✅ Fix test suite import errors
2. ✅ Begin llama.cpp integration
3. ✅ Set up CI/CD pipeline

**Long-term Vision:**
- Industry-standard semantic analysis framework
- Multi-modal support (text, vision, audio)
- Real-time drift monitoring
- Cloud deployment ready
- Active community contributions

---

**Last Updated:** 2025-01-19
**Next Review:** 2025-02-02 (after Phase 1 completion)
