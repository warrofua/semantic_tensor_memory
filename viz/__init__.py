"""Semantic Tensor Memory Visualization Package

This package provides visualization and analysis tools for the Semantic Tensor Memory system,
including PCA plots, heatmaps, and semantic analysis.

Modules:
    pca_plot: PCA visualization and drift mapping
    heatmap: Session similarity and token drift heatmaps
    pca_summary: Narrative and keyword analysis of PCA axes
    semantic_analysis: LLM-powered clinical and narrative summaries
"""

from .pca_plot import plot, plot_drift, check_tensor_health, prepare_for_pca, interpret_pca
from .heatmap import heatmap, token_heatmap
from .pca_summary import explain_pca_axes, generate_narrative_summary
from .semantic_analysis import generate_clinical_summary, print_clinical_analysis

__all__ = [
    'plot',
    'plot_drift',
    'check_tensor_health',
    'prepare_for_pca',
    'interpret_pca',
    'heatmap',
    'token_heatmap',
    'explain_pca_axes',
    'generate_narrative_summary',
    'generate_clinical_summary',
    'print_clinical_analysis'
]
