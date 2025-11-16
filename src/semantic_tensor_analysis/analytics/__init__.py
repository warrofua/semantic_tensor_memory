"""Analytics utilities for Semantic Tensor Memory."""

__all__ = []

try:
    from .trajectory import (
        calculate_semantic_trajectory_data,
        create_3d_trajectory_plot,
        display_trajectory_analysis_table,
    )

    __all__ += [
        "calculate_semantic_trajectory_data",
        "create_3d_trajectory_plot",
        "display_trajectory_analysis_table",
    ]
except ModuleNotFoundError:
    calculate_semantic_trajectory_data = None
    create_3d_trajectory_plot = None
    display_trajectory_analysis_table = None

try:
    from .dimensionality import compare_dimensionality_methods, create_alternative_visualization

    __all__ += [
        "compare_dimensionality_methods",
        "create_alternative_visualization",
    ]
except ModuleNotFoundError:
    compare_dimensionality_methods = None
    create_alternative_visualization = None

try:
    from .tensor_batching import (
        pad_and_stack,
        masked_session_means,
        flatten_with_mask,
    )

    __all__ += ["pad_and_stack", "masked_session_means", "flatten_with_mask"]
except ModuleNotFoundError:
    pad_and_stack = None
    masked_session_means = None
    flatten_with_mask = None

from .concept.concept_analysis import (
    ConceptAnalyzer,
    analyze_existing_store_concepts,
    get_concept_similarity_matrix,
)

__all__ += [
    "ConceptAnalyzer",
    "analyze_existing_store_concepts",
    "get_concept_similarity_matrix",
]
