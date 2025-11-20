"""
Vision modality placeholder (CLIP removed).

CLIP-based vision support was removed to keep the footprint lean and
avoid optional GPU-heavy deps. This module remains to satisfy imports
but raises clear errors when invoked.
"""

from __future__ import annotations

from typing import Any, List

from .universal_core import ModalityEmbedder, Modality, EventDescriptor, UniversalEmbedding

CLIP_AVAILABLE = False


class VisionEmbedder(ModalityEmbedder):
    """Disabled vision embedder placeholder."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise ImportError(
            "Vision modality is currently disabled because CLIP was removed. "
            "Re-enable by adding a CLIP-backed implementation."
        )

    @property
    def modality(self) -> Modality:
        return Modality.VISION

    @property
    def embedding_dimension(self) -> int:
        return 0

    def extract_events(self, raw_data: Any, **kwargs: Any) -> List[EventDescriptor]:
        raise ImportError(
            "Vision modality is disabled. No event extraction available."
        )

    def embed_events(self, events: List[EventDescriptor], **kwargs: Any) -> UniversalEmbedding:
        raise ImportError(
            "Vision modality is disabled. No embedding available."
        )


def create_vision_embedding(image: Any) -> UniversalEmbedding:
    """Legacy helper; now disabled."""
    raise ImportError("Vision modality is disabled. No vision embedding available.")


def get_visual_events(image: Any, threshold: float = 0.15) -> List[EventDescriptor]:
    """Legacy helper; now disabled."""
    raise ImportError("Vision modality is disabled. No vision event extraction available.")
