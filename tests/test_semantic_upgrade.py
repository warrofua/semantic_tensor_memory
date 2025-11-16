import torch
import pytest

from semantic_tensor_analysis.memory.drift import drift_series


def test_embed_sentence_similarity_matches_expected(stub_embeddings):
    """The stubbed embedder should produce deterministic similarities."""
    embed_sentence = stub_embeddings.embedder.embed_sentence

    text_a = "I feel anxious about work"
    text_b = "Work makes me feel calm"

    emb_a = embed_sentence(text_a)
    emb_b = embed_sentence(text_b)

    similarity = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=1).item()
    expected = stub_embeddings.sentence_similarity(text_a, text_b)

    assert similarity == pytest.approx(expected)


def test_drift_series_uses_stubbed_vectors(stub_embeddings):
    """Drift calculations should align with the deterministic sentence vectors."""
    sentences = [
        "Overwhelmed by the workload",
        "Finding better balance",
        "Feeling confident again",
    ]
    tensors = [stub_embeddings.embedder.embed_sentence(text) for text in sentences]

    drifts, counts = drift_series(tensors)

    expected_drifts = [
        1 - stub_embeddings.sentence_similarity(sentences[i], sentences[i - 1])
        for i in range(1, len(sentences))
    ]

    assert counts == [1, 1, 1]
    assert drifts == pytest.approx(expected_drifts)
