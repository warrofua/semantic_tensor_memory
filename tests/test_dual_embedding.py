import torch
import pytest

from semantic_tensor_analysis.memory.sequence_drift import (
    sequence_drift,
    semantic_coherence_score,
)


def test_create_dual_embedding_shapes(stub_embeddings):
    """create_dual_embedding should align with the deterministic stubs."""
    text = "Machine learning enables insight"
    embedding = stub_embeddings.dual_embedder.create_dual_embedding(text)

    expected_tokens = stub_embeddings.tokens(text)
    expected_matrix = stub_embeddings.token_embeddings(text)

    assert embedding.text == text
    assert embedding.token_count == len(expected_tokens)
    assert embedding.tokens == expected_tokens
    assert embedding.token_embeddings.shape == expected_matrix.shape
    assert torch.allclose(embedding.token_embeddings, expected_matrix)
    assert torch.allclose(
        embedding.sentence_embedding,
        stub_embeddings.sentence_vector(text),
    )


def test_multi_resolution_drift_matches_component_metrics(stub_embeddings):
    """DualMemoryStore's drift analysis should mirror direct metric calculations."""
    store = stub_embeddings.dual_embedder.DualMemoryStore()
    session_a = store.add_session("Calm focus during planning")
    session_b = store.add_session("Energetic planning after feedback")

    analysis = store.analyze_multi_resolution_drift(session_a, session_b)

    emb_a = stub_embeddings.create_dual_embedding("Calm focus during planning")
    emb_b = stub_embeddings.create_dual_embedding("Energetic planning after feedback")

    expected_drift = sequence_drift([emb_a.token_embeddings, emb_b.token_embeddings])[0]
    expected_coherence_a = semantic_coherence_score(emb_a.token_embeddings)
    expected_coherence_b = semantic_coherence_score(emb_b.token_embeddings)
    expected_sentence_similarity = torch.nn.functional.cosine_similarity(
        emb_a.sentence_embedding,
        emb_b.sentence_embedding,
        dim=0,
    ).item()
    expected_token_mean_similarity = torch.nn.functional.cosine_similarity(
        emb_a.token_embeddings.mean(0),
        emb_b.token_embeddings.mean(0),
        dim=0,
    ).item()
    expected_diff = abs(expected_sentence_similarity - expected_token_mean_similarity)

    if expected_diff < 0.1:
        expected_interpretation = "Consistent semantic and structural change"
    elif expected_sentence_similarity > expected_token_mean_similarity:
        expected_interpretation = "Semantic meaning preserved despite structural changes"
    elif expected_token_mean_similarity > expected_sentence_similarity:
        expected_interpretation = "Similar structure but semantic meaning shifted"
    else:
        expected_interpretation = "Complex semantic-structural divergence detected"

    assert analysis["token_level"]["sequence_drift"] == pytest.approx(expected_drift)
    assert analysis["token_level"]["coherence_a"] == pytest.approx(expected_coherence_a)
    assert analysis["token_level"]["coherence_b"] == pytest.approx(expected_coherence_b)
    assert analysis["token_level"]["coherence_change"] == pytest.approx(
        expected_coherence_b - expected_coherence_a
    )
    assert analysis["token_level"]["token_count_change"] == (
        emb_b.token_count - emb_a.token_count
    )
    assert analysis["sentence_level"]["semantic_similarity"] == pytest.approx(
        expected_sentence_similarity
    )
    assert analysis["sentence_level"]["semantic_drift"] == pytest.approx(
        1 - expected_sentence_similarity
    )
    assert analysis["cross_resolution"]["token_mean_similarity"] == pytest.approx(
        expected_token_mean_similarity
    )
    assert analysis["cross_resolution"]["semantic_vs_structural_ratio"] == pytest.approx(
        expected_diff
    )
    assert analysis["cross_resolution"]["interpretation"] == expected_interpretation
