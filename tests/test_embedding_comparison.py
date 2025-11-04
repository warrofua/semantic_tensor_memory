import torch
import pytest


def test_analyze_embedding_quality_matches_stub_metrics(stub_embeddings):
    """analyze_embedding_quality should align with deterministic similarities."""
    texts = ["Therapy session notes", "Planning follow-up activities"]

    results = stub_embeddings.embedder_config.analyze_embedding_quality(texts)

    cls_result = results["bert_cls"]
    dual_result = results["dual"]

    expected_cls_similarity = stub_embeddings.sentence_similarity(*texts)
    token_a = stub_embeddings.token_embeddings(texts[0])
    token_b = stub_embeddings.token_embeddings(texts[1])
    expected_dual_similarity = torch.nn.functional.cosine_similarity(
        token_a.mean(0),
        token_b.mean(0),
        dim=0,
    ).item()

    assert cls_result["semantic_similarity"] == pytest.approx(expected_cls_similarity)
    assert cls_result["emb1_shape"] == (1, stub_embeddings.dim)
    assert cls_result["emb2_shape"] == (1, stub_embeddings.dim)

    assert dual_result["semantic_similarity"] == pytest.approx(expected_dual_similarity)
    assert dual_result["emb1_shape"] == token_a.shape
    assert dual_result["emb2_shape"] == token_b.shape
    assert dual_result["token_counts"] == [token_a.shape[0], token_b.shape[0]]
