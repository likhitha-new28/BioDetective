import numpy as np
import pandas as pd

import biodetective.analysis.expression_qc as expression_qc
import biodetective.analysis.similarity as similarity
from biodetective.analysis.pca import PCAConfig, run_pca


def test_similarity_prepares_metadata_once_for_many_qualifying_pairs(monkeypatch):
    sample_ids = [f"S{i}" for i in range(20)]
    matrix = pd.DataFrame(1.0, index=sample_ids, columns=sample_ids)
    metadata = pd.DataFrame(
        {"sample_id": sample_ids, "condition": ["A", "B"] * 10, "batch": ["B1"] * 10 + ["B2"] * 10}
    )
    original = similarity._prepare_metadata
    call_count = 0

    def counted_prepare(frame):
        nonlocal call_count
        call_count += 1
        return original(frame)

    monkeypatch.setattr(similarity, "_prepare_metadata", counted_prepare)
    findings = similarity.detect_high_similarity_pairs(matrix, metadata)

    assert len(findings) == 190
    assert call_count == 1


def test_expression_qc_calculates_gene_variance_once(monkeypatch):
    expression = pd.DataFrame(
        np.arange(200, dtype=float).reshape(20, 10),
        index=[f"G{i}" for i in range(20)],
        columns=[f"S{i}" for i in range(10)],
    )
    original = expression_qc.calculate_gene_variance
    call_count = 0

    def counted_variance(frame):
        nonlocal call_count
        call_count += 1
        return original(frame)

    monkeypatch.setattr(expression_qc, "calculate_gene_variance", counted_variance)
    expression_qc.run_expression_qc(expression)

    assert call_count == 1


def test_pca_precomputed_variances_preserve_results():
    rng = np.random.default_rng(123)
    expression = pd.DataFrame(
        rng.normal(size=(100, 20)),
        index=[f"G{i}" for i in range(100)],
        columns=[f"S{i}" for i in range(20)],
    )
    config = PCAConfig(top_variable_genes=50, n_components=3)
    variances = expression.var(axis=1, ddof=0)

    calculated = run_pca(expression, config)
    reused = run_pca(expression, config, gene_variances=variances)

    np.testing.assert_allclose(calculated.coordinates, reused.coordinates)
    np.testing.assert_allclose(calculated.loadings, reused.loadings)
    pd.testing.assert_series_equal(calculated.explained_variance, reused.explained_variance)


def test_pca_rejects_misaligned_precomputed_variances():
    expression = pd.DataFrame([[1.0, 2.0], [2.0, 3.0]], index=["G1", "G2"], columns=["S1", "S2"])
    variances = pd.Series([1.0, 1.0], index=["G2", "G1"])

    try:
        run_pca(expression, PCAConfig(n_components=1), gene_variances=variances)
    except ValueError as exc:
        assert "align" in str(exc)
    else:
        raise AssertionError("misaligned variances should be rejected")
