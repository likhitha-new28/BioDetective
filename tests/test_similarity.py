import numpy as np
import pandas as pd
import pytest

from biodetective.analysis.similarity import (
    SimilarityConfig,
    analyze_sample_similarity,
    calculate_sample_correlations,
    detect_high_similarity_pairs,
)


def test_pearson_sample_correlation_matrix():
    expression = pd.DataFrame(
        {"S01": [1, 2, 3, 4], "S02": [2, 4, 6, 8], "S03": [4, 3, 2, 1]},
        index=["G1", "G2", "G3", "G4"],
    )

    result = calculate_sample_correlations(expression, method="pearson")

    assert result.shape == (3, 3)
    assert result.index.tolist() == ["S01", "S02", "S03"]
    assert np.allclose(np.diag(result), 1.0)
    assert result.loc["S01", "S02"] == pytest.approx(1.0)
    assert result.loc["S01", "S03"] == pytest.approx(-1.0)
    assert result.attrs["method"] == "pearson"


def test_spearman_supports_monotonic_relationships():
    expression = pd.DataFrame({"S01": [1, 2, 3, 4], "S02": [1, 4, 9, 16]})
    result = calculate_sample_correlations(expression, method="spearman")
    assert result.loc["S01", "S02"] == pytest.approx(1.0)
    assert result.attrs["method"] == "spearman"


def test_correlation_rejects_unknown_method():
    with pytest.raises(ValueError, match="method"):
        calculate_sample_correlations(pd.DataFrame({"S01": [1, 2]}), method="kendall")


def test_correlation_does_not_modify_expression():
    expression = pd.DataFrame({"S01": [1.0, np.inf, 3.0], "S02": [1.0, 2.0, 3.0]})
    original = expression.copy(deep=True)
    calculate_sample_correlations(expression)
    pd.testing.assert_frame_equal(expression, original)


def test_high_similarity_pair_thresholds_and_unique_ordering():
    matrix = pd.DataFrame(
        [[1.0, 0.996, 0.981], [0.996, 1.0, 0.5], [0.981, 0.5, 1.0]],
        index=["S01", "S02", "S03"],
        columns=["S01", "S02", "S03"],
    )
    matrix.attrs["method"] = "pearson"

    findings = detect_high_similarity_pairs(matrix)

    assert len(findings) == 2
    assert [finding.sample_ids for finding in findings] == [["S01", "S02"], ["S01", "S03"]]
    assert [finding.code for finding in findings] == ["highly_suspicious_similarity", "noteworthy_similarity"]
    assert all("Potential duplicate or highly similar samples" in finding.message for finding in findings)
    assert all("definitely" not in finding.message.casefold() for finding in findings)


def test_similarity_thresholds_are_configurable():
    matrix = pd.DataFrame([[1.0, 0.9], [0.9, 1.0]], index=["S1", "S2"], columns=["S1", "S2"])
    config = SimilarityConfig(noteworthy_threshold=0.85, highly_suspicious_threshold=0.95)
    findings = detect_high_similarity_pairs(matrix, config=config)
    assert len(findings) == 1
    assert findings[0].code == "noteworthy_similarity"


def test_default_similarity_thresholds_are_inclusive():
    matrix = pd.DataFrame(
        [[1.0, 0.995, 0.98], [0.995, 1.0, 0.2], [0.98, 0.2, 1.0]],
        index=["S1", "S2", "S3"],
        columns=["S1", "S2", "S3"],
    )
    findings = detect_high_similarity_pairs(matrix)
    assert [finding.code for finding in findings] == ["highly_suspicious_similarity", "noteworthy_similarity"]


def test_metadata_differences_are_recorded_without_deciding_correct_value():
    matrix = pd.DataFrame([[1.0, 0.998], [0.998, 1.0]], index=["S01", "S08"], columns=["S01", "S08"])
    metadata = pd.DataFrame(
        {
            "sample_id": ["S01", "S08"],
            "condition": ["Cancer", "Healthy"],
            "sex": ["Male", "Male"],
            "batch": ["B1", "B2"],
            "site": ["A", "B"],
        }
    )
    original = metadata.copy(deep=True)

    finding = detect_high_similarity_pairs(matrix, metadata=metadata)[0]
    differences = finding.evidence["metadata_differences"]

    assert differences == {
        "condition": {"S01": "Cancer", "S08": "Healthy"},
        "batch": {"S01": "B1", "S08": "B2"},
        "site": {"S01": "A", "S08": "B"},
    }
    assert "sex" in finding.evidence["compared_metadata_columns"]
    assert "correct" not in finding.message.casefold()
    pd.testing.assert_frame_equal(metadata, original)


def test_analyze_sample_similarity_returns_matrix_and_findings():
    expression = pd.DataFrame({"S1": [1, 2, 3], "S2": [2, 4, 6], "S3": [3, 1, 2]})
    matrix, findings = analyze_sample_similarity(expression, method="pearson")
    assert matrix.shape == (3, 3)
    assert findings[0].sample_ids == ["S1", "S2"]
