import pandas as pd
import pytest

from biodetective.core.models import Finding
from biodetective.scoring.suspicion import (
    DatasetHealthConfig,
    SuspicionScoreConfig,
    calculate_dataset_health,
    risk_label,
    score_samples,
)


def test_sample_score_is_configurable_bounded_and_has_complete_breakdown():
    config = SuspicionScoreConfig(weights={"pca_outlier": 60, "metadata_issues": 50})
    scores = score_samples(
        ["S1", "S2"],
        evidence_by_sample={"S1": {"pca_outlier": True, "metadata_issues": 0.5}},
        config=config,
    )
    assert scores[0].score == 85
    assert scores[0].contributions == {"pca_outlier": 60, "metadata_issues": 25}
    assert scores[1].score == 0
    assert scores[1].contributions == {"pca_outlier": 0, "metadata_issues": 0}


def test_missing_analyses_contribute_zero_not_penalties():
    result = score_samples(["S1"])[0]
    assert result.score == 0
    assert all(value == 0 for value in result.contributions.values())


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0, "Low"), (24, "Low"), (25, "Moderate"), (49, "Moderate"), (50, "High"), (74, "High"), (75, "Critical"), (100, "Critical")],
)
def test_risk_label_boundaries(score, expected):
    assert risk_label(score) == expected


def test_findings_are_mapped_to_sample_contributions():
    findings = [
        Finding(
            "sample_similarity",
            "highly_suspicious_similarity",
            "high",
            "Potential duplicate or highly similar samples",
            sample_ids=["S1", "S2"],
        ),
        Finding(
            "missing_metadata",
            "missing_metadata_values",
            "medium",
            "Missing metadata",
            sample_ids=["S1"],
        ),
    ]
    scores = {result.sample_id: result for result in score_samples(["S1", "S2"], findings=findings)}
    assert scores["S1"].contributions["duplicate_similarity"] == 25
    assert scores["S1"].contributions["metadata_issues"] == 2.5
    assert scores["S2"].contributions["metadata_issues"] == 0


def test_dataset_health_has_interpretable_deduction_breakdown():
    sample_scores = score_samples(
        ["S1", "S2"],
        evidence_by_sample={"S1": {"pca_outlier": 1, "duplicate_similarity": 1}},
    )
    metadata = pd.DataFrame({"sample_id": ["S1", "S2"], "condition": [None, "A"]})
    duplicates = [Finding("sample_similarity", "highly_suspicious_similarity", "high", "Pair", sample_ids=["S1"])]
    expression_findings = [Finding("expression_quality", "missing_expression_values", "high", "Missing")]

    result = calculate_dataset_health(
        sample_scores=sample_scores,
        metadata=metadata,
        duplicate_findings=duplicates,
        batch_risk="Moderate",
        confounding_risk="High",
        expression_findings=expression_findings,
    )

    assert 0 <= result.score <= 100
    assert set(result.deductions) == {
        "suspicious_sample_percentage",
        "metadata_completeness",
        "duplicate_risk",
        "batch_risk",
        "confounding_risk",
        "expression_qc",
    }
    assert all("deduction" in item and "available" in item for item in result.deductions.values())


def test_missing_health_analyses_do_not_reduce_score():
    result = calculate_dataset_health()
    assert result.score == 100
    assert all(not item["available"] and item["deduction"] == 0 for item in result.deductions.values())


def test_health_deduction_weights_are_configurable():
    config = DatasetHealthConfig(maximum_deductions={"batch_risk": 40})
    result = calculate_dataset_health(batch_risk="High", config=config)
    assert result.score == 60
    assert result.deductions["batch_risk"]["deduction"] == 40
