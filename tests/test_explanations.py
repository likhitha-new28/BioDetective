import pytest

from biodetective.core.models import Finding
from biodetective.reporting.explanations import EXPLANATION_TEMPLATES, explain_finding


EXPECTED_FINDING_CODES = {
    "missing_metadata_values",
    "duplicate_sample_ids",
    "identical_metadata_rows",
    "inconsistent_categorical_labels",
    "constant_metadata_column",
    "high_cardinality_categorical_column",
    "class_imbalance",
    "missing_expression_values",
    "positive_infinity_values",
    "negative_infinity_values",
    "zero_variance_genes",
    "very_low_variance_genes",
    "highly_suspicious_similarity",
    "noteworthy_similarity",
    "pca_distance_outlier",
    "combined_sample_outlier",
    "molecular_profile_closer_to_another_group",
    "cross_validated_label_disagreement",
    "sex_marker_metadata_inconsistency",
    "batch_pca_association",
}


def test_every_existing_finding_code_has_an_explanation_template():
    assert EXPECTED_FINDING_CODES <= set(EXPLANATION_TEMPLATES)


@pytest.mark.parametrize("code", sorted(EXPECTED_FINDING_CODES))
def test_explanation_contains_all_required_parts(code):
    finding = Finding(
        category="test",
        code=code,
        severity="medium",
        message="Observed test evidence.",
        evidence={"value": 1},
        recommendation="Original recommendation.",
    )

    explanation = explain_finding(finding)

    assert explanation.observation == finding.message
    assert explanation.evidence == {"value": 1}
    assert explanation.interpretation
    assert explanation.possible_explanations
    assert explanation.recommendation


def test_unknown_finding_uses_cautious_fallback_and_original_recommendation():
    finding = Finding(
        category="future",
        code="future_detector",
        severity="low",
        message="A future observation was recorded.",
        recommendation="Ask a domain expert to review it.",
    )

    explanation = explain_finding(finding)

    assert explanation.recommendation == finding.recommendation
    assert "may" in explanation.interpretation


def test_explanation_copies_evidence_without_modifying_finding():
    finding = Finding("test", "missing_metadata_values", "low", "Missing", evidence={"count": 2})
    explanation = explain_finding(finding)

    explanation.evidence["count"] = 99

    assert finding.evidence["count"] == 2
