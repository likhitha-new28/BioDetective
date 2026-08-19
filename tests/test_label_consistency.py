import numpy as np
import pandas as pd

from biodetective.analysis.label_consistency import (
    LabelConsistencyConfig,
    analyze_centroid_similarity,
    analyze_label_consistency,
    cross_validated_label_consistency,
)


def separated_groups(seed=42, samples_per_group=12, swapped=()):
    rng = np.random.default_rng(seed)
    gene_count = 40
    group_a = np.concatenate([np.full(gene_count // 2, 5.0), np.full(gene_count // 2, -5.0)])
    group_b = -group_a
    values_a = group_a[:, None] + rng.normal(0, 0.35, size=(gene_count, samples_per_group))
    values_b = group_b[:, None] + rng.normal(0, 0.35, size=(gene_count, samples_per_group))
    values = np.hstack([values_a, values_b])
    sample_ids = [f"S{i:02d}" for i in range(1, samples_per_group * 2 + 1)]
    labels = ["GroupA"] * samples_per_group + ["GroupB"] * samples_per_group
    for index in swapped:
        labels[index] = "GroupB" if labels[index] == "GroupA" else "GroupA"
    expression = pd.DataFrame(values, index=[f"G{i:03d}" for i in range(gene_count)], columns=sample_ids)
    metadata = pd.DataFrame({"sample_id": sample_ids, "condition": labels, "batch": ["B1", "B2"] * samples_per_group})
    return expression, metadata


def test_centroid_similarity_separates_clear_groups_without_modifying_inputs():
    expression, metadata = separated_groups()
    expression_before = expression.copy(deep=True)
    metadata_before = metadata.copy(deep=True)

    result = analyze_centroid_similarity(expression, metadata, "condition")

    assert result.results["appears_more_similar_to_another_group"].sum() == 0
    assert (result.results["recorded_group"] == result.results["molecular_closest_group"]).all()
    assert result.similarities.shape == (24, 2)
    assert result.centroids.shape == (2, 40)
    pd.testing.assert_frame_equal(expression, expression_before)
    pd.testing.assert_frame_equal(metadata, metadata_before)


def test_swapped_labels_have_stronger_mismatch_evidence_than_correct_labels():
    swapped_indices = (1, 15)
    expression, metadata = separated_groups(seed=7, swapped=swapped_indices)
    result = analyze_centroid_similarity(
        expression,
        metadata,
        "condition",
        LabelConsistencyConfig(minimum_similarity_margin=0.05),
    )
    swapped_ids = [metadata.loc[index, "sample_id"] for index in swapped_indices]
    correct_ids = [sample_id for sample_id in metadata["sample_id"] if sample_id not in swapped_ids]

    assert result.results.loc[swapped_ids, "appears_more_similar_to_another_group"].all()
    assert result.results.loc[swapped_ids, "similarity_margin"].median() > result.results.loc[correct_ids, "similarity_margin"].median()
    assert {finding.sample_ids[0] for finding in result.findings} == set(swapped_ids)
    assert all("molecular profile appears more similar to another metadata group" in finding.message for finding in result.findings)
    assert all("mislabeled" not in finding.message.casefold() for finding in result.findings)


def test_cross_validated_predictions_are_out_of_fold_and_accurate_for_separated_groups():
    expression, metadata = separated_groups(seed=11)
    result = cross_validated_label_consistency(expression, metadata, "condition")

    assert result.available
    assert result.reason is None
    assert result.results.shape == (24, 4)
    assert result.probabilities.shape == (24, 2)
    assert result.results["prediction_matches_recorded"].mean() >= 0.95
    assert ((result.results["confidence"] >= 0) & (result.results["confidence"] <= 1)).all()


def test_cross_validation_requires_minimum_samples_per_class():
    expression, metadata = separated_groups(samples_per_group=2)
    config = LabelConsistencyConfig(min_samples_per_class=3)
    result = cross_validated_label_consistency(expression, metadata, "condition", config)

    assert not result.available
    assert "at least 3" in result.reason
    assert result.results.empty


def test_cross_validated_result_contains_cautious_disagreement_evidence_for_swaps():
    expression, metadata = separated_groups(seed=21, swapped=(0, 13))
    result = cross_validated_label_consistency(expression, metadata, "condition")

    assert result.available
    assert {"S01", "S14"} <= {finding.sample_ids[0] for finding in result.findings}
    assert all("review evidence only" in finding.recommendation for finding in result.findings)


def test_combined_label_consistency_can_disable_classifier():
    expression, metadata = separated_groups()
    result = analyze_label_consistency(expression, metadata, "condition", include_classifier=False)
    assert result.cross_validated is None
