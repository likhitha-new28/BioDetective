import numpy as np
import pandas as pd

from biodetective.analysis.sex_consistency import (
    analyze_sex_marker_consistency,
    check_sex_marker_availability,
)
from biodetective.core.config import SexMarkerConfig


DEFAULT_MARKERS = ["XIST", "RPS4Y1", "KDM5D", "DDX3Y", "UTY", "EIF1AY"]


def test_default_marker_configuration_contains_requested_examples():
    config = SexMarkerConfig()
    assert config.x_associated_markers == ("XIST",)
    assert config.y_associated_markers == ("RPS4Y1", "KDM5D", "DDX3Y", "UTY", "EIF1AY")


def test_marker_availability_reports_found_missing_and_group_counts_case_insensitively():
    expression = pd.DataFrame({"S1": [1, 2, 3]}, index=["xist", "RPS4Y1", "KDM5D"])
    result = check_sex_marker_availability(expression)

    assert result.markers_found == ("XIST", "RPS4Y1", "KDM5D")
    assert result.x_marker_count == 1
    assert result.y_marker_count == 2
    assert set(result.markers_missing) == {"DDX3Y", "UTY", "EIF1AY"}
    assert result.sufficient_evidence
    assert result.status == "sufficient evidence"


def test_marker_availability_returns_insufficient_evidence_when_too_few_exist():
    expression = pd.DataFrame({"S1": [1, 2]}, index=["XIST", "RPS4Y1"])
    result = check_sex_marker_availability(expression)
    assert not result.sufficient_evidence
    assert result.status == "insufficient evidence"


def synthetic_marker_data(seed=42, swapped_indices=()):
    rng = np.random.default_rng(seed)
    sample_count = 20
    sample_ids = [f"S{i:02d}" for i in range(1, sample_count + 1)]
    recorded = ["Female"] * 10 + ["Male"] * 10
    for index in swapped_indices:
        recorded[index] = "Male" if recorded[index] == "Female" else "Female"

    values = np.empty((len(DEFAULT_MARKERS) + 4, sample_count))
    values[0, :10] = rng.normal(10, 0.4, 10)
    values[0, 10:] = rng.normal(1, 0.4, 10)
    for marker_index in range(1, len(DEFAULT_MARKERS)):
        values[marker_index, :10] = rng.normal(1, 0.4, 10)
        values[marker_index, 10:] = rng.normal(10, 0.4, 10)
    values[len(DEFAULT_MARKERS):] = rng.normal(5, 1, size=(4, sample_count))
    expression = pd.DataFrame(
        values,
        index=[*DEFAULT_MARKERS, "G1", "G2", "G3", "G4"],
        columns=sample_ids,
    )
    metadata = pd.DataFrame({"sample_id": sample_ids, "reported_sex": recorded})
    return expression, metadata


def test_sex_marker_patterns_match_clear_synthetic_groups_without_certainty_claims():
    expression, metadata = synthetic_marker_data()
    expression_before = expression.copy(deep=True)
    metadata_before = metadata.copy(deep=True)

    result = analyze_sex_marker_consistency(expression, metadata, "reported_sex")

    assert result.availability.sufficient_evidence
    assert set(result.results.iloc[:10]["observed_marker_pattern"]) == {"X-associated expression pattern"}
    assert set(result.results.iloc[10:]["observed_marker_pattern"]) == {"Y-associated expression pattern"}
    assert result.findings == ()
    assert result.results["supporting_genes"].map(len).min() >= 2
    pd.testing.assert_frame_equal(expression, expression_before)
    pd.testing.assert_frame_equal(metadata, metadata_before)


def test_swapped_metadata_produces_cautious_multi_marker_findings():
    expression, metadata = synthetic_marker_data(seed=7, swapped_indices=(1, 15))
    result = analyze_sex_marker_consistency(expression, metadata, "reported_sex")

    assert {finding.sample_ids[0] for finding in result.findings} == {"S02", "S16"}
    assert all(
        finding.message.endswith("Sex-associated expression markers appear inconsistent with recorded metadata.")
        for finding in result.findings
    )
    assert all(len(finding.evidence["supporting_genes"]) >= 2 for finding in result.findings)
    assert all("with certainty" in finding.recommendation for finding in result.findings)
    assert all("biological sex is" not in finding.message.casefold() for finding in result.findings)


def test_analysis_returns_no_pattern_results_when_marker_evidence_is_insufficient():
    expression = pd.DataFrame({"S1": [5, 1], "S2": [1, 5]}, index=["XIST", "RPS4Y1"])
    metadata = pd.DataFrame({"sample_id": ["S1", "S2"], "sex": ["Female", "Male"]})
    result = analyze_sex_marker_consistency(expression, metadata, "sex")
    assert result.availability.status == "insufficient evidence"
    assert result.results.empty
    assert result.findings == ()


def test_custom_marker_configuration_drives_analysis_logic():
    config = SexMarkerConfig(
        x_associated_markers=("CUSTOM_X",),
        y_associated_markers=("CUSTOM_Y1", "CUSTOM_Y2"),
    )
    expression = pd.DataFrame(
        {
            "S1": [10, 1, 1],
            "S2": [9, 1, 1],
            "S3": [1, 9, 10],
            "S4": [1, 10, 9],
        },
        index=["CUSTOM_X", "CUSTOM_Y1", "CUSTOM_Y2"],
    )
    metadata = pd.DataFrame({"sample_id": ["S1", "S2", "S3", "S4"], "sex": ["Female", "Female", "Male", "Male"]})
    result = analyze_sex_marker_consistency(expression, metadata, "sex", config)
    assert result.availability.markers_found == ("CUSTOM_X", "CUSTOM_Y1", "CUSTOM_Y2")
    assert result.findings == ()
