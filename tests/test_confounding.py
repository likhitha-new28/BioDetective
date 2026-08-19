import numpy as np
import pandas as pd

from biodetective.analysis.confounding import (
    COMPLETE_CONFOUNDING_EXPLANATION,
    analyze_confounding,
    build_contingency_table,
)


def metadata_from_counts(counts):
    rows = []
    for biological, technical_counts in counts.items():
        for technical, count in technical_counts.items():
            rows.extend({"condition": biological, "batch": technical} for _ in range(count))
    return pd.DataFrame(rows)


def test_contingency_counts_and_conditional_proportions():
    metadata = metadata_from_counts({"Cancer": {"B1": 8, "B2": 2}, "Healthy": {"B1": 3, "B2": 7}})
    result = build_contingency_table(metadata, "condition", "batch")

    assert result.contingency_table.loc["Cancer", "B1"] == 8
    assert result.biological_group_counts == {"Cancer": 10, "Healthy": 10}
    assert result.technical_group_counts == {"B1": 11, "B2": 9}
    assert result.technical_given_biological.loc["Cancer", "B1"] == 0.8
    assert result.biological_given_technical.loc["Cancer", "B1"] == 8 / 11


def test_perfect_association_is_critical_and_explained_as_complete_confounding():
    metadata = metadata_from_counts({"Cancer": {"B1": 10, "B2": 0}, "Healthy": {"B1": 0, "B2": 10}})
    result = analyze_confounding(metadata, "condition", "batch")

    assert result.perfect_association
    assert result.deterministic_relationship
    assert result.cramers_v == 1.0
    assert result.risk == "Critical"
    assert result.interpretation == COMPLETE_CONFOUNDING_EXPLANATION


def test_near_perfect_association_is_detected():
    metadata = metadata_from_counts({"Cancer": {"B1": 9, "B2": 1}, "Healthy": {"B1": 1, "B2": 9}})
    result = analyze_confounding(metadata, "condition", "batch")
    assert result.near_perfect_association
    assert result.risk == "High"
    assert result.cramers_v == 0.8


def test_independent_balanced_variables_have_low_risk():
    metadata = metadata_from_counts({"Cancer": {"B1": 5, "B2": 5}, "Healthy": {"B1": 5, "B2": 5}})
    result = analyze_confounding(metadata, "condition", "batch")
    assert result.cramers_v == 0
    assert result.risk == "Low"
    assert not result.perfect_association


def test_sparse_contingency_table_is_handled_safely():
    metadata = metadata_from_counts(
        {"A": {"B1": 2, "B2": 1, "B3": 0}, "B": {"B1": 0, "B2": 2, "B3": 1}, "C": {"B1": 1, "B2": 0, "B3": 2}}
    )
    result = analyze_confounding(metadata, "condition", "batch")
    assert result.sparse_table
    assert result.chi_square is not None
    assert np.isfinite(result.cramers_v)
    assert result.risk in {"Low", "Moderate", "High", "Critical"}


def test_single_category_table_returns_statistics_safely():
    metadata = pd.DataFrame({"condition": ["Cancer"] * 4, "batch": ["B1", "B1", "B2", "B2"]})
    result = analyze_confounding(metadata, "condition", "batch")
    assert result.chi_square is None
    assert result.p_value is None
    assert result.cramers_v == 0
    assert result.risk in {"Low", "Moderate"}
