"""Two-variable contingency and confounding analysis."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from biodetective.core.config import (
    DEFAULT_CONFOUNDING_HIGH_CRAMERS_V,
    DEFAULT_CONFOUNDING_MODERATE_CONDITIONAL_PROPORTION,
    DEFAULT_CONFOUNDING_MODERATE_CRAMERS_V,
    DEFAULT_CONFOUNDING_NEAR_PERFECT_PROPORTION,
)


COMPLETE_CONFOUNDING_EXPLANATION = (
    "The biological and technical variables cannot be cleanly separated using this dataset alone."
)


@dataclass(frozen=True)
class ConfoundingConfig:
    """Effect-size and conditional-proportion thresholds for confounding risk."""

    moderate_cramers_v: float = DEFAULT_CONFOUNDING_MODERATE_CRAMERS_V
    high_cramers_v: float = DEFAULT_CONFOUNDING_HIGH_CRAMERS_V
    near_perfect_proportion: float = DEFAULT_CONFOUNDING_NEAR_PERFECT_PROPORTION
    moderate_conditional_proportion: float = DEFAULT_CONFOUNDING_MODERATE_CONDITIONAL_PROPORTION

    def __post_init__(self) -> None:
        if not 0 <= self.moderate_cramers_v <= self.high_cramers_v <= 1:
            raise ValueError("Cramer's V thresholds must be ordered between 0 and 1")
        if not 0 < self.moderate_conditional_proportion <= self.near_perfect_proportion <= 1:
            raise ValueError("conditional-proportion thresholds must be ordered in (0, 1]")


@dataclass(frozen=True)
class ContingencyResult:
    contingency_table: pd.DataFrame
    technical_given_biological: pd.DataFrame
    biological_given_technical: pd.DataFrame
    biological_group_counts: dict[str, int]
    technical_group_counts: dict[str, int]


@dataclass(frozen=True)
class ConfoundingResult:
    contingency: ContingencyResult
    chi_square: float | None
    p_value: float | None
    degrees_of_freedom: int | None
    expected_counts: pd.DataFrame | None
    cramers_v: float
    sparse_table: bool
    perfect_association: bool
    near_perfect_association: bool
    deterministic_relationship: bool
    risk: str
    interpretation: str
    evidence: dict[str, object]


def build_contingency_table(
    metadata: pd.DataFrame,
    biological_column: str,
    technical_column: str,
) -> ContingencyResult:
    """Build counts and both directions of conditional proportions."""
    if biological_column == technical_column:
        raise ValueError("biological and technical variables must be different columns")
    for column in (biological_column, technical_column):
        if column not in metadata.columns:
            raise ValueError(f"metadata does not contain column '{column}'")

    complete_rows = metadata.loc[:, [biological_column, technical_column]].dropna().astype(str)
    if complete_rows.empty:
        raise ValueError("no complete rows are available for the selected variables")
    contingency = pd.crosstab(
        complete_rows[biological_column],
        complete_rows[technical_column],
        dropna=False,
    )
    contingency.index.name = biological_column
    contingency.columns.name = technical_column
    technical_given_biological = contingency.div(contingency.sum(axis=1), axis=0)
    biological_given_technical = contingency.div(contingency.sum(axis=0), axis=1)
    biological_counts = {str(group): int(count) for group, count in contingency.sum(axis=1).items()}
    technical_counts = {str(group): int(count) for group, count in contingency.sum(axis=0).items()}
    return ContingencyResult(
        contingency,
        technical_given_biological,
        biological_given_technical,
        biological_counts,
        technical_counts,
    )


def analyze_confounding(
    metadata: pd.DataFrame,
    biological_column: str,
    technical_column: str,
    config: ConfoundingConfig | None = None,
) -> ConfoundingResult:
    """Calculate association statistics and an effect-size-driven confounding risk."""
    config = config or ConfoundingConfig()
    contingency = build_contingency_table(metadata, biological_column, technical_column)
    table = contingency.contingency_table
    row_count, column_count = table.shape
    sample_count = int(table.to_numpy().sum())

    chi_square = p_value = None
    degrees_of_freedom = None
    expected_counts = None
    sparse_table = True
    cramers_v = 0.0
    if row_count >= 2 and column_count >= 2 and sample_count > 0:
        statistic, probability, dof, expected = stats.chi2_contingency(table.to_numpy(), correction=False)
        chi_square = float(statistic)
        p_value = float(probability)
        degrees_of_freedom = int(dof)
        expected_counts = pd.DataFrame(expected, index=table.index, columns=table.columns)
        sparse_table = bool((expected_counts < 5).any().any())
        denominator = sample_count * min(row_count - 1, column_count - 1)
        cramers_v = float(np.sqrt(chi_square / denominator)) if denominator > 0 else 0.0

    row_maxima = contingency.technical_given_biological.max(axis=1)
    column_maxima = contingency.biological_given_technical.max(axis=0)
    rows_deterministic = bool(len(row_maxima) > 0 and np.isclose(row_maxima, 1.0).all())
    columns_deterministic = bool(len(column_maxima) > 0 and np.isclose(column_maxima, 1.0).all())
    meaningful_dimensions = row_count >= 2 and column_count >= 2
    perfect_association = meaningful_dimensions and rows_deterministic and columns_deterministic
    deterministic_relationship = meaningful_dimensions and (rows_deterministic or columns_deterministic)
    minimum_directional_maximum = max(float(row_maxima.min()), float(column_maxima.min()))
    near_perfect = bool(
        meaningful_dimensions
        and not perfect_association
        and minimum_directional_maximum >= config.near_perfect_proportion
    )
    maximum_conditional = max(float(row_maxima.max()), float(column_maxima.max()))

    if perfect_association or (deterministic_relationship and cramers_v >= config.high_cramers_v):
        risk = "Critical"
        interpretation = COMPLETE_CONFOUNDING_EXPLANATION
    elif near_perfect or cramers_v >= config.high_cramers_v:
        risk = "High"
        interpretation = "The selected biological and technical variables are strongly associated and may be difficult to separate."
    elif cramers_v >= config.moderate_cramers_v or maximum_conditional >= config.moderate_conditional_proportion:
        risk = "Moderate"
        interpretation = "The selected variables show partial association that should be considered during analysis."
    else:
        risk = "Low"
        interpretation = "The selected variables show limited evidence of confounding in this dataset."

    evidence: dict[str, object] = {
        "biological_column": biological_column,
        "technical_column": technical_column,
        "sample_count": sample_count,
        "cramers_v": cramers_v,
        "maximum_conditional_proportion": maximum_conditional,
        "rows_deterministic": rows_deterministic,
        "columns_deterministic": columns_deterministic,
        "sparse_table": sparse_table,
    }
    return ConfoundingResult(
        contingency,
        chi_square,
        p_value,
        degrees_of_freedom,
        expected_counts,
        cramers_v,
        sparse_table,
        perfect_association,
        near_perfect,
        deterministic_relationship,
        risk,
        interpretation,
        evidence,
    )
