"""Sample correlation and potential high-similarity detection."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from biodetective.core.config import (
    DEFAULT_CORRELATION_MIN_PERIODS,
    DEFAULT_SIMILARITY_HIGHLY_SUSPICIOUS_THRESHOLD,
    DEFAULT_SIMILARITY_NOTEWORTHY_THRESHOLD,
)
from biodetective.core.models import Finding


SUPPORTED_CORRELATION_METHODS = frozenset({"pearson", "spearman"})


@dataclass(frozen=True)
class SimilarityConfig:
    """Thresholds used to classify noteworthy sample correlations."""

    noteworthy_threshold: float = DEFAULT_SIMILARITY_NOTEWORTHY_THRESHOLD
    highly_suspicious_threshold: float = DEFAULT_SIMILARITY_HIGHLY_SUSPICIOUS_THRESHOLD

    def __post_init__(self) -> None:
        if not -1 <= self.noteworthy_threshold <= 1:
            raise ValueError("noteworthy_threshold must be between -1 and 1")
        if not -1 <= self.highly_suspicious_threshold <= 1:
            raise ValueError("highly_suspicious_threshold must be between -1 and 1")
        if self.noteworthy_threshold > self.highly_suspicious_threshold:
            raise ValueError("noteworthy_threshold cannot exceed highly_suspicious_threshold")


def calculate_sample_correlations(
    expression: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = DEFAULT_CORRELATION_MIN_PERIODS,
) -> pd.DataFrame:
    """Return a sample-by-sample Pearson or Spearman correlation matrix."""
    normalized_method = method.casefold()
    if normalized_method not in SUPPORTED_CORRELATION_METHODS:
        allowed = ", ".join(sorted(SUPPORTED_CORRELATION_METHODS))
        raise ValueError(f"method must be one of: {allowed}")

    values = expression.to_numpy(copy=False)
    try:
        has_infinity = bool(np.isinf(values).any())
    except TypeError:
        has_infinity = bool(expression.isin([np.inf, -np.inf]).to_numpy().any())
    finite_expression = expression.replace([np.inf, -np.inf], np.nan) if has_infinity else expression
    correlations = finite_expression.corr(method=normalized_method, min_periods=min_periods)
    correlations.index = correlations.index.map(str)
    correlations.columns = correlations.columns.map(str)
    correlations.index.name = "sample_id"
    correlations.columns.name = "sample_id"
    correlations.attrs["method"] = normalized_method
    return correlations


def _python_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _values_differ(first: Any, second: Any) -> bool:
    if pd.isna(first) and pd.isna(second):
        return False
    if pd.isna(first) or pd.isna(second):
        return True
    return bool(first != second)


def _prepare_metadata(
    metadata: pd.DataFrame | None,
) -> tuple[pd.DataFrame | None, list[str]]:
    if metadata is None or metadata.empty:
        return None, []
    compared_columns = [str(column) for column in metadata.columns if column != "sample_id"]
    if "sample_id" in metadata.columns:
        indexed = metadata.assign(sample_id=metadata["sample_id"].astype(str)).drop_duplicates("sample_id", keep="first")
        return indexed.set_index("sample_id", drop=False), compared_columns
    indexed = metadata.set_axis(metadata.index.map(str), axis="index", copy=False)
    return indexed, compared_columns


def _prepared_metadata_differences(
    indexed: pd.DataFrame | None,
    compared_columns: list[str],
    first_sample: str,
    second_sample: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    if indexed is None:
        return {}, []
    if first_sample not in indexed.index or second_sample not in indexed.index:
        return {}, []
    differences: dict[str, dict[str, Any]] = {}
    for column in compared_columns:
        first_value = indexed.at[first_sample, column]
        second_value = indexed.at[second_sample, column]
        if _values_differ(first_value, second_value):
            differences[column] = {
                first_sample: _python_value(first_value),
                second_sample: _python_value(second_value),
            }
    return differences, compared_columns


def _metadata_differences(
    metadata: pd.DataFrame | None,
    first_sample: str,
    second_sample: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed, compared_columns = _prepare_metadata(metadata)
    return _prepared_metadata_differences(indexed, compared_columns, first_sample, second_sample)


def detect_high_similarity_pairs(
    correlation_matrix: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    config: SimilarityConfig | None = None,
) -> list[Finding]:
    """Return one cautious Finding for each noteworthy sample pair."""
    config = config or SimilarityConfig()
    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("correlation_matrix must be square")
    if list(map(str, correlation_matrix.index)) != list(map(str, correlation_matrix.columns)):
        raise ValueError("correlation_matrix rows and columns must contain the same samples in the same order")

    sample_ids = [str(value) for value in correlation_matrix.columns]
    values = correlation_matrix.to_numpy(dtype=float)
    row_indices, column_indices = np.triu_indices(len(sample_ids), k=1)
    pair_correlations = values[row_indices, column_indices]
    qualifying = np.isfinite(pair_correlations) & (pair_correlations >= config.noteworthy_threshold)
    indexed_metadata, compared_columns = _prepare_metadata(metadata)

    findings: list[Finding] = []
    for row_index, column_index, correlation in zip(
        row_indices[qualifying],
        column_indices[qualifying],
        pair_correlations[qualifying],
    ):
        first_sample = sample_ids[int(row_index)]
        second_sample = sample_ids[int(column_index)]
        is_highly_suspicious = correlation >= config.highly_suspicious_threshold
        level = "highly suspicious similarity" if is_highly_suspicious else "noteworthy similarity"
        differences, pair_compared_columns = _prepared_metadata_differences(
            indexed_metadata,
            compared_columns,
            first_sample,
            second_sample,
        )

        findings.append(
            Finding(
                category="sample_similarity",
                code="highly_suspicious_similarity" if is_highly_suspicious else "noteworthy_similarity",
                severity="high" if is_highly_suspicious else "medium",
                message=(
                    "Potential duplicate or highly similar samples: "
                    f"{first_sample} and {second_sample} have correlation {correlation:.4f}."
                ),
                sample_ids=[first_sample, second_sample],
                evidence={
                    "sample_1": first_sample,
                    "sample_2": second_sample,
                    "correlation": float(correlation),
                    "similarity_level": level,
                    "correlation_method": correlation_matrix.attrs.get("method"),
                    "compared_metadata_columns": pair_compared_columns,
                    "metadata_differences": differences,
                },
                recommendation=(
                    "Review sample provenance and the listed metadata differences; similarity alone does not prove duplication."
                ),
            )
        )
    return findings


def analyze_sample_similarity(
    expression: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    method: str = "pearson",
    config: SimilarityConfig | None = None,
    min_periods: int = DEFAULT_CORRELATION_MIN_PERIODS,
) -> tuple[pd.DataFrame, list[Finding]]:
    """Calculate correlations and return potential high-similarity findings."""
    correlations = calculate_sample_correlations(expression, method=method, min_periods=min_periods)
    findings = detect_high_similarity_pairs(correlations, metadata=metadata, config=config)
    return correlations, findings
