"""Cautious checks of configured sex-associated expression marker patterns."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from biodetective.core.config import SexMarkerConfig
from biodetective.core.models import Finding


@dataclass(frozen=True)
class SexMarkerAvailability:
    """Availability of configured markers in an expression matrix."""

    markers_found: tuple[str, ...]
    markers_missing: tuple[str, ...]
    x_markers_found: tuple[str, ...]
    y_markers_found: tuple[str, ...]
    x_marker_count: int
    y_marker_count: int
    sufficient_evidence: bool
    status: str


@dataclass(frozen=True)
class SexConsistencyResult:
    """Per-sample marker patterns and cautious inconsistency findings."""

    availability: SexMarkerAvailability
    results: pd.DataFrame
    findings: tuple[Finding, ...]


def _marker_lookup(expression: pd.DataFrame) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for gene_id in expression.index:
        lookup.setdefault(str(gene_id).strip().casefold(), gene_id)
    return lookup


def check_sex_marker_availability(
    expression: pd.DataFrame,
    config: SexMarkerConfig | None = None,
) -> SexMarkerAvailability:
    """Report which configured X- and Y-associated markers are available."""
    config = config or SexMarkerConfig()
    lookup = _marker_lookup(expression)

    x_found = tuple(marker for marker in config.x_associated_markers if marker.casefold() in lookup)
    y_found = tuple(marker for marker in config.y_associated_markers if marker.casefold() in lookup)
    configured_markers = (*config.x_associated_markers, *config.y_associated_markers)
    found = (*x_found, *y_found)
    missing = tuple(marker for marker in configured_markers if marker not in found)
    sufficient = (
        len(x_found) >= config.minimum_x_markers
        and len(y_found) >= config.minimum_y_markers
        and len(found) >= config.minimum_total_markers
    )
    return SexMarkerAvailability(
        markers_found=found,
        markers_missing=missing,
        x_markers_found=x_found,
        y_markers_found=y_found,
        x_marker_count=len(x_found),
        y_marker_count=len(y_found),
        sufficient_evidence=sufficient,
        status="sufficient evidence" if sufficient else "insufficient evidence",
    )


def _empty_results() -> pd.DataFrame:
    columns = [
        "recorded_metadata",
        "observed_marker_pattern",
        "evidence_strength",
        "supporting_genes",
        "x_marker_score",
        "y_marker_score",
        "marker_pattern_score",
    ]
    return pd.DataFrame(columns=columns, index=pd.Index([], name="sample_id"))


def _aligned_metadata(metadata: pd.DataFrame, sex_column: str) -> pd.DataFrame:
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must contain a 'sample_id' column")
    if sex_column not in metadata.columns:
        raise ValueError(f"metadata does not contain column '{sex_column}'")
    if metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("metadata sample IDs must be unique for marker consistency analysis")
    aligned = metadata.loc[:, ["sample_id", sex_column]].copy()
    aligned["sample_id"] = aligned["sample_id"].astype(str)
    return aligned.set_index("sample_id")


def _robust_standardize(marker_profiles: pd.DataFrame) -> pd.DataFrame:
    medians = marker_profiles.median(axis=0)
    absolute_deviations = marker_profiles.sub(medians).abs()
    mad = absolute_deviations.median(axis=0)
    standard_deviation = marker_profiles.std(axis=0, ddof=0)
    scale = (1.4826 * mad).where(mad.gt(0), standard_deviation.where(standard_deviation.gt(0), 1.0))
    return marker_profiles.sub(medians).div(scale)


def _evidence_strength(score: float, supporting_count: int, config: SexMarkerConfig) -> str:
    magnitude = abs(score)
    if supporting_count < config.minimum_supporting_markers or magnitude < config.pattern_score_threshold:
        return "insufficient"
    if magnitude >= config.strong_evidence_threshold:
        return "strong"
    if magnitude >= config.moderate_evidence_threshold:
        return "moderate"
    return "limited"


def analyze_sex_marker_consistency(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    sex_column: str,
    config: SexMarkerConfig | None = None,
) -> SexConsistencyResult:
    """Compare configured marker patterns with recorded metadata cautiously."""
    config = config or SexMarkerConfig()
    availability = check_sex_marker_availability(expression, config)
    if not availability.sufficient_evidence:
        return SexConsistencyResult(availability, _empty_results(), ())
    if not all(pd.api.types.is_numeric_dtype(expression[column]) for column in expression.columns):
        raise ValueError("expression values must be numeric")

    lookup = _marker_lookup(expression)
    marker_names = [*availability.x_markers_found, *availability.y_markers_found]
    actual_marker_ids = [lookup[marker.casefold()] for marker in marker_names]
    marker_expression = expression.loc[actual_marker_ids].copy(deep=True)
    marker_expression.index = marker_names
    marker_profiles = marker_expression.T
    marker_profiles.index = marker_profiles.index.map(str)
    if not np.isfinite(marker_profiles.to_numpy(dtype=float)).all():
        raise ValueError("sex-associated marker analysis requires finite expression values")

    recorded = _aligned_metadata(metadata, sex_column)
    common_sample_ids = [sample_id for sample_id in marker_profiles.index if sample_id in recorded.index]
    if not common_sample_ids:
        raise ValueError("expression and metadata have no samples in common")
    marker_profiles = marker_profiles.loc[common_sample_ids]
    recorded = recorded.loc[common_sample_ids]

    standardized = _robust_standardize(marker_profiles)
    x_scores = standardized.loc[:, list(availability.x_markers_found)].median(axis=1)
    y_scores = standardized.loc[:, list(availability.y_markers_found)].median(axis=1)
    pattern_scores = y_scores - x_scores

    x_metadata_values = {value.strip().casefold() for value in config.x_associated_metadata_values}
    y_metadata_values = {value.strip().casefold() for value in config.y_associated_metadata_values}
    rows: list[dict[str, object]] = []
    findings: list[Finding] = []

    for sample_id in common_sample_ids:
        score = float(pattern_scores.loc[sample_id])
        if score >= config.pattern_score_threshold:
            observed_pattern = "Y-associated expression pattern"
            supporting = [
                marker for marker in availability.y_markers_found if standardized.loc[sample_id, marker] > 0
            ] + [
                marker for marker in availability.x_markers_found if standardized.loc[sample_id, marker] < 0
            ]
            expected_values = y_metadata_values
        elif score <= -config.pattern_score_threshold:
            observed_pattern = "X-associated expression pattern"
            supporting = [
                marker for marker in availability.x_markers_found if standardized.loc[sample_id, marker] > 0
            ] + [
                marker for marker in availability.y_markers_found if standardized.loc[sample_id, marker] < 0
            ]
            expected_values = x_metadata_values
        else:
            observed_pattern = "indeterminate expression pattern"
            supporting = []
            expected_values = set()

        strength = _evidence_strength(score, len(supporting), config)
        recorded_value = recorded.at[sample_id, sex_column]
        normalized_recorded = "" if pd.isna(recorded_value) else str(recorded_value).strip().casefold()
        recorded_known = normalized_recorded in x_metadata_values or normalized_recorded in y_metadata_values
        inconsistent = bool(
            recorded_known
            and expected_values
            and normalized_recorded not in expected_values
            and strength != "insufficient"
        )
        rows.append(
            {
                "sample_id": sample_id,
                "recorded_metadata": None if pd.isna(recorded_value) else str(recorded_value),
                "observed_marker_pattern": observed_pattern,
                "evidence_strength": strength,
                "supporting_genes": supporting,
                "x_marker_score": float(x_scores.loc[sample_id]),
                "y_marker_score": float(y_scores.loc[sample_id]),
                "marker_pattern_score": score,
            }
        )

        if inconsistent:
            findings.append(
                Finding(
                    category="sex_marker_consistency",
                    code="sex_marker_metadata_inconsistency",
                    severity="high" if strength == "strong" else "medium",
                    message=(
                        f"Sample {sample_id}: Sex-associated expression markers appear inconsistent with recorded metadata."
                    ),
                    sample_ids=[sample_id],
                    column=sex_column,
                    evidence={
                        "recorded_metadata": str(recorded_value),
                        "observed_marker_pattern": observed_pattern,
                        "evidence_strength": strength,
                        "supporting_genes": supporting,
                        "x_marker_score": float(x_scores.loc[sample_id]),
                        "y_marker_score": float(y_scores.loc[sample_id]),
                        "marker_pattern_score": score,
                    },
                    recommendation=(
                        "Review sample provenance and metadata independently; marker patterns do not determine biological sex with certainty."
                    ),
                )
            )

    results = pd.DataFrame.from_records(rows).set_index("sample_id")
    return SexConsistencyResult(availability, results, tuple(findings))
