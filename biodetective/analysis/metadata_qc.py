"""Metadata-only quality checks for BioDetective datasets."""

from collections.abc import Mapping
from dataclasses import dataclass
import re

import pandas as pd

from biodetective.core.config import (
    DEFAULT_METADATA_HIGH_CARDINALITY_MIN_UNIQUE,
    DEFAULT_METADATA_HIGH_CARDINALITY_RATIO,
    DEFAULT_METADATA_IMBALANCE_MAX_CATEGORIES,
    DEFAULT_METADATA_IMBALANCE_MIN_FRACTION,
    DEFAULT_METADATA_MISSING_HIGH_MAX,
    DEFAULT_METADATA_MISSING_LOW_MAX,
    DEFAULT_METADATA_MISSING_MEDIUM_MAX,
)
from biodetective.core.models import Finding


@dataclass(frozen=True)
class MetadataQCConfig:
    """Configurable thresholds for metadata quality checks."""

    missing_low_max: float = DEFAULT_METADATA_MISSING_LOW_MAX
    missing_medium_max: float = DEFAULT_METADATA_MISSING_MEDIUM_MAX
    missing_high_max: float = DEFAULT_METADATA_MISSING_HIGH_MAX
    high_cardinality_ratio: float = DEFAULT_METADATA_HIGH_CARDINALITY_RATIO
    high_cardinality_min_unique: int = DEFAULT_METADATA_HIGH_CARDINALITY_MIN_UNIQUE
    imbalance_min_fraction: float = DEFAULT_METADATA_IMBALANCE_MIN_FRACTION
    imbalance_max_categories: int = DEFAULT_METADATA_IMBALANCE_MAX_CATEGORIES


def _missing_severity(percentage: float, config: MetadataQCConfig) -> str:
    if percentage < config.missing_low_max:
        return "low"
    if percentage <= config.missing_medium_max:
        return "medium"
    if percentage <= config.missing_high_max:
        return "high"
    return "critical"


def _sample_ids(metadata: pd.DataFrame, mask: pd.Series) -> list[str]:
    if "sample_id" in metadata.columns:
        return [str(value) for value in metadata.loc[mask, "sample_id"].dropna().tolist()]
    return [str(value) for value in metadata.index[mask].tolist()]


def _categorical_columns(metadata: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in metadata.columns:
        dtype = metadata[column].dtype
        if isinstance(dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            columns.append(str(column))
    return columns


def detect_missing_metadata(
    metadata: pd.DataFrame,
    config: MetadataQCConfig | None = None,
) -> list[Finding]:
    """Return one finding for every metadata column containing missing values."""
    config = config or MetadataQCConfig()
    findings: list[Finding] = []
    row_count = len(metadata)
    if row_count == 0:
        return findings

    for column in metadata.columns:
        missing_mask = metadata[column].isna()
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            continue
        percentage = missing_count / row_count * 100
        affected_samples = _sample_ids(metadata, missing_mask)
        findings.append(
            Finding(
                category="missing_metadata",
                code="missing_metadata_values",
                severity=_missing_severity(percentage, config),
                message=f"Column '{column}' has {missing_count} missing value(s) ({percentage:.1f}%).",
                sample_ids=affected_samples,
                column=str(column),
                evidence={"missing_count": missing_count, "missing_percentage": round(percentage, 2)},
                recommendation=f"Review and complete missing values in '{column}', or document why they are unavailable.",
            )
        )
    return findings


def detect_metadata_duplicates(metadata: pd.DataFrame) -> list[Finding]:
    """Detect duplicate sample IDs and samples with identical annotations."""
    findings: list[Finding] = []

    if "sample_id" in metadata.columns:
        duplicate_id_mask = metadata["sample_id"].notna() & metadata["sample_id"].duplicated(keep=False)
        if duplicate_id_mask.any():
            duplicate_ids = sorted({str(value) for value in metadata.loc[duplicate_id_mask, "sample_id"]})
            findings.append(
                Finding(
                    category="metadata_duplicates",
                    code="duplicate_sample_ids",
                    severity="critical",
                    message=f"Metadata contains {len(duplicate_ids)} duplicate sample ID(s).",
                    sample_ids=duplicate_ids,
                    column="sample_id",
                    evidence={"duplicate_sample_ids": duplicate_ids, "affected_rows": int(duplicate_id_mask.sum())},
                    recommendation="Assign one unique sample_id to each metadata row before analysis.",
                )
            )

    comparison_columns = [column for column in metadata.columns if column != "sample_id"]
    if comparison_columns:
        identical_mask = metadata.duplicated(subset=comparison_columns, keep=False)
        if identical_mask.any():
            affected = _sample_ids(metadata, identical_mask)
            findings.append(
                Finding(
                    category="metadata_duplicates",
                    code="identical_metadata_rows",
                    severity="medium",
                    message=f"{int(identical_mask.sum())} samples have identical metadata annotations.",
                    sample_ids=affected,
                    evidence={"compared_columns": [str(column) for column in comparison_columns], "row_count": int(identical_mask.sum())},
                    recommendation="Confirm that these samples are distinct and that their annotations were not copied accidentally.",
                )
            )
    return findings


def _normalize_label(value: object) -> str:
    normalized = str(value).strip().casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def detect_category_inconsistencies(
    metadata: pd.DataFrame,
    aliases: Mapping[str, Mapping[str, str]] | None = None,
) -> list[Finding]:
    """Find categorical variants that differ only by case, spacing, or punctuation."""
    aliases = aliases or {}
    findings: list[Finding] = []

    for column in _categorical_columns(metadata):
        if column == "sample_id":
            continue
        column_aliases = {
            _normalize_label(alias): _normalize_label(canonical)
            for alias, canonical in aliases.get(column, {}).items()
        }
        groups: dict[str, dict[str, list[int]]] = {}
        for index, value in metadata[column].items():
            if pd.isna(value):
                continue
            original = str(value)
            comparison_key = _normalize_label(original)
            comparison_key = column_aliases.get(comparison_key, comparison_key)
            groups.setdefault(comparison_key, {}).setdefault(original, []).append(index)

        for comparison_key, variants in groups.items():
            if len(variants) < 2:
                continue
            row_mask = metadata.index.isin([index for indices in variants.values() for index in indices])
            variant_counts = {variant: len(indices) for variant, indices in sorted(variants.items())}
            findings.append(
                Finding(
                    category="category_consistency",
                    code="inconsistent_categorical_labels",
                    severity="medium",
                    message=f"Column '{column}' contains equivalent labels with inconsistent formatting: {', '.join(variant_counts)}.",
                    sample_ids=_sample_ids(metadata, pd.Series(row_mask, index=metadata.index)),
                    column=column,
                    evidence={"comparison_label": comparison_key, "variants": variant_counts},
                    recommendation=f"Choose one canonical label for these variants in '{column}'.",
                )
            )
    return findings


def detect_metadata_structure(
    metadata: pd.DataFrame,
    config: MetadataQCConfig | None = None,
) -> list[Finding]:
    """Detect constant, high-cardinality, and imbalanced metadata columns."""
    config = config or MetadataQCConfig()
    findings: list[Finding] = []
    row_count = len(metadata)

    for column in metadata.columns:
        non_missing = metadata[column].dropna()
        unique_count = int(non_missing.nunique())
        if row_count > 0 and unique_count == 1:
            findings.append(
                Finding(
                    category="metadata_structure",
                    code="constant_metadata_column",
                    severity="low",
                    message=f"Column '{column}' contains only one non-missing value.",
                    column=str(column),
                    evidence={"unique_count": 1, "value": str(non_missing.iloc[0])},
                    recommendation=f"Confirm whether constant column '{column}' is informative for this dataset.",
                )
            )

    for column in _categorical_columns(metadata):
        if column == "sample_id":
            continue
        non_missing = metadata[column].dropna()
        if non_missing.empty:
            continue
        counts = non_missing.astype(str).value_counts()
        unique_count = int(counts.size)
        unique_ratio = unique_count / len(non_missing)

        if unique_count >= config.high_cardinality_min_unique and unique_ratio >= config.high_cardinality_ratio:
            findings.append(
                Finding(
                    category="metadata_structure",
                    code="high_cardinality_categorical_column",
                    severity="medium",
                    message=f"Categorical column '{column}' has {unique_count} unique values ({unique_ratio:.1%} of non-missing rows).",
                    column=column,
                    evidence={"unique_count": unique_count, "unique_ratio": round(unique_ratio, 4)},
                    recommendation="Confirm whether this column is an identifier or free text rather than an analysis category.",
                )
            )

        if 2 <= unique_count <= config.imbalance_max_categories:
            smallest_fraction = float(counts.min() / counts.sum())
            if smallest_fraction < config.imbalance_min_fraction:
                findings.append(
                    Finding(
                        category="metadata_structure",
                        code="class_imbalance",
                        severity="medium",
                        message=f"Column '{column}' has a minority class representing {smallest_fraction:.1%} of non-missing samples.",
                        column=column,
                        evidence={"class_counts": counts.to_dict(), "smallest_class_fraction": round(smallest_fraction, 4)},
                        recommendation="Consider class balance when designing comparisons and interpreting results.",
                    )
                )
    return findings


def run_metadata_qc(
    metadata: pd.DataFrame,
    config: MetadataQCConfig | None = None,
    aliases: Mapping[str, Mapping[str, str]] | None = None,
) -> list[Finding]:
    """Run all metadata QC checks implemented through Phase 2E."""
    config = config or MetadataQCConfig()
    return [
        *detect_missing_metadata(metadata, config),
        *detect_metadata_duplicates(metadata),
        *detect_category_inconsistencies(metadata, aliases),
        *detect_metadata_structure(metadata, config),
    ]
