"""Association checks between a selected batch variable and PCA components."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from biodetective.core.config import (
    DEFAULT_BATCH_ALPHA,
    DEFAULT_BATCH_HIGH_EFFECT_SIZE,
    DEFAULT_BATCH_MIN_SAMPLES,
    DEFAULT_BATCH_MODERATE_EFFECT_SIZE,
)
from biodetective.core.models import Finding


@dataclass(frozen=True)
class BatchEffectConfig:
    """Thresholds used for batch association tests and risk summaries."""

    alpha: float = DEFAULT_BATCH_ALPHA
    moderate_effect_size: float = DEFAULT_BATCH_MODERATE_EFFECT_SIZE
    high_effect_size: float = DEFAULT_BATCH_HIGH_EFFECT_SIZE
    minimum_samples_per_batch: int = DEFAULT_BATCH_MIN_SAMPLES

    def __post_init__(self) -> None:
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 <= self.moderate_effect_size <= self.high_effect_size <= 1:
            raise ValueError("effect-size thresholds must be ordered between 0 and 1")
        if self.minimum_samples_per_batch < 2:
            raise ValueError("minimum_samples_per_batch must be at least 2")


@dataclass(frozen=True)
class BatchEffectResult:
    associations: pd.DataFrame
    batch_counts: dict[str, int]
    risk: str
    explanation: str
    evidence: dict[str, object]
    findings: tuple[Finding, ...]


def _align_batches(
    pca_coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must contain a 'sample_id' column")
    if batch_column not in metadata.columns:
        raise ValueError(f"metadata does not contain column '{batch_column}'")
    if metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("metadata sample IDs must be unique for batch analysis")
    if pca_coordinates.empty or pca_coordinates.shape[1] == 0:
        raise ValueError("PCA coordinates must contain samples and components")
    if not np.isfinite(pca_coordinates.to_numpy(dtype=float)).all():
        raise ValueError("PCA coordinates must contain finite values")

    batch_metadata = metadata.loc[metadata[batch_column].notna(), ["sample_id", batch_column]].copy()
    batch_metadata["sample_id"] = batch_metadata["sample_id"].astype(str)
    batch_metadata[batch_column] = batch_metadata[batch_column].astype(str)
    batch_metadata = batch_metadata.set_index("sample_id")
    coordinate_lookup = {str(sample_id): sample_id for sample_id in pca_coordinates.index}
    common_ids = [sample_id for sample_id in coordinate_lookup if sample_id in batch_metadata.index]
    if not common_ids:
        raise ValueError("PCA coordinates and metadata have no samples in common")
    original_ids = [coordinate_lookup[sample_id] for sample_id in common_ids]
    coordinates = pca_coordinates.loc[original_ids].copy(deep=True)
    coordinates.index = common_ids
    batches = batch_metadata.loc[common_ids, batch_column]
    return coordinates, batches


def _eta_squared(values: np.ndarray, labels: pd.Series) -> float:
    overall_mean = float(np.mean(values))
    total_sum_squares = float(np.sum((values - overall_mean) ** 2))
    if total_sum_squares == 0:
        return 0.0
    between_sum_squares = 0.0
    for label in labels.unique():
        group_values = values[labels.to_numpy() == label]
        between_sum_squares += len(group_values) * (float(np.mean(group_values)) - overall_mean) ** 2
    return float(between_sum_squares / total_sum_squares)


def analyze_batch_pca_association(
    pca_coordinates: pd.DataFrame,
    metadata: pd.DataFrame,
    batch_column: str,
    config: BatchEffectConfig | None = None,
) -> BatchEffectResult:
    """Test batch association per PCA component without correcting any data."""
    config = config or BatchEffectConfig()
    coordinates, batches = _align_batches(pca_coordinates, metadata, batch_column)
    batch_counts = {str(label): int(count) for label, count in batches.value_counts().sort_index().items()}
    if len(batch_counts) < 2:
        raise ValueError("batch analysis requires at least two batch groups")

    records: list[dict[str, object]] = []
    for component in coordinates.columns:
        values = coordinates[component].to_numpy(dtype=float)
        groups = [coordinates.loc[batches.eq(label), component].to_numpy(dtype=float) for label in sorted(batch_counts)]
        minimum_group_size = min(len(group) for group in groups)

        anova_statistic = anova_p_value = float("nan")
        if minimum_group_size >= 2 and any(np.var(group) > 0 for group in groups):
            anova = stats.f_oneway(*groups)
            anova_statistic = float(anova.statistic)
            anova_p_value = float(anova.pvalue)

        kruskal_statistic = kruskal_p_value = float("nan")
        try:
            kruskal = stats.kruskal(*groups)
            kruskal_statistic = float(kruskal.statistic)
            kruskal_p_value = float(kruskal.pvalue)
        except ValueError:
            pass

        levene_p_value = float("nan")
        if minimum_group_size >= 2:
            levene = stats.levene(*groups, center="median")
            levene_p_value = float(levene.pvalue)

        eta_squared = _eta_squared(values, batches)
        sample_count = len(values)
        group_count = len(groups)
        epsilon_squared = (
            max(0.0, (kruskal_statistic - group_count + 1) / (sample_count - group_count))
            if np.isfinite(kruskal_statistic) and sample_count > group_count
            else float("nan")
        )
        use_robust_test = minimum_group_size < config.minimum_samples_per_batch or (
            np.isfinite(levene_p_value) and levene_p_value < config.alpha
        )
        preferred_test = "Kruskal-Wallis" if use_robust_test else "ANOVA"
        preferred_p_value = kruskal_p_value if use_robust_test else anova_p_value
        effect_size = epsilon_squared if use_robust_test and np.isfinite(epsilon_squared) else eta_squared
        records.append(
            {
                "component": str(component),
                "preferred_test": preferred_test,
                "preferred_p_value": preferred_p_value,
                "effect_size": effect_size,
                "anova_statistic": anova_statistic,
                "anova_p_value": anova_p_value,
                "eta_squared": eta_squared,
                "kruskal_statistic": kruskal_statistic,
                "kruskal_p_value": kruskal_p_value,
                "epsilon_squared": epsilon_squared,
                "levene_p_value": levene_p_value,
                "statistically_associated": bool(np.isfinite(preferred_p_value) and preferred_p_value < config.alpha),
            }
        )

    associations = pd.DataFrame.from_records(records).set_index("component")
    maximum_effect = float(associations["effect_size"].max())
    strongest_component = str(associations["effect_size"].idxmax())
    associated_components = associations.index[associations["statistically_associated"]].astype(str).tolist()
    if maximum_effect >= config.high_effect_size:
        risk = "High"
    elif maximum_effect >= config.moderate_effect_size:
        risk = "Moderate"
    else:
        risk = "Low"

    explanation = (
        f"Batch Effect Risk is {risk}. The strongest association is on {strongest_component} "
        f"with effect size {maximum_effect:.3f}. Risk is based on effect magnitude, with statistical tests as supporting evidence."
    )
    evidence: dict[str, object] = {
        "batch_column": batch_column,
        "batch_counts": batch_counts,
        "maximum_effect_size": maximum_effect,
        "strongest_component": strongest_component,
        "statistically_associated_components": associated_components,
    }
    findings: list[Finding] = []
    if risk != "Low":
        findings.append(
            Finding(
                category="batch_effects",
                code="batch_pca_association",
                severity="high" if risk == "High" else "medium",
                message=f"PCA components show {risk.lower()} evidence of association with batch column '{batch_column}'.",
                column=batch_column,
                evidence=evidence,
                recommendation="Review study design and batch balance; BioDetective has not corrected the expression data.",
            )
        )
    return BatchEffectResult(associations, batch_counts, risk, explanation, evidence, tuple(findings))
