"""Cautious molecular-profile checks against recorded metadata groups."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from biodetective.core.config import (
    DEFAULT_LABEL_CV_FOLDS,
    DEFAULT_LABEL_MIN_SAMPLES_PER_CLASS,
    DEFAULT_LABEL_SIMILARITY_MARGIN,
    DEFAULT_LOGISTIC_REGRESSION_C,
    DEFAULT_LOGISTIC_REGRESSION_MAX_ITERATIONS,
    DEFAULT_RANDOM_STATE,
)
from biodetective.core.models import Finding


@dataclass(frozen=True)
class LabelConsistencyConfig:
    """Thresholds and cross-validation settings for label consistency."""

    minimum_similarity_margin: float = DEFAULT_LABEL_SIMILARITY_MARGIN
    min_samples_per_class: int = DEFAULT_LABEL_MIN_SAMPLES_PER_CLASS
    cv_folds: int = DEFAULT_LABEL_CV_FOLDS
    logistic_c: float = DEFAULT_LOGISTIC_REGRESSION_C
    logistic_max_iterations: int = DEFAULT_LOGISTIC_REGRESSION_MAX_ITERATIONS
    random_state: int = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        if self.minimum_similarity_margin < 0:
            raise ValueError("minimum_similarity_margin must be non-negative")
        if self.min_samples_per_class < 2:
            raise ValueError("min_samples_per_class must be at least 2")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
        if self.logistic_c <= 0:
            raise ValueError("logistic_c must be positive")
        if self.logistic_max_iterations < 1:
            raise ValueError("logistic_max_iterations must be positive")


@dataclass(frozen=True)
class CentroidSimilarityResult:
    results: pd.DataFrame
    similarities: pd.DataFrame
    centroids: pd.DataFrame
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class CrossValidatedLabelResult:
    results: pd.DataFrame
    probabilities: pd.DataFrame
    findings: tuple[Finding, ...]
    available: bool
    reason: str | None = None


@dataclass(frozen=True)
class LabelConsistencyResult:
    centroid: CentroidSimilarityResult
    cross_validated: CrossValidatedLabelResult | None


def _aligned_profiles_and_labels(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    label_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if "sample_id" not in metadata.columns:
        raise ValueError("metadata must contain a 'sample_id' column")
    if label_column not in metadata.columns:
        raise ValueError(f"metadata does not contain column '{label_column}'")
    if metadata["sample_id"].astype(str).duplicated().any():
        raise ValueError("metadata sample IDs must be unique for label consistency analysis")
    if not all(pd.api.types.is_numeric_dtype(expression[column]) for column in expression.columns):
        raise ValueError("expression values must be numeric")

    metadata_subset = metadata.loc[metadata[label_column].notna(), ["sample_id", label_column]].copy()
    metadata_subset["sample_id"] = metadata_subset["sample_id"].astype(str)
    metadata_subset = metadata_subset.set_index("sample_id")

    expression_column_lookup = {str(column): column for column in expression.columns}
    common_sample_ids = [sample_id for sample_id in expression_column_lookup if sample_id in metadata_subset.index]
    if not common_sample_ids:
        raise ValueError("expression and metadata have no labeled samples in common")

    original_columns = [expression_column_lookup[sample_id] for sample_id in common_sample_ids]
    profiles = expression.loc[:, original_columns].T.copy(deep=True)
    profiles.index = common_sample_ids
    profiles.index.name = "sample_id"
    if not np.isfinite(profiles.to_numpy(dtype=float)).all():
        raise ValueError("label consistency requires finite expression values")

    labels = metadata_subset.loc[common_sample_ids, label_column].astype(str)
    labels.name = "recorded_group"
    return profiles, labels


def _profile_centroid_correlations(profiles: pd.DataFrame, centroids: pd.DataFrame) -> pd.DataFrame:
    profile_values = profiles.to_numpy(dtype=float)
    centroid_values = centroids.to_numpy(dtype=float)
    centered_profiles = profile_values - profile_values.mean(axis=1, keepdims=True)
    centered_centroids = centroid_values - centroid_values.mean(axis=1, keepdims=True)
    numerator = centered_profiles @ centered_centroids.T
    denominator = np.linalg.norm(centered_profiles, axis=1)[:, None] * np.linalg.norm(centered_centroids, axis=1)[None, :]
    correlations = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )
    return pd.DataFrame(correlations, index=profiles.index, columns=centroids.index)


def analyze_centroid_similarity(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    label_column: str,
    config: LabelConsistencyConfig | None = None,
) -> CentroidSimilarityResult:
    """Compare each sample profile with every recorded category centroid."""
    config = config or LabelConsistencyConfig()
    profiles, labels = _aligned_profiles_and_labels(expression, metadata, label_column)
    if labels.nunique() < 2:
        raise ValueError("centroid similarity requires at least two metadata groups")

    centroids = profiles.groupby(labels, sort=True).mean()
    centroids.index.name = label_column
    similarities = _profile_centroid_correlations(profiles, centroids)
    similarities.columns = similarities.columns.map(str)
    similarities.columns.name = "molecular_group"

    rows: list[dict[str, object]] = []
    findings: list[Finding] = []
    for sample_id in profiles.index:
        sample_similarities = similarities.loc[sample_id]
        if sample_similarities.notna().sum() == 0:
            closest_group = None
            closest_similarity = float("nan")
        else:
            closest_group = str(sample_similarities.idxmax())
            closest_similarity = float(sample_similarities.max())
        recorded_group = str(labels.loc[sample_id])
        recorded_similarity = (
            float(sample_similarities[recorded_group])
            if recorded_group in sample_similarities.index and pd.notna(sample_similarities[recorded_group])
            else float("nan")
        )
        similarity_margin = closest_similarity - recorded_similarity
        appears_closer_elsewhere = bool(
            closest_group is not None
            and closest_group != recorded_group
            and np.isfinite(similarity_margin)
            and similarity_margin >= config.minimum_similarity_margin
        )
        similarity_values = {
            str(group): (float(value) if pd.notna(value) else None)
            for group, value in sample_similarities.items()
        }
        rows.append(
            {
                "sample_id": str(sample_id),
                "recorded_group": recorded_group,
                "molecular_closest_group": closest_group,
                "recorded_group_similarity": recorded_similarity,
                "closest_group_similarity": closest_similarity,
                "similarity_margin": similarity_margin,
                "appears_more_similar_to_another_group": appears_closer_elsewhere,
                "similarity_values": similarity_values,
            }
        )

        if appears_closer_elsewhere:
            findings.append(
                Finding(
                    category="label_consistency",
                    code="molecular_profile_closer_to_another_group",
                    severity="medium",
                    message=(
                        f"Sample {sample_id}: molecular profile appears more similar to another metadata group "
                        f"('{closest_group}') than its recorded '{label_column}' group ('{recorded_group}')."
                    ),
                    sample_ids=[str(sample_id)],
                    column=label_column,
                    evidence={
                        "recorded_group": recorded_group,
                        "molecular_closest_group": closest_group,
                        "recorded_group_similarity": recorded_similarity,
                        "closest_group_similarity": closest_similarity,
                        "similarity_margin": float(similarity_margin),
                        "similarity_values": similarity_values,
                    },
                    recommendation="Review the sample and metadata provenance; molecular similarity does not establish the correct label.",
                )
            )

    results = pd.DataFrame.from_records(rows).set_index("sample_id")
    return CentroidSimilarityResult(results, similarities, centroids, tuple(findings))


def cross_validated_label_consistency(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    label_column: str,
    config: LabelConsistencyConfig | None = None,
) -> CrossValidatedLabelResult:
    """Generate out-of-fold Logistic Regression predictions without leakage."""
    config = config or LabelConsistencyConfig()
    profiles, labels = _aligned_profiles_and_labels(expression, metadata, label_column)
    class_counts = labels.value_counts()
    empty_results = pd.DataFrame(
        columns=["recorded_class", "cross_validated_predicted_class", "confidence", "prediction_matches_recorded"]
    )
    empty_results.index.name = "sample_id"

    if len(class_counts) < 2:
        return CrossValidatedLabelResult(
            empty_results,
            pd.DataFrame(index=profiles.index),
            (),
            False,
            "Cross-validation requires at least two classes.",
        )
    if int(class_counts.min()) < config.min_samples_per_class:
        return CrossValidatedLabelResult(
            empty_results,
            pd.DataFrame(index=profiles.index),
            (),
            False,
            f"Each class requires at least {config.min_samples_per_class} samples.",
        )

    n_splits = min(config.cv_folds, int(class_counts.min()))
    classifier = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    C=config.logistic_c,
                    max_iter=config.logistic_max_iterations,
                    random_state=config.random_state,
                ),
            ),
        ]
    )
    cross_validation = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.random_state)
    probabilities_array = cross_val_predict(
        classifier,
        profiles.to_numpy(dtype=float),
        labels.to_numpy(),
        cv=cross_validation,
        method="predict_proba",
    )
    classes = np.sort(labels.unique())
    probabilities = pd.DataFrame(probabilities_array, index=profiles.index, columns=classes)
    probabilities.index.name = "sample_id"
    predicted = probabilities.idxmax(axis=1).astype(str)
    confidence = probabilities.max(axis=1)
    results = pd.DataFrame(
        {
            "recorded_class": labels,
            "cross_validated_predicted_class": predicted,
            "confidence": confidence,
            "prediction_matches_recorded": predicted.eq(labels),
        }
    )
    results.index.name = "sample_id"

    findings: list[Finding] = []
    for sample_id in results.index[~results["prediction_matches_recorded"]]:
        row = results.loc[sample_id]
        probability_values = {str(group): float(value) for group, value in probabilities.loc[sample_id].items()}
        findings.append(
            Finding(
                category="label_consistency",
                code="cross_validated_label_disagreement",
                severity="medium",
                message=(
                    f"Sample {sample_id}: a cross-validated expression model favors group "
                    f"'{row['cross_validated_predicted_class']}' over recorded group '{row['recorded_class']}'."
                ),
                sample_ids=[str(sample_id)],
                column=label_column,
                evidence={
                    "recorded_class": str(row["recorded_class"]),
                    "cross_validated_predicted_class": str(row["cross_validated_predicted_class"]),
                    "confidence": float(row["confidence"]),
                    "class_probabilities": probability_values,
                    "cv_folds": n_splits,
                },
                recommendation="Treat this as review evidence only and verify the sample and metadata independently.",
            )
        )

    return CrossValidatedLabelResult(results, probabilities, tuple(findings), True)


def analyze_label_consistency(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    label_column: str,
    config: LabelConsistencyConfig | None = None,
    include_classifier: bool = True,
) -> LabelConsistencyResult:
    """Run centroid comparison and optionally cross-validated classification."""
    config = config or LabelConsistencyConfig()
    centroid_result = analyze_centroid_similarity(expression, metadata, label_column, config)
    classification_result = (
        cross_validated_label_consistency(expression, metadata, label_column, config)
        if include_classifier
        else None
    )
    return LabelConsistencyResult(centroid_result, classification_result)
