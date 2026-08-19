"""Interpretable sample-outlier detection in PCA space."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from biodetective.core.config import (
    DEFAULT_ISOLATION_FOREST_CONTAMINATION,
    DEFAULT_PCA_DISTANCE_PERCENTILE,
    DEFAULT_RANDOM_STATE,
)
from biodetective.core.models import Finding


@dataclass(frozen=True)
class OutlierConfig:
    """Thresholds and deterministic settings for outlier detection."""

    distance_percentile_threshold: float = DEFAULT_PCA_DISTANCE_PERCENTILE
    isolation_contamination: float | str = DEFAULT_ISOLATION_FOREST_CONTAMINATION
    random_state: int = DEFAULT_RANDOM_STATE

    def __post_init__(self) -> None:
        if not 0 < self.distance_percentile_threshold <= 100:
            raise ValueError("distance_percentile_threshold must be in (0, 100]")
        if self.isolation_contamination != "auto":
            contamination = float(self.isolation_contamination)
            if not 0 < contamination <= 0.5:
                raise ValueError("isolation_contamination must be 'auto' or in (0, 0.5]")


@dataclass(frozen=True)
class PCADistanceResult:
    scores: pd.DataFrame
    findings: tuple[Finding, ...]


@dataclass(frozen=True)
class IsolationForestResult:
    scores: pd.DataFrame


@dataclass(frozen=True)
class CombinedOutlierResult:
    results: pd.DataFrame
    findings: tuple[Finding, ...]


def _validate_coordinates(coordinates: pd.DataFrame) -> None:
    if coordinates.empty or coordinates.shape[1] == 0:
        raise ValueError("PCA coordinates must contain samples and components")
    if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
        raise ValueError("PCA coordinates must contain only finite values")


def detect_pca_distance_outliers(
    coordinates: pd.DataFrame,
    config: OutlierConfig | None = None,
) -> PCADistanceResult:
    """Detect samples far from a robust median center in standardized PCA space."""
    config = config or OutlierConfig()
    _validate_coordinates(coordinates)

    values = coordinates.to_numpy(dtype=float)
    center = np.median(values, axis=0)
    absolute_deviations = np.abs(values - center)
    mad = np.median(absolute_deviations, axis=0)
    fallback_scale = np.std(values, axis=0, ddof=0)
    scale = np.where(mad > 0, 1.4826 * mad, np.where(fallback_scale > 0, fallback_scale, 1.0))
    robust_coordinates = (values - center) / scale
    distances = np.sqrt(np.sum(robust_coordinates**2, axis=1))

    distance_series = pd.Series(distances, index=coordinates.index.map(str), name="distance_score")
    percentiles = distance_series.rank(method="average", pct=True) * 100
    triggered = percentiles.ge(config.distance_percentile_threshold)
    scores = pd.DataFrame(
        {
            "distance_score": distance_series,
            "percentile": percentiles,
            "is_distance_outlier": triggered,
        }
    )
    scores.index.name = "sample_id"

    findings: list[Finding] = []
    for sample_id in scores.index[triggered]:
        row = scores.loc[sample_id]
        findings.append(
            Finding(
                category="sample_outlier",
                code="pca_distance_outlier",
                severity="high",
                message=(
                    f"Sample {sample_id} has an unusually large robust PCA distance "
                    f"at the {row['percentile']:.1f} percentile."
                ),
                sample_ids=[str(sample_id)],
                evidence={
                    "distance_score": float(row["distance_score"]),
                    "percentile": float(row["percentile"]),
                    "percentile_threshold": config.distance_percentile_threshold,
                },
                recommendation="Review this sample's metadata and expression profile; distance alone does not prove an error.",
            )
        )
    return PCADistanceResult(scores=scores, findings=tuple(findings))


def detect_isolation_forest_outliers(
    coordinates: pd.DataFrame,
    config: OutlierConfig | None = None,
) -> IsolationForestResult:
    """Fit Isolation Forest on PCA components and return per-sample results."""
    config = config or OutlierConfig()
    _validate_coordinates(coordinates)

    model = IsolationForest(
        contamination=config.isolation_contamination,
        random_state=config.random_state,
    )
    values = coordinates.to_numpy(dtype=float)
    raw_predictions = model.fit_predict(values)
    anomaly_scores = -model.decision_function(values)
    scores = pd.DataFrame(
        {
            "anomaly_score": anomaly_scores,
            "outlier_prediction": raw_predictions == -1,
        },
        index=coordinates.index.map(str),
    )
    scores.index.name = "sample_id"
    return IsolationForestResult(scores=scores)


def combine_outlier_results(
    distance_result: PCADistanceResult,
    isolation_result: IsolationForestResult,
) -> CombinedOutlierResult:
    """Combine detector triggers into Normal, Review, or Suspicious statuses."""
    combined = distance_result.scores.join(isolation_result.scores, how="outer")
    if combined.isna().any().any():
        raise ValueError("distance and Isolation Forest results must contain the same samples")

    distance_triggered = combined["is_distance_outlier"].astype(bool)
    isolation_triggered = combined["outlier_prediction"].astype(bool)
    combined["status"] = np.select(
        [distance_triggered & isolation_triggered, distance_triggered | isolation_triggered],
        ["Suspicious", "Review"],
        default="Normal",
    )
    combined["explanation"] = np.select(
        [
            distance_triggered & isolation_triggered,
            distance_triggered,
            isolation_triggered,
        ],
        [
            "Triggered both robust PCA distance and Isolation Forest.",
            "Triggered robust PCA distance only.",
            "Triggered Isolation Forest only.",
        ],
        default="No outlier detector triggered.",
    )

    findings: list[Finding] = []
    for sample_id, row in combined.loc[combined["status"] != "Normal"].iterrows():
        status = str(row["status"])
        findings.append(
            Finding(
                category="sample_outlier",
                code="combined_sample_outlier",
                severity="high" if status == "Suspicious" else "medium",
                message=f"Sample {sample_id} has outlier status {status}: {row['explanation']}",
                sample_ids=[str(sample_id)],
                evidence={
                    "status": status,
                    "distance_score": float(row["distance_score"]),
                    "distance_percentile": float(row["percentile"]),
                    "pca_distance_triggered": bool(row["is_distance_outlier"]),
                    "anomaly_score": float(row["anomaly_score"]),
                    "isolation_forest_triggered": bool(row["outlier_prediction"]),
                },
                recommendation="Review the sample and detector evidence before deciding whether any correction is needed.",
            )
        )
    return CombinedOutlierResult(results=combined, findings=tuple(findings))


def analyze_outliers(
    coordinates: pd.DataFrame,
    config: OutlierConfig | None = None,
) -> CombinedOutlierResult:
    """Run both outlier detectors and combine their interpretable results."""
    config = config or OutlierConfig()
    distance_result = detect_pca_distance_outliers(coordinates, config)
    isolation_result = detect_isolation_forest_outliers(coordinates, config)
    return combine_outlier_results(distance_result, isolation_result)
