"""Configurable sample suspicion and dataset health scoring."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from biodetective.core.config import (
    DEFAULT_DATASET_HEALTH_DEDUCTIONS,
    DEFAULT_DATASET_RISK_FACTORS,
    DEFAULT_RISK_CRITICAL_MINIMUM,
    DEFAULT_RISK_HIGH_MINIMUM,
    DEFAULT_RISK_MODERATE_MINIMUM,
    DEFAULT_SAMPLE_SUSPICION_WEIGHTS,
    DEFAULT_SEVERITY_STRENGTH,
)
from biodetective.core.models import Finding


DEFAULT_SAMPLE_WEIGHTS = DEFAULT_SAMPLE_SUSPICION_WEIGHTS
DEFAULT_HEALTH_DEDUCTIONS = DEFAULT_DATASET_HEALTH_DEDUCTIONS
SEVERITY_STRENGTH = DEFAULT_SEVERITY_STRENGTH


@dataclass(frozen=True)
class SuspicionScoreConfig:
    """Weights and boundaries for sample-level suspicion scoring."""

    weights: dict[str, float] = field(default_factory=lambda: DEFAULT_SAMPLE_WEIGHTS.copy())
    moderate_minimum: float = DEFAULT_RISK_MODERATE_MINIMUM
    high_minimum: float = DEFAULT_RISK_HIGH_MINIMUM
    critical_minimum: float = DEFAULT_RISK_CRITICAL_MINIMUM

    def __post_init__(self) -> None:
        if any(weight < 0 for weight in self.weights.values()):
            raise ValueError("sample suspicion weights must be non-negative")
        if not 0 <= self.moderate_minimum <= self.high_minimum <= self.critical_minimum <= 100:
            raise ValueError("risk boundaries must be ordered between 0 and 100")


@dataclass(frozen=True)
class SampleSuspicionScore:
    sample_id: str
    score: float
    risk: str
    contributions: dict[str, float]


@dataclass(frozen=True)
class DatasetHealthConfig:
    """Maximum deduction per interpretable dataset-health source."""

    maximum_deductions: dict[str, float] = field(default_factory=lambda: DEFAULT_HEALTH_DEDUCTIONS.copy())
    suspicious_score_threshold: float = DEFAULT_RISK_HIGH_MINIMUM

    def __post_init__(self) -> None:
        if any(deduction < 0 for deduction in self.maximum_deductions.values()):
            raise ValueError("health deductions must be non-negative")
        if not 0 <= self.suspicious_score_threshold <= 100:
            raise ValueError("suspicious_score_threshold must be between 0 and 100")


@dataclass(frozen=True)
class DatasetHealthResult:
    score: float
    deductions: dict[str, dict[str, Any]]


def risk_label(score: float, config: SuspicionScoreConfig | None = None) -> str:
    """Map a score to configurable Low, Moderate, High, or Critical risk."""
    config = config or SuspicionScoreConfig()
    bounded_score = min(100.0, max(0.0, float(score)))
    if bounded_score >= config.critical_minimum:
        return "Critical"
    if bounded_score >= config.high_minimum:
        return "High"
    if bounded_score >= config.moderate_minimum:
        return "Moderate"
    return "Low"


def findings_to_sample_evidence(
    sample_ids: list[str],
    findings: list[Finding] | tuple[Finding, ...],
) -> dict[str, dict[str, float]]:
    """Convert existing Findings into normalized per-sample evidence strengths."""
    evidence = {str(sample_id): {} for sample_id in sample_ids}
    for finding in findings:
        sources: dict[str, float] = {}
        if finding.code == "combined_sample_outlier":
            if finding.evidence.get("pca_distance_triggered"):
                sources["pca_outlier"] = 1.0
            if finding.evidence.get("isolation_forest_triggered"):
                sources["isolation_forest"] = 1.0
        elif finding.code == "pca_distance_outlier":
            sources["pca_outlier"] = 1.0
        elif finding.code in {"highly_suspicious_similarity", "noteworthy_similarity"}:
            sources["duplicate_similarity"] = 1.0 if finding.code == "highly_suspicious_similarity" else 0.5
        elif finding.category == "label_consistency":
            sources["label_inconsistency"] = SEVERITY_STRENGTH[finding.severity]
        elif finding.category == "sex_marker_consistency":
            sources["sex_consistency"] = SEVERITY_STRENGTH[finding.severity]
        elif finding.category in {"missing_metadata", "metadata_duplicates", "category_consistency", "metadata_structure"}:
            sources["metadata_issues"] = SEVERITY_STRENGTH[finding.severity]

        for sample_id in finding.sample_ids:
            sample_key = str(sample_id)
            if sample_key not in evidence:
                continue
            for source, strength in sources.items():
                evidence[sample_key][source] = max(evidence[sample_key].get(source, 0.0), strength)
    return evidence


def score_samples(
    sample_ids: list[str],
    evidence_by_sample: dict[str, dict[str, float | bool]] | None = None,
    findings: list[Finding] | tuple[Finding, ...] = (),
    config: SuspicionScoreConfig | None = None,
) -> list[SampleSuspicionScore]:
    """Return bounded scores and a complete contribution breakdown per sample."""
    config = config or SuspicionScoreConfig()
    normalized_ids = [str(sample_id) for sample_id in sample_ids]
    finding_evidence = findings_to_sample_evidence(normalized_ids, findings)
    supplied_evidence = evidence_by_sample or {}
    results: list[SampleSuspicionScore] = []

    for sample_id in normalized_ids:
        combined_evidence = finding_evidence.get(sample_id, {}).copy()
        for source, value in supplied_evidence.get(sample_id, {}).items():
            strength = float(value)
            combined_evidence[source] = max(combined_evidence.get(source, 0.0), strength)

        contributions: dict[str, float] = {}
        for source, weight in config.weights.items():
            strength = min(1.0, max(0.0, float(combined_evidence.get(source, 0.0))))
            contributions[source] = round(float(weight) * strength, 4)
        score = round(min(100.0, sum(contributions.values())), 4)
        results.append(SampleSuspicionScore(sample_id, score, risk_label(score, config), contributions))
    return results


def _breakdown_entry(available: bool, deduction: float = 0.0, evidence: Any = None) -> dict[str, Any]:
    return {"available": available, "deduction": round(float(deduction), 4), "evidence": evidence}


def calculate_dataset_health(
    sample_scores: list[SampleSuspicionScore] | None = None,
    metadata: pd.DataFrame | None = None,
    duplicate_findings: list[Finding] | tuple[Finding, ...] | None = None,
    batch_risk: str | None = None,
    confounding_risk: str | None = None,
    expression_findings: list[Finding] | tuple[Finding, ...] | None = None,
    config: DatasetHealthConfig | None = None,
) -> DatasetHealthResult:
    """Calculate health from available analyses only, with every deduction shown."""
    config = config or DatasetHealthConfig()
    maximums = config.maximum_deductions
    deductions: dict[str, dict[str, Any]] = {}

    if sample_scores is None:
        deductions["suspicious_sample_percentage"] = _breakdown_entry(False)
    else:
        suspicious_count = sum(score.score >= config.suspicious_score_threshold for score in sample_scores)
        percentage = suspicious_count / len(sample_scores) if sample_scores else 0.0
        deduction = maximums.get("suspicious_sample_percentage", 0.0) * percentage
        deductions["suspicious_sample_percentage"] = _breakdown_entry(
            True, deduction, {"suspicious_count": suspicious_count, "sample_count": len(sample_scores), "fraction": percentage}
        )

    if metadata is None:
        deductions["metadata_completeness"] = _breakdown_entry(False)
    else:
        cell_count = metadata.size
        missing_fraction = float(metadata.isna().sum().sum() / cell_count) if cell_count else 0.0
        deduction = maximums.get("metadata_completeness", 0.0) * missing_fraction
        deductions["metadata_completeness"] = _breakdown_entry(
            True, deduction, {"missing_fraction": missing_fraction, "cell_count": cell_count}
        )

    if duplicate_findings is None:
        deductions["duplicate_risk"] = _breakdown_entry(False)
    else:
        sample_count = len(sample_scores) if sample_scores is not None else 0
        affected = {sample_id for finding in duplicate_findings for sample_id in finding.sample_ids}
        affected_fraction = len(affected) / sample_count if sample_count else 0.0
        deduction = maximums.get("duplicate_risk", 0.0) * min(1.0, affected_fraction)
        deductions["duplicate_risk"] = _breakdown_entry(
            True, deduction, {"affected_samples": len(affected), "affected_fraction": affected_fraction}
        )

    risk_factors = {
        **DEFAULT_DATASET_RISK_FACTORS,
    }
    for source, risk in (("batch_risk", batch_risk), ("confounding_risk", confounding_risk)):
        if risk is None:
            deductions[source] = _breakdown_entry(False)
        else:
            factor = risk_factors.get(risk, 0.0)
            deduction = maximums.get(source, 0.0) * factor
            deductions[source] = _breakdown_entry(True, deduction, {"risk": risk})

    if expression_findings is None:
        deductions["expression_qc"] = _breakdown_entry(False)
    else:
        maximum_strength = max((SEVERITY_STRENGTH[finding.severity] for finding in expression_findings), default=0.0)
        deduction = maximums.get("expression_qc", 0.0) * maximum_strength
        deductions["expression_qc"] = _breakdown_entry(
            True, deduction, {"finding_count": len(expression_findings), "maximum_severity_strength": maximum_strength}
        )

    total_deduction = sum(entry["deduction"] for entry in deductions.values())
    return DatasetHealthResult(round(max(0.0, min(100.0, 100.0 - total_deduction)), 4), deductions)
