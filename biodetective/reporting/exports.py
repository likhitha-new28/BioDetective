"""CSV and JSON exports for completed BioDetective pipeline results."""

from dataclasses import asdict, is_dataclass
import json
import math
from typing import Any

import numpy as np
import pandas as pd

from biodetective.core.models import Finding
from biodetective.core.pipeline import PipelineResult
from biodetective.reporting.explanations import explain_finding


SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        frame = value.reset_index()
        return _json_safe(frame.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if pd.isna(value):
        return None
    return str(value)


def finding_to_record(finding: Finding) -> dict[str, Any]:
    """Convert a Finding to a stable, JSON-safe record."""
    explanation = explain_finding(finding)
    return {
        "category": finding.category,
        "code": finding.code,
        "severity": finding.severity,
        "message": finding.message,
        "sample_ids": list(finding.sample_ids),
        "column": finding.column,
        "evidence": _json_safe(finding.evidence),
        "recommendation": explanation.recommendation,
        "scientific_context": {
            "observation": explanation.observation,
            "evidence": _json_safe(explanation.evidence),
            "interpretation": explanation.interpretation,
            "possible_explanations": list(explanation.possible_explanations),
            "recommendation": explanation.recommendation,
        },
    }


def sorted_findings(findings: tuple[Finding, ...] | list[Finding]) -> list[Finding]:
    """Sort findings from critical to low while preserving order within severity."""
    return sorted(findings, key=lambda finding: SEVERITY_ORDER.get(finding.severity, 99))


def generate_findings_csv(result: PipelineResult) -> str:
    """Return findings.csv contents without including uploaded expression values."""
    records = []
    for finding in sorted_findings(result.findings):
        record = finding_to_record(finding)
        record["sample_ids"] = ";".join(record["sample_ids"])
        record["evidence"] = json.dumps(record["evidence"], sort_keys=True)
        records.append(record)
    columns = ["category", "code", "severity", "message", "sample_ids", "column", "evidence", "recommendation"]
    return pd.DataFrame(records, columns=columns).to_csv(index=False, lineterminator="\n")


def generate_sample_scores_csv(result: PipelineResult) -> str:
    """Return sample_scores.csv with one column per score contribution."""
    contribution_names = sorted(
        {source for score in result.sample_scores for source in score.contributions}
    )
    records = []
    for score in result.sample_scores:
        record: dict[str, Any] = {"sample_id": score.sample_id, "score": score.score, "risk": score.risk}
        record.update({f"contribution_{source}": score.contributions.get(source, 0.0) for source in contribution_names})
        records.append(record)
    columns = ["sample_id", "score", "risk", *[f"contribution_{name}" for name in contribution_names]]
    return pd.DataFrame(records, columns=columns).to_csv(index=False, lineterminator="\n")


def _completed_result(result: PipelineResult, module_name: str) -> Any | None:
    module = result.modules.get(module_name)
    return module.result if module is not None and module.status == "completed" else None


def build_analysis_payload(result: PipelineResult) -> dict[str, Any]:
    """Build a complete analysis payload while deliberately excluding raw expression data."""
    metadata_cells = result.dataset.metadata.size
    missing_metadata = int(result.dataset.metadata.isna().sum().sum())
    completeness = 100.0 * (1 - missing_metadata / metadata_cells) if metadata_cells else 100.0
    expression_qc = _completed_result(result, "expression_qc")
    similarity = _completed_result(result, "similarity")
    pca = _completed_result(result, "pca")
    outliers = _completed_result(result, "outliers")
    labels = _completed_result(result, "label_consistency")
    sex = _completed_result(result, "sex_consistency")
    batch = _completed_result(result, "batch_analysis")
    confounding = _completed_result(result, "confounding")

    analyses: dict[str, Any] = {
        "metadata_qc": {
            "finding_count": sum(finding.category.startswith("metadata") or finding.category in {"missing_metadata", "category_consistency"} for finding in result.findings),
        },
        "expression_qc": None,
        "similarity": None,
        "pca": None,
        "outliers": None,
        "label_consistency": None,
        "sex_consistency": None,
        "batch_effects": None,
        "confounding": None,
    }
    if expression_qc is not None:
        analyses["expression_qc"] = {
            "variance_summary": expression_qc[1].summary,
            "sample_statistics": expression_qc[2],
        }
    if similarity is not None:
        analyses["similarity"] = {
            "method": similarity[0].attrs.get("method"),
            "suspicious_pairs": [finding_to_record(finding) for finding in similarity[1]],
        }
    if pca is not None:
        analyses["pca"] = {
            "explained_variance": pca.explained_variance,
            "preprocessing": pca.preprocessing_config,
        }
    if outliers is not None:
        analyses["outliers"] = outliers.results
    if labels is not None:
        analyses["label_consistency"] = {
            "centroid_results": labels.centroid.results,
            "cross_validated_results": labels.cross_validated.results if labels.cross_validated is not None else None,
            "cross_validation_available": labels.cross_validated.available if labels.cross_validated is not None else False,
        }
    if sex is not None:
        analyses["sex_consistency"] = {"availability": sex.availability, "results": sex.results}
    if batch is not None:
        analyses["batch_effects"] = {
            "risk": batch.risk,
            "explanation": batch.explanation,
            "batch_counts": batch.batch_counts,
            "associations": batch.associations,
        }
    if confounding is not None:
        analyses["confounding"] = {
            "risk": confounding.risk,
            "cramers_v": confounding.cramers_v,
            "interpretation": confounding.interpretation,
            "contingency_table": confounding.contingency.contingency_table,
            "evidence": confounding.evidence,
        }

    payload = {
        "report": {"name": "BioDetective Analysis", "schema_version": "1.0"},
        "dataset_summary": {
            "name": result.dataset.name,
            "samples": result.dataset.n_samples,
            "genes_features": result.dataset.n_features,
            "metadata_columns": len(result.dataset.metadata_columns),
            "metadata_completeness_percentage": round(completeness, 4),
        },
        "dataset_health": result.dataset_health,
        "analysis_settings": result.analysis_settings,
        "module_statuses": {
            name: {"status": module.status, "message": module.message}
            for name, module in result.modules.items()
        },
        "findings": [finding_to_record(finding) for finding in sorted_findings(result.findings)],
        "sample_scores": list(result.sample_scores),
        "analyses": analyses,
    }
    return _json_safe(payload)


def generate_analysis_json(result: PipelineResult) -> str:
    """Return a standards-compliant, human-readable analysis.json document."""
    return json.dumps(build_analysis_payload(result), indent=2, sort_keys=True, allow_nan=False)
