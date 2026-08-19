"""Deterministic comparisons between planted and detected data-quality issues."""

from typing import Iterable

from biodetective.analysis.pca import PCAConfig
from biodetective.core.models import BioDataset
from biodetective.core.pipeline import PipelineConfig, run_biodetective_pipeline
from biodetective.synthetic.generator import generate_synthetic_dataset


def classification_metrics(
    planted_ids: Iterable[str],
    detected_ids: Iterable[str],
) -> dict[str, float | int]:
    """Calculate sample-level detection metrics from two ID collections."""
    planted = {str(sample_id) for sample_id in planted_ids}
    detected = {str(sample_id) for sample_id in detected_ids}
    true_positives = len(planted & detected)
    false_positives = len(detected - planted)
    false_negatives = len(planted - detected)
    precision = true_positives / (true_positives + false_positives) if detected else 0.0
    recall = true_positives / (true_positives + false_negatives) if planted else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def run_synthetic_benchmark(random_state: int = 2026) -> dict[str, object]:
    """Run one fixed benchmark without changing detector thresholds to fit it."""
    expression, metadata, truth = generate_synthetic_dataset(
        n_genes=300,
        n_samples=60,
        n_conditions=3,
        n_batches=3,
        random_state=random_state,
        n_exact_duplicates=2,
        n_near_duplicates=2,
        n_outliers=3,
        n_label_swaps=3,
        batch_effect_strength=3.0,
        confounding="partial",
        confounding_strength=0.75,
    )
    result = run_biodetective_pipeline(
        BioDataset(expression, metadata, name=f"synthetic-benchmark-{random_state}"),
        PipelineConfig(
            pca=PCAConfig(remove_zero_variance_genes=True, top_variable_genes=200, n_components=5),
            label_column="condition",
            batch_column="batch",
            biological_column="condition",
            technical_column="batch",
        ),
    )

    detected_duplicates = {
        sample_id
        for finding in result.findings
        if finding.category == "sample_similarity"
        for sample_id in finding.sample_ids
    }
    detected_outliers = {
        sample_id
        for finding in result.findings
        if finding.category == "sample_outlier" and finding.code == "combined_sample_outlier"
        for sample_id in finding.sample_ids
    }
    detected_label_issues = {
        sample_id
        for finding in result.findings
        if finding.category == "label_consistency"
        for sample_id in finding.sample_ids
    }
    planted_duplicates = {
        *truth["exact_duplicates"]["affected_sample_ids"],
        *truth["near_duplicates"]["affected_sample_ids"],
    }
    batch_module = result.modules["batch_analysis"]
    confounding_module = result.modules["confounding"]
    return {
        "random_state": random_state,
        "dataset": {"genes": expression.shape[0], "samples": expression.shape[1]},
        "sample_level_metrics": {
            "duplicates": classification_metrics(planted_duplicates, detected_duplicates),
            "outliers": classification_metrics(truth["outliers"]["affected_sample_ids"], detected_outliers),
            "label_swaps": classification_metrics(
                truth["label_swaps"]["affected_sample_ids"], detected_label_issues
            ),
        },
        "dataset_level_results": {
            "batch_effect": {
                "planted_strength": truth["batch_effect"]["strength"],
                "detected_risk": batch_module.result.risk if batch_module.status == "completed" else None,
                "status": batch_module.status,
            },
            "confounding": {
                "planted_mode": truth["confounding"]["mode"],
                "detected_risk": confounding_module.result.risk if confounding_module.status == "completed" else None,
                "status": confounding_module.status,
            },
        },
    }
