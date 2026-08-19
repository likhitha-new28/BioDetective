from io import StringIO
import json

import numpy as np
import pandas as pd
import pytest

from biodetective.core.models import BioDataset
from biodetective.core.pipeline import PipelineConfig, run_biodetective_pipeline
from biodetective.reporting import (
    generate_analysis_json,
    generate_findings_csv,
    generate_html_report,
    generate_sample_scores_csv,
)


@pytest.fixture
def pipeline_result():
    rng = np.random.default_rng(123)
    sample_ids = [f"S{i:02d}" for i in range(1, 9)]
    expression = pd.DataFrame(
        rng.normal(size=(8, 8)),
        index=["XIST", "RPS4Y1", "KDM5D", "DDX3Y", "G1", "G2", "G3", "G4"],
        columns=sample_ids,
    )
    expression.loc[["G1", "G2", "G3", "G4"], sample_ids[4:]] += 5
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "condition": ["Healthy"] * 4 + ["Cancer"] * 4,
            "sex": ["Female", "Male"] * 4,
            "batch": ["Batch1", "Batch2"] * 4,
        }
    )
    dataset = BioDataset(expression, metadata, "<unsafe dataset>")
    return run_biodetective_pipeline(
        dataset,
        PipelineConfig(
            label_column="condition",
            sex_column="sex",
            batch_column="batch",
            biological_column="condition",
            technical_column="batch",
        ),
    )


def test_findings_csv_has_stable_columns_and_parseable_evidence(pipeline_result):
    csv_text = generate_findings_csv(pipeline_result)
    frame = pd.read_csv(StringIO(csv_text))

    assert list(frame.columns) == [
        "category",
        "code",
        "severity",
        "message",
        "sample_ids",
        "column",
        "evidence",
        "recommendation",
    ]
    if not frame.empty:
        assert isinstance(json.loads(frame.iloc[0]["evidence"]), dict)


def test_sample_scores_csv_contains_score_and_contribution_breakdown(pipeline_result):
    frame = pd.read_csv(StringIO(generate_sample_scores_csv(pipeline_result)))

    assert len(frame) == pipeline_result.dataset.n_samples
    assert {"sample_id", "score", "risk"}.issubset(frame.columns)
    assert "contribution_pca_outlier" in frame.columns
    assert "contribution_duplicate_similarity" in frame.columns


def test_analysis_json_is_valid_and_excludes_raw_expression_matrix(pipeline_result):
    payload = json.loads(generate_analysis_json(pipeline_result))

    assert payload["dataset_summary"]["samples"] == 8
    assert payload["dataset_summary"]["genes_features"] == 8
    assert payload["analysis_settings"]["label_column"] == "condition"
    assert len(payload["sample_scores"]) == 8
    if payload["findings"]:
        assert set(payload["findings"][0]["scientific_context"]) == {
            "observation",
            "evidence",
            "interpretation",
            "possible_explanations",
            "recommendation",
        }
    assert "expression" not in payload
    assert "expression_matrix" not in payload
    assert "loadings" not in payload["analyses"]["pca"]


@pytest.mark.parametrize(
    "heading",
    [
        "Dataset Summary",
        "Dataset Health Score",
        "Analysis Settings",
        "Top Findings",
        "High/Critical Risk Samples",
        "Metadata QC",
        "Expression QC",
        "Duplicates",
        "Outliers",
        "Label Consistency",
        "Batch Effects",
        "Confounding",
        "Methodology",
        "Limitations",
        "Disclaimer",
    ],
)
def test_html_report_contains_required_sections(pipeline_result, heading):
    assert f">{heading}<" in generate_html_report(pipeline_result)


def test_html_report_is_standalone_escapes_user_values_and_excludes_raw_matrix(pipeline_result):
    html = generate_html_report(pipeline_result)

    assert html.startswith("<!doctype html>")
    assert "&lt;unsafe dataset&gt;" in html
    assert "<unsafe dataset>" not in html
    assert "gene-by-sample" not in html
    assert all(label in html for label in ("Observation", "Evidence", "Interpretation", "Recommendation"))
