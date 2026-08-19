import numpy as np
import pandas as pd

from biodetective.analysis.expression_qc import run_expression_qc
from biodetective.analysis.similarity import analyze_sample_similarity
from biodetective.core.models import BioDataset
from biodetective.core.pipeline import PipelineConfig, run_biodetective_pipeline
from biodetective.io.validators import validate_dataset


def _metadata(sample_ids, **columns):
    return pd.DataFrame({"sample_id": sample_ids, **columns})


def _pipeline_completes(dataset, config=None):
    result = run_biodetective_pipeline(dataset, config)
    assert result.modules["scoring"].status == "completed"
    assert len(result.sample_scores) == dataset.n_samples
    return result


def test_empty_dataset_is_reported_without_pipeline_crash():
    dataset = BioDataset(pd.DataFrame(), pd.DataFrame())

    validation = validate_dataset(dataset)
    result = _pipeline_completes(dataset)

    assert not validation.is_valid
    assert {issue.code for issue in validation.issues} >= {
        "missing_expression_samples",
        "missing_sample_id",
        "empty_expression",
        "empty_metadata",
    }
    assert result.modules["pca"].status == "failed"
    assert result.modules["outliers"].status == "skipped"


def test_one_sample_and_one_gene_are_handled_gracefully():
    one_sample = BioDataset(
        pd.DataFrame({"S1": [1.0, 2.0, 3.0]}, index=["G1", "G2", "G3"]),
        _metadata(["S1"], condition=["Control"]),
    )
    one_gene = BioDataset(
        pd.DataFrame([[1.0, 2.0, 3.0]], index=["G1"], columns=["S1", "S2", "S3"]),
        _metadata(["S1", "S2", "S3"], condition=["Control", "Control", "Case"]),
    )

    one_sample_result = _pipeline_completes(one_sample, PipelineConfig(label_column="condition"))
    one_gene_result = _pipeline_completes(one_gene, PipelineConfig(label_column="condition"))

    assert one_sample_result.modules["similarity"].result[0].shape == (1, 1)
    assert one_sample_result.modules["pca"].status == "failed"
    assert one_gene_result.modules["similarity"].status == "completed"
    assert one_gene_result.modules["pca"].status == "failed"


def test_one_metadata_class_skips_optional_label_analysis():
    expression = pd.DataFrame(
        np.arange(24, dtype=float).reshape(6, 4),
        index=[f"G{i}" for i in range(6)],
        columns=[f"S{i}" for i in range(4)],
    )
    dataset = BioDataset(expression, _metadata(expression.columns, condition=["Control"] * 4))

    result = _pipeline_completes(dataset, PipelineConfig(label_column="condition"))

    assert result.modules["label_consistency"].status == "skipped"
    assert "at least two metadata groups" in result.modules["label_consistency"].message


def test_missing_and_duplicate_sample_ids_are_validated():
    expression = pd.DataFrame([[1.0, 2.0]], index=["G1"], columns=["S1", "S2"])
    missing = BioDataset(expression, _metadata(["S1", None], condition=["A", "B"]))
    duplicated = BioDataset(expression, _metadata(["S1", "S1"], condition=["A", "B"]))

    missing_codes = {issue.code for issue in validate_dataset(missing).issues}
    duplicate_codes = {issue.code for issue in validate_dataset(duplicated).issues}

    assert {"missing_metadata_sample_ids", "sample_id_mismatch"} <= missing_codes
    assert {"duplicate_sample_ids", "sample_id_mismatch"} <= duplicate_codes
    _pipeline_completes(missing)
    _pipeline_completes(duplicated)


def test_missing_and_duplicate_expression_sample_ids_are_validated():
    missing_expression = pd.DataFrame([[1.0, 2.0]], index=["G1"], columns=["S1", ""])
    duplicate_expression = pd.DataFrame([[1.0, 2.0]], index=["G1"], columns=["S1", "S1"])

    missing_result = validate_dataset(BioDataset(missing_expression, _metadata(["S1", "S2"])))
    duplicate_result = validate_dataset(BioDataset(duplicate_expression, _metadata(["S1"])))

    assert "missing_expression_sample_ids" in {issue.code for issue in missing_result.issues}
    assert "duplicate_expression_sample_ids" in {issue.code for issue in duplicate_result.issues}


def test_duplicated_genes_are_validated_without_modification():
    expression = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["G1", "G1"], columns=["S1", "S2"])
    dataset = BioDataset(expression, _metadata(["S1", "S2"]))

    validation = validate_dataset(dataset)

    assert "duplicate_gene_ids" in {issue.code for issue in validation.issues}
    assert dataset.feature_ids == ["G1", "G1"]
    _pipeline_completes(dataset)


def test_all_zero_expression_reports_zero_variance_and_skips_dependent_pca():
    expression = pd.DataFrame(0.0, index=["G1", "G2"], columns=["S1", "S2", "S3"])
    dataset = BioDataset(expression, _metadata(expression.columns))

    result = _pipeline_completes(dataset)
    expression_findings = result.modules["expression_qc"].result[0]

    assert any(finding.code == "zero_variance_genes" for finding in expression_findings)
    assert result.modules["pca"].status == "failed"
    assert result.modules["outliers"].status == "skipped"


def test_all_identical_samples_are_flagged_as_highly_similar():
    expression = pd.DataFrame(
        {"S1": [1.0, 2.0, 4.0], "S2": [1.0, 2.0, 4.0], "S3": [1.0, 2.0, 4.0]},
        index=["G1", "G2", "G3"],
    )
    matrix, findings = analyze_sample_similarity(expression)

    assert matrix.shape == (3, 3)
    assert len(findings) == 3
    assert all(finding.code == "highly_suspicious_similarity" for finding in findings)
    _pipeline_completes(BioDataset(expression, _metadata(expression.columns)))


def test_nan_and_infinity_are_reported_and_do_not_reach_pca():
    expression = pd.DataFrame(
        {"S1": [1.0, np.nan, np.inf], "S2": [2.0, 3.0, -np.inf]},
        index=["G1", "G2", "G3"],
    )
    dataset = BioDataset(expression, _metadata(expression.columns))

    findings, _, _ = run_expression_qc(expression)
    result = _pipeline_completes(dataset)
    codes = {finding.code for finding in findings}

    assert {"missing_expression_values", "positive_infinity_values", "negative_infinity_values"} <= codes
    assert result.modules["pca"].status == "failed"
    assert result.modules["outliers"].status == "skipped"


def test_tiny_classes_leave_cross_validated_label_result_unavailable():
    rng = np.random.default_rng(10)
    expression = pd.DataFrame(
        rng.normal(size=(20, 6)),
        index=[f"G{i}" for i in range(20)],
        columns=[f"S{i}" for i in range(6)],
    )
    metadata = _metadata(expression.columns, condition=["Rare", "Common", "Common", "Common", "Common", "Common"])

    result = _pipeline_completes(BioDataset(expression, metadata), PipelineConfig(label_column="condition"))
    label_result = result.modules["label_consistency"].result

    assert result.modules["label_consistency"].status == "completed"
    assert label_result.cross_validated.available is False
    assert "at least" in label_result.cross_validated.reason


def test_missing_optional_metadata_marks_optional_modules_skipped():
    expression = pd.DataFrame(
        np.arange(40, dtype=float).reshape(10, 4),
        index=[f"G{i}" for i in range(10)],
        columns=[f"S{i}" for i in range(4)],
    )
    dataset = BioDataset(expression, _metadata(expression.columns))

    result = _pipeline_completes(dataset)

    for module in ("label_consistency", "sex_consistency", "batch_analysis", "confounding"):
        assert result.modules[module].status == "skipped"
