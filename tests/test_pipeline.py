import numpy as np
import pandas as pd

from biodetective.analysis.pca import PCAConfig
from biodetective.core.models import BioDataset
from biodetective.core.pipeline import PipelineConfig, run_biodetective_pipeline


def make_dataset() -> BioDataset:
    rng = np.random.default_rng(42)
    sample_ids = [f"S{i:02d}" for i in range(1, 9)]
    genes = ["XIST", "RPS4Y1", "KDM5D", "DDX3Y", "G1", "G2", "G3", "G4"]
    values = rng.normal(size=(len(genes), len(sample_ids)))
    values[4:, 4:] += 4.0
    expression = pd.DataFrame(values, index=genes, columns=sample_ids)
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "condition": ["Healthy"] * 4 + ["Cancer"] * 4,
            "sex": ["Female", "Male"] * 4,
            "batch": ["Batch1", "Batch2"] * 4,
        }
    )
    return BioDataset(expression, metadata, "integration")


def test_pipeline_runs_required_modules_and_skips_unconfigured_optional_modules():
    result = run_biodetective_pipeline(make_dataset())

    for name in ("validation", "metadata_qc", "expression_qc", "similarity", "pca", "outliers", "scoring"):
        assert result.modules[name].status == "completed"
    for name in ("label_consistency", "sex_consistency", "batch_analysis", "confounding"):
        assert result.modules[name].status == "skipped"
    assert len(result.sample_scores) == 8
    assert result.dataset_health is not None
    assert 0 <= result.dataset_health.score <= 100


def test_pipeline_runs_configured_optional_modules_and_reports_progress():
    progress = []
    config = PipelineConfig(
        pca=PCAConfig(n_components=3),
        label_column="condition",
        sex_column="sex",
        batch_column="batch",
        biological_column="condition",
        technical_column="batch",
        metadata_mappings={"condition": "condition", "sex": "sex", "batch": "batch", "age": None},
    )
    result = run_biodetective_pipeline(
        make_dataset(),
        config,
        progress_callback=lambda name, current, total: progress.append((name, current, total)),
    )

    for name in ("label_consistency", "sex_consistency", "batch_analysis", "confounding"):
        assert result.modules[name].status == "completed"
    assert result.analysis_settings["metadata_mappings"]["age"] is None
    assert [entry[0] for entry in progress] == list(result.modules)
    assert progress[-1] == ("scoring", 11, 11)


def test_pipeline_does_not_crash_when_one_analysis_cannot_run():
    dataset = make_dataset()
    dataset.expression.iloc[0, 0] = np.nan

    result = run_biodetective_pipeline(dataset)

    assert result.modules["validation"].status == "completed"
    assert result.modules["expression_qc"].status == "completed"
    assert result.modules["pca"].status == "failed"
    assert result.modules["outliers"].status == "skipped"
    assert result.modules["scoring"].status == "completed"


def test_unavailable_configured_optional_module_is_skipped():
    config = PipelineConfig(label_column="does_not_exist")

    result = run_biodetective_pipeline(make_dataset(), config)

    assert result.modules["label_consistency"].status == "skipped"
    assert "does not contain" in result.modules["label_consistency"].message


def test_progress_callback_failure_does_not_stop_pipeline():
    def broken_callback(name, current, total):
        raise RuntimeError("UI disconnected")

    result = run_biodetective_pipeline(make_dataset(), progress_callback=broken_callback)

    assert result.modules["scoring"].status == "completed"
