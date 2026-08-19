from biodetective.synthetic.benchmark import classification_metrics, run_synthetic_benchmark


def test_classification_metrics_counts_and_rates():
    metrics = classification_metrics(["S1", "S2", "S3"], ["S2", "S3", "S4"])

    assert metrics == {
        "true_positives": 2,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.6667,
        "recall": 0.6667,
        "f1": 0.6667,
    }


def test_synthetic_benchmark_is_deterministic_and_reports_requested_metrics():
    first = run_synthetic_benchmark(77)
    second = run_synthetic_benchmark(77)

    assert first == second
    assert first["dataset"] == {"genes": 300, "samples": 60}
    for problem in ("duplicates", "outliers", "label_swaps"):
        metrics = first["sample_level_metrics"][problem]
        assert set(metrics) == {
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
        }
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

    assert first["dataset_level_results"]["batch_effect"]["planted_strength"] == 3
    assert first["dataset_level_results"]["confounding"]["planted_mode"] == "partial"
