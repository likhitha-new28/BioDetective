import numpy as np
import pandas as pd

from biodetective.analysis.outliers import (
    IsolationForestResult,
    OutlierConfig,
    PCADistanceResult,
    analyze_outliers,
    combine_outlier_results,
    detect_isolation_forest_outliers,
    detect_pca_distance_outliers,
)


def planted_outlier_coordinates():
    rng = np.random.default_rng(42)
    normal = rng.normal(0, 0.3, size=(20, 3))
    values = np.vstack([normal, [8.0, 8.0, 8.0]])
    return pd.DataFrame(values, index=[f"S{i:02d}" for i in range(1, 22)], columns=["PC1", "PC2", "PC3"])


def test_robust_pca_distance_detects_planted_outlier():
    coordinates = planted_outlier_coordinates()
    original = coordinates.copy(deep=True)
    config = OutlierConfig(distance_percentile_threshold=95, isolation_contamination=0.05)

    result = detect_pca_distance_outliers(coordinates, config)

    assert result.scores.loc["S21", "is_distance_outlier"]
    assert result.scores.loc["S21", "percentile"] == 100
    assert any(finding.sample_ids == ["S21"] for finding in result.findings)
    pd.testing.assert_frame_equal(coordinates, original)


def test_distance_threshold_is_configurable():
    coordinates = planted_outlier_coordinates()
    strict = detect_pca_distance_outliers(coordinates, OutlierConfig(distance_percentile_threshold=100))
    relaxed = detect_pca_distance_outliers(coordinates, OutlierConfig(distance_percentile_threshold=90))
    assert strict.scores["is_distance_outlier"].sum() == 1
    assert relaxed.scores["is_distance_outlier"].sum() > 1


def test_isolation_forest_detects_planted_outlier_and_is_deterministic():
    coordinates = planted_outlier_coordinates()
    config = OutlierConfig(isolation_contamination=0.05, random_state=7)
    first = detect_isolation_forest_outliers(coordinates, config)
    second = detect_isolation_forest_outliers(coordinates, config)

    assert first.scores.loc["S21", "outlier_prediction"]
    assert first.scores.loc["S21", "anomaly_score"] == second.scores.loc["S21", "anomaly_score"]
    pd.testing.assert_frame_equal(first.scores, second.scores)


def test_combined_result_statuses_and_explanations():
    index = pd.Index(["both", "distance", "isolation", "normal"], name="sample_id")
    distance = PCADistanceResult(
        scores=pd.DataFrame(
            {
                "distance_score": [10.0, 8.0, 1.0, 1.0],
                "percentile": [100.0, 99.0, 20.0, 10.0],
                "is_distance_outlier": [True, True, False, False],
            },
            index=index,
        ),
        findings=(),
    )
    isolation = IsolationForestResult(
        scores=pd.DataFrame(
            {"anomaly_score": [0.5, -0.1, 0.4, -0.2], "outlier_prediction": [True, False, True, False]},
            index=index,
        )
    )

    result = combine_outlier_results(distance, isolation)

    assert result.results["status"].to_dict() == {
        "both": "Suspicious",
        "distance": "Review",
        "isolation": "Review",
        "normal": "Normal",
    }
    assert "both" in result.results.loc["both", "explanation"]
    assert len(result.findings) == 3


def test_analyze_outliers_returns_interpretable_per_sample_result():
    result = analyze_outliers(
        planted_outlier_coordinates(),
        OutlierConfig(distance_percentile_threshold=95, isolation_contamination=0.05),
    )
    assert result.results.loc["S21", "status"] == "Suspicious"
    assert result.results.loc["S21", "explanation"] == "Triggered both robust PCA distance and Isolation Forest."
