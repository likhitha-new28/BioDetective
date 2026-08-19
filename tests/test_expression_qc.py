import numpy as np
import pandas as pd

from biodetective.analysis.expression_qc import (
    ExpressionQCConfig,
    analyze_gene_variance,
    calculate_gene_variance,
    calculate_sample_statistics,
    detect_expression_issues,
    run_expression_qc,
)


def test_basic_expression_qc_detects_invalid_values_and_zero_variance_without_mutation():
    expression = pd.DataFrame(
        {
            "S01": [1.0, np.nan, np.inf, -np.inf, 5.0],
            "S02": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=["constant", "missing", "positive_inf", "negative_inf", "also_constant"],
    )
    original = expression.copy(deep=True)

    findings = detect_expression_issues(expression)
    codes = {finding.code for finding in findings}

    assert codes == {
        "missing_expression_values",
        "positive_infinity_values",
        "negative_infinity_values",
        "zero_variance_genes",
    }
    assert next(f for f in findings if f.code == "missing_expression_values").sample_ids == ["S01"]
    assert next(f for f in findings if f.code == "positive_infinity_values").evidence["positive_infinity_count"] == 1
    pd.testing.assert_frame_equal(expression, original)


def test_gene_variance_uses_population_variance_and_ignores_non_finite_values():
    expression = pd.DataFrame(
        {"S1": [1.0, 1.0], "S2": [3.0, np.inf], "S3": [5.0, 1.0]},
        index=["variable", "finite_constant"],
    )
    variances = calculate_gene_variance(expression)
    assert variances["variable"] == 8 / 3
    assert variances["finite_constant"] == 0


def test_variance_qc_returns_summary_and_low_variance_finding():
    expression = pd.DataFrame(
        {"S1": [1.0, 1.0, 1.0], "S2": [1.0, 1.1, 5.0], "S3": [1.0, 1.0, 9.0]},
        index=["zero", "very_low", "variable"],
    )

    result = analyze_gene_variance(expression, ExpressionQCConfig(low_variance_threshold=0.01))

    assert result.summary["gene_count"] == 3
    assert result.summary["zero_variance_count"] == 1
    assert result.summary["very_low_variance_count"] == 1
    assert result.findings[0].evidence["gene_ids"] == ["very_low"]


def test_low_variance_threshold_is_configurable():
    expression = pd.DataFrame({"S1": [1.0], "S2": [1.1], "S3": [1.0]}, index=["gene"])
    strict = analyze_gene_variance(expression, ExpressionQCConfig(low_variance_threshold=0.001))
    relaxed = analyze_gene_variance(expression, ExpressionQCConfig(low_variance_threshold=0.01))
    assert strict.findings == ()
    assert len(relaxed.findings) == 1


def test_sample_expression_statistics():
    expression = pd.DataFrame(
        {"S01": [0.0, 2.0, 4.0, np.nan], "S02": [1.0, 1.0, 1.0, 1.0]},
        index=["G1", "G2", "G3", "G4"],
    )

    statistics = calculate_sample_statistics(expression)

    assert statistics.index.tolist() == ["S01", "S02"]
    assert statistics.loc["S01", "mean"] == 2.0
    assert statistics.loc["S01", "median"] == 2.0
    assert statistics.loc["S01", "standard_deviation"] == 2.0
    assert statistics.loc["S01", "min"] == 0.0
    assert statistics.loc["S01", "max"] == 4.0
    assert statistics.loc["S01", "percentage_zeros"] == 25.0
    assert statistics.loc["S01", "missing_percentage"] == 25.0


def test_sample_statistics_support_empty_sample_axis():
    statistics = calculate_sample_statistics(pd.DataFrame(index=["G1", "G2"]))
    assert statistics.empty
    assert statistics.index.name == "sample_id"


def test_run_expression_qc_combines_outputs():
    expression = pd.DataFrame({"S1": [1.0, 1.0], "S2": [1.0, 1.1]}, index=["zero", "low"])
    findings, variance_result, statistics = run_expression_qc(expression)
    assert {finding.code for finding in findings} == {"zero_variance_genes", "very_low_variance_genes"}
    assert len(variance_result.variances) == 2
    assert statistics.shape == (2, 7)
