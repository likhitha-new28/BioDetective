"""Expression-matrix quality checks and descriptive statistics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from biodetective.core.config import DEFAULT_EXPRESSION_LOW_VARIANCE_THRESHOLD
from biodetective.core.models import Finding


@dataclass(frozen=True)
class ExpressionQCConfig:
    """Configurable thresholds for expression quality checks."""

    low_variance_threshold: float = DEFAULT_EXPRESSION_LOW_VARIANCE_THRESHOLD


@dataclass(frozen=True)
class VarianceQCResult:
    """Gene-level variances, summary statistics, and related findings."""

    variances: pd.Series
    summary: dict[str, float | int]
    findings: tuple[Finding, ...]


def _gene_ids(mask: pd.Series) -> list[str]:
    return [str(gene_id) for gene_id in mask.index[mask].tolist()]


def _affected_samples(mask: pd.DataFrame) -> list[str]:
    return [str(column) for column in mask.columns[mask.any(axis=0)].tolist()]


def _finite_expression(expression: pd.DataFrame) -> pd.DataFrame:
    values = expression.to_numpy(copy=False)
    try:
        has_infinity = bool(np.isinf(values).any())
    except TypeError:
        has_infinity = bool(expression.isin([np.inf, -np.inf]).to_numpy().any())
    return expression.replace([np.inf, -np.inf], np.nan) if has_infinity else expression


def calculate_gene_variance(expression: pd.DataFrame) -> pd.Series:
    """Calculate population variance per gene without changing expression."""
    finite_expression = _finite_expression(expression)
    variances = finite_expression.var(axis=1, skipna=True, ddof=0)
    variances.name = "variance"
    return variances


def detect_expression_issues(
    expression: pd.DataFrame,
    gene_variances: pd.Series | None = None,
) -> list[Finding]:
    """Detect missing, infinite, and zero-variance expression values."""
    findings: list[Finding] = []

    missing_mask = expression.isna()
    missing_count = int(missing_mask.sum().sum())
    if missing_count:
        genes = _gene_ids(missing_mask.any(axis=1))
        findings.append(
            Finding(
                category="expression_quality",
                code="missing_expression_values",
                severity="medium",
                message=f"Expression data contains {missing_count} missing value(s).",
                sample_ids=_affected_samples(missing_mask),
                evidence={"missing_count": missing_count, "gene_ids": genes, "gene_count": len(genes)},
                recommendation="Review the affected measurements and select an appropriate missing-data strategy before analysis.",
            )
        )

    positive_infinity_mask = expression.eq(np.inf)
    positive_infinity_count = int(positive_infinity_mask.sum().sum())
    if positive_infinity_count:
        genes = _gene_ids(positive_infinity_mask.any(axis=1))
        findings.append(
            Finding(
                category="expression_quality",
                code="positive_infinity_values",
                severity="high",
                message=f"Expression data contains {positive_infinity_count} positive infinity value(s).",
                sample_ids=_affected_samples(positive_infinity_mask),
                evidence={"positive_infinity_count": positive_infinity_count, "gene_ids": genes, "gene_count": len(genes)},
                recommendation="Trace infinite values to their source and correct the calculation or import before analysis.",
            )
        )

    negative_infinity_mask = expression.eq(-np.inf)
    negative_infinity_count = int(negative_infinity_mask.sum().sum())
    if negative_infinity_count:
        genes = _gene_ids(negative_infinity_mask.any(axis=1))
        findings.append(
            Finding(
                category="expression_quality",
                code="negative_infinity_values",
                severity="high",
                message=f"Expression data contains {negative_infinity_count} negative infinity value(s).",
                sample_ids=_affected_samples(negative_infinity_mask),
                evidence={"negative_infinity_count": negative_infinity_count, "gene_ids": genes, "gene_count": len(genes)},
                recommendation="Trace infinite values to their source and correct the calculation or import before analysis.",
            )
        )

    variances = gene_variances if gene_variances is not None else calculate_gene_variance(expression)
    zero_variance_mask = variances.eq(0)
    if zero_variance_mask.any():
        genes = _gene_ids(zero_variance_mask)
        findings.append(
            Finding(
                category="expression_quality",
                code="zero_variance_genes",
                severity="low",
                message=f"Expression data contains {len(genes)} zero-variance gene(s).",
                evidence={"gene_ids": genes, "gene_count": len(genes)},
                recommendation="Review zero-variance genes before downstream modeling; BioDetective has not removed them.",
            )
        )

    return findings


def analyze_gene_variance(
    expression: pd.DataFrame,
    config: ExpressionQCConfig | None = None,
) -> VarianceQCResult:
    """Summarize gene variance and flag non-zero genes below the threshold."""
    config = config or ExpressionQCConfig()
    variances = calculate_gene_variance(expression)
    valid_variances = variances.dropna()

    summary: dict[str, float | int] = {
        "gene_count": int(len(variances)),
        "valid_variance_count": int(len(valid_variances)),
        "missing_variance_count": int(variances.isna().sum()),
        "zero_variance_count": int(variances.eq(0).sum()),
        "mean_variance": float(valid_variances.mean()) if not valid_variances.empty else float("nan"),
        "median_variance": float(valid_variances.median()) if not valid_variances.empty else float("nan"),
        "minimum_variance": float(valid_variances.min()) if not valid_variances.empty else float("nan"),
        "maximum_variance": float(valid_variances.max()) if not valid_variances.empty else float("nan"),
    }

    low_variance_mask = variances.gt(0) & variances.le(config.low_variance_threshold)
    low_variance_genes = _gene_ids(low_variance_mask)
    summary["very_low_variance_count"] = len(low_variance_genes)

    findings: list[Finding] = []
    if low_variance_genes:
        findings.append(
            Finding(
                category="expression_variance",
                code="very_low_variance_genes",
                severity="low",
                message=(
                    f"{len(low_variance_genes)} gene(s) have non-zero variance at or below "
                    f"{config.low_variance_threshold:g}."
                ),
                evidence={
                    "gene_ids": low_variance_genes,
                    "gene_count": len(low_variance_genes),
                    "variance_threshold": config.low_variance_threshold,
                },
                recommendation="Review low-variance genes for relevance before modeling; they were not removed.",
            )
        )

    return VarianceQCResult(variances=variances, summary=summary, findings=tuple(findings))


def calculate_sample_statistics(expression: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive expression statistics for every sample.

    Infinite values are excluded from descriptive calculations, while the
    missing percentage reflects only values that were originally missing.
    """
    feature_count = len(expression)
    columns = [
        "mean",
        "median",
        "standard_deviation",
        "min",
        "max",
        "percentage_zeros",
        "missing_percentage",
    ]
    if expression.shape[1] == 0:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="sample_id"))
    finite_expression = _finite_expression(expression)
    statistics = finite_expression.agg(["mean", "median", "std", "min", "max"]).T
    statistics = statistics.rename(columns={"std": "standard_deviation"})
    denominator = feature_count or 1
    statistics["percentage_zeros"] = expression.eq(0).sum(axis=0) / denominator * 100
    statistics["missing_percentage"] = expression.isna().sum(axis=0) / denominator * 100
    statistics.index = statistics.index.map(str)
    statistics.index.name = "sample_id"
    return statistics[columns]


def run_expression_qc(
    expression: pd.DataFrame,
    config: ExpressionQCConfig | None = None,
) -> tuple[list[Finding], VarianceQCResult, pd.DataFrame]:
    """Run expression checks implemented through Phase 3C."""
    variance_result = analyze_gene_variance(expression, config)
    findings = [*detect_expression_issues(expression, variance_result.variances), *variance_result.findings]
    statistics = calculate_sample_statistics(expression)
    return findings, variance_result, statistics
