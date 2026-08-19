"""Standalone HTML reporting for BioDetective pipeline results."""

from html import escape
import json
from typing import Any

import pandas as pd

from biodetective.core.models import Finding
from biodetective.core.pipeline import PipelineResult
from biodetective.reporting.explanations import explain_finding
from biodetective.reporting.exports import build_analysis_payload, finding_to_record, sorted_findings


def _table(records: list[dict[str, Any]], empty_message: str = "No results available.") -> str:
    if not records:
        return f"<p>{escape(empty_message)}</p>"
    return pd.DataFrame(records).to_html(index=False, escape=True, border=0, classes="report-table")


def _finding_table(findings: list[Finding]) -> str:
    records = []
    for finding in findings:
        record = finding_to_record(finding)
        explanation = explain_finding(finding)
        interpretation = explanation.interpretation + " Possible explanations include: " + " ".join(
            explanation.possible_explanations
        )
        records.append(
            {
                "Severity": record["severity"].title(),
                "Category": record["category"],
                "Observation": explanation.observation,
                "Evidence": json.dumps(explanation.evidence, sort_keys=True),
                "Interpretation": interpretation,
                "Samples": ", ".join(record["sample_ids"]),
                "Recommendation": explanation.recommendation,
            }
        )
    return _table(records, "No findings in this section.")


def _module_note(result: PipelineResult, module_name: str) -> str:
    module = result.modules.get(module_name)
    if module is None:
        return "<p>Module result unavailable.</p>"
    message = f" — {escape(module.message)}" if module.message else ""
    return f"<p><strong>Status:</strong> {escape(module.status.title())}{message}</p>"


def generate_html_report(result: PipelineResult) -> str:
    """Generate a self-contained HTML report with no raw expression matrix."""
    payload = build_analysis_payload(result)
    summary = payload["dataset_summary"]
    health_score = result.dataset_health.score if result.dataset_health is not None else "Unavailable"
    ordered_findings = sorted_findings(result.findings)
    suspicious_scores = [score for score in result.sample_scores if score.risk in {"High", "Critical"}]

    metadata_categories = {"missing_metadata", "metadata_duplicates", "category_consistency", "metadata_structure"}
    sections = {
        "Metadata QC": [finding for finding in ordered_findings if finding.category in metadata_categories],
        "Expression QC": [finding for finding in ordered_findings if finding.category.startswith("expression")],
        "Duplicates": [finding for finding in ordered_findings if finding.category == "sample_similarity"],
        "Outliers": [finding for finding in ordered_findings if finding.category == "sample_outlier"],
        "Label Consistency": [finding for finding in ordered_findings if finding.category == "label_consistency"],
    }
    summary_records = [
        {"Metric": "Dataset", "Value": summary["name"] or "Unnamed dataset"},
        {"Metric": "Samples", "Value": summary["samples"]},
        {"Metric": "Genes/features", "Value": summary["genes_features"]},
        {"Metric": "Metadata columns", "Value": summary["metadata_columns"]},
        {"Metric": "Metadata completeness", "Value": f"{summary['metadata_completeness_percentage']:.2f}%"},
    ]
    suspicious_records = [
        {
            "Sample": score.sample_id,
            "Score": score.score,
            "Risk": score.risk,
            "Contributions": json.dumps(score.contributions, sort_keys=True),
        }
        for score in sorted(suspicious_scores, key=lambda item: item.score, reverse=True)
    ]
    settings = escape(json.dumps(payload["analysis_settings"], indent=2, sort_keys=True))
    batch = payload["analyses"]["batch_effects"]
    confounding = payload["analyses"]["confounding"]
    batch_content = _module_note(result, "batch_analysis")
    if batch is not None:
        batch_content += _table(
            [{"Risk": batch["risk"], "Explanation": batch["explanation"], "Batch counts": json.dumps(batch["batch_counts"])}]
        )
        batch_content += _finding_table(
            list(result.modules["batch_analysis"].result.findings)
        )
    confounding_content = _module_note(result, "confounding")
    if confounding is not None:
        confounding_content += _table(
            [
                {
                    "Observation": f"The selected variables have a confounding risk rating of {confounding['risk']}.",
                    "Evidence": f"Cramer's V = {confounding['cramers_v']:.4f}",
                    "Interpretation": confounding["interpretation"],
                    "Recommendation": "Review study design and conditional counts before attributing variation to either variable.",
                }
            ]
        )

    finding_sections = "".join(
        f"<section><h2>{escape(title)}</h2>{_finding_table(findings)}</section>"
        for title, findings in sections.items()
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BioDetective Analysis Report</title>
  <style>
    body {{ color: #172033; font-family: Arial, sans-serif; line-height: 1.45; margin: 0 auto; max-width: 1100px; padding: 32px; }}
    h1, h2 {{ color: #173f5f; }} section {{ margin: 30px 0; }}
    .health {{ background: #e8f5ee; border-left: 6px solid #25855a; font-size: 1.4rem; padding: 18px; }}
    .report-table {{ border-collapse: collapse; width: 100%; }}
    .report-table th, .report-table td {{ border: 1px solid #d8dee8; padding: 8px; text-align: left; vertical-align: top; }}
    .report-table th {{ background: #eef3f8; }} pre {{ background: #f5f7fa; overflow-x: auto; padding: 14px; }}
    .disclaimer {{ background: #fff4dd; border-left: 6px solid #d99200; padding: 16px; }}
  </style>
</head>
<body>
  <h1>BioDetective Analysis Report</h1>
  <section><h2>Dataset Summary</h2>{_table(summary_records)}</section>
  <section><h2>Dataset Health Score</h2><div class="health">{escape(str(health_score))} / 100</div></section>
  <section><h2>Analysis Settings</h2><pre>{settings}</pre></section>
  <section><h2>Top Findings</h2>{_finding_table(ordered_findings[:10])}</section>
  <section><h2>High/Critical Risk Samples</h2>{_table(suspicious_records, "No High or Critical risk samples were identified.")}</section>
  {finding_sections}
  <section><h2>Batch Effects</h2>{batch_content}</section>
  <section><h2>Confounding</h2>{confounding_content}</section>
  <section><h2>Methodology</h2><p>BioDetective validates dataset structure and summarizes configured metadata, expression, similarity, PCA, outlier, label-consistency, sex-marker, batch, confounding, and scoring analyses. Findings are evidence for review rather than automatic corrections.</p></section>
  <section><h2>Limitations</h2><p>Results depend on dataset size, available markers, metadata quality, configured thresholds, and suitability of the statistical assumptions. Correlation, classification, and anomaly evidence cannot establish sample identity or the correct metadata value by themselves.</p></section>
  <section class="disclaimer"><h2>Disclaimer</h2><p>This report is a data-quality screening aid and is not a clinical diagnosis or a substitute for expert biological, statistical, or provenance review.</p></section>
</body>
</html>"""
