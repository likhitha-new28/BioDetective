from pathlib import Path

from biodetective.core.models import Finding
from biodetective.reporting.explanations import explain_finding


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_user_facing_sources_avoid_categorical_error_claims():
    paths = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "pages" / "1_GEO_Import.py",
        PROJECT_ROOT / "biodetective" / "reporting" / "explanations.py",
        PROJECT_ROOT / "biodetective" / "reporting" / "html_report.py",
        *sorted((PROJECT_ROOT / "biodetective" / "analysis").glob("*.py")),
    ]
    text = "\n".join(path.read_text(encoding="utf-8").casefold() for path in paths)
    prohibited = (
        "sample is mislabeled",
        "sample is mislabelled",
        "sex is wrong",
        "dataset is invalid",
        "definitely duplicate",
        "is definitely a duplicate",
    )

    assert all(phrase not in text for phrase in prohibited)


def test_finding_explanation_exposes_four_scientific_sections():
    finding = Finding(
        category="sample_similarity",
        code="noteworthy_similarity",
        severity="medium",
        message="Potential duplicate or highly similar samples were observed.",
        evidence={"correlation": 0.98},
    )

    explanation = explain_finding(finding)

    assert explanation.observation
    assert explanation.evidence
    assert explanation.interpretation
    assert explanation.recommendation
