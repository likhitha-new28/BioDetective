"""Scientifically cautious, centralized explanations for BioDetective Findings."""

from dataclasses import dataclass
from typing import Any

from biodetective.core.models import Finding


@dataclass(frozen=True)
class FindingExplanation:
    observation: str
    evidence: dict[str, Any]
    interpretation: str
    possible_explanations: tuple[str, ...]
    recommendation: str

    @property
    def what_was_detected(self) -> str:
        return self.observation

    @property
    def why_it_matters(self) -> str:
        return self.interpretation

    @property
    def recommended_action(self) -> str:
        return self.recommendation


@dataclass(frozen=True)
class _ExplanationTemplate:
    why_it_matters: str
    possible_explanations: tuple[str, ...]
    recommended_action: str


EXPLANATION_TEMPLATES: dict[str, _ExplanationTemplate] = {
    "missing_metadata_values": _ExplanationTemplate(
        "Missing annotations can reduce usable sample counts and can bias comparisons when absence is systematic.",
        ("The field was not collected.", "Values were lost during export or merging.", "The field is not applicable to every sample."),
        "Review source records, restore values when justified, and document any values that remain unavailable.",
    ),
    "duplicate_sample_ids": _ExplanationTemplate(
        "Sample identifiers are used to align molecular measurements with metadata; repeated IDs make that alignment ambiguous.",
        ("Rows may have been duplicated during a merge.", "Distinct samples may have been assigned the same identifier."),
        "Resolve each duplicate against source records and assign one unambiguous identifier per sample before analysis.",
    ),
    "identical_metadata_rows": _ExplanationTemplate(
        "Identical annotations may be legitimate, but they can also indicate copied or duplicated records.",
        ("Samples may genuinely share all recorded attributes.", "Metadata rows may have been copied accidentally.", "Important distinguishing fields may be absent."),
        "Confirm sample provenance and add distinguishing annotations where they are available.",
    ),
    "inconsistent_categorical_labels": _ExplanationTemplate(
        "Formatting variants can split one biological group into multiple computational categories.",
        ("Capitalization or punctuation may differ across data-entry sources.", "Labels may have been entered manually without a controlled vocabulary."),
        "Verify that the variants have the same intended meaning, then apply a documented canonical vocabulary.",
    ),
    "constant_metadata_column": _ExplanationTemplate(
        "A constant field cannot explain differences among samples, although it may still document study context.",
        ("All samples may legitimately share the attribute.", "Variation may have been omitted during export."),
        "Confirm the value and retain it for documentation or exclude it from comparative modeling as appropriate.",
    ),
    "high_cardinality_categorical_column": _ExplanationTemplate(
        "A near-unique categorical field may behave like an identifier and is usually unsuitable as a grouping variable.",
        ("The column may contain sample identifiers.", "It may contain free text or overly detailed labels."),
        "Determine the field's intended role before using it for grouping, coloring, or statistical comparisons.",
    ),
    "class_imbalance": _ExplanationTemplate(
        "Strong imbalance can reduce power for the minority group and make model performance appear better than it generalizes.",
        ("The study design may intentionally use unequal groups.", "Samples may be missing or filtered unevenly."),
        "Review group counts and use analysis and validation procedures appropriate for imbalanced data.",
    ),
    "missing_expression_values": _ExplanationTemplate(
        "Missing measurements can prevent some analyses and may distort summaries if their pattern is not considered.",
        ("Assay measurements may have failed.", "Values may have become missing during transformation or merging."),
        "Trace missing measurements to their source and choose a documented, analysis-appropriate handling strategy.",
    ),
    "positive_infinity_values": _ExplanationTemplate(
        "Infinite values are not finite measurements and can invalidate distance, correlation, PCA, and modeling calculations.",
        ("A division by zero may have occurred.", "A transformation or import step may have overflowed."),
        "Inspect the generating calculation and correct or explicitly handle affected values before downstream analysis.",
    ),
    "negative_infinity_values": _ExplanationTemplate(
        "Infinite values are not finite measurements and can invalidate distance, correlation, PCA, and modeling calculations.",
        ("A logarithm of zero may have occurred.", "A transformation or import step may have produced an invalid value."),
        "Inspect the generating calculation and correct or explicitly handle affected values before downstream analysis.",
    ),
    "zero_variance_genes": _ExplanationTemplate(
        "Features with no variation cannot distinguish samples and may cause numerical problems in some models.",
        ("The gene may be unexpressed in this dataset.", "Values may have been rounded, capped, or filled uniformly."),
        "Confirm the measurements and consider excluding these features only within analyses that require informative variance.",
    ),
    "very_low_variance_genes": _ExplanationTemplate(
        "Very-low-variance features contribute little sample separation and may add noise to some models.",
        ("Expression may genuinely be stable across samples.", "Preprocessing may have compressed the observed range."),
        "Review the configured threshold and retain or filter these genes according to the downstream scientific objective.",
    ),
    "highly_suspicious_similarity": _ExplanationTemplate(
        "Extremely similar profiles may indicate related, replicated, or duplicated material, but correlation alone cannot establish identity.",
        ("The samples may be technical replicates.", "They may be biologically very similar.", "A file or sample may have been duplicated."),
        "Compare provenance, processing records, and metadata differences before deciding whether either sample requires action.",
    ),
    "noteworthy_similarity": _ExplanationTemplate(
        "High profile similarity is worth reviewing, but it does not by itself demonstrate a duplicate.",
        ("Samples may share a strong biological state.", "They may be related replicates.", "Technical processing may have reduced variation."),
        "Review sample relationships and provenance, especially when the paired metadata disagree.",
    ),
    "pca_distance_outlier": _ExplanationTemplate(
        "The sample lies far from the robust center in the analyzed PCA space, indicating an unusual multivariate profile.",
        ("The sample may represent genuine biology.", "Technical quality or preprocessing may differ.", "Metadata may be incorrect."),
        "Inspect assay quality, PCA loadings, metadata, and study context before excluding or relabeling the sample.",
    ),
    "combined_sample_outlier": _ExplanationTemplate(
        "One or both configured outlier detectors identified an unusual PCA profile; agreement provides stronger evidence, not certainty.",
        ("The profile may reflect genuine biological heterogeneity.", "A technical artifact or sample handling difference may be present."),
        "Review the detector-specific evidence and source data before making any correction or exclusion decision.",
    ),
    "molecular_profile_closer_to_another_group": _ExplanationTemplate(
        "The sample profile is more similar to another recorded group's centroid under the configured comparison.",
        ("The recorded label may need review.", "Groups may overlap biologically.", "Batch effects or outliers may influence similarity."),
        "Check provenance and potential confounders; do not change the label from molecular similarity alone.",
    ),
    "cross_validated_label_disagreement": _ExplanationTemplate(
        "A model trained without this sample predicted a different group from its recorded label.",
        ("The label may need review.", "The sample may be biologically atypical.", "Class overlap or limited sample size may reduce reliability."),
        "Consider confidence, cross-validation design, class size, provenance, and independent evidence before interpreting the disagreement.",
    ),
    "sex_marker_metadata_inconsistency": _ExplanationTemplate(
        "Available sex-associated expression markers appear inconsistent with the recorded metadata under the configured marker rules.",
        ("Metadata or sample identity may need review.", "Marker expression may vary biologically or technically.", "Too few informative markers may be available."),
        "Review marker coverage, assay context, and source metadata; do not infer biological sex from this signal alone.",
    ),
    "batch_pca_association": _ExplanationTemplate(
        "One or more PCA components are associated with the selected batch variable with a non-trivial effect size.",
        ("A technical batch effect may be present.", "Batch may be confounded with real biological groups.", "Chance association remains possible."),
        "Inspect study design and confounding before choosing a documented batch-aware analysis; BioDetective has not corrected the values.",
    ),
}


FALLBACK_TEMPLATE = _ExplanationTemplate(
    "This finding may affect interpretation, but its importance depends on the study design and supporting evidence.",
    ("The observation may reflect genuine biology.", "It may reflect metadata, processing, or measurement variation."),
    "Review the supplied evidence and source records before changing data or excluding samples.",
)


def explain_finding(finding: Finding) -> FindingExplanation:
    """Return one complete, cautious explanation for any Finding."""
    template = EXPLANATION_TEMPLATES.get(finding.code, FALLBACK_TEMPLATE)
    recommendation = template.recommended_action
    if finding.recommendation and finding.code not in EXPLANATION_TEMPLATES:
        recommendation = finding.recommendation
    return FindingExplanation(
        observation=finding.message,
        evidence=dict(finding.evidence),
        interpretation=template.why_it_matters,
        possible_explanations=template.possible_explanations,
        recommendation=recommendation,
    )
