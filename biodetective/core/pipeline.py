"""Fault-tolerant orchestration of BioDetective analysis modules."""

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from biodetective.analysis.batch_effects import BatchEffectConfig, analyze_batch_pca_association
from biodetective.analysis.confounding import ConfoundingConfig, analyze_confounding
from biodetective.analysis.expression_qc import ExpressionQCConfig, run_expression_qc
from biodetective.analysis.label_consistency import LabelConsistencyConfig, analyze_label_consistency
from biodetective.analysis.metadata_qc import MetadataQCConfig, run_metadata_qc
from biodetective.analysis.outliers import OutlierConfig, analyze_outliers
from biodetective.analysis.pca import PCAConfig, run_pca
from biodetective.analysis.sex_consistency import analyze_sex_marker_consistency
from biodetective.analysis.similarity import SimilarityConfig, analyze_sample_similarity
from biodetective.core.config import SexMarkerConfig
from biodetective.core.models import BioDataset, Finding
from biodetective.io.validators import validate_dataset
from biodetective.scoring.suspicion import (
    DatasetHealthConfig,
    DatasetHealthResult,
    SampleSuspicionScore,
    SuspicionScoreConfig,
    calculate_dataset_health,
    score_samples,
)


ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for required and optional pipeline modules."""

    metadata_qc: MetadataQCConfig = field(default_factory=MetadataQCConfig)
    expression_qc: ExpressionQCConfig = field(default_factory=ExpressionQCConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    similarity_method: str = "pearson"
    pca: PCAConfig = field(default_factory=PCAConfig)
    outliers: OutlierConfig = field(default_factory=OutlierConfig)
    label_column: str | None = None
    label_consistency: LabelConsistencyConfig = field(default_factory=LabelConsistencyConfig)
    sex_column: str | None = None
    sex_markers: SexMarkerConfig = field(default_factory=SexMarkerConfig)
    batch_column: str | None = None
    batch_effects: BatchEffectConfig = field(default_factory=BatchEffectConfig)
    biological_column: str | None = None
    technical_column: str | None = None
    confounding: ConfoundingConfig = field(default_factory=ConfoundingConfig)
    suspicion: SuspicionScoreConfig = field(default_factory=SuspicionScoreConfig)
    health: DatasetHealthConfig = field(default_factory=DatasetHealthConfig)
    metadata_mappings: dict[str, str | None] = field(default_factory=dict)


@dataclass(frozen=True)
class ModuleResult:
    """Outcome of one module without exposing exceptions to the whole pipeline."""

    status: str
    result: Any = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"completed", "skipped", "failed"}:
            raise ValueError("module status must be completed, skipped, or failed")


@dataclass(frozen=True)
class PipelineResult:
    """Structured results from a complete pipeline attempt."""

    dataset: BioDataset
    modules: dict[str, ModuleResult]
    findings: tuple[Finding, ...]
    sample_scores: tuple[SampleSuspicionScore, ...]
    dataset_health: DatasetHealthResult | None
    analysis_settings: dict[str, Any] = field(default_factory=dict)


def _findings_from_module(name: str, result: Any) -> list[Finding]:
    if name == "metadata_qc":
        return list(result)
    if name == "expression_qc":
        return list(result[0])
    if name == "similarity":
        return list(result[1])
    if name in {"outliers", "sex_consistency", "batch_analysis"}:
        return list(result.findings)
    if name == "label_consistency":
        findings = list(result.centroid.findings)
        if result.cross_validated is not None:
            findings.extend(result.cross_validated.findings)
        return findings
    return []


def run_biodetective_pipeline(
    dataset: BioDataset,
    config: PipelineConfig | None = None,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Run all configured analyses in order while isolating module failures."""
    config = config or PipelineConfig()
    modules: dict[str, ModuleResult] = {}
    findings: list[Finding] = []
    module_names = (
        "validation",
        "metadata_qc",
        "expression_qc",
        "similarity",
        "pca",
        "outliers",
        "label_consistency",
        "sex_consistency",
        "batch_analysis",
        "confounding",
        "scoring",
    )

    def record(name: str, outcome: ModuleResult) -> None:
        modules[name] = outcome
        if outcome.status == "completed":
            findings.extend(_findings_from_module(name, outcome.result))
        if progress_callback is not None:
            try:
                progress_callback(name, len(modules), len(module_names))
            except Exception:
                pass

    def execute(name: str, operation: Callable[[], Any]) -> None:
        try:
            record(name, ModuleResult("completed", operation()))
        except Exception as exc:  # A module failure must not terminate the pipeline.
            record(name, ModuleResult("failed", message=str(exc)))

    execute("validation", lambda: validate_dataset(dataset))
    execute("metadata_qc", lambda: run_metadata_qc(dataset.metadata, config.metadata_qc))
    execute("expression_qc", lambda: run_expression_qc(dataset.expression, config.expression_qc))
    execute(
        "similarity",
        lambda: analyze_sample_similarity(
            dataset.expression,
            dataset.metadata,
            method=config.similarity_method,
            config=config.similarity,
        ),
    )
    expression_module = modules["expression_qc"]
    precomputed_variances = expression_module.result[1].variances if expression_module.status == "completed" else None
    execute("pca", lambda: run_pca(dataset.expression, config.pca, gene_variances=precomputed_variances))

    pca_module = modules["pca"]
    if pca_module.status == "completed":
        execute("outliers", lambda: analyze_outliers(pca_module.result.coordinates, config.outliers))
    else:
        record("outliers", ModuleResult("skipped", message="PCA results are unavailable."))

    if config.label_column is None:
        record("label_consistency", ModuleResult("skipped", message="No label metadata column was configured."))
    else:
        try:
            result = analyze_label_consistency(
                dataset.expression,
                dataset.metadata,
                config.label_column,
                config.label_consistency,
            )
            record("label_consistency", ModuleResult("completed", result))
        except Exception as exc:
            record("label_consistency", ModuleResult("skipped", message=str(exc)))

    if config.sex_column is None:
        record("sex_consistency", ModuleResult("skipped", message="No sex metadata column was configured."))
    else:
        try:
            result = analyze_sex_marker_consistency(
                dataset.expression,
                dataset.metadata,
                config.sex_column,
                config.sex_markers,
            )
            record("sex_consistency", ModuleResult("completed", result))
        except Exception as exc:
            record("sex_consistency", ModuleResult("skipped", message=str(exc)))

    if config.batch_column is None:
        record("batch_analysis", ModuleResult("skipped", message="No batch metadata column was configured."))
    elif pca_module.status != "completed":
        record("batch_analysis", ModuleResult("skipped", message="PCA results are unavailable."))
    else:
        try:
            result = analyze_batch_pca_association(
                pca_module.result.coordinates,
                dataset.metadata,
                config.batch_column,
                config.batch_effects,
            )
            record("batch_analysis", ModuleResult("completed", result))
        except Exception as exc:
            record("batch_analysis", ModuleResult("skipped", message=str(exc)))

    if config.biological_column is None or config.technical_column is None:
        record(
            "confounding",
            ModuleResult("skipped", message="Biological and technical metadata columns were not both configured."),
        )
    else:
        try:
            result = analyze_confounding(
                dataset.metadata,
                config.biological_column,
                config.technical_column,
                config.confounding,
            )
            record("confounding", ModuleResult("completed", result))
        except Exception as exc:
            record("confounding", ModuleResult("skipped", message=str(exc)))

    sample_scores: tuple[SampleSuspicionScore, ...] = ()
    dataset_health: DatasetHealthResult | None = None
    try:
        sample_scores = tuple(score_samples(dataset.sample_ids, findings=findings, config=config.suspicion))
        duplicate_findings = (
            modules["similarity"].result[1] if modules["similarity"].status == "completed" else None
        )
        expression_findings = (
            modules["expression_qc"].result[0] if modules["expression_qc"].status == "completed" else None
        )
        batch_risk = (
            modules["batch_analysis"].result.risk if modules["batch_analysis"].status == "completed" else None
        )
        confounding_risk = (
            modules["confounding"].result.risk if modules["confounding"].status == "completed" else None
        )
        dataset_health = calculate_dataset_health(
            sample_scores=list(sample_scores),
            metadata=dataset.metadata if modules["metadata_qc"].status == "completed" else None,
            duplicate_findings=duplicate_findings,
            batch_risk=batch_risk,
            confounding_risk=confounding_risk,
            expression_findings=expression_findings,
            config=config.health,
        )
        record(
            "scoring",
            ModuleResult("completed", {"sample_scores": sample_scores, "dataset_health": dataset_health}),
        )
    except Exception as exc:
        record("scoring", ModuleResult("failed", message=str(exc)))

    return PipelineResult(
        dataset,
        modules,
        tuple(findings),
        sample_scores,
        dataset_health,
        analysis_settings=asdict(config),
    )
