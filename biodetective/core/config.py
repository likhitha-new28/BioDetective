"""Reusable configuration definitions for BioDetective analyses."""

from dataclasses import dataclass


# Central defaults preserve the detector behavior established in Phases 1-18.
DEFAULT_RANDOM_STATE = 42

DEFAULT_METADATA_MISSING_LOW_MAX = 5.0
DEFAULT_METADATA_MISSING_MEDIUM_MAX = 15.0
DEFAULT_METADATA_MISSING_HIGH_MAX = 30.0
DEFAULT_METADATA_HIGH_CARDINALITY_RATIO = 0.8
DEFAULT_METADATA_HIGH_CARDINALITY_MIN_UNIQUE = 10
DEFAULT_METADATA_IMBALANCE_MIN_FRACTION = 0.1
DEFAULT_METADATA_IMBALANCE_MAX_CATEGORIES = 20

DEFAULT_EXPRESSION_LOW_VARIANCE_THRESHOLD = 0.01
DEFAULT_SIMILARITY_NOTEWORTHY_THRESHOLD = 0.98
DEFAULT_SIMILARITY_HIGHLY_SUSPICIOUS_THRESHOLD = 0.995
DEFAULT_CORRELATION_MIN_PERIODS = 2
DEFAULT_PCA_REMOVE_ZERO_VARIANCE = True
DEFAULT_PCA_COMPONENTS = 3
DEFAULT_PCA_DISTANCE_PERCENTILE = 97.5
DEFAULT_ISOLATION_FOREST_CONTAMINATION = 0.1

DEFAULT_LABEL_SIMILARITY_MARGIN = 0.05
DEFAULT_LABEL_MIN_SAMPLES_PER_CLASS = 3
DEFAULT_LABEL_CV_FOLDS = 5
DEFAULT_LOGISTIC_REGRESSION_C = 1.0
DEFAULT_LOGISTIC_REGRESSION_MAX_ITERATIONS = 2000

DEFAULT_BATCH_ALPHA = 0.05
DEFAULT_BATCH_MODERATE_EFFECT_SIZE = 0.06
DEFAULT_BATCH_HIGH_EFFECT_SIZE = 0.14
DEFAULT_BATCH_MIN_SAMPLES = 2

DEFAULT_CONFOUNDING_MODERATE_CRAMERS_V = 0.3
DEFAULT_CONFOUNDING_HIGH_CRAMERS_V = 0.7
DEFAULT_CONFOUNDING_NEAR_PERFECT_PROPORTION = 0.9
DEFAULT_CONFOUNDING_MODERATE_CONDITIONAL_PROPORTION = 0.75

DEFAULT_SAMPLE_SUSPICION_WEIGHTS = {
    "pca_outlier": 25.0,
    "isolation_forest": 20.0,
    "duplicate_similarity": 25.0,
    "label_inconsistency": 15.0,
    "sex_consistency": 10.0,
    "metadata_issues": 5.0,
}
DEFAULT_DATASET_HEALTH_DEDUCTIONS = {
    "suspicious_sample_percentage": 25.0,
    "metadata_completeness": 20.0,
    "duplicate_risk": 15.0,
    "batch_risk": 15.0,
    "confounding_risk": 15.0,
    "expression_qc": 10.0,
}
DEFAULT_SEVERITY_STRENGTH = {"low": 0.25, "medium": 0.5, "high": 0.75, "critical": 1.0}
DEFAULT_RISK_MODERATE_MINIMUM = 25.0
DEFAULT_RISK_HIGH_MINIMUM = 50.0
DEFAULT_RISK_CRITICAL_MINIMUM = 75.0
DEFAULT_DATASET_RISK_FACTORS = {"Low": 0.0, "Moderate": 0.5, "High": 1.0, "Critical": 1.0}

DEFAULT_X_ASSOCIATED_MARKERS = ("XIST",)
DEFAULT_Y_ASSOCIATED_MARKERS = ("RPS4Y1", "KDM5D", "DDX3Y", "UTY", "EIF1AY")
DEFAULT_MINIMUM_X_MARKERS = 1
DEFAULT_MINIMUM_Y_MARKERS = 2
DEFAULT_MINIMUM_TOTAL_SEX_MARKERS = 3
DEFAULT_SEX_PATTERN_SCORE_THRESHOLD = 0.5
DEFAULT_SEX_MODERATE_EVIDENCE_THRESHOLD = 1.0
DEFAULT_SEX_STRONG_EVIDENCE_THRESHOLD = 2.0
DEFAULT_MINIMUM_SUPPORTING_SEX_MARKERS = 2
DEFAULT_X_ASSOCIATED_METADATA_VALUES = ("female",)
DEFAULT_Y_ASSOCIATED_METADATA_VALUES = ("male",)


@dataclass(frozen=True)
class SexMarkerConfig:
    """Configurable sex-associated expression markers and evidence thresholds."""

    x_associated_markers: tuple[str, ...] = DEFAULT_X_ASSOCIATED_MARKERS
    y_associated_markers: tuple[str, ...] = DEFAULT_Y_ASSOCIATED_MARKERS
    minimum_x_markers: int = DEFAULT_MINIMUM_X_MARKERS
    minimum_y_markers: int = DEFAULT_MINIMUM_Y_MARKERS
    minimum_total_markers: int = DEFAULT_MINIMUM_TOTAL_SEX_MARKERS
    pattern_score_threshold: float = DEFAULT_SEX_PATTERN_SCORE_THRESHOLD
    moderate_evidence_threshold: float = DEFAULT_SEX_MODERATE_EVIDENCE_THRESHOLD
    strong_evidence_threshold: float = DEFAULT_SEX_STRONG_EVIDENCE_THRESHOLD
    minimum_supporting_markers: int = DEFAULT_MINIMUM_SUPPORTING_SEX_MARKERS
    x_associated_metadata_values: tuple[str, ...] = DEFAULT_X_ASSOCIATED_METADATA_VALUES
    y_associated_metadata_values: tuple[str, ...] = DEFAULT_Y_ASSOCIATED_METADATA_VALUES

    def __post_init__(self) -> None:
        if not self.x_associated_markers or not self.y_associated_markers:
            raise ValueError("both X-associated and Y-associated marker definitions are required")
        if min(self.minimum_x_markers, self.minimum_y_markers, self.minimum_total_markers) < 1:
            raise ValueError("minimum marker counts must be positive")
        if self.pattern_score_threshold < 0:
            raise ValueError("pattern_score_threshold must be non-negative")
        if not self.pattern_score_threshold <= self.moderate_evidence_threshold <= self.strong_evidence_threshold:
            raise ValueError("evidence thresholds must be ordered from pattern to moderate to strong")
        if self.minimum_supporting_markers < 1:
            raise ValueError("minimum_supporting_markers must be positive")
