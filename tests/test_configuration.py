from biodetective.analysis.batch_effects import BatchEffectConfig
from biodetective.analysis.confounding import ConfoundingConfig
from biodetective.analysis.expression_qc import ExpressionQCConfig
from biodetective.analysis.label_consistency import LabelConsistencyConfig
from biodetective.analysis.metadata_qc import MetadataQCConfig
from biodetective.analysis.outliers import OutlierConfig
from biodetective.analysis.pca import PCAConfig
from biodetective.analysis.similarity import SimilarityConfig
from biodetective.core import config as defaults
from biodetective.scoring.suspicion import DatasetHealthConfig, SuspicionScoreConfig


def test_analysis_configs_use_central_scientific_defaults():
    metadata = MetadataQCConfig()
    assert metadata.missing_low_max == defaults.DEFAULT_METADATA_MISSING_LOW_MAX
    assert metadata.missing_medium_max == defaults.DEFAULT_METADATA_MISSING_MEDIUM_MAX
    assert metadata.missing_high_max == defaults.DEFAULT_METADATA_MISSING_HIGH_MAX
    assert ExpressionQCConfig().low_variance_threshold == defaults.DEFAULT_EXPRESSION_LOW_VARIANCE_THRESHOLD
    assert SimilarityConfig().noteworthy_threshold == defaults.DEFAULT_SIMILARITY_NOTEWORTHY_THRESHOLD
    assert SimilarityConfig().highly_suspicious_threshold == defaults.DEFAULT_SIMILARITY_HIGHLY_SUSPICIOUS_THRESHOLD
    assert PCAConfig().n_components == defaults.DEFAULT_PCA_COMPONENTS
    assert OutlierConfig().distance_percentile_threshold == defaults.DEFAULT_PCA_DISTANCE_PERCENTILE
    assert LabelConsistencyConfig().min_samples_per_class == defaults.DEFAULT_LABEL_MIN_SAMPLES_PER_CLASS
    assert BatchEffectConfig().high_effect_size == defaults.DEFAULT_BATCH_HIGH_EFFECT_SIZE
    assert ConfoundingConfig().high_cramers_v == defaults.DEFAULT_CONFOUNDING_HIGH_CRAMERS_V


def test_scoring_configs_copy_central_defaults_without_sharing_mutable_state():
    first = SuspicionScoreConfig()
    second = SuspicionScoreConfig()
    health = DatasetHealthConfig()

    assert first.weights == defaults.DEFAULT_SAMPLE_SUSPICION_WEIGHTS
    assert health.maximum_deductions == defaults.DEFAULT_DATASET_HEALTH_DEDUCTIONS
    assert first.high_minimum == defaults.DEFAULT_RISK_HIGH_MINIMUM
    assert health.suspicious_score_threshold == defaults.DEFAULT_RISK_HIGH_MINIMUM
    assert first.weights is not second.weights
    assert health.maximum_deductions is not defaults.DEFAULT_DATASET_HEALTH_DEDUCTIONS


def test_central_defaults_preserve_previous_behavior():
    assert defaults.DEFAULT_SIMILARITY_NOTEWORTHY_THRESHOLD == 0.98
    assert defaults.DEFAULT_SIMILARITY_HIGHLY_SUSPICIOUS_THRESHOLD == 0.995
    assert defaults.DEFAULT_PCA_DISTANCE_PERCENTILE == 97.5
    assert (
        defaults.DEFAULT_RISK_MODERATE_MINIMUM,
        defaults.DEFAULT_RISK_HIGH_MINIMUM,
        defaults.DEFAULT_RISK_CRITICAL_MINIMUM,
    ) == (25.0, 50.0, 75.0)
