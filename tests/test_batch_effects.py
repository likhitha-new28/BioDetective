import numpy as np
import pandas as pd

from biodetective.analysis.batch_effects import BatchEffectConfig, analyze_batch_pca_association
from biodetective.analysis.pca import PCAConfig, run_pca


def batch_shifted_expression(seed=42):
    rng = np.random.default_rng(seed)
    gene_count = 60
    samples_per_batch = 15
    batch1 = rng.normal(0, 0.5, size=(gene_count, samples_per_batch))
    batch2 = rng.normal(0, 0.5, size=(gene_count, samples_per_batch))
    batch2[:30] += 5.0
    values = np.hstack([batch1, batch2])
    sample_ids = [f"S{i:02d}" for i in range(1, samples_per_batch * 2 + 1)]
    expression = pd.DataFrame(values, index=[f"G{i:03d}" for i in range(gene_count)], columns=sample_ids)
    metadata = pd.DataFrame(
        {"sample_id": sample_ids, "batch": ["Batch1"] * samples_per_batch + ["Batch2"] * samples_per_batch}
    )
    return expression, metadata


def test_batch_shift_produces_strong_pca_association_and_high_risk():
    expression, metadata = batch_shifted_expression()
    pca = run_pca(expression, PCAConfig(n_components=3))
    coordinates_before = pca.coordinates.copy(deep=True)
    metadata_before = metadata.copy(deep=True)

    result = analyze_batch_pca_association(pca.coordinates, metadata, "batch")

    assert result.risk == "High"
    assert result.batch_counts == {"Batch1": 15, "Batch2": 15}
    assert result.associations.loc["PC1", "effect_size"] >= 0.14
    assert result.evidence["strongest_component"] == "PC1"
    assert result.findings[0].code == "batch_pca_association"
    pd.testing.assert_frame_equal(pca.coordinates, coordinates_before)
    pd.testing.assert_frame_equal(metadata, metadata_before)


def test_balanced_component_values_have_low_batch_risk():
    coordinates = pd.DataFrame(
        {"PC1": [-1, 0, 1, -1, 0, 1], "PC2": [1, 0, -1, 1, 0, -1]},
        index=["S1", "S2", "S3", "S4", "S5", "S6"],
    )
    metadata = pd.DataFrame({"sample_id": coordinates.index, "batch": ["B1"] * 3 + ["B2"] * 3})
    result = analyze_batch_pca_association(coordinates, metadata, "batch")
    assert result.risk == "Low"
    assert result.associations["effect_size"].max() == 0


def test_batch_effect_thresholds_are_configurable():
    coordinates = pd.DataFrame(
        {"PC1": [0, 1, 2, 1, 2, 3]},
        index=["S1", "S2", "S3", "S4", "S5", "S6"],
    )
    metadata = pd.DataFrame({"sample_id": coordinates.index, "batch": ["B1"] * 3 + ["B2"] * 3})
    relaxed = BatchEffectConfig(moderate_effect_size=0.01, high_effect_size=0.05)
    result = analyze_batch_pca_association(coordinates, metadata, "batch", relaxed)
    assert result.risk == "High"


def test_batch_analysis_reports_anova_and_robust_alternative():
    expression, metadata = batch_shifted_expression(seed=9)
    pca = run_pca(expression, PCAConfig(n_components=2))
    result = analyze_batch_pca_association(pca.coordinates, metadata, "batch")
    assert {"ANOVA", "Kruskal-Wallis"} >= set(result.associations["preferred_test"])
    assert result.associations["anova_p_value"].notna().all()
    assert result.associations["kruskal_p_value"].notna().all()
    assert result.associations["eta_squared"].between(0, 1).all()
