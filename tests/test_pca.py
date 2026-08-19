import numpy as np
import pandas as pd
import pytest

from biodetective.analysis.pca import PCAConfig, run_pca


def expression_fixture():
    return pd.DataFrame(
        {
            "S1": [1.0, 1.0, 2.0, 4.0],
            "S2": [1.0, 2.0, 3.0, 3.0],
            "S3": [1.0, 3.0, 5.0, 2.0],
            "S4": [1.0, 4.0, 7.0, 1.0],
        },
        index=["constant", "G2", "G3", "G4"],
    )


def test_pca_returns_coordinates_variance_loadings_and_configuration_without_mutation():
    expression = expression_fixture()
    original = expression.copy(deep=True)
    config = PCAConfig(remove_zero_variance_genes=True, top_variable_genes=2, n_components=2)

    result = run_pca(expression, config)

    assert result.coordinates.shape == (4, 2)
    assert result.coordinates.index.tolist() == ["S1", "S2", "S3", "S4"]
    assert result.explained_variance.index.tolist() == ["PC1", "PC2"]
    assert result.loadings.shape == (2, 2)
    assert result.preprocessing_config["removed_zero_variance_gene_count"] == 1
    assert result.preprocessing_config["selected_gene_count"] == 2
    assert result.preprocessing_config["top_variable_genes"] == 2
    pd.testing.assert_frame_equal(expression, original)


def test_pca_can_retain_zero_variance_genes():
    result = run_pca(expression_fixture(), PCAConfig(remove_zero_variance_genes=False, n_components=2))
    assert "constant" in result.loadings.index
    assert result.preprocessing_config["removed_zero_variance_gene_count"] == 0


def test_pca_rejects_invalid_values():
    expression = expression_fixture()
    expression.loc["G2", "S1"] = np.nan
    with pytest.raises(ValueError, match="missing or infinite"):
        run_pca(expression, PCAConfig(n_components=2))


def test_pca_rejects_too_many_components():
    with pytest.raises(ValueError, match="n_components"):
        run_pca(expression_fixture(), PCAConfig(top_variable_genes=2, n_components=3))


def test_pca_rejects_dataset_with_only_zero_variance_genes():
    expression = pd.DataFrame({"S1": [1.0], "S2": [1.0]}, index=["constant"])
    with pytest.raises(ValueError, match="no genes remain"):
        run_pca(expression)
