"""Principal component analysis for expression datasets."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from biodetective.core.config import DEFAULT_PCA_COMPONENTS, DEFAULT_PCA_REMOVE_ZERO_VARIANCE


@dataclass(frozen=True)
class PCAConfig:
    """Preprocessing and component options for PCA."""

    remove_zero_variance_genes: bool = DEFAULT_PCA_REMOVE_ZERO_VARIANCE
    top_variable_genes: int | None = None
    n_components: int = DEFAULT_PCA_COMPONENTS

    def __post_init__(self) -> None:
        if self.top_variable_genes is not None and self.top_variable_genes < 1:
            raise ValueError("top_variable_genes must be at least 1")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")


@dataclass(frozen=True)
class PCAResult:
    """Coordinates, explained variance, loadings, and preprocessing details."""

    coordinates: pd.DataFrame
    explained_variance: pd.Series
    loadings: pd.DataFrame
    preprocessing_config: dict[str, object]


def run_pca(
    expression: pd.DataFrame,
    config: PCAConfig | None = None,
    gene_variances: pd.Series | None = None,
) -> PCAResult:
    """Run PCA on samples without modifying the source expression matrix."""
    config = config or PCAConfig()
    if expression.empty or expression.shape[1] == 0:
        raise ValueError("expression must contain at least one gene and one sample")
    if not all(pd.api.types.is_numeric_dtype(expression[column]) for column in expression.columns):
        raise ValueError("expression values must be numeric")
    if not np.isfinite(expression.to_numpy(dtype=float)).all():
        raise ValueError("expression must not contain missing or infinite values for PCA")

    if gene_variances is None:
        gene_variances = expression.var(axis=1, ddof=0)
    elif len(gene_variances) != len(expression) or not gene_variances.index.equals(expression.index):
        raise ValueError("precomputed gene variances must align with the expression rows")
    selected_expression = expression
    zero_variance_mask = gene_variances.eq(0)
    removed_zero_variance = 0
    if config.remove_zero_variance_genes:
        removed_zero_variance = int(zero_variance_mask.sum())
        selected_expression = selected_expression.loc[~zero_variance_mask]
        gene_variances = gene_variances.loc[~zero_variance_mask]

    if config.top_variable_genes is not None and len(selected_expression) > config.top_variable_genes:
        selected_ids = gene_variances.sort_values(ascending=False, kind="stable").head(config.top_variable_genes).index
        selected_expression = selected_expression.loc[selected_ids]

    if selected_expression.empty:
        raise ValueError("no genes remain after PCA preprocessing")

    maximum_components = min(selected_expression.shape[0], selected_expression.shape[1])
    if config.n_components > maximum_components:
        raise ValueError(
            f"n_components cannot exceed {maximum_components} for the selected genes and samples"
        )

    model = PCA(n_components=config.n_components)
    transformed = model.fit_transform(selected_expression.T.to_numpy(dtype=float, copy=False))
    component_names = [f"PC{index}" for index in range(1, config.n_components + 1)]

    coordinates = pd.DataFrame(transformed, index=selected_expression.columns.map(str), columns=component_names)
    coordinates.index.name = "sample_id"
    explained_variance = pd.Series(model.explained_variance_ratio_, index=component_names, name="explained_variance_ratio")
    loadings = pd.DataFrame(model.components_.T, index=selected_expression.index.map(str), columns=component_names)
    loadings.index.name = "gene_id"

    preprocessing_config: dict[str, object] = {
        **asdict(config),
        "input_gene_count": int(expression.shape[0]),
        "selected_gene_count": int(selected_expression.shape[0]),
        "removed_zero_variance_gene_count": removed_zero_variance,
        "selected_gene_ids": [str(gene_id) for gene_id in selected_expression.index],
    }
    return PCAResult(coordinates, explained_variance, loadings, preprocessing_config)
