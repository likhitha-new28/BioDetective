"""Core dataset models used throughout BioDetective."""

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


SEVERITIES = frozenset({"low", "medium", "high", "critical"})


@dataclass
class Finding:
    """A structured data-quality observation produced by an analysis."""

    category: str
    code: str
    severity: str
    message: str
    sample_ids: list[str] = field(default_factory=list)
    column: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    recommendation: str = "Review the affected records and confirm the expected metadata."

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            allowed = ", ".join(sorted(SEVERITIES))
            raise ValueError(f"severity must be one of: {allowed}")


@dataclass
class BioDataset:
    """A gene-expression matrix and its associated sample metadata.

    Expression rows represent features/genes and columns represent samples.
    Feature IDs are stored in the expression index. Metadata retains a
    ``sample_id`` column so its original contents can be validated unchanged.
    """

    expression: pd.DataFrame
    metadata: pd.DataFrame
    name: str | None = None

    @property
    def sample_ids(self) -> list[str]:
        return [str(sample_id) for sample_id in self.expression.columns]

    @property
    def feature_ids(self) -> list[str]:
        return [str(feature_id) for feature_id in self.expression.index]

    @property
    def n_samples(self) -> int:
        return self.expression.shape[1]

    @property
    def n_features(self) -> int:
        return self.expression.shape[0]

    @property
    def metadata_columns(self) -> list[str]:
        return [str(column) for column in self.metadata.columns]
