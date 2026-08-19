"""CSV loaders for expression data and sample metadata."""

from os import PathLike
from typing import IO

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from biodetective.core.exceptions import DataLoadError
from biodetective.core.models import BioDataset

CSVSource = str | PathLike[str] | IO[bytes] | IO[str]


def _read_csv(source: CSVSource, label: str) -> pd.DataFrame:
    try:
        return pd.read_csv(source)
    except EmptyDataError as exc:
        raise DataLoadError(f"The {label} CSV is empty.") from exc
    except (ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        raise DataLoadError(f"The {label} CSV could not be read: {exc}") from exc


def load_expression_csv(source: CSVSource) -> pd.DataFrame:
    """Load an expression CSV as a feature-by-sample numeric matrix."""
    frame = _read_csv(source, "expression")
    if "gene_id" not in frame.columns:
        raise DataLoadError("Expression CSV must contain a 'gene_id' column.")

    expression = frame.set_index("gene_id", drop=True)
    try:
        expression = expression.apply(pd.to_numeric, errors="raise")
    except (TypeError, ValueError) as exc:
        raise DataLoadError("Expression values must be numeric.") from exc

    expression.index.name = "gene_id"
    return expression


def load_metadata_csv(source: CSVSource) -> pd.DataFrame:
    """Load a metadata CSV while preserving its sample_id column."""
    frame = _read_csv(source, "metadata")
    if "sample_id" not in frame.columns:
        raise DataLoadError("Metadata CSV must contain a 'sample_id' column.")
    return frame


def load_biodataset(
    expression_source: CSVSource,
    metadata_source: CSVSource,
    dataset_name: str | None = None,
) -> BioDataset:
    """Load expression and metadata CSVs into a :class:`BioDataset`."""
    return BioDataset(
        expression=load_expression_csv(expression_source),
        metadata=load_metadata_csv(metadata_source),
        name=dataset_name,
    )
