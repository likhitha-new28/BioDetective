from io import StringIO

import pytest

from biodetective.core.exceptions import DataLoadError
from biodetective.io.loaders import load_biodataset, load_expression_csv, load_metadata_csv


EXPRESSION = "gene_id,S01,S02\nTP53,10,12\nBRCA1,5,6\n"
METADATA = "sample_id,condition,sex\nS01,Healthy,Male\nS02,Cancer,Female\n"


def test_load_biodataset_from_csv():
    dataset = load_biodataset(StringIO(EXPRESSION), StringIO(METADATA), "study")

    assert dataset.name == "study"
    assert dataset.expression.loc["TP53", "S01"] == 10
    assert dataset.expression.dtypes.apply(lambda dtype: dtype.kind in "iuf").all()
    assert dataset.metadata["sample_id"].tolist() == ["S01", "S02"]


def test_expression_requires_gene_id():
    with pytest.raises(DataLoadError, match="gene_id"):
        load_expression_csv(StringIO("feature,S01\nTP53,10\n"))


def test_metadata_requires_sample_id():
    with pytest.raises(DataLoadError, match="sample_id"):
        load_metadata_csv(StringIO("sample,condition\nS01,Healthy\n"))


def test_expression_values_must_be_numeric():
    with pytest.raises(DataLoadError, match="numeric"):
        load_expression_csv(StringIO("gene_id,S01\nTP53,unknown\n"))


def test_empty_csv_has_friendly_error():
    with pytest.raises(DataLoadError, match="empty"):
        load_metadata_csv(StringIO(""))
