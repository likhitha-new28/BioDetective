import pandas as pd
import pytest

from biodetective.core.models import BioDataset
from biodetective.io.validators import validate_dataset


def make_dataset(expression=None, metadata=None):
    if expression is None:
        expression = pd.DataFrame({"S01": [10.0, 5.0], "S02": [12.0, 6.0]}, index=["TP53", "BRCA1"])
    if metadata is None:
        metadata = pd.DataFrame({"sample_id": ["S01", "S02"], "condition": ["Healthy", "Cancer"]})
    return BioDataset(expression, metadata)


def issue_codes(dataset):
    return {issue.code for issue in validate_dataset(dataset).issues}


def test_valid_dataset():
    assert validate_dataset(make_dataset()).is_valid


def test_missing_expression_sample_columns():
    expression = pd.DataFrame(index=["TP53"])
    codes = issue_codes(make_dataset(expression=expression))
    assert {"missing_expression_samples", "empty_expression"} <= codes


def test_missing_metadata_sample_id():
    metadata = pd.DataFrame({"condition": ["Healthy"]})
    assert "missing_sample_id" in issue_codes(make_dataset(metadata=metadata))


def test_duplicate_metadata_sample_ids():
    metadata = pd.DataFrame({"sample_id": ["S01", "S01"]})
    assert "duplicate_sample_ids" in issue_codes(make_dataset(metadata=metadata))


def test_duplicate_gene_ids():
    expression = pd.DataFrame({"S01": [1, 2], "S02": [3, 4]}, index=["TP53", "TP53"])
    assert "duplicate_gene_ids" in issue_codes(make_dataset(expression=expression))


def test_sample_id_mismatch():
    metadata = pd.DataFrame({"sample_id": ["S01", "S03"]})
    assert "sample_id_mismatch" in issue_codes(make_dataset(metadata=metadata))


@pytest.mark.parametrize(
    ("expression", "metadata", "expected"),
    [
        (pd.DataFrame(columns=["S01"]), None, "empty_expression"),
        (None, pd.DataFrame(columns=["sample_id"]), "empty_metadata"),
    ],
)
def test_empty_inputs(expression, metadata, expected):
    assert expected in issue_codes(make_dataset(expression=expression, metadata=metadata))


def test_non_numeric_expression():
    expression = pd.DataFrame({"S01": ["bad"], "S02": [2]}, index=["TP53"])
    assert "non_numeric_expression" in issue_codes(make_dataset(expression=expression))


def test_completely_empty_expression_row_and_column():
    expression = pd.DataFrame({"S01": [None, 1.0], "S02": [None, None]}, index=["TP53", "BRCA1"])
    codes = issue_codes(make_dataset(expression=expression))
    assert "empty_expression_rows" in codes
    assert "empty_expression_columns" in codes


def test_completely_empty_metadata_row_and_column():
    metadata = pd.DataFrame(
        {"sample_id": ["S01", "S02", None], "condition": ["Healthy", "Cancer", None], "unused": [None, None, None]}
    )
    codes = issue_codes(make_dataset(metadata=metadata))
    assert "empty_metadata_rows" in codes
    assert "empty_metadata_columns" in codes


def test_validation_does_not_modify_data():
    dataset = make_dataset()
    expression_before = dataset.expression.copy(deep=True)
    metadata_before = dataset.metadata.copy(deep=True)

    validate_dataset(dataset)

    pd.testing.assert_frame_equal(dataset.expression, expression_before)
    pd.testing.assert_frame_equal(dataset.metadata, metadata_before)
