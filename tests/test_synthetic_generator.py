import pandas as pd
import pytest

from biodetective.synthetic.generator import generate_synthetic_dataset


def test_clean_dataset_has_expected_shape_metadata_and_ground_truth():
    expression, metadata, truth = generate_synthetic_dataset(60, 24, 3, 2, 7)

    assert expression.shape == (60, 24)
    assert metadata.shape == (24, 4)
    assert expression.columns.tolist() == metadata["sample_id"].tolist()
    assert expression.index.name == "gene_id"
    assert metadata["condition"].nunique() == 3
    assert metadata["batch"].nunique() == 2
    assert truth["clean"] is True


def test_generation_is_reproducible():
    first = generate_synthetic_dataset(30, 16, 2, 2, 99, n_near_duplicates=1, n_outliers=1)
    second = generate_synthetic_dataset(30, 16, 2, 2, 99, n_near_duplicates=1, n_outliers=1)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    assert first[2] == second[2]


def test_exact_and_near_duplicates_are_planted_and_reported():
    expression, _, truth = generate_synthetic_dataset(
        100, 20, 2, 2, 12, n_exact_duplicates=2, n_near_duplicates=2, near_duplicate_noise=0.01
    )

    assert len(truth["exact_duplicates"]["affected_sample_ids"]) == 4
    assert len(truth["near_duplicates"]["affected_sample_ids"]) == 4
    for pair in truth["exact_duplicates"]["pairs"]:
        pd.testing.assert_series_equal(
            expression[pair["source_sample_id"]],
            expression[pair["duplicate_sample_id"]],
            check_names=False,
        )
    for pair in truth["near_duplicates"]["pairs"]:
        source = expression[pair["source_sample_id"]]
        duplicate = expression[pair["duplicate_sample_id"]]
        assert not source.equals(duplicate)
        assert source.corr(duplicate) > 0.99


def test_outliers_are_planted_and_reported():
    expression, _, truth = generate_synthetic_dataset(200, 30, 2, 2, 21, n_outliers=3, outlier_strength=12)

    outlier_ids = truth["outliers"]["affected_sample_ids"]
    assert len(outlier_ids) == 3
    regular_ids = [sample_id for sample_id in expression if sample_id not in outlier_ids]
    assert expression[outlier_ids].std(axis=0).mean() > expression[regular_ids].std(axis=0).mean() * 4


def test_label_swaps_preserve_original_and_recorded_labels():
    _, metadata, truth = generate_synthetic_dataset(50, 20, 3, 2, 5, n_label_swaps=4)

    assert len(truth["label_swaps"]["affected_sample_ids"]) == 4
    labels = metadata.set_index("sample_id")["condition"]
    for swap in truth["label_swaps"]["swaps"]:
        assert swap["original_condition"] != swap["recorded_condition"]
        assert labels[swap["sample_id"]] == swap["recorded_condition"]


def test_batch_effect_strength_is_applied_and_described():
    expression, metadata, truth = generate_synthetic_dataset(
        100, 40, 2, 2, 8, batch_effect_strength=6
    )
    affected = truth["batch_effect"]["affected_gene_ids"]
    first_mean = expression.loc[affected, metadata.loc[metadata["batch"].eq("Batch1"), "sample_id"]].to_numpy().mean()
    second_mean = expression.loc[affected, metadata.loc[metadata["batch"].eq("Batch2"), "sample_id"]].to_numpy().mean()

    assert truth["batch_effect"]["strength"] == 6
    assert len(affected) == 20
    assert second_mean - first_mean > 8


def test_complete_and_partial_confounding_are_supported():
    _, complete_metadata, complete_truth = generate_synthetic_dataset(
        30, 40, 2, 2, 3, confounding="complete"
    )
    _, partial_metadata, partial_truth = generate_synthetic_dataset(
        30, 80, 2, 2, 3, confounding="partial", confounding_strength=0.75
    )

    complete_table = pd.crosstab(complete_metadata["condition"], complete_metadata["batch"])
    partial_table = pd.crosstab(partial_metadata["condition"], partial_metadata["batch"])
    assert (complete_table.gt(0).sum(axis=1) == 1).all()
    assert (partial_table.gt(0).sum(axis=1) > 1).all()
    assert complete_truth["confounding"]["mode"] == "complete"
    assert partial_truth["confounding"]["mode"] == "partial"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_genes": 0},
        {"n_conditions": 21},
        {"n_exact_duplicates": -1},
        {"n_label_swaps": 1, "n_conditions": 1},
        {"batch_effect_strength": -1},
        {"confounding": "invalid"},
    ],
)
def test_invalid_generator_parameters_raise_friendly_errors(kwargs):
    with pytest.raises(ValueError):
        generate_synthetic_dataset(**kwargs)
