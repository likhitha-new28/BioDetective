import pandas as pd
import pytest

from biodetective.analysis.metadata_mapper import (
    SEMANTIC_ROLES,
    create_mapping_approval,
    ranked_columns_for_role,
    suggest_metadata_roles,
)


def example_metadata():
    return pd.DataFrame(
        {
            "sample_id": [f"S{i}" for i in range(1, 7)],
            "disease_status": ["Healthy", "Cancer"] * 3,
            "gender": ["Female", "Male"] * 3,
            "sequencing_batch": ["Batch1"] * 3 + ["Batch2"] * 3,
            "age_years": [22, 35, 48, 51, 67, 73],
            "notes": ["a", "b", "c", "d", "e", "f"],
        }
    )


def test_mapper_ranks_expected_semantic_roles_deterministically():
    metadata = example_metadata()
    first = suggest_metadata_roles(metadata)
    second = suggest_metadata_roles(metadata)

    assert first == second
    assert first["disease_status"][0].role == "condition"
    assert first["gender"][0].role == "sex"
    assert first["sequencing_batch"][0].role == "batch"
    assert first["age_years"][0].role == "age"
    assert first["sample_id"] == ()


def test_value_heuristics_work_when_column_names_are_uninformative():
    metadata = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "field_a": ["Male", "Female", "Female", "Male"],
            "field_b": ["Control", "Treated", "Control", "Treated"],
            "field_c": ["Plate-A", "Plate-B", "Plate-A", "Plate-B"],
            "field_d": [21, 42, 63, 84],
        }
    )
    suggestions = suggest_metadata_roles(metadata)

    assert suggestions["field_a"][0].role == "sex"
    assert suggestions["field_b"][0].role == "condition"
    assert suggestions["field_c"][0].role == "batch"
    assert suggestions["field_d"][0].role == "age"


def test_ranked_columns_for_role_orders_score_then_column_name():
    candidates = ranked_columns_for_role(suggest_metadata_roles(example_metadata()), "condition")

    assert candidates[0].column == "disease_status"
    assert all(candidates[index].score >= candidates[index + 1].score for index in range(len(candidates) - 1))


def test_mapper_does_not_modify_metadata():
    metadata = example_metadata()
    original = metadata.copy(deep=True)

    suggest_metadata_roles(metadata)

    pd.testing.assert_frame_equal(metadata, original)


def test_mapping_requires_explicit_approval_for_every_role():
    metadata = example_metadata()
    mappings = {"condition": "disease_status", "sex": "gender", "batch": None, "age": "age_years"}

    partial = create_mapping_approval(metadata, mappings, ["condition", "sex", "age"])
    complete = create_mapping_approval(metadata, mappings, SEMANTIC_ROLES)

    assert partial.fully_approved is False
    assert complete.fully_approved is True
    assert complete.mappings["batch"] is None


def test_mapping_override_must_reference_an_existing_column():
    with pytest.raises(ValueError, match="do not exist"):
        create_mapping_approval(example_metadata(), {"condition": "missing_column"}, SEMANTIC_ROLES)


@pytest.mark.parametrize("minimum_score", [-0.1, 1.1])
def test_invalid_minimum_score_is_rejected(minimum_score):
    with pytest.raises(ValueError):
        suggest_metadata_roles(example_metadata(), minimum_score)
