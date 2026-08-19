import pandas as pd

from biodetective.analysis.metadata_qc import (
    MetadataQCConfig,
    detect_category_inconsistencies,
    detect_metadata_duplicates,
    detect_metadata_structure,
    detect_missing_metadata,
    run_metadata_qc,
)


def test_missing_metadata_reports_count_percentage_samples_and_severity():
    metadata = pd.DataFrame(
        {
            "sample_id": [f"S{i:02d}" for i in range(1, 21)],
            "condition": [None, None, "Healthy", "Cancer"] + ["Healthy"] * 16,
        }
    )

    findings = detect_missing_metadata(metadata)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.column == "condition"
    assert finding.severity == "medium"
    assert finding.sample_ids == ["S01", "S02"]
    assert finding.evidence == {"missing_count": 2, "missing_percentage": 10.0}


def test_missing_thresholds_are_configurable():
    metadata = pd.DataFrame({"sample_id": ["S1", "S2"], "group": [None, "A"]})
    config = MetadataQCConfig(missing_low_max=60, missing_medium_max=70, missing_high_max=80)
    assert detect_missing_metadata(metadata, config)[0].severity == "low"


def test_missing_severity_boundaries():
    sample_ids = [f"S{i:03d}" for i in range(100)]
    expected = [(4, "low"), (5, "medium"), (15, "medium"), (16, "high"), (30, "high"), (31, "critical")]
    for missing_count, severity in expected:
        values = [None] * missing_count + ["A"] * (100 - missing_count)
        metadata = pd.DataFrame({"sample_id": sample_ids, "group": values})
        assert detect_missing_metadata(metadata)[0].severity == severity


def test_duplicate_sample_ids_and_identical_rows_are_reported_without_mutation():
    metadata = pd.DataFrame(
        {
            "sample_id": ["S01", "S01", "S03"],
            "condition": ["Healthy", "Healthy", "Healthy"],
            "batch": ["B1", "B1", "B1"],
        }
    )
    original = metadata.copy(deep=True)

    findings = detect_metadata_duplicates(metadata)
    codes = {finding.code for finding in findings}

    assert codes == {"duplicate_sample_ids", "identical_metadata_rows"}
    assert next(f for f in findings if f.code == "duplicate_sample_ids").sample_ids == ["S01"]
    pd.testing.assert_frame_equal(metadata, original)


def test_category_consistency_handles_case_space_and_punctuation():
    metadata = pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3", "S4"],
            "sex": ["Male", " male ", "MALE", "Female"],
            "condition": ["Case-Control", "case control", "Control", "control"],
        }
    )

    findings = detect_category_inconsistencies(metadata)

    assert len(findings) == 3
    assert {finding.column for finding in findings} == {"sex", "condition"}
    assert all(finding.code == "inconsistent_categorical_labels" for finding in findings)


def test_category_consistency_does_not_assume_gender_aliases():
    metadata = pd.DataFrame({"sample_id": ["S1", "S2"], "sex": ["M", "Male"]})
    assert detect_category_inconsistencies(metadata) == []

    findings = detect_category_inconsistencies(metadata, aliases={"sex": {"M": "Male"}})
    assert len(findings) == 1
    assert findings[0].evidence["variants"] == {"M": 1, "Male": 1}


def test_structural_checks_find_constant_high_cardinality_and_imbalance():
    metadata = pd.DataFrame(
        {
            "sample_id": [f"S{i:02d}" for i in range(20)],
            "study": ["StudyA"] * 20,
            "note": [f"note-{i}" for i in range(20)],
            "condition": ["Rare"] + ["Common"] * 19,
        }
    )

    findings = detect_metadata_structure(metadata)
    codes_by_column = {(finding.code, finding.column) for finding in findings}

    assert ("constant_metadata_column", "study") in codes_by_column
    assert ("high_cardinality_categorical_column", "note") in codes_by_column
    assert ("high_cardinality_categorical_column", "sample_id") not in codes_by_column
    assert ("class_imbalance", "condition") in codes_by_column


def test_structural_thresholds_are_configurable():
    metadata = pd.DataFrame({"sample_id": ["S1", "S2", "S3"], "label": ["A", "B", "C"]})
    strict = MetadataQCConfig(high_cardinality_ratio=1.0, high_cardinality_min_unique=4)
    assert not any(f.code == "high_cardinality_categorical_column" for f in detect_metadata_structure(metadata, strict))


def test_run_metadata_qc_combines_detectors():
    metadata = pd.DataFrame(
        {"sample_id": ["S1", "S2", "S2"], "condition": ["Control", " control ", None]}
    )
    codes = {finding.code for finding in run_metadata_qc(metadata)}
    assert "missing_metadata_values" in codes
    assert "duplicate_sample_ids" in codes
    assert "inconsistent_categorical_labels" in codes
