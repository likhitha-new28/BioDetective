import pytest

from biodetective.core.models import Finding


def test_finding_defaults_are_independent():
    first = Finding("metadata", "example", "low", "Example finding")
    second = Finding("metadata", "example", "medium", "Another finding")

    first.sample_ids.append("S01")
    first.evidence["count"] = 1

    assert second.sample_ids == []
    assert second.evidence == {}
    assert second.column is None
    assert second.recommendation


@pytest.mark.parametrize("severity", ["low", "medium", "high", "critical"])
def test_finding_accepts_supported_severities(severity):
    assert Finding("metadata", "example", severity, "Message").severity == severity


def test_finding_rejects_unknown_severity():
    with pytest.raises(ValueError, match="severity"):
        Finding("metadata", "example", "urgent", "Message")
