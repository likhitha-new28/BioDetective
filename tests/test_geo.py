import pytest

from biodetective.integrations.geo import (
    GEONetworkError,
    GEORecordNotFoundError,
    InvalidGEOAccessionError,
    fetch_geo_metadata,
    is_valid_gse_accession,
    normalize_gse_accession,
)


SERIES_SOFT = """^SERIES = GSE1234
!Series_title = Example study
!Series_summary = First description line
!Series_summary = Second description line
!Series_sample_id = GSM10
!Series_sample_id = GSM11
!Series_platform_id = GPL100
!Series_platform_id = GPL200
!Series_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE1nnn/GSE1234/suppl/raw.tar
"""

SAMPLES_SOFT = """^SAMPLE = GSM10
!Sample_title = Control sample
!Sample_platform_id = GPL100
!Sample_supplementary_file = ftp://ftp.ncbi.nlm.nih.gov/geo/samples/GSMnnn/GSM10/suppl/GSM10.CEL.gz
^SAMPLE = GSM11
!Sample_title = Treated sample
!Sample_platform_id = GPL200
"""

MATRIX_LISTING = """<html><body>
<a href="GSE1234-GPL100_series_matrix.txt.gz">first</a>
<a href="GSE1234-GPL200_series_matrix.txt.gz">second</a>
</body></html>"""


def fixture_transport(url, timeout):
    if "targ=self" in url:
        return SERIES_SOFT
    if "targ=gsm" in url:
        return SAMPLES_SOFT
    if url.endswith("/matrix/"):
        return MATRIX_LISTING
    raise AssertionError(f"unexpected URL: {url}")


@pytest.mark.parametrize("value", ["GSE1", "gse1000", "  GSE12345  "])
def test_gse_accession_validation_accepts_and_normalizes_valid_values(value):
    assert normalize_gse_accession(value).startswith("GSE")
    assert is_valid_gse_accession(value)


@pytest.mark.parametrize("value", ["", "GSM100", "GSE0", "GSEABC", "123"])
def test_gse_accession_validation_rejects_invalid_values(value):
    assert not is_valid_gse_accession(value)
    with pytest.raises(InvalidGEOAccessionError):
        normalize_gse_accession(value)


def test_fetch_geo_metadata_parses_series_samples_platforms_and_files():
    result = fetch_geo_metadata("gse1234", transport=fixture_transport)

    assert result.accession == "GSE1234"
    assert result.title == "Example study"
    assert result.description == "First description line\nSecond description line"
    assert result.platforms == ("GPL100", "GPL200")
    assert [sample.accession for sample in result.samples] == ["GSM10", "GSM11"]
    assert result.samples[0].title == "Control sample"
    assert result.samples[0].supplementary_files[0].url.startswith("https://ftp.ncbi.nlm.nih.gov")
    assert [file.platform_accession for file in result.expression_files] == ["GPL100", "GPL200"]
    assert result.supplementary_files[0].name == "raw.tar"


def test_fetch_geo_metadata_reports_missing_record():
    def empty_transport(url, timeout):
        return "Could not find a public GEO record"

    with pytest.raises(GEORecordNotFoundError, match="GSE999"):
        fetch_geo_metadata("GSE999", transport=empty_transport)


def test_fetch_geo_metadata_wraps_network_errors():
    def broken_transport(url, timeout):
        raise TimeoutError("timed out")

    with pytest.raises(GEONetworkError, match="timed out"):
        fetch_geo_metadata("GSE1000", transport=broken_transport)


def test_fetch_does_not_download_expression_files():
    requested_urls = []

    def recording_transport(url, timeout):
        requested_urls.append(url)
        return fixture_transport(url, timeout)

    result = fetch_geo_metadata("GSE1234", transport=recording_transport)

    assert len(requested_urls) == 3
    assert all(not url.endswith(".txt.gz") for url in requested_urls)
    assert len(result.expression_files) == 2
