"""Metadata-only integration with NCBI Gene Expression Omnibus (GEO)."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

from biodetective.core.exceptions import BioDetectiveError


GSE_PATTERN = re.compile(r"^GSE[1-9]\d*$", re.IGNORECASE)
GEO_ACCESSION_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
GEO_FTP_HTTPS_ROOT = "https://ftp.ncbi.nlm.nih.gov/geo/series"


class GEOIntegrationError(BioDetectiveError):
    """Base exception for expected GEO integration failures."""


class InvalidGEOAccessionError(GEOIntegrationError):
    """Raised when an accession is not a valid GSE identifier."""


class GEORecordNotFoundError(GEOIntegrationError):
    """Raised when NCBI does not return the requested public GSE record."""


class GEONetworkError(GEOIntegrationError):
    """Raised when NCBI cannot be reached or returns an unusable response."""


@dataclass(frozen=True)
class GEOFile:
    name: str
    url: str
    kind: str
    source_accession: str
    platform_accession: str | None = None


@dataclass(frozen=True)
class GEOSample:
    accession: str
    title: str
    platform_accession: str | None
    supplementary_files: tuple[GEOFile, ...] = ()


@dataclass(frozen=True)
class GEOSeriesMetadata:
    accession: str
    title: str
    description: str
    platforms: tuple[str, ...]
    samples: tuple[GEOSample, ...]
    expression_files: tuple[GEOFile, ...]
    supplementary_files: tuple[GEOFile, ...]


class GEOClient(Protocol):
    """Interface for metadata-only GEO clients."""

    def fetch_metadata(self, accession: str) -> GEOSeriesMetadata:
        """Fetch one public GSE metadata record without downloading expression data."""


Transport = Callable[[str, float], str]


def normalize_gse_accession(accession: str) -> str:
    """Validate and normalize a GEO Series accession."""
    normalized = str(accession).strip().upper()
    if not GSE_PATTERN.fullmatch(normalized):
        raise InvalidGEOAccessionError("Enter a valid GEO Series accession such as GSE1000.")
    return normalized


def is_valid_gse_accession(accession: str) -> bool:
    try:
        normalize_gse_accession(accession)
    except InvalidGEOAccessionError:
        return False
    return True


def _default_transport(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": "BioDetective/1.0 (GEO metadata client)"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise GEONetworkError(f"NCBI GEO returned HTTP {exc.code}.") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise GEONetworkError(f"Could not reach NCBI GEO: {exc}") from exc


def _request_text(transport: Transport, url: str, timeout: float) -> str:
    try:
        return transport(url, timeout)
    except GEOIntegrationError:
        raise
    except Exception as exc:
        raise GEONetworkError(f"Could not retrieve GEO metadata: {exc}") from exc


def _soft_entities(text: str) -> list[dict[str, object]]:
    entities: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("^") and "=" in line:
            entity_type, accession = line[1:].split("=", 1)
            current = {"type": entity_type.strip().upper(), "accession": accession.strip(), "fields": {}}
            entities.append(current)
        elif line.startswith("!") and "=" in line and current is not None:
            key, value = line[1:].split("=", 1)
            fields = current["fields"]
            fields.setdefault(key.strip(), []).append(value.strip())
    return entities


def _values(entity: dict[str, object], key: str) -> list[str]:
    return list(entity["fields"].get(key, []))


def _first(entity: dict[str, object], key: str, default: str = "") -> str:
    values = _values(entity, key)
    return values[0] if values else default


def _https_ncbi_url(url: str) -> str:
    return url.replace("ftp://ftp.ncbi.nlm.nih.gov", "https://ftp.ncbi.nlm.nih.gov", 1)


def _file_from_url(
    url: str,
    kind: str,
    source_accession: str,
    platform_accession: str | None = None,
) -> GEOFile:
    normalized_url = _https_ncbi_url(url)
    name = PurePosixPath(urlparse(normalized_url).path).name or normalized_url
    return GEOFile(name, normalized_url, kind, source_accession, platform_accession)


def _series_range(accession: str) -> str:
    digits = accession[3:]
    prefix = digits[:-3]
    return f"GSE{prefix}nnn" if prefix else "GSEnnn"


def _accession_url(accession: str, target: str) -> str:
    query = urlencode({"acc": accession, "targ": target, "view": "brief", "form": "text"})
    return f"{GEO_ACCESSION_URL}?{query}"


def _matrix_directory_url(accession: str) -> str:
    return f"{GEO_FTP_HTTPS_ROOT}/{_series_range(accession)}/{accession}/matrix/"


def _parse_matrix_files(html: str, directory_url: str, accession: str, platforms: tuple[str, ...]) -> tuple[GEOFile, ...]:
    names = sorted(set(re.findall(r'href=["\']([^"\']+_series_matrix\.txt\.gz)["\']', html, re.IGNORECASE)))
    files = []
    for name in names:
        platform_match = re.search(r"(GPL\d+)", name, re.IGNORECASE)
        platform = platform_match.group(1).upper() if platform_match else (platforms[0] if len(platforms) == 1 else None)
        files.append(_file_from_url(urljoin(directory_url, name), "series_matrix", accession, platform))
    return tuple(files)


@dataclass
class NCBIGEOClient:
    """Small client for NCBI's documented GEO SOFT metadata interface."""

    timeout: float = 20.0
    transport: Transport = _default_transport

    def fetch_metadata(self, accession: str) -> GEOSeriesMetadata:
        normalized = normalize_gse_accession(accession)
        series_text = _request_text(self.transport, _accession_url(normalized, "self"), self.timeout)
        series_entities = [entity for entity in _soft_entities(series_text) if entity["type"] == "SERIES"]
        if not series_entities or str(series_entities[0]["accession"]).upper() != normalized:
            raise GEORecordNotFoundError(f"No public GEO Series record was found for {normalized}.")
        series = series_entities[0]
        platforms = tuple(dict.fromkeys(value.upper() for value in _values(series, "Series_platform_id")))
        series_sample_ids = tuple(dict.fromkeys(value.upper() for value in _values(series, "Series_sample_id")))
        supplementary_files = tuple(
            _file_from_url(url, "supplementary", normalized)
            for url in _values(series, "Series_supplementary_file")
        )

        sample_text = _request_text(self.transport, _accession_url(normalized, "gsm"), self.timeout)
        sample_entities = {
            str(entity["accession"]).upper(): entity
            for entity in _soft_entities(sample_text)
            if entity["type"] == "SAMPLE"
        }
        samples: list[GEOSample] = []
        for sample_id in series_sample_ids:
            entity = sample_entities.get(sample_id)
            if entity is None:
                samples.append(GEOSample(sample_id, sample_id, None))
                continue
            platform = _first(entity, "Sample_platform_id").upper() or None
            sample_files = tuple(
                _file_from_url(url, "supplementary", sample_id, platform)
                for url in _values(entity, "Sample_supplementary_file")
            )
            samples.append(GEOSample(sample_id, _first(entity, "Sample_title", sample_id), platform, sample_files))

        matrix_url = _matrix_directory_url(normalized)
        matrix_listing = _request_text(self.transport, matrix_url, self.timeout)
        expression_files = _parse_matrix_files(matrix_listing, matrix_url, normalized, platforms)
        description_parts = _values(series, "Series_summary") or _values(series, "Series_description")
        return GEOSeriesMetadata(
            accession=normalized,
            title=_first(series, "Series_title", normalized),
            description="\n".join(description_parts),
            platforms=platforms,
            samples=tuple(samples),
            expression_files=expression_files,
            supplementary_files=supplementary_files,
        )


def fetch_geo_metadata(
    accession: str,
    *,
    timeout: float = 20.0,
    transport: Transport | None = None,
) -> GEOSeriesMetadata:
    """Convenience interface for fetching basic public GEO Series metadata."""
    client = NCBIGEOClient(timeout=timeout, transport=transport or _default_transport)
    return client.fetch_metadata(accession)
