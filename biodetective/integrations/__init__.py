"""External data-source integrations kept separate from core analysis."""

from biodetective.integrations.geo import (
    GEOClient,
    GEOFile,
    GEOIntegrationError,
    GEONetworkError,
    GEORecordNotFoundError,
    GEOSample,
    GEOSeriesMetadata,
    InvalidGEOAccessionError,
    NCBIGEOClient,
    fetch_geo_metadata,
    is_valid_gse_accession,
    normalize_gse_accession,
)

__all__ = [
    "GEOClient",
    "GEOFile",
    "GEOIntegrationError",
    "GEONetworkError",
    "GEORecordNotFoundError",
    "GEOSample",
    "GEOSeriesMetadata",
    "InvalidGEOAccessionError",
    "NCBIGEOClient",
    "fetch_geo_metadata",
    "is_valid_gse_accession",
    "normalize_gse_accession",
]
