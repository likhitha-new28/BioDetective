"""Friendly exceptions raised by BioDetective."""


class BioDetectiveError(Exception):
    """Base exception for expected BioDetective errors."""


class DataLoadError(BioDetectiveError):
    """Raised when an uploaded dataset file cannot be loaded safely."""
