# Changelog

All notable changes to BioDetective are documented here.

The project follows the principles of [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and intends to use semantic versioning for tagged releases.

## [Unreleased]

### Changed

- Standardized major finding presentation into Observation, Evidence, Interpretation, and Recommendation.
- Audited user-facing language for cautious scientific interpretation.
- Expanded repository documentation and contribution guidance.

## [0.1.0] — 2026-08-20

### Added

- CSV dataset loading, structural validation, and deterministic demonstration data.
- Metadata and expression quality-control analyses.
- Sample similarity, PCA, combined outlier, label-consistency, sex-marker, batch-effect, and confounding analyses.
- Configurable sample suspicion and dataset health scoring.
- Unified fault-tolerant analysis pipeline.
- Overview dashboard, sample explorer, and CSV/JSON/HTML exports.
- Central scientific explanation templates and explicit metadata-role approval.
- Deterministic synthetic generator and benchmark script.
- Metadata-only NCBI GEO Series discovery page.
- Edge-case and performance regression coverage.
- Centralized scientific thresholds and scoring defaults.

### Safety and interpretation

- Analysis modules do not silently modify uploaded expression or metadata values.
- Findings use evidence-based language and recommend researcher review rather than automatic correction.
- Reports exclude the raw uploaded expression matrix.
