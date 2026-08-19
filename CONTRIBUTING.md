# Contributing to BioDetective

Thank you for helping improve BioDetective. Contributions should preserve its role as a cautious, non-destructive research data-quality tool.

## Development setup

1. Fork and clone the repository.
2. Create a Python 3.11+ virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Run `python -m pytest` before making changes.

## Contribution workflow

1. Create a focused branch.
2. Keep changes small and modular.
3. Add or update tests for changed behavior.
4. Run the complete test suite.
5. Describe the scientific and user-facing impact in the pull request.

Avoid unrelated refactoring in detector changes. Preserve input data and keep optional analyses fault-tolerant.

## Scientific language

User-facing findings must separate:

- **Observation:** what the analysis observed;
- **Evidence:** relevant values, samples, thresholds, or comparisons;
- **Interpretation:** why the observation may matter and plausible alternatives;
- **Recommendation:** a cautious next step for researcher review.

Do not state that a sample is definitely duplicated, mislabeled, biologically male/female, invalid, or unsuitable solely from a BioDetective signal. Prefer language such as “potential duplicate or highly similar samples,” “possible label inconsistency,” and “review recommended.”

## Tests

Run all tests:

```bash
python -m pytest
```

New thresholds must be configurable and defined through the centralized defaults in `biodetective/core/config.py`. Synthetic tests must use deterministic seeds. Do not change an algorithm solely to improve synthetic benchmark metrics.

## Data and secrets

Never commit:

- uploaded or identifiable research datasets;
- downloaded GEO expression or supplementary files;
- credentials, API keys, tokens, or Streamlit secrets;
- `.env` files, virtual environments, caches, or generated reports containing research data.

Tests should use the deterministic generator or small fabricated fixtures. The files under `sample_data/` must remain synthetic and non-identifiable.

## Reporting security or privacy concerns

Do not open a public issue containing credentials, private study data, or participant information. Remove sensitive material from the repository history and contact the repository maintainer through a private channel.
