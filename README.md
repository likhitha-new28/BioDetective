# BioDetective

BioDetective is a local-first quality-control application for auditing gene-expression matrices and their sample metadata. It gathers multiple signals that can help researchers identify records that deserve review, while keeping observations separate from scientific interpretation and never modifying uploaded measurements automatically.

## Project purpose

Gene-expression studies often combine molecular measurements, manually curated metadata, and files produced across instruments or processing batches. Sample identifiers, category labels, missing values, technical effects, or unusually similar profiles can introduce ambiguity that is difficult to notice by inspecting tables alone.

BioDetective organizes these checks into a reproducible pipeline and presents the resulting evidence through an interactive Streamlit dashboard and downloadable reports.

## Problem statement

Data-quality concerns such as sample mix-ups, copied metadata, confounding, batch-associated variation, or atypical molecular profiles can affect downstream conclusions. No single statistical signal proves that a sample is wrong. BioDetective therefore:

- validates data structure before analysis;
- reports multiple independent or complementary observations;
- uses cautious, evidence-based wording;
- requires researcher approval for semantic metadata mappings;
- provides recommendations for review rather than automatic corrections.

## Screenshots

| Overview dashboard | Sample explorer |
| --- | --- |
| _Screenshot placeholder — add `docs/screenshots/overview.png`._ | _Screenshot placeholder — add `docs/screenshots/sample-explorer.png`._ |

| Metadata mapping | GEO import |
| --- | --- |
| _Screenshot placeholder — add `docs/screenshots/metadata-mapping.png`._ | _Screenshot placeholder — add `docs/screenshots/geo-import.png`._ |

## Features

- CSV loading with friendly structural validation
- Deterministic metadata-role suggestions for condition, sex, batch, and age
- Explicit researcher approval or override of every semantic mapping
- Missing, duplicate, inconsistent, constant, high-cardinality, and imbalanced metadata checks
- Missing, infinite, zero-variance, and very-low-variance expression checks
- Pearson and Spearman sample-correlation analysis
- Cautious potential duplicate or highly similar sample evidence
- Configurable PCA preprocessing and visualization
- Robust PCA-distance and Isolation Forest outlier evidence
- Centroid and cross-validated label-consistency analysis
- Configurable sex-associated marker consistency analysis
- Batch association and effect-size summaries
- Condition-versus-batch confounding analysis
- Interpretable sample suspicion and dataset health scores
- Overview dashboard and per-sample explorer
- CSV, JSON, and standalone HTML reports without raw expression export
- Deterministic synthetic datasets with planted quality problems
- Synthetic benchmark reporting precision, recall, and F1 where meaningful
- Metadata-only GEO Series discovery through NCBI, with explicit file selection

## Installation

Requirements:

- Python 3.11 or newer
- Internet access only when using GEO metadata discovery

Create an isolated environment and install dependencies:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick start

Start the application:

```bash
streamlit run app.py
```

Then upload:

1. `sample_data/expression_example.csv` as the expression matrix.
2. `sample_data/metadata_example.csv` as the sample metadata.
3. Review and approve or override every suggested metadata mapping.
4. Select **Run BioDetective Analysis**.

The included example is deterministic synthetic data generated with random seed `42`; it is not a research dataset.

## Deploy to the web

The included `render.yaml` deploys BioDetective as a responsive Streamlit web service for mobile and desktop browsers. In Render, choose **New > Blueprint**, connect this repository, and apply the detected `biodetective` service. Render supplies the public HTTPS URL and deploys new commits automatically.

For Streamlit Community Cloud, select `app.py` as the entrypoint and Python 3.12. No secrets are required for the core application. Uploaded research files stay in the active Streamlit session and are not committed to the repository.

To run tests:

```bash
python -m pytest
```

## Expected data format

### Expression CSV

Rows are genes or features. Columns after `gene_id` are sample identifiers. Values must be numeric.

```csv
gene_id,S01,S02,S03
TP53,10,12,8
BRCA1,5,6,7
```

### Metadata CSV

Each row describes one sample. The `sample_id` values must align with the expression sample columns. Other column names are researcher-defined.

```csv
sample_id,condition,sex,batch,age
S01,Healthy,Female,Batch1,42
S02,Healthy,Male,Batch1,51
S03,Cancer,Female,Batch2,47
```

BioDetective does not silently normalize labels, remove genes, transform values, correct batches, or change metadata.

## Architecture

```text
CSV uploads ──> biodetective.io ──> BioDataset ──> validation
                                              │
                                              v
                                      unified pipeline
                                              │
                   ┌──────────────────────────┼──────────────────────────┐
                   v                          v                          v
          metadata/expression QC     similarity/PCA/outliers   label/sex/batch/confounding
                   └──────────────────────────┼──────────────────────────┘
                                              v
                                  suspicion and health scoring
                                              │
                              ┌───────────────┴───────────────┐
                              v                               v
                     Streamlit dashboard             CSV/JSON/HTML reports

NCBI GEO ──> biodetective.integrations.geo ──> metadata and file discovery only
```

Main packages:

- `biodetective/core`: models, centralized defaults, and pipeline orchestration
- `biodetective/io`: CSV loading and non-mutating validation
- `biodetective/analysis`: independent quality-control analyses
- `biodetective/scoring`: sample suspicion and dataset health scoring
- `biodetective/reporting`: explanations and report generation
- `biodetective/synthetic`: deterministic generators and benchmarking
- `biodetective/integrations`: external metadata integrations kept separate from core analysis

## Analysis methods

| Area | Method | Interpretation boundary |
| --- | --- | --- |
| Metadata | Completeness, duplicate rows/IDs, normalized category comparison, cardinality, imbalance | Flags records or fields for review; does not infer the correct value. |
| Expression | Missing/infinite values and gene variance summaries | Reports measurements and features; does not impute or filter automatically. |
| Similarity | Pearson or Spearman sample correlation | High correlation is potential duplicate or similarity evidence, not proof of identity. |
| PCA | Optional zero-variance removal and top-variable-gene selection on an analysis copy | Coordinates describe major variance directions and do not establish causality. |
| Outliers | Robust PCA distance and Isolation Forest | Unusual profiles may reflect biology, technical variation, or metadata concerns. |
| Label consistency | Group-centroid similarity and leakage-safe cross-validated logistic regression | Disagreement is possible label inconsistency evidence, not an automatic relabeling decision. |
| Sex-marker consistency | Configurable X- and Y-associated marker patterns | Marker evidence is assay-dependent and does not determine biological sex with certainty. |
| Batch effects | PCA-component association, ANOVA or robust alternative, and effect size | Association does not distinguish technical effects from confounded biology. |
| Confounding | Contingency tables, chi-square where suitable, conditional proportions, and Cramer's V | Strong association can limit separability within the available dataset. |
| Scoring | Configurable weighted evidence and transparent deductions | Scores prioritize review; they are not probabilities of error. |

Scientific thresholds and scoring defaults are centralized in `biodetective/core/config.py` and remain configurable through analysis-specific configuration objects.

## Synthetic benchmark

The generator can create clean datasets and plant exact duplicates, near duplicates, expression outliers, condition-label swaps, batch effects, and partial or complete condition/batch confounding. Every generated dataset includes ground truth.

Run the deterministic benchmark:

```bash
python scripts/benchmark.py
```

The script compares planted and detected sample-level issues and reports true positives, false positives, false negatives, precision, recall, and F1 where those metrics are meaningful. Batch and confounding results are reported as dataset-level risk comparisons. The detectors are not tuned solely for the synthetic benchmark.

## GEO metadata discovery

Open the **GEO Import** page in Streamlit and enter a public GSE accession. BioDetective retrieves basic Series and Sample metadata plus discoverable SeriesMatrix and supplementary-file links. It does not download expression data, infer ambiguous mappings, or automatically start analysis. Multiple platforms, samples, or files require explicit selection.

## Limitations

- Results depend on sample size, study design, preprocessing, metadata quality, and configured thresholds.
- Small or highly imbalanced classes can make cross-validation unavailable or unstable.
- PCA, correlations, and anomaly detectors may highlight genuine biological heterogeneity.
- Batch association can be indistinguishable from biology when variables are confounded.
- Sex-associated marker analysis depends on marker availability and assay context.
- GEO records vary in annotation quality and file organization; BioDetective does not infer ambiguous mappings.
- BioDetective does not replace provenance records, laboratory review, domain expertise, or a prespecified statistical analysis plan.

## Scientific disclaimer

BioDetective is a research data-quality screening tool. Its findings, risk labels, and scores are evidence summaries for review—not diagnoses, proof of sample identity, proof of metadata error, or instructions to exclude data. Researchers remain responsible for validating observations against source records, experimental design, assay context, and appropriate independent evidence.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and data-handling expectations. BioDetective is available under the [MIT License](LICENSE).
