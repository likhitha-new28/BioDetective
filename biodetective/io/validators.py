"""Non-mutating structural validation for BioDataset objects."""

from dataclasses import dataclass

import pandas as pd

from biodetective.core.models import BioDataset


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    severity: str = "error"


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")


def validate_dataset(dataset: BioDataset) -> ValidationResult:
    """Return all structural validation issues without modifying the dataset."""
    issues: list[ValidationIssue] = []
    expression = dataset.expression
    metadata = dataset.metadata

    if expression.shape[1] == 0:
        issues.append(ValidationIssue("missing_expression_samples", "Expression data has no sample columns."))
    else:
        expression_sample_ids = pd.Series(list(expression.columns), dtype="object")
        missing_expression_ids = expression_sample_ids.isna() | expression_sample_ids.map(
            lambda value: not str(value).strip() if not pd.isna(value) else False
        )
        if missing_expression_ids.any():
            issues.append(
                ValidationIssue("missing_expression_sample_ids", "Expression sample columns contain missing or blank IDs.")
            )
        normalized_expression_ids = expression_sample_ids.loc[~missing_expression_ids].map(str)
        if normalized_expression_ids.duplicated().any():
            issues.append(
                ValidationIssue("duplicate_expression_sample_ids", "Expression data contains duplicate sample IDs.")
            )

    if "sample_id" not in metadata.columns:
        issues.append(ValidationIssue("missing_sample_id", "Metadata must contain a 'sample_id' column."))

    if expression.shape[0] == 0 or expression.shape[1] == 0:
        issues.append(ValidationIssue("empty_expression", "Expression matrix must contain genes and samples."))

    if metadata.empty:
        issues.append(ValidationIssue("empty_metadata", "Metadata must contain at least one sample row."))

    if expression.index.duplicated().any():
        issues.append(ValidationIssue("duplicate_gene_ids", "Expression data contains duplicate gene IDs."))

    if "sample_id" in metadata.columns:
        missing_metadata_ids = metadata["sample_id"].isna() | metadata["sample_id"].map(
            lambda value: not str(value).strip() if not pd.isna(value) else False
        )
        if missing_metadata_ids.any():
            issues.append(
                ValidationIssue("missing_metadata_sample_ids", "Metadata contains missing or blank sample IDs.")
            )
        if metadata["sample_id"].duplicated().any():
            issues.append(ValidationIssue("duplicate_sample_ids", "Metadata contains duplicate sample IDs."))

        expression_ids = {str(value) for value in expression.columns}
        metadata_ids = {str(value) for value in metadata["sample_id"].dropna()}
        if expression_ids != metadata_ids:
            missing = sorted(expression_ids - metadata_ids)
            extra = sorted(metadata_ids - expression_ids)
            details = []
            if missing:
                details.append(f"missing from metadata: {', '.join(missing)}")
            if extra:
                details.append(f"missing from expression: {', '.join(extra)}")
            suffix = f" ({'; '.join(details)})" if details else "."
            issues.append(
                ValidationIssue(
                    "sample_id_mismatch",
                    "Expression and metadata sample IDs do not match" + suffix,
                )
            )

    non_numeric = [column for column in expression.columns if not pd.api.types.is_numeric_dtype(expression[column])]
    if non_numeric:
        issues.append(
            ValidationIssue(
                "non_numeric_expression",
                "Expression values must be numeric; invalid sample columns: " + ", ".join(map(str, non_numeric)) + ".",
            )
        )

    if not expression.empty:
        if expression.isna().all(axis=1).any():
            issues.append(ValidationIssue("empty_expression_rows", "Expression data contains completely empty gene rows."))
        if expression.isna().all(axis=0).any():
            issues.append(ValidationIssue("empty_expression_columns", "Expression data contains completely empty sample columns."))

    if not metadata.empty and metadata.isna().all(axis=1).any():
        issues.append(ValidationIssue("empty_metadata_rows", "Metadata contains completely empty rows."))
    if metadata.shape[1] > 0 and metadata.isna().all(axis=0).any():
        issues.append(ValidationIssue("empty_metadata_columns", "Metadata contains completely empty columns."))

    return ValidationResult(tuple(issues))
