"""Deterministic semantic-role suggestions for metadata columns."""

from dataclasses import dataclass
import re

import pandas as pd


SEMANTIC_ROLES = ("condition", "sex", "batch", "age")
ROLE_ORDER = {role: index for index, role in enumerate(SEMANTIC_ROLES)}
NAME_ALIASES = {
    "condition": {"condition", "group", "status", "disease", "diagnosis", "phenotype", "treatment", "class", "cohort"},
    "sex": {"sex", "gender"},
    "batch": {"batch", "plate", "lane", "run", "site", "center", "centre", "chip", "technical"},
    "age": {"age", "age years", "age year", "years old", "age at collection"},
}


@dataclass(frozen=True)
class MetadataRoleSuggestion:
    column: str
    role: str
    score: float
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in SEMANTIC_ROLES:
            raise ValueError(f"unsupported metadata role: {self.role}")
        if not 0 <= self.score <= 1:
            raise ValueError("suggestion score must be between 0 and 1")


@dataclass(frozen=True)
class MetadataMappingApproval:
    mappings: dict[str, str | None]
    approved_roles: tuple[str, ...]

    @property
    def fully_approved(self) -> bool:
        return set(self.approved_roles) == set(SEMANTIC_ROLES)


def _normalized_name(column: object) -> str:
    words = re.sub(r"[^a-z0-9]+", " ", str(column).strip().casefold())
    return re.sub(r"\s+", " ", words).strip()


def _name_score(column: object, role: str) -> tuple[float, list[str]]:
    normalized = _normalized_name(column)
    tokens = set(normalized.split())
    aliases = NAME_ALIASES[role]
    if normalized in aliases:
        return 0.8, [f"column name exactly matches a common {role} label"]
    matches = [alias for alias in aliases if alias in normalized or set(alias.split()).issubset(tokens)]
    if matches:
        return 0.55, [f"column name contains {role}-related term '{sorted(matches)[0]}'"]
    return 0.0, []


def _normalized_values(series: pd.Series) -> set[str]:
    return {str(value).strip().casefold() for value in series.dropna().unique()}


def _value_score(series: pd.Series, role: str) -> tuple[float, list[str]]:
    non_missing = series.dropna()
    if non_missing.empty:
        return 0.0, []
    values = _normalized_values(series)
    unique_count = len(values)
    reasons: list[str] = []
    score = 0.0

    if role == "sex":
        recognized = {"male", "female", "m", "f", "unknown", "other", "intersex", "not reported"}
        if 1 < unique_count <= 8 and values.issubset(recognized):
            score = 0.65
            reasons.append("values resemble commonly recorded sex or gender categories")
    elif role == "batch":
        pattern_fraction = sum(
            bool(re.match(r"^(batch|plate|lane|run|site|center|centre|chip)[ _-]*[a-z0-9]+$", value))
            for value in values
        ) / unique_count
        if 1 < unique_count <= max(20, len(non_missing) // 2) and pattern_fraction >= 0.5:
            score = 0.55
            reasons.append("values resemble technical batch, plate, lane, run, or site labels")
    elif role == "condition":
        condition_terms = {"control", "case", "healthy", "disease", "diseased", "treated", "untreated", "vehicle", "cancer", "normal"}
        matching_values = sum(any(term in value for term in condition_terms) for value in values)
        if 1 < unique_count <= 20 and matching_values / unique_count >= 0.5:
            score = 0.55
            reasons.append("values resemble biological condition, case/control, or treatment groups")
        elif 1 < unique_count <= min(10, max(2, len(non_missing) // 2)):
            score = 0.15
            reasons.append("values form a small categorical grouping")
    elif role == "age":
        numeric = pd.to_numeric(non_missing, errors="coerce")
        numeric_fraction = float(numeric.notna().mean())
        plausible_fraction = float(numeric.dropna().between(0, 120).mean()) if numeric.notna().any() else 0.0
        if numeric_fraction >= 0.9 and plausible_fraction >= 0.9 and numeric.nunique() > 2:
            score = 0.35
            reasons.append("values are predominantly numeric and within a plausible age range")
    return score, reasons


def suggest_metadata_roles(
    metadata: pd.DataFrame,
    minimum_score: float = 0.1,
) -> dict[str, tuple[MetadataRoleSuggestion, ...]]:
    """Return ranked role suggestions for each column without modifying metadata."""
    if not 0 <= minimum_score <= 1:
        raise ValueError("minimum_score must be between 0 and 1")
    suggestions: dict[str, tuple[MetadataRoleSuggestion, ...]] = {}
    for column in metadata.columns:
        column_name = str(column)
        column_suggestions: list[MetadataRoleSuggestion] = []
        if _normalized_name(column) not in {"sample id", "sampleid", "id"}:
            for role in SEMANTIC_ROLES:
                name_score, name_reasons = _name_score(column, role)
                value_score, value_reasons = _value_score(metadata[column], role)
                score = round(min(1.0, name_score + value_score), 4)
                if score >= minimum_score:
                    column_suggestions.append(
                        MetadataRoleSuggestion(column_name, role, score, tuple([*name_reasons, *value_reasons]))
                    )
        column_suggestions.sort(key=lambda item: (-item.score, ROLE_ORDER[item.role]))
        suggestions[column_name] = tuple(column_suggestions)
    return suggestions


def ranked_columns_for_role(
    suggestions: dict[str, tuple[MetadataRoleSuggestion, ...]],
    role: str,
) -> tuple[MetadataRoleSuggestion, ...]:
    """Return all candidate columns for one role in deterministic rank order."""
    if role not in SEMANTIC_ROLES:
        raise ValueError(f"unsupported metadata role: {role}")
    candidates = [suggestion for values in suggestions.values() for suggestion in values if suggestion.role == role]
    return tuple(sorted(candidates, key=lambda item: (-item.score, item.column.casefold())))


def create_mapping_approval(
    metadata: pd.DataFrame,
    mappings: dict[str, str | None],
    approved_roles: list[str] | tuple[str, ...] | set[str],
) -> MetadataMappingApproval:
    """Validate explicit researcher selections; this function never auto-approves them."""
    unknown_roles = set(mappings) - set(SEMANTIC_ROLES)
    if unknown_roles:
        raise ValueError(f"unsupported metadata roles: {', '.join(sorted(unknown_roles))}")
    invalid_columns = {
        str(column)
        for column in mappings.values()
        if column is not None and str(column) not in {str(value) for value in metadata.columns}
    }
    if invalid_columns:
        raise ValueError(f"metadata columns do not exist: {', '.join(sorted(invalid_columns))}")
    invalid_approvals = set(approved_roles) - set(SEMANTIC_ROLES)
    if invalid_approvals:
        raise ValueError(f"unsupported approved roles: {', '.join(sorted(invalid_approvals))}")
    complete_mappings = {role: mappings.get(role) for role in SEMANTIC_ROLES}
    ordered_approvals = tuple(role for role in SEMANTIC_ROLES if role in set(approved_roles))
    return MetadataMappingApproval(complete_mappings, ordered_approvals)
