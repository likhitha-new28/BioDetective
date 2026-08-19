"""Reproducible synthetic gene-expression datasets with optional planted issues."""

from typing import Literal

import numpy as np
import pandas as pd


ConfoundingMode = Literal["none", "partial", "complete"]


def _validate_parameters(
    n_genes: int,
    n_samples: int,
    n_conditions: int,
    n_batches: int,
    n_exact_duplicates: int,
    n_near_duplicates: int,
    n_outliers: int,
    n_label_swaps: int,
    batch_effect_strength: float,
    confounding: str,
    confounding_strength: float,
) -> None:
    if min(n_genes, n_samples, n_conditions, n_batches) < 1:
        raise ValueError("gene, sample, condition, and batch counts must be positive")
    if n_conditions > n_samples or n_batches > n_samples:
        raise ValueError("condition and batch counts cannot exceed the sample count")
    if min(n_exact_duplicates, n_near_duplicates, n_outliers, n_label_swaps) < 0:
        raise ValueError("planted issue counts must be non-negative")
    if 2 * (n_exact_duplicates + n_near_duplicates) + n_outliers > n_samples:
        raise ValueError("not enough samples to plant disjoint duplicates and outliers")
    if n_label_swaps > n_samples:
        raise ValueError("label swap count cannot exceed the sample count")
    if n_label_swaps and n_conditions < 2:
        raise ValueError("label swaps require at least two conditions")
    if batch_effect_strength < 0:
        raise ValueError("batch_effect_strength must be non-negative")
    if confounding not in {"none", "partial", "complete"}:
        raise ValueError("confounding must be 'none', 'partial', or 'complete'")
    if not 0 <= confounding_strength <= 1:
        raise ValueError("confounding_strength must be between 0 and 1")


def _assign_batches(
    conditions: np.ndarray,
    n_batches: int,
    mode: ConfoundingMode,
    strength: float,
    rng: np.random.Generator,
) -> np.ndarray:
    condition_numbers = np.array([int(value.removeprefix("Condition")) - 1 for value in conditions])
    preferred = condition_numbers % n_batches
    if mode == "complete":
        batch_numbers = preferred
    elif mode == "partial":
        use_preferred = rng.random(len(conditions)) < strength
        random_batches = rng.integers(0, n_batches, size=len(conditions))
        if n_batches > 1:
            alternative = (preferred + rng.integers(1, n_batches, size=len(conditions))) % n_batches
            random_batches = np.where(random_batches == preferred, alternative, random_batches)
        batch_numbers = np.where(use_preferred, preferred, random_batches)
    else:
        batch_numbers = np.arange(len(conditions)) % n_batches
        rng.shuffle(batch_numbers)
    return np.array([f"Batch{number + 1}" for number in batch_numbers])


def generate_synthetic_dataset(
    n_genes: int = 100,
    n_samples: int = 20,
    n_conditions: int = 2,
    n_batches: int = 2,
    random_state: int = 42,
    *,
    n_exact_duplicates: int = 0,
    n_near_duplicates: int = 0,
    near_duplicate_noise: float = 0.02,
    n_outliers: int = 0,
    outlier_strength: float = 8.0,
    n_label_swaps: int = 0,
    batch_effect_strength: float = 0.0,
    confounding: ConfoundingMode = "none",
    confounding_strength: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Return expression, metadata, and exact ground truth for planted problems."""
    _validate_parameters(
        n_genes,
        n_samples,
        n_conditions,
        n_batches,
        n_exact_duplicates,
        n_near_duplicates,
        n_outliers,
        n_label_swaps,
        batch_effect_strength,
        confounding,
        confounding_strength,
    )
    if near_duplicate_noise <= 0:
        raise ValueError("near_duplicate_noise must be positive")
    if outlier_strength <= 0:
        raise ValueError("outlier_strength must be positive")

    rng = np.random.default_rng(random_state)
    sample_ids = np.array([f"S{index:03d}" for index in range(1, n_samples + 1)])
    gene_ids = [f"GENE{index:04d}" for index in range(1, n_genes + 1)]
    conditions = np.array([f"Condition{index % n_conditions + 1}" for index in range(n_samples)])
    rng.shuffle(conditions)
    batches = _assign_batches(conditions, n_batches, confounding, confounding_strength, rng)

    values = rng.normal(loc=8.0, scale=1.0, size=(n_genes, n_samples))
    condition_effect_gene_count = max(1, n_genes // max(4, n_conditions * 2))
    condition_effects: dict[str, list[str]] = {}
    for condition_index in range(n_conditions):
        start = condition_index * condition_effect_gene_count
        stop = min(start + condition_effect_gene_count, n_genes)
        affected_indices = np.arange(start, stop)
        condition_name = f"Condition{condition_index + 1}"
        values[np.ix_(affected_indices, conditions == condition_name)] += 2.5
        condition_effects[condition_name] = [gene_ids[index] for index in affected_indices]

    batch_gene_count = max(1, n_genes // 5)
    batch_gene_ids = gene_ids[-batch_gene_count:]
    batch_shifts = np.linspace(-batch_effect_strength, batch_effect_strength, n_batches)
    for batch_index, shift in enumerate(batch_shifts):
        values[-batch_gene_count:, batches == f"Batch{batch_index + 1}"] += shift

    available_indices = list(rng.permutation(n_samples))
    exact_pairs: list[dict[str, str]] = []
    near_pairs: list[dict[str, object]] = []
    for _ in range(n_exact_duplicates):
        source_index, duplicate_index = available_indices.pop(), available_indices.pop()
        values[:, duplicate_index] = values[:, source_index]
        exact_pairs.append(
            {"source_sample_id": str(sample_ids[source_index]), "duplicate_sample_id": str(sample_ids[duplicate_index])}
        )
    for _ in range(n_near_duplicates):
        source_index, duplicate_index = available_indices.pop(), available_indices.pop()
        values[:, duplicate_index] = values[:, source_index] + rng.normal(0, near_duplicate_noise, n_genes)
        near_pairs.append(
            {
                "source_sample_id": str(sample_ids[source_index]),
                "duplicate_sample_id": str(sample_ids[duplicate_index]),
                "noise_standard_deviation": near_duplicate_noise,
            }
        )

    outlier_ids: list[str] = []
    for _ in range(n_outliers):
        sample_index = available_indices.pop()
        direction = rng.choice([-1.0, 1.0], size=n_genes)
        values[:, sample_index] += direction * outlier_strength
        outlier_ids.append(str(sample_ids[sample_index]))

    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "condition": conditions,
            "sex": np.where(np.arange(n_samples) % 2 == 0, "Female", "Male"),
            "batch": batches,
        }
    )
    label_swap_records: list[dict[str, str]] = []
    if n_label_swaps:
        swap_indices = rng.choice(n_samples, size=n_label_swaps, replace=False)
        condition_names = [f"Condition{index + 1}" for index in range(n_conditions)]
        for sample_index in swap_indices:
            original = str(metadata.at[sample_index, "condition"])
            replacement = condition_names[(condition_names.index(original) + 1) % n_conditions]
            metadata.at[sample_index, "condition"] = replacement
            label_swap_records.append(
                {
                    "sample_id": str(sample_ids[sample_index]),
                    "original_condition": original,
                    "recorded_condition": replacement,
                }
            )

    expression = pd.DataFrame(values, index=gene_ids, columns=sample_ids)
    expression.index.name = "gene_id"
    exact_affected = sorted(
        {sample_id for pair in exact_pairs for sample_id in (pair["source_sample_id"], pair["duplicate_sample_id"])}
    )
    near_affected = sorted(
        {str(sample_id) for pair in near_pairs for sample_id in (pair["source_sample_id"], pair["duplicate_sample_id"])}
    )
    confounding_table = pd.crosstab(metadata["condition"], metadata["batch"]).to_dict()
    ground_truth: dict[str, object] = {
        "random_state": random_state,
        "clean": not any(
            [n_exact_duplicates, n_near_duplicates, n_outliers, n_label_swaps, batch_effect_strength, confounding != "none"]
        ),
        "condition_effects": condition_effects,
        "exact_duplicates": {"pairs": exact_pairs, "affected_sample_ids": exact_affected},
        "near_duplicates": {"pairs": near_pairs, "affected_sample_ids": near_affected},
        "outliers": {"affected_sample_ids": sorted(outlier_ids), "strength": outlier_strength},
        "label_swaps": {
            "swaps": label_swap_records,
            "affected_sample_ids": sorted(record["sample_id"] for record in label_swap_records),
        },
        "batch_effect": {
            "strength": batch_effect_strength,
            "affected_gene_ids": batch_gene_ids if batch_effect_strength else [],
            "batch_shifts": {
                f"Batch{index + 1}": float(shift) for index, shift in enumerate(batch_shifts)
            },
        },
        "confounding": {
            "mode": confounding,
            "strength": confounding_strength if confounding == "partial" else (1.0 if confounding == "complete" else 0.0),
            "condition_batch_counts": confounding_table,
        },
    }
    return expression, metadata, ground_truth
