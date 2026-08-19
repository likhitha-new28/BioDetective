"""Synthetic datasets and benchmarking helpers."""

from biodetective.synthetic.benchmark import classification_metrics, run_synthetic_benchmark
from biodetective.synthetic.generator import ConfoundingMode, generate_synthetic_dataset

__all__ = [
    "ConfoundingMode",
    "classification_metrics",
    "generate_synthetic_dataset",
    "run_synthetic_benchmark",
]
