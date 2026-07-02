"""Recall@10 and conversational quality metric contracts."""

from __future__ import annotations

from collections.abc import Sequence
from statistics import mean


def recall_at_k(expected: Sequence[str], actual: Sequence[str], k: int = 10) -> float:
    expected_set = {_normalize(value) for value in expected if value}
    if not expected_set:
        return 1.0
    actual_set = {_normalize(value) for value in actual[:k] if value}
    return len(expected_set & actual_set) / len(expected_set)


def overlap_accuracy(expected: Sequence[str], actual: Sequence[str]) -> float:
    expected_set = {_normalize(value) for value in expected if value}
    actual_set = {_normalize(value) for value in actual if value}
    if not expected_set and not actual_set:
        return 1.0
    if not expected_set:
        return 0.0
    return len(expected_set & actual_set) / len(expected_set | actual_set)


def hallucination_rate(actual: Sequence[str], catalog_names: set[str]) -> float:
    if not actual:
        return 0.0
    normalized_catalog = {_normalize(name) for name in catalog_names}
    hallucinated = [
        value for value in actual if _normalize(value) not in normalized_catalog
    ]
    return len(hallucinated) / len(actual)


def safe_mean(values: Sequence[float]) -> float:
    return mean(values) if values else 0.0


def _normalize(value: str) -> str:
    return " ".join(value.lower().strip().split())
