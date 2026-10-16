from __future__ import annotations

import random
from collections import Counter
from math import isfinite
from typing import Callable, Iterable, Literal, Sequence


PipelineBucket = Literal[
    "write_blocked",
    "admitted_not_retrieved",
    "retrieved_noncausal",
    "retrieved_residual_causal",
]


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    """Return numerator / denominator, using 0.0 when the denominator is empty."""
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def mean(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def bootstrap_ci(
    values: Sequence[float],
    *,
    statistic: Callable[[Iterable[float]], float] = mean,
    confidence: float = 0.95,
    num_resamples: int = 2000,
    seed: int = 13,
) -> dict[str, float]:
    """Percentile bootstrap confidence interval for a scalar statistic."""
    items = [float(value) for value in values]
    if not items:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    if len(items) == 1:
        point = float(statistic(items))
        return {"point": point, "ci_low": point, "ci_high": point}

    rng = random.Random(seed)
    estimates = []
    for _ in range(num_resamples):
        sample = [items[rng.randrange(len(items))] for _ in items]
        estimates.append(float(statistic(sample)))
    estimates.sort()

    alpha = max(0.0, min(1.0, 1.0 - confidence))
    low_index = int((alpha / 2.0) * (len(estimates) - 1))
    high_index = int((1.0 - alpha / 2.0) * (len(estimates) - 1))
    return {
        "point": float(statistic(items)),
        "ci_low": estimates[low_index],
        "ci_high": estimates[high_index],
    }


def poisoned_to_clean_flip(
    baseline_label: str,
    modified_label: str,
    *,
    clean_label: str = "clean",
    poison_label: str = "poisoned",
) -> bool:
    """Label-level flip indicator: poisoned under full context, clean after intervention."""
    return baseline_label == poison_label and modified_label == clean_label


def poisoned_to_clean_flip_rate(
    label_pairs: Iterable[tuple[str, str]],
    *,
    clean_label: str = "clean",
    poison_label: str = "poisoned",
) -> float:
    pairs = list(label_pairs)
    denominator = sum(1 for baseline, _ in pairs if baseline == poison_label)
    numerator = sum(
        poisoned_to_clean_flip(
            baseline,
            modified,
            clean_label=clean_label,
            poison_label=poison_label,
        )
        for baseline, modified in pairs
    )
    return safe_rate(numerator, denominator)


def delta_i(full_poisoned_match_score: float, minus_i_poisoned_match_score: float) -> float:
    """Individual removal effect: Delta_i = s(full) - s(remove i)."""
    return float(full_poisoned_match_score) - float(minus_i_poisoned_match_score)


def delta_ij(full_poisoned_match_score: float, minus_ij_poisoned_match_score: float) -> float:
    """Joint removal effect: Delta_ij = s(full) - s(remove i,j)."""
    return float(full_poisoned_match_score) - float(minus_ij_poisoned_match_score)


def omega_ij(
    full_poisoned_match_score: float,
    minus_i_poisoned_match_score: float,
    minus_j_poisoned_match_score: float,
    minus_ij_poisoned_match_score: float,
) -> float:
    """Pair interaction: Omega_ij = Delta_i + Delta_j - Delta_ij."""
    joint = delta_ij(full_poisoned_match_score, minus_ij_poisoned_match_score)
    left = delta_i(full_poisoned_match_score, minus_i_poisoned_match_score)
    right = delta_i(full_poisoned_match_score, minus_j_poisoned_match_score)
    return left + right - joint


def activation_shift_i(delta_trigger: float, delta_control: float) -> float:
    """Trigger-specific influence: ActivationShift_i = Delta_i(trigger) - Delta_i(control)."""
    return float(delta_trigger) - float(delta_control)


def pipeline_bucket(
    *,
    admitted: bool,
    retrieved: bool,
    causal_effect: float | None = None,
) -> PipelineBucket:
    """Classify a designated unit using its counterfactual poisoned-score drop."""
    if not admitted:
        return "write_blocked"
    if not retrieved:
        return "admitted_not_retrieved"
    if causal_effect is None or not isfinite(causal_effect):
        raise ValueError("A finite causal_effect is required for an admitted and retrieved unit.")
    if causal_effect <= 0:
        return "retrieved_noncausal"
    return "retrieved_residual_causal"


def pipeline_bucket_rates(buckets: Iterable[PipelineBucket]) -> dict[str, float]:
    items = list(buckets)
    counts = Counter(items)
    total = len(items)
    return {
        "write_block_rate": safe_rate(counts["write_blocked"], total),
        "admitted_not_retrieved_rate": safe_rate(counts["admitted_not_retrieved"], total),
        "retrieved_noncausal_rate": safe_rate(counts["retrieved_noncausal"], total),
        "residual_causal_rate": safe_rate(counts["retrieved_residual_causal"], total),
    }

