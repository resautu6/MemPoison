"""Core utilities for the MemPoison benchmark scaffold."""

from .metrics import (
    AttackMetrics,
    AttackRunCounts,
    DefenseMetrics,
    DefenseRunCounts,
    ExperimentTrajectoryRow,
    compute_attack_metrics,
    compute_defense_metrics,
)
from .metric_formulas import (
    activation_shift_i,
    bootstrap_ci,
    delta_i,
    delta_ij,
    omega_ij,
    pipeline_bucket,
    pipeline_bucket_rates,
    poisoned_to_clean_flip_rate,
)

__all__ = [
    "AttackMetrics",
    "AttackRunCounts",
    "DefenseMetrics",
    "DefenseRunCounts",
    "ExperimentTrajectoryRow",
    "compute_attack_metrics",
    "compute_defense_metrics",
    "activation_shift_i",
    "bootstrap_ci",
    "delta_i",
    "delta_ij",
    "omega_ij",
    "pipeline_bucket",
    "pipeline_bucket_rates",
    "poisoned_to_clean_flip_rate",
]
