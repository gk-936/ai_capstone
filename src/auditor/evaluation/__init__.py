"""evaluation module: reliance-gap and metrics."""

from auditor.evaluation.reliance import (
    balanced_eval_subset,
    evaluate_accuracy,
    reliance_gap,
)

__all__ = ["balanced_eval_subset", "evaluate_accuracy", "reliance_gap"]
