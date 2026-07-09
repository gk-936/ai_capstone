"""mitigation module: Deep Feature Reweighting (V5)."""

from auditor.mitigation.baselines import current_head, erm_predictions, fit_jtt_head
from auditor.mitigation.dfr import (
    apply_dfr,
    extract_features,
    fit_dfr_head,
    group_balanced_sample,
)

__all__ = [
    "apply_dfr", "extract_features", "fit_dfr_head", "group_balanced_sample",
    "current_head", "erm_predictions", "fit_jtt_head",
]
