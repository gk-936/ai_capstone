"""Single entry point for deterministic seeding across the stack.

Call `set_seed(cfg.seed)` once at the start of every run. torch is imported lazily so
that config/logging tests don't require a torch install.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and (if available) torch/CUDA for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
