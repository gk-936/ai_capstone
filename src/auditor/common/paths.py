"""Path and device helpers.

Resolves external data/ paths (handling Windows spaces) and selects the compute device,
degrading gracefully to CPU when CUDA is unavailable so tests and CPU-only runs work.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the repo root, into an absolute Path.

    Absolute paths are returned as-is. Spaces in Windows paths are handled by Path itself.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def resolve_device(requested: str = "cuda") -> str:
    """Return the usable device string, falling back to cpu if CUDA is unavailable."""
    if requested == "cuda":
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return requested
