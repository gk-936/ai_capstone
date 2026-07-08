"""discovery module: VLM-naming crop export + scoring (V4)."""

from auditor.discovery.score_naming import (
    load_and_score,
    names_confound,
    names_tube,
    score,
)

__all__ = ["load_and_score", "names_confound", "names_tube", "score"]
