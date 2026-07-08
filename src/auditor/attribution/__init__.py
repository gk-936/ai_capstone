"""attribution module: Grad-CAM + confound localization (V3)."""

from auditor.attribution.gradcam import (
    compute_gradcam,
    mass_fraction,
    peak_in_region,
    resolve_target_layer,
    salient_crop,
    tag_box,
)

__all__ = [
    "compute_gradcam",
    "mass_fraction",
    "peak_in_region",
    "resolve_target_layer",
    "salient_crop",
    "tag_box",
]
