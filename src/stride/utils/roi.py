"""
Small ROI-file helpers shared by the stages (reporting only).
"""

from pathlib import Path
from typing import Optional

import numpy as np


def roi_stem_angle_deg(yml_path) -> Optional[float]:
    """Direction of the maze stem in the image, read from an ROI yml.

    Returns the angle (degrees) of the vector from the centre of ``segment1`` (start end of
    the stem) to the centre of ``junction``; 0 means the stem runs left to right in the
    image, +-90 means it runs along the image's vertical axis. None when the file or the two
    regions are not available.
    """
    try:
        import yaml

        with open(yml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        cent = {}
        for r in y.get("rois", []):
            pts = np.asarray(r.get("coordinates", []), dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                cent[r.get("name")] = pts.mean(axis=0)
        if "segment1" not in cent or "junction" not in cent:
            return None
        d = cent["junction"] - cent["segment1"]
        if not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-6:
            return None
        return float(np.degrees(np.arctan2(d[1], d[0])))
    except Exception:
        return None
