"""
Convert ROI SLP files to polygon YAML format.

Ported from the ROI scripts of the original study.

This module converts SLEAP keypoint predictions (23 ROI keypoints) to
polygon YAML files with 7 regions (arm_right, junction, arm_left, segment1-4).
"""

from pathlib import Path
from typing import Optional
import re
import numpy as np
import yaml

from stride.config import STRICT_REQUIRED_23, ROI_PALETTE


def convert_batch(
    slp_dir: Path,
    yml_dir: Path,
    overwrite: bool = False,
    patterns: tuple[str, ...] = ("*.preds.v2.best1.slp", "*.best1.slp"),
) -> dict:
    """
    Convert all SLP files in a directory to YAML.

    Args:
        slp_dir: Directory containing .slp files
        yml_dir: Output directory for .yml files
        overwrite: Whether to overwrite existing files
        patterns: Glob patterns to match SLP files

    Returns:
        Summary dict with counts
    """
    yml_dir.mkdir(parents=True, exist_ok=True)

    # Find all matching SLP files
    slp_files = []
    for pattern in patterns:
        slp_files.extend(slp_dir.glob(pattern))
    slp_files = sorted(set(slp_files))

    results = {"total": len(slp_files), "ok": 0, "skip": 0, "missing_keypoints": 0, "failed": 0}

    for slp_path in slp_files:
        yml_path = yml_dir / (slp_path.stem + ".yml")

        if yml_path.exists() and not overwrite:
            results["skip"] += 1
            continue

        try:
            slp_to_roi_yaml(str(slp_path), str(yml_path))
            results["ok"] += 1
        except ValueError as e:
            if "Missing required keypoints" in str(e):
                results["missing_keypoints"] += 1
            else:
                results["failed"] += 1
        except Exception:
            results["failed"] += 1

    return results


def slp_to_roi_yaml(slp_path: str, yaml_path: str) -> None:
    """
    Convert a single SLP file to ROI YAML.

    Args:
        slp_path: Path to input .slp file
        yaml_path: Path to output .yml file

    Raises:
        ValueError: If required keypoints are missing
    """
    import sleap_io as sio

    try:
        labels = sio.load_file(slp_path)
    except OSError:  # the embedded video path is unreadable on this machine: the points suffice
        labels = sio.load_slp(str(slp_path), open_videos=False)

    # Validate keypoints: the 16 corners are required; the 7 region centroids are not used by
    # the polygon builder (same rule as the frame selector, stride.stages.roi_selection).
    valid, msg = validate_corner_keypoints(labels)
    if not valid:
        raise ValueError(f"Missing required keypoints: {msg}")

    lf = labels[0]
    inst = lf.instances[0]

    # Get frame dimensions
    video = labels.videos[0] if getattr(labels, "videos", None) else None
    try:
        shape = getattr(video, "shape", None) if video else None
    except Exception:
        shape = None
    if shape is not None and len(shape) >= 3 and all(v is not None for v in shape[-3:-1]):
        if len(shape) == 4:
            _, H, W, _ = shape
        else:
            H, W, _ = shape
    else:
        H, W = 4096, 4096

    # Build skeleton lookup
    skel_names = get_skeleton_node_names(labels)
    exact = {name: i for i, name in enumerate(skel_names)}
    norm = {_norm(name): i for i, name in enumerate(skel_names)}

    def xy(name):
        return _xy_by_name(inst, exact, norm, name)

    # Build polygon corners for horizontal T-maze
    # Junction
    J_TL = xy("junction.top_left")
    J_TR = xy("junction.top_right")
    J_BL = xy("junction.bottom_left")
    J_BR = xy("junction.bottom_right")

    # arm_right: TL, TR from keypoints; BL, BR shared with junction
    AR_TL = xy("arm_right.top_left")
    AR_TR = xy("arm_right.top_right")
    AR_BL = J_TL
    AR_BR = J_TR

    # arm_left: BL, BR from keypoints; TL, TR shared with junction
    AL_TL = J_BL
    AL_TR = J_BR
    AL_BL = xy("arm_left.bottom_left")
    AL_BR = xy("arm_left.bottom_right")

    # Segment 4
    S4_TL = xy("segment4.top_left")
    S4_BL = xy("segment4.bottom_left")
    S4_TR = J_TL
    S4_BR = J_BL

    # Segment 3
    S3_TL = xy("segment3.top_left")
    S3_BL = xy("segment3.bottom_left")
    S3_TR = S4_TL
    S3_BR = S4_BL

    # Segment 2
    S2_TL = xy("segment2.top_left")
    S2_BL = xy("segment2.bottom_left")
    S2_TR = S3_TL
    S2_BR = S3_BL

    # Segment 1
    S1_TL = xy("segment1.top_left")
    S1_BL = xy("segment1.bottom_left")
    S1_TR = S2_TL
    S1_BR = S2_BL

    # Clamp all corners to frame bounds
    def clamp(pt):
        if pt is None:
            return None
        x, y = pt
        return [float(min(max(x, 0.0), W - 1)), float(min(max(y, 0.0), H - 1))]

    polys = {
        "arm_right": [clamp(AR_TL), clamp(AR_BL), clamp(AR_BR), clamp(AR_TR)],
        "junction": [clamp(J_TL), clamp(J_BL), clamp(J_BR), clamp(J_TR)],
        "arm_left": [clamp(AL_TL), clamp(AL_BL), clamp(AL_BR), clamp(AL_TR)],
        "segment4": [clamp(S4_TL), clamp(S4_BL), clamp(S4_BR), clamp(S4_TR)],
        "segment3": [clamp(S3_TL), clamp(S3_BL), clamp(S3_BR), clamp(S3_TR)],
        "segment2": [clamp(S2_TL), clamp(S2_BL), clamp(S2_BR), clamp(S2_TR)],
        "segment1": [clamp(S1_TL), clamp(S1_BL), clamp(S1_BR), clamp(S1_TR)],
    }

    # Validate polygons
    for name, pts in polys.items():
        if any(p is None for p in pts):
            raise ValueError(f"Cannot construct polygon '{name}' - missing corners")

    # Build YAML structure
    video_path = video.filename if (video and hasattr(video, "filename")) else ""

    rois = []
    order = ["arm_right", "junction", "arm_left", "segment4", "segment3", "segment2", "segment1"]
    for i, name in enumerate(order, start=1):
        coords = [[float(x), float(y)] for (x, y) in polys[name]]
        rois.append({
            "id": i,
            "name": name,
            "type": "polygon",
            "coordinates": coords,
            "color": ROI_PALETTE.get(name, "#808080"),
            "properties": {
                "vertex_count": 4,
                "perimeter": _perimeter(coords),
                "area": _area(coords),
            },
        })

    data = {
        "image_file": video_path,
        "roi_count": len(rois),
        "rois": rois,
        "metadata": {
            "created_with": "tmaze-pipeline",
            "format_version": "1.0",
        },
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


# Helper functions

def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def get_skeleton_node_names(labels) -> list[str]:
    if hasattr(labels, "skeleton") and hasattr(labels.skeleton, "nodes"):
        return [n.name for n in labels.skeleton.nodes]
    lf0 = labels[0]
    if lf0.instances and hasattr(lf0.instances[0], "skeleton"):
        return [n.name for n in lf0.instances[0].skeleton.nodes]
    raise ValueError("Cannot determine skeleton node names")


def _xy_by_name(inst, exact, norm, name):
    j = exact.get(name) or norm.get(_norm(name))
    if j is None:
        return None
    return _get_point_xy(inst, j)


def _get_point_xy(instance, idx):
    if hasattr(instance, "points"):
        pts = instance.points
        if 0 <= idx < len(pts):
            rec = pts[idx]
            if hasattr(rec, "dtype") and hasattr(rec.dtype, "names") and "xy" in rec.dtype.names:
                xy = rec["xy"]
                arr = np.asarray(xy, dtype=np.float64).reshape(-1)
                if arr.size >= 2:
                    pt = [float(arr[0]), float(arr[1])]
                    if np.isfinite(pt).all():
                        return pt
            try:
                arr = np.asarray(rec, dtype=np.float64).reshape(-1)
                if arr.size >= 2:
                    pt = [float(arr[0]), float(arr[1])]
                    if np.isfinite(pt).all():
                        return pt
            except Exception:
                pass
    return None


def validate_corner_keypoints(labels) -> tuple[bool, str]:
    """Check that all 16 maze corners are present and finite (centroids are optional)."""
    from stride.stages.roi_selection import REQUIRED_CORNERS_16
    return _validate_keypoints(labels, REQUIRED_CORNERS_16)


def validate_23_keypoints(labels) -> tuple[bool, str]:
    """Check if all 23 keypoints (16 corners + 7 centroids) are present and finite."""
    return _validate_keypoints(labels, STRICT_REQUIRED_23)


def _validate_keypoints(labels, required) -> tuple[bool, str]:
    if len(labels) < 1 or not labels[0].instances:
        return False, "No instances in SLP"

    lf = labels[0]
    inst = lf.instances[0]
    skel_names = get_skeleton_node_names(labels)
    exact = {name: i for i, name in enumerate(skel_names)}
    norm = {_norm(name): i for i, name in enumerate(skel_names)}

    missing = []
    for name in required:
        pt = _xy_by_name(inst, exact, norm, name)
        if pt is None:
            missing.append(name)

    if missing:
        preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
        return False, f"Missing {len(missing)} keypoints: {preview}"
    return True, "OK"


def _perimeter(poly):
    per = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        per += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return per


def _area(poly):
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return abs(s) * 0.5
