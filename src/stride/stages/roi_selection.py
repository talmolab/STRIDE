"""
Choosing the frame (and the instance) that defines a video's maze ROI.

The ROI model is a top-down SLEAP model with one 23-node "maze" skeleton (16 corners + 7 region
centroids). It is run on a few frames of each video; this module picks the prediction to keep:

* every predicted instance of a frame is evaluated, not only the first one. The model sometimes
  emits a second, shifted, low-score instance with missing corners, and it is not always listed
  second; looking only at ``instances[0]`` rejected whole recording days;
* a frame qualifies when one instance has all 23 nodes (strict) or, failing that, all 16 corners
  (the centroids are derived from the corners downstream, so a missing centroid is acceptable; a
  missing corner is not);
* strict frames take precedence over corner-only frames; within a class the best is chosen by
  (mean point score over the 23 nodes, mean point score over the required nodes, instance score);
* the kept frame carries ONLY the chosen instance.

Frames for the first ("slice") pass are sampled evenly from the middle of the video (20 %-80 %),
not from its start, where the experimenter's hand or the mouse being placed often hides a corner.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from stride.config import STRICT_REQUIRED_23

# Acceptance set = ALL 16 CORNERS. A frame is only accepted if every one of these is present.
REQUIRED_CORNERS_16 = [
    "arm_right.top_left", "arm_right.top_right",
    "junction.top_left", "junction.top_right", "junction.bottom_left", "junction.bottom_right",
    "arm_left.bottom_left", "arm_left.bottom_right",
    "segment1.top_left", "segment1.bottom_left",
    "segment2.top_left", "segment2.bottom_left",
    "segment3.top_left", "segment3.bottom_left",
    "segment4.top_left", "segment4.bottom_left",
]
SCORING_NODES_DEFAULT = STRICT_REQUIRED_23

NUM_SAMPLE_FRAMES = 10       # frames sampled in the slice pass
SAMPLE_START_RATIO = 0.2     # start sampling 20 % into the video
SAMPLE_END_RATIO = 0.8       # stop sampling 80 % into the video


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def get_skeleton_node_names(pred_labels, fallback=None):
    if hasattr(pred_labels, "skeleton") and hasattr(pred_labels.skeleton, "nodes"):
        return [n.name for n in pred_labels.skeleton.nodes]
    try:
        lf0 = pred_labels[0]
        if lf0.instances and hasattr(lf0.instances[0], "skeleton"):
            return [n.name for n in lf0.instances[0].skeleton.nodes]
    except Exception:
        pass
    if fallback is not None:
        return list(fallback)
    raise ValueError("Cannot determine skeleton node order; pass skeleton_nodes explicitly.")


def _build_name_indices(skeleton_nodes):
    exact = {name: i for i, name in enumerate(skeleton_nodes)}
    norm = {_norm(name): i for i, name in enumerate(skeleton_nodes)}
    return exact, norm


def _lookup_idx(name, exact, norm):
    if name in exact:
        return exact[name]
    return norm.get(_norm(name), None)


def _extract_from_structured_point(rec, field):
    try:
        if hasattr(rec, "dtype") and getattr(rec.dtype, "names", None):
            if field in rec.dtype.names:
                return rec[field]
    except Exception:
        pass
    try:
        if isinstance(rec, dict) and (field in rec):
            return rec[field]
    except Exception:
        pass
    return None


def get_point_xy(instance, idx):
    if hasattr(instance, "points"):
        pts = instance.points
        if 0 <= idx < len(pts):
            rec = pts[idx]
            xy = _extract_from_structured_point(rec, "xy")
            if xy is not None:
                arr = np.asarray(xy, dtype=np.float64).reshape(-1)
                if arr.size >= 2:
                    return [float(arr[0]), float(arr[1])]
            try:
                arr = np.asarray(rec, dtype=np.float64).reshape(-1)
                if arr.size >= 2:
                    return [float(arr[0]), float(arr[1])]
            except Exception:
                pass
    try:
        d = instance[idx]
        if isinstance(d, dict):
            if ("x" in d) and ("y" in d):
                return [float(d["x"]), float(d["y"])]
            if "xy" in d:
                arr = np.asarray(d["xy"], dtype=np.float64).reshape(-1)
                if arr.size >= 2:
                    return [float(arr[0]), float(arr[1])]
    except Exception:
        pass
    return [np.nan, np.nan]


def get_point_score(instance, idx):
    if hasattr(instance, "points"):
        pts = instance.points
        if 0 <= idx < len(pts):
            rec = pts[idx]
            s = _extract_from_structured_point(rec, "score")
            if s is not None:
                try:
                    return float(np.asarray(s).reshape(-1)[0])
                except Exception:
                    pass
    try:
        d = instance[idx]
        if isinstance(d, dict) and ("score" in d):
            return float(np.asarray(d["score"]).reshape(-1)[0])
    except Exception:
        pass
    for attr in ("scores", "confidence"):
        if hasattr(instance, attr):
            try:
                arr = np.asarray(getattr(instance, attr), dtype=np.float64).reshape(-1)
                if 0 <= idx < arr.size:
                    return float(arr[idx])
            except Exception:
                pass
    return 0.0


def is_finite_xy(pt):
    arr = np.asarray(pt, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return False
    return np.isfinite(arr[:2]).all()


def _eval_inst(inst, exact, norm, required_nodes):
    scores = []
    for name in required_nodes:
        j = _lookup_idx(name, exact, norm)
        if j is None:
            return False, -1.0, -1.0
        xy = get_point_xy(inst, j)
        if not is_finite_xy(xy):
            return False, -1.0, -1.0
        scores.append(get_point_score(inst, j))
    avg_req = float(np.mean(scores)) if scores else -1.0
    inst_score = float(getattr(inst, "score", 0.0) or 0.0)
    return True, avg_req, inst_score


def _eval_frame(lf, exact, norm, required_nodes):
    """Evaluate EVERY predicted instance of the frame; return the best passing one as a 4th value."""
    best = (False, -1.0, -1.0, None)
    for inst in (getattr(lf, "instances", None) or []):
        ok, avg_req, inst_score = _eval_inst(inst, exact, norm, required_nodes)
        if ok and (best[3] is None or (avg_req, inst_score) > (best[1], best[2])):
            best = (True, avg_req, inst_score, inst)
    return best


def _avg_over_indices(inst, idx_list):
    if not idx_list:
        return -1.0
    scores = []
    for j in idx_list:
        xy = get_point_xy(inst, j)
        if is_finite_xy(xy):
            scores.append(get_point_score(inst, j))
    return float(np.mean(scores)) if scores else -1.0


def select_best_frame_v2(pred_labels, skeleton_nodes=None, max_check=10):
    """Strict 23 -> all 16 corners; best by (avg over 23) -> (avg required) -> instance.score.

    ``max_check=None`` checks every predicted frame (the full-video pass).

    Returns ``(frame_index, labeled_frame)``; the labeled frame keeps ONLY the chosen instance.
    When no frame qualifies returns ``(<video name or "UNKNOWN_VIDEO">, None)``.
    """
    if skeleton_nodes is None:
        skeleton_nodes = get_skeleton_node_names(pred_labels)
    exact, norm = _build_name_indices(skeleton_nodes)

    scoring_idx = [_lookup_idx(n, exact, norm) for n in SCORING_NODES_DEFAULT]
    scoring_idx = [j for j in scoring_idx if j is not None]
    use_all_for_scoring = len(scoring_idx) == 0

    nframes = len(pred_labels) if max_check is None else min(max_check, len(pred_labels))
    best_strict = None   # (avg_scoring, avg_required, inst_score, idx, lf, inst)
    best_relaxed = None

    for idx in range(nframes):
        lf = pred_labels[idx]

        ok_s, avg_req_s, inst_s, inst = _eval_frame(lf, exact, norm, STRICT_REQUIRED_23)
        if ok_s:
            avg_scoring = (_avg_over_indices(inst, scoring_idx)
                           if not use_all_for_scoring else
                           _avg_over_indices(inst, list(range(len(skeleton_nodes)))))
            key = (avg_scoring, avg_req_s, inst_s)
            if (best_strict is None) or (key > best_strict[:3]):
                best_strict = (*key, idx, lf, inst)
            continue  # strict precedence

        ok_r, avg_req_r, inst_r, inst = _eval_frame(lf, exact, norm, REQUIRED_CORNERS_16)
        if ok_r:
            avg_scoring = (_avg_over_indices(inst, scoring_idx)
                           if not use_all_for_scoring else
                           _avg_over_indices(inst, list(range(len(skeleton_nodes)))))
            key = (avg_scoring, avg_req_r, inst_r)
            if (best_relaxed is None) or (key > best_relaxed[:3]):
                best_relaxed = (*key, idx, lf, inst)

    for best in (best_strict, best_relaxed):
        if best is not None:
            lf = best[4]
            lf.instances = [best[5]]   # keep ONLY the chosen instance (the converter reads instances[0])
            return best[3], lf

    try:
        return pred_labels[0].video.filename, None
    except Exception:
        return "UNKNOWN_VIDEO", None


def build_single_frame_labels(lbl, best_lf):
    """A Labels object holding only ``best_lf`` (same videos and skeletons as ``lbl``)."""
    import sleap_io as sio

    LabelsClass = getattr(sio, "Labels", None)
    skels = getattr(lbl, "skeletons", None)
    if skels is None:
        s = getattr(lbl, "skeleton", None)
        skels = [s] if s is not None else None
    vids = getattr(lbl, "videos", None)

    if LabelsClass is not None:
        try:
            return LabelsClass(labeled_frames=[best_lf], videos=vids, skeletons=skels)
        except Exception:
            pass

    lbl.labeled_frames = [best_lf]
    return lbl


def get_video_frame_count(video_path) -> int:
    """Total frame count of a video (OpenCV); 0 when it cannot be read."""
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count


def get_sample_frames(video_path, num_frames: int = NUM_SAMPLE_FRAMES) -> list[int]:
    """Frame indices sampled evenly from the middle portion (20 %-80 %) of the video."""
    total_frames = get_video_frame_count(video_path)
    if total_frames <= 0:
        # Fallback to the first frames if the video cannot be read
        return list(range(num_frames))

    start_frame = int(total_frames * SAMPLE_START_RATIO)
    end_frame = int(total_frames * SAMPLE_END_RATIO)

    if end_frame - start_frame < num_frames:
        # Video too short, sample what we can from the middle
        middle = total_frames // 2
        start_frame = max(0, middle - num_frames // 2)
        end_frame = min(total_frames, start_frame + num_frames)

    # Evenly space frames across the range
    step = max(1, (end_frame - start_frame) // num_frames)
    frames = list(range(start_frame, end_frame, step))[:num_frames]
    return frames
