"""Where the maze sits in the image on each recording day, read from the ROI files: a second check that the
camera was not moved. Report and warning only; no output of the analysis depends on it.

Per day: the median position of every region corner over the day's DETECTED ROI files (clips that borrowed the
session's consensus ROI are left out), the largest distance of a corner from where it was on the cohort's first
day, and the direction of the stem (segment1 -> junction).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

import numpy as np

COLS = ["day", "folder", "roi_files", "consensus_files", "corner_drift_max_px", "corner_drift_mean_px", "stem_angle_deg",
        "stem_angle_change_deg", "status"]


def _corners(yml_path) -> Optional[dict]:
    import yaml
    try:
        with open(yml_path, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f) or {}
        out = {}
        for r in y.get("rois", []):
            pts = np.asarray(r.get("coordinates", []), dtype=float)
            if pts.ndim == 2 and pts.shape[0] >= 3 and np.all(np.isfinite(pts)):
                out[str(r.get("name"))] = pts
        return out or None
    except Exception:
        return None


def roi_position_report(day_dirs: dict, roi_dirs, camera, out_csv=None, drift_warn_px: float = 25.0,
                        stem_angle_warn_deg: float = 1.0) -> dict:
    """{"days": [row, ...], "warnings": [...], "reference_day": N}; writes `out_csv` when given."""
    from stride.stages.decision_analysis import resolve_roi_yml
    from stride.stages.roi_consensus import is_borrowed_roi
    from stride.utils.roi import roi_stem_angle_deg

    per_day = {}
    for n, d in sorted(day_dirs.items()):
        stacks, angles, n_cons = {}, [], 0
        for v in camera.videos(d):
            y = resolve_roi_yml(v, roi_dirs)
            if y is None:
                continue
            if is_borrowed_roi(y):
                n_cons += 1
                continue
            c = _corners(y)
            if c is None:
                continue
            for name, pts in c.items():
                stacks.setdefault((name, pts.shape[0]), []).append(pts)
            a = roi_stem_angle_deg(y)
            if a is not None:
                angles.append(a)
        med = {k: np.median(np.stack(v), axis=0) for k, v in stacks.items()}
        n_files = max((len(v) for v in stacks.values()), default=0)
        per_day[n] = dict(folder=d.name, med=med, n=n_files, n_cons=n_cons, angle=float(np.median(angles)) if angles else None)
    ref_day = next((n for n in sorted(per_day) if per_day[n]["n"] > 0), None)
    rows, warnings = [], []
    for n in sorted(per_day):
        p = per_day[n]
        row = dict(day=n, folder=p["folder"], roi_files=p["n"], consensus_files=p["n_cons"], corner_drift_max_px="",
                   corner_drift_mean_px="", stem_angle_deg="" if p["angle"] is None else round(p["angle"], 3),
                   stem_angle_change_deg="", status="no ROI files" if p["n"] == 0 else "ok")
        if p["n"] and ref_day is not None:
            ref = per_day[ref_day]
            dists = [np.linalg.norm(p["med"][k] - ref["med"][k], axis=1) for k in p["med"] if k in ref["med"]]
            if dists:
                allv = np.concatenate(dists)
                row["corner_drift_max_px"], row["corner_drift_mean_px"] = round(float(allv.max()), 2), round(float(allv.mean()), 2)
            if p["angle"] is not None and ref["angle"] is not None:
                row["stem_angle_change_deg"] = round(p["angle"] - ref["angle"], 3)
            why = []
            if row["corner_drift_max_px"] != "" and row["corner_drift_max_px"] > drift_warn_px:
                why.append(f"a maze corner sits {row['corner_drift_max_px']} px from where it was on DAY{ref_day} (level {drift_warn_px:g} px)")
            if row["stem_angle_change_deg"] != "" and abs(row["stem_angle_change_deg"]) > stem_angle_warn_deg:
                why.append(f"the stem direction changed by {row['stem_angle_change_deg']:+.2f} deg against DAY{ref_day} (level {stem_angle_warn_deg:g} deg)")
            if why:
                row["status"] = "warn"
                warnings.append(f"DAY{n}: maze position — " + "; ".join(why) + ": was the camera or the maze moved?")
        rows.append(row)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
    return dict(days=rows, warnings=warnings, reference_day=ref_day)
