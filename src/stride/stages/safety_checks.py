#!/usr/bin/env python
"""STRIDE safety gates — automatic sanity checks run BETWEEN pipeline steps.

Motivation: the pipeline is a fixed sequence of steps, but steps were being skipped/reordered by
judgement and a silent calibration bug (px/cm doubling on angled mazes) slipped all the way to the
figures. These checks make failure LOUD and EARLY. A gate returns a list of Findings; severity FAIL
means STOP the pipeline and fix before the next step, WARN means inspect but may continue.

Run standalone as a gate, e.g. after gait extraction:
    python -m stride.stages.safety_checks --gait <gait_per_stride.csv> [--expect-pxcm 26.2] [--strict]
    python -m stride.stages.safety_checks --roi-coverage <video_dir>

Exit code is nonzero if any FAIL (with --strict, also if any WARN) — so it can gate a shell script:
    python -m stride.stages.safety_checks --gait g.csv --strict || { echo STOP; exit 1; }
"""
from __future__ import annotations
import argparse, csv, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Plausible ranges (adult mouse; overhead T-maze). Deliberately wide — these catch GROSS errors
# (a 2x calibration bug, a swapped unit), not subtle ones.
BODY_LEN_CM_MIN, BODY_LEN_CM_MAX = 4.0, 11.0      # snout->tailbase, projected, during walking
PXCM_MIN, PXCM_MAX = 5.0, 40.0                    # any of the lab's cameras (Basler ~12, GoPro/DJI ~26)


@dataclass
class Finding:
    severity: str   # "FAIL" | "WARN" | "OK"
    check: str
    message: str


def _col(rows, name):
    out = []
    for r in rows:
        try:
            v = float(r[name])
            if v == v:  # not nan
                out.append(v)
        except (KeyError, ValueError, TypeError):
            pass
    return np.array(out)


def check_calibration(rows) -> list[Finding]:
    """px/cm must be UNIMODAL and in a plausible range. Bimodal px/cm = the segment2 orientation bug
    (some clips 2x off); it is the single failure that corrupted a whole cohort silently."""
    f = []
    ppcm = 1.0 / _col(rows, "px_per_cm") if "px_per_cm" in (rows[0] if rows else {}) else np.array([])
    # px_per_cm column is stored as cm/px in these CSVs; invert to px/cm
    if ppcm.size == 0:
        return [Finding("WARN", "calibration", "no px_per_cm column found — cannot verify calibration")]
    med = float(np.median(ppcm)); lo, hi = np.percentile(ppcm, [2, 98])
    if not (PXCM_MIN <= med <= PXCM_MAX):
        f.append(Finding("FAIL", "calibration",
            f"median px/cm={med:.1f} outside plausible [{PXCM_MIN},{PXCM_MAX}] — wrong calibration/unit"))
    # bimodality: if p98/p2 spread is large, px/cm is NOT a single value across the batch
    spread = hi / max(lo, 1e-6)
    if spread > 1.4:
        f.append(Finding("FAIL", "calibration",
            f"px/cm is NOT uniform across clips (p2={lo:.1f}, p98={hi:.1f}, spread {spread:.2f}x). "
            f"This is the segment2 orientation bug — some clips have doubled px/cm. Pin PX_PER_CM_OVERRIDE."))
    if not f:
        f.append(Finding("OK", "calibration", f"px/cm uniform ~{med:.1f} (spread {spread:.2f}x)"))
    return f


def check_body_length(rows, expect_pxcm: float | None = None) -> list[Finding]:
    """Median body length must be anatomically plausible. If it isn't, the ruler is wrong."""
    bl = _col(rows, "body_length_cm")
    if bl.size == 0:
        return [Finding("WARN", "body_length", "no body_length_cm column")]
    med = float(np.median(bl))
    if not (BODY_LEN_CM_MIN <= med <= BODY_LEN_CM_MAX):
        return [Finding("FAIL", "body_length",
            f"median body_length_cm={med:.2f} is anatomically implausible "
            f"(expect {BODY_LEN_CM_MIN}-{BODY_LEN_CM_MAX}). Calibration is almost certainly wrong.")]
    return [Finding("OK", "body_length", f"median body_length_cm={med:.2f} (plausible)")]


def check_roi_coverage(video_dir: str) -> list[Finding]:
    """Every video must have a co-located .rois.yml before decisions/gait."""
    d = Path(video_dir)
    mp4 = sorted([p for p in d.rglob("*.MP4")] + [p for p in d.rglob("*.mp4")])
    miss = [p.name for p in mp4 if not p.with_suffix("").with_suffix(".rois.yml").exists()]
    if not mp4:
        return [Finding("WARN", "roi_coverage", f"no videos found under {video_dir}")]
    if miss:
        return [Finding("FAIL", "roi_coverage",
            f"{len(miss)}/{len(mp4)} videos have NO .rois.yml (e.g. {miss[:3]}). "
            f"Run the ROI consensus fallback (roi_inference/propagate_roi.py) — do NOT proceed.")]
    return [Finding("OK", "roi_coverage", f"all {len(mp4)} videos have a .rois.yml")]


def report(findings: list[Finding]) -> int:
    """Print findings; return 2 if any FAIL, 1 if any WARN, else 0."""
    order = {"FAIL": 0, "WARN": 1, "OK": 2}
    for x in sorted(findings, key=lambda f: order[f.severity]):
        mark = {"FAIL": "✗ FAIL", "WARN": "! WARN", "OK": "✓ OK  "}[x.severity]
        print(f"  {mark}  [{x.check}] {x.message}")
    if any(f.severity == "FAIL" for f in findings):
        return 2
    if any(f.severity == "WARN" for f in findings):
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description="STRIDE safety gates")
    ap.add_argument("--gait", help="gait_per_stride[_filtered].csv to sanity-check calibration + body length")
    ap.add_argument("--roi-coverage", help="video dir: assert every video has a .rois.yml")
    ap.add_argument("--expect-pxcm", type=float, default=None)
    ap.add_argument("--strict", action="store_true", help="treat WARN as failure (nonzero exit)")
    a = ap.parse_args()
    findings = []
    if a.gait:
        rows = list(csv.DictReader(open(a.gait)))
        findings += check_calibration(rows) + check_body_length(rows, a.expect_pxcm)
    if a.roi_coverage:
        findings += check_roi_coverage(a.roi_coverage)
    if not findings:
        print("nothing to check — pass --gait and/or --roi-coverage"); sys.exit(0)
    code = report(findings)
    sys.exit(code if (a.strict or code == 2) else 0)


if __name__ == "__main__":
    main()
