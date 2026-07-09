#!/usr/bin/env python
"""STRIDE safety gates — automatic sanity checks run BETWEEN pipeline steps, plus a gate ledger that
enforces step ORDER (so steps cannot be self-picked out of sequence).

Why: steps were being reordered/skipped by judgement and a silent px/cm calibration bug reached the
figures. A gate returns Findings; severity FAIL means STOP, WARN means inspect. Nonzero exit (2 on any
FAIL; 1 on any WARN with --strict) gates a shell step.

Hardened after an adversarial review — closes the ways a bad cohort could pass a green gate:
  * a pinned PX_PER_CM_OVERRIDE makes px/cm uniform, so uniformity alone is NOT enough: pass
    --expect-pxcm (from calibration_manifest) and the gate FAILs if the median disagrees >5%.
  * per-ROW range check catches the silent _px_per_cm=1.0 fallback and per-clip garbage (not just the
    batch median/percentiles).
  * direct bimodality check catches a MINORITY 2x population (the original bug was ~40% of 24mo).
  * fps uniformity (30-fps Mimo clips corrupt speed/timing).
  * single-camera / single-pose-model assertion (body length is NOT comparable across cameras).
  * missing calibration column or empty CSV is a FAIL, not a silent pass.

Gate ledger:
    python -m stride.stages.safety_checks --stamp rename   --vid $VID
    python -m stride.stages.safety_checks --require rename,roi --vid $VID   # FAIL if a prior gate unstamped

Checks:
    python -m stride.stages.safety_checks --gait g.csv --expect-pxcm 26.15 --strict
    python -m stride.stages.safety_checks --roi-coverage $VID --expect-count 128
"""
from __future__ import annotations
import argparse, csv, json, sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np

BODY_LEN_CM_MIN, BODY_LEN_CM_MAX = 4.0, 11.0     # adult mouse snout->tailbase, projected, walking
PXCM_MIN, PXCM_MAX = 5.0, 40.0                    # any lab camera (Basler ~12, GoPro/DJI ~26)
PXCM_TOL = 0.05                                   # allowed deviation of median px/cm from --expect-pxcm
GATE_LEDGER = ".stride_gates.json"                # written in the video dir; records which gates passed


@dataclass
class Finding:
    severity: str   # FAIL | WARN | OK
    check: str
    message: str


def _col(rows, name):
    out = []
    for r in rows:
        try:
            v = float(r[name])
            if v == v:
                out.append(v)
        except (KeyError, ValueError, TypeError):
            pass
    return np.array(out)


def check_calibration(rows, expect_pxcm=None) -> list[Finding]:
    f = []
    if not rows:
        return [Finding("FAIL", "calibration", "empty CSV — no data to verify (truncated file?)")]
    if "px_per_cm" not in rows[0]:
        return [Finding("FAIL", "calibration", "no px_per_cm column — calibration UNVERIFIABLE (stop)")]
    cmpx = _col(rows, "px_per_cm")                       # column stores cm/px in these CSVs
    if cmpx.size == 0:
        return [Finding("FAIL", "calibration", "px_per_cm column present but all-empty/non-numeric")]
    ppcm = 1.0 / cmpx
    med = float(np.median(ppcm)); lo, hi = np.percentile(ppcm, [2, 98])

    # (a) per-ROW range: catches the _px_per_cm=1.0 fallback sentinel + per-clip garbage
    bad = int(np.sum((ppcm < PXCM_MIN) | (ppcm > PXCM_MAX)))
    if bad:
        f.append(Finding("FAIL", "calibration",
            f"{bad}/{ppcm.size} strides have px/cm outside [{PXCM_MIN},{PXCM_MAX}] "
            f"(min {ppcm.min():.2f}, max {ppcm.max():.2f}) — 1.0-fallback or garbage calibration"))
    # (b) median plausibility
    if not (PXCM_MIN <= med <= PXCM_MAX):
        f.append(Finding("FAIL", "calibration", f"median px/cm={med:.1f} outside plausible range"))
    # (c) uniformity (percentile spread)
    spread = hi / max(lo, 1e-6)
    if spread > 1.4:
        f.append(Finding("FAIL", "calibration",
            f"px/cm NOT uniform (p2={lo:.1f}, p98={hi:.1f}, {spread:.2f}x) — segment2 orientation bug"))
    # (d) minority bimodality near 2x or 0.5x the median (the original bug was ~40% affected)
    for factor, tag in ((2.0, "2x"), (0.5, "0.5x")):
        frac = float(np.mean(np.abs(ppcm - factor * med) < 0.15 * factor * med))
        if frac > 0.02:
            f.append(Finding("FAIL", "calibration",
                f"{frac*100:.0f}% of strides sit at ~{tag} the median px/cm — bimodal calibration bug"))
    # (e) absolute anchor: the ONE check the override cannot fool
    if expect_pxcm:
        dev = abs(med - expect_pxcm) / expect_pxcm
        if dev > PXCM_TOL:
            f.append(Finding("FAIL", "calibration",
                f"median px/cm={med:.2f} deviates {dev*100:.1f}% from board-verified {expect_pxcm:.2f} "
                f"(> {PXCM_TOL*100:.0f}%) — override/segment2 disagrees with the checkerboard"))
    elif spread < 1.02:
        f.append(Finding("WARN", "calibration",
            "px/cm is a single pinned constant but no --expect-pxcm given: a uniform-but-WRONG override "
            "cannot be caught. Pass the board-verified value from calibration_manifest to verify it."))
    if not f:
        f.append(Finding("OK", "calibration", f"px/cm uniform ~{med:.1f}"
                         + (f", matches board {expect_pxcm:.2f}" if expect_pxcm else "")))
    return f


def check_body_length(rows) -> list[Finding]:
    f = []
    bl = _col(rows, "body_length_cm")
    if bl.size == 0:
        return [Finding("FAIL", "body_length", "no/empty body_length_cm column")]
    med = float(np.median(bl))
    if not (BODY_LEN_CM_MIN <= med <= BODY_LEN_CM_MAX):
        f.append(Finding("FAIL", "body_length",
            f"median body_length_cm={med:.2f} implausible ({BODY_LEN_CM_MIN}-{BODY_LEN_CM_MAX}) — wrong ruler"))
    # never pool cm lengths across cameras / pose-model versions
    for colname, label in (("camera", "camera"), ("pose_model", "pose-model")):
        if colname in rows[0]:
            vals = {str(r.get(colname, "")).strip() for r in rows if str(r.get(colname, "")).strip()}
            if len(vals) > 1:
                f.append(Finding("FAIL", "body_length",
                    f"rows mix {len(vals)} {label}s ({sorted(vals)}) — body length is NOT comparable "
                    f"across {label}s; do not pool"))
    if not f:
        f.append(Finding("OK", "body_length", f"median body_length_cm={med:.2f} (plausible)"))
    return f


def check_fps(rows) -> list[Finding]:
    if not rows or "fps" not in rows[0]:
        return [Finding("WARN", "fps", "no fps column")]
    fps = _col(rows, "fps")
    uniq = sorted({round(x, 1) for x in fps})
    if len(uniq) > 1:
        return [Finding("FAIL", "fps",
            f"non-uniform fps across clips {uniq} — speed/timing not comparable; exclude off-cadence clips")]
    return [Finding("OK", "fps", f"uniform fps {uniq[0] if uniq else '?'}")]


def check_roi_coverage(video_dir, expect_count=None) -> list[Finding]:
    d = Path(video_dir)
    mp4 = sorted([p for p in d.rglob("*.MP4")] + [p for p in d.rglob("*.mp4")])
    if not mp4:
        return [Finding("FAIL", "roi_coverage", f"NO videos found under {video_dir} (wrong path / glob?)")]
    if expect_count is not None and len(mp4) != expect_count:
        return [Finding("FAIL", "roi_coverage",
            f"found {len(mp4)} videos but expected {expect_count} (Step-0 count mismatch)")]
    miss = [p.name for p in mp4 if not p.with_suffix("").with_suffix(".rois.yml").exists()]
    if miss:
        return [Finding("FAIL", "roi_coverage",
            f"{len(miss)}/{len(mp4)} videos have NO .rois.yml (e.g. {miss[:3]}) — run propagate_roi.py")]
    return [Finding("OK", "roi_coverage", f"all {len(mp4)} videos have a .rois.yml")]


# ---- gate ledger: enforce step ORDER ----
def _ledger_path(vid): return Path(vid) / GATE_LEDGER
def stamp_gate(vid, name):
    p = _ledger_path(vid); d = json.loads(p.read_text()) if p.exists() else {}
    d[name] = True; p.write_text(json.dumps(d, indent=2))
    print(f"  ✓ stamped gate '{name}' in {p}")
def require_gates(vid, names) -> list[Finding]:
    p = _ledger_path(vid); d = json.loads(p.read_text()) if p.exists() else {}
    missing = [n for n in names if not d.get(n)]
    if missing:
        return [Finding("FAIL", "gate_order", f"prior gate(s) not passed: {missing} — run steps in order")]
    return [Finding("OK", "gate_order", f"prior gates {names} all passed")]


def report(findings) -> int:
    order = {"FAIL": 0, "WARN": 1, "OK": 2}
    mark = {"FAIL": "✗ FAIL", "WARN": "! WARN", "OK": "✓ OK  "}
    for x in sorted(findings, key=lambda f: order[f.severity]):
        print(f"  {mark[x.severity]}  [{x.check}] {x.message}")
    if any(f.severity == "FAIL" for f in findings): return 2
    if any(f.severity == "WARN" for f in findings): return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description="STRIDE safety gates")
    ap.add_argument("--gait"); ap.add_argument("--roi-coverage")
    ap.add_argument("--expect-pxcm", type=float, default=None)
    ap.add_argument("--expect-count", type=int, default=None)
    ap.add_argument("--vid", help="video dir for the gate ledger (--stamp/--require)")
    ap.add_argument("--stamp", help="stamp this gate name as passed (needs --vid)")
    ap.add_argument("--require", help="comma-list of gate names that must already be stamped (needs --vid)")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    findings = []
    if a.require and a.vid: findings += require_gates(a.vid, [s.strip() for s in a.require.split(",")])
    if a.gait:
        rows = list(csv.DictReader(open(a.gait)))
        findings += check_calibration(rows, a.expect_pxcm) + check_body_length(rows) + check_fps(rows)
    if a.roi_coverage: findings += check_roi_coverage(a.roi_coverage, a.expect_count)
    code = report(findings) if findings else 0
    if a.stamp and a.vid and code != 2:
        stamp_gate(a.vid, a.stamp)
    if not findings and not a.stamp:
        print("nothing to check"); sys.exit(0)
    sys.exit(code if (a.strict or code == 2) else 0)


if __name__ == "__main__":
    main()
