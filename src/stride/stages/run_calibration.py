"""Config-driven scale calibration (pipeline stage 0): px/cm of a batch from a ChArUco board in view.

Opt-in per batch. A batch is calibrated only if it has a `calibrate:` entry; other batches are skipped.

    batches:
      my_batch:
        video_dir: /data/my_cohort
        calibrate: true                  # the default board (14x9, 20 mm squares, 15 mm markers, DICT_5X5_250)
        # or
        calibrate:
          squares_x: 7                   # the same keyword arguments as calibrate_px_per_cm(...)
          squares_y: 5
          square_mm: 30
          marker_mm: 22
          dictionary: DICT_4X4_50
          max_clips: 12                  # optional; also min_clips, frames_per_clip, min_readings, max_spread_pct, workers

Writes <output>/<batch>/calibration/px_per_cm.txt (+ summary, per-clip table, QC images). The gait stage uses
that value when the batch has no explicit `px_per_cm_override`.

    python -m stride.stages.run_calibration --config config.yaml --output output [--batch my_batch]

Exit code 1 if a requested calibration cannot be accepted (the reason is printed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from stride.stages.scale_calibration import CalibrationError, calibrate_px_per_cm

CALIBRATION_DIR = "calibration"
ALLOWED = {"min_clips", "max_clips", "frames_per_clip", "scan_seconds", "step_seconds", "min_readings", "max_spread_pct", "workers"}


def calibration_file(output_dir, batch_key: str) -> Path:
    """Where a batch's accepted px/cm is written (and where the gait stage looks for it)."""
    return Path(output_dir) / batch_key / CALIBRATION_DIR / "px_per_cm.txt"


def read_batch_px_per_cm(output_dir, batch_key: str):
    """The batch's calibrated px/cm, or None if it has not been calibrated."""
    f = calibration_file(output_dir, batch_key)
    try:
        return float(f.read_text(encoding="utf-8").strip()) if f.is_file() else None
    except ValueError:
        return None


def run_batch(batch_key: str, batch_cfg: dict, output_dir: Path) -> str:
    """-> "skipped" | "ok". Raises CalibrationError when a requested calibration fails."""
    spec = batch_cfg.get("calibrate")
    if not spec:
        print(f"  [{batch_key}] no `calibrate:` entry in the batch — scale calibration skipped")
        return "skipped"
    spec = dict(spec) if isinstance(spec, dict) else {}
    video_dir = spec.pop("video_dir", None) or batch_cfg.get("video_dir")
    if not video_dir:
        raise CalibrationError(f"[{batch_key}] calibrate needs the batch's `video_dir`")
    board = {k: spec.pop(k) for k in ("board_squares", "squares", "squares_x", "squares_y", "square_mm", "marker_mm", "dictionary") if k in spec}
    if "clips" in spec:
        spec["min_clips"] = spec.pop("clips")
    unknown = sorted(set(spec) - ALLOWED)
    if unknown:
        raise CalibrationError(f"[{batch_key}] unknown key(s) under calibrate: {', '.join(unknown)}")
    out = calibration_file(output_dir, batch_key).parent
    r = calibrate_px_per_cm(Path(video_dir), out, board or None, log=lambda m: print(f"  [{batch_key}] {m}", flush=True), **spec)
    if not r["ok"]:
        raise CalibrationError(f"[{batch_key}] {r['reason']}")
    pinned = batch_cfg.get("px_per_cm_override")
    if pinned:
        diff = abs(float(pinned) - r["px_per_cm"]) / r["px_per_cm"] * 100
        print(f"  [{batch_key}] {'WARNING: ' if diff > 1.0 else ''}the batch pins px_per_cm_override = {pinned}; "
              f"the board reads {r['px_per_cm']:.2f} ({diff:.2f} % apart). The pinned value is the one the gait stage uses.")
    return "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Config-driven ChArUco scale calibration (stage 0)")
    ap.add_argument("--config", default="config.yaml", help="Pipeline config")
    ap.add_argument("--output", default="output", help="Output directory")
    ap.add_argument("--batch", default=None, help="Process only this batch key")
    a = ap.parse_args(argv)
    try:
        config = yaml.safe_load(Path(a.config).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        print(f"ERROR: cannot read the config {a.config}: {e}", file=sys.stderr)
        return 1
    batches = config.get("batches") or {}
    if a.batch:
        if a.batch not in batches:
            print(f"ERROR: batch '{a.batch}' is not in {a.config} (batches: {', '.join(batches) or 'none'})", file=sys.stderr)
            return 1
        batches = {a.batch: batches[a.batch]}
    print("=" * 60 + "\nSCALE CALIBRATION (stage 0)\n" + "=" * 60)
    failed = []
    for key, cfg in batches.items():
        try:
            run_batch(key, cfg or {}, Path(a.output))
        except CalibrationError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            failed.append(key)
    if failed:
        print(f"Scale calibration FAILED for: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
