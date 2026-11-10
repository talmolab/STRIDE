"""Intake settings from a YAML file, so nothing about a lab's folders or rig is written into the code.

    cohort_dir: /data/cohorts/my_cohort          # raw clips + the recorder export, or its <DATE>-DAY{N} folders
    records_dir: /data/intake_records/my_cohort  # optional; default <cohort_dir>/_intake
    workers: 8
    verifier: "name of the person who reads the boards"
    allow_placeholders: false
    camera:
      raw_pattern: '^DJI_(?P<date>\\d{8})(?P<time>\\d{6})_(?P<seq>\\d{4})_D\\.MP4$'
      extensions: mp4
      crop_rotation: ccw90
      rail_crop: 820,1320,850,2500               # or "none"
    board:
      squares: 14x9
      square_mm: 20
      marker_mm: 15
      dictionary: DICT_5X5_250

Every key is optional; a command-line option always wins over the file.
"""
from __future__ import annotations

from pathlib import Path

from ._log import IntakeError

_TOP = {"cohort_dir": "cohort_dir", "records_dir": "records_dir", "workers": "workers", "verifier": "verifier",
        "allow_placeholders": "allow_placeholders", "allow_mismatch": "allow_mismatch", "fixed_run_order": "fixed_run_order",
        "days": "days"}
_CAMERA = {"raw_pattern": "raw_pattern", "extensions": "extensions", "crop_rotation": "crop_rotation", "rail_crop": "rail_crop"}
_BOARD = {"squares": "board_squares", "square_mm": "square_mm", "marker_mm": "marker_mm", "dictionary": "dictionary"}


def load_defaults(path) -> dict:
    """YAML -> {option name: value}, the defaults for every intake command."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise IntakeError(f"config file not found: {p}")
    try:
        cfg = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as e:
        raise IntakeError(f"{p}: not valid YAML ({e})")
    if not isinstance(cfg, dict):
        raise IntakeError(f"{p}: expected a mapping at the top level")
    out, unknown = {}, []
    for k, v in cfg.items():
        if k in _TOP:
            out[_TOP[k]] = v
        elif k in ("camera", "board") and isinstance(v, dict):
            table = _CAMERA if k == "camera" else _BOARD
            for kk, vv in v.items():
                if kk in table:
                    out[table[kk]] = vv if not isinstance(vv, (list, tuple)) else ",".join(str(x) for x in vv)
                else:
                    unknown.append(f"{k}.{kk}")
        else:
            unknown.append(str(k))
    if unknown:
        raise IntakeError(f"{p}: unknown key(s) {', '.join(unknown)}")
    for k in ("cohort_dir", "records_dir"):
        if k in out and out[k] is not None:
            out[k] = str(out[k])
    return out
