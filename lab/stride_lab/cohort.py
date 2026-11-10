"""The cohort config: one YAML per cohort, kept OUTSIDE the repository (it names the lab's folders and models).

    stride-lab run --config my_cohort.yaml

`lab/cohort.example.yaml` documents every key. Nothing about a lab's folders, rig or models is written in the code.
"""
from __future__ import annotations

import copy
from pathlib import Path

from stride.stages.scale_calibration import BoardSpec, make_board

from .intake._log import IntakeError
from .intake.camera import CameraProfile, profile_from_options

DEFAULTS: dict = {
    "cohort": None,                       # label used in titles and the summary; default: the raw folder's name
    "raw_dir": None,                      # raw clips + the recorder export, or the <DATE>-DAY{N} folders
    "metadata": None,                     # recorder export (*_trials_full.csv); default: the most complete one under raw_dir
    "records_dir": None,                  # plans, crops, sign-offs, calibration, logs; default <raw_dir>/_intake
    "output_dir": None,                   # tables, inference outputs, summaries
    "verifier": "",
    "allow_placeholders": False,
    "allow_mismatch": False,
    "fixed_run_order": False,
    "genotype": None,                     # one genotype for every mouse; default: the export's genotype column
    "camera": {"raw_pattern": None, "extensions": None, "crop_rotation": None, "rail_crop": None},
    "board": {"squares": "14x9", "square_mm": 20.0, "marker_mm": 15.0, "dictionary": "DICT_5X5_250"},
    "calibration": {
        "clips_per_day": 3,               # clips measured on every recording day (spread over the day)
        "warn_pct": 0.5,                  # a day this far from the cohort median: warning
        "stop_pct": 1.0,                  # ... this far: the day is held until a person records a decision
        "reference_px_per_cm": None,      # the previous cohort on the same rig (warning only)
        "px_per_cm": None,                # explicit value for the analysis; the measurement is still made and compared
        "on_unmeasured_day": "warn",      # warn | stop: a day on which no board reading was possible
    },
    "roi_position": {"drift_warn_px": 25.0, "stem_angle_warn_deg": 1.0},
    "models": {"pose": [], "roi": []},
    "inference": {
        "devices": ["cuda:0"],            # one entry per GPU
        "workers_per_device": 2,
        "batch_size": 16,
        "sidecars": "videos",             # videos: pose and ROI files end up next to the clips (lab convention) | output
        "skip": False,                    # true: analyse only clips that already have pose and ROI files
    },
    "analysis": {
        "workers": 8,
        "decision_window_s": 150.0,
        "distortion_check": False,
        "stride_detection": {},
        "stride_filtering": {},
    },
}

_ALIASES = {"cohort_dir": "raw_dir", "output": "output_dir"}
_LEGACY_TOP = {"workers": ("analysis", "workers"), "days": None}


def _merge(base: dict, new: dict, where: str, unknown: list) -> dict:
    for k, v in new.items():
        if k not in base:
            unknown.append(f"{where}{k}")
            continue
        if isinstance(base[k], dict) and base[k] and isinstance(v, dict):
            _merge(base[k], v, f"{where}{k}.", unknown)
        elif isinstance(base[k], dict) and base[k] and v is not None:
            unknown.append(f"{where}{k} (expected a mapping)")
        else:
            base[k] = v
    return base


def load_cohort_config(path, overrides: dict | None = None) -> dict:
    """YAML -> validated settings. Unknown keys are refused with their names."""
    import yaml

    p = Path(path)
    if not p.is_file():
        raise IntakeError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise IntakeError(f"{p}: not valid YAML ({e})")
    if not isinstance(raw, dict):
        raise IntakeError(f"{p}: expected a mapping at the top level")
    raw = dict(raw)
    for old, new in _ALIASES.items():
        if old in raw and new not in raw:
            raw[new] = raw.pop(old)
    days = raw.pop("days", None)
    if "workers" in raw:
        raw.setdefault("analysis", {})
        if isinstance(raw["analysis"], dict):
            raw["analysis"].setdefault("workers", raw.pop("workers"))
    cfg, unknown = copy.deepcopy(DEFAULTS), []
    _merge(cfg, raw, "", unknown)
    if unknown:
        raise IntakeError(f"{p}: unknown key(s): {', '.join(unknown)}. Valid top-level keys: {', '.join(DEFAULTS)}")
    for k, v in (overrides or {}).items():
        if v is not None:
            cfg[k] = v
    cfg["days"] = days
    cfg["config_file"] = str(p)
    return finish(cfg)


def finish(cfg: dict) -> dict:
    """Resolve defaults that depend on other keys and check the values."""
    if not cfg.get("raw_dir"):
        raise IntakeError("the cohort config needs `raw_dir` (the folder with the raw clips and the recorder export)")
    raw = Path(str(cfg["raw_dir"])).expanduser()
    cfg["raw_dir"] = raw
    cfg["records_dir"] = Path(str(cfg["records_dir"])).expanduser() if cfg.get("records_dir") else raw / "_intake"
    cfg["output_dir"] = Path(str(cfg["output_dir"])).expanduser() if cfg.get("output_dir") else None
    cfg["cohort"] = str(cfg.get("cohort") or raw.name)
    cal = cfg["calibration"]
    try:
        cal["clips_per_day"] = int(cal["clips_per_day"])
        cal["warn_pct"], cal["stop_pct"] = float(cal["warn_pct"]), float(cal["stop_pct"])
        for k in ("reference_px_per_cm", "px_per_cm"):
            cal[k] = float(cal[k]) if cal.get(k) not in (None, "") else None
    except (TypeError, ValueError) as e:
        raise IntakeError(f"calibration: {e}")
    if cal["clips_per_day"] < 1:
        raise IntakeError("calibration.clips_per_day must be at least 1")
    if not 0 < cal["warn_pct"] <= cal["stop_pct"]:
        raise IntakeError(f"calibration: need 0 < warn_pct <= stop_pct (got {cal['warn_pct']}, {cal['stop_pct']})")
    if cal["on_unmeasured_day"] not in ("warn", "stop"):
        raise IntakeError(f"calibration.on_unmeasured_day must be warn or stop, not {cal['on_unmeasured_day']!r}")
    inf = cfg["inference"]
    if isinstance(inf.get("devices"), str):
        inf["devices"] = [d.strip() for d in inf["devices"].split(",") if d.strip()]
    if not inf.get("devices"):
        raise IntakeError("inference.devices must name at least one device (e.g. [cuda:0] or [cpu])")
    if inf["sidecars"] not in ("videos", "output"):
        raise IntakeError(f"inference.sidecars must be videos or output, not {inf['sidecars']!r}")
    for k in ("pose", "roi"):
        v = cfg["models"].get(k) or []
        cfg["models"][k] = [str(v)] if isinstance(v, (str, Path)) else [str(x) for x in v]
    return cfg


def camera_of(cfg: dict) -> CameraProfile:
    c = cfg.get("camera") or {}
    rail = c.get("rail_crop")
    if isinstance(rail, (list, tuple)):
        rail = ",".join(str(x) for x in rail)
    ext = c.get("extensions")
    if isinstance(ext, (list, tuple)):
        ext = ",".join(str(x) for x in ext)
    try:
        return profile_from_options(c.get("raw_pattern"), c.get("crop_rotation"), ext, rail)
    except Exception as e:
        raise IntakeError(f"camera: {e}")


def board_of(cfg: dict) -> BoardSpec:
    b = cfg.get("board") or {}
    try:
        return make_board({"squares": b.get("squares", "14x9"), "square_mm": b.get("square_mm", 20.0),
                           "marker_mm": b.get("marker_mm", 15.0), "dictionary": b.get("dictionary", "DICT_5X5_250")})
    except Exception as e:
        raise IntakeError(f"board: {e}")


def need_output(cfg: dict) -> Path:
    if cfg.get("output_dir") is None:
        raise IntakeError("the cohort config needs `output_dir` (where the tables and summaries go)")
    return cfg["output_dir"]
