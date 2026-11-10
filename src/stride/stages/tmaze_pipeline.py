"""
STRIDE T-maze: the fully automated pipeline (`stride tmaze`).

Videos + a trained pose model + a trained ROI model in; per-trial and per-stride CSV tables out::

    discover -> distortion check (report only) -> scale (px/cm) -> pose inference
             -> ROI inference (+ consensus ROI) -> decisions -> decision rules -> gait

Nothing here computes a metric. Every step calls the stage that the rest of the package uses
(``run_decision_analysis``, ``apply_decision_rules``, ``run_gait_extraction``,
``compute_keypoint_confidence``, ``filter_strides``, the two inference stages), so the numbers
are the same whichever way the stages are started.

Settings come from an optional YAML config (`stride init-config tmaze` writes a commented
template); any command-line option overrides the config value. Every step can be switched off.

Output folder::

    SUMMARY.md, tmaze_run.log, tmaze_run_manifest.yaml, tmaze_state.json
    scale/       px_per_cm.json, distortion_check.csv, (ChArUco QC files)
    pose/        <video stem>.slp               only for videos that had no pose file next to them
    roi/         <video stem>.preds.v2.best1.slp, <video stem>.rois.yml, roi_sources.csv,
                 roi_consensus_qc/
    decisions/   decisions.csv, decisions_clean.csv, events.csv, metrics.csv
    gait/        gait_per_stride.csv, gait_per_step.csv, gait_per_video.csv,
                 gait_per_stride_filtered.csv, pose_keypoint_confidence_by_video.csv,
                 metrics_meta.csv, run_params.json
    stats/       only when the optional statistics steps are switched on
"""

from __future__ import annotations

import copy
import glob as _glob
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

STEPS = ["distortion_check", "calibrate", "pose_inference", "roi_inference", "roi_consensus",
         "decisions", "decision_rules", "gait", "mixed_models", "classifier"]
STEPS_DEFAULT_OFF = {"mixed_models", "classifier"}
VIDEO_EXTENSIONS = (".mp4",)

DEFAULTS: dict = {
    "videos": None,                 # folder, file, glob pattern, or a list of these
    "output": None,
    "pose_model": [],               # one or two paths (centroid, centered instance)
    "roi_model": [],
    "metadata": None,
    "pose_dir": None,               # folder holding existing pose files (when not next to the videos)
    "roi_dir": None,                # folder holding existing ROI files (when not next to the videos)
    "exclude_dirs": ["waste"],      # folder names skipped during the video search
    "pose_suffix": ".slp",
    "device": "auto",               # auto | cuda | cuda:0 | cpu
    "workers": 4,
    "batch_size": 16,
    "fps_default": 120.0,           # used only when the frame rate cannot be read from a video
    "scale": {
        "px_per_cm": None,          # a value you measured: wins over everything else
        "maze_width_cm": 10.0,      # corridor width, for the ROI-based px/cm
        "charuco": {
            "squares_x": None, "squares_y": None, "square_mm": None, "marker_mm": None,
            "dictionary": None, "min_clips": 6, "max_clips": None,
            "clips": None,          # measure exactly these videos instead of choosing them
        },
    },
    "roi": {"session": "folder", "drift_max_px": 25.0, "full_pass": True, "frames": 10},
    "distortion": {"max_clips": 6, "frames": 10},
    "decision_rules": {"window_s": 150.0},
    "stride_detection": {},         # optional overrides; defaults are the published values
    "stride_filtering": {},
    "stats_config": None,           # a STRIDE config with metrics / models / statistics sections
    "steps": {s: (s not in STEPS_DEFAULT_OFF) for s in STEPS},
    "stop_after": None,
    "force": False,
    "strict": False,
}

CONFIG_TEMPLATE = """\
# STRIDE T-maze configuration  (`stride tmaze --config this_file.yaml`)
# Any command-line option overrides the value written here. Paths may contain spaces; relative
# paths are resolved from the folder you start the command in.

videos: /path/to/videos            # a folder (searched recursively), one file, a pattern such as
                                   # "D:/recordings/*/*.mp4", or a list of these. Any file names.
output: /path/to/output            # everything STRIDE writes goes here (your videos are not touched)

pose_model:                        # your trained SLEAP pose model: one or two folders
  - /path/to/pose/centroid
  - /path/to/pose/centered_instance
roi_model:                         # your trained SLEAP maze-ROI model: one or two folders
  - /path/to/roi/centroid
  - /path/to/roi/centered_instance

pose_dir: null                     # folder with existing pose files, if they are not next to the videos
roi_dir: null                      # folder with existing ROI files (<video name>.rois.yml / .yml)

metadata: null                     # optional CSV (or a list of CSVs). Trial rows: a `video` column
                                   # (file name) or `day` + `mouse`, plus `reward` (L or R);
                                   # optional `trial`, `sex`, `genotype`. Mouse rows (one per
                                   # mouse): `mouse` plus `sex` / `genotype`. Without metadata the
                                   # choice, timing and probe columns are written and the
                                   # correctness columns stay empty.

scale:
  px_per_cm: null                  # a value you measured yourself; wins over everything below
  maze_width_cm: 10.0              # corridor width; used for px/cm from the ROI when there is
                                   # neither a value above nor a ChArUco board in view
  charuco:                         # the calibration board visible in the videos; null = the default
    squares_x: null                # board (14 x 9 squares, 20 mm squares, 15 mm markers, DICT_5X5_250)
    squares_y: null
    square_mm: null                # measure the printed square: it is what turns pixels into cm
    marker_mm: null
    dictionary: null               # e.g. DICT_5X5_250
    min_clips: 6                   # videos measured: the first of every folder, at least this many
    max_clips: null                # ... and at most this many
    clips: null                    # or: a list of videos to measure instead of that choice

device: auto                       # auto | cuda | cuda:0 | cpu   (inference only)
workers: 4                         # CPU worker processes for the analysis
batch_size: 16                     # pose inference batch size
pose_suffix: .slp                  # pose file = <video name><this suffix>
exclude_dirs: [waste]              # folder names skipped when searching for videos

roi:
  session: folder                  # clips sharing a ROI geometry: `folder` (same folder) or `all`
  drift_max_px: 25                 # consensus ROI is refused when detected ROIs differ by more
  full_pass: true                  # re-run the whole video when 10 sampled frames show no full maze

decision_rules:
  window_s: 150                    # trial length: an arm entry after this many seconds is no decision

steps:                             # switch any step off with `false`
  distortion_check: true           # report only
  calibrate: true                  # px/cm from the ChArUco board
  pose_inference: true             # skipped per video when its pose file already exists
  roi_inference: true              # skipped per video when its ROI file already exists
  roi_consensus: true              # borrow the session's ROI for clips with a hidden maze corner
  decisions: true
  decision_rules: true
  gait: true
  mixed_models: false              # genotype statistics: off by default, need `stats_config`
  classifier: false

stats_config: null                 # only for mixed_models / classifier: a STRIDE config holding
                                   # the metrics / covariates / models / statistics sections
                                   # (config.example.yaml in the repository)
stop_after: null                   # stop after this step, e.g. roi_inference
"""


class TmazeError(Exception):
    """The pipeline cannot run or cannot continue; the message says why (printed, exit code 1)."""


# --------------------------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------------------------

def _deep_update(base: dict, new: dict) -> dict:
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_settings(config_path=None, overrides: Optional[dict] = None) -> dict:
    """Defaults <- YAML config <- overrides (command-line options that were given)."""
    settings = copy.deepcopy(DEFAULTS)
    if config_path:
        p = Path(config_path)
        if not p.exists():
            raise TmazeError(f"config file not found: {p} (write a template with `stride init-config tmaze`)")
        with open(p, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise TmazeError(f"{p} is not a YAML mapping (see `stride init-config tmaze`)")
        unknown = sorted(set(cfg) - set(DEFAULTS))
        if unknown:
            raise TmazeError(f"{p}: unknown setting(s) {unknown}. Known settings: {sorted(DEFAULTS)}")
        bad_steps = sorted(set(cfg.get("steps") or {}) - set(STEPS))
        if bad_steps:
            raise TmazeError(f"{p}: unknown step(s) {bad_steps} under `steps:`. Steps: {STEPS}")
        _deep_update(settings, cfg)
        settings["config_file"] = str(p)
    _deep_update(settings, overrides or {})
    for key in ("pose_model", "roi_model", "exclude_dirs"):
        v = settings.get(key)
        settings[key] = [] if v is None else ([v] if isinstance(v, (str, Path)) else list(v))
    return settings


def write_config_template(path, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise TmazeError(f"{path} already exists (pass --overwrite to replace it)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------

class _Tee:
    """Write everything printed during the run to the log file as well."""

    def __init__(self, stream, log_file):
        self.stream, self.log_file = stream, log_file

    def write(self, text):
        self.stream.write(text)
        try:
            self.log_file.write(text)
        except Exception:
            pass
        return len(text)

    def flush(self):
        self.stream.flush()
        try:
            self.log_file.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _file_sig(path) -> list:
    try:
        st = Path(path).stat()
        return [str(path), st.st_size, int(st.st_mtime)]
    except OSError:
        return [str(path), None, None]


def _fingerprint(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def discover_videos(spec, exclude_dirs=()) -> tuple[list, int]:
    """Videos named by ``spec`` (folder / file / glob pattern / list of these).

    Returns (sorted unique paths, number excluded because they sit under an excluded folder).
    """
    from stride.utils.parallel import find_videos

    entries = spec if isinstance(spec, (list, tuple)) else [spec]
    found: list = []
    for entry in entries:
        if entry is None or str(entry).strip() == "":
            continue
        p = Path(str(entry)).expanduser()
        if p.is_dir():
            found.extend(find_videos(p, extensions=VIDEO_EXTENSIONS))
        elif p.is_file():
            found.append(p)
        elif any(ch in str(entry) for ch in "*?["):
            for hit in sorted(_glob.glob(str(p), recursive=True)):
                hp = Path(hit)
                if hp.is_dir():
                    found.extend(find_videos(hp, extensions=VIDEO_EXTENSIONS))
                elif hp.suffix.lower() in VIDEO_EXTENSIONS:
                    found.append(hp)
        else:
            raise TmazeError(f"videos: {entry} does not exist")
    excl = {str(d).lower() for d in (exclude_dirs or [])}
    kept, n_excluded, seen = [], 0, set()
    for v in found:
        if any(part.lower() in excl for part in v.parent.parts):
            n_excluded += 1
            continue
        key = str(v.resolve()) if v.exists() else str(v)
        if key not in seen:
            seen.add(key)
            kept.append(v)
    return sorted(kept), n_excluded


# --------------------------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------------------------

_VIDEO_KEYS = ("video", "file", "filename", "file_name", "video_name", "video_file", "stem", "video_path")


def _norm_day(value: str) -> str:
    d = str(value).strip().upper()
    if d.isdigit():
        return f"DAY{int(d)}"
    if d.startswith("DAY") and d[3:].strip().isdigit():
        return f"DAY{int(d[3:])}"
    return d


def _norm_reward(value) -> str:
    r = str(value or "").strip().upper()[:1]
    return r if r in ("L", "R") else ""


def load_metadata(paths) -> dict:
    """Read the optional metadata CSV file(s).

    A file may describe trials (a `video` column with the file name, or `day` + `mouse`; plus
    `reward`, optional `trial`, `sex`, `genotype`) or mice (`mouse` plus `sex` / `genotype`, one
    row per mouse). Several files are merged. Column names are case-insensitive; `Gender` = `sex`.

    Returns a dict with the lookups used by `build_trials`.
    """
    import pandas as pd

    if isinstance(paths, (str, Path)):
        paths = [paths]
    meta = {"path": ", ".join(Path(str(p)).name for p in paths), "n_rows": 0, "columns": [],
            "video_col": None, "by_stem": {}, "by_key": {}, "by_key_trial": {}, "by_mouse": {},
            "used": set()}
    row_id = 0
    for path in paths:
        p = Path(str(path)).expanduser()
        if not p.exists():
            raise TmazeError(f"metadata file not found: {p}")
        df = pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df.columns = [str(c).strip().lower() for c in df.columns]
        df = df.rename(columns={"gender": "sex"})
        video_col = next((c for c in _VIDEO_KEYS if c in df.columns), None)
        if video_col is None and "mouse" not in df.columns:
            raise TmazeError(f"{p.name}: needs a `video` column (file name) or a `mouse` column "
                             f"(with `day` for per-trial rows); found {list(df.columns)}")
        has_dm = "day" in df.columns and "mouse" in df.columns
        for c in df.columns:
            df[c] = df[c].astype(str).str.strip()
        meta["columns"] = sorted(set(meta["columns"]) | set(df.columns))
        meta["video_col"] = meta["video_col"] or video_col
        for r in df.to_dict("records"):
            trial_row = False
            if video_col is not None and r.get(video_col):
                meta["by_stem"][Path(r[video_col].replace("\\", "/")).stem] = r
                trial_row = True
            if has_dm and r.get("mouse") and r.get("day"):
                day = _norm_day(r["day"])
                meta["by_key"][(day, r["mouse"])] = r                # later rows win, as in the lab path
                if r.get("trial"):
                    meta["by_key_trial"][(day, r["mouse"], str(r["trial"]))] = r
                trial_row = True
            if r.get("mouse"):
                m = meta["by_mouse"].setdefault(r["mouse"], {})
                for k in ("sex", "genotype"):
                    if r.get(k) and not m.get(k):
                        m[k] = r[k]
            if trial_row:
                r["_row"] = row_id
                row_id += 1
    meta["n_rows"] = row_id
    return meta


def build_trials(videos: list, meta: Optional[dict]) -> list:
    """One dict per video: video_path, day, mouse, trial, reward, sex, genotype.

    day / mouse / trial come from the file name when it is ``Day<N>_<mouse>_Trial<T>``, otherwise
    from the metadata row of that video, otherwise they stay empty.
    """
    from stride.stages.decision_analysis import parse_filename

    trials = []
    for v in videos:
        key = parse_filename(v) or {}
        t = {"video_path": str(v), "day": key.get("day", ""), "mouse": key.get("mouse", ""),
             "trial": key.get("trial"), "reward": "", "sex": "", "genotype": "", "meta_row": None}
        row = None
        if meta is not None:
            row = meta["by_stem"].get(v.stem)
            if row is None and meta["by_stem"]:
                low = {k.lower(): r for k, r in meta["by_stem"].items()}
                row = low.get(v.stem.lower())
            if row is not None and not key:
                t["day"] = _norm_day(row.get("day", "")) if row.get("day") else ""
                t["mouse"] = row.get("mouse", "") or ""
                tr = row.get("trial", "")
                t["trial"] = int(float(tr)) if str(tr).replace(".", "", 1).isdigit() else None
            if row is None and t["mouse"]:
                if t["trial"] is not None:
                    row = meta["by_key_trial"].get((t["day"], t["mouse"], str(t["trial"])))
                if row is None:
                    row = meta["by_key"].get((t["day"], t["mouse"]))
            if row is not None:
                meta["used"].add(row["_row"])
                t["reward"] = _norm_reward(row.get("reward", ""))
                t["meta_row"] = row["_row"]
            per_mouse = meta["by_mouse"].get(t["mouse"], {}) if t["mouse"] else {}
            t["sex"] = (row or {}).get("sex", "") or per_mouse.get("sex", "")
            t["genotype"] = (row or {}).get("genotype", "") or per_mouse.get("genotype", "")
        trials.append(t)
    return trials


# --------------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------------

class TmazeRun:
    def __init__(self, settings: dict):
        self.s = settings
        if not settings.get("videos"):
            raise TmazeError("no videos given: pass a folder / file / pattern (`stride tmaze VIDEOS ...`) "
                             "or set `videos:` in the config")
        if not settings.get("output"):
            raise TmazeError("no output folder given: pass --out or set `output:` in the config")
        self.out = Path(str(settings["output"])).expanduser()
        self.steps = dict(settings["steps"])
        stop = settings.get("stop_after")
        if stop and stop not in STEPS:
            raise TmazeError(f"--stop-after {stop}: unknown step. Steps: {', '.join(STEPS)}")
        self.suffix = settings.get("pose_suffix") or ".slp"
        self.pose_dir = self.out / "pose"
        self.roi_dir = self.out / "roi"
        self.dec_dir = self.out / "decisions"
        self.gait_dir = self.out / "gait"
        self.scale_dir = self.out / "scale"
        # where pose / ROI files are looked for besides next to the video: the output folder
        # first, then the folder the user named for files that already exist
        self.pose_dirs = [self.pose_dir] + ([Path(str(settings["pose_dir"])).expanduser()] if settings.get("pose_dir") else [])
        self.roi_dirs = [self.roi_dir] + ([Path(str(settings["roi_dir"])).expanduser()] if settings.get("roi_dir") else [])
        for label, d in (("pose_dir", settings.get("pose_dir")), ("roi_dir", settings.get("roi_dir"))):
            if d and not Path(str(d)).expanduser().is_dir():
                raise TmazeError(f"{label}: folder not found: {d}")
        self.warnings: list = []
        self.stage_lines: list = []
        self.results: dict = {}
        self.skipped: dict = {}         # video name -> reason (first reason wins)
        self.state_path = self.out / "tmaze_state.json"
        self.state = {}
        self.t0 = time.time()

    # ---- bookkeeping -----------------------------------------------------------------------
    def on(self, step: str) -> bool:
        return bool(self.steps.get(step, False))

    def line(self, step: str, text: str):
        msg = f"[{step}] {text}"
        self.stage_lines.append(msg)
        print(msg, flush=True)

    def warn(self, text: str):
        self.warnings.append(text)
        print(f"WARNING: {text}", flush=True)

    def skip(self, video, reason: str):
        self.skipped.setdefault(Path(video).name, reason)

    def _load_state(self):
        if self.s.get("force"):
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_state(self):
        try:
            self.state_path.write_text(json.dumps(self.state, indent=1), encoding="utf-8")
        except OSError:
            pass

    def _done(self, step: str, fp: str, outputs: list) -> bool:
        return (self.state.get(step) == fp) and all(Path(o).exists() for o in outputs)

    def _stop(self, step: str) -> bool:
        return self.s.get("stop_after") == step

    # ---- file lookups ----------------------------------------------------------------------
    def pose_of(self, video: Path) -> Path:
        from stride.utils.pose_io import resolve_pose_path
        return resolve_pose_path(video, self.suffix, self.pose_dirs)

    def roi_of(self, video: Path) -> Optional[Path]:
        from stride.stages.decision_analysis import resolve_roi_yml
        return resolve_roi_yml(video, self.roi_dirs)

    def device(self) -> str:
        dev = str(self.s.get("device") or "auto")
        if dev != "auto":
            return dev
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    # ---- steps -----------------------------------------------------------------------------
    def step_discover(self):
        videos, n_excl = discover_videos(self.s["videos"], self.s.get("exclude_dirs"))
        if not videos:
            raise TmazeError(f"no .mp4 videos found for {self.s['videos']} (folders are searched "
                             f"recursively, any letter case)")
        stems: dict = {}
        for v in videos:
            stems.setdefault(v.stem, []).append(v)
        dup = {k: v for k, v in stems.items() if len(v) > 1}
        if dup:
            k, v = next(iter(dup.items()))
            raise TmazeError(f"{len(dup)} file name(s) are used by more than one video (e.g. {k}: "
                             f"{v[0]} and {v[1]}). Every output table is keyed by the video file name, "
                             f"so the names must be unique - rename the files or run the folders separately.")
        self.videos = videos
        extra = f"; {n_excl} under excluded folder(s) {self.s.get('exclude_dirs')} left out" if n_excl else ""
        self.line("discover", f"{len(videos)} videos in {len({v.parent for v in videos})} folder(s){extra}")
        self.results["discover"] = {"videos": len(videos), "excluded": n_excl}

        meta = load_metadata(self.s["metadata"]) if self.s.get("metadata") else None
        self.trials = build_trials(videos, meta)
        n_named = sum(1 for t in self.trials if t["day"] and t["mouse"])
        n_rew = sum(1 for t in self.trials if t["reward"])
        if meta is None:
            self.line("metadata", f"none given: choice, timing and probe columns are written; the correctness "
                                  f"columns stay empty for all {len(videos)} videos")
        else:
            unused = meta["n_rows"] - len(meta["used"])
            self.line("metadata", f"{meta['path']}: reward known for {n_rew} of {len(videos)} videos"
                                  + (f"; correctness stays empty for the other {len(videos) - n_rew}" if n_rew < len(videos) else "")
                                  + (f"; {unused} metadata row(s) match no video" if unused else ""))
            if n_rew == 0:
                self.warn(f"no video matched a trial row of {meta['path']} (keys: "
                          f"{'file name' if meta['video_col'] else 'day + mouse from Day<N>_<mouse>_Trial<T> file names'})")
        self.results["metadata"] = {"file": self.s.get("metadata"), "reward_known": n_rew,
                                    "day_mouse_known": n_named, "videos": len(videos)}

    def step_distortion(self):
        step = "distortion_check"
        if not self.on(step):
            self.line(step, "off")
            return
        from stride.stages.distortion_check import test_video_distortion
        cfg = self.s.get("distortion") or {}
        n = max(1, int(cfg.get("max_clips", 6)))
        idx = sorted({int(round(i * (len(self.videos) - 1) / max(1, n - 1))) for i in range(min(n, len(self.videos)))})
        sample = [self.videos[i] for i in idx]
        out_csv = self.scale_dir / "distortion_check.csv"
        fp = _fingerprint([[_file_sig(v) for v in sample], cfg, self.s["scale"]["charuco"]])
        if self._done(step, fp, [out_csv]):
            self.line(step, f"already done for the same {len(sample)} videos ({out_csv.name})")
            return
        ch = self.s["scale"]["charuco"]
        kw = {}
        if ch.get("squares_x"):
            kw["squares_x"] = int(ch["squares_x"])
        if ch.get("squares_y"):
            kw["squares_y"] = int(ch["squares_y"])
        rows = []
        for v in sample:
            try:
                rows.append(test_video_distortion(v, num_frames=int(cfg.get("frames", 10)), **kw))
            except Exception as e:
                rows.append({"video": v.name, "error": f"{type(e).__name__}: {e}", "needs_undistortion": None})
        self.scale_dir.mkdir(parents=True, exist_ok=True)
        import csv
        keys: list = []
        for r in rows:
            keys.extend(k for k in r if k not in keys)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        need = sum(1 for r in rows if r.get("needs_undistortion"))
        no_board = sum(1 for r in rows if r.get("needs_undistortion") is None)
        self.line(step, f"{len(rows)} videos sampled: {need} may need undistortion, {no_board} without a "
                        f"ChArUco board in view (not assessed) -> {out_csv.name}")
        if need:
            self.warn(f"{need} of {len(rows)} sampled videos show lens distortion above the thresholds "
                      f"(see scale/distortion_check.csv); STRIDE does not undistort")
        self.results[step] = {"sampled": len(rows), "may_need_undistortion": need, "no_board": no_board}
        self.state[step] = fp

    def step_scale(self):
        sc = self.s["scale"]
        out_json = self.scale_dir / "px_per_cm.json"
        info = {"px_per_cm": None, "source": None, "maze_width_cm": float(sc.get("maze_width_cm") or 10.0)}
        if sc.get("px_per_cm"):
            try:
                val = float(sc["px_per_cm"])
            except (TypeError, ValueError):
                raise TmazeError(f"px_per_cm must be a number, got {sc['px_per_cm']!r}")
            if not val > 0:
                raise TmazeError(f"px_per_cm must be positive, got {val}")
            info.update(px_per_cm=val, source="given by the user")
            self.line("scale", f"px/cm = {val:g} (given by the user)")
        elif self.on("calibrate"):
            fp = _fingerprint([[_file_sig(v) for v in self.videos], sc["charuco"]])
            prev = None
            if self._done("calibrate", fp, [out_json]):
                try:
                    prev = json.loads(out_json.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    prev = None
            if prev is not None and prev.get("source", "").startswith(("ChArUco", "ROI")):
                info = prev
                self.line("scale", f"already done: px/cm = {info['px_per_cm'] if info['px_per_cm'] else 'per video'} ({info['source']})")
            else:
                res = self._charuco()
                if res.get("ok") and res.get("px_per_cm"):
                    info.update(px_per_cm=float(res["px_per_cm"]),
                                source=f"ChArUco board ({res.get('n_clips_used')} clips, spread "
                                       f"{res.get('spread_pct')} %, {res.get('method')})",
                                charuco={"spread_pct": res.get("spread_pct"), "n_clips_used": res.get("n_clips_used"),
                                         "n_clips": res.get("n_clips"), "method": res.get("method"),
                                         "board": res.get("board"),
                                         "readings": {str(r.get("clip")): r.get("px_per_cm") for r in (res.get("per_clip") or [])
                                                      if isinstance(r, dict)}})
                    self.line("scale", f"px/cm = {info['px_per_cm']:.4g} from the ChArUco board in view "
                                       f"({res.get('n_clips_used')} clips, spread {res.get('spread_pct')} %)")
                else:
                    info.update(source="ROI corridor width (per video)", charuco_reason=res.get("reason"))
                    self.line("scale", f"no ChArUco measurement ({res.get('reason')}); px/cm is derived per video "
                                       f"from the ROI corridor width ({info['maze_width_cm']:g} cm)")
                self.state["calibrate"] = fp
        else:
            info.update(source="ROI corridor width (per video)")
            self.line("scale", f"calibrate is off and no px_per_cm given: px/cm is derived per video from the "
                               f"ROI corridor width ({info['maze_width_cm']:g} cm)")
        self.scale = info
        self.scale_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        self.results["scale"] = {k: info.get(k) for k in ("px_per_cm", "source", "maze_width_cm")}

    def _charuco(self) -> dict:
        """px/cm from the ChArUco board in view (stride.stages.scale_calibration).

        Clips measured: the first video of every folder (recording session), at least
        ``min_clips`` in total, at most ``max_clips``.
        """
        import re
        try:
            from stride.stages.scale_calibration import calibrate_px_per_cm
        except ImportError as e:
            return {"ok": False, "px_per_cm": None,
                    "reason": f"the ChArUco calibration module could not be loaded ({e})"}
        ch = dict(self.s["scale"]["charuco"])
        max_clips = ch.pop("max_clips", None)
        min_clips = int(ch.pop("min_clips", None) or 6)
        given = ch.pop("clips", None)
        board = {k: v for k, v in ch.items() if v is not None} or None
        if given:
            clips = [Path(str(c)).expanduser() for c in ([given] if isinstance(given, (str, Path)) else given)]
            missing = [c for c in clips if not c.is_file()]
            if missing:
                raise TmazeError(f"scale.charuco.clips: video not found: {missing[0]}")
            try:
                res = calibrate_px_per_cm(clips, self.scale_dir, board=board,
                                          workers=max(1, min(int(self.s.get("workers") or 1), 8)))
            except Exception as e:
                return {"ok": False, "px_per_cm": None, "reason": f"calibration failed: {type(e).__name__}: {e}"}
            if not res.get("ok"):
                res["reason"] = res.get("reason") or "no ChArUco board found in the given clips"
            return res
        groups: dict = {}
        for v in self.videos:
            groups.setdefault(v.parent, []).append(v)
        natural = lambda d: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(d))]  # noqa: E731
        ordered = [sorted(groups[d]) for d in sorted(groups, key=natural)]
        try:
            from stride.stages.scale_calibration import pick_clips
            clips = pick_clips(ordered, minimum=min_clips)
        except Exception:
            clips = [g[0] for g in ordered]
        try:
            res = calibrate_px_per_cm(clips, self.scale_dir, board=board,
                                      max_clips=int(max_clips) if max_clips else None,
                                      workers=max(1, min(int(self.s.get("workers") or 1), 8)))
        except Exception as e:
            return {"ok": False, "px_per_cm": None, "reason": f"calibration failed: {type(e).__name__}: {e}"}
        if not isinstance(res, dict):
            return {"ok": False, "px_per_cm": None, "reason": "calibration returned no result"}
        if not res.get("ok"):
            res["reason"] = res.get("reason") or "no ChArUco board found in the sampled videos"
        return res

    def step_pose(self):
        step = "pose_inference"
        need = [v for v in self.videos if not self.pose_of(v).exists()]
        have = len(self.videos) - len(need)
        res = {"existing": have, "done": 0, "failed": 0}
        if not need:
            self.line(step, f"all {have} pose files exist: nothing to do")
        elif not self.on(step):
            self.line(step, f"off: {have} videos have a pose file, {len(need)} do not and are left out")
            for v in need:
                self.skip(v, f"no pose file (<video name>{self.suffix}) and pose_inference is off")
        else:
            from stride.config import check_device, require_sleap_nn, resolve_model_paths
            from stride.stages.pose_inference import run_single_pose_inference
            if not self.s["pose_model"]:
                raise TmazeError(f"{len(need)} video(s) have no pose file: pose inference needs your trained "
                                 f"pose model (--pose-model <centroid> --pose-model <centered_instance>, or "
                                 f"`pose_model:` in the config). Use --skip-inference to analyse only the "
                                 f"videos that already have pose and ROI files.")
            dev = self.device()
            try:
                models = resolve_model_paths(self.s["pose_model"], "pose")
                require_sleap_nn()
                check_device(dev)
            except (ValueError, RuntimeError) as e:
                raise TmazeError(str(e))
            self.pose_dir.mkdir(parents=True, exist_ok=True)
            print(f"[{step}] running on {len(need)} video(s), device {dev}", flush=True)
            for i, v in enumerate(need, 1):
                r = run_single_pose_inference(video_path=v, output_dir=self.pose_dir, model_paths=models,
                                              overwrite=False, batch_size=int(self.s["batch_size"]), device=dev)
                made = self.pose_dir / f"{v.stem}.slp"
                if r["status"] in ("ok", "skip") and made.exists() and self.suffix != ".slp":
                    made.replace(self.pose_dir / f"{v.stem}{self.suffix}")
                if r["status"] in ("ok", "skip"):
                    res["done"] += 1
                else:
                    res["failed"] += 1
                    why = r.get("error") or r.get("reason") or "failed"
                    self.skip(v, f"pose inference failed: {why}")
                print(f"  [{i}/{len(need)}] {r['status'].upper():5} {v.name}"
                      + (f"  ({r.get('error') or r.get('reason')})" if r["status"] not in ("ok", "skip") else ""), flush=True)
            self.line(step, f"{have} existing, {res['done']} inferred, {res['failed']} failed (device {dev})")
            res["device"] = dev
        self.results[step] = res

    def step_roi(self):
        step = "roi_inference"
        sources: dict = {}
        for v in self.videos:
            y = self.roi_of(v)
            if y is not None:
                from stride.stages.roi_consensus import is_borrowed_roi
                sources[v] = ("consensus" if is_borrowed_roi(y) else "detected", y, "existing file")
        need = [v for v in self.videos if v not in sources]
        res = {"existing": len(sources), "detected": 0, "failed": 0, "consensus": 0, "missing": 0}
        if not need:
            self.line(step, f"all {len(sources)} ROI files exist: nothing to do")
        elif not self.on(step):
            self.line(step, f"off: {len(sources)} videos have an ROI file, {len(need)} do not")
        else:
            from stride.config import check_device, require_sleap_nn, resolve_model_paths
            from stride.stages.roi_inference import roi_output_path, run_single_video_inference
            from stride.stages.slp_to_yaml import slp_to_roi_yaml
            if not self.s["roi_model"]:
                raise TmazeError(f"{len(need)} video(s) have no ROI file: ROI inference needs your trained maze-ROI "
                                 f"model (--roi-model <centroid> --roi-model <centered_instance>, or `roi_model:` "
                                 f"in the config). Use --skip-inference to analyse only the videos that already "
                                 f"have pose and ROI files.")
            dev = self.device()
            todo = [v for v in need if not roi_output_path(v, self.roi_dir).exists()]
            try:
                models = resolve_model_paths(self.s["roi_model"], "ROI")
                if todo:
                    require_sleap_nn()
                    check_device(dev)
            except (ValueError, RuntimeError) as e:
                raise TmazeError(str(e))
            self.roi_dir.mkdir(parents=True, exist_ok=True)
            rcfg = self.s.get("roi") or {}
            print(f"[{step}] running on {len(need)} video(s), device {dev}", flush=True)
            for i, v in enumerate(need, 1):
                r = run_single_video_inference(
                    video_path=v, output_dir=self.roi_dir, model_paths=models, overwrite=False,
                    frames_to_check=int(rcfg.get("frames", 10)), device=dev,
                    full_pass=bool(rcfg.get("full_pass", True)))
                note = ""
                if r["status"] in ("ok", "skip"):
                    yml = self.roi_dir / f"{v.stem}.rois.yml"
                    try:
                        slp_to_roi_yaml(str(roi_output_path(v, self.roi_dir)), str(yml))
                        sources[v] = ("detected", yml, f"{r.get('pass', 'earlier run')} pass")
                        res["detected"] += 1
                    except Exception as e:
                        note = f"ROI conversion failed: {e}"
                else:
                    note = r.get("error") or r.get("reason") or "failed"
                if note:
                    res["failed"] += 1
                    sources[v] = ("missing", None, note)
                print(f"  [{i}/{len(need)}] {('OK' if not note else 'FAIL'):5} {v.name}" + (f"  ({note})" if note else ""), flush=True)
            self.line(step, f"{res['existing']} existing, {res['detected']} detected, {res['failed']} without a usable frame")
            res["device"] = dev
        for v in self.videos:
            sources.setdefault(v, ("missing", None, "no ROI file"))
        self.roi_sources = sources
        self.results[step] = res
        if self._stop(step):
            self._write_roi_sources()

    def step_consensus(self):
        step = "roi_consensus"
        from stride.stages.roi_consensus import ConsensusError, DRIFT_WARN_PX, propagate_consensus
        sources = self.roi_sources
        missing = [v for v in self.videos if sources[v][0] == "missing"]
        res = self.results.setdefault("roi_inference", {})
        if not missing:
            self.line(step, "no clip without an ROI: nothing to do")
        elif not self.on(step):
            self.line(step, f"off: {len(missing)} clip(s) without an ROI are left out")
        else:
            rcfg = self.s.get("roi") or {}
            drift = float(rcfg.get("drift_max_px") or DRIFT_WARN_PX)
            by_folder = (rcfg.get("session") or "folder") == "folder"
            sessions: dict = {}
            for v in self.videos:
                sessions.setdefault(v.parent if by_folder else Path("all videos"), []).append(v)
            n_rec = 0
            for sess, vids in sessions.items():
                miss = [v for v in vids if sources[v][0] == "missing"]
                if not miss:
                    continue
                donors = [sources[v][1] for v in vids if sources[v][0] == "detected"]
                qc = self.roi_dir / "roi_consensus_qc" / (sess.name if by_folder else "all")
                try:
                    out = propagate_consensus(miss, donors, lambda v: self.roi_dir / f"{v.stem}.rois.yml", qc,
                                              drift_warn=drift, on_drift="fail", verbose=False)
                    for v in miss:
                        yml = self.roi_dir / f"{v.stem}.rois.yml"
                        if yml.exists():
                            sources[v] = ("consensus", yml, out["tag"])
                            n_rec += 1
                    print(f"  {sess}: {len(out['recovered'])} clip(s) got the consensus ROI ({out['tag']}); "
                          f"check {qc}", flush=True)
                except ConsensusError as e:
                    for v in miss:
                        sources[v] = ("missing", None, f"{sources[v][2]}; {e}")
                    print(f"  {sess}: {len(miss)} clip(s) stay without an ROI - {e}", flush=True)
            res["consensus"] = n_rec
            still = sum(1 for v in self.videos if sources[v][0] == "missing")
            self.line(step, f"{n_rec} clip(s) got the consensus ROI of their {'folder' if by_folder else 'run'}, "
                            f"{still} stay without an ROI")
            if n_rec:
                self.warn(f"{n_rec} clip(s) use a consensus (borrowed) ROI: open the montage(s) under "
                          f"roi/roi_consensus_qc/ and confirm the ROI lands on the maze")
        for v in self.videos:
            if sources[v][0] == "missing":
                self.skip(v, f"no ROI: {sources[v][2]}")
        res["missing"] = sum(1 for v in self.videos if sources[v][0] == "missing")
        self._write_roi_sources()

    def _write_roi_sources(self):
        import csv
        self.roi_dir.mkdir(parents=True, exist_ok=True)
        with open(self.roi_dir / "roi_sources.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "video_path", "roi_source", "roi_file", "note"])
            for v in self.videos:
                src, yml, note = self.roi_sources[v]
                w.writerow([v.stem, str(v), src, str(yml) if yml else "", note])

    def step_decisions(self):
        step = "decisions"
        outs = [self.dec_dir / n for n in ("decisions.csv", "events.csv", "metrics.csv")]
        if not self.on(step):
            self.line(step, "off" + ("" if all(o.exists() for o in outs) else
                                     " (no decisions files in the output folder: the gait step uses the video list)"))
            return
        from stride.stages.decision_analysis import run_decision_analysis
        usable = [t for t in self.trials
                  if self.pose_of(Path(t["video_path"])).exists() and self.roi_of(Path(t["video_path"])) is not None]
        if not usable:
            raise TmazeError("no video has both a pose file and an ROI file, so there is nothing to analyse. "
                             + ("Inference is switched off: " if not (self.on("pose_inference") and self.on("roi_inference")) else "")
                             + f"Pose files are looked for as <video name>{self.suffix} next to the video or in "
                               f"{self.pose_dir}; ROI files as <video name>.rois.yml next to the video or in {self.roi_dir}.")
        fp = _fingerprint([[t["video_path"], t["day"], t["mouse"], t["trial"], t["reward"],
                            _file_sig(self.pose_of(Path(t["video_path"]))), _file_sig(self.roi_of(Path(t["video_path"])))]
                           for t in usable])
        if self._done(step, fp, outs):
            self.line(step, f"already done for the same {len(usable)} videos")
        else:
            payload = [{k: t[k] for k in ("video_path", "day", "mouse", "trial", "reward")} for t in usable]
            summ = run_decision_analysis(
                video_dir=Path(str(self.s["videos"] if not isinstance(self.s["videos"], (list, tuple)) else self.out)),
                yml_dir=self.roi_dirs, meta_csv=None, output_dir=self.dec_dir,
                n_workers=int(self.s["workers"]), pose_slp_suffix=self.suffix,
                trials=payload, pose_dir=self.pose_dirs)
            for name, status in (summ.get("skipped_videos") or {}).items():
                self.skip(name, f"decision analysis: {status}")
            if summ.get("ok", 0) == 0:
                raise TmazeError("the decision analysis produced no row (see the reasons above)")
            self.results[step] = {k: summ.get(k) for k in ("total", "ok", "skipped", "no_arm_entry", "reward_unknown")}
            self.line(step, f"{summ['ok']} of {len(self.videos)} videos analysed"
                            + (f", {summ.get('no_arm_entry')} with no arm entry" if summ.get("no_arm_entry") else "")
                            + (f", correctness empty for {summ.get('reward_unknown')} (reward unknown)" if summ.get("reward_unknown") else ""))
            self.state[step] = fp
        self.decisions_fp = fp

    def step_rules(self):
        step = "decision_rules"
        raw = self.dec_dir / "decisions.csv"
        clean = self.dec_dir / "decisions_clean.csv"
        if not self.on(step):
            self.line(step, "off")
            return
        if not raw.exists():
            self.line(step, f"left out: {raw} does not exist (the decisions step is off or produced nothing)")
            return
        from stride.stages.decision_rules import apply_decision_rules
        window = float((self.s.get("decision_rules") or {}).get("window_s", 150.0))
        fp = _fingerprint([_file_sig(raw), window])
        if self._done(step, fp, [clean]):
            self.line(step, "already done")
            return
        r = apply_decision_rules(raw, window_s=window, verbose=False)
        self.line(step, f"{r['reclassified']} of {r['trials']} trials reclassified to no decision "
                        f"({r['soft']} without a full commit, {r['late']} after {window:g} s) -> decisions_clean.csv "
                        f"({r['decisions']} decisions)")
        self.results[step] = {k: r[k] for k in ("trials", "soft", "late", "decisions", "no_decision", "window_s")}
        self.state[step] = fp

    def step_gait(self):
        step = "gait"
        if not self.on(step):
            self.line(step, "off")
            return
        import pandas as pd
        from stride.stages.gait_extraction import compute_keypoint_confidence, run_gait_extraction
        from stride.stages.run_gait_extraction import build_params, filter_strides

        import csv
        metrics = self.dec_dir / "metrics.csv"
        by_path = {t["video_path"]: t for t in self.trials}
        self.gait_dir.mkdir(parents=True, exist_ok=True)
        meta_csv = self.gait_dir / "metrics_meta.csv"
        # The table the gait stage reads: the decisions metrics + Gender + Genotype, written as text
        # and read back with pandas exactly as the stage does when it is given the file.
        if metrics.exists():
            with open(metrics, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                cols = list(reader.fieldnames or [])
                text_rows = [r for r in reader if str(r.get("video_path")) in by_path]
            src = "the videos with a decisions row"
        else:
            cols = ["video_path", "stem", "day", "mouse", "trial"]
            text_rows = [{"video_path": t["video_path"], "stem": Path(t["video_path"]).stem, "day": t["day"],
                          "mouse": t["mouse"], "trial": "" if t["trial"] is None else t["trial"]}
                         for t in self.trials if self.pose_of(Path(t["video_path"])).exists()]
            src = "the videos with a pose file (no decisions table)"
        for r in text_rows:
            t = by_path[str(r["video_path"])]
            r["Gender"], r["Genotype"] = t["sex"], t["genotype"]
        with open(meta_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols + ["Gender", "Genotype"])
            w.writeheader()
            w.writerows(text_rows)
        rows = []
        if text_rows:
            for r in pd.read_csv(meta_csv).to_dict("records"):
                t = by_path[str(r["video_path"])]
                # ids as text (a column with empty cells would otherwise turn 101 into 101.0);
                # unknown values stay empty instead of "nan"
                r["day"], r["mouse"] = t["day"], t["mouse"]
                r["Gender"], r["Genotype"] = t["sex"], t["genotype"]
                rows.append(r)
        if not rows:
            raise TmazeError("the gait step has no video to work on (no decisions row and no pose file)")

        cfg = {"stride_detection": dict(self.s.get("stride_detection") or {}),
               "stride_filtering": dict(self.s.get("stride_filtering") or {})}
        params = build_params(cfg)
        params["DEFAULT_FPS"] = float(self.s.get("fps_default") or 120.0)
        params["MAZE_WIDTH_CM"] = float(self.scale.get("maze_width_cm") or 10.0)
        if self.scale.get("px_per_cm"):
            params["PX_PER_CM_OVERRIDE"] = float(self.scale["px_per_cm"])
        outs = [self.gait_dir / n for n in ("gait_per_stride.csv", "gait_per_step.csv", "gait_per_video.csv",
                                            "gait_per_stride_filtered.csv", "pose_keypoint_confidence_by_video.csv")]
        fp = _fingerprint([params, cfg, [[r["video_path"], r.get("Gender"), r.get("Genotype"),
                                          _file_sig(self.pose_of(Path(r["video_path"]))),
                                          _file_sig(self.roi_of(Path(r["video_path"])) or "")] for r in rows]])
        if self._done(step, fp, outs):
            self.line(step, f"already done for the same {len(rows)} videos")
            return
        if self.state.get(step) == fp and self.state.get("gait_no_strides") and (self.gait_dir / "gait_per_video.csv").exists():
            self.line(step, f"already done for the same {len(rows)} videos (no stride detected)")
            return
        column_map = {"mouse_id": ["mouse"], "mouse_id_sep": "", "sex": "Gender"}
        print(f"[{step}] stride detection on {len(rows)} videos ({src})", flush=True)
        summ = run_gait_extraction(meta_csv=meta_csv, yml_dir=self.roi_dirs, output_dir=self.gait_dir, params=params,
                                   pose_slp_suffix=self.suffix, column_map=column_map,
                                   n_workers=int(self.s["workers"]), meta_rows=rows, pose_dir=self.pose_dirs)
        conf = self.gait_dir / "pose_keypoint_confidence_by_video.csv"
        compute_keypoint_confidence(video_dir=None, pose_slp_suffix=self.suffix, output_path=conf,
                                    videos=self.videos, pose_dir=self.pose_dirs)
        stride_csv = self.gait_dir / "gait_per_stride.csv"
        filtered = self.gait_dir / "gait_per_stride_filtered.csv"
        n_filtered = None
        if stride_csv.exists():
            n_filtered = filter_strides(stride_csv, conf, filtered, cfg)
        total = summ.get("total_strides") if isinstance(summ, dict) else None
        if not total:
            self.line(step, f"{summ.get('ok') if isinstance(summ, dict) else '?'} videos processed, no stride "
                            f"detected (clips too short or no walking): no stride tables written")
            self.warn("the gait step found no stride in any video")
        else:
            self.line(step, f"{summ.get('ok') if isinstance(summ, dict) else '?'} videos, {total} strides, "
                            f"{n_filtered} after the filters (px/cm: {self.scale.get('source')})")
        self.results[step] = {"videos": len(rows), "strides": total, "filtered_strides": n_filtered,
                              "px_per_cm": self.scale.get("px_per_cm"), "px_per_cm_source": self.scale.get("source")}
        try:   # the gait stage's own warnings (scale, frame rate, stem direction) go into the summary too
            man = yaml.safe_load((self.gait_dir / "gait_run_manifest.yaml").read_text(encoding="utf-8")) or {}
            for w in (man.get("results") or {}).get("warnings") or []:
                self.warnings.append(f"gait: {w}")
        except Exception:
            pass
        self.state[step] = fp
        self.state["gait_no_strides"] = not total

    def step_stats(self):
        """Optional genotype statistics: the existing mixed-model and classifier stages, unchanged."""
        wanted = [s for s in ("mixed_models", "classifier") if self.on(s)]
        if not wanted:
            return
        import subprocess
        stats_cfg = self.s.get("stats_config")
        if not stats_cfg or not Path(str(stats_cfg)).exists():
            raise TmazeError(f"step(s) {wanted} are switched on but need `stats_config`: a STRIDE config with the "
                             f"metrics / covariates / models / statistics sections (config.example.yaml in the "
                             f"repository). Got: {stats_cfg}")
        strides = self.gait_dir / "gait_per_stride_filtered.csv"
        if not strides.exists():
            raise TmazeError(f"step(s) {wanted} need {strides}, which does not exist (the gait step is off or produced nothing)")
        if not any(t["genotype"] for t in self.trials):
            raise TmazeError(f"step(s) {wanted} compare genotypes, but no video has a genotype: add a `genotype` "
                             f"column to the metadata file")
        with open(stats_cfg, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg["batches"] = {"tmaze": {
            "name": "tmaze", "stride_csv": str(strides),
            "decisions_csv": str(self.dec_dir / "decisions.csv"), "events_csv": str(self.dec_dir / "events.csv"),
            "metrics_csv": str(self.dec_dir / "metrics.csv"), "metrics_meta_csv": str(self.gait_dir / "metrics_meta.csv"),
            "has_sex": any(t["sex"] for t in self.trials), "mouse_id_cols": ["mouse"], "mouse_id_sep": "",
            "column_map": {"mouse_id": ["mouse"], "mouse_id_sep": "", "sex": "Gender"},
        }}
        stats_dir = self.out / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        gen = stats_dir / "stats_config.generated.yaml"
        with open(gen, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        mods = {"mixed_models": ("stride.stages.gait_lmm", ["--batch", "tmaze"]),
                "classifier": ("stride.stages.classification", [])}
        for s in wanted:
            mod, extra = mods[s]
            cmd = [sys.executable, "-m", mod, "--config", str(gen), "--output", str(stats_dir)] + extra
            rc = subprocess.run(cmd).returncode
            if rc != 0:
                raise TmazeError(f"step {s} failed (exit code {rc}); its messages are above")
            self.line(s, f"done -> {stats_dir}")
            self.results[s] = {"output": str(stats_dir)}

    # ---- run -------------------------------------------------------------------------------
    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        log_file = open(self.out / "tmaze_run.log", "a", encoding="utf-8")
        old_out = sys.stdout
        sys.stdout = _Tee(old_out, log_file)
        ok, error = True, None
        try:
            print(f"\n===== stride tmaze  {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
            off = [s for s in STEPS if not self.on(s)]
            print(f"output: {self.out}\nsteps off: {', '.join(off) if off else 'none'}", flush=True)
            sequence = [("discover", self.step_discover), ("distortion_check", self.step_distortion),
                        ("calibrate", self.step_scale), ("pose_inference", self.step_pose),
                        ("roi_inference", self.step_roi), ("roi_consensus", self.step_consensus),
                        ("decisions", self.step_decisions), ("decision_rules", self.step_rules),
                        ("gait", self.step_gait), ("mixed_models", self.step_stats)]
            for name, fn in sequence:
                fn()
                self._save_state()
                if self.s.get("stop_after") == name:
                    self.line("stop", f"stopped after {name} as requested")
                    break
        except TmazeError as e:
            ok, error = False, str(e)
            print(f"\nERROR: {e}", flush=True)
        finally:
            self._save_state()
            summary = self._finish(ok, error)
            sys.stdout = old_out
            log_file.close()
        return summary

    def _finish(self, ok: bool, error: Optional[str]) -> dict:
        n_videos = len(getattr(self, "videos", []))
        summary = {"ok": ok, "error": error, "output": str(self.out), "videos": n_videos,
                   "stages": self.results, "skipped": dict(self.skipped), "warnings": list(self.warnings),
                   "elapsed_s": round(time.time() - self.t0, 1)}
        if self.skipped and ok:
            print(f"\n{len(self.skipped)} of {n_videos} video(s) were left out:", flush=True)
            for name, why in list(self.skipped.items())[:20]:
                print(f"  {name}: {why}", flush=True)
            if len(self.skipped) > 20:
                print(f"  ... and {len(self.skipped) - 20} more (SUMMARY.md lists all)", flush=True)
        lines = ["# STRIDE T-maze run", "", f"- finished: {time.strftime('%Y-%m-%d %H:%M:%S')}  ({summary['elapsed_s']} s)",
                 f"- result: {'completed' if ok else 'STOPPED: ' + str(error)}", f"- output folder: `{self.out}`",
                 f"- videos: {n_videos}", ""]
        lines += ["## Steps", ""] + [f"- {l}" for l in self.stage_lines] + [""]
        if getattr(self, "scale", None):
            lines += ["## Scale", "", f"- px/cm: {self.scale.get('px_per_cm') if self.scale.get('px_per_cm') else 'derived per video'}",
                      f"- source: {self.scale.get('source')}", ""]
        if self.warnings:
            lines += ["## Warnings", ""] + [f"- {w}" for w in self.warnings] + [""]
        if self.skipped:
            lines += [f"## Videos left out ({len(self.skipped)})", "", "| video | reason |", "|---|---|"]
            lines += [f"| {n} | {str(w).replace('|', '/')} |" for n, w in self.skipped.items()] + [""]
        try:
            (self.out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            pass
        try:
            from stride.run_manifest import write_run_manifest
            settings = {k: v for k, v in self.s.items()}
            write_run_manifest(
                self.out, "tmaze",
                models=[m for m in (list(self.s.get("pose_model") or []) + list(self.s.get("roi_model") or []))],
                params=settings,
                inputs={"videos": self.s.get("videos"), "n_videos": n_videos, "metadata": self.s.get("metadata")},
                outputs=[p for p in [self.dec_dir / "decisions.csv", self.dec_dir / "decisions_clean.csv",
                                     self.dec_dir / "events.csv", self.dec_dir / "metrics.csv",
                                     self.gait_dir / "gait_per_stride.csv", self.gait_dir / "gait_per_stride_filtered.csv"]
                         if p.exists()],
                results={"ok": ok, "error": error, "stages": self.results, "n_skipped": len(self.skipped),
                         "skipped": dict(self.skipped), "warnings": list(self.warnings),
                         "roi_sources": self._roi_source_counts()},
            )
        except Exception:
            pass
        if ok:
            print(f"\nDone in {summary['elapsed_s']} s. Tables: {self.dec_dir} and {self.gait_dir}; summary: {self.out / 'SUMMARY.md'}", flush=True)
        return summary

    def _roi_source_counts(self) -> dict:
        counts: dict = {}
        for src, _, _ in getattr(self, "roi_sources", {}).values():
            counts[src] = counts.get(src, 0) + 1
        return counts


def run_tmaze_pipeline(settings: Optional[dict] = None, **overrides) -> dict:
    """Run the pipeline. ``settings`` as returned by :func:`load_settings` (or pass keyword overrides).

    Returns a summary dict (``ok``, ``error``, ``stages``, ``skipped``, ``warnings``, ``output``).
    Raises nothing for expected problems: ``ok`` is False and ``error`` says why.
    """
    if settings is None:
        settings = load_settings(None, overrides)
    elif overrides:
        settings = _deep_update(copy.deepcopy(settings), overrides)
    try:
        run = TmazeRun(settings)
    except TmazeError as e:
        print(f"ERROR: {e}")
        return {"ok": False, "error": str(e), "stages": {}, "skipped": {}, "warnings": []}
    return run.run()
