"""
STRIDE Gait: the gait-only pipeline (`stride gait`).

Videos with pose tracking in (pose files, or a trained pose model to make them), per-stride and
per-video gait tables out. No maze ROI is used::

    discover -> scale (px/cm) -> pose inference (only when a pose model is given)
             -> stride detection -> keypoint confidence -> stride filtering

Nothing here computes a metric. Every step calls the stage that the rest of the package uses
(``run_gait_extraction``, ``compute_keypoint_confidence``, ``filter_strides``, the pose inference
stage), so the numbers are the same whichever way the stages are started.

Settings come from an optional YAML config (`stride init-config gait` writes a commented
template); any command-line option overrides the config value. Every step can be switched off.

Output folder::

    SUMMARY.md, gait_run.log, stride_gait_run_manifest.yaml, gait_state.json
    scale/   px_per_cm.json, (ChArUco QC files)
    pose/    <video stem>.slp                 only for videos that had no pose file
    gait/    gait_per_stride.csv, gait_per_step.csv, gait_per_video.csv,
             gait_per_stride_filtered.csv, pose_keypoint_confidence_by_video.csv,
             videos.csv, run_params.json
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

from stride.stages.tmaze_pipeline import (
    TmazeError, TmazeRun, _Tee, _VIDEO_KEYS, _deep_update, _file_sig, _fingerprint, _norm_day,
    load_settings as _load_settings, write_config_template as _write_config_template,
)

STEPS = ["calibrate", "pose_inference", "gait_extraction", "keypoint_confidence", "stride_filtering"]

# keypoints the gait stage reads (names as in the STRIDE pose skeleton)
REQUIRED_KEYPOINTS = ["snout", "tailbase"]
HINDPAW_KEYPOINTS = {"left": ["hindpawL1", "hindpawL2"], "right": ["hindpawR1", "hindpawR2"]}
OPTIONAL_KEYPOINTS = ["tailtip"]

DEFAULTS: dict = {
    "videos": None,                 # folder, file, glob pattern, or a list of these
    "output": None,
    "pose_model": [],               # one or two paths; pose inference runs only when given
    "pose_dir": None,               # folder holding existing pose files (when not next to the videos)
    "pose_suffix": ".slp",
    "metadata": None,               # optional CSV keyed by video file name
    "exclude_dirs": ["waste"],
    "device": "auto",
    "workers": 4,
    "batch_size": 16,
    "fps_default": 120.0,           # used only when the frame rate cannot be read from a video
    "scale": {
        "px_per_cm": None,
        "charuco": {
            "squares_x": None, "squares_y": None, "square_mm": None, "marker_mm": None,
            "dictionary": None, "min_clips": 6, "max_clips": None, "clips": None,
        },
    },
    "keypoints": {},                # node name in your pose files -> STRIDE keypoint name
    "stride_detection": {},         # optional overrides; defaults are the published values
    "stride_filtering": {},
    "steps": {s: True for s in STEPS},
    "stop_after": None,
    "force": False,
    "strict": False,
}

CONFIG_TEMPLATE = """\
# STRIDE Gait configuration  (`stride gait --config this_file.yaml`)
# Any command-line option overrides the value written here. Paths may contain spaces; relative
# paths are resolved from the folder you start the command in.

videos: /path/to/videos            # a folder (searched recursively), one file, a pattern such as
                                   # "D:/recordings/*/*.mp4", or a list of these. Any file names.
output: /path/to/output            # everything STRIDE writes goes here (your videos are not touched)

pose_dir: null                     # folder with existing pose files, if they are not next to the videos
pose_suffix: .slp                  # pose file = <video name><this suffix>
pose_model: []                     # OR your trained SLEAP pose model (one or two folders): pose
                                   # inference then runs for the videos that have no pose file
                                   #   - /path/to/pose/centroid
                                   #   - /path/to/pose/centered_instance

scale:
  px_per_cm: null                  # a value you measured yourself; wins over the board
  charuco:                         # else: the calibration board visible in the videos; null = the
    squares_x: null                # default board (14 x 9 squares, 20 mm squares, 15 mm markers,
    squares_y: null                # DICT_5X5_250). Without a value and without a board the run
    square_mm: null                # stops: there is no other way to get the scale.
    marker_mm: null
    dictionary: null
    min_clips: 6                   # videos measured: the first of every folder, at least this many
    max_clips: null
    clips: null                    # or: a list of videos to measure instead of that choice

keypoints: {}                      # only when your pose skeleton uses other node names:
                                   #   nose: snout
                                   #   tail_base: tailbase
                                   #   tail_end: tailtip
                                   #   left_hindpaw: hindpawL1      # one landmark per hind paw is enough
                                   #   right_hindpaw: hindpawR1
                                   # needed: snout, tailbase, at least one landmark per hind paw
                                   # (hindpawL1 / hindpawL2, hindpawR1 / hindpawR2); tailtip optional

metadata: null                     # optional CSV with a `video` column (file name) and any other
                                   # columns (animal, sex, group, ...): they are added to the tables

device: auto                       # auto | cuda | cuda:0 | cpu   (pose inference only)
workers: 4                         # CPU worker processes
batch_size: 16                     # pose inference batch size
fps_default: 120                   # used only when a video's frame rate cannot be read
exclude_dirs: [waste]              # folder names skipped when searching for videos

steps:                             # switch any step off with `false`
  calibrate: true                  # px/cm from the ChArUco board (not needed with scale.px_per_cm)
  pose_inference: true             # runs only when pose_model is given and a video has no pose file
  gait_extraction: true            # stride detection and the per-stride metrics
  keypoint_confidence: true        # mean keypoint confidence per video (used by the filter)
  stride_filtering: true           # confidence, first/last stride, turning, forward direction

stop_after: null                   # stop after this step, e.g. pose_inference
"""


class GaitError(TmazeError):
    """The gait pipeline cannot run or cannot continue; the message says why (exit code 1)."""


def load_settings(config_path=None, overrides: Optional[dict] = None) -> dict:
    """Defaults <- YAML config <- overrides (command-line options that were given)."""
    try:
        settings = _load_settings(config_path, overrides, defaults=DEFAULTS, steps=STEPS, pipeline="gait",
                                  list_keys=("pose_model", "exclude_dirs"))
    except TmazeError as e:
        raise GaitError(str(e))
    kp = settings.get("keypoints")
    if kp is None:
        settings["keypoints"] = {}
    elif not isinstance(kp, dict):
        raise GaitError(f"`keypoints` must be a mapping (node name in your pose files -> STRIDE keypoint "
                        f"name), got {kp!r}")
    return settings


def write_config_template(path, overwrite: bool = False) -> Path:
    try:
        return _write_config_template(path, overwrite=overwrite, template=CONFIG_TEMPLATE)
    except TmazeError as e:
        raise GaitError(str(e))


# --------------------------------------------------------------------------------------------
# keypoints
# --------------------------------------------------------------------------------------------

def resolve_keypoints(node_names: list, mapping: Optional[dict] = None) -> dict:
    """Match the node names of a pose file with the keypoints the gait stage reads.

    ``mapping``: node name in the pose file -> STRIDE keypoint name (from the config).

    Returns a dict:
        aliases            STRIDE name -> node name in the pose file (or None for an optional
                           keypoint the skeleton does not have); only entries that differ from the
                           plain name, so an unmapped STRIDE skeleton gives {} (the stage then
                           runs exactly as it always does)
        confidence_names   keypoint names for the confidence table, or None to keep the standard 15
        notes              lines for the log
    Raises GaitError listing what is missing and what the pose file contains.
    """
    from stride.stages.gait_extraction import KEYPOINT_NAMES

    nodes = [str(n) for n in node_names]
    mapping = {str(k): str(v) for k, v in (mapping or {}).items()}
    bad_src = sorted(k for k in mapping if k not in nodes)
    if bad_src:
        raise GaitError(f"keypoints: {bad_src} {'is' if len(bad_src) == 1 else 'are'} not in the pose file. "
                        f"The pose file has these nodes: {nodes}")
    bad_dst = sorted({v for v in mapping.values() if v not in KEYPOINT_NAMES})
    if bad_dst:
        raise GaitError(f"keypoints: {bad_dst} {'is' if len(bad_dst) == 1 else 'are'} not STRIDE keypoint "
                        f"name(s). STRIDE keypoints: {KEYPOINT_NAMES}")
    by_target: dict = {}
    for src, dst in mapping.items():
        if dst in by_target:
            raise GaitError(f"keypoints: both {by_target[dst]!r} and {src!r} are mapped to {dst!r}")
        by_target[dst] = src

    def source(name):                                   # node in the pose file that plays `name`
        if name in by_target:
            return by_target[name]
        return name if name in nodes else None

    notes, aliases, missing = [], {}, []
    for name in REQUIRED_KEYPOINTS:
        if source(name) is None:
            missing.append(name)
    for side, (a, b) in HINDPAW_KEYPOINTS.items():
        sa, sb = source(a), source(b)
        if sa is None and sb is None:
            missing.append(f"{a} or {b} (a {side} hind paw landmark)")
        elif sa is None or sb is None:
            only = sa if sa is not None else sb
            aliases[a], aliases[b] = only, only
            notes.append(f"one {side} hind paw landmark ({only!r}): it is used as the {side} hind paw; "
                         f"no choice between two landmarks is made")
    if missing:
        raise GaitError(f"the pose file lacks keypoint(s) the gait analysis needs: {missing}. The pose file "
                        f"has these nodes: {nodes}. If they exist under other names, map them with "
                        f"`keypoints:` in the config or --keypoint THEIR_NAME=STRIDE_NAME (needed: snout, "
                        f"tailbase, at least one of hindpawL1 / hindpawL2 and one of hindpawR1 / hindpawR2; "
                        f"optional: tailtip).")
    for name in OPTIONAL_KEYPOINTS:
        if source(name) is None:
            aliases[name] = None
            notes.append(f"no {name!r} keypoint: the tail-tip columns stay empty")
    for name in REQUIRED_KEYPOINTS + [n for pair in HINDPAW_KEYPOINTS.values() for n in pair] + OPTIONAL_KEYPOINTS:
        src = source(name)
        if src is not None and src != name:
            aliases.setdefault(name, src)

    confidence_names = None
    if any(source(k) is None for k in KEYPOINT_NAMES):
        # not the 15-keypoint STRIDE skeleton: the per-video mean confidence is taken over the
        # nodes this skeleton has (under their STRIDE name where mapped)
        back = {src: dst for dst, src in by_target.items()}
        confidence_names = []
        for n in nodes:
            name = back.get(n, n)
            if name not in confidence_names:
                confidence_names.append(name)
        notes.append(f"the pose skeleton is not the 15-keypoint STRIDE skeleton: the mean keypoint confidence of "
                     f"a video is taken over its {len(confidence_names)} nodes")
    conf_aliases = {dst: src for dst, src in by_target.items() if dst != src}
    return {"aliases": aliases, "confidence_names": confidence_names, "confidence_aliases": conf_aliases,
            "notes": notes}


def read_pose_node_names(pose_path, video_path=None) -> list:
    from stride.utils.pose_io import load_pose_labels
    labels = load_pose_labels(pose_path, video_path)
    return list(labels.skeleton.node_names)


# --------------------------------------------------------------------------------------------
# metadata (keyed by video file name)
# --------------------------------------------------------------------------------------------

_ID_KEYS = ("mouse_id", "mouse", "animal", "animal_id", "subject", "id")
_SEX_KEYS = ("sex", "gender")


def load_gait_metadata(path) -> dict:
    """Read the optional metadata CSV: one row per video, found by its file name.

    Needs a `video` column (file name, with or without folder and extension). Every other column
    is kept. Recognised (case-insensitive): an animal id (`mouse_id`, `mouse`, `animal`,
    `animal_id`, `subject`, `id`), `sex` / `gender`, `genotype`, `day`.
    """
    p = Path(str(path)).expanduser()
    if not p.exists():
        raise GaitError(f"metadata file not found: {p}")
    with open(p, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header = [str(c) for c in (reader.fieldnames or [])]
        rows = [{str(k): ("" if v is None else str(v).strip()) for k, v in r.items() if k is not None} for r in reader]
    low = {c.strip().lower(): c for c in header}
    video_col = next((low[k] for k in _VIDEO_KEYS if k in low), None)
    if video_col is None:
        raise GaitError(f"{p.name}: needs a `video` column holding the video file name; found {header}")
    by_stem: dict = {}
    for i, r in enumerate(rows):
        name = r.get(video_col, "")
        if name:
            r["_row"] = i
            by_stem[Path(name.replace("\\", "/")).stem] = r
    pick = lambda keys: next((low[k] for k in keys if k in low), None)   # noqa: E731
    return {"path": p.name, "header": header, "video_col": video_col, "by_stem": by_stem,
            "by_stem_lower": {k.lower(): r for k, r in by_stem.items()}, "n_rows": len(by_stem),
            "id_col": pick(_ID_KEYS), "sex_col": pick(_SEX_KEYS), "genotype_col": pick(("genotype",)),
            "day_col": pick(("day",)), "used": set()}


def _append_columns(csv_path: Path, key_col: str, extra: dict, names: list) -> int:
    """Add columns to a CSV written by a stage, as text (existing cells are not re-formatted).

    ``extra``: key value -> {column name: text}. Returns the number of rows that got a value.
    """
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)
    if not header or key_col not in header:
        return 0
    k = header.index(key_col)
    n = 0
    out_rows = []
    for r in rows:
        vals = extra.get(r[k] if k < len(r) else "", None)
        if vals is not None:
            n += 1
        out_rows.append(r + [(vals or {}).get(c, "") for c in names])
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator=os.linesep)
        w.writerow(header + names)
        w.writerows(out_rows)
    tmp.replace(csv_path)
    return n


# --------------------------------------------------------------------------------------------
# the pipeline
# --------------------------------------------------------------------------------------------

class GaitRun(TmazeRun):
    """`stride gait`. Reuses the video search, the ChArUco measurement, the pose-inference step
    and the bookkeeping of the T-maze run; nothing that needs a maze ROI is called."""

    _no_pose_model_hint = "Or put the pose files next to the videos / name their folder with --pose-dir."

    def __init__(self, settings: dict):                 # noqa: D107 - not calling TmazeRun.__init__ (ROI folders, T-maze steps)
        self.s = settings
        if not settings.get("videos"):
            raise GaitError("no videos given: pass a folder / file / pattern (`stride gait VIDEOS ...`) "
                            "or set `videos:` in the config")
        if not settings.get("output"):
            raise GaitError("no output folder given: pass --out or set `output:` in the config")
        self.out = Path(str(settings["output"])).expanduser()
        self.steps = dict(settings["steps"])
        stop = settings.get("stop_after")
        if stop and stop not in STEPS:
            raise GaitError(f"--stop-after {stop}: unknown step. Steps: {', '.join(STEPS)}")
        self.suffix = settings.get("pose_suffix") or ".slp"
        self.pose_dir = self.out / "pose"
        self.gait_dir = self.out / "gait"
        self.scale_dir = self.out / "scale"
        self.pose_dirs = [self.pose_dir] + ([Path(str(settings["pose_dir"])).expanduser()] if settings.get("pose_dir") else [])
        if settings.get("pose_dir") and not Path(str(settings["pose_dir"])).expanduser().is_dir():
            raise GaitError(f"pose_dir: folder not found: {settings['pose_dir']}")
        self.warnings: list = []
        self.stage_lines: list = []
        self.results: dict = {}
        self.skipped: dict = {}
        self.state_path = self.out / "gait_state.json"
        self.state = {}
        self.t0 = time.time()
        self.kp: dict = {"aliases": {}, "confidence_names": None, "confidence_aliases": {}, "notes": []}
        self.meta = None

    # ---- steps -----------------------------------------------------------------------------
    def step_discover(self):
        from stride.stages.decision_analysis import parse_filename
        videos = self._find_videos()
        meta = load_gait_metadata(self.s["metadata"]) if self.s.get("metadata") else None
        self.meta = meta
        rows = []
        for v in videos:
            key = parse_filename(v) or {}
            row = None
            if meta is not None:
                row = meta["by_stem"].get(v.stem) or meta["by_stem_lower"].get(v.stem.lower())
                if row is not None:
                    meta["used"].add(row["_row"])
            get = lambda col: (row.get(col, "") if (row is not None and col) else "")   # noqa: E731
            day = key.get("day", "") or (_norm_day(get(meta["day_col"])) if (meta and get(meta["day_col"])) else "")
            mouse = key.get("mouse", "") or (get(meta["id_col"]) if meta else "")
            rows.append({"video_path": str(v), "stem": v.stem, "day": day, "mouse": mouse,
                         "Gender": get(meta["sex_col"]) if meta else "",
                         "Genotype": get(meta["genotype_col"]) if meta else "",
                         "_meta": row})
        self.rows = rows
        if meta is None:
            self.line("metadata", "none given: the animal, sex and genotype columns stay empty unless the file "
                                  "name is Day<N>_<animal>_Trial<T>")
        else:
            n_hit = sum(1 for r in rows if r["_meta"] is not None)
            unused = meta["n_rows"] - len(meta["used"])
            self.line("metadata", f"{meta['path']}: {n_hit} of {len(videos)} videos have a row"
                                  + (f"; {unused} metadata row(s) match no video" if unused else ""))
            if n_hit == 0:
                self.warn(f"no video matched a row of {meta['path']} (rows are found by the file name in "
                          f"column `{meta['video_col']}`)")
        self.results["metadata"] = {"file": self.s.get("metadata"),
                                    "videos_with_row": sum(1 for r in rows if r["_meta"] is not None),
                                    "videos": len(videos)}

    def step_scale(self):
        sc = self.s["scale"]
        out_json = self.scale_dir / "px_per_cm.json"
        info = {"px_per_cm": None, "source": None}
        if sc.get("px_per_cm"):
            try:
                val = float(sc["px_per_cm"])
            except (TypeError, ValueError):
                raise GaitError(f"px_per_cm must be a number, got {sc['px_per_cm']!r}")
            if not val > 0:
                raise GaitError(f"px_per_cm must be positive, got {val}")
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
            if prev is not None and prev.get("px_per_cm") and str(prev.get("source", "")).startswith("ChArUco"):
                info = prev
                self.line("scale", f"already done: px/cm = {info['px_per_cm']} ({info['source']})")
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
                    self.state["calibrate"] = fp
                else:
                    raise GaitError(f"no scale: the ChArUco measurement gave no value ({res.get('reason')}). "
                                    f"STRIDE Gait has no maze to take the scale from: give the scale you measured "
                                    f"(--px-per-cm VALUE, or `scale: px_per_cm:` in the config), or record with a "
                                    f"ChArUco board in view and describe it under `scale: charuco:`.")
        else:
            raise GaitError("no scale: `calibrate` is off and no px/cm was given. Give the scale you measured "
                            "(--px-per-cm VALUE, or `scale: px_per_cm:` in the config) or switch `calibrate` on "
                            "to measure it from a ChArUco board in view.")
        self.scale = info
        self.scale_dir.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        self.results["scale"] = {k: info.get(k) for k in ("px_per_cm", "source")}

    def step_pose(self):
        step = "pose_inference"
        need = [v for v in self.videos if not self.pose_of(v).exists()]
        if need and self.on(step) and not self.s["pose_model"]:
            # no model given: pose inference cannot run; work with the pose files that exist
            have = len(self.videos) - len(need)
            if have == 0:
                raise GaitError(f"none of the {len(self.videos)} videos has a pose file (<video name>{self.suffix} "
                                f"next to the video" + (f" or in {self.s['pose_dir']}" if self.s.get("pose_dir") else "")
                                + "), and no pose model was given. Give the pose files (--pose-dir, --pose-suffix) "
                                  "or your trained pose model (--pose-model, once or twice).")
            self.line(step, f"no pose model given: {have} videos have a pose file, {len(need)} do not and are left out")
            for v in need:
                self.skip(v, f"no pose file (<video name>{self.suffix}) and no pose model given")
            self.results[step] = {"existing": have, "done": 0, "failed": 0}
            return
        super().step_pose()

    def step_keypoints(self):
        first = next((v for v in self.videos if self.pose_of(v).exists()), None)
        if first is None:
            raise GaitError("no video has a pose file, so there is nothing to analyse")
        try:
            nodes = read_pose_node_names(self.pose_of(first), first)
        except GaitError:
            raise
        except Exception as e:
            raise GaitError(f"the pose file {self.pose_of(first)} could not be read: {type(e).__name__}: {e}")
        self.kp = resolve_keypoints(nodes, self.s.get("keypoints") or {})
        mapped = {k: v for k, v in self.kp["aliases"].items() if v is not None}
        self.line("keypoints", f"checked on {self.pose_of(first).name} ({len(nodes)} nodes): all needed keypoints found"
                               + (f"; read as {mapped}" if mapped else ""))
        for n in self.kp["notes"]:
            self.line("keypoints", n)
        self.results["keypoints"] = {"pose_nodes": nodes, "aliases": self.kp["aliases"],
                                     "confidence_names": self.kp["confidence_names"]}

    def _params_cfg(self):
        from stride.stages.run_gait_extraction import build_params
        cfg = {"stride_detection": dict(self.s.get("stride_detection") or {}),
               "stride_filtering": dict(self.s.get("stride_filtering") or {})}
        params = build_params(cfg)
        params["DEFAULT_FPS"] = float(self.s.get("fps_default") or 120.0)
        params["PX_PER_CM_OVERRIDE"] = float(self.scale["px_per_cm"])
        if self.kp["aliases"]:
            params["KEYPOINT_ALIASES"] = dict(self.kp["aliases"])
        return params, cfg

    def step_extraction(self):
        step = "gait_extraction"
        stride_csv = self.gait_dir / "gait_per_stride.csv"
        if not self.on(step):
            self.line(step, "off" + ("" if stride_csv.exists() else " (no gait_per_stride.csv in the output folder)"))
            return
        from stride.stages.gait_extraction import run_gait_extraction

        rows = [r for r in self.rows if self.pose_of(Path(r["video_path"])).exists()]
        if not rows:
            raise GaitError("no video has a pose file, so there is nothing to analyse")
        params, cfg = self._params_cfg()
        extra_cols = self._extra_columns()
        outs = [self.gait_dir / n for n in ("gait_per_stride.csv", "gait_per_step.csv", "gait_per_video.csv")]
        fp = _fingerprint([params, extra_cols,
                           [[r["video_path"], r["day"], r["mouse"], r["Gender"], r["Genotype"],
                             {c: (r["_meta"] or {}).get(c, "") for c in extra_cols},
                             _file_sig(self.pose_of(Path(r["video_path"])))] for r in rows]])
        self.extraction_fp = fp
        if self._done(step, fp, outs):
            self.line(step, f"already done for the same {len(rows)} videos")
            return
        if self.state.get(step) == fp and self.state.get("no_strides") and (self.gait_dir / "gait_per_video.csv").exists():
            self.line(step, f"already done for the same {len(rows)} videos (no stride detected)")
            return
        self.gait_dir.mkdir(parents=True, exist_ok=True)
        videos_csv = self.gait_dir / "videos.csv"                      # the video list this run worked on
        with open(videos_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["video_path", "stem", "day", "mouse", "Gender", "Genotype"])
            w.writeheader()
            w.writerows([{k: r[k] for k in w.fieldnames} for r in rows])
        payload = [{k: r[k] for k in ("video_path", "stem", "day", "mouse", "Gender", "Genotype")} for r in rows]
        column_map = {"mouse_id": ["mouse"], "mouse_id_sep": "", "sex": "Gender"}
        print(f"[{step}] stride detection on {len(rows)} videos", flush=True)
        summ = run_gait_extraction(meta_csv=videos_csv, yml_dir=None, output_dir=self.gait_dir, params=params,
                                   pose_slp_suffix=self.suffix, column_map=column_map,
                                   n_workers=int(self.s["workers"]), meta_rows=payload, pose_dir=self.pose_dirs)
        for e in summ.get("error_list") or []:
            name, _, why = str(e).partition(": ")
            self.skip(name, f"stride detection failed: {why}")
        total = summ.get("total_strides") or 0
        # metadata columns: added as text to the three tables, keyed by the video name
        if extra_cols:
            extra = {r["stem"]: {c: (r["_meta"] or {}).get(c, "") for c in extra_cols} for r in rows if r["_meta"] is not None}
            for path in outs:
                if path.exists():
                    _append_columns(path, "stem", extra, extra_cols)
        # walking direction (reporting only)
        walk = summ.get("walk_axis") or {}
        n_walk = sum(int(v[0]) for v in walk.values())
        n_along = sum(int(v[1]) for v in walk.values())
        share = (n_along / n_walk) if n_walk else None
        self.results[step] = {"videos": len(rows), "processed": summ.get("ok"), "errors": summ.get("errors"),
                              "strides": total, "px_per_cm": self.scale.get("px_per_cm"),
                              "px_per_cm_source": self.scale.get("source"),
                              "walking_frames": n_walk, "walking_frames_along_image_x": n_along,
                              "share_along_image_x": None if share is None else round(share, 4)}
        if not total:
            self.line(step, f"{summ.get('ok')} videos processed, no stride detected (clips too short or no "
                            f"walking): no stride tables written")
            self.warn("no stride was found in any video")
        else:
            self.line(step, f"{summ.get('ok')} videos, {total} strides (px/cm: {self.scale.get('px_per_cm'):g}, "
                            f"{self.scale.get('source')})" + (f"; {summ.get('errors')} video(s) failed" if summ.get("errors") else ""))
        if share is not None:
            self.line("direction", f"{100 * share:.0f} % of the walking frames have the body axis within 45 degrees "
                                   f"of the image x-axis ({n_along} of {n_walk} frames)")
            if share < 0.5:
                off = sorted((k for k, v in walk.items() if v[0] and (v[1] / v[0]) < 0.5))
                self.warn(f"only {100 * share:.0f} % of the walking frames have the body axis within 45 degrees of "
                          f"the image x-axis ({len(off)} of {len(walk)} videos mostly walk across it, e.g. "
                          f"{off[0] if off else '?'}). stride_length_cm, step_length_cm and the forward-direction "
                          f"filter assume the animal walks left-right in the image; rotate the videos or the "
                          f"camera so the runway is horizontal.")
        try:   # the stage's own warnings (frame rate, scale) go into the summary too
            man = yaml.safe_load((self.gait_dir / "gait_run_manifest.yaml").read_text(encoding="utf-8")) or {}
            for w in (man.get("results") or {}).get("warnings") or []:
                self.warnings.append(f"gait: {w}")
        except Exception:
            pass
        self.state[step] = fp
        self.state["no_strides"] = not total

    def _extra_columns(self) -> list:
        """Metadata columns added to the tables: every column except the video name, skipping
        names the stride table already has."""
        if self.meta is None:
            return []
        from stride.stages.gait_extraction import STEP_HEADER, STRIDE_HEADER
        taken = {c.lower() for c in STRIDE_HEADER + STEP_HEADER}
        return [c for c in self.meta["header"] if c != self.meta["video_col"] and c.strip().lower() not in taken]

    def step_confidence(self):
        step = "keypoint_confidence"
        conf = self.gait_dir / "pose_keypoint_confidence_by_video.csv"
        if not self.on(step):
            self.line(step, "off" + ("" if conf.exists() else " (no confidence table: the confidence filter is not applied)"))
            return
        from stride.stages.gait_extraction import compute_keypoint_confidence
        vids = [v for v in self.videos if self.pose_of(v).exists()]
        fp = _fingerprint([[_file_sig(self.pose_of(v)) for v in vids], self.kp["confidence_names"], self.kp["confidence_aliases"]])
        if self._done(step, fp, [conf]):
            self.line(step, f"already done for the same {len(vids)} videos")
            return
        self.gait_dir.mkdir(parents=True, exist_ok=True)
        compute_keypoint_confidence(video_dir=None, pose_slp_suffix=self.suffix, output_path=conf,
                                    videos=vids, pose_dir=self.pose_dirs,
                                    keypoint_names=self.kp["confidence_names"],
                                    aliases=self.kp["confidence_aliases"] or None)
        self.line(step, f"{len(vids)} videos -> {conf.name}")
        self.results[step] = {"videos": len(vids)}
        self.state[step] = fp

    def step_filter(self):
        step = "stride_filtering"
        stride_csv = self.gait_dir / "gait_per_stride.csv"
        filtered = self.gait_dir / "gait_per_stride_filtered.csv"
        conf = self.gait_dir / "pose_keypoint_confidence_by_video.csv"
        if not self.on(step):
            self.line(step, "off")
            return
        if not stride_csv.exists():
            self.line(step, f"left out: {stride_csv.name} does not exist (no stride was detected, or the "
                            f"gait_extraction step is off)")
            return
        from stride.stages.run_gait_extraction import filter_strides
        cfg = {"stride_detection": dict(self.s.get("stride_detection") or {}),
               "stride_filtering": dict(self.s.get("stride_filtering") or {})}
        fp = _fingerprint([_file_sig(stride_csv), _file_sig(conf), cfg["stride_filtering"]])
        if self._done(step, fp, [filtered]):
            self.line(step, "already done")
            return
        n = filter_strides(stride_csv, conf, filtered, cfg)
        self.line(step, f"{n} strides after the filters -> {filtered.name}")
        self.results[step] = {"filtered_strides": n}
        self.state[step] = fp

    # ---- run -------------------------------------------------------------------------------
    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        log_file = open(self.out / "gait_run.log", "a", encoding="utf-8")
        old_out = sys.stdout
        sys.stdout = _Tee(old_out, log_file)
        ok, error = True, None
        try:
            print(f"\n===== stride gait  {time.strftime('%Y-%m-%d %H:%M:%S')} =====", flush=True)
            off = [s for s in STEPS if not self.on(s)]
            print(f"output: {self.out}\nsteps off: {', '.join(off) if off else 'none'}", flush=True)
            sequence = [("discover", self.step_discover), ("calibrate", self.step_scale),
                        ("pose_inference", self.step_pose), ("keypoints", self.step_keypoints),
                        ("gait_extraction", self.step_extraction), ("keypoint_confidence", self.step_confidence),
                        ("stride_filtering", self.step_filter)]
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
        lines = ["# STRIDE Gait run", "", f"- finished: {time.strftime('%Y-%m-%d %H:%M:%S')}  ({summary['elapsed_s']} s)",
                 f"- result: {'completed' if ok else 'STOPPED: ' + str(error)}", f"- output folder: `{self.out}`",
                 f"- videos: {n_videos}",
                 "- mode: gait only. No maze ROI is used, so no trial phase and no maze region is attached to a "
                 "stride; the trial columns of gait_per_video.csv (choice, latency, reward, probes) are empty.", ""]
        lines += ["## Steps", ""] + [f"- {l}" for l in self.stage_lines] + [""]
        if getattr(self, "scale", None):
            lines += ["## Scale", "", f"- px/cm: {self.scale.get('px_per_cm')}", f"- source: {self.scale.get('source')}", ""]
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
            write_run_manifest(
                self.out, "stride_gait",
                models=list(self.s.get("pose_model") or []),
                params={k: v for k, v in self.s.items()},
                inputs={"videos": self.s.get("videos"), "n_videos": n_videos, "metadata": self.s.get("metadata")},
                outputs=[p for p in [self.gait_dir / "gait_per_stride.csv", self.gait_dir / "gait_per_step.csv",
                                     self.gait_dir / "gait_per_video.csv", self.gait_dir / "gait_per_stride_filtered.csv",
                                     self.gait_dir / "pose_keypoint_confidence_by_video.csv"] if p.exists()],
                results={"ok": ok, "error": error, "stages": self.results, "n_skipped": len(self.skipped),
                         "skipped": dict(self.skipped), "warnings": list(self.warnings)},
            )
        except Exception:
            pass
        if ok:
            print(f"\nDone in {summary['elapsed_s']} s. Tables: {self.gait_dir}; summary: {self.out / 'SUMMARY.md'}", flush=True)
        return summary


def run_gait_pipeline(settings: Optional[dict] = None, **overrides) -> dict:
    """Run the gait-only pipeline. ``settings`` as returned by :func:`load_settings` (or pass
    keyword overrides, e.g. ``run_gait_pipeline(videos=..., output=..., scale={"px_per_cm": 26.0})``).

    Returns a summary dict (``ok``, ``error``, ``stages``, ``skipped``, ``warnings``, ``output``).
    Raises nothing for expected problems: ``ok`` is False and ``error`` says why.
    """
    try:
        if settings is None:
            settings = load_settings(None, overrides)
        elif overrides:
            settings = _deep_update(copy.deepcopy(settings), overrides)
        run = GaitRun(settings)
    except TmazeError as e:
        print(f"ERROR: {e}")
        return {"ok": False, "error": str(e), "stages": {}, "skipped": {}, "warnings": []}
    return run.run()
