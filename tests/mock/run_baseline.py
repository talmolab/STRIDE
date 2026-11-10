#!/usr/bin/env python3
"""Run every user-facing STRIDE entry point against the mock dataset and record what happens.

    python tests/mock/make_mock_dataset.py --out tests/mock/_data
    python tests/mock/run_baseline.py --data tests/mock/_data --runs tests/mock/_runs

Nothing here changes STRIDE.  Each test is one command (or one library call sequence) run in its own
working directory; the exit code, runtime, the informative output lines and the files written are
stored in ``<runs>/baseline_results.json`` (+ a flat ``baseline_results.md``).  Where a command
produces numbers, they are compared with the ground truth the generator planted (``truth.csv``).

Groups
  cli      the ``stride`` console script, every subcommand
  runall   ``python run_all.py`` (all stages, single stages, --batch) with the README-style config
           and with a "full" config that carries every key the stage scripts read
  module   ``python -m stride.stages.<module>`` for every module that has a main
  safety   ``python -m stride.stages.safety_checks`` in each mode
  lib      the library path the lab's driver scripts use (run_decision_analysis, run_gait_extraction,
           compute_keypoint_confidence, filter_strides, write_run_manifest), one run per scenario
  unit     two direct function checks that explain numbers seen in the lib group

Stages 1-2 (SLEAP inference) need models and a GPU and are NOT run; only --help and the argument
plumbing without models are exercised.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
PY = sys.executable
STRIDE = str(Path(PY).parent / "stride")
KEY = re.compile(r"error|traceback|warn|fail|skip|refus|no such|not found|cannot|missing|done|complete|"
                 r"strides|trials|analy[sz]ed|converted|videos|frames|saved|stage|usage|nothing|exists|"
                 r"accuracy|removed|loaded|final|\bok\b|✓|✗|!", re.I)
RESULTS: list[dict] = []


def _has_option(args, opt):
    """True when `<args> --help` lists the option (lets one runner serve the baseline and the fixed code)."""
    try:
        out = subprocess.run([*map(str, args), "--help"], capture_output=True, text=True, timeout=120).stdout
        return opt in out
    except Exception:
        return False


AFTER_FIX = _has_option([STRIDE, "status"], "--output")
RUN_YML = (lambda core: ["--yml", core["yml_dir"]]) if _has_option([STRIDE, "run"], "--yml") else (lambda core: [])
NODES = ["snout", "mouth", "forepawR2", "forepawR1", "forepawL1", "forepawL2", "hindpawR2", "hindpawR1",
         "hindpawL2", "hindpawL1", "tailbase", "tail1", "tail2", "tail3", "tailtip"]


def tail(text: str, n=14):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    return lines[-n:]


def key_lines(text: str, n=22):
    out = [l.rstrip()[:240] for l in text.splitlines() if l.strip() and KEY.search(l)]
    return out if len(out) <= n else out[: n // 2] + ["..."] + out[-n // 2:]


def list_files(d: Path, limit=40):
    out = []
    if d and d.exists():
        for p in sorted(d.rglob("*")):
            if p.is_file():
                rows = ""
                if p.suffix == ".csv":
                    try:
                        with open(p, "rb") as f:
                            rows = max(sum(1 for _ in f) - 1, 0)
                    except OSError:
                        pass
                out.append({"file": str(p.relative_to(d)), "bytes": p.stat().st_size, "rows": rows})
    return out[:limit]


def run(tid, group, entry, scenario, expect, cmd, cwd: Path, out_dir: Path | None = None, timeout=1500,
        env=None, checks=None):
    cwd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    e = dict(os.environ)
    e.update({"PYTHONWARNINGS": "ignore", "MPLBACKEND": "Agg", "COLUMNS": "200", "NO_COLOR": "1"})
    e.update(env or {})
    try:
        p = subprocess.run(["nice", "-n", "15", *map(str, cmd)], cwd=cwd, env=e, capture_output=True,
                           text=True, timeout=timeout)
        code, so, se = p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        code, so, se = "timeout", ex.stdout or "", ex.stderr or ""
    so = "\n".join(l for l in so.splitlines() if "OpenCV: FFMPEG" not in l)
    se = "\n".join(l for l in se.splitlines() if "OpenCV: FFMPEG" not in l)
    rec = {"id": tid, "group": group, "entry": entry, "scenario": scenario, "expect": expect,
           "cmd": " ".join(str(c).replace(str(REPO), "$REPO") for c in cmd),
           "exit": code, "secs": round(time.time() - t0, 1),
           "traceback": "Traceback (most recent call last)" in (so + se),
           "key": key_lines(so + "\n" + se), "tail": tail(so + "\n" + se),
           "files": list_files(out_dir) if out_dir else ([] if cwd == REPO else list_files(cwd))}
    if checks:
        try:
            rec["checks"] = checks()
        except Exception as ex:      # a check that cannot run is itself a result
            rec["checks"] = {"check_error": f"{type(ex).__name__}: {ex}"}
    RESULTS.append(rec)
    print(f"[{tid:>5}] exit={code!s:>7} {rec['secs']:6.1f}s  {entry} :: {scenario}", flush=True)
    return rec


# ---- configs ---------------------------------------------------------------------------------
def scn_info(data: Path, name: str) -> dict:
    return json.loads((data / name / "scenario.json").read_text())


def full_config(info: dict, out_dir: Path, dst: Path, mouse_id_cols=("mouse",)) -> Path:
    """README batch + every key the later stage scripts read, pointing at <out_dir>/<batch_key>/."""
    cfg = yaml.safe_load(Path(info["config_readme"]).read_text())
    key = info["batch_key"]
    b = cfg["batches"][key]
    bo = out_dir / key
    has_sex = "Gender" in Path(info["metadata_csv"]).read_text().splitlines()[0]
    b.update({"decisions_csv": str(bo / "decisions" / "decisions.csv"),
              "events_csv": str(bo / "decisions" / "events.csv"),
              "metrics_meta_csv": str(bo / "metrics_meta.csv"),
              "stride_csv": str(bo / "gait_per_stride_filtered.csv"),
              "step_csv": str(bo / "gait_per_step.csv"),
              "mouse_id_cols": list(mouse_id_cols), "mouse_id_sep": "", "has_sex": has_sex,
              "px_per_cm": info["px_per_cm"]})
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return dst


# ---- comparison with the planted truth -------------------------------------------------------
def check_decisions(dec_dir: Path, truth_csv: Path) -> dict:
    import numpy as np
    import pandas as pd
    tr = pd.read_csv(truth_csv, dtype={"mouse": str})
    out = {"truth_clips": len(tr)}
    files = {n: dec_dir / f"{n}.csv" for n in ("decisions", "events", "metrics")}
    out["files_present"] = {n: p.exists() for n, p in files.items()}
    if not files["decisions"].exists():
        return out
    de, ev = pd.read_csv(files["decisions"]), pd.read_csv(files["events"])
    me = pd.read_csv(files["metrics"]) if files["metrics"].exists() else pd.DataFrame({"stem": []})
    out["rows"] = len(de)
    out["clips_without_a_row"] = sorted(set(tr.stem) - set(de.stem))
    m = tr.merge(de, on="stem", how="inner", suffixes=("_t", "")).merge(
        ev[["stem", "enter_seg1_frame", "enter_seg4_frame", "probes_L", "probes_R", "probe_frames_L",
            "probe_frames_R", "n_reversals", "reversal_frames"]], on="stem", how="left").merge(
        me[["stem", "latency_choice_ms", "stem_latency_ms"]] if len(me) else me, on="stem", how="left")
    ent = m[m.planted_choice.notna() & (m.planted_choice != "")]
    out["entering_clips"] = len(ent)
    out["choice_matches_planted"] = int((ent.entry_side == ent.planted_choice).sum())
    out["choice_mismatch"] = ent.loc[ent.entry_side != ent.planted_choice, "stem"].tolist()
    exp = ent.expected_correct.astype(float)
    out["correct_matches_planted"] = int((ent.correct_bin.astype(float) == exp).sum())
    out["correct_mismatch"] = [f"{r.stem}: planted {int(e)} got {r.correct_bin} (reward in table: {r.reward!r})"
                               for (_, r), e in zip(ent.iterrows(), exp) if float(r.correct_bin) != e]
    out["entry_frame_minus_planted_median"] = None if ent.empty else float(
        (ent.entry_frame - ent.planted_entry_frame.astype(float)).median())
    nod = m[m.planted_choice.isna() | (m.planted_choice == "")]
    out["no_decision_clips"] = [f"{r.stem}: ok={r.ok} reason={r.reason} correct_bin={r.correct_bin} correct_TF={r.correct_TF}"
                                for _, r in nod.iterrows()]
    pm = []
    for _, r in m.iterrows():
        got = (int(r.probes_L), int(r.probes_R), int(r.probe_frames_L), int(r.probe_frames_R))
        want = (int(r.planted_probes_L), int(r.planted_probes_R), int(r.planted_probe_frames_L), int(r.planted_probe_frames_R))
        if got[:2] != want[:2] and (r.planted_choice == r.planted_choice and r.planted_choice != ""):
            pm.append(f"{r.stem}: planted L{want[0]}/R{want[1]} ({want[2]}/{want[3]} fr) got L{got[0]}/R{got[1]} ({got[2]}/{got[3]} fr)")
    out["probe_count_mismatch"] = pm
    out["gate_frames"] = sorted({int(g) for g in m.gate_frame.dropna()})
    out["gate_reasons"] = sorted({str(g) for g in m.gate_reason.dropna()})
    fwd = m[(m.planted_withdrawals == 0) & m.planted_choice.notna() & (m.planted_choice != "")]
    out["n_reversals_on_trials_with_no_withdrawal"] = {str(k): int(v) for k, v in fwd.n_reversals.value_counts().items()}
    out["n_reversals_by_clip"] = {r.stem: [int(r.planted_withdrawals), int(r.n_reversals), int(r.reversal_frames)]
                                  for _, r in m.iterrows()} if len(m) <= 10 else "see events.csv"
    out["latency_choice_ms_nan"] = ent.loc[ent.latency_choice_ms.isna(), "stem"].tolist()
    out["stem_latency_ms_nan"] = ent.loc[ent.stem_latency_ms.isna(), "stem"].tolist()
    out["enter_seg1_frame_is_0"] = m.loc[m.enter_seg1_frame == 0, "stem"].tolist()
    out["enter_seg4_frame_is_0"] = m.loc[m.enter_seg4_frame == 0, "stem"].tolist()
    out["fps_values"] = sorted({float(x) for x in m.fps.dropna()})
    out["entry_thresholds"] = sorted({str(x) for x in m.entry_threshold.dropna()})
    return out


def check_gait(gait_dir: Path, truth_csv: Path, filtered_name="gait_per_stride_filtered.csv") -> dict:
    import numpy as np
    import pandas as pd
    tr = pd.read_csv(truth_csv, dtype={"mouse": str})
    out = {}
    raw = gait_dir / "gait_per_stride.csv"
    out["raw_exists"] = raw.exists()
    if not raw.exists():
        return out
    g = pd.read_csv(raw)
    out["raw_strides"] = len(g)
    out["clips_with_strides"] = int(g.stem.nunique())
    out["clips_without_strides"] = sorted(set(tr.stem) - set(g.stem))
    a = g.groupby("stem").agg(n=("stride_id", "size"), dur=("duration_ms", "median"),
                              speed=("stride_speed_cm_s_mean", "median"), sl=("stride_length_cm", "median"),
                              sle=("stride_length_euclid_cm", "median"), stepl=("step_length_cm", "median"),
                              stepw=("step_width_cm", "median"), duty=("limb_duty_factor_mean", "median"),
                              cmpx=("px_per_cm", "median"), body=("body_length_cm", "median"),
                              fps=("fps", "median"), neg=("stride_length_cm", lambda s: int((s < 0).sum())),
                              pawL=("hindL_ind", "first"), pawR=("hindR_ind", "first"),
                              tip_nan=("tailtip_lat_disp_norm", lambda s: int(s.isna().sum())),
                              nose_nan=("nose_lat_disp_norm", lambda s: int(s.isna().sum()))).reset_index()
    m = tr.merge(a, on="stem", how="inner", suffixes=("_t", ""))
    r3 = lambda x: None if x != x else round(float(x), 3)   # noqa: E731
    out["median_over_clips"] = {
        "strides_per_clip": r3(m.n.median()), "planted_left_landings_minus_1": r3((m.planted_left_landings - 1).median()),
        "duration_ms": r3(m.dur.median()), "planted_period_ms": r3(m.planted_stride_period_ms.median()),
        "duration_minus_period_in_frames": r3(((m.dur - m.planted_stride_period_ms) * m.fps / 1000).median()),
        "speed_cm_s": r3(m.speed.median()), "planted_speed_cm_s": r3(m.planted_speed_cm_s.median()),
        "stride_length_cm": r3(m.sl.median()), "stride_length_euclid_cm": r3(m.sle.median()),
        "planted_ground_stride_length_cm": r3(m.planted_stride_length_cm.median()),
        "planted_body_frame_excursion_cm": r3(m.planted_body_frame_excursion_cm.median()),
        "step_length_cm": r3(m.stepl.median()), "step_width_cm": r3(m.stepw.median()),
        "duty": r3(m.duty.median()), "planted_duty": r3(m.planted_duty.median()),
        "px_per_cm_column": r3(m.cmpx.median()), "one_over_true_px_per_cm": r3((1 / m.px_per_cm).median()),
        "body_length_cm": r3(m.body.median()), "planted_body_length_cm": r3(m.body_length_cm.median()),
        "fps_column": r3(m.fps.median()), "true_fps": r3(m.fps_t.median()) if "fps_t" in m else None,
    }
    if len(m) <= 8:
        out["per_clip"] = {r.stem: {"n": int(r.n), "dur_ms": r3(r.dur), "speed": r3(r.speed), "stride_len": r3(r.sl),
                                    "stride_len_euclid": r3(r.sle), "step_len": r3(r.stepl),
                                    "paws_picked_L/R": f"{NODES[int(r.pawL)]}/{NODES[int(r.pawR)]}", "px_per_cm_col": r3(r.cmpx),
                                    "implied_px_per_cm": r3(1 / r.cmpx) if r.cmpx else None, "body_cm": r3(r.body),
                                    "fps": r3(r.fps), "negative_stride_len": int(r.neg), "tailtip_nan": int(r.tip_nan),
                                    "nose_nan": int(r.nose_nan), "note": r.note if isinstance(r.note, str) else ""}
                           for _, r in m.iterrows()}
    # where the strides with a negative stride_length_cm are (before or after the mouse turned into the arm)
    ent = dict(zip(tr.stem, pd.to_numeric(tr.planted_entry_frame, errors="coerce")))
    neg = g[g.stride_length_cm < 0]
    after = int(sum(1 for _, r in neg.iterrows() if ent.get(r.stem) == ent.get(r.stem) and r.frame_start >= ent[r.stem] - 0.6 * r.fps))
    out["negative_stride_length"] = {"n": len(neg), "of_which_in_the_turn_or_arm": after,
                                     "planted_backward_walking_clips": int((tr.planted_withdrawals > 0).sum())}
    f = gait_dir / filtered_name
    out["filtered_exists"] = f.exists()
    if f.exists():
        fd = pd.read_csv(f)
        out["filtered_strides"] = len(fd)
        out["filtered_has_stem_column"] = "stem" in fd.columns
        out["filtered_columns_lost"] = sorted(set(g.columns) - set(fd.columns))
        if len(fd) and "video_path" in fd.columns:
            fd["stem"] = fd.video_path.map(lambda p: Path(p).stem)
            n = fd.groupby("stem").size()
            mm = tr.merge(n.rename("n_f").reset_index(), on="stem", how="left")
            out["filtered_per_clip_median"] = r3(mm.n_f.median())
            out["planted_straight_strides_median"] = r3(mm.planted_straight_strides.median())
            out["clips_with_no_filtered_stride"] = mm.loc[mm.n_f.isna(), "stem"].tolist()
    return out


# ---- the library path (run in a subprocess per scenario: `run_baseline.py --lib <scn> <out>`) ----
def lib_run(scn_dir: Path, out: Path, workers: int):
    import pandas as pd
    info = json.loads((scn_dir / "scenario.json").read_text())
    cfg = yaml.safe_load(Path(info["config_readme"]).read_text())
    batch = cfg["batches"][info["batch_key"]]
    vdir = Path(info["video_dir"])
    yml_dir = None if info["roi_mode"] == "colocated" else Path(info["yml_dir"])
    from stride.run_manifest import write_run_manifest
    from stride.stages.decision_analysis import run_decision_analysis
    from stride.stages.gait_extraction import compute_keypoint_confidence, run_gait_extraction
    from stride.stages.run_gait_extraction import build_params, filter_strides
    summary = {}
    print("== decisions ==", flush=True)
    # small scenarios run sequentially (that path prints a status per video); the core batch runs in
    # parallel, as the lab drivers do
    dec_kw = {}
    if batch.get("pose_slp_suffix") and "pose_slp_suffix" in run_decision_analysis.__code__.co_varnames:
        dec_kw["pose_slp_suffix"] = batch["pose_slp_suffix"]     # argument added after the baseline
    res = run_decision_analysis(video_dir=vdir, yml_dir=yml_dir, meta_csv=Path(info["meta_trials_csv"]),
                                output_dir=out / "decisions", n_workers=workers if info["n_clips"] > 8 else 1,
                                **dec_kw)
    summary["decisions"] = res
    print("RESULT decisions", res, flush=True)
    metrics = out / "decisions" / "metrics.csv"
    if not metrics.exists():
        print("ERROR-LIB no metrics.csv: the decision stage wrote nothing; gait is not run")
        print("LIBSUMMARY " + json.dumps(summary, default=str)); return
    # what the lab drivers do: add Gender + Genotype to metrics.csv -> metrics_meta.csv
    me = pd.read_csv(metrics, dtype={"mouse": str})
    md = pd.read_csv(info["metadata_csv"], dtype=str).rename(columns={"Mouse": "mouse"})
    me = me.merge(md, on="mouse", how="left")
    for col in ("Gender", "Genotype"):
        if col not in me.columns:
            me[col] = ""
    (out / "gait").mkdir(parents=True, exist_ok=True)
    me.to_csv(out / "gait" / "metrics_meta.csv", index=False)
    params = build_params(cfg)
    params["DEFAULT_FPS"] = float(batch.get("fps", 120))
    if batch.get("px_per_cm_override"):
        params["PX_PER_CM_OVERRIDE"] = float(batch["px_per_cm_override"])
    suffix = batch.get("pose_slp_suffix", ".slp")
    print("== gait extraction ==", flush=True)
    r = run_gait_extraction(meta_csv=out / "gait" / "metrics_meta.csv", yml_dir=yml_dir, output_dir=out / "gait",
                            params=params, pose_slp_suffix=suffix,
                            column_map={"mouse_id": ["mouse"], "mouse_id_sep": "", "sex": "Gender"}, n_workers=workers)
    summary["gait"] = {k: v for k, v in r.items() if not k.endswith("_csv")}
    print("RESULT gait", summary["gait"], flush=True)
    conf = out / "gait" / "pose_keypoint_confidence_by_video.csv"
    compute_keypoint_confidence(video_dir=vdir, pose_slp_suffix=suffix, output_path=conf)
    print("== stride filtering ==", flush=True)
    stride_csv = out / "gait" / "gait_per_stride.csv"
    if stride_csv.exists():
        try:
            summary["filtered"] = filter_strides(stride_csv, conf, out / "gait" / "gait_per_stride_filtered.csv", cfg)
        except Exception as ex:
            summary["filter_error"] = f"{type(ex).__name__}: {ex}"
            print("ERROR-LIB filter_strides raised", summary["filter_error"])
    else:
        print("ERROR-LIB no gait_per_stride.csv was written (0 strides)")
    mp = write_run_manifest(out / "gait", "gait", params=params, inputs={"video_dir": vdir},
                            outputs=[stride_csv], results=summary.get("gait"))
    summary["manifest"] = str(mp) if mp else None
    print("LIBSUMMARY " + json.dumps(summary, default=str))


def unit_checks():
    """Two direct function checks (no data files)."""
    import numpy as np
    from stride.stages.decision_analysis import interpolate_gaps
    from stride.stages.gait_extraction import normalize_to_egocentric
    res = {}
    # 1. egocentric transform: tail base at the origin, snout 7 cm ahead, a point 2 cm ahead and 1 cm to one side
    ego = {}
    for deg in (0, 15, 45, 90, 180):
        a = np.radians(deg)
        fwd, lat = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
        x = np.stack([np.zeros(2), 7 * fwd, 2 * fwd + 1 * lat])[None]
        e = normalize_to_egocentric(x.copy(), ctr_ind=0, fwd_ind=1)[0]
        ego[f"heading_{deg}deg"] = {"snout_expected": [7.0, 0.0], "snout_got": [round(float(v), 3) for v in e[1]],
                                    "point_expected": [2.0, 1.0], "point_got": [round(float(v), 3) for v in e[2]]}
    res["normalize_to_egocentric"] = ego
    # 2. gap filling: how long a gap does max_gap=7 bridge?
    gaps = {}
    for n in (1, 7, 8, 14, 15, 20):
        t = np.arange(60, dtype=float)
        tr = np.stack([t, t], 1)
        tr[20:20 + n] = np.nan
        out = interpolate_gaps(tr, max_gap=7)
        gaps[f"gap_{n}_frames"] = {"frames_still_missing": int(np.isnan(out[:, 0]).sum())}
    res["interpolate_gaps_max_gap_7"] = gaps
    print("UNIT " + json.dumps(res))


def parse_tag(rec, tag):
    for l in list(reversed(rec["tail"])) + list(reversed(rec["key"])):      # tail lines are not truncated
        if l.startswith(tag + " "):
            try:
                return json.loads(l[len(tag) + 1:])
            except json.JSONDecodeError:
                return {"unparsed": l[:200]}
    return None


# ---- the test list ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(Path(__file__).parent / "_data"))
    ap.add_argument("--runs", default=str(Path(__file__).parent / "_runs"))
    ap.add_argument("--groups", default="cli,runall,module,safety,lib,unit")
    ap.add_argument("--workers", type=int, default=4, help="max worker processes any stage may start")
    ap.add_argument("--lib", nargs=2, metavar=("SCENARIO_DIR", "OUT"), help=argparse.SUPPRESS)
    ap.add_argument("--unit", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.lib:
        return lib_run(Path(a.lib[0]), Path(a.lib[1]), min(a.workers, 4))
    if a.unit:
        return unit_checks()

    data, runs = Path(a.data).resolve(), Path(a.runs).resolve()
    if runs.exists():
        shutil.rmtree(runs)
    runs.mkdir(parents=True)
    groups = set(a.groups.split(","))
    names = [s["name"] for s in json.loads((data / "scenarios.json").read_text())]
    info = {n: scn_info(data, n) for n in names}
    core = info["core"]
    W = str(min(a.workers, 4))
    me = str(Path(__file__).resolve())

    # ---------------- lib: one run per scenario (also feeds the cli / safety tests) ----------------
    lib_out = {}
    if "lib" in groups:
        for n in names:
            o = runs / "lib" / n
            lib_out[n] = o
            rec = run(f"L-{n}", "lib", "library path (run_decision_analysis + run_gait_extraction + filter_strides)",
                      n, info[n]["about"], [PY, me, "--lib", data / n, o, "--workers", W], runs / "lib" / "_cwd", o,
                      checks=lambda n=n, o=o: {"decisions": check_decisions(o / "decisions", data / n / "truth.csv"),
                                                "gait": check_gait(o / "gait", data / n / "truth.csv")})
            rec["lib_summary"] = parse_tag(rec, "LIBSUMMARY")
    if "unit" in groups:
        rec = run("U-1", "unit", "normalize_to_egocentric / interpolate_gaps", "-", "ego frame puts the snout on +x; "
                  "gaps up to 7 frames are filled", [PY, me, "--unit"], runs / "unit")
        rec["unit"] = parse_tag(rec, "UNIT")

    # ---------------- cli ----------------
    if "cli" in groups:
        c = runs / "cli"
        run("C-01", "cli", "stride --help", "-", "lists the subcommands", [STRIDE, "--help"], c / "help")
        run("C-02", "cli", "stride --version", "-", "prints the version", [STRIDE, "--version"], c / "help")
        for i, sub in enumerate(["run", "pose-inference", "check-distortion", "roi-inference", "convert-roi",
                                 "analyze-decisions", "analyze-gait", "count-frames", "status"]):
            run(f"C-1{i}", "cli", f"stride {sub} --help", "-", "prints usage", [STRIDE, sub, "--help"], c / "help")
        run("C-20", "cli", "stride run", "core", "full pipeline on videos that already have pose + ROI: decisions written",
            [STRIDE, "run", "-i", core["video_dir"], "-o", c / "run" / "out", "-m", core["meta_trials_csv"],
             "--skip-distortion", "--skip-pose", "--skip-roi", "-w", W, *RUN_YML(core)], c / "run", c / "run" / "out")
        run("C-21", "cli", "stride run (no --skip flags)", "behaviour", "starts with the distortion check; inference needs models",
            [STRIDE, "run", "-i", info["behaviour"]["video_dir"], "-o", c / "run2" / "out", "--device", "cpu", "-w", W],
            c / "run2", c / "run2" / "out")
        run("C-30", "cli", "stride analyze-decisions (README command)", "core", "decisions for 52 trials",
            [STRIDE, "analyze-decisions", "--videos", core["video_dir"], "--yml", core["yml_dir"], "--meta",
             core["meta_trials_csv"]], c / "analyze_decisions")
        run("C-31", "cli", "stride analyze-decisions -o out/decisions.csv -w 4", "core", "decisions for 52 trials",
            [STRIDE, "analyze-decisions", "--videos", core["video_dir"], "--yml", core["yml_dir"], "--meta",
             core["meta_trials_csv"], "-o", c / "analyze_decisions2" / "out" / "decisions.csv", "-w", W],
            c / "analyze_decisions2")
        lp = info["layout_per_day_colocated"]
        run("C-32", "cli", "stride analyze-decisions (per-day folders, co-located ROI)", "layout_per_day_colocated",
            "decisions for 4 trials", [STRIDE, "analyze-decisions", "--videos", lp["video_dir"], "--yml", lp["video_dir"],
                                       "--meta", lp["meta_trials_csv"]], c / "analyze_decisions3")
        if "core" in lib_out and (lib_out["core"] / "gait" / "gait_per_stride.csv").exists():
            g = lib_out["core"] / "gait"

            def cmp_filter(csv_path):
                import pandas as pd
                o = {"cli_filtered_exists": Path(csv_path).exists()}
                if Path(csv_path).exists():
                    o["cli_filtered_strides"] = len(pd.read_csv(csv_path))
                    ref = g / "gait_per_stride_filtered.csv"
                    o["config_path_filtered_strides"] = len(pd.read_csv(ref)) if ref.exists() else None
                    o["raw_strides"] = len(pd.read_csv(g / "gait_per_stride.csv"))
                return o
            run("C-40", "cli", "stride analyze-gait -i strides -c confidence", "core",
                "same filtered strides as the config-driven stage 6", [STRIDE, "analyze-gait", "-i", g / "gait_per_stride.csv",
                "-c", g / "pose_keypoint_confidence_by_video.csv", "-o", c / "analyze_gait" / "filtered.csv"],
                c / "analyze_gait", checks=lambda: cmp_filter(c / "analyze_gait" / "filtered.csv"))
            run("C-41", "cli", "stride analyze-gait -i strides (no confidence file)", "core", "filtered strides",
                [STRIDE, "analyze-gait", "-i", g / "gait_per_stride.csv"], c / "analyze_gait2",
                checks=lambda: cmp_filter(c / "analyze_gait2" / "gait_filtered.csv"))
        run("C-50", "cli", "stride check-distortion", "behaviour", "a result per video (no board in a blank video)",
            [STRIDE, "check-distortion", "-i", info["behaviour"]["video_dir"], "-o", c / "distortion" / "dist.csv", "-n", "3"],
            c / "distortion")
        run("C-51", "cli", "stride check-distortion", "layout_per_day_colocated", "finds the 4 .MP4 files in the day folders",
            [STRIDE, "check-distortion", "-i", lp["video_dir"], "-n", "3"], c / "distortion2")

        def cmp_roi(yml_out):
            import numpy as np
            o = {"yml_written": sorted(p.name for p in Path(yml_out).glob("*.yml"))}
            worst = 0.0
            for p in Path(yml_out).glob("*.yml"):
                ref = Path(core["yml_dir"]) / p.name
                if ref.exists():
                    A = {r["name"]: np.array(r["coordinates"]) for r in yaml.safe_load(p.read_text())["rois"]}
                    B = {r["name"]: np.array(r["coordinates"]) for r in yaml.safe_load(ref.read_text())["rois"]}
                    worst = max(worst, max(float(np.abs(A[k] - B[k]).max()) for k in B))
            o["max_abs_diff_px_vs_generator_polygons"] = round(worst, 3)
            return o
        run("C-60", "cli", "stride convert-roi", "core", "3 ROI .slp -> 2 yml (one .slp lacks a keypoint)",
            [STRIDE, "convert-roi", "-i", core["roi_slp_dir"], "-o", c / "convert_roi" / "yml"], c / "convert_roi",
            checks=lambda: cmp_roi(c / "convert_roi" / "yml"))
        run("C-70", "cli", "stride count-frames", "core", "52 videos, 52 x 720 or 1020 frames",
            [STRIDE, "count-frames", "-i", core["video_dir"]], c / "count")
        run("C-71", "cli", "stride count-frames", "layout_per_day_colocated", "4 videos (.MP4, in day folders)",
            [STRIDE, "count-frames", "-i", lp["video_dir"]], c / "count")
        run("C-72", "cli", "stride count-frames", "pose_foreign_video_path", "2 videos",
            [STRIDE, "count-frames", "-i", info["pose_foreign_video_path"]["video_dir"]], c / "count")
        run("C-80", "cli", "stride status", "-", "shows pipeline progress", [STRIDE, "status"], c / "status")
        if AFTER_FIX:   # `stride status` now lists the run manifests under an output folder
            run("C-81", "cli", "stride status -o <run output>", "core", "shows pipeline progress of the run above",
                [STRIDE, "status", "-o", c / "run" / "out"], c / "status")
        else:
            run("C-81", "cli", "stride status -c <run output>/progress.log", "core", "shows pipeline progress of the run above",
                [STRIDE, "status", "-c", c / "run" / "out" / "progress.log"], c / "status")
        dummy = c / "infer" / "dummy_model"
        dummy.mkdir(parents=True, exist_ok=True)
        bv = info["behaviour"]["video_dir"]
        run("C-90", "cli", "stride pose-inference (no --model)", "behaviour", "says which models are needed",
            [STRIDE, "pose-inference", "-i", bv, "--device", "cpu"], c / "infer")
        run("C-91", "cli", "stride pose-inference --model <dir> (sleap-nn not installed)", "pose_low_confidence",
            "says sleap-nn is required", [STRIDE, "pose-inference", "-i", info["pose_low_confidence"]["video_dir"], "-m", dummy,
                                          "--device", "cpu", "-o", c / "infer" / "pose_out"], c / "infer")
        run("C-92", "cli", "stride roi-inference (no --model)", "behaviour", "says which models are needed",
            [STRIDE, "roi-inference", "-i", bv, "-o", c / "infer" / "roi_out", "--device", "cpu"], c / "infer")
        run("C-93", "cli", "stride roi-inference --model <dir> (sleap-nn not installed)", "behaviour", "says sleap-nn is required",
            [STRIDE, "roi-inference", "-i", bv, "-o", c / "infer" / "roi_out2", "-m", dummy, "--device", "cpu"], c / "infer")

    # ---------------- run_all.py ----------------
    if "runall" in groups:
        r = runs / "runall"
        ra = [PY, REPO / "run_all.py"]
        run("R-01", "runall", "python run_all.py --help", "-", "prints usage", ra + ["--help"], REPO)
        o = r / "readme_all"
        run("R-10", "runall", "run_all.py --config <README-style config> (README quick start)", "core",
            "all stages run; decisions, strides, LMM table, classifier figures", ra + ["--config", core["config_readme"],
            "--output", o / "out"], REPO, o / "out",
            checks=lambda: {"decisions": check_decisions(o / "out" / core["batch_key"] / "decisions", data / "core" / "truth.csv"),
                            "gait": check_gait(o / "out" / core["batch_key"], data / "core" / "truth.csv")})
        o2 = r / "readme_batch"
        run("R-11", "runall", "run_all.py --config <README-style config> --batch <key> (README 'adding a new batch')", "core",
            "all stages run for that batch", ra + ["--config", core["config_readme"], "--output", o2 / "out", "--batch",
            core["batch_key"]], REPO, o2 / "out")
        o3 = r / "readme_stages"
        for i, st in enumerate(["1-2", "3", "4-6", "7-9", "10"]):
            run(f"R-2{i}", "runall", f"run_all.py --stages {st} (README-style config, same output dir, in order)", "core",
                "that stage runs on the previous stage's output", ra + ["--config", core["config_readme"], "--output", o3 / "out",
                "--stages", st], REPO, o3 / "out")
        # relative --config from another directory (the README runs from the repo root)
        rel = r / "relative_cfg"
        rel.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(core["config_readme"], rel / "config.yaml")
        run("R-30", "runall", "python <repo>/run_all.py --config config.yaml --stages 3 (run from another folder)", "core",
            "relative paths resolve from the current folder", ra + ["--config", "config.yaml", "--output", "out", "--stages", "3"],
            rel, rel / "out")
        run("R-31", "runall", "run_all.py --stages 5 (unknown stage)", "core", "clear error", ra + ["--config", core["config_readme"],
            "--output", r / "bad" / "out", "--stages", "5"], REPO)
        # the config as shipped, on a machine that does not have the lab's file share
        sh = r / "shipped"
        sh.mkdir(parents=True, exist_ok=True)
        (sh / "config.shipped_no_lab_share.yaml").write_text(
            (REPO / "config.yaml").read_text().replace("/home/exx/vast/leo", "/nonexistent/lab_share"))
        run("R-40", "runall", "run_all.py with the shipped config.yaml (lab paths do not exist on this machine)", "-",
            "tells the user to edit the config", ra + ["--config", sh / "config.shipped_no_lab_share.yaml", "--output", sh / "out"],
            REPO, sh / "out")
        # full config: every key the stage scripts read
        full_set = ["core", "layout_per_day_colocated", "layout_custom_pose_suffix", "ids_string", "meta_no_sex",
                    "meta_other_genotype_labels", "meta_missing_rows", "meta_day_as_number", "calib_roi_problems",
                    "calib_override", "camera_fps30", "pose_before_rename", "pose_low_confidence",
                    "batch_one_genotype", "batch_wt_tau", "batch_too_few_mice"]
        for n in full_set:
            if n not in info:
                continue
            o = r / "full" / n
            cfg = full_config(info[n], o / "out", o / "config.full.yaml")
            key = info[n]["batch_key"]
            run(f"R-5:{n}", "runall", "run_all.py --config <full config> (all stages)", n,
                "all stages run end to end", ra + ["--config", cfg, "--output", o / "out"], REPO, o / "out",
                checks=lambda o=o, n=n, key=key: {
                    "decisions": check_decisions(o / "out" / key / "decisions", data / n / "truth.csv"),
                    "gait": check_gait(o / "out" / key, data / n / "truth.csv"),
                    "lmm": lmm_summary(o / "out"), "classifier": [p.name for p in (o / "out").glob("regression_*")]})
        # stage 10 with the other possible id-column setting
        o = r / "full_stage10_mouse_id"
        cfg = full_config(core, r / "full" / "core" / "out", o / "config.full.yaml", mouse_id_cols=("mouse_id",))
        run("R-60", "runall", "run_all.py --stages 10 (full config, mouse_id_cols: [mouse_id])", "core",
            "classifier runs on the pipeline's own outputs", ra + ["--config", cfg, "--output", r / "full" / "core" / "out",
            "--stages", "10"], REPO)

    # ---------------- stage modules ----------------
    if "module" in groups:
        m = runs / "module"
        mods = ["run_inference", "run_tmaze", "run_gait_extraction", "gait_lmm", "classification", "safety_checks"]
        for i, mod in enumerate(mods):
            run(f"M-0{i}", "module", f"python -m stride.stages.{mod} --help", "-", "prints usage",
                [PY, "-m", f"stride.stages.{mod}", "--help"], m / "help")
        for i, mod in enumerate(mods[:5]):
            run(f"M-1{i}", "module", f"python -m stride.stages.{mod} (no arguments)", "-", "uses ./config.yaml or says what is needed",
                [PY, "-m", f"stride.stages.{mod}"], m / f"noargs_{mod}")
        # README "Individual stages": --config only
        rd = m / "readme_individual"
        rd.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(core["config_readme"], rd / "config.yaml")
        for i, mod in enumerate(["run_gait_extraction", "gait_lmm", "classification"]):
            run(f"M-2{i}", "module", f"python -m stride.stages.{mod} --config config.yaml (README 'Individual stages')", "core",
                "runs that stage", [PY, "-m", f"stride.stages.{mod}", "--config", "config.yaml"], rd, rd)
        # every module in order with a full config
        o = m / "full_core"
        cfg = full_config(core, o / "out", o / "config.full.yaml")
        key = core["batch_key"]
        seq = [("run_inference", ["--batch", key]), ("run_tmaze", ["--batch", key]), ("run_gait_extraction", ["--batch", key]),
               ("gait_lmm", []), ("gait_lmm", ["--trial-phase", "all"]), ("gait_lmm", ["--batch", key]), ("classification", [])]
        for i, (mod, extra) in enumerate(seq):
            run(f"M-3{i}", "module", f"python -m stride.stages.{mod} --config <full> --output <out> {' '.join(extra)}".strip(),
                "core", "runs that stage on the previous stage's output",
                [PY, "-m", f"stride.stages.{mod}", "--config", cfg, "--output", o / "out", *extra], o, o / "out",
                checks=(lambda: {"lmm": lmm_summary(o / "out")}) if mod == "gait_lmm" else None)
        # a second run of stage 3 / 4-6 into the same output (resume behaviour)
        run("M-40", "module", "python -m stride.stages.run_gait_extraction (second run, same output)", "core",
            "skips finished work or redoes it", [PY, "-m", "stride.stages.run_gait_extraction", "--config", cfg, "--output", o / "out"],
            o, o / "out")

    # ---------------- safety checks ----------------
    if "safety" in groups and lib_out:
        s = runs / "safety"
        sc = [PY, "-m", "stride.stages.safety_checks"]
        gc = lib_out["core"] / "gait" / "gait_per_stride.csv"
        run("S-01", "safety", "safety_checks --gait <strides>", "core", "OK (px/cm 16, body 7 cm, 120 fps)", sc + ["--gait", gc], s)
        run("S-02", "safety", "safety_checks --gait <strides> --expect-pxcm 16 --strict", "core", "OK, exit 0",
            sc + ["--gait", gc, "--expect-pxcm", "16", "--strict"], s)
        run("S-03", "safety", "safety_checks --gait <strides> --expect-pxcm 26", "core", "FAIL (wrong calibration), exit 2",
            sc + ["--gait", gc, "--expect-pxcm", "26"], s)
        run("S-04", "safety", "safety_checks --gait <strides>", "calib_roi_problems", "FAIL: clips with no / broken ROI have px/cm = 1.0",
            sc + ["--gait", lib_out["calib_roi_problems"] / "gait" / "gait_per_stride.csv"], s)
        run("S-05", "safety", "safety_checks --gait <strides>", "calib_override", "WARN: pinned override without --expect-pxcm",
            sc + ["--gait", lib_out["calib_override"] / "gait" / "gait_per_stride.csv", "--strict"], s)
        mixed = s / "mixed_fps.csv"
        try:
            import pandas as pd
            s.mkdir(parents=True, exist_ok=True)
            pd.concat([pd.read_csv(lib_out[n] / "gait" / "gait_per_stride.csv") for n in ("camera_fps100", "behaviour")]).to_csv(mixed, index=False)
            run("S-06", "safety", "safety_checks --gait <100 fps + 120 fps strides>", "camera_fps100 + behaviour", "FAIL: mixed fps",
                sc + ["--gait", mixed], s)
        except Exception as ex:
            print("  (mixed-fps safety test not built:", ex, ")")
        run("S-07", "safety", "safety_checks --gait <filtered strides>", "camera_lowres", "FAIL or OK on a 10 px/cm camera",
            sc + ["--gait", lib_out["camera_lowres"] / "gait" / "gait_per_stride.csv"], s)
        lp = info["layout_per_day_colocated"]
        run("S-10", "safety", "safety_checks --roi-coverage <videos>", "layout_per_day_colocated", "OK: 4 videos, 4 .rois.yml",
            sc + ["--roi-coverage", lp["video_dir"], "--expect-count", "4"], s)
        run("S-11", "safety", "safety_checks --roi-coverage <videos>" + (" --yml-dir <yml folder>" if AFTER_FIX else ""), "core",
            "OK: 52 videos have an ROI yml (in the yml folder)",
            sc + ["--roi-coverage", core["video_dir"]] + (["--yml-dir", core["yml_dir"]] if AFTER_FIX else []), s)
        run("S-12", "safety", "safety_checks --roi-coverage <videos> --expect-count 5", "layout_per_day_colocated", "FAIL: count mismatch",
            sc + ["--roi-coverage", lp["video_dir"], "--expect-count", "5"], s)
        crp = info["calib_roi_problems"]
        run("S-13", "safety", "safety_checks --roi-coverage <videos>" + (" --yml-dir <yml folder>" if AFTER_FIX else ""),
            "calib_roi_problems", "FAIL: one clip has no ROI file",
            sc + ["--roi-coverage", crp["video_dir"]]
            + (["--yml-dir", crp["yml_dir"]] if AFTER_FIX and crp["roi_mode"] != "colocated" else []), s)
        led = s / "ledger_dir"
        led.mkdir(parents=True, exist_ok=True)
        run("S-20", "safety", "safety_checks --require rename,roi --vid <dir> (nothing stamped)", "-", "FAIL, exit 2",
            sc + ["--require", "rename,roi", "--vid", led], s)
        run("S-21", "safety", "safety_checks --stamp rename --vid <dir>", "-", "stamps the gate", sc + ["--stamp", "rename", "--vid", led], s)
        run("S-22", "safety", "safety_checks --require rename --vid <dir>", "-", "OK", sc + ["--require", "rename", "--vid", led], s)
        run("S-23", "safety", "safety_checks (no arguments)", "-", "prints usage or 'nothing to check'", sc, s)
        run("S-24", "safety", "safety_checks --gait <missing file>", "-", "clear error", sc + ["--gait", s / "nope.csv"], s)

    (runs / "baseline_results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    with open(runs / "baseline_results.md", "w") as f:
        f.write("| id | entry point | scenario | exit | s | traceback | last line |\n|---|---|---|---|---|---|---|\n")
        for r_ in RESULTS:
            last = (r_["tail"][-1] if r_["tail"] else "").replace("|", "\\|")[:150]
            f.write(f"| {r_['id']} | {r_['entry']} | {r_['scenario']} | {r_['exit']} | {r_['secs']} | "
                    f"{'yes' if r_['traceback'] else ''} | {last} |\n")
    n_tb = sum(r_["traceback"] for r_ in RESULTS)
    n_nz = sum(r_["exit"] != 0 for r_ in RESULTS)
    print(f"\n{len(RESULTS)} runs: {n_nz} with a non-zero exit, {n_tb} with a traceback -> {runs / 'baseline_results.json'}")


def lmm_summary(out_dir: Path) -> dict:
    import pandas as pd
    p = out_dir / "lmm_results_all.csv"
    o = {"lmm_results_all_exists": p.exists(), "heatmaps": sorted(x.name for x in out_dir.glob("heatmap_*.png"))}
    if p.exists():
        d = pd.read_csv(p)
        o["rows"] = len(d)
        o["columns"] = list(d.columns)
        o["has_fdr_column"] = any("fdr" in c.lower() or c.lower().startswith("q") for c in d.columns)
        sl = d[(d.variable == "stride_length_euclid_cm")]
        o["stride_length_rows"] = [{k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in r.items()
                                    if k in ("model", "n_obs", "n_mice", "Tau_coef", "Tau_pval", "APP_coef", "APP_pval", "converged")}
                                   for r in sl.to_dict("records")]
    return o


if __name__ == "__main__":
    main()
