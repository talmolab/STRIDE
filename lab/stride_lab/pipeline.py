"""From named clips to the STRIDE tables, for a lab cohort.

    per-day scale check  ->  pose + ROI inference (workers spread over the GPUs)  ->  the public T-maze pipeline
    (ROI files, consensus ROI, decisions, decision rules, gait)  ->  gait calibration gate  ->  cohort summary

No analysis is done here: the tables come from `stride.stages.tmaze_pipeline.run_tmaze_pipeline`, inference from the
public `stride pose-inference` / `stride roi-inference` commands, the gate from `stride.stages.safety_checks`.
This module decides WHICH days run with WHICH px/cm, builds the metadata table from the recorder export exactly as
the lab's drivers did, and writes the cohort summary and the notes a person should read.

Layout written under `output_dir`:
    decisions/ gait/ scale/ roi/ pose/ SUMMARY.md tmaze_run.log      the public pipeline's own outputs (whole cohort)
    own_scale/DAY{N}/...                                              a day accepted with its own px/cm (separate run)
    metadata_for_stride.csv                                           day, mouse, reward, sex, genotype
    roi_position_per_day.csv  cohort_summary.csv  COHORT_SUMMARY.md  DATA_QUALITY.md
"""
from __future__ import annotations

import csv
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from . import calibration as cal
from .cohort import board_of, camera_of, need_output
from .intake._log import CMD, IntakeError, StageLog, write_manifest
from .intake.orchestrate import day_dirs, resolve_days
from .roi_position import roi_position_report

PLACEHOLDER = re.compile(r"^\d{1,2}$")          # recorder placeholders 1..16; real ids are longer


# ------------------------------------------------------------------------------------------------- metadata
def best_trials_full(raw_dir, skip_dirs=()) -> Optional[Path]:
    """The most complete recorder export under the cohort folder: most distinct days, then the last by name
    (the rule of the lab's drivers)."""
    skip = {Path(s).resolve() for s in skip_dirs}
    files = []
    for f in sorted(Path(raw_dir).rglob("*trials_full*.csv")):
        if any(s in f.resolve().parents for s in skip) or "waste" in {p.lower() for p in f.parent.parts}:
            continue
        try:
            with open(f, newline="", encoding="utf-8-sig") as fh:
                n_days = len({(r.get("day") or "").strip() for r in csv.DictReader(fh)})
        except (OSError, UnicodeDecodeError, csv.Error):
            continue
        files.append((n_days, str(f), f))
    return max(files)[2] if files else None


def build_analysis_metadata(trials_full, out_csv, allow_placeholders: bool = False, genotype: Optional[str] = None,
                            log: Optional[StageLog] = None) -> dict:
    """day, mouse, reward (+ sex, genotype) for the analysis, from the recorder export.

    One row per (day, mouse): the first row of the export with a valid reward (L / R), all phases. Sex and
    genotype are the first non-empty value the export holds for the mouse. `trials_in_export` (the number of
    non-empty T1..T9 cells) is carried for the notes only. Refuses an export whose mouse ids
    are all 1-2 digit numbers (the recorder's placeholders) unless `allow_placeholders`.
    """
    say = log or (lambda m: print(m, flush=True))
    with open(trials_full, newline="", encoding="utf-8-sig") as fh:
        src = list(csv.DictReader(fh))
    sex, geno = {}, {}
    for r in src:
        m = (r.get("mouse") or "").strip()
        if m and m not in sex and (r.get("sex") or "").strip():
            sex[m] = r["sex"].strip()
        if m and m not in geno and (r.get("genotype") or "").strip():
            geno[m] = r["genotype"].strip()
    seen, rows, ids = set(), [], set()
    for r in src:
        day = (r.get("day") or "").strip().upper()
        mouse = (r.get("mouse") or "").strip()
        rew = (r.get("reward") or "").strip().upper()[:1]
        if not mouse or rew not in ("L", "R") or (day, mouse) in seen:
            continue
        seen.add((day, mouse))
        ids.add(mouse)
        n_trials = sum(1 for i in range(1, 10) if (r.get(f"T{i}") or "").strip())
        rows.append(dict(day=day, mouse=mouse, reward=rew, sex=sex.get(mouse, ""), genotype=genotype or geno.get(mouse, ""),
                         trials_in_export=n_trials))
    if not rows:
        raise IntakeError(f"{trials_full}: no (day, mouse) row with a reward of L or R")
    if all(PLACEHOLDER.match(m) for m in ids) and not allow_placeholders:
        raise IntakeError(f"{Path(trials_full).name} still has placeholder mouse ids ({', '.join(sorted(ids, key=int)[:5])}...): "
                          "enter the real ids in the recorder, re-export and copy the CSVs — or set allow_placeholders: true "
                          "when these numbers ARE the ids written on the whiteboards")
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["day", "mouse", "reward", "sex", "genotype", "trials_in_export"])
        w.writeheader()
        w.writerows(rows)
    no_sex = sorted(m for m in ids if not sex.get(m))
    if no_sex:
        say(f"WARN no sex in the export for mice {', '.join(no_sex)}")
    say(f"metadata: {len(rows)} (day, mouse) rows from {Path(trials_full).name} -> {out_csv}")
    return dict(rows=rows, n_rows=len(rows), mice=sorted(ids), path=out_csv, source=str(trials_full))


# ------------------------------------------------------------------------------------------------ inference
def _pose_missing(videos, pose_dirs) -> list:
    from stride.utils.pose_io import resolve_pose_path
    return [v for v in videos if not resolve_pose_path(v, ".slp", pose_dirs).exists()]


def _roi_missing(videos, roi_dir: Path) -> list:
    from stride.stages.decision_analysis import resolve_roi_yml
    from stride.stages.roi_inference import roi_output_path
    return [v for v in videos if resolve_roi_yml(v, [roi_dir]) is None and not roi_output_path(v, roi_dir).exists()]


def inference_plan(cfg: dict, days: dict, camera) -> list:
    """The worker commands for pose and ROI inference, per day: `workers_per_device` workers on every device, each
    taking its share of the day's clips through the public commands' --worker-id / --num-workers options."""
    out = need_output(cfg)
    inf = cfg["inference"]
    devices = list(inf["devices"])
    wpd = max(1, int(inf["workers_per_device"]))
    n_workers = len(devices) * wpd
    pose_dir, roi_dir = out / "pose", out / "roi"
    plan = []
    for n, d in sorted(days.items()):
        vids = camera.videos(d)
        for stage, missing, models in (("pose", _pose_missing(vids, [pose_dir]), cfg["models"]["pose"]),
                                       ("roi", _roi_missing(vids, roi_dir), cfg["models"]["roi"])):
            if not missing:
                continue
            for i in range(n_workers):
                dev = devices[i // wpd]
                cmd = [sys.executable, "-m", "stride.cli", f"{stage}-inference", "-i", str(d)]
                for m in models:
                    cmd += ["-m", str(m)]
                cmd += ["--device", dev, "--worker-id", str(i), "--num-workers", str(n_workers)]
                if stage == "pose":
                    cmd += ["--batch-size", str(int(inf["batch_size"]))]
                    if inf["sidecars"] == "output":
                        cmd += ["-o", str(pose_dir)]
                else:
                    cmd += ["-o", str(roi_dir)]
                plan.append(dict(day=n, folder=d.name, stage=stage, worker=i, n_workers=n_workers, device=dev,
                                 clips_missing=len(missing), cmd=cmd))
    return plan


def print_plan(plan: list, log) -> None:
    if not plan:
        log("inference plan: nothing to do (every clip has its pose and ROI file)")
        return
    for (n, stage), rows in _group(plan).items():
        log(f"plan DAY{n} {stage}: {rows[0]['clips_missing']} clip(s) to do, {len(rows)} worker(s) on "
            + ", ".join(f"{dev} x{c}" for dev, c in Counter(r["device"] for r in rows).items()))
        for r in rows:
            log("  " + " ".join(_quote(c) for c in r["cmd"]))


def _quote(s: str) -> str:
    return f'"{s}"' if (" " in s) else s


def _group(plan: list) -> dict:
    g = defaultdict(list)
    for r in plan:
        g[(r["day"], r["stage"])].append(r)
    return dict(sorted(g.items(), key=lambda kv: (kv[0][0], kv[0][1] != "pose")))


def run_inference_fleet(cfg: dict, plan: list, log) -> None:
    """Run the plan: per day, the pose workers together, then the ROI workers together. A worker that fails stops
    the run with its log file named."""
    rec = cfg["records_dir"]
    logs = rec / "inference"
    logs.mkdir(parents=True, exist_ok=True)
    for (n, stage), rows in _group(plan).items():
        log(f"DAY{n} {stage} inference: {rows[0]['clips_missing']} clip(s), {len(rows)} worker(s) "
            f"({', '.join(sorted({r['device'] for r in rows}))})")
        procs = []
        for r in rows:
            lf = logs / f"day{n}_{stage}_w{r['worker']}.log"
            fh = open(lf, "w", encoding="utf-8")
            procs.append((r, lf, fh, subprocess.Popen(r["cmd"], stdout=fh, stderr=subprocess.STDOUT)))
        failed = []
        for r, lf, fh, p in procs:
            rc = p.wait()
            fh.close()
            if rc != 0:
                failed.append((r["worker"], rc, lf))
        if failed:
            w, rc, lf = failed[0]
            log.refuse(f"DAY{n} {stage} inference: worker {w} exited with code {rc} — see {lf}")
        log(f"DAY{n} {stage} inference done (worker logs: {logs}/day{n}_{stage}_w*.log)")


# ------------------------------------------------------------------------------------------------- pipeline
def _tmaze(cfg: dict, videos: list, out: Path, px_per_cm: float, meta_csv: Path, steps: dict, log, stop_after=None,
           extra: Optional[dict] = None) -> dict:
    from stride.stages.tmaze_pipeline import load_settings, run_tmaze_pipeline

    an, inf = cfg["analysis"], cfg["inference"]
    ov = dict(videos=[str(v) for v in videos], output=str(out), pose_model=list(cfg["models"]["pose"]),
              roi_model=list(cfg["models"]["roi"]), metadata=str(meta_csv), device=str(inf["devices"][0]),
              workers=int(an["workers"]), batch_size=int(inf["batch_size"]),
              scale={"px_per_cm": float(px_per_cm)}, decision_rules={"window_s": float(an["decision_window_s"])},
              stride_detection=dict(an.get("stride_detection") or {}), stride_filtering=dict(an.get("stride_filtering") or {}),
              steps=dict(steps), stop_after=stop_after)
    ov.update(extra or {})
    summ = run_tmaze_pipeline(load_settings(None, ov))
    if not summ.get("ok"):
        log.refuse(f"the STRIDE T-maze pipeline stopped: {summ.get('error')}")
    return summ


def _gait_gate(gait_dir: Path, px_per_cm: float, log) -> None:
    """The lab's gait calibration gate: `python -m stride.stages.safety_checks --gait <filtered> --expect-pxcm <value>`."""
    g = gait_dir / "gait_per_stride_filtered.csv"
    if not g.exists():
        log.warn(f"no filtered stride table in {gait_dir} (no stride detected): the gait calibration gate was not run")
        return
    r = subprocess.run([sys.executable, "-m", "stride.stages.safety_checks", "--gait", str(g), "--expect-pxcm", f"{px_per_cm:g}"],
                       capture_output=True, text=True)
    tail = [ln for ln in (r.stdout or "").strip().splitlines() if ln.strip()][-6:]
    for ln in tail:
        log(f"gate: {ln.strip()}")
    if r.returncode != 0:
        log.refuse(f"gait calibration gate FAILED for {g} (expected px/cm {px_per_cm:g}): see the lines above")
    log(f"gait calibration gate OK ({g.name}, expected px/cm {px_per_cm:g})")


def _copy_sidecars(days: dict, out: Path, camera, log) -> int:
    """Lab convention: pose and ROI files sit next to the clips. Copies what the pipeline wrote under the output
    folder next to its clip when no such file is there yet."""
    n = 0
    for _n, d in sorted(days.items()):
        for v in camera.videos(d):
            for src, dst in ((out / "roi" / f"{v.stem}.rois.yml", v.with_suffix(".rois.yml")),
                             (out / "pose" / f"{v.stem}.slp", v.with_suffix(".slp"))):
                if src.exists() and not dst.exists():
                    shutil.copyfile(src, dst)
                    n += 1
    if n:
        log(f"{n} pose / ROI file(s) copied next to their clips (inference.sidecars: videos)")
    return n


def ready_days(cfg: dict, camera) -> tuple:
    """({day: folder} fully renamed with clips, {day: n camera-named clips} not renamed yet) for DAY >= 1."""
    ready, raw = {}, {}
    for n, d in day_dirs(cfg["raw_dir"]).items():
        if n < 1:
            continue
        vids = camera.videos(d)
        n_raw = sum(1 for p in vids if camera.is_raw(p.name))
        if n_raw:
            raw[n] = n_raw
        elif vids:
            ready[n] = d
    return ready, raw


def run_calibration(cfg: dict, days=None, force: bool = False, log=None) -> dict:
    c = cfg["calibration"]
    return cal.monitor(cfg["raw_dir"], cfg["records_dir"], days, board=board_of(cfg), camera=camera_of(cfg),
                       clips_per_day=c["clips_per_day"], warn_pct=c["warn_pct"], stop_pct=c["stop_pct"],
                       reference_px_per_cm=c["reference_px_per_cm"], given_px_per_cm=c["px_per_cm"],
                       on_unmeasured_day=c["on_unmeasured_day"], workers=int(cfg["analysis"]["workers"]), force=force, log=log)


def pipeline(cfg: dict, days=None, skip_inference: Optional[bool] = None, plan_only: bool = False,
             log: Optional[StageLog] = None) -> dict:
    """Scale check -> inference -> the public T-maze pipeline -> gate -> summary. Safe to re-run.

    `days` limits the scale measurement and the inference; the analysis always covers every day of the cohort
    that is ready (renamed, not held, not excluded), as the lab's whole-cohort STRIDE stage did.
    Returns {"status": "complete" | "held" | "plan", ...}.
    """
    out = need_output(cfg)
    rec, camera = cfg["records_dir"], camera_of(cfg)
    log = log or StageLog("pipeline", rec / "intake.log")
    skip_inference = bool(cfg["inference"]["skip"]) if skip_inference is None else bool(skip_inference)
    ready, raw = ready_days(cfg, camera)
    sel_nums = resolve_days(cfg["raw_dir"], days) if days else sorted(ready) + sorted(raw)
    bad = [n for n in sel_nums if n in raw]
    if bad:
        log.refuse(f"day(s) {', '.join(map(str, bad))} still hold camera-named clips ({', '.join(str(raw[n]) for n in bad)}): "
                   f"finish the whiteboard check and the rename first ({CMD} auto --apply)")
    unknown = [n for n in sel_nums if n not in ready]
    if unknown:
        log.refuse(f"no renamed clips for day(s) {', '.join(map(str, unknown))} under {cfg['raw_dir']}")
    if not ready:
        log.refuse(f"no renamed day folder under {cfg['raw_dir']}: nothing to analyse")
    selected = {n: ready[n] for n in sel_nums}

    # ---- scale check -------------------------------------------------------------------------------
    log.stage = "calibrate"
    state = run_calibration(cfg, days=",".join(map(str, sorted(ready))), log=log)
    use = {n: state["days"].get(str(n), {}).get("use", "cohort") for n in ready}
    main = {n: d for n, d in ready.items() if use[n] == "cohort"}
    own = {n: d for n, d in ready.items() if use[n] == "own"}
    held = sorted(n for n in ready if use[n] == "hold")
    excluded = sorted(n for n in ready if use[n] == "exclude")
    px = float(state["used_px_per_cm"])
    log.stage = "pipeline"
    log(f"days: {len(main)} with the cohort px/cm {px:g}"
        + (f"; own value: {', '.join(f'DAY{n}' for n in sorted(own))}" if own else "")
        + (f"; HELD (scale): {', '.join(f'DAY{n}' for n in held)}" if held else "")
        + (f"; excluded: {', '.join(f'DAY{n}' for n in excluded)}" if excluded else ""))

    # ---- metadata ----------------------------------------------------------------------------------
    tf = Path(str(cfg["metadata"])).expanduser() if cfg.get("metadata") else best_trials_full(cfg["raw_dir"], [rec, out])
    if tf is None or not Path(tf).is_file():
        log.refuse(f"no recorder export (*_trials_full.csv) found under {cfg['raw_dir']}"
                   if not cfg.get("metadata") else f"metadata file not found: {tf}")
    meta = build_analysis_metadata(tf, out / "metadata_for_stride.csv", allow_placeholders=bool(cfg["allow_placeholders"]),
                                   genotype=cfg.get("genotype"), log=log)

    # ---- inference ---------------------------------------------------------------------------------
    infer_days = {n: d for n, d in selected.items() if n in main or n in own}
    plan = [] if skip_inference else inference_plan(cfg, infer_days, camera)
    if plan_only:
        log.stage = "plan"
        print_plan(plan, log)
        return dict(status="plan", plan=plan, held=held, excluded=excluded, px_per_cm=px)
    summaries = {}
    if skip_inference:
        log("inference skipped: only clips that already have a pose file and an ROI file are analysed")
    elif infer_days:
        if plan:
            for k in ("pose", "roi"):
                if any(r["stage"] == k for r in plan) and not cfg["models"][k]:
                    log.refuse(f"{k} inference is needed but the cohort config has no models.{k}")
            log.stage = "inference"
            run_inference_fleet(cfg, plan, log)
            log.stage = "pipeline"
        # ROI regions files + consensus ROI for the days just inferred; inference of anything the workers left
        summaries["inference"] = _tmaze(
            cfg, list(infer_days.values()), out, px, meta["path"], log=log, stop_after="roi_consensus",
            steps=dict(distortion_check=False, calibrate=False, pose_inference=True, roi_inference=True, roi_consensus=True,
                       decisions=False, decision_rules=False, gait=False))
        if cfg["inference"]["sidecars"] == "videos":
            _copy_sidecars(infer_days, out, camera, log)

    # ---- analysis: the whole cohort that is ready --------------------------------------------------
    an_steps = dict(distortion_check=bool(cfg["analysis"]["distortion_check"]), calibrate=False, pose_inference=False,
                    roi_inference=False, roi_consensus=False, decisions=True, decision_rules=True, gait=True)
    if main:
        summaries["cohort"] = _tmaze(cfg, [main[n] for n in sorted(main)], out, px, meta["path"], an_steps, log)
        _gait_gate(out / "gait", px, log)
    else:
        log.warn("no day runs with the cohort px/cm: the cohort tables were not (re)written")
    for n, d in sorted(own.items()):
        own_px = float(state["days"][str(n)]["value_used"])
        o = out / "own_scale" / f"DAY{n}"
        extra = {k: str(out / sub) for k, sub in (("pose_dir", "pose"), ("roi_dir", "roi")) if (out / sub).is_dir()}
        log(f"DAY{n}: analysed on its own with its own px/cm {own_px:g} (recorded decision) -> {o}")
        summaries[f"own_DAY{n}"] = _tmaze(cfg, [d], o, own_px, meta["path"], an_steps, log, extra=extra)
        _gait_gate(o / "gait", own_px, log)

    # ---- maze position, summary, notes --------------------------------------------------------------
    rp = roi_position_report({n: ready[n] for n in sorted(ready) if n not in excluded}, [out / "roi"], camera,
                             out / "roi_position_per_day.csv", float(cfg["roi_position"]["drift_warn_px"]),
                             float(cfg["roi_position"]["stem_angle_warn_deg"]))
    for w in rp["warnings"]:
        log.warn(w)
    info = write_summary(cfg, state=state, roi_pos=rp, summaries=summaries, meta=meta, log=log)
    status = "held" if held else "complete"
    write_manifest(out, "lab_pipeline",
                   models=list(cfg["models"]["pose"]) + list(cfg["models"]["roi"]),
                   params=dict(cohort=cfg["cohort"], px_per_cm=px, px_per_cm_source=state["source"], skip_inference=skip_inference,
                               days_selected=sorted(selected), calibration=cfg["calibration"], roi_position=cfg["roi_position"],
                               inference={k: v for k, v in cfg["inference"].items()}, analysis=cfg["analysis"]),
                   inputs=dict(raw_dir=str(cfg["raw_dir"]), metadata=meta["source"], config=cfg.get("config_file")),
                   outputs=[str(p) for p in (out / "decisions" / "decisions_clean.csv", out / "gait" / "gait_per_stride_filtered.csv",
                                             out / "COHORT_SUMMARY.md", out / "DATA_QUALITY.md") if p.exists()],
                   results=dict(status=status, days_cohort=sorted(main), days_own=sorted(own), days_held=held, days_excluded=excluded,
                                per_day=info["per_day"], warnings=info["n_notes"]))
    if held:
        log(f"STOP: day(s) {', '.join(map(str, held))} are held by the scale check and were not analysed. Read "
            f"{cal.cal_dir(cfg['raw_dir'], rec) / 'CALIBRATION.md'}, record a decision "
            f"({CMD} calibrate --decide N=accept_own|accept_cohort|exclude --by NAME) and re-run.")
    else:
        log(f"PIPELINE COMPLETE: {out / 'COHORT_SUMMARY.md'}; notes for a person: {out / 'DATA_QUALITY.md'}")
    return dict(status=status, held=held, excluded=excluded, own=sorted(own), px_per_cm=px, summaries=summaries, output=str(out))


# -------------------------------------------------------------------------------------------------- summary
def _read(path) -> list:
    try:
        with open(path, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _dayn(text) -> Optional[int]:
    m = re.search(r"(\d+)", str(text or ""))
    return int(m.group(1)) if m else None


def write_summary(cfg: dict, state: Optional[dict] = None, roi_pos: Optional[dict] = None, summaries: Optional[dict] = None,
                  meta: Optional[dict] = None, log=None) -> dict:
    """COHORT_SUMMARY.md + cohort_summary.csv (counts per day; accuracy = share of decided trials that entered the
    rewarded arm, as the lab's summary script reports it) and DATA_QUALITY.md (what a person should look at)."""
    from stride.stages.decision_analysis import resolve_roi_yml
    from stride.stages.roi_consensus import is_borrowed_roi
    from stride.utils.pose_io import resolve_pose_path

    out = need_output(cfg)
    rec, camera = cfg["records_dir"], camera_of(cfg)
    state = state or cal.load_state(cfg["raw_dir"], rec) or {"days": {}}
    summaries = summaries or {}
    ready, raw = ready_days(cfg, camera)
    all_days = {n: d for n, d in day_dirs(cfg["raw_dir"]).items() if n >= 1}
    sources = [(out, None)] + [(p, _dayn(p.name)) for p in sorted((out / "own_scale").glob("DAY*")) if p.is_dir()]
    dec_rows, gait_rows = [], []
    for o, _n in sources:
        dec_rows += _read(o / "decisions" / "decisions_clean.csv")
        gait_rows += _read(o / "gait" / "gait_per_stride_filtered.csv")
    by_day_dec = defaultdict(list)
    for r in dec_rows:
        by_day_dec[_dayn(r.get("day"))].append(r)
    strides = Counter(_dayn(r.get("day")) for r in gait_rows)
    consensus, per_day = defaultdict(list), []
    for n, d in sorted(all_days.items()):
        vids = camera.videos(d)
        n_pose = sum(1 for v in vids if resolve_pose_path(v, ".slp", [out / "pose"]).exists())
        n_roi = 0
        for v in vids:
            y = resolve_roi_yml(v, [out / "roi"])
            if y is not None:
                n_roi += 1
                if is_borrowed_roi(y):
                    consensus[n].append(v.name)
        rows = by_day_dec.get(n, [])
        decided = [r for r in rows if (r.get("entry_side") or "") in ("L", "R")]
        correct = sum(1 for r in decided if r.get("entry_side") == r.get("reward"))
        cd = state["days"].get(str(n), {})
        per_day.append(dict(
            day=n, folder=d.name, clips=len(vids), camera_named=raw.get(n, 0), pose_files=n_pose, roi_files=n_roi,
            consensus_roi=len(consensus[n]), trials_analysed=len(rows), mice=len({r.get("mouse") for r in decided}),
            decisions=len(decided), no_decision=len(rows) - len(decided),
            accuracy=round(correct / len(decided), 4) if decided else "", filtered_strides=strides.get(n, 0),
            px_per_cm=cd.get("px_per_cm", ""), scale_status=cd.get("status", ""), scale_use=cd.get("use", "")))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cohort_summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_day[0].keys()) if per_day else ["day"])
        w.writeheader()
        w.writerows(per_day)
    tot = lambda k: sum(int(r[k] or 0) for r in per_day)   # noqa: E731
    decided_all = [r for r in dec_rows if (r.get("entry_side") or "") in ("L", "R")]
    L = [f"# Cohort summary — {cfg['cohort']}", "",
         f"- raw folder: `{cfg['raw_dir']}`", f"- output folder: `{out}`",
         f"- px/cm used for the cohort: **{state.get('used_px_per_cm')}** ({state.get('source', 'no scale check yet')})",
         f"- clips: {tot('clips')} in {len(per_day)} day folder(s); trials analysed: {len(dec_rows)}; decisions: {len(decided_all)}; "
         f"no decision: {len(dec_rows) - len(decided_all)}; filtered strides: {len(gait_rows)}",
         f"- mice with a decision: {len({r.get('mouse') for r in decided_all})}", "",
         "Accuracy = share of the decided trials (entry side L or R) that entered the rewarded arm, per day. No learning "
         "criterion is computed here.", "",
         "| day | folder | clips | pose | ROI (consensus) | trials | mice | decisions | no decision | accuracy | filtered strides | px/cm | scale |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in per_day:
        L.append(f"| DAY{r['day']} | {r['folder']} | {r['clips']}" + (f" ({r['camera_named']} camera-named)" if r["camera_named"] else "")
                 + f" | {r['pose_files']} | {r['roi_files']} ({r['consensus_roi']}) | {r['trials_analysed']} | {r['mice']} | {r['decisions']} | "
                 f"{r['no_decision']} | {r['accuracy'] if r['accuracy'] == '' else format(r['accuracy'], '.3f')} | {r['filtered_strides']} | "
                 f"{r['px_per_cm']} | {r['scale_status']}{'' if r['scale_use'] in ('cohort', '') else ' -> ' + r['scale_use']} |")
    (out / "COHORT_SUMMARY.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---- what a person should look at ---------------------------------------------------------------
    Q = [f"# Data-quality notes — {cfg['cohort']}", "",
         "Written by `stride-lab`. Each section lists what a person should look at; an empty section says so.", ""]
    n_notes = 0

    def section(title, items, empty="nothing to report"):
        nonlocal n_notes
        Q.extend([f"## {title}", ""])
        Q.extend([f"- {i}" for i in items] if items else [f"- {empty}"])
        Q.append("")
        n_notes += len(items)

    section("Scale (px/cm per day)", list(state.get("warnings") or [])
            + [f"DAY{n}: {d['decision']} (recorded decision; day {d.get('px_per_cm')} px/cm, {d.get('deviation_pct')} % from the cohort median)"
               for n, d in sorted(state["days"].items(), key=lambda kv: int(kv[0])) if d.get("decision")],
            f"every measured day is within {(state.get('thresholds') or {}).get('warn_pct', '?')} % of the cohort median")
    section("Maze position (from the ROI files)", list((roi_pos or {}).get("warnings") or []),
            "no day's maze position differs from the first day beyond the warning levels"
            if roi_pos else "not computed in this run (it is written by the pipeline stage)")
    qc = sorted({str(p.parent) for p in out.rglob("MONTAGE_*.png")})
    section("Clips that use the session's consensus ROI",
            [f"DAY{n}: {len(c)} clip(s): {', '.join(sorted(c))}" for n, c in sorted(consensus.items()) if c]
            + ([f"check that every borrowed ROI lands on the maze: QC montage(s) in {', '.join(qc)}"] if qc and any(consensus.values()) else []))
    waste = sorted(p for p in (Path(cfg["raw_dir"]) / "waste").rglob("*") if camera.is_video(p)) if (Path(cfg["raw_dir"]) / "waste").is_dir() else []
    section("Clips taken out (waste)", [f"{p.parent.name}/{p.name}" for p in waste])
    left = {}
    for key, s in summaries.items():
        for name, why in (s.get("skipped") or {}).items():
            left.setdefault(name, why)
    section("Clips left out by the analysis", [f"{n}: {w}" for n, w in sorted(left.items())])
    meta_rows = (meta or {}).get("rows") or _read(out / "metadata_for_stride.csv")
    have_meta = {(str(r.get("day", "")).upper(), str(r.get("mouse", ""))) for r in meta_rows}
    ran = {(str(r.get("day", "")).upper(), str(r.get("mouse", ""))) for r in meta_rows if int(r.get("trials_in_export") or 0) > 0}
    no_reward = sorted({f"{r.get('day')} mouse {r.get('mouse')}" for r in dec_rows
                        if (str(r.get("day", "")).upper(), str(r.get("mouse", ""))) not in have_meta})
    seen = {(str(r.get("day", "")).upper(), str(r.get("mouse", ""))) for r in dec_rows}
    done_days = {f"DAY{r['day']}" for r in per_day if r["trials_analysed"]}
    no_clip = sorted(f"{d} mouse {m}" for d, m in ran - seen if d in done_days)
    section("Metadata that does not match",
            [f"clips with no reward row in the export: {x}" for x in no_reward]
            + [f"the export lists trials but no clip was analysed: {x}" for x in no_clip])
    warn = []
    for key, s in summaries.items():
        warn += [f"({key}) {w}" for w in (s.get("warnings") or [])]
    section("Warnings from the STRIDE pipeline", warn)
    not_ready = [f"DAY{n}: {c} camera-named clip(s) — whiteboard check and rename not finished" for n, c in sorted(raw.items())]
    section("Days not ready", not_ready)
    (out / "DATA_QUALITY.md").write_text("\n".join(Q) + "\n", encoding="utf-8")
    if log is not None:
        log(f"summary: {out / 'COHORT_SUMMARY.md'} ({len(dec_rows)} trials, {len(gait_rows)} filtered strides); "
            f"notes: {out / 'DATA_QUALITY.md'} ({n_notes} item(s))")
    return dict(per_day=per_day, n_notes=n_notes)
