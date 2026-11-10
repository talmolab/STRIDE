"""px/cm of the cohort, measured on every recording day, as a check that the camera was not moved.

The scale is not a pinned number. On every recording day a few clips (spread over the day, chosen automatically) are
read with the public ChArUco measurement (`stride.stages.scale_calibration.measure_clip`). The day's value is the
median of its clip readings; the cohort value is the median over the days. The cohort value is what the gait stage
uses when the days agree.

    |day - cohort median| / cohort median   <= warn_pct          ok
                                            >  warn_pct          WARNING (processed with the cohort value)
                                            >  stop_pct          STOP: the day is held until a person records a
                                                                 decision (accept_own | accept_cohort | exclude)

The cohort value is also compared with `reference_px_per_cm` (the previous cohort on the same rig): warning only.

Files, all under `<records>/calibration/`:
    day{N}/<clip>.json, <clip>_checkerboard.png    per-clip readings and QC images
    per_day_px_per_cm.csv                          the per-day table
    px_per_cm.txt                                  the value the analysis uses for the cohort
    calibration_state.json                         everything above, machine-readable
    day_decisions.csv                              the recorded decisions for flagged days (append-only)
    CALIBRATION.md                                 the report for a person
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import statistics as st
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from stride.stages.scale_calibration import BoardSpec, DEFAULT_BOARD, measure_clip

from .intake._log import CMD, IntakeError, StageLog, write_manifest
from .intake.camera import CameraProfile, DJI
from .intake.orchestrate import _need_days, records_dir_of

DECISIONS = ("accept_own", "accept_cohort", "exclude", "clear")
DAY_COLS = ["day", "folder", "px_per_cm", "spread_pct", "clips_measured", "clips_used", "deviation_pct", "status",
            "decision", "use", "value_used", "clips"]


def cal_dir(cohort_dir, records_dir=None) -> Path:
    return records_dir_of(cohort_dir, records_dir) / "calibration"


def pick_day_clips(clips: list, k: int) -> list:
    """`k` clips spread evenly over the day's recording order (first, ..., last); no hand-picking."""
    n = len(clips)
    if n == 0:
        return []
    k = max(1, min(int(k), n))
    if k == 1:
        return [clips[0]]
    return [clips[i] for i in sorted({round(j * (n - 1) / (k - 1)) for j in range(k)})]


def _read_clip(clip: Path, out_dir: Path, board: BoardSpec, camera: CameraProfile, force: bool) -> dict:
    cache = out_dir / f"{clip.stem}.json"
    if cache.exists() and not force:
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("stem") == clip.stem and "charuco" in d:
                return d
        except (OSError, ValueError):
            pass
    out_dir.mkdir(parents=True, exist_ok=True)      # measure_clip writes its JSON there also when no board is found
    return measure_clip(clip, out_dir, board, camera.rail_crop, camera.frame_size)


# ------------------------------------------------------------------------------------------------- decisions
def read_decisions(cdir: Path) -> dict:
    """{day number: the latest recorded decision row}; a `clear` row removes the day's decision."""
    f = Path(cdir) / "day_decisions.csv"
    out = {}
    if f.exists():
        with open(f, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                n = int(r["day"])
                if r["decision"] == "clear":
                    out.pop(n, None)
                else:
                    out[n] = r
    return out


def record_decision(cohort_dir, records_dir, day: int, decision: str, by: str = "", note: str = "",
                    log: Optional[StageLog] = None) -> dict:
    """Append a person's decision for a flagged day to day_decisions.csv (the explicit, logged override)."""
    cdir = cal_dir(cohort_dir, records_dir)
    log = log or StageLog("calibrate", records_dir_of(cohort_dir, records_dir) / "intake.log")
    if decision not in DECISIONS:
        log.refuse(f"decision must be one of {', '.join(DECISIONS)} (got {decision!r})")
    state = load_state(cohort_dir, records_dir)
    if state is None:
        log.refuse(f"no calibration yet: run `{CMD} calibrate` first")
    d = state["days"].get(str(day))
    if d is None:
        log.refuse(f"DAY{day} is not in the calibration table (days measured: {', '.join(state['days']) or 'none'})")
    if decision == "accept_own" and d.get("px_per_cm") is None:
        log.refuse(f"DAY{day} has no board reading of its own, so `accept_own` is not possible: use accept_cohort or exclude")
    if decision != "clear" and not str(by).strip():
        log.refuse("a decision needs --by (who decided)")
    cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / "day_decisions.csv"
    cols = ["day", "decision", "by", "note", "recorded_at", "day_px_per_cm", "cohort_px_per_cm", "deviation_pct", "status_then"]
    new = not f.exists()
    row = dict(day=day, decision=decision, by=by, note=note, recorded_at=f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}",
               day_px_per_cm=d.get("px_per_cm"), cohort_px_per_cm=state.get("cohort_px_per_cm"),
               deviation_pct=d.get("deviation_pct"), status_then=d.get("status"))
    with open(f, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)
    log(f"DAY{day}: decision recorded: {decision}" + (f" by {by}" if by else "") + (f" — {note}" if note else "")
        + f" (day {d.get('px_per_cm')} px/cm, cohort {state.get('cohort_px_per_cm')}, deviation {d.get('deviation_pct')} %)")
    return row


# ---------------------------------------------------------------------------------------------------- monitor
def _use(status: str, decision: Optional[str], on_unmeasured: str) -> str:
    if decision == "exclude":
        return "exclude"
    if decision == "accept_own":
        return "own"
    if decision == "accept_cohort":
        return "cohort"
    if status == "stop" or (status == "unmeasured" and on_unmeasured == "stop"):
        return "hold"
    return "cohort"


def monitor(cohort_dir, records_dir=None, days=None, board: BoardSpec = DEFAULT_BOARD, camera: CameraProfile = DJI,
            clips_per_day: int = 3, warn_pct: float = 0.5, stop_pct: float = 1.0, reference_px_per_cm: Optional[float] = None,
            given_px_per_cm: Optional[float] = None, on_unmeasured_day: str = "warn", workers: int = 1, force: bool = False,
            log: Optional[StageLog] = None) -> dict:
    """Measure every day (cached per clip), classify the days and write the report. Returns the state dict.

    `days` limits which days are (re)measured; days measured earlier stay in the table, and the cohort median is
    always taken over every measured day of the cohort.
    """
    root, rec = Path(cohort_dir), records_dir_of(cohort_dir, records_dir)
    cdir = rec / "calibration"
    log = log or StageLog("calibrate", rec / "intake.log")
    cdir.mkdir(parents=True, exist_ok=True)
    prev = load_state(cohort_dir, records_dir) or {}
    table = {int(k): v for k, v in (prev.get("days") or {}).items()}

    dd = _need_days(root, days, log)
    jobs = []
    for n, d in dd.items():
        clips = [p for p in camera.videos(d)]
        if any(camera.is_raw(p.name) for p in clips):
            log(f"DAY{n}: still holds camera-named clips — measured after the rename")
            continue
        chosen = pick_day_clips(clips, clips_per_day)
        if not chosen:
            log.warn(f"DAY{n}: no clips in {d.name}")
        jobs.append((n, d, chosen))
    flat = [(n, c) for n, _d, chosen in jobs for c in chosen]
    log(f"{board.describe()}; {clips_per_day} clip(s) per day on {len(jobs)} day(s) = {len(flat)} clips")

    def one(item):
        n, clip = item
        return n, clip, _read_clip(clip, cdir / f"day{n}", board, camera, force)

    if int(workers) > 1 and len(flat) > 1:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(flat))) as ex:      # OpenCV releases the GIL while decoding
            readings = list(ex.map(one, flat))
    else:
        readings = [one(x) for x in flat]
    by_day: dict = {}
    for n, clip, r in readings:
        by_day.setdefault(n, []).append((clip, r))
    for n, d, chosen in jobs:
        vals = [(c.name, r["charuco"]["pxcm_H"]) for c, r in by_day.get(n, []) if r.get("charuco") and r["charuco"].get("pxcm_H")]
        xs = [v for _, v in vals]
        med = st.median(xs) if xs else None
        table[n] = dict(day=n, folder=d.name, px_per_cm=round(med, 3) if med is not None else None,
                        spread_pct=round((max(xs) - min(xs)) / med * 100, 3) if xs else None,
                        clips_measured=len(chosen), clips_used=len(xs),
                        clips=";".join(f"{name}={v:.3f}" for name, v in vals),
                        no_board=";".join(c.name for c, r in by_day.get(n, []) if not (r.get("charuco") and r["charuco"].get("pxcm_H"))))

    measured = {n: t["px_per_cm"] for n, t in table.items() if t.get("px_per_cm") is not None}
    cohort_median = st.median(measured.values()) if measured else None
    cohort_value = round(cohort_median, 2) if cohort_median is not None else None
    if cohort_median is None and given_px_per_cm is None:
        _write_report(cdir, dict(days={}, cohort_px_per_cm=None), table, log)
        log.refuse("no ChArUco reading on any day, so the cohort has no px/cm: check that the board is in view and that the "
                   "board specification matches the print, or give `calibration.px_per_cm` in the cohort config")
    used = float(given_px_per_cm) if given_px_per_cm is not None else cohort_value
    decisions = read_decisions(cdir)
    for n, t in sorted(table.items()):
        if t.get("px_per_cm") is None or cohort_median is None:
            t["deviation_pct"], t["status"] = None, "unmeasured"
        else:
            dev = (t["px_per_cm"] - cohort_median) / cohort_median * 100
            t["deviation_pct"] = round(dev, 3)
            t["status"] = "stop" if abs(dev) > stop_pct else ("warn" if abs(dev) > warn_pct else "ok")
        dec = decisions.get(n, {}).get("decision")
        t["decision"] = dec or ""
        t["use"] = _use(t["status"], dec, on_unmeasured_day)
        t["value_used"] = (round(t["px_per_cm"], 2) if t["use"] == "own" else (used if t["use"] == "cohort" else None))

    warnings = []
    for n, t in sorted(table.items()):
        if t["status"] == "warn":
            warnings.append(f"DAY{n}: px/cm {t['px_per_cm']} is {t['deviation_pct']:+.2f} % from the cohort median "
                            f"({cohort_median:.3f}); above the warning level of {warn_pct:g} %")
        elif t["status"] == "stop":
            warnings.append(f"DAY{n}: px/cm {t['px_per_cm']} is {t['deviation_pct']:+.2f} % from the cohort median "
                            f"({cohort_median:.3f}); above the stop level of {stop_pct:g} % — the camera or the board moved"
                            + (f" (decision recorded: {t['decision']})" if t["decision"] else ""))
        elif t["status"] == "unmeasured":
            warnings.append(f"DAY{n}: no board reading ({t.get('clips_measured', 0)} clip(s) examined), so this day could not "
                            f"be checked" + (f" (decision recorded: {t['decision']})" if t["decision"] else ""))
    ref_dev = None
    if reference_px_per_cm and cohort_median is not None:
        ref_dev = round((cohort_median - reference_px_per_cm) / reference_px_per_cm * 100, 3)
        if abs(ref_dev) > warn_pct:
            warnings.append(f"cohort px/cm {cohort_median:.3f} is {ref_dev:+.2f} % from the reference {reference_px_per_cm:g} "
                            f"(the previous cohort on this rig); above the {'stop' if abs(ref_dev) > stop_pct else 'warning'} "
                            f"level — warning only")
    given_dev = None
    if given_px_per_cm is not None and cohort_median is not None:
        given_dev = round((given_px_per_cm - cohort_median) / cohort_median * 100, 3)
        if abs(given_dev) > warn_pct:
            warnings.append(f"the px/cm given in the cohort config ({given_px_per_cm:g}) is {given_dev:+.2f} % from the "
                            f"measured cohort median ({cohort_median:.3f})")
    state = dict(
        cohort_px_per_cm=cohort_value, cohort_median=round(cohort_median, 4) if cohort_median is not None else None,
        days_measured=len(measured), used_px_per_cm=used,
        source=("given in the cohort config (calibration.px_per_cm)" if given_px_per_cm is not None
                else f"measured: median of {len(measured)} recording day(s)"),
        given_px_per_cm=given_px_per_cm, given_deviation_pct=given_dev,
        reference_px_per_cm=reference_px_per_cm, reference_deviation_pct=ref_dev,
        thresholds=dict(warn_pct=warn_pct, stop_pct=stop_pct, clips_per_day=clips_per_day, on_unmeasured_day=on_unmeasured_day),
        board=board.describe(), warnings=warnings,
        days={str(n): t for n, t in sorted(table.items())},
        hold_days=[n for n, t in sorted(table.items()) if t["use"] == "hold"],
        updated=f"{_dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    _write_report(cdir, state, table, log)
    for w in warnings:
        log.warn(w)
    span = (f"days {min(measured.values()):.3f}..{max(measured.values()):.3f}" if measured else "no day measured")
    log(f"cohort px/cm = {used:g} ({state['source']}; {span})"
        + (f"; HELD: day(s) {', '.join(map(str, state['hold_days']))} — record a decision: "
           f"{CMD} calibrate --decide N=accept_own|accept_cohort|exclude --by NAME" if state["hold_days"] else ""))
    write_manifest(cdir, "lab_calibration",
                   params=dict(board=board.describe(), clips_per_day=clips_per_day, warn_pct=warn_pct, stop_pct=stop_pct,
                               reference_px_per_cm=reference_px_per_cm, given_px_per_cm=given_px_per_cm,
                               on_unmeasured_day=on_unmeasured_day, camera=camera.name),
                   inputs=dict(cohort_dir=str(root), days=sorted(dd)),
                   outputs=[str(cdir / "per_day_px_per_cm.csv"), str(cdir / "px_per_cm.txt"), str(cdir / "CALIBRATION.md")],
                   results=dict(cohort_px_per_cm=cohort_value, used_px_per_cm=used, hold_days=state["hold_days"],
                                per_day={str(n): t.get("px_per_cm") for n, t in sorted(table.items())}, warnings=warnings))
    return state


def _write_report(cdir: Path, state: dict, table: dict, log) -> None:
    with open(cdir / "per_day_px_per_cm.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=DAY_COLS, extrasaction="ignore")
        w.writeheader()
        for n, t in sorted(table.items()):
            w.writerow({k: ("" if t.get(k) is None else t.get(k)) for k in DAY_COLS})
    pxcm = cdir / "px_per_cm.txt"
    if state.get("used_px_per_cm"):
        pxcm.write_text(f"{state['used_px_per_cm']:.2f}\n", encoding="utf-8")
    elif pxcm.exists():
        pxcm.unlink()
    (cdir / "calibration_state.json").write_text(json.dumps(state, indent=1, default=str) + "\n", encoding="utf-8")
    th = state.get("thresholds") or {}
    L = ["# Scale check (px/cm per recording day)", "",
         f"- updated: {state.get('updated', '')}", f"- board: {state.get('board', '')}",
         f"- cohort value used by the analysis: **{state.get('used_px_per_cm')}** ({state.get('source', 'none')})",
         f"- measured cohort median: {state.get('cohort_median')} over {state.get('days_measured', 0)} day(s)",
         f"- reference (previous cohort on this rig): {state.get('reference_px_per_cm')}"
         + (f" -> this cohort is {state['reference_deviation_pct']:+.2f} % from it" if state.get("reference_deviation_pct") is not None else ""),
         f"- levels: warning above {th.get('warn_pct')} %, stop above {th.get('stop_pct')} %; {th.get('clips_per_day')} clip(s) per day", "",
         "| day | folder | px/cm | spread % | clips used | vs cohort % | status | decision | used for the analysis |", "|---|---|---|---|---|---|---|---|---|"]
    for n, t in sorted(table.items()):
        used = {"cohort": f"cohort value {t.get('value_used')}", "own": f"its own value {t.get('value_used')}",
                "hold": "HELD — waiting for a decision", "exclude": "excluded"}.get(t.get("use"), "")
        L.append(f"| DAY{n} | {t.get('folder', '')} | {t.get('px_per_cm') if t.get('px_per_cm') is not None else 'no reading'} | "
                 f"{t.get('spread_pct') if t.get('spread_pct') is not None else ''} | {t.get('clips_used', 0)}/{t.get('clips_measured', 0)} | "
                 f"{t.get('deviation_pct') if t.get('deviation_pct') is not None else ''} | {t.get('status', '')} | {t.get('decision', '')} | {used} |")
    if state.get("warnings"):
        L += ["", "## Look at", ""] + [f"- {w}" for w in state["warnings"]]
    if state.get("hold_days"):
        L += ["", "## Decision needed", "",
              f"Day(s) {', '.join(map(str, state['hold_days']))} are held. Look at the QC images in `day<N>/` (is the board flat and "
              "in the same place? was the camera touched?), then record one of:", "",
              f"    {CMD} calibrate --config <cohort.yaml> --decide <N>=accept_own --by <name> --note \"...\"      # process the day with its own value",
              f"    {CMD} calibrate --config <cohort.yaml> --decide <N>=accept_cohort --by <name> --note \"...\"   # process it with the cohort value",
              f"    {CMD} calibrate --config <cohort.yaml> --decide <N>=exclude --by <name> --note \"...\"         # leave the day out"]
    (cdir / "CALIBRATION.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def load_state(cohort_dir, records_dir=None) -> Optional[dict]:
    f = cal_dir(cohort_dir, records_dir) / "calibration_state.json"
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
