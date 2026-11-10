"""The staged flow over a cohort folder, safe to re-run:

    check  ->  sort  ->  boards (plan + whiteboard crops + montage)  ->  STOP until a person signs
    BOARD_VERIFICATION.md  ->  rename  ->  calibrate

Layout (the one the inference and analysis stages expect):

    <cohort>/                       flat raw clips + the recorder export CSVs (before `sort`)
    <cohort>/<DATE>-DAY{N}/         one folder per recording day; `metadata/` holds the export CSVs
    <cohort>/waste/<DATE>-DAY{N}/   extra / aborted clips taken out of a day
    <records>/day{N}/               rename_plan.csv, wb/, wb_montage.png, BOARD_VERIFICATION.md, rename_log.csv
    <records>/calibration/          per-clip readings, QC images, calibration_summary.txt, px_per_cm.txt
    <records>/intake.log           one line per stage event

`<records>` defaults to `<cohort>/_intake`.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

from stride.stages.scale_calibration import BoardSpec, DEFAULT_BOARD, CalibrationError, calibrate_scale, pick_clips

from ._log import CMD, StageLog, IntakeError, write_manifest
from .apply import apply_rename
from .boards import boards_stage, signed_off, current_verdict, fill_verification
from .camera import CameraProfile, DJI
from .metadata import day_blocks, placeholder_ids, trials_full_for_day, newest_trials_full
from .plan import build_rename_plan
from .sort import sort_days

DAY_DIR = re.compile(r"(?:^|-)DAY(\d+)$")          # <DATE>-DAY{N}; a bare DAY{N} (older cohorts) is accepted too


def records_dir_of(cohort_dir, records_dir=None) -> Path:
    return Path(records_dir) if records_dir else Path(cohort_dir) / "_intake"


def day_dirs(cohort_dir) -> dict:
    """{day number: folder} for every `<DATE>-DAY{N}` folder of the cohort."""
    out = {}
    root = Path(cohort_dir)
    if root.is_dir():
        for p in sorted(root.iterdir()):
            m = DAY_DIR.search(p.name)
            if p.is_dir() and m:
                out.setdefault(int(m.group(1)), p)
    return dict(sorted(out.items()))


def resolve_days(cohort_dir, days=None, include_day0: bool = False) -> list:
    """Day numbers to work on. `days`: "1,2" | "1-3" | "DAY1 DAY2" | "2026-10-14-DAY1"; default every DAY >= 1 on disk."""
    have = day_dirs(cohort_dir)
    if not days:
        return [n for n in have if n >= 1 or include_day0]
    sel = []
    for tok in re.split(r"[,\s]+", str(days).strip()):
        if not tok:
            continue
        m = re.fullmatch(r".*-DAY(\d+)", tok) or re.fullmatch(r"(?i)DAY(\d+)", tok)
        if m:
            sel.append(int(m.group(1)))
        elif re.fullmatch(r"\d+-\d+", tok):
            a, b = (int(x) for x in tok.split("-"))
            sel.extend(range(a, b + 1))
        elif re.fullmatch(r"\d+", tok):
            sel.append(int(tok))
        else:
            raise IntakeError(f"bad --days token: {tok!r} (use 1,2 | 1-3 | DAY1 | 2026-10-14-DAY1)")
    return sorted(set(sel))


def _need_days(cohort_dir, days, log, include_day0=False) -> dict:
    have = day_dirs(cohort_dir)
    nums = resolve_days(cohort_dir, days, include_day0)
    if not nums:
        log.refuse(f"no day folders to process under {cohort_dir} (run the sort stage first, or name the days)")
    for n in nums:
        if n not in have:
            log.refuse(f"no folder for DAY{n} under {cohort_dir} (run the sort stage first)")
    return {n: have[n] for n in nums}


def plan_is_current(plan_csv, day_dir, camera: CameraProfile = DJI) -> bool:
    """True when the plan's source clips are exactly the un-renamed clips now in the day folder."""
    if not Path(plan_csv).exists():
        return False
    planned = {r["old_name"] for r in csv.DictReader(open(plan_csv))}
    return planned == {p.name for p in camera.videos(day_dir)}


def _renamed_state(day_dir, camera) -> tuple:
    vids = camera.videos(day_dir)
    ren = camera.renamed_videos(day_dir)
    return len(vids), len(vids) - len(ren), len(ren)          # clips, not yet renamed, renamed


# ----------------------------------------------------------------------------------------------------- check
def check(cohort_dir, records_dir=None, camera: CameraProfile = DJI, quiet: bool = False) -> dict:
    """What exists per day and what the next step is. Writes nothing."""
    root, rec = Path(cohort_dir), records_dir_of(cohort_dir, records_dir)
    out = dict(cohort=str(root), exists=root.is_dir(), flat_clips=0, days=[], px_per_cm=None, trials_full=None)

    def say(m):
        if not quiet:
            print(m, flush=True)

    say(f"== intake check @ {root} ==")
    if not root.is_dir():
        say("  folder MISSING")
        return out
    flat = camera.raw_videos(root)
    out["flat_clips"] = len(flat)
    tf = newest_trials_full(root)
    out["trials_full"] = str(tf) if tf else None
    say(f"  flat raw clips (unsorted): {len(flat)}")
    say(f"  recorder export in the cohort folder: {tf.name if tf else 'none'}")
    pxcm = rec / "calibration" / "px_per_cm.txt"
    if pxcm.exists():
        out["px_per_cm"] = float(pxcm.read_text().strip())
    say("  calibration: " + (f"px/cm {out['px_per_cm']} ({pxcm})" if out["px_per_cm"] else "NOT calibrated (run the calibrate stage once clips are renamed)"))
    days = day_dirs(root)
    if not days:
        say("  day folders: none (run the sort stage after the clips and the export are in the folder)")
    else:
        say(f"  {'day':6} {'folder':22} {'clips':>5} {'raw':>4} {'ren':>4} {'meta':>5} {'plan':>4} {'wb':>3} {'mont':>4} {'sign':>5}  next")
    for n, d in days.items():
        n_clips, n_raw, n_ren = _renamed_state(d, camera)
        D = rec / f"day{n}"
        meta = trials_full_for_day(d, root)
        mt, ph = 0, False
        if meta is not None:
            try:
                blocks, _ = day_blocks(meta, f"DAY{n}")
                mt, ph = sum(b[1] for b in blocks), placeholder_ids(blocks)
            except IntakeError:
                pass
        plan = (D / "rename_plan.csv").exists()
        wb = len(list((D / "wb").glob("*_wb.png"))) if (D / "wb").is_dir() else 0
        mont = (D / "wb_montage.png").exists()
        sign = signed_off(D / "BOARD_VERIFICATION.md")
        if n == 0:
            nxt = "DAY0 (preference phase): sorted only, not renamed unless asked for"
        elif not n_clips:
            nxt = "empty folder"
        elif n_raw and not plan:
            nxt = ("boards (ids look like recorder placeholders: enter the real ids and re-export, or --allow-placeholders)" if ph
                   else ("boards" if mt == n_clips else f"count mismatch ({n_clips} clips vs {mt} trials): finish the transfer / move the extra to waste, then boards"))
        elif n_raw and plan and not sign:
            nxt = f"PERSON: read wb_montage.png -> BOARD_VERIFICATION.md 'VERDICT: APPLY' ({CMD} verify)"
        elif n_raw and sign:
            nxt = "rename --apply"
        elif out["px_per_cm"] is None:
            nxt = "calibrate"
        else:
            nxt = "ready for inference"
        row = dict(day=n, folder=d.name, clips=n_clips, raw=n_raw, renamed=n_ren, metadata_trials=mt, plan=plan, crops=wb,
                   montage=mont, signed=sign, next=nxt)
        out["days"].append(row)
        say(f"  DAY{n:<3} {d.name:22} {n_clips:5d} {n_raw:4d} {n_ren:4d} {mt:5d} {'yes' if plan else 'no':>4} {wb:3d} "
            f"{'yes' if mont else 'no':>4} {'APPLY' if sign else 'no':>5}  {nxt}")
    return out


# ---------------------------------------------------------------------------------------------------- boards
def boards_days(cohort_dir, records_dir=None, days=None, allow_placeholders: bool = False, sweep_all: bool = False,
                whole_clip: bool = False, include_day0: bool = False, camera: CameraProfile = DJI, workers: int = 1,
                title_prefix: str = "", force: bool = False, log: Optional[StageLog] = None) -> dict:
    """Plan + whiteboard crops + montage + verification template for each un-renamed day. Never renames."""
    root, rec = Path(cohort_dir), records_dir_of(cohort_dir, records_dir)
    log = log or StageLog("boards", rec / "intake.log")
    out = {}
    for n, d in _need_days(root, days, log, include_day0).items():
        if n < 1 and not include_day0:
            log(f"DAY{n} (preference phase) is not renamed — skipping {d.name}")
            continue
        n_clips, n_raw, n_ren = _renamed_state(d, camera)
        if n_ren > 0 and not force:
            log(f"DAY{n} already renamed ({n_ren} clips) — skip")
            out[n] = dict(status="renamed")
            continue
        meta = trials_full_for_day(d, root)
        if meta is None:
            log.refuse(f"DAY{n}: no *_trials_full.csv (looked in {d / 'metadata'} and {root})")
        blocks, _ = day_blocks(meta, f"DAY{n}")
        log(f"DAY{n}: {n_clips} clips vs {sum(b[1] for b in blocks)} metadata trials / {len(blocks)} mice ({meta.name}): "
            + " ".join(f"{b[0]}x{b[1]}" for b in blocks))
        if placeholder_ids(blocks) and not allow_placeholders:
            log.refuse(f"DAY{n}: every mouse id in the export is a 1-2 digit number, which is what the recorder writes before "
                       f"the real ids are typed in — plans would read Day{n}_1_Trial1. Enter the ids and re-export, or pass "
                       "--allow-placeholders if these numbers ARE the ids shown on the whiteboards")
        D = rec / f"day{n}"
        vmd = D / "BOARD_VERIFICATION.md"
        if signed_off(vmd) and plan_is_current(D / "rename_plan.csv", d, camera) and not force:
            log(f"DAY{n}: boards already signed off (VERDICT: APPLY) and the plan matches the clips on disk — skip")
            out[n] = dict(status="signed")
            continue
        if signed_off(vmd) and not plan_is_current(D / "rename_plan.csv", d, camera):
            log.refuse(f"DAY{n}: {vmd} is signed off, but the clips in {d.name} are no longer the ones in the plan — "
                       "check what changed; delete the day's records folder to start that day again")
        if vmd.exists() and not plan_is_current(D / "rename_plan.csv", d, camera):
            vmd.unlink()                                          # unsigned and stale: the table is rebuilt below
        p = build_rename_plan(meta, f"DAY{n}", n, d, D, camera=camera, log=log)
        if not p["plan_csv"].exists() or p["n_videos"] == 0:
            log.refuse(f"DAY{n}: no rename plan produced (no clips in {d})")
        title = f"{title_prefix or root.name} — DAY{n} ({d.name})"
        b = boards_stage(d, D, meta, f"DAY{n}", n, title, p["perfect"], sweep_all=sweep_all, whole_clip=whole_clip,
                         camera=camera, workers=workers, log=log)
        log(f"DAY{n}: STOP — read the boards: open {D / 'wb_montage.png'} (crops in {D / 'wb'}), fill {vmd} and set "
            f"'VERDICT: APPLY' ({CMD} verify --days {n} ...), then: {CMD} rename --days {n} --apply")
        out[n] = dict(status="pending", perfect=p["perfect"], crops=b["crops"], candidates=b["candidates"], blocks=p["blocks"])
    return out


# ---------------------------------------------------------------------------------------------------- rename
def rename_days(cohort_dir, records_dir=None, days=None, apply: bool = False, force: bool = False,
                include_day0: bool = False, camera: CameraProfile = DJI, log: Optional[StageLog] = None) -> dict:
    """Validate (dry run) and, with apply, rename each signed-off day."""
    root, rec = Path(cohort_dir), records_dir_of(cohort_dir, records_dir)
    log = log or StageLog("rename", rec / "intake.log")
    out = {}
    for n, d in _need_days(root, days, log, include_day0).items():
        if n < 1 and not include_day0:
            continue
        n_clips, n_raw, n_ren = _renamed_state(d, camera)
        if n_raw == 0 and n_ren > 0:
            log(f"DAY{n} already renamed — skip")
            out[n] = dict(applied=False, status="renamed")
            continue
        D = rec / f"day{n}"
        r = apply_rename(D / "rename_plan.csv", d, verification_md=D / "BOARD_VERIFICATION.md", apply=apply, force=force,
                         camera=camera, log=log)
        if apply:
            _c, left, ren = _renamed_state(d, camera)
            log(f"DAY{n}: renamed {ren} clips ({left} not renamed left in the folder)")
        else:
            log(f"DAY{n}: dry run OK — re-run with --apply to commit (this is the explicit apply step)")
        out[n] = dict(applied=r["applied"], n=r["n"], status="renamed" if r["applied"] else "validated")
    return out


# ------------------------------------------------------------------------------------------------- calibrate
def calibrate(cohort_dir, records_dir=None, days=None, clips=None, board: BoardSpec = DEFAULT_BOARD, camera: CameraProfile = DJI,
              workers: int = 1, include_day0: bool = False, log: Optional[StageLog] = None) -> dict:
    """In-view ChArUco px/cm for the cohort -> <records>/calibration/px_per_cm.txt (the public stride.stages.scale_calibration)."""
    root, rec = Path(cohort_dir), records_dir_of(cohort_dir, records_dir)
    log = log or StageLog("calibrate", rec / "intake.log")
    if clips:
        chosen = [Path(c) for c in clips]
        missing = [c for c in chosen if not c.exists()]
        if missing:
            log.refuse(f"clip not found: {missing[0]}")
    else:
        dd = _need_days(root, days, log, include_day0)
        chosen = pick_clips([camera.videos(d) for n, d in dd.items() if n >= 1 or include_day0])   # first clip of each day
        if not chosen:
            log.refuse("no clips in the day folders to calibrate on")
    try:
        r = calibrate_scale(chosen, rec / "calibration", board=board, workers=workers, rail_crop=camera.rail_crop,
                            rail_frame_size=camera.frame_size, log=log)
    except CalibrationError as e:
        log.refuse(str(e))
    return r


# ------------------------------------------------------------------------------------------------------ auto
def auto(cohort_dir, records_dir=None, days=None, apply: bool = False, fixed_run_order: bool = False, verifier: str = "",
         allow_placeholders: bool = False, allow_mismatch: bool = False, sweep_all: bool = False, whole_clip: bool = False,
         board: BoardSpec = DEFAULT_BOARD, camera: CameraProfile = DJI, workers: int = 1, skip_calibration: bool = False,
         log: Optional[StageLog] = None) -> dict:
    """sort -> boards -> (stop until signed) -> rename -> calibrate. Re-run it until it reports `complete`.

    Without apply=True nothing is moved or renamed (sort and rename run as dry runs).
    Returns {"status": "complete" | "waiting_for_signoff" | "dry_run", "pending_days": [...], "px_per_cm": ...}.
    """
    root, rec = Path(cohort_dir), records_dir_of(cohort_dir, records_dir)
    log = log or StageLog("auto", rec / "intake.log")
    res = dict(status="complete", pending_days=[], px_per_cm=None)

    def stage(name):
        log.stage = name
        return log

    # ---- sort ----------------------------------------------------------------------------------------
    if camera.raw_videos(root):
        s = sort_days(root, apply=apply, allow_mismatch=allow_mismatch, camera=camera, log=stage("sort"))
        if not apply:
            stage("auto")("dry run: the clips are still flat in the cohort folder — re-run with --apply to sort them into day folders")
            res["status"] = "dry_run"
            return res
    else:
        stage("sort")("no flat raw clips in the cohort folder — already sorted")
    dd = _need_days(root, days, stage("auto"))
    # ---- boards / sign-off -----------------------------------------------------------------------------
    todo = [n for n, d in dd.items() if n >= 1 and not (_renamed_state(d, camera)[1] == 0 and _renamed_state(d, camera)[2] > 0)]
    waiting = []
    if todo:
        b = boards_days(root, rec, ",".join(map(str, todo)), allow_placeholders=allow_placeholders, sweep_all=sweep_all,
                        whole_clip=whole_clip, camera=camera, workers=workers, log=stage("boards"))
        for n in todo:
            vmd = rec / f"day{n}" / "BOARD_VERIFICATION.md"
            if signed_off(vmd):
                continue
            st = b.get(n, {})
            if fixed_run_order and st.get("status") == "pending":
                starts_without_crop = _block_starts_without_crop(rec / f"day{n}")
                if st.get("perfect") and all(k == 1 for k in starts_without_crop):
                    stage("verify")(f"DAY{n}: !! FIXED-RUN-ORDER SKIP !! the board NUMBERS are NOT read. Signing off because the "
                                    "counts reconcile exactly and a whiteboard was found at every block start"
                                    + (" except the first mouse of the day" if starts_without_crop else "")
                                    + ". This is valid ONLY if the mice are always run in the metadata order.")
                    fill_verification(vmd, verdict="APPLY", verifier=verifier or "intake auto --fixed-run-order",
                                      numbers_read=False, log=log)
                    continue
                stage("verify")(f"DAY{n}: fixed-run-order skip NOT possible ("
                                + ("counts do not reconcile" if not st.get("perfect") else f"no whiteboard found at the start of block(s) {starts_without_crop}")
                                + ") — a person must read the boards")
            waiting.append(n)
    if waiting:
        res.update(status="waiting_for_signoff", pending_days=waiting)
        stage("auto")(f"STOP: day(s) {', '.join(map(str, waiting))} need the whiteboards read and 'VERDICT: APPLY' in "
                      f"{rec}/day<N>/BOARD_VERIFICATION.md. Nothing is renamed before that. Re-run this command afterwards.")
        return res
    # ---- rename ----------------------------------------------------------------------------------------
    if not todo:
        stage("rename")(f"day(s) {', '.join(map(str, dd))} already renamed — skip")
    if todo:
        rename_days(root, rec, ",".join(map(str, todo)), apply=apply, camera=camera, log=stage("rename"))
        if not apply:
            stage("auto")("dry run: the signed-off days were validated but not renamed — re-run with --apply")
            res["status"] = "dry_run"
            return res
    # ---- calibrate ---------------------------------------------------------------------------------------
    pxcm = rec / "calibration" / "px_per_cm.txt"
    if skip_calibration:
        stage("calibrate")("skipped (--skip-calibration)")
    elif pxcm.exists() and pxcm.read_text().strip():
        res["px_per_cm"] = float(pxcm.read_text().strip())
        stage("calibrate")(f"calibration present ({res['px_per_cm']}) — skip")
    else:
        c = calibrate(root, rec, board=board, camera=camera, workers=workers, log=stage("calibrate"))
        res["px_per_cm"] = c["px_per_cm"]
    stage("auto")(f"INTAKE COMPLETE for day(s) {', '.join(map(str, dd))}"
                  + (f"; px/cm = {res['px_per_cm']}" if res["px_per_cm"] else ""))
    write_manifest(rec, "intake_auto", params=dict(apply=apply, fixed_run_order=fixed_run_order, board=board.describe(), camera=camera.name),
                   inputs=dict(cohort_dir=str(root)), outputs=[str(pxcm)] if pxcm.exists() else [],
                   results=dict(status=res["status"], days=list(dd), px_per_cm=res["px_per_cm"]))
    return res


def _block_starts_without_crop(day_records) -> list:
    """Block numbers whose first clip produced no whiteboard crop."""
    D = Path(day_records)
    out, seen = [], set()
    for r in csv.DictReader(open(D / "rename_plan.csv")):
        if r["block_index"] in seen:
            continue
        seen.add(r["block_index"])
        if not (D / "wb" / f"{Path(r['old_name']).stem}_wb.png").exists():
            out.append(int(r["block_index"]))
    return out
