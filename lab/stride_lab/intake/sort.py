"""Stage `sort`: a flat folder of raw clips + the recorder export -> `<DATE>-DAY{N}/` folders.

The DATE comes from the raw clip name, the DAY and the expected number of clips from the export (non-empty
T1..T9 cells per mouse row). Clips keep their camera names; the export CSVs are copied into each day's
`metadata/`. Nothing moves unless every day reconciles exactly (clips == metadata trials) or the caller
overrides that explicitly. Dry run by default.
"""
from __future__ import annotations

import csv
import datetime as dt
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .apply import SIDECARS
from ._log import StageLog, IntakeError, write_manifest
from .camera import CameraProfile, DJI
from .metadata import newest_trials_full, read_trials_full


def sort_days(cohort_dir, trials_full=None, apply: bool = False, allow_mismatch: bool = False,
              camera: CameraProfile = DJI, log: Optional[StageLog] = None) -> dict:
    """Returns {"rows": [...per day...], "mismatches": [...], "warnings": [...], "moved": n, "applied": bool}.

    Raises IntakeError when there is no export, or on --apply with unreconciled days (unless allow_mismatch).
    """
    root = Path(cohort_dir)
    log = log or StageLog("sort")
    if not root.is_dir():
        log.refuse(f"cohort folder not found: {root}")
    csvs = sorted(c for c in root.glob("*.csv") if c.name not in ("sort_log.csv",))
    tf = Path(trials_full) if trials_full else newest_trials_full(root)
    if tf is None or not tf.exists():
        log.refuse(f"no *_trials_full.csv in {root} — put the recorder export next to the clips (or pass --trials-full)")
    day_date, trials, mice = read_trials_full(tf)
    date_day = {}
    for d, date in day_date.items():
        date_day.setdefault(date.replace("-", ""), d)
    clips = camera.raw_videos(root)
    by_date, warnings = defaultdict(list), []
    for p in clips:
        d8 = camera.date_of(p.name)
        mtime = dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y%m%d")
        if mtime != d8:
            warnings.append(f"{p.name}: filename date {d8} != mtime date {mtime}")
        by_date[d8].append(p)
    rows, bad = [], []
    for d8 in sorted(by_date):
        if d8 not in date_day:
            bad.append(f"{d8}: {len(by_date[d8])} clips but no metadata day for this date")
            continue
        day, n_clips, n_tr = date_day[d8], len(by_date[d8]), trials.get(date_day[d8], 0)
        date = f"{d8[:4]}-{d8[4:6]}-{d8[6:]}"
        rows.append(dict(date=date, day=day, folder=f"{date}-{day}", n_clips=n_clips, n_trials=n_tr,
                         n_mice=len(mice.get(day, [])), delta=n_clips - n_tr,
                         seq=f"{camera.seq_of(by_date[d8][0].name)}..{camera.seq_of(by_date[d8][-1].name)}"))
        if n_clips != n_tr:
            bad.append(f"{day} ({d8}): {n_clips} clips vs {n_tr} metadata trials (delta {n_clips - n_tr:+d})")
    for day in day_date:
        if trials.get(day, 0) and day_date[day].replace("-", "") not in by_date:
            warnings.append(f"{day} ({day_date[day]}): {trials[day]} metadata trials but NO clips on disk")

    log(f"cohort {root} | metadata {tf.name} | {len(clips)} flat clips | metadata trials on the days with clips: {sum(r['n_trials'] for r in rows)}")
    for r in rows:
        log(f"  {r['folder']:24s} clips {r['n_clips']:4d}  trials {r['n_trials']:4d}  mice {r['n_mice']:3d}  delta {r['delta']:+d}  seq {r['seq']}")
    date_warn = [w for w in warnings if "!= mtime date" in w]
    if date_warn:                                                  # one line, not one per clip (the report lists them all)
        log.warn(f"{len(date_warn)} clip(s): the date in the file name differs from the file's modification date, e.g. {date_warn[0]}")
    for w in warnings:
        if w not in date_warn:
            log.warn(w)
    for b in bad:
        log(f"MISMATCH: {b}")
    result = dict(rows=rows, mismatches=bad, warnings=warnings, moved=0, applied=False, trials_full=str(tf))
    if not clips:
        log("no flat raw clips in the cohort folder — nothing to sort")
        return result
    if not apply:
        log("dry run — nothing moved (add --apply)")
        return result
    if bad and not allow_mismatch:
        log.refuse("a date on disk does not reconcile with the metadata trial count: finish that day's transfer or resolve "
                   "the extra / missing clips first (--allow-mismatch overrides; never use it blindly)")
    moved, n_side = [], 0
    for r in rows:
        dest = root / r["folder"]
        (dest / "metadata").mkdir(parents=True, exist_ok=True)
        for c in csvs:
            shutil.copy2(c, dest / "metadata" / c.name)
        for p in by_date[r["date"].replace("-", "")]:
            target = dest / p.name
            if target.exists():
                log.refuse(f"target exists: {target}")
            p.rename(target)                                   # same filesystem: a rename, no copy
            moved.append((str(p), str(target)))
            for suf in SIDECARS:                               # a pose / ROI file made before the sort follows its clip
                side = p.parent / f"{p.stem}{suf}"
                if side.exists() and not (dest / side.name).exists():
                    side.rename(dest / side.name)
                    n_side += 1
    with open(root / "sort_log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_path", "new_path"])
        w.writerows(moved)
    with open(root / "SORT_REPORT.md", "w") as f:
        f.write(f"# {root.name} — raw clips sorted into day folders ({dt.date.today()})\n\n")
        f.write("Source of the DATE: the raw clip name. Source of DAY and of the expected clip count: "
                f"`{tf.name}` (non-empty T1..T9 cells per mouse row). Clips keep their camera names — the whiteboard-verified "
                "rename is the next step. The recorder CSVs were copied into every day's `metadata/`.\n\n")
        f.write("| folder | clips | metadata trials | mice run | delta | clip numbers |\n|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['folder']} | {r['n_clips']} | {r['n_trials']} | {r['n_mice']} | {r['delta']:+d} | {r['seq']} |\n")
        f.write(f"\nTotal {len(moved)} clips moved. " + ("Notes: " + "; ".join(warnings + bad) if (warnings or bad) else "No warnings.") + "\n")
        f.write("\nPer day, mice run (metadata row order) and trial counts:\n")
        for r in rows:
            f.write(f"- {r['day']}: " + ", ".join(f"{m}×{n}" for m, n in mice[r["day"]]) + "\n")
    log(f"moved {len(moved)} clips into {len(rows)} day folders" + (f" (+ {n_side} pose / ROI sidecar files)" if n_side else "")
        + "; wrote sort_log.csv and SORT_REPORT.md")
    result.update(moved=len(moved), applied=True)
    write_manifest(root, "intake_sort", params=dict(allow_mismatch=allow_mismatch, camera=camera.name, raw_pattern=camera.raw_pattern),
                   inputs=dict(cohort_dir=str(root), trials_full=str(tf), n_flat_clips=len(clips)),
                   outputs=[str(root / "sort_log.csv"), str(root / "SORT_REPORT.md")],
                   results=dict(days=len(rows), moved=len(moved), mismatches=bad, warnings=warnings))
    return result
