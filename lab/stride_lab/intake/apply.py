"""Stage `rename`: apply a signed-off rename plan. Dry run by default.

Two-phase ATOMIC rename (old -> temporary name -> final name) so a collision or an interruption can never
overwrite a source clip. Sidecar files that share the clip's stem (.slp, .rois.yml, .pred.slp) move with it.
Writes rename_log.csv (old_name, new_name, mouse, trial, timestamp) — the rename is reversible from that file.

All of these must hold before ANY file is touched:
  * the day's BOARD_VERIFICATION.md carries `VERDICT: APPLY` (unless the caller explicitly skips the gate),
  * every clip of the plan exists, no two rows map to the same new name, no new name already exists,
  * the day does not already look renamed (Day*_*_Trial* clips present) unless forced.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from ._log import StageLog, write_manifest
from .boards import signed_off, current_verdict
from .camera import CameraProfile, DJI

SIDECARS = (".slp", ".rois.yml", ".pred.slp")


def _companions(vdir: Path, stem: str) -> list:
    return [suf for suf in SIDECARS if (vdir / f"{stem}{suf}").exists()]


def load_plan(plan_csv) -> list:
    rows = list(csv.DictReader(open(plan_csv)))
    need = {"old_name", "new_name", "mouse", "trial"}
    if not rows or not need.issubset(rows[0].keys()):
        raise ValueError(f"plan is empty or misses the columns {sorted(need)}: {plan_csv}")
    return rows


def apply_rename(plan_csv, videos_dir, verification_md=None, apply: bool = False, force: bool = False,
                 require_signoff: bool = True, log_csv=None, camera: CameraProfile = DJI,
                 log: Optional[StageLog] = None) -> dict:
    """Validate (always) and, with apply=True, rename. Returns {"n", "applied", "log_csv", "errors"}.

    Raises IntakeError when the sign-off is missing or the validation fails.
    """
    log = log or StageLog("rename")
    plan_csv, vdir = Path(plan_csv), Path(videos_dir)
    if not plan_csv.exists() or not plan_csv.stat().st_size:
        log.refuse(f"no rename plan at {plan_csv} (run the boards stage first)")
    if require_signoff:
        vm = Path(verification_md) if verification_md else plan_csv.parent / "BOARD_VERIFICATION.md"
        if not vm.exists() or not vm.stat().st_size:
            log.refuse(f"no {vm} — run the boards stage, then read the boards and sign off")
        if not signed_off(vm):
            log.refuse(f"{vm} has no 'VERDICT: APPLY' line — the boards have not been signed off (current: {current_verdict(vm)})")
    try:
        rows = load_plan(plan_csv)
    except ValueError as e:
        log.refuse(str(e))
    log_path = Path(log_csv) if log_csv else plan_csv.parent / "rename_log.csv"

    errs = []
    already = sorted(p.name for p in camera.renamed_videos(vdir))
    if already and not force:
        errs.append(f"{len(already)} Day*_*_Trial* clips already present (the day looks renamed; force to override), e.g. {already[:3]}")
    seen_new = {}
    for r in rows:
        if r["new_name"] in seen_new:
            errs.append(f"collision: {r['new_name']} claimed by both {seen_new[r['new_name']]} and {r['old_name']}")
        seen_new[r["new_name"]] = r["old_name"]
    old_set = {r["old_name"] for r in rows}
    for r in rows:
        if not (vdir / r["old_name"]).exists():
            errs.append(f"missing source: {r['old_name']}")
        if (vdir / r["new_name"]).exists() and r["new_name"] not in old_set:
            errs.append(f"target exists (not itself a source): {r['new_name']}")
    if errs:
        for e in errs[:20]:
            log(f"  - {e}")
        log.refuse(f"rename plan validation failed with {len(errs)} problem(s); first: {errs[0]}")
    extra = [p.name for p in camera.videos(vdir) if p.name not in old_set and p.name not in seen_new]
    if extra:
        log.warn(f"{len(extra)} clip(s) in {vdir.name} are not in the plan (e.g. {extra[0]}) — extras must be moved to waste "
                 "before the apply, or the inference stages will meet un-renamed clips")
    log(f"plan validated: {len(rows)} clips, 0 collisions, all sources present, {len(already)} already renamed")
    if not apply:
        for r in rows[:5]:
            comp = _companions(vdir, Path(r["old_name"]).stem)
            log(f"  {r['old_name']} -> {r['new_name']}" + (f"  (+{','.join(comp)})" if comp else ""))
        log(f"DRY RUN: {len(rows)} clips would be renamed — re-run with --apply to commit")
        return dict(n=len(rows), applied=False, log_csv=None, errors=[])

    # ---- phase 1: old -> temporary -------------------------------------------------------------
    ts = datetime.now().replace(microsecond=0).isoformat()
    tmp_map = []
    for i, r in enumerate(rows):
        old_stem, new_stem = Path(r["old_name"]).stem, Path(r["new_name"]).stem
        tmp = vdir / f".__rntmp_{i:04d}__{r['new_name']}"
        os.rename(vdir / r["old_name"], tmp)
        moves = [(tmp, vdir / r["new_name"])]
        for suf in _companions(vdir, old_stem):
            t = vdir / f".__rntmp_{i:04d}__{new_stem}{suf}"
            os.rename(vdir / f"{old_stem}{suf}", t)
            moves.append((t, vdir / f"{new_stem}{suf}"))
        tmp_map.append(moves)
    # ---- phase 2: temporary -> final -------------------------------------------------------------
    for moves in tmp_map:
        for tmp, final in moves:
            os.rename(tmp, final)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["old_name", "new_name", "mouse", "trial", "timestamp"])
        for r in rows:
            w.writerow([r["old_name"], r["new_name"], r["mouse"], r["trial"], ts])
    applied_copy = plan_csv.with_name("rename_plan.applied.csv")
    applied_copy.write_text(plan_csv.read_text())
    log(f"APPLIED: {len(rows)} clips renamed (two-phase); log -> {log_path} (the rename is reversible from it)")
    write_manifest(plan_csv.parent, "intake_rename", params=dict(force=force, require_signoff=require_signoff),
                   inputs=dict(plan=str(plan_csv), videos_dir=str(vdir)),
                   outputs=[str(log_path), str(applied_copy)], results=dict(renamed=len(rows), not_in_plan=extra))
    return dict(n=len(rows), applied=True, log_csv=log_path, errors=[])


def move_to_waste(cohort_dir, day_dir, clips, reason: str = "", log: Optional[StageLog] = None) -> list:
    """Move extra / aborted clips (and their sidecars) out of a day folder into `<cohort>/waste/<day folder>/`.

    Kept outside the day folder so no later stage picks them up; every move is appended to
    `<cohort>/waste/waste_log.csv` (reversible).
    """
    log = log or StageLog("waste")
    cohort_dir, day_dir = Path(cohort_dir), Path(day_dir)
    dest = cohort_dir / "waste" / day_dir.name
    moved, ts = [], datetime.now().replace(microsecond=0).isoformat()
    for name in clips:
        src = day_dir / Path(name).name
        if not src.exists():
            log.refuse(f"{src} not found")
    dest.mkdir(parents=True, exist_ok=True)
    for name in clips:
        src = day_dir / Path(name).name
        for p in [src] + [day_dir / f"{src.stem}{suf}" for suf in _companions(day_dir, src.stem)]:
            if (dest / p.name).exists():
                log.refuse(f"{dest / p.name} already exists")
            os.rename(p, dest / p.name)
            moved.append((str(p), str(dest / p.name), reason, ts))
    wl = cohort_dir / "waste" / "waste_log.csv"
    new = not wl.exists()
    with open(wl, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["old_path", "new_path", "reason", "timestamp"])
        w.writerows(moved)
    log(f"{day_dir.name}: moved {len(clips)} clip(s) to {dest}" + (f" — {reason}" if reason else ""))
    return moved
