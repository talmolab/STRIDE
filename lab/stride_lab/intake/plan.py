"""Stage `plan`: recording order x metadata trial counts -> a PROVISIONAL rename plan. Never renames.

The clips of one day, in recording order, are cut into per-mouse blocks by walking the metadata trial counts
in row order. That plan is only a preview: the whiteboard shown at the start of each mouse's block is the
anchor, and the plan is applied only after a person has read the boards (stage `boards` + sign-off).
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from ._log import StageLog, write_manifest
from .camera import CameraProfile, DJI
from .metadata import day_blocks

PLAN_COLS = ["old_name", "new_name", "mouse", "trial", "block_index", "direction",
             "metadata_trials", "video_block_size", "block_size_match", "notes"]


def new_name(day_num, mouse, trial, src_name: str) -> str:
    """Day{N}_{Mouse}_Trial{T} with the source clip's own extension (the lab's clips are .MP4)."""
    return f"Day{day_num}_{mouse}_Trial{trial}{Path(src_name).suffix}"


def build_rename_plan(metadata, day: str, day_num, videos_dir, out_dir, camera: CameraProfile = DJI,
                      log: Optional[StageLog] = None) -> dict:
    """Writes <out_dir>/rename_plan.csv and reconciliation_report.txt.

    Returns {"perfect": bool, "n_videos", "n_trials", "delta", "blocks": [...], "plan_csv", "leftover": [names]}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = log or StageLog("plan")
    meta_blocks, excluded = day_blocks(Path(metadata), day)
    videos = camera.videos(Path(videos_dir))
    n_trials = sum(b[1] for b in meta_blocks)
    rep = []

    def w(line=""):
        rep.append(line)

    w("=" * 70)
    w(f"RENAME MATCH — {day}  (NO FILES RENAMED)")
    w("=" * 70)
    w(f"Metadata: {Path(metadata).name}")
    w(f"Mice with trials : {len(meta_blocks)}")
    w(f"Excluded (0 trial): {len(excluded)}  -> " + ", ".join(f"{m}({nt or 'no note'})" for m, nt in excluded))
    w(f"Expected trials  : {n_trials}")
    w(f"Videos present   : {len(videos)}")
    delta = len(videos) - n_trials
    w(f"Count delta      : {delta:+d}  (videos - expected_trials)")
    w("")
    w("Block-split mode : PROVISIONAL (sequential; read the whiteboards to verify)")
    # walk the metadata trial counts in order to slice the clip list (preview only)
    vblocks, idx = [], 0
    for (_m, n, _d, _nt) in meta_blocks:
        vblocks.append(videos[idx:idx + n])
        idx += n
    leftover = videos[idx:]
    w("")
    w(f"{'#':<3}{'mouse':<8}{'meta_n':>7}{'video_n':>8}{'match':>7}   first_clip -> last_clip")
    w("-" * 70)
    plan_rows, blocks, all_match = [], [], (delta == 0)
    for i, (mouse, n, dirs, notes) in enumerate(meta_blocks):
        vb = vblocks[i] if i < len(vblocks) else []
        ok = (len(vb) == n)
        all_match = all_match and ok
        first = camera.seq_of(vb[0].name) if vb else "----"
        last = camera.seq_of(vb[-1].name) if vb else "----"
        flag = "OK" if ok else f"!! {len(vb)}≠{n}"
        w(f"{i+1:<3}{mouse:<8}{n:>7}{len(vb):>8}{flag:>9}   {first} -> {last}   {notes[:24]}")
        blocks.append(dict(block_index=i + 1, mouse=mouse, metadata_trials=n, video_block_size=len(vb), match=ok,
                           first=vb[0].name if vb else "", notes=notes))
        for t, v in enumerate(vb, 1):
            plan_rows.append({
                "old_name": v.name, "new_name": new_name(day_num, mouse, t, v.name),
                "mouse": mouse, "trial": t, "block_index": i + 1,
                "direction": dirs[t - 1] if t - 1 < len(dirs) else "",
                "metadata_trials": n, "video_block_size": len(vb), "block_size_match": ok, "notes": notes,
            })
    if leftover:
        w(f"   LEFTOVER (provisional): {len(leftover)} clip(s) past the last metadata block "
          f"-> {', '.join(camera.seq_of(v.name) for v in leftover)}")
        all_match = False
    w("")
    w("VERDICT: " + ("PERFECT MATCH ✓" if all_match else "MISMATCH — see flags above ✗"))
    plan_csv = out_dir / "rename_plan.csv"
    with open(plan_csv, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=PLAN_COLS)
        wtr.writeheader()
        wtr.writerows(plan_rows)
    (out_dir / "reconciliation_report.txt").write_text("\n".join(rep) + "\n")
    log(f"{day}: {len(videos)} clips vs {n_trials} metadata trials / {len(meta_blocks)} mice (delta {delta:+d}) -> "
        f"{'PERFECT MATCH' if all_match else 'MISMATCH'}; plan {plan_csv} ({len(plan_rows)} rows)")
    for line in rep:
        if "!!" in line or "LEFTOVER" in line:
            log("  " + line.strip())
    write_manifest(out_dir, "intake_plan", params=dict(day=day, day_num=str(day_num), camera=camera.name),
                   inputs=dict(metadata=str(metadata), videos_dir=str(videos_dir), n_videos=len(videos)),
                   outputs=[str(plan_csv), str(out_dir / "reconciliation_report.txt")],
                   results=dict(perfect=all_match, n_trials=n_trials, delta=delta, n_mice=len(meta_blocks)))
    return dict(perfect=all_match, n_videos=len(videos), n_trials=n_trials, delta=delta, blocks=blocks,
                plan_csv=plan_csv, leftover=[v.name for v in leftover], excluded=excluded, report=rep)
