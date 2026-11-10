"""Stage `boards`: find the whiteboard held up at the start of each mouse's block, for a person to read.

The whiteboard number is the only anchor that binds a block of clips to a mouse (counts can agree by
coincidence, and the run order can differ from the metadata order). This stage does NOT read the number: it
finds the board (a transient, bright, smooth region), saves a readable crop per candidate clip and one montage
per day, and writes a BOARD_VERIFICATION.md table. A person (or a supervising agent) reads every number, fills
the table and sets `VERDICT: APPLY`; only then will `rename --apply` run.

Detector = the lab's transient detector (find_whiteboard_sweep3/4): frames sampled every 12, down-scaled to
960x540; background = median of ~9 sampled frames; only brightness that is NEW against the background is kept
(threshold 45), so the lit floor panel and a taped ChArUco board vanish; a candidate region must cover >= 1500
px, fill >= 50 % of its bounding box and have a Canny edge density <= 10 % (rejects glare and grids);
score = area fraction x fill x (10 - edge %) / 10. The best region of the clip is saved, turned upright.
"""
from __future__ import annotations

import csv
import datetime as dt
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from ._log import CMD, StageLog, IntakeError, write_manifest
from .camera import CameraProfile, DJI
from .metadata import day_blocks
from .plan import new_name, PLAN_COLS

VERDICT_APPLY = re.compile(r"^VERDICT:\s*APPLY", re.M)
_ROW_PENDING = re.compile(r"^\| (\d+) \| (\S+) \| (\S+) \| (\d+) \| ____ \|\s*\|(.*)$")
_ROTATE = {"ccw90": "ROTATE_90_COUNTERCLOCKWISE", "cw90": "ROTATE_90_CLOCKWISE", "180": "ROTATE_180"}


# ------------------------------------------------------------------------------------------------ detector
def sweep_clip(clip, out_dir, sample_every: int = 12, max_seconds: Optional[float] = 12.0,
               crop_rotation: str = "ccw90") -> dict:
    """Best whiteboard-like transient in one clip -> {"clip", "score", "frame"}; writes <stem>_wb.png if found."""
    import cv2
    import numpy as np

    clip, out_dir = Path(clip), Path(out_dir)
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 120.0
    max_frames = int(fps * max_seconds) if max_seconds else 10 ** 9      # first N s only (fast), or the whole clip
    frames, fi = [], 0                                                    # (frame index, full-resolution frame)
    while fi < max_frames:
        if not cap.grab():
            break
        if fi % sample_every == 0:
            ok, fr = cap.retrieve()
            if ok:
                frames.append((fi, fr))
        fi += 1
    cap.release()
    if not frames:
        return {"clip": clip.name, "score": 0, "frame": -1}
    grays = [cv2.cvtColor(cv2.resize(f, (960, 540)), cv2.COLOR_BGR2GRAY) for _, f in frames]
    bg = np.median(np.stack(grays[::max(1, len(grays) // 9)]), 0).astype(np.uint8)
    best = (0.0, None, -1)
    for (fi, fr), gray in zip(frames, grays):
        diff = cv2.subtract(gray, bg)                                     # only NEW brightness
        _, thr = cv2.threshold(diff, 45, 255, cv2.THRESH_BINARY)
        thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 1500:
                continue
            x, y, w, h = cv2.boundingRect(c)
            fill = area / (w * h + 1e-9)
            if fill < 0.5:
                continue
            roi = gray[y:y + h, x:x + w]
            ed = 100.0 * float((cv2.Canny(roi, 60, 140) > 0).mean())
            if ed > 10.0:
                continue
            score = (area / (960 * 540)) * fill * max(0.0, (10.0 - ed)) / 10.0
            if score > best[0]:
                sx, sy = fr.shape[1] / 960, fr.shape[0] / 540
                X, Y, W, H = int(x * sx), int(y * sy), int(w * sx), int(h * sy)
                pad = int(0.3 * max(W, H))
                crop = fr[max(Y - pad, 0):min(Y + H + pad, fr.shape[0]), max(X - pad, 0):min(X + W + pad, fr.shape[1])]
                best = (score, crop, fi)
    if best[1] is not None:
        big = best[1] if crop_rotation == "none" else cv2.rotate(best[1], getattr(cv2, _ROTATE[crop_rotation]))
        big = cv2.resize(big, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / f"{clip.stem}_wb.png"), big)
    return {"clip": clip.name, "score": round(best[0], 4), "frame": best[2]}


def _sweep_star(a):
    return sweep_clip(*a)


def board_candidates(plan_csv, videos_dir, sweep_all: bool = False, camera: CameraProfile = DJI) -> list:
    """Clips to search: every block start (trial 1) + any clip whose metadata note mentions the board;
    with sweep_all, every clip of the day (slow; for days whose counts do not reconcile)."""
    videos_dir = Path(videos_dir)
    if sweep_all:
        return camera.videos(videos_dir)
    pick = []
    for r in csv.DictReader(open(plan_csv)):
        if r["trial"] == "1" or "board" in (r.get("notes") or "").lower():
            pick.append(videos_dir / r["old_name"])
    return list(dict.fromkeys(pick))


def find_whiteboards(clips, out_dir, sample_every: int = 12, max_seconds: Optional[float] = 12.0,
                     camera: CameraProfile = DJI, workers: int = 1, log: Optional[StageLog] = None) -> list:
    """Sweep the candidate clips -> <out_dir>/<stem>_wb.png crops + whiteboard_sweep.csv. Returns the rows."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = log or StageLog("boards")
    jobs = [(str(c), str(out_dir), sample_every, max_seconds, camera.crop_rotation) for c in clips]
    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as ex:     # OpenCV releases the GIL while decoding
            rows = list(ex.map(_sweep_star, jobs))
    else:
        rows = [_sweep_star(j) for j in jobs]
    with open(out_dir / "whiteboard_sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "score", "frame"])
        w.writeheader()
        w.writerows(rows)
    found = sum(1 for r in rows if r["frame"] >= 0 and r["score"] > 0)
    log(f"whiteboard sweep: {len(rows)} candidate clips, crop found in {found} -> {out_dir}")
    return rows


# ------------------------------------------------------------------------------------------------- montage
def make_montage(wb_dir, plan_csv, out_png, camera: CameraProfile = DJI) -> int:
    """Tile the crops into one image per day, each captioned `clip <number>  expected <mouse> T<trial>`."""
    import cv2
    import numpy as np

    wb_dir = Path(wb_dir)
    prov = {Path(r["old_name"]).stem: (r["old_name"], r["mouse"], r["trial"]) for r in csv.DictReader(open(plan_csv))}
    cells = []
    for p in sorted(wb_dir.glob("*_wb.png")):
        stem = p.name[:-7]
        old_name, mouse, trial = prov.get(stem, (stem, "?", "?"))
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = min(520 / h, 420 / w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        cell = np.full((560, 440, 3), 25, np.uint8)
        cell[20:20 + img.shape[0], 10:10 + img.shape[1]] = img
        cv2.putText(cell, f"clip {camera.seq_of(old_name)[-14:]}  expected {mouse} T{trial}",
                    (10, 548), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
        cells.append(cell)
    if not cells:
        return 0
    cols = 5
    rows = (len(cells) + cols - 1) // cols
    canvas = np.full((rows * 560, cols * 440, 3), 25, np.uint8)
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        canvas[r * 560:(r + 1) * 560, k * 440:(k + 1) * 440] = c
    cv2.imwrite(str(out_png), canvas)
    return len(cells)


# ------------------------------------------------------------------------------------------ verification file
def _plan_blocks(plan_csv) -> dict:
    blocks = {}
    for r in csv.DictReader(open(plan_csv)):
        b = blocks.setdefault(r["block_index"], dict(mouse=r["mouse"], n=0, first=r["old_name"], notes=r.get("notes", "")))
        b["n"] += 1
    return blocks


def write_verification_template(plan_csv, out_md, title: str, perfect: bool, day_num, camera: CameraProfile = DJI) -> Path:
    """BOARD_VERIFICATION.md with one row per block and `VERDICT: PENDING`. Never overwrites a filled file."""
    out_md = Path(out_md)
    if out_md.exists() and out_md.stat().st_size:
        return out_md
    verdict = "PERFECT MATCH" if perfect else "MISMATCH (see reconciliation_report.txt)"
    L = [f"# {title} — whiteboard verification — PENDING", "",
         f"Provisional plan (metadata row order, counts only): {verdict}. Crops: wb/*_wb.png; montage: wb_montage.png.",
         "Read EVERY board number off the montage/crops and fill the table. The board number is the ONLY anchor:",
         "the run order may differ from the metadata order; an over-sized block hides an extra clip (usually the shortest — move",
         "it to waste and say which); a missing board is searched in the 3-5 clips around the block start. If a block binds to",
         f"another mouse, re-bind the plan (`{CMD} verify --reads ... --rebind`) before applying.", "",
         "| block | start clip | expected mouse (metadata order) | clips | board read | OK? |",
         "|---|---|---|---|---|---|"]
    for k, b in _plan_blocks(plan_csv).items():
        L.append(f"| {k} | {camera.seq_of(b['first'])} | {b['mouse']} | {b['n']} | ____ |  |"
                 + (f"  <!-- note: {b['notes']} -->" if b["notes"] else ""))
    L += ["", "Extras / camera gaps / aborted takes: (none found | list them)", "",
          "VERDICT: PENDING   <- change to 'VERDICT: APPLY' when every block binds to its mouse and clips == trials;",
          f"                      or 'VERDICT: FIX <what>' and fix the plan / waste first. Then: {CMD} rename --days {day_num} --apply",
          ""]
    out_md.write_text("\n".join(L))
    return out_md


def signed_off(verification_md) -> bool:
    p = Path(verification_md)
    return p.exists() and bool(VERDICT_APPLY.search(p.read_text()))


def current_verdict(verification_md) -> str:
    p = Path(verification_md)
    if not p.exists():
        return "(no BOARD_VERIFICATION.md)"
    m = re.search(r"^VERDICT.*$", p.read_text(), re.M)
    return m.group(0).strip() if m else "(no VERDICT line)"


def rebind_plan(plan_csv, metadata, day: str, day_num, reads: dict, log: Optional[StageLog] = None) -> None:
    """Bind each block to the mouse READ off its board and rewrite the plan.

    `reads` = {block_index: mouse id read}. Refuses unless every block has a read, every read is a mouse of
    that day used exactly once, and the block's clip count equals that mouse's metadata trial count. The
    provisional plan is kept as rename_plan.provisional.csv.
    """
    log = log or StageLog("verify")
    plan_csv = Path(plan_csv)
    if not plan_csv.exists():
        log.refuse(f"no rename plan at {plan_csv} — run the boards stage first")
    rows = list(csv.DictReader(open(plan_csv)))
    meta = {m: (n, dirs, notes) for m, n, dirs, notes in day_blocks(Path(metadata), day)[0]}
    blocks = {}
    for r in rows:
        blocks.setdefault(int(r["block_index"]), []).append(r)
    missing = [b for b in blocks if b not in reads]
    if missing:
        log.refuse(f"rebind needs a board read for every block; missing blocks {missing}")
    used = {}
    for b, mouse in reads.items():
        if b not in blocks:
            log.refuse(f"rebind: block {b} is not in the plan")
        if mouse not in meta:
            log.refuse(f"rebind: block {b} was read as '{mouse}', which is not a mouse with trials on {day}")
        if mouse in used:
            log.refuse(f"rebind: mouse '{mouse}' read on blocks {used[mouse]} and {b}")
        used[mouse] = b
        if len(blocks[b]) != meta[mouse][0]:
            log.refuse(f"rebind: block {b} has {len(blocks[b])} clips but mouse '{mouse}' has {meta[mouse][0]} metadata trials "
                       "— an extra or missing clip must be resolved first")
    backup = plan_csv.with_name("rename_plan.provisional.csv")
    if not backup.exists():
        backup.write_text(plan_csv.read_text())
    out = []
    for b in sorted(blocks):
        mouse = reads[b]
        n, dirs, notes = meta[mouse]
        for t, r in enumerate(blocks[b], 1):
            out.append({**r, "new_name": new_name(day_num, mouse, t, r["old_name"]), "mouse": mouse, "trial": t,
                        "direction": dirs[t - 1] if t - 1 < len(dirs) else "", "metadata_trials": n,
                        "video_block_size": len(blocks[b]), "block_size_match": True, "notes": notes})
    with open(plan_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PLAN_COLS)
        w.writeheader()
        w.writerows(out)
    changed = sum(1 for b in blocks if blocks[b][0]["mouse"] != reads[b])
    log(f"plan re-bound from the board reads: {changed} of {len(blocks)} blocks changed mouse (provisional plan kept as {backup.name})")


def fill_verification(verification_md, reads: Optional[dict] = None, verdict: str = "APPLY", note: str = "",
                      verifier: str = "", numbers_read: bool = True, log: Optional[StageLog] = None) -> bool:
    """Record what was read off the boards and the verdict. Returns True when every read equals the expected mouse.

    `reads` = {block_index: number read}; blocks not listed are recorded as read == expected (the lab's
    convention when the whole montage was read and everything matched). A verdict of APPLY is refused while
    any row is a MISMATCH — re-bind the plan first (then the expected column is the read mouse).
    """
    log = log or StageLog("verify")
    p = Path(verification_md)
    if not p.exists():
        log.refuse(f"{p} not found — run the boards stage first")
    reads = reads or {}
    out, ok_all, n_rows = [], True, 0
    for line in p.read_text().split("\n"):
        m = _ROW_PENDING.match(line)
        if m:
            n_rows += 1
            blk, start, exp, n = int(m.group(1)), m.group(2), m.group(3), m.group(4)
            read = str(reads.get(blk, exp))
            ok = (read == exp)
            ok_all &= ok
            line = f"| {blk} | {start} | {exp} | {n} | {read if numbers_read else 'not read'} | {'✓' if ok else 'MISMATCH'} |{m.group(5)}"
        out.append(line)
    if n_rows == 0:
        log.refuse(f"{p}: no pending rows to fill (already verified? current: {current_verdict(p)})")
    if verdict.upper().startswith("APPLY") and not ok_all:
        log.refuse("a board read differs from the expected mouse: re-bind the plan from the reads (--rebind) or use "
                   "--verdict 'FIX <what>' — APPLY is only for a table without MISMATCH rows")
    s = "\n".join(out)
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    how = ("every board number read off wb_montage.png / wb/*_wb.png" if numbers_read else
           "board NUMBERS NOT READ — fixed, documented run order: run order == metadata row order; boards present at the block starts and clean counts")
    line = (f"\nVerifier: {verifier or 'unnamed'}, {stamp} — {how}; "
            f"{'all blocks bind to their metadata mouse' if ok_all else 'see MISMATCH rows'}." + (f" {note}" if note else "") + "\n")
    s = re.sub(r"\nVERDICT: PENDING.*?(?=\n\S|\Z)", lambda _m: line + f"\nVERDICT: {verdict}", s, count=1, flags=re.S)
    s = s.replace("— whiteboard verification — PENDING",
                  f"— whiteboard verification — {'VERIFIED' if verdict.upper().startswith('APPLY') else 'FIX'}", 1)
    p.write_text(s if s.endswith("\n") else s + "\n")
    log(f"{p.parent.name}: {'all boards match' if ok_all else 'MISMATCHES'}; VERDICT: {verdict}"
        + ("" if numbers_read else "  !! NUMBERS NOT READ (fixed-run-order skip) !!"))
    return ok_all


def boards_stage(day_dir, records_dir, metadata, day: str, day_num, title: str, perfect: bool,
                 sweep_all: bool = False, whole_clip: bool = False, sample_every: int = 12,
                 camera: CameraProfile = DJI, workers: int = 1, log: Optional[StageLog] = None) -> dict:
    """Candidates -> sweep -> montage -> verification template for one day. Returns counts."""
    log = log or StageLog("boards")
    D = Path(records_dir)
    plan_csv = D / "rename_plan.csv"
    cands = board_candidates(plan_csv, day_dir, sweep_all=sweep_all, camera=camera)
    (D / "board_candidates.txt").write_text("\n".join(str(c) for c in cands) + "\n")
    missing = [c for c in cands if not Path(c).exists()]
    if missing:
        log.refuse(f"{day}: {len(missing)} candidate clip(s) of the plan are not on disk, e.g. {Path(missing[0]).name}")
    log(f"{day}: whiteboard sweep over {len(cands)} candidate clips (transient detector, "
        f"{'whole clip' if whole_clip else 'first 12 s each'}) -> {D / 'wb'}")
    rows = find_whiteboards(cands, D / "wb", sample_every=sample_every, max_seconds=None if whole_clip else 12.0,
                            camera=camera, workers=workers, log=log)
    n_crops = make_montage(D / "wb", plan_csv, D / "wb_montage.png", camera=camera)
    if n_crops == 0:
        log.warn(f"{day}: the sweep found NO whiteboard crops — re-run with --sweep-all (or --whole-clip), or look at the clips by hand")
    write_verification_template(plan_csv, D / "BOARD_VERIFICATION.md", title, perfect, day_num, camera=camera)
    write_manifest(D, "intake_boards", params=dict(day=day, sweep_all=sweep_all, whole_clip=whole_clip, sample_every=sample_every,
                                                     crop_rotation=camera.crop_rotation),
                   inputs=dict(day_dir=str(day_dir), metadata=str(metadata), n_candidates=len(cands)),
                   outputs=[str(D / "wb" / "whiteboard_sweep.csv"), str(D / "wb_montage.png"), str(D / "BOARD_VERIFICATION.md")],
                   results=dict(crops=n_crops, candidates=len(cands)))
    return dict(candidates=len(cands), crops=n_crops, rows=rows)
