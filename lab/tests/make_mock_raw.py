#!/usr/bin/env python
"""Synthetic RAW-CAMERA cohorts for the `stride prepare` front-end (sort -> plan -> boards -> rename -> calibrate).

No animal video: every clip is drawn. Each scenario is one cohort folder as it comes off the camera — clips with
camera names in recording order, a recorder-style `*_trials_full.csv`, a whiteboard with the mouse number held up
for a moment at the start of each mouse's block, and a ChArUco board lying in view at a planted px/cm — plus
`truth.csv` / `truth.json` with the answer (the name every clip must end up with, the planted scale).

    python tests/mock/make_mock_raw.py --out <dir> [--only base,extra_clip]

Scenarios (the rename procedure's known traps):
    base                2 days x 3 mice x 3 trials; a gap in the camera numbering; day 2's first mouse has no board
    extra_clip          an aborted short take inside the day: clips = trials + 1
    missing_clip        one clip never recorded: clips = trials - 1
    swapped_order       the mice were run in another order than the metadata rows (only the boards tell)
    board_second_trial  one mouse's board is shown on its second trial (says so in `notes`)
    placeholder_ids     mouse ids 1, 2, 3 (what the recorder writes before real ids are typed in)
    other_camera        another naming scheme, lowercase .mp4, no ChArUco board in view
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1920, 1080, 30
TRIAL_FRAMES, SHORT_FRAMES, BOARD_FRAMES = 60, 15, 18
PXCM = 26.0                                   # planted scale: one 20 mm ChArUco square = 52 px
TF_COLS = ["day", "date", "phase", "mouse", "condition", "genotype", "sex", "weight", "reward",
           "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "accuracy", "learned", "notes"]


def charuco_patch(pxcm: float):
    """The lab's board (14 x 9 squares of 20 mm, 15 mm markers, DICT_5X5_250) drawn at `pxcm`; -> (image, true px/cm)."""
    ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((14, 9), 0.02, 0.015, ad)
    big = board.generateImage((1400, 900))                          # 100 px per square
    w = int(round(14 * pxcm * 2.0))
    img = cv2.resize(big, (w, int(round(w * 9 / 14))), interpolation=cv2.INTER_AREA)
    return img, (w / 14.0) / 2.0


def background(with_charuco: bool = True):
    """A static scene: dark floor, a lit (bright, smooth, STATIC) floor panel, a maze outline, the ChArUco board."""
    bg = np.full((H, W, 3), 55, np.uint8)
    cv2.rectangle(bg, (1500, 80), (1860, 330), (215, 215, 215), -1)             # lit panel: bright but static
    cv2.rectangle(bg, (150, 620), (1450, 760), (95, 80, 70), -1)                # stem
    cv2.rectangle(bg, (1450, 360), (1590, 1020), (95, 80, 70), -1)              # arms
    true_pxcm = None
    if with_charuco:
        patch, true_pxcm = charuco_patch(PXCM)
        y0, x0 = 60, 80
        pad = 14
        cv2.rectangle(bg, (x0 - pad, y0 - pad), (x0 + patch.shape[1] + pad, y0 + patch.shape[0] + pad), (255, 255, 255), -1)
        bg[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return bg, true_pxcm


def whiteboard_patch(number: str):
    """A whiteboard with the mouse number, turned the way the lab rig sees it (reads upright after a 90 deg CCW turn)."""
    wb = np.full((300, 440, 3), 246, np.uint8)
    size = 4.2 if len(number) <= 3 else 3.0
    (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, size, 9)
    cv2.putText(wb, number, ((440 - tw) // 2, (300 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, size, (20, 20, 20), 9, cv2.LINE_AA)
    return cv2.rotate(wb, cv2.ROTATE_90_CLOCKWISE)                              # 440 x 300 -> 300 wide x 440 high


def write_clip(path: Path, bg, n_frames: int, board_number=None, board_frames: int = BOARD_FRAMES, when: dt.datetime = None):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    patch = whiteboard_patch(str(board_number)) if board_number is not None else None
    for i in range(n_frames):
        fr = bg.copy()
        x = 200 + int(1200 * i / max(1, n_frames - 1))                          # a "mouse" walking the stem
        cv2.ellipse(fr, (x, 690), (38, 16), 0, 0, 360, (30, 30, 30), -1)
        if patch is not None and i < board_frames:
            fr[560:560 + patch.shape[0], 860:860 + patch.shape[1]] = patch
        vw.write(fr)
    vw.release()
    if when is not None:
        ts = when.timestamp()
        os.utime(path, (ts, ts))


# One scenario = a list of days; one day = date + the mice in METADATA row order + the order they were RUN in.
#   mouse: dict(id, trials, notes="", board="first" | "second" | "none")
#   extras: {(run position of the mouse, trial index before which the extra take happens): "short"}
#   drop:   {(mouse id, trial)} clips that were never recorded;   gap_after: camera numbers skipped after N clips
def _mice(ids, n=3, **kw):
    return [dict(id=i, trials=n, notes=kw.get("notes", {}).get(i, ""), board=kw.get("board", {}).get(i, "first")) for i in ids]


SCENARIOS = {
    "base": dict(days=[
        dict(date="2027-01-05", day=1, mice=_mice(["301", "302", "303"]), gap_after=4),
        dict(date="2027-01-06", day=2, mice=_mice(["301", "302", "303"], board={"301": "none"})),
    ]),
    "extra_clip": dict(days=[dict(date="2027-01-05", day=1, mice=_mice(["301", "302", "303"]), extras={(1, 1): "short"})]),
    "missing_clip": dict(days=[dict(date="2027-01-05", day=1, mice=_mice(["301", "302", "303"]), drop={("302", 3)})]),
    "swapped_order": dict(days=[dict(date="2027-01-05", day=1, mice=_mice(["301", "302", "303"]), run_order=["302", "301", "303"])]),
    "board_second_trial": dict(days=[dict(date="2027-01-05", day=1, mice=_mice(
        ["301", "302", "303"], notes={"302": "Board show on second trial"}, board={"302": "second"}))]),
    "placeholder_ids": dict(days=[dict(date="2027-01-05", day=1, mice=_mice(["1", "2", "3"], n=2))]),
    "other_camera": dict(charuco=False, name_fmt="CAM_{date}_{seq:04d}.mp4",
                         raw_pattern=r"^CAM_(?P<date>\d{8})_(?P<seq>\d{4})\.mp4$",
                         days=[dict(date="2027-01-05", day=1, mice=_mice(["301", "302"], n=2))]),
}


def make_scenario(name: str, out: Path) -> dict:
    spec = SCENARIOS[name]
    root = out / name
    root.mkdir(parents=True, exist_ok=True)
    bg, true_pxcm = background(spec.get("charuco", True))
    fmt = spec.get("name_fmt", "DJI_{date}{time}_{seq:04d}_D.MP4")
    seq, truth, tf_rows = 0, [], []
    for day in spec["days"]:
        d8 = day["date"].replace("-", "")
        t = dt.datetime.strptime(day["date"] + " 09:00:00", "%Y-%m-%d %H:%M:%S")
        by_id = {m["id"]: m for m in day["mice"]}
        for m in day["mice"]:                                                  # metadata rows, in METADATA order
            row = dict.fromkeys(TF_COLS, "")
            row.update(day=f"DAY{day['day']}", date=day["date"], phase="acquisition", mouse=m["id"], condition="healthy",
                       genotype="WT", sex="F", weight="21.0", reward="L", notes=m["notes"])
            for k in range(m["trials"]):
                row[f"T{k + 1}"] = "LR"[k % 2]
            tf_rows.append(row)
        n_in_day = 0
        for pos, mid in enumerate(day.get("run_order") or [m["id"] for m in day["mice"]]):
            m = by_id[mid]
            for trial in range(1, m["trials"] + 1):
                takes = []
                if day.get("extras", {}).get((pos, trial)) == "short":          # an aborted take: board shown, then re-shown
                    takes.append(("extra", SHORT_FRAMES, mid, 6))
                show = (m["board"] == "first" and trial == 1) or (m["board"] == "second" and trial == 2)
                takes.append(("trial", TRIAL_FRAMES, mid if (show or takes) else None, BOARD_FRAMES))
                for kind, n_frames, board_no, bf in takes:
                    if kind == "trial" and (mid, trial) in day.get("drop", set()):
                        continue                                                # never recorded: no clip, no camera number
                    seq += 1
                    if day.get("gap_after") and n_in_day == day["gap_after"]:
                        seq += 1                                                # the camera deleted one clip: numbering gap
                    n_in_day += 1
                    t += dt.timedelta(seconds=40)
                    raw = fmt.format(date=d8, time=t.strftime("%H%M%S"), seq=seq)
                    write_clip(root / raw, bg, n_frames, board_number=board_no, board_frames=bf, when=t)
                    ext = Path(raw).suffix
                    truth.append(dict(scenario=name, raw_name=raw, day=day["day"], kind=kind, mouse=mid, trial=trial if kind == "trial" else "",
                                      true_new_name=f"Day{day['day']}_{mid}_Trial{trial}{ext}" if kind == "trial" else "",
                                      board_shown=int(board_no is not None), board_number=board_no or ""))
    export = root / f"{spec['days'][-1]['date']}_mock_trials_full.csv"
    with open(export, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TF_COLS)
        w.writeheader()
        w.writerows(tf_rows)
    with open(root / "truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(truth[0].keys()))
        w.writeheader()
        w.writerows(truth)
    info = dict(scenario=name, n_clips=len(truth), px_per_cm=true_pxcm, raw_pattern=spec.get("raw_pattern"),
                frame=[W, H], fps=FPS, export=export.name)
    (root / "truth.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default=",".join(SCENARIOS))
    a = ap.parse_args()
    for name in a.only.split(","):
        info = make_scenario(name, a.out)
        print(f"{name}: {info['n_clips']} clips, px/cm {info['px_per_cm']} -> {a.out / name}")


if __name__ == "__main__":
    main()
