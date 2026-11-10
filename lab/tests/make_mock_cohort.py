#!/usr/bin/env python
"""A synthetic RAW cohort for the whole lab pipeline (`stride-lab run`): several recording days as they come off the
camera, with everything the pipeline needs and a known answer.

    python lab/tests/make_mock_cohort.py --out <dir> [--day-pxcm 3=26.39] [--maze-shift 2=40,0]

No animal video. Per clip:
  * a drawn video with a camera name (DJI_...), the ChArUco board lying in view at the px/cm planted for that DAY,
    and a whiteboard with the mouse number at the start of each mouse's block (not for the first mouse of a day);
  * a synthetic pose file and ROI file from the public mock generator (tests/mock/make_mock_dataset.py), stored
    under the camera name as the sidecars an earlier inference would have left, so the run can skip inference;
plus a recorder-style `*_trials_full.csv` and `truth.csv` (final name, planted arm choice, reward, planted px/cm).

The maze is drawn by the public generator at 16 px/cm while the board is planted near 26 px/cm: the two are
independent here, the cohort only has to exercise the plumbing with a known answer.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import importlib.util
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
W, H, FPS, N_FRAMES, BOARD_FRAMES = 1920, 1080, 120.0, 720, 72
TF_COLS = ["day", "date", "phase", "mouse", "condition", "genotype", "sex", "weight", "reward",
           "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "accuracy", "learned", "notes"]
MICE = [("301", "F", "L"), ("302", "M", "R")]          # id, sex, rewarded arm


def public_mock():
    """The public mock generator module (pose + ROI with a planted answer)."""
    p = REPO / "tests" / "mock" / "make_mock_dataset.py"
    spec = importlib.util.spec_from_file_location("stride_public_mock", p)
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["stride_public_mock"] = mod
    spec.loader.exec_module(mod)
    return mod


def charuco_patch(pxcm: float):
    ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    board = cv2.aruco.CharucoBoard((14, 9), 0.02, 0.015, ad)
    big = board.generateImage((1400, 900))
    w = int(round(14 * pxcm * 2.0))
    return cv2.resize(big, (w, int(round(w * 9 / 14))), interpolation=cv2.INTER_AREA), (w / 14.0) / 2.0


def background(pxcm):
    bg = np.full((H, W, 3), 55, np.uint8)
    cv2.rectangle(bg, (1500, 700), (1860, 1000), (215, 215, 215), -1)            # a lit panel: bright but static
    true = None
    if pxcm:
        patch, true = charuco_patch(pxcm)
        y0, x0, pad = 60, 80, 14
        cv2.rectangle(bg, (x0 - pad, y0 - pad), (x0 + patch.shape[1] + pad, y0 + patch.shape[0] + pad), (255, 255, 255), -1)
        bg[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return bg, true


def whiteboard_patch(number: str):
    wb = np.full((300, 440, 3), 246, np.uint8)
    (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 4.2, 9)
    cv2.putText(wb, number, ((440 - tw) // 2, (300 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 4.2, (20, 20, 20), 9, cv2.LINE_AA)
    return cv2.rotate(wb, cv2.ROTATE_90_CLOCKWISE)


_VIDEO_CACHE: dict = {}


def clip_video(cache_dir: Path, pxcm, board_number) -> Path:
    """One drawn clip per (planted px/cm, whiteboard number); clips with the same content are copies."""
    key = (pxcm, board_number)
    if key in _VIDEO_CACHE and _VIDEO_CACHE[key].exists():
        return _VIDEO_CACHE[key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"clip_{pxcm}_{board_number}.mp4"
    if not p.exists():
        bg, _ = background(pxcm)
        with_board = bg.copy()
        if board_number is not None:
            patch = whiteboard_patch(str(board_number))
            with_board[560:560 + patch.shape[0], 1000:1000 + patch.shape[1]] = patch
        vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for i in range(N_FRAMES):
            vw.write(with_board if i < BOARD_FRAMES else bg)
        vw.release()
    _VIDEO_CACHE[key] = p
    return p


def make_cohort(out: Path, days=(1, 2, 3), trials: int = 2, day_pxcm: dict | None = None, maze_shift: dict | None = None,
                sidecars: bool = True, placeholder_ids: bool = False, cache_dir: Path | None = None,
                start_date: str = "2027-02-01") -> dict:
    """Write the flat raw cohort into `out`. `day_pxcm` {day: planted px/cm (None = no board that day)},
    `maze_shift` {day: (dx, dy) px by which the maze sits elsewhere in the image that day}."""
    pm = public_mock()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir else out.parent / "_clip_cache"
    day_pxcm = {int(k): v for k, v in (day_pxcm or {}).items()}
    maze_shift = {int(k): v for k, v in (maze_shift or {}).items()}
    maze = pm.Maze()
    assert (maze.frame_w, maze.frame_h) == (W, H)
    mice = [(str(i + 1), s, r) for i, (_m, s, r) in enumerate(MICE)] if placeholder_ids else MICE
    truth, tf_rows, seq, true_pxcm = [], [], 0, {}
    d0 = dt.datetime.strptime(start_date, "%Y-%m-%d")
    for day in days:
        date = d0 + dt.timedelta(days=day - 1)
        d8, iso = date.strftime("%Y%m%d"), date.strftime("%Y-%m-%d")
        pxcm = day_pxcm.get(day, 26.0)
        true_pxcm[day] = background(pxcm)[1] if pxcm else None
        shift = np.asarray(maze_shift.get(day, (0.0, 0.0)), float)
        t = dt.datetime.strptime(iso + " 09:00:00", "%Y-%m-%d %H:%M:%S")
        day_dir = out / f"{iso}-DAY{day}"
        for pos, (mid, sex, reward) in enumerate(mice):
            row = dict.fromkeys(TF_COLS, "")
            row.update(day=f"DAY{day}", date=iso, phase="acquisition", mouse=mid, condition="healthy", genotype="WT", sex=sex,
                       weight="21.0", reward=reward)
            for trial in range(1, trials + 1):
                # the last trial of mouse 2 goes to the unrewarded arm, every other trial to the rewarded one
                choice = ("L" if reward == "R" else "R") if (pos == 1 and trial == trials) else reward
                row[f"T{trial}"] = choice
                seq += 1
                t += dt.timedelta(seconds=40)
                raw = f"DJI_{d8}{t.strftime('%H%M%S')}_{seq:04d}_D.MP4"
                final = f"Day{day}_{mid}_Trial{trial}.MP4"
                board_no = mid if (trial == 1 and pos > 0) else None
                shutil.copyfile(clip_video(cache_dir, pxcm, board_no), out / raw)
                if sidecars:
                    c = pm.Clip(day=day, mouse=mid, trial=trial, sex=sex, reward=reward, behaviour=f"enter:{choice}", B=8.0)
                    rng = pm.clip_rng("lab_cohort", f"Day{day}_{mid}_Trial{trial}")
                    S, side, vis, _ev = pm.simulate(c, FPS)
                    pad = N_FRAMES - len(S)
                    assert pad >= 0
                    S, side, vis = np.r_[S, [S[-1]] * pad], np.r_[side, [side[-1]] * pad], np.r_[vis, [True] * pad]
                    px = maze.to_px(pm.pose_cm(S, side, c)) + rng.normal(0, 0.3, (N_FRAMES, len(pm.POSE_NODES), 2)) + shift
                    px[~vis] = np.nan
                    scores = rng.uniform(0.85, 0.98, px.shape[:2])
                    pm.write_pose_slp((out / raw).with_suffix(".slp"), out / raw, px, scores, vis, embed_as=str(day_dir / final))
                    doc = maze.roi_yaml(str(day_dir / final))
                    for r in doc["rois"]:
                        r["coordinates"] = [[x + float(shift[0]), y + float(shift[1])] for x, y in r["coordinates"]]
                    (out / raw).with_suffix(".rois.yml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
                truth.append(dict(raw_name=raw, final_name=final, day=day, mouse=mid, trial=trial, reward=reward,
                                  planted_choice=choice, expected_correct=int(choice == reward), board_shown=int(board_no is not None),
                                  planted_px_per_cm=true_pxcm[day] if true_pxcm[day] else ""))
            tf_rows.append(row)
    export = out / f"{(d0 + dt.timedelta(days=max(days) - 1)).strftime('%Y-%m-%d')}_mock_trials_full.csv"
    with open(export, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=TF_COLS)
        w.writeheader()
        w.writerows(tf_rows)
    with open(out / "truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(truth[0].keys()))
        w.writeheader()
        w.writerows(truth)
    info = dict(n_clips=len(truth), days=list(days), mice=[m[0] for m in mice], trials=trials,
                planted_px_per_cm={str(k): v for k, v in true_pxcm.items()}, export=export.name, frame=[W, H], fps=FPS)
    (out / "truth.json").write_text(json.dumps(info, indent=1) + "\n", encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--days", default="1,2,3")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--day-pxcm", action="append", default=[], metavar="DAY=PXCM", help="planted px/cm for one day (repeatable)")
    ap.add_argument("--maze-shift", action="append", default=[], metavar="DAY=DX,DY", help="maze displaced on one day (repeatable)")
    a = ap.parse_args()
    pxcm = {int(k): float(v) for k, v in (x.split("=") for x in a.day_pxcm)}
    shift = {int(k): tuple(float(z) for z in v.split(",")) for k, v in (x.split("=") for x in a.maze_shift)}
    info = make_cohort(a.out, tuple(int(d) for d in a.days.split(",")), a.trials, pxcm, shift)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
