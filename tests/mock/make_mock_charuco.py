#!/usr/bin/env python
"""Synthetic clips with a ChArUco board in view, for the scale calibration (`stride calibrate`).

No animal video: every frame is drawn. The board is rendered at a planted px/cm, so the calibration has a
known answer (`truth.json` in each scenario folder).

    python tests/mock/make_mock_charuco.py --out <dir> [--only days,flat]

Scenarios
    days         three day folders x 4 clips, board at 26.0 px/cm, .MP4 and .mp4 mixed
    flat         one folder, 5 clips, board at 23.5 px/cm
    partly       5 clips, the board is missing in two of them
    no_board     3 clips without a board
    moved        the board lies at two different distances (26.0 and 27.0 px/cm): the readings disagree
    other_board  a 7 x 5 board, 30 mm squares, 22 mm markers, DICT_4X4_50, at 18.0 px/cm
    nested       clips two folders deep, plus a `waste/` and a `_private/` folder that must be ignored
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, N_FRAMES = 1920, 1080, 30, 20
LAB = dict(squares=(14, 9), square_mm=20.0, marker_mm=15.0, dictionary="DICT_5X5_250")
SMALL = dict(squares=(7, 5), square_mm=30.0, marker_mm=22.0, dictionary="DICT_4X4_50")


def charuco_patch(pxcm: float, board: dict = LAB):
    """The board drawn so that one square is `pxcm * square_cm` pixels; -> (grey image, the exact px/cm drawn)."""
    sx, sy = board["squares"]
    ad = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board["dictionary"]))
    b = cv2.aruco.CharucoBoard((sx, sy), board["square_mm"] / 1000.0, board["marker_mm"] / 1000.0, ad)
    big = b.generateImage((sx * 100, sy * 100))                       # 100 px per square
    w = int(round(sx * pxcm * board["square_mm"] / 10.0))
    img = cv2.resize(big, (w, int(round(w * sy / sx))), interpolation=cv2.INTER_AREA)
    return img, (w / sx) / (board["square_mm"] / 10.0)


def scene(pxcm=None, board: dict = LAB):
    """A static arena; with `pxcm`, the board lies in view. -> (frame, exact px/cm or None)"""
    fr = np.full((H, W, 3), 60, np.uint8)
    cv2.rectangle(fr, (150, 640), (1450, 780), (95, 80, 70), -1)
    cv2.rectangle(fr, (1450, 380), (1590, 1040), (95, 80, 70), -1)
    true = None
    if pxcm is not None:
        patch, true = charuco_patch(pxcm, board)
        y0, x0, pad = 60, 80, 14
        cv2.rectangle(fr, (x0 - pad, y0 - pad), (x0 + patch.shape[1] + pad, y0 + patch.shape[0] + pad), (255, 255, 255), -1)
        fr[y0:y0 + patch.shape[0], x0:x0 + patch.shape[1]] = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    return fr, true


def write_clip(path: Path, frame, n_frames: int = N_FRAMES):
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(n_frames):
        fr = frame.copy()
        cv2.ellipse(fr, (200 + 60 * i, 710), (38, 16), 0, 0, 360, (30, 30, 30), -1)     # something moving
        vw.write(fr)
    vw.release()


def make_scenario(name: str, out: Path) -> dict:
    root = out / name
    info = dict(scenario=name, px_per_cm=None, board=LAB, clips=[])

    def add(rel, pxcm, board=LAB):
        fr, true = scene(pxcm, board)
        write_clip(root / rel, fr)
        info["clips"].append(dict(clip=rel, px_per_cm=true))
        return true

    if name == "days":
        for d in (1, 2, 3):
            for k in range(4):
                info["px_per_cm"] = add(f"2027-01-0{d}-DAY{d}/Day{d}_30{k + 1}_Trial1.{'MP4' if k % 2 == 0 else 'mp4'}", 26.0)
    elif name == "flat":
        for k in range(5):
            info["px_per_cm"] = add(f"clip_{k + 1:02d}.mp4", 23.5)
    elif name == "partly":
        for k in range(5):
            v = add(f"clip_{k + 1:02d}.mp4", None if k in (1, 3) else 26.0)
            info["px_per_cm"] = v or info["px_per_cm"]
    elif name == "no_board":
        for k in range(3):
            add(f"clip_{k + 1:02d}.mp4", None)
    elif name == "moved":
        for k in range(4):
            add(f"clip_{k + 1:02d}.mp4", 26.0 if k < 2 else 27.0)
    elif name == "other_board":
        info["board"] = SMALL
        for k in range(3):
            info["px_per_cm"] = add(f"clip_{k + 1:02d}.mp4", 18.0, SMALL)
    elif name == "nested":
        for k in range(3):
            info["px_per_cm"] = add(f"session_a/cam1/take_{k + 1}.{'MP4' if k else 'mp4'}", 26.0)
        add("waste/aborted.mp4", None)
        add("_private/notes.mp4", None)
    else:
        raise KeyError(name)
    (root / "truth.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


SCENARIOS = ("days", "flat", "partly", "no_board", "moved", "other_board", "nested")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default=",".join(SCENARIOS))
    a = ap.parse_args()
    for name in a.only.split(","):
        info = make_scenario(name, a.out)
        print(f"{name}: {len(info['clips'])} clips, planted px/cm {info['px_per_cm']} -> {a.out / name}")


if __name__ == "__main__":
    main()
