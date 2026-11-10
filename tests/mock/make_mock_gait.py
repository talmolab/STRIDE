#!/usr/bin/env python
"""Synthetic runway clips for `stride gait` (the gait-only pipeline).

No animal video and no maze: a kinematic mouse (the one of make_mock_dataset.py) walks a straight
line. There are NO ROI files and the file names do NOT follow Day<N>_<mouse>_Trial<T>. Each
scenario folder holds `videos/` (video + pose file per clip), `truth.csv` (what was planted) and
`scenario.json` (paths and facts for the tests).

    python tests/mock/make_mock_gait.py --out <dir> [--only runway,renamed]

Scenarios
    runway        6 clips, names with spaces, `.mp4` and `.MP4`, 24 px/cm, 120 fps, metadata.csv
    renamed       the first 3 runway clips with other node names in the pose skeleton (scenario.json holds the mapping)
    one_landmark  the first 3 runway clips as a 5-node skeleton with one landmark per hind paw
    no_tailtip    the first 2 runway clips, skeleton without a tail tip
    no_tailbase   2 clips, skeleton without the tail base (a required keypoint)
    vertical      2 clips, the mouse walks top to bottom in the image
    board         4 clips with a ChArUco board drawn into the video at 24 px/cm
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_mock_charuco as MC          # noqa: E402
import make_mock_dataset as M           # noqa: E402

FPS = 120.0
N_FRAMES = 720                           # 6 s

RENAME = {"snout": "nose", "mouth": "jaw", "tailbase": "tail_root", "tailtip": "tail_end",
          "hindpawL1": "LH_heel", "hindpawL2": "LH_toe", "hindpawR1": "RH_heel", "hindpawR2": "RH_toe",
          "forepawL1": "LF_heel", "forepawL2": "LF_toe", "forepawR1": "RF_heel", "forepawR2": "RF_toe",
          "tail1": "tail_a", "tail2": "tail_b", "tail3": "tail_c"}

# (file stem, extension, animal, sex, group, stride length L cm, duty d, speed v cm/s)
RUNWAY = [
    ("runway A1", ".mp4", "A1", "F", "control", 6.0, 0.60, 20.0),
    ("runway A2", ".mp4", "A1", "F", "control", 6.0, 0.60, 24.0),
    ("runway B1", ".MP4", "B7", "M", "control", 6.6, 0.58, 22.0),
    ("runway B2", ".mp4", "B7", "M", "control", 6.6, 0.58, 26.0),
    ("run_C-01", ".mp4", "C3", "F", "treated", 5.4, 0.62, 18.0),
    ("run_C-02", ".mp4", "C3", "F", "treated", 5.4, 0.62, 21.0),
]


def write_pose(path: Path, video_path: Path, pts_px, scores, visible, node_names):
    import sleap_io as sio
    skel = sio.Skeleton(nodes=list(node_names))
    video = sio.Video.from_filename(str(video_path))
    lfs = []
    for t in np.flatnonzero(visible):
        inst = sio.PredictedInstance.from_numpy(pts_px[t], skeleton=skel, point_scores=scores[t],
                                                score=float(np.nanmean(scores[t])))
        lfs.append(sio.LabeledFrame(video=video, frame_idx=int(t), instances=[inst]))
    sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel]).save(str(path))


def board_video(path: Path, w: int, h: int, fps: float, n: int, pxcm: float) -> float:
    """A static clip with a ChArUco board in the top-left corner; returns the exact px/cm drawn."""
    import cv2
    patch, exact = MC.charuco_patch(pxcm)
    frame = np.full((h, w, 3), 90, np.uint8)
    ph, pw = patch.shape[:2]
    frame[20:20 + ph, 20:20 + pw] = patch[..., None] if patch.ndim == 2 else patch
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    for _ in range(n):
        wr.write(frame)
    wr.release()
    return exact


def make_clip(vdir: Path, cache: Path, scn: str, stem: str, ext: str, L: float, d: float, v: float,
              maze: M.Maze, nodes_keep=None, rename=None, board_pxcm=None, n_frames=N_FRAMES) -> dict:
    c = M.Clip(day=1, mouse="x", trial=1, behaviour="none", L=L, d=d, v=v)
    rng = M.clip_rng("gait/runway", stem)     # same coordinates for the same clip in every scenario
    S, side, vis, _ = M.simulate(c, FPS)
    assert len(S) <= n_frames, (stem, len(S))
    pad = n_frames - len(S)
    S, side, vis = np.r_[S, [S[-1]] * pad], np.r_[side, [side[-1]] * pad], np.r_[vis, [True] * pad]
    cm = M.pose_cm(S, side, c)
    px = maze.to_px(cm) + rng.normal(0, 0.3, cm.shape)
    scores = rng.uniform(0.85, 0.98, px.shape[:2])
    px[~vis] = np.nan
    names = list(M.POSE_NODES)
    if nodes_keep is not None:
        keep = [M.NI[n] for n in nodes_keep]
        px, scores, names = px[:, keep], scores[:, keep], list(nodes_keep)
    if rename:
        names = [rename.get(n, n) for n in names]
    video = vdir / f"{stem}{ext}"
    exact = None
    if board_pxcm is not None:
        exact = board_video(video, maze.frame_w, maze.frame_h, FPS, n_frames, board_pxcm)
    else:
        shutil.copyfile(M.blank_video(cache, maze.frame_w, maze.frame_h, FPS, n_frames), video)
    write_pose(video.with_suffix(".slp"), video, px, scores, vis, names)
    return {"stem": stem, "video": video.name, "planted_stride_length_cm": L, "planted_duty": d,
            "planted_speed_cm_s": v, "planted_body_frame_excursion_cm": round(d * L, 4),
            "planted_stride_period_ms": round(1000.0 * L / v, 2), "px_per_cm": maze.px_per_cm,
            "maze_rotation_deg": maze.rotation_deg, "fps": FPS, "n_frames": n_frames,
            "pose_nodes": "|".join(names), "board_px_per_cm": exact}


def build(name: str, out: Path, cache: Path) -> dict:
    sdir = out / name
    if sdir.exists():
        shutil.rmtree(sdir)
    vdir = sdir / "videos"
    vdir.mkdir(parents=True)
    flat = M.Maze(px_per_cm=24.0)
    info = {"name": name, "dir": str(sdir), "video_dir": str(vdir), "px_per_cm": 24.0, "fps": FPS}
    rows = []
    if name == "runway":
        for stem, ext, animal, sex, group, L, d, v in RUNWAY:
            rows.append(make_clip(vdir, cache, name, stem, ext, L, d, v, flat))
        with open(sdir / "metadata.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["video", "animal", "sex", "group", "weight_g"])
            for i, (stem, ext, animal, sex, group, *_rest) in enumerate(RUNWAY):
                if i == len(RUNWAY) - 1:
                    continue                                    # the last clip has no metadata row
                w.writerow([f"{stem}{ext}", animal, sex, group, 21.5 + i])
        info["metadata_csv"] = str(sdir / "metadata.csv")
        info["no_metadata_stem"] = RUNWAY[-1][0]
    elif name == "renamed":
        for stem, ext, *_m, L, d, v in RUNWAY[:3]:
            rows.append(make_clip(vdir, cache, name, stem, ext, L, d, v, flat, rename=RENAME))
        mapping = {v: k for k, v in RENAME.items() if k in ("snout", "tailbase", "tailtip", "hindpawL1", "hindpawL2",
                                                            "hindpawR1", "hindpawR2")}
        (sdir / "keypoints.json").write_text(json.dumps(mapping, indent=1), encoding="utf-8")
        info["keypoints"] = mapping
    elif name == "one_landmark":
        keep = ["snout", "tailbase", "tailtip", "hindpawL2", "hindpawR2"]
        ren = {"hindpawL2": "hind_left", "hindpawR2": "hind_right"}
        for stem, ext, *_m, L, d, v in RUNWAY[:3]:
            rows.append(make_clip(vdir, cache, name, stem, ext, L, d, v, flat, nodes_keep=keep, rename=ren))
        info["keypoints"] = {"hind_left": "hindpawL2", "hind_right": "hindpawR2"}
    elif name == "no_tailtip":
        keep = [n for n in M.POSE_NODES if n != "tailtip"]
        for stem, ext, *_m, L, d, v in RUNWAY[:2]:
            rows.append(make_clip(vdir, cache, name, stem, ext, L, d, v, flat, nodes_keep=keep))
    elif name == "no_tailbase":
        keep = [n for n in M.POSE_NODES if n != "tailbase"]
        for stem, ext, *_m, L, d, v in RUNWAY[:2]:
            rows.append(make_clip(vdir, cache, name, stem, ext, L, d, v, flat, nodes_keep=keep))
    elif name == "vertical":
        vert = M.Maze(px_per_cm=16.0, rotation_deg=90.0)
        info["px_per_cm"] = 16.0
        for stem, ext, *_m, L, d, v in RUNWAY[:2]:
            rows.append(make_clip(vdir, cache, name, stem, ext, L, d, v, vert))
    elif name == "board":
        for stem, ext, *_m, L, d, v in RUNWAY[:4]:
            rows.append(make_clip(vdir, cache, name, stem, ".mp4", L, d, max(v, 24.0), flat, board_pxcm=24.0,
                                  n_frames=600))
        info["board_px_per_cm"] = rows[0]["board_px_per_cm"]
    else:
        raise SystemExit(f"unknown scenario {name}")
    with open(sdir / "truth.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    info.update(n_clips=len(rows), truth_csv=str(sdir / "truth.csv"), stems=[r["stem"] for r in rows])
    (sdir / "scenario.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


SCENARIOS = ["runway", "renamed", "one_landmark", "no_tailtip", "no_tailbase", "vertical", "board"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="output directory (scenario folders are replaced)")
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    names = [n.strip() for n in a.only.split(",")] if a.only else SCENARIOS
    for n in names:
        info = build(n, out, out / "_blank_cache")
        print(f"{n}: {info['n_clips']} clips -> {info['dir']}")


if __name__ == "__main__":
    main()
