#!/usr/bin/env python3
"""Deterministic synthetic T-maze dataset for exercising STRIDE end to end.

Everything written here is synthetic: no animal video, no lab paths, no real mouse ids.

    python tests/mock/make_mock_dataset.py --out /tmp/stride_mock            # all scenarios
    python tests/mock/make_mock_dataset.py --out /tmp/stride_mock --only core,behaviour

What one "clip" is
------------------
* ``<stem>.mp4``   blank video with the right frame size / fps / frame count (STRIDE only reads
                   those three things from the video file).
* ``<stem>.slp``   15-keypoint pose written with sleap-io: a kinematic mouse that stands in the
                   start segment (segment1), walks the stem with an alternating hind-paw gait,
                   reaches the junction and enters an arm.
* ROI yml          the 7 maze polygons in the format ``stride convert-roi`` writes.

Each scenario folder holds a small batch that isolates ONE user case (layout, ids, metadata,
pose quality, calibration, camera, behaviour, batch composition) plus:

* ``meta_trials.csv``      day,mouse,reward            (decision stage input)
* ``metadata.csv``         Mouse,Gender,Genotype       (enrichment input)
* ``config.readme.yaml``   the repo config.example.yaml with ``batches:`` replaced by ONE batch that has
                           exactly the fields the README shows for a new batch
* ``truth.csv``            what was planted per clip (choice, probes, stride length, speed, ...)
* ``scenario.json``        description + the paths a test needs

Maze geometry (cm), horizontal T as ``stride.stages.slp_to_yaml`` lays it out::

      y=-32.5 ........ +------+
                       | arm  |
                       | right|
      y=-5  +----+----+----+----+------+
            |seg1|seg2|seg3|seg4| junc |      mouse walks +x
      y=+5  +----+----+----+----+------+
                       | arm  |
                       | left |
      y=+32.5 ........ +------+
            x=0  13.75 27.5 41.25 55    65

Gait model: the body advances along a path (straight stem -> quarter-circle -> arm) with arc
position S(t).  Each paw is fixed on the ground during stance and moves one ground stride length L
forward during swing; left/right hind paws are half a cycle apart.  With duty factor d the paw
moves d*L in the BODY frame per swing (planted_body_frame_excursion_cm) and L on the ground
(planted_stride_length_cm).  The stride period is L / speed.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
SEED = 20260101

POSE_NODES = [
    "snout", "mouth",
    "forepawR2", "forepawR1", "forepawL1", "forepawL2",
    "hindpawR2", "hindpawR1", "hindpawL2", "hindpawL1",
    "tailbase", "tail1", "tail2", "tail3", "tailtip",
]
NI = {n: i for i, n in enumerate(POSE_NODES)}
HULL_NODES = ["snout", "forepawL2", "forepawR2", "hindpawL2", "hindpawR2", "tailbase"]

ROI_ORDER = ["arm_right", "junction", "arm_left", "segment4", "segment3", "segment2", "segment1"]
ROI_PALETTE = {
    "arm_right": "#2ca02c", "junction": "#98df8a", "arm_left": "#d62728",
    "segment4": "#ff9896", "segment3": "#9467bd", "segment2": "#c5b0d5", "segment1": "#8c564b",
}
ROI_NODES_23 = [
    "arm_right.top_left", "arm_right.top_right", "arm_right.centroid",
    "junction.top_left", "junction.top_right", "junction.bottom_left",
    "junction.bottom_right", "junction.centroid",
    "arm_left.bottom_left", "arm_left.bottom_right", "arm_left.centroid",
    "segment1.top_left", "segment1.bottom_left", "segment1.centroid",
    "segment2.top_left", "segment2.bottom_left", "segment2.centroid",
    "segment3.top_left", "segment3.bottom_left", "segment3.centroid",
    "segment4.top_left", "segment4.bottom_left", "segment4.centroid",
]

# ---- maze (cm) -------------------------------------------------------------------------------
SEG = 13.75
HALF_W = 5.0
ARM = 27.5
REGIONS_CM = {  # name: (xmin, xmax, ymin, ymax)
    "segment1": (0.0, SEG, -HALF_W, HALF_W),
    "segment2": (SEG, 2 * SEG, -HALF_W, HALF_W),
    "segment3": (2 * SEG, 3 * SEG, -HALF_W, HALF_W),
    "segment4": (3 * SEG, 4 * SEG, -HALF_W, HALF_W),
    "junction": (55.0, 65.0, -HALF_W, HALF_W),
    "arm_right": (55.0, 65.0, -HALF_W - ARM, -HALF_W),
    "arm_left": (55.0, 65.0, HALF_W, HALF_W + ARM),
}
ARC_START = 56.0          # tail-base arc position where the turn into an arm starts
ARC_R = 4.0
ARC_LEN = ARC_R * math.pi / 2
S_ENTRY = ARC_START + ARC_LEN + 2.0   # tail base ~1 cm inside the arm: the whole body hull is in
S_END = ARC_START + ARC_LEN + 14.0    # where an entering mouse stops
S_PROBE = 58.28           # snout ~2 cm inside the arm, body still in the junction
S_BACK = 48.0             # where a mouse that withdraws from a probe backs up to


@dataclass
class Maze:
    px_per_cm: float = 16.0
    frame_w: int = 1920
    frame_h: int = 1080
    rotation_deg: float = 0.0

    def to_px(self, p_cm):
        p = (np.asarray(p_cm, float) - np.array([32.5, 0.0])) * self.px_per_cm
        a = math.radians(self.rotation_deg)
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return p @ rot.T + np.array([self.frame_w / 2.0, self.frame_h / 2.0])

    def region_px(self, name):
        """4 corners in the order slp_to_yaml writes them: TL, BL, BR, TR (of the unrotated maze)."""
        x0, x1, y0, y1 = REGIONS_CM[name]
        return self.to_px([[x0, y0], [x0, y1], [x1, y1], [x1, y0]])

    def check_fits(self):
        for n in REGIONS_CM:
            c = self.region_px(n)
            assert c[:, 0].min() >= 0 and c[:, 0].max() <= self.frame_w - 1, (n, c)
            assert c[:, 1].min() >= 0 and c[:, 1].max() <= self.frame_h - 1, (n, c)

    def roi_yaml(self, image_file: str, drop=()):
        rois = []
        for i, name in enumerate(ROI_ORDER, start=1):
            if name in drop:
                continue
            coords = [[float(x), float(y)] for x, y in self.region_px(name)]
            per = sum(math.dist(coords[k], coords[(k + 1) % 4]) for k in range(4))
            area = abs(sum(coords[k][0] * coords[(k + 1) % 4][1] - coords[(k + 1) % 4][0] * coords[k][1]
                           for k in range(4))) * 0.5
            rois.append({"id": i, "name": name, "type": "polygon", "coordinates": coords,
                         "color": ROI_PALETTE[name],
                         "properties": {"vertex_count": 4, "perimeter": per, "area": area}})
        return {"image_file": image_file, "roi_count": len(rois), "rois": rois,
                "metadata": {"created_with": "stride-mock", "format_version": "1.0"}}

    def roi_keypoints_px(self):
        """The 23 ROI keypoints an ROI model would predict, for ``stride convert-roi``."""
        out = {}
        for name in ROI_ORDER:
            tl, bl, br, tr = self.region_px(name)
            out[f"{name}.top_left"], out[f"{name}.bottom_left"] = tl, bl
            out[f"{name}.bottom_right"], out[f"{name}.top_right"] = br, tr
            out[f"{name}.centroid"] = (tl + bl + br + tr) / 4.0
        return np.array([out[n] for n in ROI_NODES_23])


# ---- path and kinematics ---------------------------------------------------------------------
def path_xy(s, side):
    """Point and unit tangent on the body path at arc position s (cm).
    side = -1 -> arm_right (y<0), +1 -> arm_left (y>0), 0 -> straight stem only."""
    s = np.asarray(s, float)
    if side == 0:
        return np.stack([s, np.zeros_like(s)], -1), np.stack([np.ones_like(s), np.zeros_like(s)], -1)
    a = np.clip((s - ARC_START) / ARC_R, 0.0, math.pi / 2)
    extra = np.clip(s - ARC_START - ARC_LEN, 0.0, None)
    x = np.where(s < ARC_START, s, ARC_START + ARC_R * np.sin(a))
    y = side * (ARC_R * (1 - np.cos(a)) + extra)
    return np.stack([x, y], -1), np.stack([np.cos(a), side * np.sin(a)], -1)


def offset(s, side, lateral):
    p, t = path_xy(s, side)
    n = np.stack([-t[..., 1], t[..., 0]], -1)       # +n = the mouse's left in this image convention
    return p + n * np.asarray(lateral, float)[..., None]


def paw_arc(S, L, d, phase0, hip):
    """Arc position of a paw as a function of body arc position S (stance fixed, swing = one L)."""
    c = S / L - phase0
    k = np.floor(c)
    psi = c - k
    land = (k + phase0) * L + hip + d * L / 2.0
    u = np.clip((psi - d) / (1.0 - d), 0.0, 1.0)
    return land + L * (u - np.sin(2 * np.pi * u) / (2 * np.pi))


def trapezoid(S0, S1, v, fps, acc=120.0):
    dist = abs(S1 - S0)
    if dist < 1e-9:
        return np.array([])
    sgn = 1.0 if S1 > S0 else -1.0
    t_acc = v / acc
    if acc * t_acc ** 2 >= dist:
        t_acc = math.sqrt(dist / acc)
        t_cruise, vpk = 0.0, acc * t_acc
    else:
        t_cruise, vpk = (dist - acc * t_acc ** 2) / v, v
    total = 2 * t_acc + t_cruise
    t = (np.arange(int(math.ceil(total * fps))) + 1) / fps
    s = np.where(t < t_acc, 0.5 * acc * t ** 2,
                 np.where(t < t_acc + t_cruise, 0.5 * acc * t_acc ** 2 + vpk * (t - t_acc),
                          dist - 0.5 * acc * np.clip(total - t, 0, None) ** 2))
    return S0 + sgn * np.clip(s, 0, dist)


@dataclass
class Clip:
    day: int
    mouse: str
    trial: int
    genotype: str = "WT"
    sex: str = "F"
    reward: str = "L"
    behaviour: str = "enter:L"     # enter:L|enter:R|none|probe:R>L|probe:L>R
    start: str = "seg1"            # seg1 | seg4
    absent_s: float = 0.2          # leading seconds with no pose (mouse being placed)
    L: float = 6.0                 # ground stride length, cm
    d: float = 0.60                # duty factor (stance fraction)
    v: float = 20.0                # cruise speed, cm/s
    B: float = 7.0                 # snout - tail base, cm
    score: str = "ok"              # ok | low
    missing: tuple = ()            # nodes with no coordinates for the whole clip
    dropouts: tuple = ()           # (node, offset from probe-pause start in frames, length)
    embed: str = "same"            # same | missing | unreadable | renamed
    roi: str = "ok"                # ok | missing | no_segment2 | invalid_yaml
    in_meta_trials: bool = True    # a (day, mouse) row in meta_trials.csv
    in_metadata: bool = True       # a Mouse row in metadata.csv
    note: str = ""


def simulate(c: Clip, fps: float):
    """Return S, side, visible per frame plus planted event frames."""
    S0 = 43.0 if c.start == "seg4" else 2.0
    S, side, vis = [], [], []
    ev = {"probe_pause_start": None, "probe_pause_len": 0, "entry_frame": None, "withdrawals": 0}

    def hold(sec, s, sd, visible=True):
        n = int(round(sec * fps))
        S.extend([s] * n); side.extend([sd] * n); vis.extend([visible] * n)

    def go(s0, s1, v, sd):
        seg = trapezoid(s0, s1, v, fps)
        S.extend(seg.tolist()); side.extend([sd] * len(seg)); vis.extend([True] * len(seg))

    beh = c.behaviour
    if beh.startswith("enter:"):
        sd = -1 if beh.endswith("R") else 1
        hold(c.absent_s, S0, sd, visible=False)
        hold(0.1 if c.start == "seg4" else 0.3, S0, sd)
        f0 = len(S)
        go(S0, S_END, c.v, sd)
        ev["entry_frame"] = f0 + int(np.argmax(np.asarray(S[f0:]) >= S_ENTRY))
        hold(0.4, S_END, sd)
    elif beh == "none":
        hold(c.absent_s, S0, 0, visible=False)
        hold(0.3, S0, 0)
        go(S0, 52.0, c.v, 0)
        hold(1.5, 52.0, 0)
    elif beh.startswith("probe:"):
        first, final = beh[6:].split(">")
        sd1 = -1 if first == "R" else 1
        sd2 = -1 if final == "R" else 1
        hold(c.absent_s, S0, sd1, visible=False)
        hold(0.3, S0, sd1)
        go(S0, S_PROBE, c.v, sd1)
        ev["probe_pause_start"] = len(S)
        hold(0.5, S_PROBE, sd1)
        ev["probe_pause_len"] = len(S) - ev["probe_pause_start"]
        go(S_PROBE, S_BACK, 12.0, sd1)
        ev["withdrawals"] = 1
        hold(0.15, S_BACK, sd2)
        f0 = len(S)
        go(S_BACK, S_END, c.v, sd2)
        ev["entry_frame"] = f0 + int(np.argmax(np.asarray(S[f0:]) >= S_ENTRY))
        hold(0.4, S_END, sd2)
    else:
        raise ValueError(beh)
    return np.asarray(S), np.asarray(side), np.asarray(vis, bool), ev


def pose_cm(S, side, c: Clip):
    """(T, 15, 2) keypoints in maze cm for body arc positions S on path `side`."""
    T = len(S)
    out = np.full((T, len(POSE_NODES), 2), np.nan)
    ph = 2 * np.pi * S / c.L
    for sd in np.unique(side):
        m = side == sd
        s, p = S[m], ph[m]

        def put(name, arc, lat):
            out[m, NI[name]] = offset(arc, sd, lat)

        put("tailbase", s, 0.10 * np.sin(p + 0.4))
        put("snout", s + c.B, 0.25 * np.sin(p + 1.9))
        put("mouth", s + c.B - 0.6, 0.22 * np.sin(p + 1.9))
        for i, name in enumerate(["tail1", "tail2", "tail3", "tailtip"], start=1):
            put(name, s - 2.0 * i, 0.15 * i * np.sin(p + 0.4 - 0.5 * i))
        # hind paws: left phase 0, right phase 0.5; fore paws diagonal to the hind paws
        for name_toe, name_heel, ph0, hip, lat in [
            ("hindpawL2", "hindpawL1", 0.0, 1.0, +1.3), ("hindpawR2", "hindpawR1", 0.5, 1.0, -1.3),
            ("forepawL2", "forepawL1", 0.5, 4.8, +1.0), ("forepawR2", "forepawR1", 0.0, 4.8, -1.0),
        ]:
            arc = paw_arc(s, c.L, c.d, ph0, hip)
            put(name_toe, arc, np.full_like(arc, lat))
            put(name_heel, arc - 0.5, np.full_like(arc, lat))
    return out


def probe_runs(pts_px, maze: Maze, before_frame, min_len=2, arm_max=40.0, junc_min=30.0):
    """Probe episodes per the definition in decision_analysis.count_snout_probes, computed here
    independently on the complete (no-dropout) pose: snout inside the arm polygon, body hull < 40%
    in that arm and >= 30% in the junction, for >= 2 consecutive frames, before the entry."""
    from shapely.geometry import MultiPoint, Point, Polygon
    polys = {n: Polygon(maze.region_px(n)) for n in ("arm_left", "arm_right", "junction")}
    hull_idx = [NI[n] for n in HULL_NODES]
    res = {}
    for arm, tag in (("arm_left", "L"), ("arm_right", "R")):
        flags = []
        for t in range(min(before_frame, len(pts_px))):
            sn = pts_px[t, NI["snout"]]
            body = pts_px[t, hull_idx]
            body = body[np.all(np.isfinite(body), axis=1)]
            ok = False
            if np.all(np.isfinite(sn)) and len(body) >= 3 and Point(sn).within(polys[arm]):
                hull = MultiPoint(body).convex_hull
                if hull.area > 0:
                    a = 100 * hull.intersection(polys[arm]).area / hull.area
                    j = 100 * hull.intersection(polys["junction"]).area / hull.area
                    ok = a < arm_max and j >= junc_min
            flags.append(ok)
        runs, n, frames = 0, 0, int(np.sum(flags))
        for f in flags + [False]:
            if f:
                n += 1
            else:
                runs += n >= min_len
                n = 0
        res[f"planted_probes_{tag}"], res[f"planted_probe_frames_{tag}"] = int(runs), frames
    return res


# ---- writers ---------------------------------------------------------------------------------
_BLANK = {}


def blank_video(cache_dir: Path, w, h, fps, n) -> Path:
    """One blank clip per (size, fps, length); every clip with the same spec is a copy of it."""
    import cv2
    key = (w, h, round(fps, 3), n)
    if key in _BLANK:
        return _BLANK[key]
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"blank_{w}x{h}_{fps:g}fps_{n}f.mp4"
    if not p.exists():
        frame = np.zeros((h, w, 3), np.uint8)
        for fourcc in ("VP90", "mp4v"):      # VP9 keeps a blank clip at a few tens of KB
            vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*fourcc), float(fps), (w, h))
            if vw.isOpened():
                for _ in range(n):
                    vw.write(frame)
                vw.release()
                break
        cap = cv2.VideoCapture(str(p))
        ok = cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == n
        cap.release()
        if not ok:
            raise RuntimeError(f"could not write a readable blank video: {p}")
    _BLANK[key] = p
    return p


def write_pose_slp(path: Path, video_path: Path, pts_px, scores, visible, embed_as=None):
    import sleap_io as sio
    skel = sio.Skeleton(nodes=list(POSE_NODES))   # sleap-io converts the list it is given in place
    video = sio.Video.from_filename(str(video_path))
    lfs = []
    for t in np.flatnonzero(visible):
        inst = sio.PredictedInstance.from_numpy(pts_px[t], skeleton=skel, point_scores=scores[t],
                                                score=float(np.nanmean(scores[t])))
        lfs.append(sio.LabeledFrame(video=video, frame_idx=int(t), instances=[inst]))
    labels = sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel])
    if embed_as is not None:        # the .slp remembers a path that is not where the video is now
        video.replace_filename(str(embed_as), open=False)
    labels.save(str(path))


def write_roi_slp(path: Path, video_path: Path, maze: Maze, drop_node=None):
    import sleap_io as sio
    skel = sio.Skeleton(nodes=list(ROI_NODES_23))
    video = sio.Video.from_filename(str(video_path))
    pts = maze.roi_keypoints_px().copy()
    if drop_node:
        pts[ROI_NODES_23.index(drop_node)] = np.nan
    inst = sio.PredictedInstance.from_numpy(pts, skeleton=skel, point_scores=np.full(len(pts), 0.8),
                                            score=0.8)
    lf = sio.LabeledFrame(video=video, frame_idx=0, instances=[inst])
    sio.Labels(labeled_frames=[lf], videos=[video], skeletons=[skel]).save(str(path))


def clip_rng(scn, stem):
    return np.random.default_rng([SEED, zlib.crc32(f"{scn}/{stem}".encode())])


@dataclass
class Scenario:
    name: str
    about: str
    clips: list = field(default_factory=list)
    maze: Maze = field(default_factory=Maze)
    fps: float = 120.0
    layout: str = "flat"                 # flat | per_day
    ext: str = ".mp4"
    roi_mode: str = "yml_dir"            # yml_dir (<stem>.preds.v2.best1.yml) | colocated (<stem>.rois.yml)
    pose_suffix: str = ".slp"
    sex_column: bool = True
    day_format: str = "Day{n}"           # how the day is written in meta_trials.csv
    batch_extra: dict = field(default_factory=dict)
    view_of: str | None = None           # config-only scenario reusing another scenario's files
    view_genotypes: dict | None = None   # mouse -> genotype override for a view (None value = drop)
    roi_slp: int = 0                     # how many ROI .slp files to write for `stride convert-roi`


def build_scenario(scn: Scenario, out: Path, cache: Path, base_cfg: dict):
    sdir = out / scn.name
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    vdir = sdir / "videos"
    vdir.mkdir()
    ydir = sdir / "roi_yml" if scn.roi_mode == "yml_dir" else vdir
    ydir.mkdir(exist_ok=True)
    scn.maze.check_fits()
    std_n, long_n = int(round(6.0 * scn.fps)), int(round(8.5 * scn.fps))
    truth = []
    for c in scn.clips:
        stem = f"Day{c.day}_{c.mouse}_Trial{c.trial}"
        rng = clip_rng(scn.name, stem)
        S, side, vis, ev = simulate(c, scn.fps)
        n = std_n if len(S) <= std_n else long_n
        assert len(S) <= n, (stem, len(S), n)
        pad = n - len(S)
        S, side, vis = np.r_[S, [S[-1]] * pad], np.r_[side, [side[-1]] * pad], np.r_[vis, [True] * pad]
        cm = pose_cm(S, side, c)
        px = scn.maze.to_px(cm) + rng.normal(0, 0.3, cm.shape)       # ~0.3 px keypoint noise
        complete = px.copy()                                          # pose before any dropout
        scores = rng.uniform(0.85, 0.98, px.shape[:2]) if c.score == "ok" else rng.uniform(0.10, 0.20, px.shape[:2])
        for node in c.missing:
            px[:, NI[node]] = np.nan
            scores[:, NI[node]] = 0.0
        for node, off, length in c.dropouts:
            a = ev["probe_pause_start"] + off
            px[a:a + length, NI[node]] = np.nan
        px[~vis] = np.nan

        cdir = vdir / f"DAY{c.day}" if scn.layout == "per_day" else vdir
        cdir.mkdir(exist_ok=True)
        video = cdir / f"{stem}{scn.ext}"
        shutil.copyfile(blank_video(cache, scn.maze.frame_w, scn.maze.frame_h, scn.fps, n), video)
        embed = {"same": None,
                 "missing": f"/data/other_machine/tmaze/{stem}.mp4",
                 "unreadable": f"/root/stride_mock_other_machine/CAM_{zlib.crc32(stem.encode()) % 10000:04d}.MP4",
                 "renamed": str(cdir / f"CAM_{zlib.crc32(stem.encode()) % 10000:04d}.MP4")}[c.embed]
        write_pose_slp(video.with_suffix(scn.pose_suffix), video, px, scores, vis, embed_as=embed)

        roi_name = f"{stem}.preds.v2.best1.yml" if scn.roi_mode == "yml_dir" else f"{stem}.rois.yml"
        roi_dir = ydir if scn.roi_mode == "yml_dir" else cdir
        if c.roi == "ok":
            (roi_dir / roi_name).write_text(yaml.safe_dump(scn.maze.roi_yaml(str(video)), sort_keys=False))
        elif c.roi == "no_segment2":
            (roi_dir / roi_name).write_text(yaml.safe_dump(scn.maze.roi_yaml(str(video), drop=("segment2",)), sort_keys=False))
        elif c.roi == "invalid_yaml":
            (roi_dir / roi_name).write_text("rois: [ {name: segment1, coordinates: [[0, 0], [1,\n")

        choice = "" if c.behaviour == "none" else c.behaviour[-1]
        entry = ev["entry_frame"]
        cL = S[vis] / c.L
        landings = int(np.clip(np.floor(cL[1:]) - np.floor(cL[:-1]), 0, None).sum())
        S_start = 43.0 if c.start == "seg4" else 2.0
        straight = max(int(math.floor((ARC_START - c.B) / c.L) - math.ceil((S_start + 3.0) / c.L)), 0)
        row = {
            "stem": stem, "video": str(video.relative_to(sdir)), "day": f"DAY{c.day}", "mouse": c.mouse,
            "trial": c.trial, "genotype": c.genotype, "sex": c.sex, "reward": c.reward,
            "behaviour": c.behaviour, "planted_choice": choice,
            "expected_correct": "" if choice == "" else int(choice == c.reward),
            "planted_entry_frame": "" if entry is None else int(entry),
            "planted_withdrawals": ev["withdrawals"],
            **probe_runs(complete, scn.maze, entry if entry is not None else len(S)),
            "fps": scn.fps, "n_frames": n, "frame_w": scn.maze.frame_w, "frame_h": scn.maze.frame_h,
            "px_per_cm": scn.maze.px_per_cm, "maze_rotation_deg": scn.maze.rotation_deg,
            "body_length_cm": c.B, "planted_stride_length_cm": c.L, "planted_duty": c.d,
            "planted_body_frame_excursion_cm": round(c.d * c.L, 4),
            "planted_speed_cm_s": c.v, "planted_stride_period_ms": round(1000 * c.L / c.v, 2),
            "planted_left_landings": landings, "planted_straight_strides": straight,
            "score_level": c.score, "missing_nodes": "|".join(c.missing),
            "dropouts": "|".join(f"{a}@{b}x{l}" for a, b, l in c.dropouts),
            "embedded_video": c.embed, "roi_file": c.roi,
            "in_meta_trials": int(c.in_meta_trials), "in_metadata": int(c.in_metadata), "note": c.note,
        }
        truth.append(row)

    with open(sdir / "truth.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(truth[0].keys()))
        w.writeheader(); w.writerows(truth)

    for i, c in enumerate(scn.clips[:scn.roi_slp]):      # ROI model output, for `stride convert-roi`
        rdir = sdir / "roi_slp"
        rdir.mkdir(exist_ok=True)
        stem = f"Day{c.day}_{c.mouse}_Trial{c.trial}"
        cdir = vdir / f"DAY{c.day}" if scn.layout == "per_day" else vdir
        write_roi_slp(rdir / f"{stem}.preds.v2.best1.slp", cdir / f"{stem}{scn.ext}", scn.maze,
                      drop_node="segment1.top_left" if i == scn.roi_slp - 1 and scn.roi_slp > 2 else None)

    write_tables(scn, sdir, scn.clips)
    write_config(scn, sdir, vdir, ydir, base_cfg)
    return len(scn.clips)


def write_tables(scn: Scenario, sdir: Path, clips, genotypes: dict | None = None):
    seen, rows = set(), []
    for c in clips:
        if c.in_meta_trials and (c.day, c.mouse) not in seen:
            seen.add((c.day, c.mouse))
            rows.append({"day": scn.day_format.format(n=c.day), "mouse": c.mouse, "reward": c.reward})
    with open(sdir / "meta_trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["day", "mouse", "reward"])
        w.writeheader(); w.writerows(rows)
    seen, rows = set(), []
    for c in clips:
        if not c.in_metadata or c.mouse in seen:
            continue
        seen.add(c.mouse)
        g = c.genotype
        if genotypes is not None:
            if c.mouse not in genotypes:
                continue
            g = genotypes[c.mouse]
        r = {"Mouse": c.mouse, "Gender": c.sex, "Genotype": g}
        if not scn.sex_column:
            r.pop("Gender")
        rows.append(r)
    with open(sdir / "metadata.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def write_config(scn: Scenario, sdir: Path, vdir: Path, ydir: Path, base_cfg: dict):
    """The shipped config.example.yaml with `batches:` replaced by one batch carrying exactly the fields the
    README lists for a new batch (plus pose_slp_suffix / px_per_cm_override where the scenario is
    about them).  gait_extraction.n_workers is lowered to 4 to keep the mock run light."""
    key = f"mock_{scn.name}"
    batch = {"name": f"Mock {scn.name}", "video_dir": str(vdir) + "/", "yml_dir": str(ydir) + "/",
             "meta_trials_csv": str(sdir / "meta_trials.csv"), "metadata_csv": str(sdir / "metadata.csv"),
             "fps": scn.fps if scn.fps % 1 else int(scn.fps),
             "column_map": {"mouse_id": ["mouse"], "sex": "Gender"}}
    if scn.pose_suffix != ".slp":
        batch["pose_slp_suffix"] = scn.pose_suffix
    batch.update(scn.batch_extra)
    cfg = copy.deepcopy(base_cfg)
    cfg["batches"] = {key: batch}
    cfg.setdefault("gait_extraction", {})["n_workers"] = 4
    (sdir / "config.readme.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    info = {"name": scn.name, "about": scn.about, "batch_key": key, "video_dir": str(vdir),
            "yml_dir": str(ydir), "roi_mode": scn.roi_mode, "pose_suffix": scn.pose_suffix,
            "layout": scn.layout, "ext": scn.ext, "fps": scn.fps, "px_per_cm": scn.maze.px_per_cm,
            "frame": [scn.maze.frame_w, scn.maze.frame_h], "maze_rotation_deg": scn.maze.rotation_deg,
            "meta_trials_csv": str(sdir / "meta_trials.csv"), "metadata_csv": str(sdir / "metadata.csv"),
            "config_readme": str(sdir / "config.readme.yaml"), "truth_csv": str(sdir / "truth.csv"),
            "roi_slp_dir": str(sdir / "roi_slp") if scn.roi_slp else None, "view_of": scn.view_of,
            "batch_extra": scn.batch_extra, "n_clips": len(scn.clips)}
    (sdir / "scenario.json").write_text(json.dumps(info, indent=2))


def build_view(scn: Scenario, src: Scenario, out: Path, base_cfg: dict):
    """A config-only scenario: same videos / pose / ROI as `src`, different genotype table."""
    sdir, src_dir = out / scn.name, out / src.name
    if sdir.exists():
        shutil.rmtree(sdir)
    sdir.mkdir(parents=True)
    scn.clips, scn.maze, scn.fps = src.clips, src.maze, src.fps
    scn.roi_mode, scn.pose_suffix, scn.layout, scn.ext = src.roi_mode, src.pose_suffix, src.layout, src.ext
    write_tables(scn, sdir, src.clips, genotypes=scn.view_genotypes)
    shutil.copyfile(src_dir / "truth.csv", sdir / "truth.csv")
    vdir = src_dir / "videos"
    write_config(scn, sdir, vdir, src_dir / "roi_yml" if src.roi_mode == "yml_dir" else vdir, base_cfg)
    return 0


# ---- the scenarios ---------------------------------------------------------------------------
def core_clips():
    rng = np.random.default_rng([SEED, 1])
    mice = [(str(101 + i), "WT") for i in range(5)] + [(str(106 + i), "Tau") for i in range(5)] + \
           [(str(111 + i), "APP") for i in range(3)]
    correct = {"WT": [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
               "Tau": [[1, 0, 0, 1], [0, 1, 0, 1], [1, 0, 1, 0], [0, 0, 1, 1], [0, 1, 1, 0]],
               "APP": [[1, 1, 0, 1], [1, 1, 1, 1], [1, 0, 1, 1]]}
    L0 = {"WT": 6.0, "APP": 6.0, "Tau": 5.0}
    clips, n_geno = [], {"WT": 0, "Tau": 0, "APP": 0}
    for idx, (mouse, geno) in enumerate(mice):
        k = n_geno[geno]; n_geno[geno] += 1
        Lm = L0[geno] + rng.normal(0, 0.2)
        vm = rng.uniform(18.5, 21.5)
        Bm = (6.8 if geno == "Tau" else 7.0) + rng.normal(0, 0.15)
        base = "R" if int(mouse) % 2 else "L"
        for j, (day, trial) in enumerate([(1, 1), (1, 2), (2, 1), (2, 2)]):
            reward = base
            if mouse in ("101", "106") and day == 2:          # reward reversal for two mice
                reward = "L" if base == "R" else "R"
            choice = reward if correct[geno][k][j] else ("L" if reward == "R" else "R")
            other = "L" if choice == "R" else "R"
            beh = f"probe:{other}>{choice}" if (day, trial) == (2, 2) else f"enter:{choice}"
            clips.append(Clip(day, mouse, trial, genotype=geno, sex="F" if idx % 2 == 0 else "M",
                              reward=reward, behaviour=beh,
                              L=round(float(Lm + rng.normal(0, 0.1)), 3),
                              d=0.62 if geno == "Tau" else 0.60,
                              v=round(float(np.clip(vm + rng.normal(0, 0.5), 18.0, 22.5)), 3),
                              B=round(float(Bm), 3)))
    return clips


def scenarios():
    S = []
    core = Scenario("core", "Reference batch: 13 mice (5 WT, 5 Tau, 3 APP), 2 days x 2 trials, flat folder, "
                    ".mp4, ROI ymls in a separate folder, 120 fps, 1920x1080 at 16 px/cm. Tau are planted with "
                    "a shorter stride (5 vs 6 cm) and lower accuracy; two mice have a reward reversal on day 2; "
                    "the last trial of every mouse is probe-then-withdraw-then-enter.", core_clips(), roi_slp=3)
    S.append(core)

    def two(mouse, **kw):
        return [Clip(1, mouse, 1, reward="L", behaviour="enter:L", **kw),
                Clip(1, mouse, 2, reward="L", behaviour="enter:R", **kw)]

    S.append(Scenario("layout_per_day_colocated", "Per-day subfolders (DAY1/, DAY2/), upper-case .MP4, ROI "
                      "co-located as <stem>.rois.yml (yml_dir in the config = the video root).",
                      [Clip(d, m, 1, genotype=g, reward="R", behaviour=f"enter:{ch}")
                       for d in (1, 2) for m, g, ch in (("501", "WT", "R"), ("502", "Tau", "L"))],
                      layout="per_day", ext=".MP4", roi_mode="colocated"))
    S.append(Scenario("layout_custom_pose_suffix", "Pose files named <stem>.pose.v2.slp (pose_slp_suffix in the "
                      "config).", two("511"), pose_suffix=".pose.v2.slp"))
    S.append(Scenario("ids_string", "String mouse ids: M4-1 and WT_3_1 (underscores inside the id).",
                      two("M4-1") + [Clip(1, "WT_3_1", 1, genotype="Tau", reward="R", behaviour="enter:R"),
                                     Clip(1, "WT_3_1", 2, genotype="Tau", reward="R", behaviour="enter:L")]))
    S.append(Scenario("meta_no_sex", "metadata.csv has no Gender column.",
                      two("521") + [Clip(1, "522", 1, genotype="Tau", reward="R", behaviour="enter:R")],
                      sex_column=False))
    S.append(Scenario("meta_other_genotype_labels", "Genotype labels are KO / Ctrl, not WT / APP / Tau.",
                      [Clip(1, m, t, genotype=g, reward="L", behaviour="enter:L")
                       for m, g in (("531", "Ctrl"), ("532", "KO")) for t in (1, 2)]))
    S.append(Scenario("meta_missing_rows", "Mouse 541 has a reward row for day 1 but none for day 2; mouse 542 "
                      "has no row in meta_trials.csv and none in metadata.csv.",
                      [Clip(1, "541", 1, reward="L", behaviour="enter:L"),
                       Clip(2, "541", 1, reward="L", behaviour="enter:L", in_meta_trials=False,
                            note="no reward row for (DAY2, 541)"),
                       Clip(1, "542", 1, reward="R", behaviour="enter:R", in_meta_trials=False, in_metadata=False,
                            note="mouse absent from both tables"),
                       Clip(1, "542", 2, reward="R", behaviour="enter:L", in_meta_trials=False, in_metadata=False,
                            note="mouse absent from both tables")]))
    S.append(Scenario("meta_day_as_number", "meta_trials.csv writes the day as 1 / 2, not Day1 / Day2.",
                      [Clip(1, "551", 1, reward="L", behaviour="enter:L"),
                       Clip(2, "551", 1, reward="R", behaviour="enter:R")], day_format="{n}"))

    S.append(Scenario("behaviour", "One mouse, one clip per behaviour (day 1 reward L, day 2 reward R).", [
        Clip(1, "201", 1, reward="L", behaviour="enter:L", note="correct, left"),
        Clip(1, "201", 2, reward="L", behaviour="enter:R", note="wrong arm"),
        Clip(1, "201", 3, reward="L", behaviour="none", note="no decision: stops before the junction"),
        Clip(1, "201", 4, reward="L", behaviour="probe:R>L", note="probe right, withdraw, enter left"),
        Clip(1, "201", 5, reward="L", behaviour="enter:L", start="seg4", absent_s=0.0,
             note="already in segment4 (next to the junction) at frame 0"),
        Clip(1, "201", 6, reward="L", behaviour="enter:L", absent_s=0.0,
             note="pose present in the start segment from frame 0"),
        Clip(2, "201", 1, reward="R", behaviour="enter:R", note="correct, right"),
        Clip(2, "201", 2, reward="R", behaviour="probe:L>R", note="probe left, withdraw, enter right"),
    ]))

    S.append(Scenario("pose_snout_dropout", "Same probe-right-then-enter-left trial twice; in trial 2 the snout "
                      "keypoint is missing for 1, 3 and 7 frames INSIDE the probe (tracking dropouts).", [
        Clip(1, "301", 1, reward="L", behaviour="probe:R>L", note="complete pose"),
        Clip(1, "301", 2, reward="L", behaviour="probe:R>L",
             dropouts=(("snout", 10, 1), ("snout", 25, 3), ("snout", 40, 7)), note="snout dropouts in the probe"),
    ]))
    S.append(Scenario("pose_missing_keypoint", "One keypoint absent for the whole clip.", [
        Clip(1, "311", 1, reward="L", behaviour="enter:L", missing=("tailtip",), note="tailtip never tracked"),
        Clip(1, "311", 2, reward="L", behaviour="enter:L", missing=("hindpawL2",), note="hindpawL2 never tracked"),
        Clip(1, "311", 3, reward="L", behaviour="enter:L", missing=("snout",), note="snout never tracked"),
    ]))
    S.append(Scenario("pose_low_confidence", "Trial 2 has keypoint scores of 0.10-0.20 (below the 0.3 filter).", [
        Clip(1, "321", 1, reward="L", behaviour="enter:L"),
        Clip(1, "321", 2, reward="L", behaviour="enter:L", score="low", note="low keypoint confidence"),
    ]))
    S.append(Scenario("pose_foreign_video_path", "The .slp remembers the video path of another machine: trial 1 the "
                      "same file name in a folder that does not exist here; trial 2 a camera file name under /root "
                      "(pose run as root in a container before the videos were renamed; a normal user cannot read /root).", [
        Clip(1, "331", 1, reward="L", behaviour="enter:L", embed="missing", note="embedded path does not exist"),
        Clip(1, "331", 2, reward="L", behaviour="enter:L", embed="unreadable", note="embedded path under /root"),
    ]))
    S.append(Scenario("pose_before_rename", "Pose was run before the videos were renamed: the .slp remembers a "
                      "camera file name, the file on disk is Day1_341_TrialN.mp4.",
                      [Clip(1, "341", t, reward="L", behaviour="enter:L", embed="renamed") for t in (1, 2)]))

    S.append(Scenario("calib_roi_problems", "Trial 1 has a good ROI file, trial 2 none, trial 3 an ROI file without "
                      "segment2, trial 4 an ROI file that is not valid YAML.", [
        Clip(1, "351", 1, reward="L", behaviour="enter:L"),
        Clip(1, "351", 2, reward="L", behaviour="enter:L", roi="missing", note="no ROI file"),
        Clip(1, "351", 3, reward="L", behaviour="enter:L", roi="no_segment2", note="ROI file lacks segment2"),
        Clip(1, "351", 4, reward="L", behaviour="enter:L", roi="invalid_yaml", note="ROI file is not valid YAML"),
    ]))
    S.append(Scenario("calib_override", "Batch-level px_per_cm_override in the config (16.0, the true value).",
                      two("356"), batch_extra={"px_per_cm_override": 16.0}))

    S.append(Scenario("camera_fps100", "100 fps camera.", two("361"), fps=100.0))
    S.append(Scenario("camera_fps30", "30 fps camera.", two("371"), fps=30.0))
    S.append(Scenario("camera_4k", "3840x2160 at 26 px/cm (the scale of a 4K action camera under the maze).",
                      two("381"), maze=Maze(26.0, 3840, 2160)))
    S.append(Scenario("camera_lowres", "1280x720 at 10 px/cm (a low-resolution machine-vision camera).",
                      two("391"), maze=Maze(10.0, 1280, 720)))
    S.append(Scenario("camera_rotated15", "Camera rotated 15 degrees against the maze (13 px/cm).",
                      two("401"), maze=Maze(13.0, 1920, 1080, rotation_deg=15.0)))
    S.append(Scenario("camera_vertical90", "Maze drawn vertically in the image (stem runs top to bottom).",
                      two("411"), maze=Maze(16.0, 1920, 1080, rotation_deg=90.0)))

    core_mice = {c.mouse: c.genotype for c in core.clips}
    S.append(Scenario("batch_one_genotype", "Core videos, every mouse labelled WT.", view_of="core",
                      view_genotypes={m: "WT" for m in core_mice}))
    S.append(Scenario("batch_wt_tau", "Core videos, only the WT and Tau mice are in metadata.csv.", view_of="core",
                      view_genotypes={m: g for m, g in core_mice.items() if g != "APP"}))
    S.append(Scenario("batch_too_few_mice", "Core videos, only one WT and one Tau mouse in metadata.csv.",
                      view_of="core", view_genotypes={"101": "WT", "106": "Tau"}))
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=None, help="output directory (created; scenario folders are replaced)")
    ap.add_argument("--only", default=None, help="comma-separated scenario names")
    ap.add_argument("--list", action="store_true", help="list the scenarios and exit")
    a = ap.parse_args()
    scns = scenarios()
    if a.list:
        for s in scns:
            print(f"{s.name:28s} {len(s.clips):3d} clips  {s.about}")
        return
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base_cfg = yaml.safe_load((REPO / "config.example.yaml").read_text(encoding="utf-8"))
    by_name = {s.name: s for s in scns}
    want = set(a.only.split(",")) if a.only else set(by_name)
    total = 0
    for s in scns:
        if s.name not in want:
            continue
        if s.view_of:
            if not (out / s.view_of / "truth.csv").exists():
                raise SystemExit(f"{s.name} needs scenario '{s.view_of}' in {out} — generate it first")
            build_view(s, by_name[s.view_of], out, base_cfg)
            print(f"  {s.name:28s} view of {s.view_of}")
        else:
            n = build_scenario(s, out, out / "_blank_cache", base_cfg)
            total += n
            print(f"  {s.name:28s} {n:3d} clips", flush=True)
    # the index lists every scenario present in the output folder (so --only does not drop the others)
    index = [{"name": s.name, "about": s.about, "dir": str(out / s.name)}
             for s in scns if (out / s.name / "scenario.json").exists()]
    (out / "scenarios.json").write_text(json.dumps(index, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and "_blank_cache" not in p.parts)
    print(f"wrote {total} clips; {len(index)} scenarios in {out}, {size / 1e6:.1f} MB (excluding the blank-video cache)")


if __name__ == "__main__":
    main()
