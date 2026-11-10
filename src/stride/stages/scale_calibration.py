"""Scale calibration: px/cm from a ChArUco board that is in view of the camera.

Standalone: needs only OpenCV (with the ArUco module) and NumPy. Give it a folder of videos, or a list of
clips, and it returns one scale for the recording set-up, with its spread, a per-clip table and QC images.

    from stride.stages.scale_calibration import calibrate_px_per_cm
    r = calibrate_px_per_cm("/data/my_cohort", out_dir="/data/my_cohort/calibration")
    r["ok"], r["px_per_cm"]      # True, 26.02  -> the gait stage's `px_per_cm_override`

    stride calibrate /data/my_cohort            # the same from the command line

Method. Per clip, frames are read every 0.15 s over the first 10 s. In each frame the ChArUco corners are
detected and a homography from the board's own grid (in squares) to the image is fitted (RANSAC, 3 px). The
scale of that frame is the pixel length of one square at the centre of the detected corners, averaged over
the board's two axes. The frame with the best score (many corners, low reprojection error, low anisotropy)
gives the clip's reading: px/cm = pixels per square / square size in cm.

The value for the set-up is the MEDIAN of the clip readings, to two decimals. It is accepted only with at
least 3 usable readings whose spread, (max - min) / median, is below 1 %; otherwise a CalibrationError says why.

Default board: 14 x 9 squares, 20 mm squares, 15 mm markers, dictionary DICT_5X5_250.

Portable: pathlib paths only, no shell, no links, UTF-8 text files; runs the same on Windows, macOS and Linux.
Every setting is a plain keyword argument, so a YAML block can drive it:

    calibrate:
      squares_x: 14
      squares_y: 9
      square_mm: 20
      marker_mm: 15
      dictionary: DICT_5X5_250
      max_clips: 12

    calibrate_px_per_cm(video_dir, out_dir, **config["calibrate"])
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Union

METHOD = "charuco_homography_median"
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")
SKIP_DIRS = ("waste",)                       # plus any folder whose name starts with "." or "_"


class CalibrationError(RuntimeError):
    """No scale could be accepted. The message is the reason."""


@dataclass(frozen=True)
class BoardSpec:
    """The printed ChArUco board. `square_mm` is what turns pixels into centimetres — measure it on the print."""
    squares_x: int = 14
    squares_y: int = 9
    square_mm: float = 20.0
    marker_mm: float = 15.0
    dictionary: str = "DICT_5X5_250"

    @property
    def square_cm(self) -> float:
        return self.square_mm / 10.0

    def describe(self) -> str:
        return (f"ChArUco {self.squares_x}x{self.squares_y}, {self.square_mm:g} mm squares, "
                f"{self.marker_mm:g} mm markers, {self.dictionary}")


DEFAULT_BOARD = BoardSpec()
_local = threading.local()


def _imwrite(path, img) -> bool:
    """cv2.imwrite fails on non-ASCII paths on Windows; encode in memory and let pathlib write the bytes."""
    import cv2

    path = Path(path)
    ok, buf = cv2.imencode(path.suffix, img)
    if ok:
        path.write_bytes(buf.tobytes())
    return bool(ok)


def make_board(board=None, squares_x: Optional[int] = None, squares_y: Optional[int] = None, square_mm: Optional[float] = None,
               marker_mm: Optional[float] = None, dictionary: Optional[str] = None) -> BoardSpec:
    """A BoardSpec from whatever a config gives: None (default board), a BoardSpec, a mapping
    ({squares_x, squares_y, square_mm, marker_mm, dictionary}, or {squares: "14x9", ...}), a "14x9" string;
    the individual keyword arguments override it."""
    if board is None:
        base = {}
    elif isinstance(board, BoardSpec):
        base = dict(squares_x=board.squares_x, squares_y=board.squares_y, square_mm=board.square_mm,
                    marker_mm=board.marker_mm, dictionary=board.dictionary)
    elif isinstance(board, str):
        base = {"squares": board}
    elif isinstance(board, dict):
        base = dict(board)
    else:
        raise CalibrationError(f"board must be None, a BoardSpec, a mapping or a string like '14x9' (got {type(board).__name__})")
    sq = base.pop("squares", None) or base.pop("board_squares", None)
    if sq is not None:
        try:
            base["squares_x"], base["squares_y"] = (int(v) for v in str(sq).lower().split("x"))
        except ValueError:
            raise CalibrationError(f"board squares must look like 14x9 (got {sq!r})")
    for k, v in (("squares_x", squares_x), ("squares_y", squares_y), ("square_mm", square_mm), ("marker_mm", marker_mm),
                 ("dictionary", dictionary)):
        if v is not None:
            base[k] = v
    unknown = sorted(set(base) - {"squares_x", "squares_y", "square_mm", "marker_mm", "dictionary"})
    if unknown:
        raise CalibrationError(f"unknown board setting(s): {', '.join(unknown)}")
    d = DEFAULT_BOARD
    try:
        return BoardSpec(int(base.get("squares_x", d.squares_x)), int(base.get("squares_y", d.squares_y)),
                         float(base.get("square_mm", d.square_mm)), float(base.get("marker_mm", d.marker_mm)),
                         str(base.get("dictionary", d.dictionary)))
    except (TypeError, ValueError) as e:
        raise CalibrationError(f"board settings must be numbers (squares, mm) and a dictionary name: {e}")


def _detector(board: BoardSpec):
    """(detector, board corner coordinates in metres); one detector per thread."""
    import cv2

    cache = getattr(_local, "cache", None)
    if cache is None:
        cache = _local.cache = {}
    if board not in cache:
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
            raise CalibrationError("this OpenCV build has no ChArUco detector (cv2.aruco.CharucoDetector) — "
                                   "install opencv-python >= 4.8 or opencv-contrib-python")
        if not hasattr(cv2.aruco, board.dictionary):
            raise CalibrationError(f"unknown ArUco dictionary {board.dictionary!r} (e.g. DICT_5X5_250, DICT_4X4_50)")
        if board.squares_x < 3 or board.squares_y < 3 or not (0 < board.marker_mm < board.square_mm):
            raise CalibrationError(f"impossible board: {board.describe()} (needs >= 3 squares per side and marker < square)")
        ad = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, board.dictionary))
        b = cv2.aruco.CharucoBoard((board.squares_x, board.squares_y), board.square_mm / 1000.0, board.marker_mm / 1000.0, ad)
        cache[board] = (cv2.aruco.CharucoDetector(b), b.getChessboardCorners())
    return cache[board]


def charuco_measure(gray, board: BoardSpec = DEFAULT_BOARD) -> Optional[dict]:
    """One grey frame -> dict(n, nn_px, scale_h_px, reproj_px, aniso, ux, uy, + detections), or None (< 8 corners)."""
    import cv2
    import numpy as np

    det, OBJ = _detector(board)
    chC, chI, mc, mid = det.detectBoard(gray)
    if chC is None or len(chC) < 8:
        return None
    ids = chI.ravel()
    img = chC.reshape(-1, 2).astype(np.float64)
    obj = OBJ[ids][:, :2] / (board.square_mm / 1000.0)          # metres -> square units (integer grid coordinates)
    n = len(img)
    # (1) raw nearest-neighbour spacing between detected corners (px per square, foreshortened)
    d = np.linalg.norm(img[:, None] - img[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.median(d.min(1))
    # (2) homography-corrected: plane mapping square grid -> pixels; scale = px per square at the board centre
    scale_h = res = aniso = ux = uy = None
    if n >= 6:
        H, _mask = cv2.findHomography(obj.astype(np.float32), img.astype(np.float32), cv2.RANSAC, 3.0)
        if H is not None:
            c = obj.mean(0)

            def proj(p):
                v = H @ np.array([p[0], p[1], 1.0])
                return v[:2] / v[2]

            ux = np.linalg.norm(proj(c + [1, 0]) - proj(c - [1, 0])) / 2.0
            uy = np.linalg.norm(proj(c + [0, 1]) - proj(c - [0, 1])) / 2.0
            scale_h = (ux + uy) / 2.0
            proj_all = cv2.perspectiveTransform(obj.reshape(-1, 1, 2).astype(np.float32), H).reshape(-1, 2)
            res = float(np.median(np.linalg.norm(proj_all - img, axis=1)))
            aniso = abs(ux - uy) / ((ux + uy) / 2)               # anisotropy = tilt indicator
    return dict(n=n, nn_px=float(nn), scale_h_px=None if scale_h is None else float(scale_h), reproj_px=res,
                aniso=None if scale_h is None else float(aniso), ux=None if scale_h is None else float(ux),
                uy=None if scale_h is None else float(uy), chC=chC, chI=chI, mc=mc, mid=mid)


def scan_times(scan_seconds: float = 10.0, step_seconds: float = 0.15, frames_per_clip: Optional[int] = None):
    """The time points examined in a clip: every `step_seconds` over the first `scan_seconds` (67 frames by
    default), or `frames_per_clip` evenly spaced frames over the same window."""
    import numpy as np

    if frames_per_clip:
        return np.linspace(0, scan_seconds, int(frames_per_clip), endpoint=False)
    return np.arange(0, scan_seconds, step_seconds)


def scan_charuco(path, board: BoardSpec = DEFAULT_BOARD, scan_seconds: float = 10.0, step_seconds: float = 0.15,
                 frames_per_clip: Optional[int] = None):
    """Best frame of the first `scan_seconds` -> (score, t, measurement, frame), or None when no frame has the board."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    best = None
    for t in scan_times(scan_seconds, step_seconds, frames_per_clip):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        m = charuco_measure(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), board)
        if m is None or m["scale_h_px"] is None:
            continue
        # score: many corners, low reprojection error, low anisotropy (a flat board facing the camera)
        score = m["n"] - 100 * (m["reproj_px"] or 5) - 200 * (m["aniso"] or 1)
        if best is None or score > best[0]:
            best = (score, float(t), m, fr)
    cap.release()
    return best


def rail_measure(path, crop: Sequence[int], out_png=None, corridor_cm: float = 10.0, expect_pxcm: float = 27.2):
    """Optional cross-check, never used for the value: the width of a straight corridor of known width, from the
    near-horizontal Hough lines inside `crop` = (y0, y1, x0, x1). Returns (all readings, best reading)."""
    import cv2
    import numpy as np

    y0, y1, x0, x1 = crop
    cap = cv2.VideoCapture(str(path))
    results, best, best_vis = [], None, None
    for t in [1.0, 3.0, 6.0, 9.0]:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, fr = cap.read()
        if not ok:
            continue
        sub = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        lines = cv2.HoughLinesP(cv2.Canny(sub, 40, 120), 1, np.pi / 180, threshold=120, minLineLength=300, maxLineGap=40)
        if lines is None:
            continue
        horiz = []
        for seg in np.asarray(lines).reshape(-1, 4):             # (N, 1, 4) in OpenCV 4, (N, 4) in OpenCV 5
            xa, ya, xb, yb = (int(v) for v in seg)
            if abs(np.degrees(np.arctan2(yb - ya, xb - xa))) < 8:
                horiz.append(((ya + yb) / 2.0, (xa, ya, xb, yb)))
        if len(horiz) < 2:
            continue
        ys = np.sort(np.array([h[0] for h in horiz]))
        gaps = np.diff(ys)
        if len(gaps) == 0:
            continue
        gi = int(np.argmax(gaps))
        if gaps[gi] < 120:                                       # the two rails must be well apart
            continue
        top, bot = ys[:gi + 1], ys[gi + 1:]
        width = float(np.median(bot) - np.median(top))
        results.append((t, width, len(horiz)))
        if best is None or abs(width / corridor_cm - expect_pxcm) < abs(best[1] / corridor_cm - expect_pxcm):
            best = (t, width, len(horiz))
            if out_png is not None:
                vis = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
                for _, seg in horiz:
                    xa, ya, xb, yb = seg
                    cv2.line(vis, (int(xa), int(ya)), (int(xb), int(yb)), (0, 255, 0), 2)
                cv2.line(vis, (0, int(np.median(top))), (sub.shape[1], int(np.median(top))), (0, 0, 255), 3)
                cv2.line(vis, (0, int(np.median(bot))), (sub.shape[1], int(np.median(bot))), (255, 0, 0), 3)
                cv2.putText(vis, f"t={t} width={width:.1f}px pxcm={width / corridor_cm:.2f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                best_vis = vis
    cap.release()
    if best_vis is not None and out_png is not None:
        _imwrite(out_png, best_vis)
    return results, best


def measure_clip(path, out_dir=None, board: BoardSpec = DEFAULT_BOARD, rail_crop: Optional[Sequence[int]] = None,
                 rail_frame_size: Optional[Sequence[int]] = None, scan_seconds: float = 10.0, step_seconds: float = 0.15,
                 frames_per_clip: Optional[int] = None) -> dict:
    """One clip -> its reading (JSON-able). With out_dir: also <stem>.json and the QC image <stem>_checkerboard.png.

    {"stem", "path", "charuco": None | {t, n, nn_px, scale_h_px, reproj_px, aniso, ux, uy, pxcm_nn, pxcm_H},
     "rail_all": [...], "rail_best": None | {...}}  — `pxcm_H` is the clip's px/cm.
    """
    import cv2

    path = Path(path)
    stem = path.stem
    out = {"stem": stem, "path": str(path)}
    b = scan_charuco(path, board, scan_seconds, step_seconds, frames_per_clip)
    if b is not None:
        _score, t, m, fr = b
        out["charuco"] = dict(t=round(t, 2), n=m["n"], nn_px=round(m["nn_px"], 3), scale_h_px=round(m["scale_h_px"], 3),
                              reproj_px=round(m["reproj_px"], 3), aniso=round(m["aniso"], 4), ux=round(m["ux"], 3),
                              uy=round(m["uy"], 3), pxcm_nn=round(m["nn_px"] / board.square_cm, 3),
                              pxcm_H=round(m["scale_h_px"] / board.square_cm, 3))
        if out_dir is not None:
            vis = fr.copy()
            try:                                                  # QC drawing only: must never cost the reading
                import numpy as np
                cv2.aruco.drawDetectedCornersCharuco(vis, np.asarray(m["chC"], np.float32).reshape(-1, 1, 2),
                                                     np.asarray(m["chI"], np.int32).reshape(-1, 1), (0, 0, 255))
                if m["mid"] is not None:
                    cv2.aruco.drawDetectedMarkers(vis, m["mc"], m["mid"])
            except cv2.error:
                pass
            cv2.putText(vis, f"{board.squares_x}x{board.squares_y} charuco t={t:.2f} n={m['n']} pitchH={m['scale_h_px']:.2f}px "
                             f"pxcm={m['scale_h_px'] / board.square_cm:.2f} aniso={m['aniso']:.3f} reproj={m['reproj_px']:.2f}",
                        (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            _imwrite(Path(out_dir) / f"{stem}_checkerboard.png", vis)
    else:
        out["charuco"] = None
    out["rail_all"], out["rail_best"] = [], None
    if rail_crop is not None:
        size_ok = True
        if rail_frame_size is not None:
            cap = cv2.VideoCapture(str(path))
            size_ok = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == tuple(rail_frame_size)
            cap.release()
        rr, rb = [], None
        if size_ok:
            try:                                                  # a cross-check must never cost the reading
                rr, rb = rail_measure(path, rail_crop, None if out_dir is None else Path(out_dir) / f"{stem}_rail.png")
            except Exception:
                rr, rb = [], None
        if size_ok:
            out["rail_all"] = [dict(t=r[0], width_px=round(r[1], 2), nlines=r[2], pxcm=round(r[1] / 10.0, 3)) for r in rr]
            out["rail_best"] = None if rb is None else dict(t=rb[0], width_px=round(rb[1], 2), pxcm=round(rb[1] / 10.0, 3))
    if out_dir is not None:
        (Path(out_dir) / f"{stem}.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
    return out


def find_videos(folder, suffixes: Iterable[str] = VIDEO_SUFFIXES) -> list:
    """Clips under `folder`, grouped by the sub-folder they are in: [[clips of folder 1, sorted], ...].

    Recursive and case-insensitive (.mp4 / .MP4). Folders named `waste`, and folders whose name starts with
    "." or "_", are skipped. Groups are ordered by folder name with numbers compared as numbers (DAY2 < DAY10).
    """
    import re

    root = Path(folder)
    suffixes = tuple(s.lower() for s in suffixes)
    groups = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).parts[:-1]
        if any(part.startswith((".", "_")) or part.lower() in SKIP_DIRS for part in rel):
            continue
        if p.is_file() and p.suffix.lower() in suffixes and not p.name.startswith("."):
            groups.setdefault(p.parent, []).append(p)

    def natural(d: Path):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(d.relative_to(root)))]

    return [sorted(groups[d]) for d in sorted(groups, key=natural)]


def pick_clips(groups: Sequence[Sequence[Path]], minimum: int = 6) -> list:
    """The first clip of each group (recording day); with fewer groups than `minimum`, further clips of the same
    groups are added (the 11th, 18th ... clip of each, or one from the middle of what is left on short days)."""
    lists = [list(l) for l in groups if l]
    clips = [l[0] for l in lists]
    while lists and len(clips) < minimum:
        k = len(clips)
        l = lists[k % len(lists)]
        i = k // len(lists) * 7 + 3
        if i < len(l) and l[i] not in clips:
            clips.append(l[i])
            continue
        rest = [c for c in l if c not in clips] or [c for ll in lists for c in ll if c not in clips]
        if not rest:
            break
        clips.append(rest[len(rest) // 2])
    return clips


def summarize(readings: Sequence[dict], min_readings: int = 3, max_spread_pct: float = 1.0) -> dict:
    """Clip readings -> {"value" (None if not accepted), "median", "spread_pct", "n_usable", "usable", "rails", "reason"}."""
    vals, rails = [], []
    for d in readings:
        ch = d.get("charuco")
        if ch and ch.get("pxcm_H"):
            vals.append((d["stem"], ch["pxcm_H"], ch["n"], ch["aniso"], ch["reproj_px"]))
        if d.get("rail_best"):
            rails.append(d["rail_best"]["pxcm"])
    out = dict(value=None, median=None, spread_pct=None, n_usable=len(vals), usable=vals, rails=rails, reason="")
    if len(vals) < min_readings:
        out["reason"] = (f"only {len(vals)} usable ChArUco reading(s) from {len(readings)} clip(s) (need >= {min_readings}). "
                         "Is the board in view during the first 10 s of the clips, flat and unobstructed, and does the board "
                         "specification (squares, sizes, dictionary) match the print?")
        return out
    xs = [v for _, v, *_ in vals]
    med = st.median(xs)
    spread = (max(xs) - min(xs)) / med * 100
    out.update(median=med, spread_pct=spread)
    if spread < max_spread_pct:
        out["value"] = round(med, 2)
    else:
        out["reason"] = (f"the readings spread by {spread:.2f} % (limit {max_spread_pct:g} %): the board or the camera moved "
                         "between clips, or a reading is bad — look at the *_checkerboard.png images and the per-clip table")
    return out


def _table(readings: Sequence[dict]) -> list:
    rows = []
    for d in readings:
        ch = d.get("charuco") or {}
        rows.append(dict(clip=Path(d.get("path", d["stem"])).name, px_per_cm=ch.get("pxcm_H", ""), usable=bool(ch.get("pxcm_H")),
                         corners=ch.get("n", 0), anisotropy=ch.get("aniso", ""), reprojection_px=ch.get("reproj_px", ""),
                         frame_time_s=ch.get("t", ""), px_per_square=ch.get("scale_h_px", ""),
                         rail_px_per_cm=(d.get("rail_best") or {}).get("pxcm", "")))
    return rows


def limit_clips(clips: Sequence, max_clips: Optional[int]) -> list:
    """At most `max_clips` clips, evenly spread over the list (so the days stay covered)."""
    clips = list(clips)
    if not max_clips or len(clips) <= max_clips:
        return clips
    if max_clips == 1:
        return [clips[0]]
    idx = sorted({round(i * (len(clips) - 1) / (max_clips - 1)) for i in range(max_clips)})
    return [clips[i] for i in idx]


def calibrate_px_per_cm(videos: Union[str, Path, Sequence], out_dir=None, board=None, *,
                        squares_x: Optional[int] = None, squares_y: Optional[int] = None, square_mm: Optional[float] = None,
                        marker_mm: Optional[float] = None, dictionary: Optional[str] = None,
                        min_clips: int = 6, max_clips: Optional[int] = None,
                        scan_seconds: float = 10.0, step_seconds: float = 0.15, frames_per_clip: Optional[int] = None,
                        min_readings: int = 3, max_spread_pct: float = 1.0, workers: int = 1,
                        rail_crop: Optional[Sequence[int]] = None, rail_frame_size: Optional[Sequence[int]] = None,
                        log: Optional[Callable[[str], None]] = None) -> dict:
    """px/cm of a recording set-up from the ChArUco board in view. Never raises for a calibration that cannot be
    accepted: look at `ok` and `reason`.

    videos            a folder (clips are found recursively; the first clip of each sub-folder is measured, at
                      least `min_clips` and at most `max_clips` in total), one clip, or a list of clips (all measured,
                      up to `max_clips`).
    out_dir           where the files go. Default: `<folder>/calibration`, or `calibration` in the working directory.
    board             None (default board), a BoardSpec, a mapping or "14x9"; or give the board as plain keywords:
    squares_x, squares_y, square_mm, marker_mm, dictionary
    frames_per_clip   frames examined per clip, evenly spaced over the first `scan_seconds`; default: one every
                      `step_seconds` (67 frames).
    min_readings, max_spread_pct   the acceptance rule: at least 3 usable clips spreading by less than 1 %.

    Returns a dict:
        ok             True when a value was accepted and written
        px_per_cm      the accepted value (median of the clip readings, two decimals), else None
        spread_pct     (max - min) / median of the usable readings, in %, else None
        n_clips_used   clips that gave a reading
        n_clips        clips examined
        per_clip       one dict per clip: clip, px_per_cm, usable, corners, anisotropy, reprojection_px, frame_time_s, ...
        method         "charuco_homography_median"
        reason         "" when ok, else why no value was accepted
        median, board, out_dir, px_per_cm_txt, summary_txt, readings_csv, readings
    """
    from concurrent.futures import ThreadPoolExecutor

    say = log or (lambda m: print(f"[calibrate] {m}", flush=True))
    res = dict(ok=False, px_per_cm=None, spread_pct=None, n_clips_used=0, n_clips=0, per_clip=[], method=METHOD, reason="",
               median=None, board=None, out_dir=None, px_per_cm_txt=None, summary_txt=None, readings_csv=None, readings=[])

    def fail(reason):
        res["reason"] = reason
        return res

    try:
        spec = make_board(board, squares_x, squares_y, square_mm, marker_mm, dictionary)
        _detector(spec)                                           # a bad board / OpenCV build fails here, before any work
    except CalibrationError as e:
        return fail(str(e))
    res["board"] = spec.describe()
    if isinstance(videos, (str, Path)):
        src = Path(videos)
        if src.is_file():
            clips, default_out = [src], Path("calibration")
        elif src.is_dir():
            clips, default_out = pick_clips(find_videos(src), minimum=int(min_clips)), src / "calibration"
            if not clips:
                return fail(f"no video files found under {src}")
        else:
            return fail(f"not found: {src}")
    else:
        clips, default_out = [Path(c) for c in videos], Path("calibration")
        if not clips:
            return fail("no clips given")
        missing = [c for c in clips if not c.is_file()]
        if missing:
            return fail(f"clip not found: {missing[0]}")
    clips = limit_clips(clips, max_clips)
    out_dir = Path(out_dir) if out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)
    say(f"{spec.describe()}; homography on {len(clips)} clips: {' '.join(c.name for c in clips)}")

    def one(c):
        return measure_clip(c, out_dir, spec, rail_crop, rail_frame_size, scan_seconds, step_seconds, frames_per_clip)

    if workers and int(workers) > 1 and len(clips) > 1:
        with ThreadPoolExecutor(max_workers=min(int(workers), len(clips))) as ex:   # OpenCV releases the GIL while decoding
            readings = list(ex.map(one, clips))
    else:
        readings = [one(c) for c in clips]
    s = summarize(readings, min_readings=int(min_readings), max_spread_pct=float(max_spread_pct))
    table = _table(readings)
    L = [("ChArUco readings (homography, px/cm): " + ", ".join(f"{stem}: {v:.2f} (n={n}, aniso={a:.3f}, reproj={r:.2f}px)"
                                                                 for stem, v, n, a, r in s["usable"])) if s["usable"]
         else "ChArUco readings: NONE detected"]
    no_board = [r["clip"] for r in table if not r["usable"]]
    if no_board:
        L.append(f"no board found in {len(no_board)} clip(s): {', '.join(no_board)}")
    if s["rails"]:
        L.append("rail cross-check (px/cm): " + ", ".join(f"{x:.2f}" for x in s["rails"]))
    if s["median"] is not None:
        L.append(f"median {s['median']:.2f} px/cm over {s['n_usable']} clips, spread {s['spread_pct']:.2f}%")
    pxcm_txt, summary_txt, readings_csv = out_dir / "px_per_cm.txt", out_dir / "calibration_summary.txt", out_dir / "calibration_readings.csv"
    if s["value"] is not None:
        pxcm_txt.write_text(f"{s['value']:.2f}\n", encoding="utf-8")
        L.append(f"WROTE {pxcm_txt} = {s['value']:.2f}")
        L.append(f"NEXT: give this value to the gait stage as the batch's px_per_cm_override ({s['value']:.2f}).")
    else:
        if pxcm_txt.exists():
            pxcm_txt.unlink()                                     # never leave an older value next to a refused run
        L.append(f"NOT WRITTEN: {s['reason']}")
    summary_txt.write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(readings_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader()
        w.writerows(table)
    for line in L:
        say(line)
    try:                                                           # provenance, best-effort
        from stride.run_manifest import write_run_manifest
        write_run_manifest(out_dir, "calibrate",
                           params=dict(board=spec.describe(), min_readings=min_readings, max_spread_pct=max_spread_pct,
                                       scan_seconds=scan_seconds, step_seconds=step_seconds, frames_per_clip=frames_per_clip,
                                       min_clips=min_clips, max_clips=max_clips, rail_crop=list(rail_crop) if rail_crop else None),
                           inputs=dict(clips=[str(c) for c in clips]),
                           outputs=[str(summary_txt), str(readings_csv)] + ([str(pxcm_txt)] if s["value"] is not None else []),
                           results=dict(px_per_cm=s["value"], median=s["median"], spread_pct=s["spread_pct"], n_usable=s["n_usable"],
                                        readings={stem: v for stem, v, *_ in s["usable"]}))
    except Exception:
        pass
    res.update(ok=s["value"] is not None, px_per_cm=s["value"], spread_pct=s["spread_pct"], n_clips_used=s["n_usable"],
               n_clips=len(clips), per_clip=table, median=s["median"], out_dir=out_dir,
               px_per_cm_txt=pxcm_txt if s["value"] is not None else None, summary_txt=summary_txt, readings_csv=readings_csv,
               readings=readings, reason="" if s["value"] is not None else f"{s['reason']} (details: {summary_txt})")
    return res


def calibrate_scale(videos, out_dir=None, board=None, n_clips: int = 6, **options) -> dict:
    """`calibrate_px_per_cm` that RAISES CalibrationError when no value is accepted (for callers that must stop).

    Same arguments (`n_clips` = `min_clips`). The returned dict also carries the older key names
    `n_usable` and `table`.
    """
    r = calibrate_px_per_cm(videos, out_dir, board, min_clips=n_clips, **options)
    if not r["ok"]:
        raise CalibrationError(r["reason"] if r["out_dir"] is None else f"no px/cm written: {r['reason']}")
    r["n_usable"], r["table"] = r["n_clips_used"], r["per_clip"]
    return r


# ------------------------------------------------------------------------------------------------ command line
def parse_board(squares: str, square_mm: float, marker_mm: float, dictionary: str) -> BoardSpec:
    return make_board({"squares": squares, "square_mm": square_mm, "marker_mm": marker_mm, "dictionary": dictionary})


def parse_rail_crop(text: Optional[str]):
    if text is None or str(text).strip().lower() in ("", "none", "off"):
        return None
    try:
        vals = tuple(int(v) for v in str(text).split(","))
    except ValueError:
        vals = ()
    if len(vals) != 4:
        raise CalibrationError(f"--rail-crop needs y0,y1,x0,x1 (got {text!r})")
    return vals


try:
    import click
except ImportError:                                               # the library works without the command line
    click = None

if click is not None:
    @click.command("calibrate")
    @click.argument("videos", nargs=-1, required=True, type=click.Path(path_type=Path))
    @click.option("--out", "out_dir", default=None, type=click.Path(file_okay=False, path_type=Path),
                  help="Output folder. Default: <folder>/calibration.")
    @click.option("--board-squares", default="14x9", show_default=True, help="ChArUco squares, X x Y.")
    @click.option("--square-mm", default=20.0, show_default=True, help="Square side on the print, mm.")
    @click.option("--marker-mm", default=15.0, show_default=True, help="Marker side on the print, mm.")
    @click.option("--dictionary", default="DICT_5X5_250", show_default=True, help="ArUco dictionary of the board.")
    @click.option("--clips", "min_clips", default=6, show_default=True, help="For a folder: measure at least this many clips.")
    @click.option("--max-clips", default=None, type=int, help="Measure at most this many clips (spread over the folders).")
    @click.option("--frames-per-clip", default=None, type=int, help="Frames examined per clip over the first 10 s. Default: one every 0.15 s (67).")
    @click.option("--min-readings", default=3, show_default=True, help="Usable clip readings needed to accept a value.")
    @click.option("--max-spread", default=1.0, show_default=True, help="Largest accepted spread of the readings, %.")
    @click.option("--rail-crop", default=None, help="Optional cross-check: y0,y1,x0,x1 of a straight 10-cm corridor in the frame.")
    @click.option("--workers", default=1, show_default=True, help="Clips measured in parallel.")
    def calibrate_cmd(videos, out_dir, board_squares, square_mm, marker_mm, dictionary, min_clips, max_clips, frames_per_clip,
                      min_readings, max_spread, rail_crop, workers):
        """px/cm from a ChArUco board in view of the camera.

        VIDEOS is a folder (clips are found recursively; the first clip of each sub-folder is measured) or one or
        more clip files. Writes px_per_cm.txt, a summary, a per-clip table and QC images, and prints the value.
        Use the value as the batch's `px_per_cm_override` in the gait stage.
        """
        try:
            crop = parse_rail_crop(rail_crop)
        except CalibrationError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
        r = calibrate_px_per_cm(videos[0] if len(videos) == 1 else list(videos), out_dir,
                                {"squares": board_squares, "square_mm": square_mm, "marker_mm": marker_mm, "dictionary": dictionary},
                                min_clips=min_clips, max_clips=max_clips, frames_per_clip=frames_per_clip, min_readings=min_readings,
                                max_spread_pct=max_spread, rail_crop=crop, workers=workers)
        if not r["ok"]:
            click.echo(f"ERROR: {r['reason']}", err=True)
            sys.exit(1)
        click.echo(f"px_per_cm = {r['px_per_cm']:.2f}   (median of {r['n_clips_used']} clips, spread {r['spread_pct']:.2f} %)   -> {r['px_per_cm_txt']}")

    if __name__ == "__main__":
        calibrate_cmd()
