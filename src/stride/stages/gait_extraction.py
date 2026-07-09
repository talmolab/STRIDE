# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "scipy",
#     "sleap-io",
#     "opencv-python",
#     "pyyaml",
# ]
# ///
"""
Gait extraction: keypoint preprocessing, stride detection, and metric computation.

Stages 4-5 of the T-maze gait analysis pipeline.

PROVENANCE:
  Extracted from:
    GoPro: /home/exx/vast/leo/2025-12-12-GoPro-Tmaz-gait-analysis/gait_output.ipynb, cell 1 (745 lines)
    Basler: /home/exx/vast/leo/2025-10-03-Tmaze-gait/gait_output.ipynb, cell 2 (703 lines)
  Both notebooks contain identical function code; only config (paths, column names) differs.
  Functions copied verbatim; changes documented below.

CHANGES FROM SOURCE:
  1. FIX: estimate_cm_per_px_from_yaml line 76 — `yml_path = YML_DIR / ...` changed to
     `yml_path = Path(yml_dir) / ...` (use function parameter instead of global)
  2. Globals PARAMS, YML_DIR replaced with function parameters
  3. Column names abstracted via `column_map` for GoPro vs Basler differences
  4. Added run_gait_extraction() entry point wrapping the parallel loop
  5. Added compute_keypoint_confidence() for generating the confidence CSV
"""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy import interpolate
from scipy.ndimage import median_filter, uniform_filter1d

warnings.filterwarnings("ignore")


# ============================================================================
# DEFAULT PARAMETERS (match both notebooks exactly)
# ============================================================================

DEFAULT_PARAMS = dict(
    MIN_WALK_VEL=5.0,        # cm/s tail base
    MIN_PAW_VEL=8.0,         # cm/s swing enter threshold (no hysteresis)
    MIN_PEAK_PAW_VEL=15.0,   # cm/s peak-in-swing
    MIN_FRAMES_SWING=3,      # frames
    SMOOTH_MED_WIN=3,        # median filter window
    SMOOTH_MOV_WIN=5,        # moving average window
    PHASE_SPLINE_K=3,        # cubic spline order
    PHASE_GRID=101,          # interpolation grid points
    DEFAULT_FPS=120.0,       # fallback FPS (GoPro=119.88, Basler=100)
    MAZE_WIDTH_CM=10.0,      # physical corridor width
)


# ============================================================================
# PURE FUNCTIONS — copied verbatim from gait_output.ipynb cell 1
# ============================================================================

def interpolate_missing(data, kind="linear"):
    x = np.arange(0, data.shape[0])
    def interpolate_1d(y):
        missing = np.isnan(y)
        if np.all(missing): return y
        vmask = ~missing
        f = interpolate.interp1d(x[vmask], y[vmask], kind=kind,
                                 bounds_error=False, fill_value="extrapolate")
        y[missing] = f(x[missing])
        return y
    return np.apply_along_axis(interpolate_1d, 0, data)


def median_smoothing(arr, window_size):
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    size_tuple = (window_size,) + (1,) * (arr.ndim - 1)
    return median_filter(arr, size=size_tuple, mode="mirror")


def moving_average_smoothing(arr, window_size):
    if window_size <= 1: return arr
    return uniform_filter1d(arr, size=window_size, axis=0, mode="nearest")


def connected_components1d(x, return_limits=False):
    L, n = scipy.ndimage.label(x.squeeze())
    ccs = scipy.ndimage.find_objects(L)
    starts = [cc[0].start for cc in ccs]
    ends   = [cc[0].stop  for cc in ccs]
    if return_limits:
        return np.stack([starts, ends], axis=1)
    else:
        return [np.arange(i0, i1, dtype=int) for i0, i1 in zip(starts, ends)]


def resolve_roi_yml(stem: str, yml_dir=None, video_path=None):
    """Locate a video's ROI yml, tolerating naming/layout variants.

    Tries (in order) yml_dir then co-located next to the video, for these names:
      {stem}.preds.v2.best1.yml  (STRIDE inference output)
      {stem}.rois.yml            (per-day-subdir cohorts, ROIs co-located with videos)
      {stem}.yml
    Returns the first existing Path, or None.
    """
    names = [f"{stem}.preds.v2.best1.yml", f"{stem}.rois.yml", f"{stem}.yml"]
    dirs = []
    if yml_dir is not None:
        dirs.append(Path(yml_dir))
    if video_path is not None:
        dirs.append(Path(video_path).parent)  # co-located fallback
    for d in dirs:
        for n in names:
            p = d / n
            if p.exists():
                return p
    return None


def estimate_cm_per_px_from_yaml(stem: str, yml_dir, maze_width_cm=10.0, video_path=None):
    """
    Calculate cm/px calibration from segment2 ROI polygon.

    FIX: Handle different T-maze orientations between batches.
    - Old batch: vertical T shape (segment is taller than wide)
    - New batch: horizontal T shape (segment is wider than tall)

    The maze corridor width (10cm) is always the SHORTER dimension,
    so we use min(horiz_px, vert_px) for calibration.
    """
    import yaml
    yml_path = resolve_roi_yml(stem, yml_dir, video_path)
    if yml_path is None:
        raise FileNotFoundError(f"ROI YAML not found for stem '{stem}' (yml_dir={yml_dir})")
    with open(yml_path, "r") as f:
        y = yaml.safe_load(f) or {}
    polys = {}
    for r in y.get("rois", []):
        name = str(r.get("name", "")).strip().lower()
        coords = np.asarray(r.get("coordinates", []), dtype=float)
        if name and coords.ndim == 2 and coords.shape[1] == 2 and len(coords) >= 4:
            polys[name] = coords
    seg = polys.get("segment2", None)
    if seg is None:
        raise ValueError("segment2 polygon not found in YAML.")
    q = np.asarray(seg[:4], dtype=float)

    # segment2 is a quad given as 4 CONSECUTIVE polygon vertices. Its two side lengths are the
    # means of the two pairs of OPPOSITE edges; the maze corridor width (10 cm) is the shorter side.
    #
    # BUGFIX: the previous implementation grouped vertices into "top-2 / bottom-2" by y-coordinate
    # and measured top/bottom/left/right spans. That silently assumes the rectangle is axis-aligned.
    # When the corridor runs at an ANGLE (e.g. the DJI T-maze), the y-sort pairs the wrong vertices
    # and returns a skew/diagonal span instead of the short edge — roughly DOUBLING maze_width_px and
    # halving all cm distances/speeds. It was orientation-dependent, so it hit some clips and not
    # others (e.g. 100% of one DJI cohort, ~40% of another). Using consecutive-edge lengths is
    # rotation-invariant and reads the true corridor width regardless of maze orientation.
    edges = [float(np.linalg.norm(q[i] - q[(i + 1) % 4])) for i in range(4)]
    dim_a = 0.5 * (edges[0] + edges[2])   # one pair of opposite edges
    dim_b = 0.5 * (edges[1] + edges[3])   # the other pair
    maze_width_px = min(dim_a, dim_b)     # corridor width = the shorter side

    if not np.isfinite(maze_width_px) or maze_width_px <= 0:
        raise ValueError(f"Invalid segment2 dimensions from YAML: edges={edges}")
    return float(maze_width_cm) / maze_width_px  # cm/px


def px_per_cm_from_yaml(stem: str, yml_dir, maze_width_cm=10.0, video_path=None) -> float:
    return 1.0 / estimate_cm_per_px_from_yaml(stem, yml_dir, maze_width_cm=maze_width_cm,
                                              video_path=video_path)


def read_fps(video_obj, video_path: str | Path, yml_dir=None, stem: str | None = None) -> float | None:
    for attr in ("fps", "frame_rate", "frame_rate_hz", "frameRate"):
        if hasattr(video_obj, attr):
            try:
                v = float(getattr(video_obj, attr))
                if np.isfinite(v) and v > 0: return v
            except Exception: pass
    for attr in ("metadata", "info"):
        if hasattr(video_obj, attr):
            meta = getattr(video_obj, attr)
            if isinstance(meta, dict):
                for k in ("fps", "frame_rate", "frameRate"):
                    if k in meta:
                        try:
                            v = float(meta[k])
                            if np.isfinite(v) and v > 0: return v
                        except Exception: pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        v = cap.get(cv2.CAP_PROP_FPS); cap.release()
        if np.isfinite(v) and v > 0: return float(v)
    except Exception: pass
    if yml_dir is not None and stem is not None:
        import yaml
        cand = Path(yml_dir) / f"{stem}.yml"
        if cand.exists():
            try:
                with open(cand, "r") as f:
                    y = yaml.safe_load(f) or {}
                for k in ("fps", "frame_rate", "frameRate"):
                    if k in y:
                        v = float(y[k])
                        if np.isfinite(v) and v > 0: return v
            except Exception: pass
    return None


def normalize_to_egocentric(
    x: np.ndarray,
    ctr_ind: int,
    fwd_ind: int,
    rel_to: np.ndarray | None = None,
    return_angles: bool = False,
):
    if rel_to is None: rel_to = x
    is_singleton = (x.ndim == 2)
    if x.ndim == 2: x = x[None, ...]
    if rel_to.ndim == 2: rel_to = rel_to[None, ...]
    ctr = rel_to[..., ctr_ind, :]
    fwd = rel_to[..., fwd_ind, :]
    axis = fwd - ctr
    ang = np.arctan2(axis[..., 1], axis[..., 0])
    ca, sa = np.cos(ang), np.sin(ang)
    x = x - ctr[:, None, :]
    rot = np.zeros((len(ca), 2, 2), dtype=x.dtype)
    rot[:, 0, 0] = ca; rot[:, 0, 1] = -sa
    rot[:, 1, 0] = sa; rot[:, 1, 1] =  ca
    x = np.einsum("tij,tkj->tki", rot, x)
    if is_singleton:
        x = x[0]; ang = ang[0]
    return (x, ang) if return_angles else x


def align_to_displacement(x: np.ndarray, pt_0: np.ndarray, pt_1: np.ndarray):
    disp = pt_1 - pt_0
    ang = np.arctan2(disp[1], disp[0])
    ca, sa = np.cos(ang), np.sin(ang)
    rot = np.array([[ca, -sa],[sa, ca]], dtype=x.dtype)
    x0 = x - pt_0.reshape(1,1,2)
    return np.einsum("ij,tkj->tki", rot, x0)


def pick_best_paw_indices(node_vels, tail_base_ind, left_candidates, right_candidates,
                          min_walking_vel=5.0, topk=10):
    """Choose best paw landmark per side (silent version)."""
    vel_tail = node_vels[:, tail_base_ind]
    walk_mask = np.isfinite(vel_tail) & (vel_tail >= min_walking_vel)
    def score(idx):
        v = node_vels[:, idx]
        v = v[np.isfinite(v) & walk_mask]
        if v.size == 0: return 0.0
        completeness = v.size / max(1, walk_mask.sum())
        if v.size >= topk:
            peakiness = float(np.median(np.partition(v, -topk)[-topk:]))
        else:
            peakiness = float(np.median(v))
        return completeness * peakiness
    def choose(cands):
        scored = [(i, score(i)) for i in cands]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[0][0]
    return choose(left_candidates), choose(right_candidates)


def get_strides_paper(
    node_vels: np.ndarray,
    tail_base_ind: int,
    hindL_ind: int,
    hindR_ind: int,
    min_walking_vel=5,
    min_paw_vel=4,
    min_peak_paw_vel=15,
    min_frames=3,
    return_swings: bool = False,
):
    v_track = node_vels[:, tail_base_ind]
    is_walking = np.isfinite(v_track) & (v_track >= min_walking_vel)
    def swings_for_paw(idx: int):
        v = node_vels[:, idx]
        mask = np.isfinite(v) & (v >= min_paw_vel) & is_walking
        comps = connected_components1d(mask)
        kept = []
        for c in comps:
            a, b = int(c[0]), int(c[-1])
            if (b - a + 1) < min_frames: continue
            vmax_paw   = float(np.nanmax(v[a:b+1]))
            vmax_track = float(np.nanmax(v_track[a:b+1]))
            if not np.isfinite(vmax_paw): continue
            if vmax_paw < min_peak_paw_vel: continue
            if vmax_paw < vmax_track: continue
            kept.append([a, b])
        return np.array(kept, dtype=int) if kept else np.empty((0,2), int)
    L_sw = swings_for_paw(hindL_ind)
    R_sw = swings_for_paw(hindR_ind)
    if len(L_sw) < 2:
        return (np.empty((0,2), int), L_sw, R_sw) if return_swings else np.empty((0,2), int)
    stride_starts = L_sw[:-1, 1] + 1
    stride_ends   = L_sw[1:,  1]
    R_ends = R_sw[:, 1] if len(R_sw) else np.array([], dtype=int)
    strides = []
    for s0, s1 in zip(stride_starts, stride_ends):
        if R_ends.size and ((R_ends > s0) & (R_ends < s1)).any():
            strides.append([s0, s1])
    strides = np.array(strides, dtype=int) if len(strides) else np.empty((0,2), int)
    return (strides, L_sw, R_sw) if return_swings else strides


def central_diff_deg_per_s(theta_rad, fps):
    th = np.unwrap(theta_rad)
    d = np.empty_like(th)
    d[1:-1] = (th[2:] - th[:-2]) * 0.5 * fps
    d[0]    = (th[1] - th[0]) * fps
    d[-1]   = (th[-1] - th[-2]) * fps
    return np.degrees(d)


def point_line_segment_distance(p, a, b):
    ap = p - a
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 0: return float(np.linalg.norm(ap)), 0.0
    t = float(np.clip(np.dot(ap, ab) / denom, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj)), t


def lateral_disp_norm_along_allocentric(trx_cm, kp_ind, stride_slice, body_length_cm, tail_base_ind):
    s0, s1 = stride_slice
    tb0 = trx_cm[s0, tail_base_ind, :]
    tb1 = trx_cm[s1-1, tail_base_ind, :]
    disp = tb1 - tb0
    if not np.all(np.isfinite(disp)) or np.linalg.norm(disp) < 1e-6 or not np.isfinite(body_length_cm) or body_length_cm <= 0:
        return np.nan
    u = disp / np.linalg.norm(disp)
    u_perp = np.array([-u[1], u[0]])
    ys = []
    for t in range(s0, s1):
        p = trx_cm[t, kp_ind, :]
        if np.any(~np.isfinite(p)) or np.any(~np.isfinite(tb0)):
            ys.append(np.nan); continue
        ys.append(np.dot((p - tb0), u_perp))
    ys = np.asarray(ys, float)
    if np.all(~np.isfinite(ys)): return np.nan
    return (np.nanmax(ys) - np.nanmin(ys)) / body_length_cm


def phase_of_max_lateral(trx_cm, kp_ind, stride_slice, fps, tail_base_ind, phase_grid=101):
    from scipy.interpolate import CubicSpline
    s0, s1 = stride_slice
    tb0 = trx_cm[s0, tail_base_ind, :]
    tb1 = trx_cm[s1-1, tail_base_ind, :]
    disp = tb1 - tb0
    if not np.all(np.isfinite(disp)) or np.linalg.norm(disp) < 1e-6: return np.nan
    u = disp / np.linalg.norm(disp)
    u_perp = np.array([-u[1], u[0]])
    ys = []
    for t in range(s0, s1):
        p = trx_cm[t, kp_ind, :]
        if np.any(~np.isfinite(p)) or np.any(~np.isfinite(tb0)):
            ys.append(np.nan); continue
        ys.append(np.dot((p - tb0), u_perp))
    ys = np.asarray(ys, float)
    n = len(ys)
    if n < 3 or np.all(~np.isfinite(ys)): return np.nan
    x = np.linspace(0, 100, n)
    try:
        cs = CubicSpline(x, ys)
        grid = np.linspace(0, 100, phase_grid)
        yhat = cs(grid)
        return float(grid[np.nanargmax(yhat)])
    except Exception:
        return float(x[np.nanargmax(ys)])


def build_trx_from_labels(labels, video) -> np.ndarray:
    T = len(video)
    J = len(labels.skeleton.node_names)
    trx = np.full((T, J, 2), np.nan, dtype=float)
    for f in range(T):
        lfs = labels.find(video=video, frame_idx=f)
        if len(lfs) and len(lfs[0]):
            pts = lfs[0][0].numpy()
            j = min(J, pts.shape[0])
            trx[f, :j, :] = pts[:j, :]
    return trx


# ============================================================================
# PER-VIDEO PROCESSING — adapted from process_one_video in notebook
# Changes: globals replaced with params dict, yml_dir, pose_slp_suffix, column_map
# ============================================================================

# Canonical stride CSV header (GoPro-style column names for backward compat)
STRIDE_HEADER = [
    "video_path", "stem", "day", "mouse_id", "sex", "genotype", "stride_id",
    "frame_start", "frame_end", "t_start_ms", "t_end_ms", "duration_ms",
    "fps", "px_per_cm", "body_length_cm", "hindL_ind", "hindR_ind",
    "angular_velocity_deg_s_mean", "stride_speed_cm_s_mean",
    "limb_duty_factor_mean", "temporal_symmetry",
    "step_length_cm", "step_width_cm", "stride_length_cm", "stride_length_euclid_cm",
    "nose_lat_disp_norm", "tailbase_lat_disp_norm", "tailtip_lat_disp_norm",
    "nose_lat_phase_pct", "tailbase_lat_phase_pct", "tailtip_lat_phase_pct",
    "valid_stride_TF", "has_R_end_inside_TF", "nans_in_stride_TF", "comment",
]

STEP_HEADER = [
    "video_path", "stem", "day", "mouse_id", "sex", "genotype",
    "step_id", "paw_side", "frame_a", "frame_b", "t_a_ms", "t_b_ms", "dur_ms",
    "peak_speed_cm_s", "peak_tailbase_speed_cm_s", "within_stride_id",
]


def _extract_meta_fields(meta_row: dict, column_map: dict) -> tuple[str, str, str]:
    """Extract canonical mouse_id, sex, genotype from meta_row using column_map."""
    # Mouse ID: may be single column or composite
    id_cols = column_map.get("mouse_id", ["mouse"])
    if isinstance(id_cols, str):
        id_cols = [id_cols]
    sep = column_map.get("mouse_id_sep", "_")
    mouse_id = sep.join(str(meta_row[c]) for c in id_cols)

    sex_col = column_map.get("sex", "Gender")
    sex = str(meta_row.get(sex_col, ""))

    genotype = str(meta_row.get("Genotype", ""))
    return mouse_id, sex, genotype


def process_one_video(meta_row: dict, params: dict, yml_dir: str | Path,
                      pose_slp_suffix: str, column_map: dict):
    """Process a single video: load SLEAP, detect strides, compute metrics.

    Returns (stride_rows, step_rows, per_video_row) or None if skipped.
    """
    import sleap_io as sio

    video_path = Path(meta_row["video_path"])
    if not video_path.exists():
        return None

    pose_path = video_path.with_suffix(pose_slp_suffix)
    if not pose_path.exists():
        return None

    labels = sio.load_file(str(pose_path))
    video  = labels.videos[0]
    stem   = Path(video.filename).stem

    # Pixel->cm and FPS. Pass video_path so ROIs co-located with the video (.rois.yml in a
    # per-day subdir) resolve even when a separate yml_dir isn't provided.
    # PX_PER_CM_OVERRIDE (optional): a known, externally-verified px/cm (e.g. from a ChArUco
    # calibration board) takes precedence over the per-video segment2 derivation. Use this when
    # you have a trusted per-cohort calibration constant — it's immune to any ROI/segment2 issue.
    override = params.get("PX_PER_CM_OVERRIDE") if isinstance(params, dict) else None
    if override:
        _px_per_cm = float(override)
    else:
        try:
            _px_per_cm = px_per_cm_from_yaml(stem, yml_dir, maze_width_cm=params["MAZE_WIDTH_CM"],
                                             video_path=video_path)
        except Exception:
            _px_per_cm = 1.0
    fps = read_fps(video, video_path, yml_dir=yml_dir, stem=stem) or params["DEFAULT_FPS"]

    # Node indices
    def idx(name): return labels.skeleton.index(name)
    snout_ind     = idx("snout")
    tail_base_ind = idx("tailbase")
    tail_tip_ind  = idx("tailtip")
    hindL1 = idx("hindpawL1"); hindL2 = idx("hindpawL2")
    hindR1 = idx("hindpawR1"); hindR2 = idx("hindpawR2")

    # Trajectories
    trx_px = build_trx_from_labels(labels, video)
    trx_cm = trx_px / _px_per_cm
    trx_cm = interpolate_missing(trx_cm, kind="nearest")
    trx_cm = median_smoothing(trx_cm, params["SMOOTH_MED_WIN"])
    trx_cm = moving_average_smoothing(trx_cm, params["SMOOTH_MOV_WIN"])

    node_vels = np.linalg.norm(np.diff(trx_cm, axis=0, prepend=np.nan), axis=-1) * fps

    # Body length
    body_lengths = np.linalg.norm(trx_cm[:, snout_ind] - trx_cm[:, tail_base_ind], axis=-1)
    body_length_mean = float(np.nanmean(body_lengths))

    # Auto-pick paws
    hindL_ind, hindR_ind = pick_best_paw_indices(
        node_vels=node_vels, tail_base_ind=tail_base_ind,
        left_candidates=[hindL1, hindL2], right_candidates=[hindR1, hindR2],
        min_walking_vel=params["MIN_WALK_VEL"], topk=10,
    )

    # Detect strides/swings
    strides, L_swings, R_swings = get_strides_paper(
        node_vels=node_vels,
        tail_base_ind=tail_base_ind,
        hindL_ind=hindL_ind,
        hindR_ind=hindR_ind,
        min_walking_vel=params["MIN_WALK_VEL"],
        min_paw_vel=params["MIN_PAW_VEL"],
        min_peak_paw_vel=params["MIN_PEAK_PAW_VEL"],
        min_frames=params["MIN_FRAMES_SWING"],
        return_swings=True,
    )

    # Egocentric transform
    trx_ego, ego_ang = normalize_to_egocentric(
        trx_cm, ctr_ind=tail_base_ind, fwd_ind=snout_ind, return_angles=True)
    ang_vel_deg_s = central_diff_deg_per_s(ego_ang, fps)

    # Time arrays
    T = node_vels.shape[0]
    t_ms = np.arange(T) / fps * 1000.0
    v_tail = node_vels[:, tail_base_ind]

    # Extract canonical meta fields
    mouse_id, sex, genotype = _extract_meta_fields(meta_row, column_map)

    # ---- Per-step table ----
    step_rows = []
    step_id = 0
    stride_idx_by_end = {}
    for si, (s0, s1) in enumerate(strides):
        if L_swings is not None:
            for a, b in L_swings:
                if s0 < b < s1:
                    stride_idx_by_end[("L", b)] = si
        if R_swings is not None:
            for a, b in R_swings:
                if s0 < b < s1:
                    stride_idx_by_end[("R", b)] = si

    def add_steps(swings, side):
        nonlocal step_id
        if swings is None: return
        paw_ind = hindL_ind if side == "L" else hindR_ind
        for a, b in swings:
            dur_ms_step = (b - a + 1) / fps * 1000.0
            peak_paw = float(np.nanmax(node_vels[a:b+1, paw_ind]))
            peak_tb  = float(np.nanmax(v_tail[a:b+1]))
            step_rows.append([
                str(video_path), stem, meta_row.get("day", ""),
                mouse_id, sex, genotype,
                step_id, side, int(a), int(b), float(t_ms[a]), float(t_ms[b]),
                float(dur_ms_step), peak_paw, peak_tb,
                int(stride_idx_by_end.get((side, b), -1)),
            ])
            step_id += 1

    add_steps(L_swings, "L")
    add_steps(R_swings, "R")

    # ---- Per-stride table ----
    stride_rows = []
    phase_grid = params.get("PHASE_GRID", 101)

    for si, (s0, s1) in enumerate(strides):
        if s1 <= s0 + 1: continue
        w = slice(s0, s1)
        dur_ms = (s1 - s0) / fps * 1000.0
        ang_mean = float(np.nanmean(ang_vel_deg_s[w]))
        speed_mean = float(np.nanmean(node_vels[w, tail_base_ind]))

        # Duty factors via swing masks
        def duty_from_swings(swings):
            mask = np.zeros(s1 - s0, dtype=bool)
            if swings is not None:
                for a, b in swings:
                    aa = max(a, s0); bb = min(b, s1-1)
                    if bb >= aa: mask[aa - s0: bb - s0 + 1] |= True
            stance = ~mask
            return float(np.mean(stance)) if (s1 - s0) > 0 else np.nan

        l_duty = duty_from_swings(L_swings)
        r_duty = duty_from_swings(R_swings)
        duty_mean = np.nanmean([l_duty, r_duty])
        temp_sym = (l_duty - r_duty) / (l_duty + r_duty) if np.isfinite(l_duty) and np.isfinite(r_duty) and (l_duty + r_duty) != 0 else np.nan

        # R swing that ends inside this stride
        R_end_inside = None
        if R_swings is not None:
            for a, b in R_swings:
                if s0 < b < s1:
                    R_end_inside = (a, b); break

        # Step length / width (egocentric)
        step_len = np.nan
        step_width = np.nan
        L_strike_prev = None
        if L_swings is not None and len(L_swings) > 0:
            ends = L_swings[:, 1]
            idx_prev = np.where(ends <= s0 - 1)[0]
            if idx_prev.size: L_strike_prev = int(ends[idx_prev[-1]])
        if (L_strike_prev is not None) and (R_end_inside is not None):
            R_strike = int(R_end_inside[1])
            pL = trx_ego[L_strike_prev, hindL_ind, :2]
            pR = trx_ego[R_strike, hindR_ind, :2]
            if np.all(np.isfinite([*pL, *pR])):
                step_len = float(pR[0] - pL[0])
            L_toeoff = None; L_strike = None
            if L_swings is not None and len(L_swings) > 0:
                cands = np.where((L_swings[:, 1] >= L_strike_prev) & (L_swings[:, 1] < s1))[0]
                if cands.size:
                    k = cands[0]
                    L_toeoff = int(L_swings[k][0]); L_strike = int(L_swings[k][1])
            if L_toeoff is not None and L_strike is not None:
                p  = trx_ego[R_strike, hindR_ind, :2]
                a  = trx_ego[L_toeoff, hindL_ind, :2]
                b  = trx_ego[L_strike, hindL_ind, :2]
                if np.all(np.isfinite([*p, *a, *b])):
                    step_width, _ = point_line_segment_distance(p, a, b)

        # Stride length (egocentric dX and Euclid)
        stride_len_long = np.nan
        stride_len_euc  = np.nan
        if L_swings is not None and len(L_swings) > 0:
            cands2 = np.where((L_swings[:, 1] > s0) & (L_swings[:, 1] <= s1))[0]
            if cands2.size:
                k2 = cands2[-1]
                L_toeoff2 = int(L_swings[k2][0])
                L_strike2 = int(L_swings[k2][1])
                p0 = trx_ego[L_toeoff2, hindL_ind, :2]
                p1 = trx_ego[L_strike2, hindL_ind, :2]
                if np.all(np.isfinite([*p0, *p1])):
                    stride_len_long = float(p1[0] - p0[0])
                    stride_len_euc  = float(np.linalg.norm(p1 - p0))

        # Lateral displacement norms & phases
        nose_lat  = lateral_disp_norm_along_allocentric(trx_cm, idx("snout"), (s0, s1), body_length_mean, tail_base_ind)
        base_lat  = lateral_disp_norm_along_allocentric(trx_cm, tail_base_ind, (s0, s1), body_length_mean, tail_base_ind)
        tip_lat   = lateral_disp_norm_along_allocentric(trx_cm, tail_tip_ind, (s0, s1), body_length_mean, tail_base_ind)

        nose_phase = phase_of_max_lateral(trx_cm, idx("snout"), (s0, s1), fps, tail_base_ind, phase_grid=phase_grid)
        base_phase = phase_of_max_lateral(trx_cm, tail_base_ind, (s0, s1), fps, tail_base_ind, phase_grid=phase_grid)
        tip_phase  = phase_of_max_lateral(trx_cm, tail_tip_ind, (s0, s1), fps, tail_base_ind, phase_grid=phase_grid)

        # QC flags
        has_R_end_inside = R_end_inside is not None
        nans_stride = bool(np.any(~np.isfinite(trx_cm[s0:s1])))
        comment = "" if has_R_end_inside else "No right swing end inside L-delimited window"
        valid = bool(has_R_end_inside)

        stride_rows.append([
            str(video_path), stem, meta_row.get("day", ""),
            mouse_id, sex, genotype, si,
            int(s0), int(s1), float(t_ms[s0]), float(t_ms[s1-1]), float(dur_ms),
            float(fps), float(1.0 / _px_per_cm if _px_per_cm != 0 else np.nan),
            float(body_length_mean), int(hindL_ind), int(hindR_ind),
            float(ang_mean), float(speed_mean),
            float(duty_mean), float(temp_sym) if np.isfinite(temp_sym) else np.nan,
            step_len, step_width, stride_len_long, stride_len_euc,
            nose_lat, base_lat, tip_lat,
            nose_phase, base_phase, tip_phase,
            valid, has_R_end_inside, nans_stride, comment,
        ])

    # ---- Per-video aggregates ----
    stride_df = pd.DataFrame(stride_rows, columns=STRIDE_HEADER) if stride_rows else pd.DataFrame()

    agg = {
        "video_path": str(video_path), "stem": stem,
        "day": meta_row.get("day", ""),
        "mouse_id": mouse_id, "sex": sex, "genotype": genotype,
        "choice_LR": meta_row.get("choice_LR", np.nan),
        "latency_choice_ms": meta_row.get("latency_choice_ms", np.nan),
        "stem_latency_ms": meta_row.get("stem_latency_ms", np.nan),
        "junction_explore_ms": meta_row.get("junction_explore_ms", np.nan),
        "reward": meta_row.get("reward", np.nan),
        "correct_TF": meta_row.get("correct_TF", np.nan),
        "correct_bin": meta_row.get("correct_bin", np.nan),
        "probes_L": meta_row.get("probes_L", np.nan),
        "probes_R": meta_row.get("probes_R", np.nan),
        "hindL_ind": int(hindL_ind), "hindR_ind": int(hindR_ind),
        "n_strides": int(len(strides)),
        "n_valid_strides": int(stride_df["valid_stride_TF"].sum()) if not stride_df.empty else 0,
        "frac_valid": float(stride_df["valid_stride_TF"].mean()) if not stride_df.empty else np.nan,
    }

    def add_stats(col):
        if stride_df.empty:
            agg[f"{col}_mean"] = np.nan
            agg[f"{col}_sd"] = np.nan
            agg[f"{col}_n"] = 0
            return
        s = stride_df.loc[stride_df["valid_stride_TF"], col]
        agg[f"{col}_mean"] = float(s.mean()) if s.size else np.nan
        agg[f"{col}_sd"]   = float(s.std(ddof=1)) if s.size > 1 else np.nan
        agg[f"{col}_n"]    = int(s.size)

    for col in ["stride_speed_cm_s_mean", "angular_velocity_deg_s_mean",
                "limb_duty_factor_mean", "temporal_symmetry",
                "step_length_cm", "step_width_cm", "stride_length_cm",
                "nose_lat_disp_norm", "tailbase_lat_disp_norm", "tailtip_lat_disp_norm",
                "nose_lat_phase_pct", "tailbase_lat_phase_pct", "tailtip_lat_phase_pct"]:
        add_stats(col)

    return stride_rows, step_rows, agg


# ============================================================================
# KEYPOINT CONFIDENCE COMPUTATION
# ============================================================================

KEYPOINT_NAMES = [
    "snout", "mouth",
    "forepawR2", "forepawR1", "forepawL1", "forepawL2",
    "hindpawR2", "hindpawR1", "hindpawL2", "hindpawL1",
    "tailbase", "tail1", "tail2", "tail3", "tailtip",
]


def compute_keypoint_confidence(video_dir: Path, pose_slp_suffix: str,
                                output_path: Path) -> Path:
    """Compute per-video mean keypoint confidence from .slp files.

    Iterates over all videos, loads the paired .slp file, extracts confidence
    scores, and writes a CSV with one row per video.
    """
    import sleap_io as sio

    video_dir = Path(video_dir)
    rows = []

    # recursive + case-insensitive: supports flat dirs AND nested per-day subdirs
    videos = sorted({v for ext in ("mp4", "MP4") for v in video_dir.rglob(f"*.{ext}")})
    for vp in videos:
        pose_path = vp.with_suffix(pose_slp_suffix)
        if not pose_path.exists():
            continue
        try:
            labels = sio.load_file(str(pose_path))
            video = labels.videos[0]
            T = len(video)
            node_names = labels.skeleton.node_names

            # Collect confidences from all labeled frames
            conf_sum = {n: 0.0 for n in node_names}
            conf_count = {n: 0 for n in node_names}

            for lf in labels.labeled_frames:
                if lf.video != video:
                    continue
                for inst in lf.instances:
                    for node, pt in zip(labels.skeleton.nodes, inst.points):
                        # sleap_io >=0.7 stores points as a structured array (np.void with a
                        # 'score' field, no .score attribute); older versions use objects with
                        # pt.score. Support both so confidence isn't silently read as 0.
                        if hasattr(pt, "score"):
                            score = pt.score
                        else:
                            try:
                                score = float(pt["score"])
                            except (KeyError, ValueError, TypeError, IndexError):
                                continue
                        if np.isfinite(score):
                            conf_sum[node.name] = conf_sum.get(node.name, 0.0) + score
                            conf_count[node.name] = conf_count.get(node.name, 0) + 1

            row = {"stem": vp.stem, "video_path": str(vp), "frames_total": T}
            row["frames_kept"] = max(conf_count.values()) if conf_count else 0
            for kp in KEYPOINT_NAMES:
                n = conf_count.get(kp, 0)
                row[f"avg_{kp}"] = conf_sum.get(kp, 0.0) / n if n > 0 else 0.0
            rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"    Confidence CSV: {len(df)} videos → {output_path.name}")
    return output_path


# ============================================================================
# ENTRY POINT: run_gait_extraction
# ============================================================================

def _process_wrapper(args):
    """Wrapper for ProcessPoolExecutor (must be top-level for pickling)."""
    meta_row, params, yml_dir, pose_slp_suffix, column_map = args
    try:
        res = process_one_video(meta_row, params, yml_dir, pose_slp_suffix, column_map)
        return {"status": "ok" if res else "skip", "result": res,
                "video": meta_row.get("video_path", "?")}
    except Exception as e:
        return {"status": "error", "error": str(e),
                "video": meta_row.get("video_path", "?")}


def run_gait_extraction(
    meta_csv: str | Path,
    yml_dir: str | Path,
    output_dir: str | Path,
    params: dict | None = None,
    pose_slp_suffix: str = ".slp",
    column_map: dict | None = None,
    n_workers: int = 16,
) -> dict:
    """Run Stages 4-5: keypoint preprocessing + stride detection.

    Args:
        meta_csv: Path to metrics_meta CSV with video_path, day, mouse ID, sex, genotype columns
        yml_dir: Path to directory containing ROI YAML files
        output_dir: Where to write output CSVs
        params: Stride detection parameters (defaults to DEFAULT_PARAMS)
        pose_slp_suffix: File suffix for SLEAP pose files (e.g., ".slp" or ".pose.v2.slp")
        column_map: Maps batch-specific columns to canonical names
        n_workers: Number of parallel workers

    Returns:
        Summary dict with counts
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if column_map is None:
        column_map = {"mouse_id": ["mouse"], "sex": "Gender"}

    meta_csv = Path(meta_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save params
    with open(output_dir / "run_params.json", "w") as f:
        json.dump(params, f, indent=2)

    meta = pd.read_csv(meta_csv)
    meta_rows = [row.to_dict() for _, row in meta.iterrows()]

    print(f"  Processing {len(meta_rows)} videos with {n_workers} workers...")

    all_stride_rows = []
    all_step_rows = []
    all_per_video_rows = []
    errors = []
    n_ok = n_skip = 0

    # Build args for each video
    args_list = [
        (row, params, str(yml_dir), pose_slp_suffix, column_map)
        for row in meta_rows
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_process_wrapper, a): i for i, a in enumerate(args_list)}
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "ok":
                stride_rows, step_rows, per_video_row = res["result"]
                all_stride_rows.extend(stride_rows)
                all_step_rows.extend(step_rows)
                all_per_video_rows.append(per_video_row)
                n_ok += 1
            elif res["status"] == "skip":
                n_skip += 1
            else:
                errors.append(f"{Path(res['video']).name}: {res['error']}")

    # Write CSVs
    out_stride = output_dir / "gait_per_stride.csv"
    out_step = output_dir / "gait_per_step.csv"
    out_video = output_dir / "gait_per_video.csv"

    if all_stride_rows:
        pd.DataFrame(all_stride_rows, columns=STRIDE_HEADER).to_csv(out_stride, index=False)
    if all_step_rows:
        pd.DataFrame(all_step_rows, columns=STEP_HEADER).to_csv(out_step, index=False)
    if all_per_video_rows:
        pd.DataFrame(all_per_video_rows).to_csv(out_video, index=False)

    print(f"  Done: {n_ok} ok, {n_skip} skipped, {len(errors)} errors")
    if errors and len(errors) <= 10:
        for e in errors:
            print(f"    ERROR: {e}")
    elif errors:
        for e in errors[:5]:
            print(f"    ERROR: {e}")
        print(f"    ... and {len(errors) - 5} more errors")

    print(f"  Output: {out_stride.name} ({len(all_stride_rows):,} strides)")

    return {
        "ok": n_ok, "skipped": n_skip, "errors": len(errors),
        "total_strides": len(all_stride_rows),
        "stride_csv": str(out_stride),
        "step_csv": str(out_step),
        "video_csv": str(out_video),
    }
