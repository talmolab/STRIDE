"""
T-maze arm entry decision analysis with depth-based commit detection.

Adapted from: 2025-12-10-GoPro-Tmaze-analysis/pilot_decision.eda.02.ipynb

This module analyzes pose and ROI data to determine which arm the mouse
entered first, with depth-based validation requiring:
1. Body coverage >= threshold in arm
2. Snout depth >= MIN_DEPTH_PX from junction
3. At least one hindpaw in the arm

Output files:
- events.csv: Detailed timing events per trial
- decisions.csv: Per-video decision summary
- metrics.csv: Simplified metrics for analysis
"""

from pathlib import Path
from typing import Optional
from functools import partial
import re
import numpy as np
import pandas as pd
import cv2
import yaml
from shapely.geometry import Polygon, box, Point
from shapely.ops import unary_union

from stride.config import PipelineConfig
from stride.utils.video import read_fps
from stride.utils.parallel import run_parallel
from stride.utils.pose_io import resolve_pose_path

# Default thresholds
MIN_DEPTH_PX = 50  # Minimum depth from junction to qualify as "deep entry"
FPS_DEFAULT = 120.0
MIN_VALID_SEC = 0.5
SEG4_LEAVE_THR = 30.0
SEG4_LEAVE_K = 5
STRICT_THR = 95.0
SOFT_THR = 60.0
MIN_RUN = 3
GAP_ALLOW = 1
MIN_BODY_AREA = 1500.0


def run_decision_analysis(
    video_dir: Path,
    yml_dir: Path,
    meta_csv: Path,
    output_dir: Path,
    n_workers: int = 4,
    min_depth_px: int = MIN_DEPTH_PX,
    config: Optional[PipelineConfig] = None,
    pose_slp_suffix: str = ".slp",
    *,
    trials: Optional[list] = None,
    pose_dir: Optional[Path] = None,
) -> dict:
    """
    Run T-maze decision analysis on all videos.

    Outputs three CSV files:
    - events.csv: Detailed timing events
    - decisions.csv: Per-video decision summary
    - metrics.csv: Simplified metrics

    Args:
        video_dir: Directory containing videos with co-located .slp pose files
        yml_dir: Directory containing ROI .yml files
        meta_csv: CSV with day,mouse,reward columns
        output_dir: Output directory for CSVs
        n_workers: Number of parallel workers (CPU-bound, can use many workers)
        min_depth_px: Minimum depth threshold for commit (default 50)
        config: Optional pipeline config for thresholds
        pose_slp_suffix: Suffix of the pose file next to each video (default ".slp";
            e.g. ".pose.v2.slp")
        trials: Optional explicit trial list (used by `stride tmaze`). Each item is a dict with
            ``video_path`` and optionally ``day``, ``mouse``, ``trial``, ``reward``. When given,
            the videos are exactly these (any file name; ``video_dir`` is only reported and
            ``meta_csv`` is not read), and a trial without an L/R reward gets EMPTY correctness
            columns instead of "not correct". Without it nothing changes: videos are named
            ``Day<N>_<mouse>_Trial<T>`` and the reward comes from ``meta_csv``.
        pose_dir: Optional folder searched for ``<video stem><pose suffix>`` when no pose file
            sits next to the video.

    Returns:
        Summary dict with counts
    """
    if config is None:
        config = PipelineConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if trials is not None:
        return _run_decision_analysis_trials(
            trials, video_dir, yml_dir, output_dir, n_workers, min_depth_px, config,
            pose_slp_suffix, pose_dir)

    # Load metadata
    meta_df = pd.read_csv(meta_csv)
    meta_df["day"] = meta_df["day"].astype(str).str.strip().str.upper()
    meta_df["mouse"] = meta_df["mouse"].astype(str).str.strip()  # keep string: numeric AND GENO_cage_mouse IDs
    meta_df["reward"] = meta_df["reward"].astype(str).str.strip().str.upper().str[:1]
    meta_map = {(row.day, row.mouse): row.reward for _, row in meta_df.iterrows()}

    # Find videos — supports a flat dir OR nested per-day subdirs (DAY1/, DAY2/, ...)
    videos = sorted({v for ext in ("mp4", "MP4") for v in video_dir.rglob(f"*.{ext}")})

    # Create worker function with bound arguments. video_path is left as the single
    # unbound positional so run_parallel can call worker_fn(video) directly.
    worker_fn = partial(
        process_single_video,
        yml_dir=yml_dir,
        meta_map=meta_map,
        min_depth_px=min_depth_px,
        config=config,
        pose_slp_suffix=pose_slp_suffix,
        pose_dir=pose_dir,
    )

    # Metadata matching report (warnings only: scoring is unchanged)
    _report_metadata_matching(videos, meta_df, meta_map, meta_csv)

    # Process videos in parallel (CPU-bound analysis)
    if n_workers > 1:
        print(f"Running decision analysis with {n_workers} parallel workers...")
        # Pass the picklable partial directly. A lambda wrapper here cannot be pickled
        # by ProcessPoolExecutor, which silently broke every n_workers>1 run.
        results = run_parallel(
            items=videos,
            worker_fn=worker_fn,
            n_workers=n_workers,
            desc="Decision Analysis",
        )
    else:
        # Sequential processing with progress output
        results = []
        for i, video_path in enumerate(videos, 1):
            result = worker_fn(video_path=video_path)
            results.append(result)
            status_str = result.get("status", "ok")
            print(f"[{i}/{len(videos)}] {status_str:10} {video_path.name}")

    # Collect events and decisions from results
    events_rows = []
    decisions_rows = []
    for result in results:
        if result["status"] == "ok":
            events_rows.append(result["events"])
            decisions_rows.append(result["decisions"])

    # Save events.csv
    if events_rows:
        events_df = pd.DataFrame(events_rows)
        events_cols = [
            "day", "mouse", "trial", "stem", "video_path",
            "enter_seg1_frame", "enter_seg4_frame", "enter_junction_frame",
            "commit_side", "commit_frame", "ok",
            "enter_seg1_ms", "enter_seg4_ms", "enter_junction_ms", "commit_ms",
            "junction_explore_frames_precommit", "junction_explore_ms_precommit",
            "probes_L", "probes_R", "probe_frames_L", "probe_frames_R",
            "n_reversals", "reversal_frames", "reversal_ms",
            "coverage_thr_rule"
        ]
        events_df = events_df.reindex(columns=events_cols)
        events_df = events_df.sort_values(["day", "mouse", "trial"]).reset_index(drop=True)
        events_df.to_csv(output_dir / "events.csv", index=False)

    # Save decisions.csv
    if decisions_rows:
        decisions_df = pd.DataFrame(decisions_rows)
        decisions_cols = [
            "video_path", "stem", "day", "mouse", "trial", "reward", "fps",
            "ok", "reason", "gate_frame", "gate_reason",
            "entry_side", "entry_frame", "entry_ms", "entry_threshold",
            "correct_TF", "correct_bin"
        ]
        decisions_df = decisions_df.reindex(columns=decisions_cols)
        decisions_df = decisions_df.sort_values(["day", "mouse", "trial"]).reset_index(drop=True)
        decisions_df.to_csv(output_dir / "decisions.csv", index=False)

    # Save metrics.csv
    if events_rows and decisions_rows:
        metrics_df = build_metrics(events_df, decisions_df)
        metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    summary = {
        "total": len(videos),
        "ok": len(events_rows),
        "skipped": len(videos) - len(events_rows),
    }

    # End-of-stage summary: why videos were skipped (also when several workers were used)
    skip_reasons, error_examples = _summarize_skips(results)
    _print_stage_summary(video_dir, summary, skip_reasons, error_examples, pose_slp_suffix)
    n_no_entry = sum(1 for d in decisions_rows if not d.get("ok", True))
    if decisions_rows and n_no_entry > len(decisions_rows) / 2:
        print(f"  WARNING: {n_no_entry} of {len(decisions_rows)} analysed videos have no arm entry (ok = False). "
              f"If the mice did enter an arm, check the ROI files and the camera resolution: frames in which "
              f"the body outline covers less than {MIN_BODY_AREA:g} px^2 are ignored, so a low-resolution "
              f"recording can lose every frame.")
    if n_no_entry:
        summary["no_arm_entry"] = n_no_entry
    if skip_reasons:
        summary["skip_reasons"] = skip_reasons
    if error_examples:
        summary["errors"] = error_examples

    # Provenance sidecar (best-effort; never breaks a completed run)
    try:
        from stride.run_manifest import write_run_manifest
        write_run_manifest(
            output_dir, "decisions",
            params={"n_workers": n_workers, "min_depth_px": min_depth_px,
                    "pose_slp_suffix": pose_slp_suffix},
            inputs={"video_dir": video_dir, "yml_dir": yml_dir, "meta_csv": meta_csv,
                    "n_videos": len(videos)},
            outputs=[output_dir / "decisions.csv", output_dir / "events.csv",
                     output_dir / "metrics.csv"],
            results=summary,
        )
    except Exception:
        pass

    return summary


def _run_decision_analysis_trials(trials: list, video_dir, yml_dir, output_dir: Path, n_workers: int,
                                  min_depth_px: int, config: PipelineConfig, pose_slp_suffix: str,
                                  pose_dir: Optional[Path]) -> dict:
    """Explicit-trial-list mode: same analysis and same CSV columns as the default mode."""
    worker_fn = partial(process_trial, yml_dir=yml_dir, min_depth_px=min_depth_px, config=config,
                        pose_slp_suffix=pose_slp_suffix, pose_dir=pose_dir)
    if n_workers > 1 and len(trials) > 1:
        print(f"Running decision analysis with {n_workers} parallel workers...")
        results = run_parallel(items=trials, worker_fn=worker_fn, n_workers=n_workers,
                               desc="Decision Analysis")
    else:
        results = []
        for i, t in enumerate(trials, 1):
            result = worker_fn(t)
            results.append(result)
            print(f"[{i}/{len(trials)}] {result.get('status', 'ok'):10} {Path(t['video_path']).name}")

    events_rows = [r["events"] for r in results if r["status"] == "ok"]
    decisions_rows = [r["decisions"] for r in results if r["status"] == "ok"]

    def _frame(rows, cols):
        df = pd.DataFrame(rows).reindex(columns=cols)
        df["day"] = df["day"].fillna("").astype(str)
        df["mouse"] = df["mouse"].fillna("").astype(str)
        df["trial"] = pd.to_numeric(df["trial"], errors="coerce").astype("Int64")
        return df.sort_values(["day", "mouse", "trial", "stem"], kind="mergesort",
                              na_position="last").reset_index(drop=True)

    events_df = decisions_df = None
    if events_rows:
        events_df = _frame(events_rows, EVENTS_COLUMNS)
        events_df.to_csv(output_dir / "events.csv", index=False)
    if decisions_rows:
        decisions_df = _frame(decisions_rows, DECISIONS_COLUMNS)
        if decisions_df["correct_bin"].isna().any():
            decisions_df["correct_bin"] = decisions_df["correct_bin"].astype("Int64")
        decisions_df.to_csv(output_dir / "decisions.csv", index=False)
    if events_df is not None and decisions_df is not None:
        build_metrics(events_df, decisions_df).to_csv(output_dir / "metrics.csv", index=False)

    summary = {"total": len(trials), "ok": len(events_rows), "skipped": len(trials) - len(events_rows)}
    skip_reasons, error_examples = _summarize_skips(results)
    _print_stage_summary(video_dir, summary, skip_reasons, error_examples, pose_slp_suffix)
    n_no_entry = sum(1 for d in decisions_rows if not d.get("ok", True))
    if n_no_entry:
        summary["no_arm_entry"] = n_no_entry
    n_unknown = sum(1 for d in decisions_rows if d.get("correct_TF", "") == "")
    summary["reward_unknown"] = n_unknown
    if skip_reasons:
        summary["skip_reasons"] = skip_reasons
        summary["skipped_videos"] = {str(r.get("video")): str(r.get("status")) for r in results
                                     if r.get("status") != "ok"}
    if error_examples:
        summary["errors"] = error_examples
    try:
        from stride.run_manifest import write_run_manifest
        write_run_manifest(
            output_dir, "decisions",
            params={"n_workers": n_workers, "min_depth_px": min_depth_px,
                    "pose_slp_suffix": pose_slp_suffix, "mode": "explicit trial list"},
            inputs={"video_dir": video_dir, "yml_dir": yml_dir, "pose_dir": pose_dir,
                    "n_videos": len(trials)},
            outputs=[output_dir / "decisions.csv", output_dir / "events.csv",
                     output_dir / "metrics.csv"],
            results=summary,
        )
    except Exception:
        pass
    return summary


EVENTS_COLUMNS = [
    "day", "mouse", "trial", "stem", "video_path",
    "enter_seg1_frame", "enter_seg4_frame", "enter_junction_frame",
    "commit_side", "commit_frame", "ok",
    "enter_seg1_ms", "enter_seg4_ms", "enter_junction_ms", "commit_ms",
    "junction_explore_frames_precommit", "junction_explore_ms_precommit",
    "probes_L", "probes_R", "probe_frames_L", "probe_frames_R",
    "n_reversals", "reversal_frames", "reversal_ms",
    "coverage_thr_rule"
]
DECISIONS_COLUMNS = [
    "video_path", "stem", "day", "mouse", "trial", "reward", "fps",
    "ok", "reason", "gate_frame", "gate_reason",
    "entry_side", "entry_frame", "entry_ms", "entry_threshold",
    "correct_TF", "correct_bin"
]

_SKIP_LABELS = {
    "skip_parse": "file name is not Day<N>_<mouse>_Trial<T>.mp4",
    "skip_pose": "no pose file next to the video",
    "skip_yml": "no ROI file found",
}


def _summarize_skips(results: list) -> tuple[dict, list]:
    """Count the videos that produced no row, by reason; keep a few error messages."""
    reasons: dict = {}
    examples: list = []
    for r in results:
        status = str(r.get("status", "error"))
        if status == "ok":
            continue
        if status.startswith("error"):
            key = "error while analysing"
            msg = status[len("error:"):].strip() if status.startswith("error:") else str(r.get("error", ""))
            if len(examples) < 20:
                examples.append(f"{r.get('video', r.get('item', '?'))}: {msg}")
        else:
            key = _SKIP_LABELS.get(status, status)
        reasons[key] = reasons.get(key, 0) + 1
    return reasons, examples


def _print_stage_summary(video_dir, summary: dict, skip_reasons: dict, error_examples: list,
                         pose_slp_suffix: str) -> None:
    print(f"Decision analysis summary: {summary['total']} videos found under {video_dir}, "
          f"{summary['ok']} analysed, {summary['skipped']} skipped")
    for reason, n in sorted(skip_reasons.items(), key=lambda kv: -kv[1]):
        extra = f" (looked for <video name>{pose_slp_suffix})" if reason.startswith("no pose file") else ""
        print(f"  skipped {n}: {reason}{extra}")
    for e in error_examples[:5]:
        print(f"    ERROR {e}")
    if len(error_examples) > 5:
        print(f"    ... and {len(error_examples) - 5} more errors")
    if summary["total"] and summary["ok"] == 0:
        print("  WARNING: no video was analysed, so no decisions.csv / events.csv / metrics.csv was written")


def _report_metadata_matching(videos: list, meta_df, meta_map: dict, meta_csv) -> None:
    """Warn about videos without a reward row and metadata rows without a video.

    Reporting only: how a trial is scored is not changed here (a video whose (day, mouse) has
    no reward row is still written with an empty reward, as before).
    """
    try:
        video_keys = {}
        for v in videos:
            key = parse_filename(v)
            if key is not None:
                video_keys.setdefault((key["day"], key["mouse"]), 0)
                video_keys[(key["day"], key["mouse"])] += 1
        if not video_keys:
            return
        no_reward = sorted(k for k in video_keys if meta_map.get(k, "") not in ("L", "R"))
        unused = sorted(k for k in meta_map if k not in video_keys)
        if no_reward:
            n_vid = sum(video_keys[k] for k in no_reward)
            shown = ", ".join(f"{d}/{m}" for d, m in no_reward[:8]) + (" ..." if len(no_reward) > 8 else "")
            print(f"WARNING: {n_vid} video(s) of {len(no_reward)} (day, mouse) pair(s) have no L/R reward row in "
                  f"{Path(meta_csv).name}: {shown}. Their reward is left empty and the trials are written "
                  f"as not correct (correct_bin = 0) - add the rows or exclude these trials downstream.")
        if unused:
            shown = ", ".join(f"{d}/{m}" for d, m in unused[:8]) + (" ..." if len(unused) > 8 else "")
            print(f"WARNING: {len(unused)} (day, mouse) row(s) in {Path(meta_csv).name} match no video: {shown}")
        odd_days = sorted({str(d) for d in meta_df["day"].unique() if not re.fullmatch(r"DAY\d+", str(d))})
        if odd_days:
            print(f"WARNING: metadata day value(s) {odd_days[:6]} are not written like 'Day1' / 'DAY1'; "
                  f"days are matched as text against the video names (Day<N>_...), so these rows match nothing.")
    except Exception as exc:  # reporting must never stop the stage
        print(f"(metadata matching report skipped: {type(exc).__name__}: {exc})")


def process_single_video(
    video_path: Path,
    yml_dir: Path,
    meta_map: dict,
    min_depth_px: int,
    config: PipelineConfig,
    pose_slp_suffix: str = ".slp",
    pose_dir: Optional[Path] = None,
) -> dict:
    """Process a single video for decision analysis with depth-based commit."""
    # Parse filename
    key = parse_filename(video_path)
    if key is None:
        return {"status": "skip_parse", "video": video_path.name}

    reward = meta_map.get((key["day"], key["mouse"]), "")
    return _analyse_trial(video_path, key, reward, yml_dir, min_depth_px, config,
                          pose_slp_suffix, pose_dir, blank_unknown_reward=False)


def process_trial(trial: dict, yml_dir, min_depth_px: int, config: PipelineConfig,
                  pose_slp_suffix: str = ".slp", pose_dir: Optional[Path] = None) -> dict:
    """Worker of the explicit-trial-list mode (`stride tmaze`): any file name, reward optional."""
    video_path = Path(trial["video_path"])
    key = {"day": trial.get("day") or "", "mouse": trial.get("mouse") or "",
           "trial": trial.get("trial")}
    return _analyse_trial(video_path, key, trial.get("reward") or "", yml_dir, min_depth_px,
                          config, pose_slp_suffix, pose_dir, blank_unknown_reward=True)


def _analyse_trial(video_path: Path, key: dict, reward: str, yml_dir, min_depth_px: int,
                   config: PipelineConfig, pose_slp_suffix: str, pose_dir: Optional[Path],
                   blank_unknown_reward: bool) -> dict:
    """Decision analysis of one video whose day / mouse / trial / reward are already known."""
    # Check for required files
    pose_path = resolve_pose_path(video_path, pose_slp_suffix, pose_dir)
    yml_path = resolve_roi_yml(video_path, yml_dir)

    if not pose_path.exists():
        return {"status": "skip_pose", "video": video_path.name}
    if yml_path is None:
        return {"status": "skip_yml", "video": video_path.name}

    try:
        # Load data with labels and tracks for depth calculation
        pct_cover, fps, roi_polys, labels, trx = load_analysis_data(
            video_path, pose_path, yml_path, config
        )

        # Run depth-based decision
        decision = decide_arm_entry_v2_depth(
            pct_cover, fps, reward, roi_polys, labels, trx,
            min_depth_px=min_depth_px, config=config
        )
        if blank_unknown_reward and str(reward).strip().upper()[:1] not in ("L", "R"):
            # reward unknown: the choice is reported, whether it was correct is left empty
            decision["correct_TF"] = ""
            decision["correct_bin"] = None

        # Compute dwell intervals for junction exploration counting
        dwell_df = find_dwell_intervals(
            pct_cover, ["junction", "arm_left", "arm_right", "segment1", "segment2", "segment3", "segment4"],
            fps, STRICT_THR,
        )

        # Build events row
        events = build_events_row(
            key, video_path, pct_cover, fps, decision, config,
            roi_polys=roi_polys, dwell_df=dwell_df,
        )

        # Build decisions row
        decisions = {
            "video_path": str(video_path),
            "stem": video_path.stem,
            "day": key["day"],
            "mouse": key["mouse"],
            "trial": key["trial"],
            "reward": reward,
            "fps": fps,
            **decision,
        }

        return {"status": "ok", "events": events, "decisions": decisions}

    except Exception as e:
        return {"status": f"error:{type(e).__name__}: {e}", "video": video_path.name}


def load_analysis_data(video_path, pose_path, yml_path, config):
    """Load all data needed for analysis including labels and tracks."""
    import sleap_io as sio

    # Load ROIs
    roi_polys = load_rois(yml_path)

    # Get frame dimensions and clip ROIs
    cap = cv2.VideoCapture(str(video_path))
    ret, frame0 = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read video: {video_path}")
    H, W = frame0.shape[:2]

    img_box = box(0, 0, W, H)
    roi_polys = {
        k: (v.intersection(img_box).buffer(0) if not v.is_empty else v)
        for k, v in roi_polys.items()
    }
    roi_names = sorted(roi_polys.keys())

    # Load poses (falls back to the on-disk video when the .slp embeds an unreadable video path)
    from stride.utils.pose_io import load_pose_labels
    labels = load_pose_labels(pose_path, video_path)
    video = labels.videos[0]
    fps = read_fps(video_path, labels_video=video, default=config.fps_default)

    # Build tracks
    trx = build_tracks(labels, video)

    # Select body keypoints for hull
    body_kpts = ["snout", "forepawL2", "forepawR2", "hindpawL2", "hindpawR2", "tailbase"]
    sel = [labels.skeleton.index(n) for n in body_kpts if n in labels.skeleton.node_names]

    if len(sel) < 3:
        raise ValueError("Not enough body keypoints found")

    pts = trx[:, sel, :].astype(np.float32)

    # Interpolate short gaps
    for j in range(pts.shape[1]):
        pts[:, j, :] = interpolate_gaps(pts[:, j, :], max_gap=7)

    # Column of the snout inside the gap-filled array `pts` (needed by count_snout_probes).
    # The snout stored in the coverage table is taken from `pts`, i.e. after interpolate_gaps
    # (max_gap=7), like the body keypoints. Reading the raw track instead leaves NaN on every
    # frame where the snout is not tracked, and a short tracking dropout then splits one probe
    # episode into several (probes_L/R inflated, probe_frames_L/R deflated).
    snout_col = None
    if "snout" in labels.skeleton.node_names:
        snout_col = sel.index(labels.skeleton.index("snout"))

    # Compute coverage
    rows = []
    for t in range(pts.shape[0]):
        poly = body_hull(pts[t])

        row = {"frame": t}

        # Store snout coordinates (for geometric probe detection): gap-filled, see above
        if snout_col is not None and np.all(np.isfinite(pts[t, snout_col])):
            row["snout_x"] = float(pts[t, snout_col, 0])
            row["snout_y"] = float(pts[t, snout_col, 1])
        else:
            row["snout_x"] = np.nan
            row["snout_y"] = np.nan

        if poly is None or poly.area < MIN_BODY_AREA:
            for rn in roi_names:
                row[f"pct_{rn}"] = np.nan
            rows.append(row)
            continue

        for rn in roi_names:
            inter = poly.intersection(roi_polys[rn])
            a_in = float(inter.area) if not inter.is_empty else 0.0
            row[f"pct_{rn}"] = 100.0 * a_in / poly.area
        rows.append(row)

    pct_cover = pd.DataFrame(rows)
    return pct_cover, fps, roi_polys, labels, trx


def decide_arm_entry_v2_depth(pct_cover, fps, reward_side, roi_polys, labels, trx,
                               min_depth_px=MIN_DEPTH_PX, config=None):
    """
    V2 depth-based decision algorithm.

    Strategy:
    1. Find all arm dwells after gate
    2. Measure depth from junction for each
    3. Filter: depth > min_depth_px AND hindpaw present
    4. Pick deepest penetration (not first or longest)
    """
    fv = fps if (fps and fps > 0) else FPS_DEFAULT
    min_valid_frame = int(round(MIN_VALID_SEC * fv))

    # Find gate frame
    gate_seg4 = find_seg4_gate(pct_cover, SEG4_LEAVE_THR, SEG4_LEAVE_K)
    if gate_seg4 is None:
        gate_frame = min_valid_frame
        gate_reason = "warmup_only"
    else:
        gate_frame = max(min_valid_frame, int(gate_seg4))
        gate_reason = "seg4_leave"

    # Helper: check if hindpaw is in arm
    def check_hindpaw_in_arm(pose_2d, arm_poly):
        def idx(name):
            return labels.skeleton.index(name)
        hind_parts = ["hindpawL2", "hindpawR2", "tailbase"]
        count = 0
        for name in hind_parts:
            try:
                pt = pose_2d[idx(name)]
                if np.all(np.isfinite(pt)) and arm_poly.contains(Point(pt)):
                    count += 1
            except (ValueError, IndexError):
                pass
        return count >= 1

    # Helper: calculate depth from junction
    def calc_depth(pose_2d):
        def idx(name):
            return labels.skeleton.index(name)
        try:
            snout = pose_2d[idx("snout")]
        except (ValueError, IndexError):
            snout = None

        if snout is None or not np.all(np.isfinite(snout)):
            valid_pts = pose_2d[np.all(np.isfinite(pose_2d), axis=1)]
            if len(valid_pts) == 0:
                return -999
            snout = valid_pts.mean(axis=0)

        snout_pt = Point(snout)
        junction = roi_polys.get("junction")
        if not junction:
            return 0

        if junction.contains(snout_pt):
            return -snout_pt.distance(junction.boundary)
        else:
            return snout_pt.distance(junction.boundary)

    # Find arm dwells with depth info
    def find_deep_entries(thr):
        dwells = find_dwell_intervals(pct_cover, ["arm_left", "arm_right"], fv, thr)
        if dwells.empty:
            return []

        dwells = dwells[dwells["start_frame"] >= gate_frame].copy()

        candidates = []
        for _, d in dwells.iterrows():
            side = "L" if d["region"] == "arm_left" else "R"
            f = int(d["start_frame"])

            if 0 <= f < trx.shape[0]:
                pose_2d = trx[f]
                arm_poly = roi_polys.get("arm_left" if side == "L" else "arm_right")

                has_hindpaw = check_hindpaw_in_arm(pose_2d, arm_poly) if arm_poly else False
                depth = calc_depth(pose_2d)

                candidates.append({
                    "region": d["region"],
                    "side": side,
                    "frame": f,
                    "length": int(d["length_frames"]),
                    "depth": depth,
                    "has_hindpaw": has_hindpaw,
                })
        return candidates

    # Try strict threshold
    candidates = find_deep_entries(STRICT_THR)
    entry_threshold = f">={int(STRICT_THR)}% for ≥{MIN_RUN}f"

    if not candidates:
        candidates = find_deep_entries(SOFT_THR)
        entry_threshold = f">={int(SOFT_THR)}% for ≥{MIN_RUN}f" if candidates else "none"

    if not candidates:
        return dict(
            ok=False, reason="no_arm_dwell_after_gate",
            gate_frame=int(gate_frame), gate_reason=gate_reason,
            entry_side="", entry_frame=np.nan, entry_ms=np.nan,
            entry_threshold="none", correct_TF="F", correct_bin=0
        )

    # Filter: deep entries with hindpaws
    deep_with_hind = [c for c in candidates if c["depth"] > min_depth_px and c["has_hindpaw"]]

    if deep_with_hind:
        # Pick deepest
        pick = max(deep_with_hind, key=lambda c: c["depth"])
    else:
        # Fallback: earliest with hindpaw, or just earliest
        with_hind = [c for c in candidates if c["has_hindpaw"]]
        pick = min(with_hind, key=lambda c: c["frame"]) if with_hind else candidates[0]

    side = pick["side"]
    f0 = pick["frame"]
    ms = 1000.0 * f0 / fv

    reward_side = (reward_side or "").strip().upper()[:1]
    is_correct = (side == reward_side) if reward_side in ("L", "R") else False

    return dict(
        ok=True, reason="",
        gate_frame=int(gate_frame), gate_reason=gate_reason,
        entry_side=side, entry_frame=int(f0), entry_ms=float(ms),
        entry_threshold=entry_threshold,
        correct_TF=("T" if is_correct else "F"),
        correct_bin=(1 if is_correct else 0)
    )


def count_reversals(pct_cover: pd.DataFrame, gate_frame: int,
                    commit_frame: int | None, fps: float,
                    min_coverage: float = 30.0) -> tuple[int, int]:
    """Count direction reversals (mouse retreating toward start) before commitment.

    Tracks the mouse's position along the maze longitudinal axis using the
    dominant coverage region per frame. A reversal = the mouse drops back by
    at least 1 full region after reaching a more forward position.

    Region ordering (start → goal):
      segment4 (0) → segment3 (1) → segment2 (2) → segment1 (3) → junction (4)

    Uses median filtering (11 frames) to suppress single-frame noise.
    Purely coverage-based — no gait/stride dependency.

    Returns: (n_reversals, reversal_frames)
    """
    from scipy.ndimage import median_filter as _mf

    if commit_frame is None or not np.isfinite(commit_frame):
        return 0, 0

    region_map = {
        "pct_segment4": 0, "pct_segment3": 1, "pct_segment2": 2,
        "pct_segment1": 3, "pct_junction": 4,
    }
    avail = [c for c in region_map if c in pct_cover.columns]
    if len(avail) < 3:
        return 0, 0

    scores = [region_map[c] for c in avail]
    vals = pct_cover[avail].values
    valid_mask = np.any(np.isfinite(vals), axis=1)

    # Assign position score per frame based on dominant region
    position = np.full(len(pct_cover), np.nan)
    for i in range(len(pct_cover)):
        if not valid_mask[i]:
            continue
        row = vals[i]
        best = np.nanargmax(row)
        if np.isfinite(row[best]) and row[best] >= min_coverage:
            position[i] = scores[best]

    # Smooth: interpolate short gaps, then median filter
    pos_smooth = pd.Series(position).interpolate(limit=5).values
    pos_smooth = _mf(pos_smooth, size=11)

    # Detect reversals within task window
    gate_frame = int(gate_frame)
    commit_frame = int(commit_frame)
    task_pos = pos_smooth[gate_frame:commit_frame]

    furthest = 0.0
    n_reversals = 0
    reversal_frames = 0
    in_reversal = False

    for pv in task_pos:
        if np.isnan(pv):
            continue
        if pv > furthest:
            furthest = pv
            if in_reversal:
                in_reversal = False
        elif pv <= furthest - 1:  # dropped back by 1+ region
            if not in_reversal:
                n_reversals += 1
                in_reversal = True
            reversal_frames += 1

    return n_reversals, reversal_frames


def build_events_row(key, video_path, pct_cover, fps, decision, config,
                     roi_polys=None, dwell_df=None):
    """Build events row matching events.v2.csv format."""
    # Find region entry times
    enter_seg1 = find_first_entry(pct_cover, "segment1", threshold=50.0)
    enter_seg4 = find_first_entry(pct_cover, "segment4", threshold=50.0)
    enter_junction = find_first_entry(pct_cover, "junction", threshold=50.0)

    gate_frame = decision.get("gate_frame", 0)
    commit_frame = decision.get("entry_frame", np.nan)

    # Count probes (geometric snout-in-polygon method) and junction exploration (dwell-based)
    probe_info = {"probes_L": 0, "probes_R": 0, "probe_frames_L": 0, "probe_frames_R": 0}
    junction_explore = 0
    n_reversals = 0
    reversal_frames = 0
    if not np.isnan(commit_frame):
        cf = int(commit_frame)
        if roi_polys is not None:
            probe_info = count_snout_probes(pct_cover, roi_polys, cf)
        junction_explore = count_junction_explore_frames(dwell_df, cf)
        n_reversals, reversal_frames = count_reversals(pct_cover, gate_frame, cf, fps)

    return {
        "day": key["day"],
        "mouse": key["mouse"],
        "trial": key["trial"],
        "stem": video_path.stem,
        "video_path": str(video_path),
        "enter_seg1_frame": enter_seg1,
        "enter_seg4_frame": enter_seg4,
        "enter_junction_frame": enter_junction,
        "commit_side": decision.get("entry_side", ""),
        "commit_frame": commit_frame,
        "ok": decision.get("ok", False),
        "enter_seg1_ms": enter_seg1 * 1000.0 / fps if enter_seg1 else np.nan,
        "enter_seg4_ms": enter_seg4 * 1000.0 / fps if enter_seg4 else np.nan,
        "enter_junction_ms": enter_junction * 1000.0 / fps if enter_junction else np.nan,
        "commit_ms": decision.get("entry_ms", np.nan),
        "junction_explore_frames_precommit": junction_explore,
        "junction_explore_ms_precommit": junction_explore * 1000.0 / fps,
        **probe_info,
        "n_reversals": n_reversals,
        "reversal_frames": reversal_frames,
        "reversal_ms": reversal_frames * 1000.0 / fps if fps > 0 else 0,
        "coverage_thr_rule": decision.get("entry_threshold", "none"),
    }


def build_metrics(events_df, decisions_df):
    """Build metrics.v2.csv from events and decisions."""
    # Merge events and decisions on key columns
    metrics = events_df.merge(
        decisions_df[["stem", "reward", "correct_TF", "correct_bin"]],
        on="stem", how="left"
    )

    # Calculate latencies
    metrics["choice_LR"] = metrics["commit_side"]
    metrics["latency_choice_ms"] = metrics["commit_ms"] - metrics["enter_seg4_ms"]
    metrics["stem_latency_ms"] = metrics["enter_junction_ms"] - metrics["enter_seg1_ms"]
    metrics["junction_explore_ms"] = metrics["junction_explore_ms_precommit"]

    # Select and order columns
    cols = [
        "day", "mouse", "trial", "stem", "video_path",
        "choice_LR", "latency_choice_ms", "stem_latency_ms", "junction_explore_ms",
        "reward", "correct_TF", "correct_bin",
        "probes_L", "probes_R", "probe_frames_L", "probe_frames_R"
    ]
    return metrics.reindex(columns=cols)


# ============================================================
# Helper Functions
# ============================================================

def parse_filename(video_path: Path) -> Optional[dict]:
    """Extract day, mouse, trial from filename like Day12_13982_Trial8.mp4 or Day1_WT_3_1_Trial8.mp4.

    Mouse token is kept as a string so genotype-prefixed IDs (e.g. WT_3_1, M4-1) work as well as
    purely numeric IDs; the meta side is normalized to str to match (see run_decision_analysis)."""
    pattern = re.compile(r"(?i)^day(\d+)_(.+?)_trial(\d+)\.mp4$")
    m = pattern.match(video_path.name)
    if not m:
        return None
    return {
        "day": f"DAY{int(m.group(1))}",
        "mouse": m.group(2).strip(),
        "trial": int(m.group(3)),
    }


def resolve_roi_yml(video_path: Path, yml_dir: Optional[Path]) -> Optional[Path]:
    """Locate a video's ROI yml, tolerating naming/layout variants.

    Searches (in order) the yml_dir then co-located next to the video, for these names:
      {stem}.preds.v2.best1.yml  (STRIDE inference output)
      {stem}.rois.yml            (per-day-subdir cohorts, ROIs co-located with videos)
      {stem}.yml
    Returns the first existing path, or None. Pass yml_dir=None for co-located-only.
    """
    stem = video_path.stem
    names = [f"{stem}.preds.v2.best1.yml", f"{stem}.rois.yml", f"{stem}.yml"]
    search_dirs = []
    if yml_dir is not None:
        # one folder, or several (searched in the order given)
        many = yml_dir if isinstance(yml_dir, (list, tuple)) else [yml_dir]
        search_dirs.extend(Path(d) for d in many)
    search_dirs.append(video_path.parent)  # co-located fallback
    for d in search_dirs:
        for n in names:
            p = d / n
            if p.exists():
                return p
    return None


def load_rois(yml_path: Path) -> dict[str, Polygon]:
    """Load ROI polygons from YAML file."""
    with open(yml_path, encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    name_map = {
        "segment1": "segment1", "segment2": "segment2", "segment3": "segment3",
        "segment4": "segment4", "junction": "junction",
        "arm_right": "arm_right", "arm_left": "arm_left",
    }

    polys = {}
    for r in y.get("rois", []):
        nm = name_map.get(r.get("name"))
        coords = r.get("coordinates", [])
        if nm and coords:
            poly = Polygon(coords)
            if not poly.is_valid:
                poly = poly.buffer(0)
            polys.setdefault(nm, []).append(poly)

    return {k: unary_union(vs) for k, vs in polys.items()}


def build_tracks(labels, video) -> np.ndarray:
    """Build T x J x 2 array of keypoint positions."""
    # Get frame count
    if hasattr(video, "frames") and video.frames:
        T = int(video.frames)
    elif hasattr(video, "shape") and len(video.shape) > 0:
        T = int(video.shape[0])
    else:
        T = len(labels)

    J = len(labels.skeleton.node_names)
    trx = np.full((T, J, 2), np.nan, dtype=np.float32)

    try:
        lfs = labels.find(video=video)
    except TypeError:
        lfs = [lf for lf in labels.labeled_frames if lf.video == video]

    for lf in lfs:
        t = int(lf.frame_idx)
        if not (0 <= t < T):
            continue
        insts = getattr(lf, "instances", None) or []
        if not insts:
            continue
        inst = max(insts, key=lambda i: getattr(i, "score", 0) or 0)
        pts = inst.numpy()
        if pts.ndim == 2 and pts.shape[1] == 2:
            jn = min(J, pts.shape[0])
            trx[t, :jn, :] = pts[:jn, :]

    return trx


def interpolate_gaps(track: np.ndarray, max_gap: int = 7) -> np.ndarray:
    """Interpolate short gaps in tracking data."""
    arr = track.copy()
    t = np.arange(arr.shape[0], dtype=np.float32)

    for d in (0, 1):
        y = arr[:, d]
        mask = np.isfinite(y)
        if mask.sum() < 2:
            continue
        ti, yi = t[mask], y[mask]
        y[:] = np.interp(t, ti, yi)

        # Restore long gaps
        left = np.where(mask, np.arange(len(y)), -np.inf)
        left = np.maximum.accumulate(left)
        right = np.where(mask, np.arange(len(y)), np.inf)
        for i in range(len(right) - 2, -1, -1):
            right[i] = min(right[i], right[i + 1])
        too_far = ((np.arange(len(y)) - left) > max_gap) & ((right - np.arange(len(y))) > max_gap)
        y[too_far] = np.nan
        arr[:, d] = y

    return arr


def body_hull(points: np.ndarray) -> Optional[Polygon]:
    """Compute convex hull polygon from body points."""
    P = np.asarray(points, np.float32)
    P = P[np.all(np.isfinite(P), axis=1)]
    if P.shape[0] < 3:
        return None
    hull = cv2.convexHull(P).reshape(-1, 2)
    poly = Polygon(hull)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if poly.area > 0 else None


def find_seg4_gate(pct_cover: pd.DataFrame, threshold: float, k_frames: int) -> Optional[int]:
    """Find frame when mouse leaves segment4 (start area)."""
    col = "pct_segment4"
    if col not in pct_cover.columns:
        return None

    x = pct_cover[col].to_numpy()
    good = np.isfinite(x) & (x < threshold)

    i, n = 0, len(good)
    while i < n:
        if good[i]:
            j = i
            while j < n and good[j]:
                j += 1
            if (j - i) >= k_frames:
                return int(pct_cover.iloc[i]["frame"])
            i = j
        else:
            i += 1
    return None


def find_dwell_intervals(pct_cover: pd.DataFrame, regions: list, fps: float, threshold: float) -> pd.DataFrame:
    """Find dwell intervals in each region."""
    results = []

    for region in regions:
        col = f"pct_{region}"
        if col not in pct_cover.columns:
            continue

        x = pct_cover[col].to_numpy()
        frames = pct_cover["frame"].to_numpy()
        good = np.isfinite(x) & (x >= threshold)

        # Bridge short gaps
        good = merge_short_gaps(good, GAP_ALLOW)

        # Find runs
        i, n = 0, len(good)
        while i < n:
            if good[i]:
                j = i
                while j < n and good[j]:
                    j += 1
                run_len = j - i
                if run_len >= MIN_RUN:
                    results.append({
                        "region": region,
                        "start_frame": int(frames[i]),
                        "end_frame": int(frames[j - 1]),
                        "length_frames": run_len,
                    })
                i = j
            else:
                i += 1

    return pd.DataFrame(results)


def merge_short_gaps(good: np.ndarray, gap_allow: int) -> np.ndarray:
    """Bridge short gaps in boolean array."""
    if gap_allow <= 0:
        return good

    out = good.copy()
    i, n = 0, len(good)

    while i < n:
        while i < n and not good[i]:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and good[j]:
            j += 1
        k = j
        while k < n and not good[k]:
            k += 1
        if k < n and 0 < (k - j) <= gap_allow:
            out[j:k] = True
        i = k

    return out


def find_first_entry(pct_cover: pd.DataFrame, region: str, threshold: float = 50.0) -> Optional[int]:
    """Find first frame where body coverage in region exceeds threshold."""
    col = f"pct_{region}"
    if col not in pct_cover.columns:
        return None
    x = pct_cover[col].to_numpy()
    good = np.isfinite(x) & (x >= threshold)
    idx = np.where(good)[0]
    if len(idx) > 0:
        return int(pct_cover.iloc[idx[0]]["frame"])
    return None


# Probe detection thresholds (from pilot_analysis.eda.01.ipynb)
PROBE_ARM_MAX = 40.0    # Max body coverage in arm (snout pokes, body stays in junction)
PROBE_JUNC_MIN = 30.0   # Min junction coverage (body must be in junction area)
PROBE_MIN_LEN = 2       # Min frames for a probe episode


def count_snout_probes(pct_cover: pd.DataFrame, roi_polys: dict,
                       commit_frame: int | None,
                       arm_max: float = PROBE_ARM_MAX,
                       junc_min: float = PROBE_JUNC_MIN,
                       min_len: int = PROBE_MIN_LEN) -> dict:
    """Count snout probes into each arm before commitment.

    A "probe" is a sequence of frames where:
      - Snout point is geometrically inside the arm polygon
      - Body coverage in that arm < arm_max (body not fully in arm)
      - Junction coverage >= junc_min (body is still in junction area)
      - Episode lasts >= min_len frames

    PROVENANCE: Ported verbatim from pilot_analysis.eda.01.ipynb lines 402-447.
    """
    if commit_frame is None:
        Tcut = len(pct_cover)
    else:
        if commit_frame in pct_cover["frame"].values:
            Tcut = int(pct_cover.index[pct_cover["frame"] == commit_frame].tolist()[0])
        else:
            Tcut = len(pct_cover)

    def in_poly(x, y, poly):
        if not np.isfinite(x) or not np.isfinite(y): return False
        try: return Point(float(x), float(y)).within(poly)
        except Exception: return False

    out = {}
    for arm, rn in [("L", "arm_left"), ("R", "arm_right")]:
        col_arm = f"pct_{rn}"
        col_j = "pct_junction" if "pct_junction" in pct_cover.columns else None
        probe_frames = []
        for _, r in pct_cover.iloc[:Tcut].iterrows():
            snx, sny = r.get("snout_x", np.nan), r.get("snout_y", np.nan)
            cond_snout = in_poly(snx, sny, roi_polys.get(rn, Polygon()))
            arm_pct = float(r.get(col_arm, np.nan)) if col_arm in pct_cover.columns else np.nan
            junc_pct = float(r.get(col_j, np.nan)) if col_j else np.nan
            cond_body = (np.isfinite(arm_pct) and arm_pct < arm_max)
            cond_junc = (np.isfinite(junc_pct) and junc_pct >= junc_min) if col_j else True
            if cond_snout and cond_body and cond_junc:
                probe_frames.append(int(r["frame"]))
        cnt_eps = 0; total_frames = len(probe_frames)
        if total_frames:
            pf = np.array(sorted(probe_frames), dtype=int)
            run_breaks = np.where(np.diff(pf, prepend=pf[0]) != 1)[0]
            run_breaks = np.append(run_breaks, len(pf))
            st = 0
            for ed in run_breaks:
                run_len = ed - st
                if run_len >= min_len: cnt_eps += 1
                st = ed
        out[f"probes_{arm}"] = int(cnt_eps)
        out[f"probe_frames_{arm}"] = int(total_frames)
    return out


def count_junction_explore_frames(dwell_df: pd.DataFrame, commit_frame: int | None) -> int:
    """Count frames spent in junction dwelling before commitment.

    Uses pre-computed dwell intervals (not raw coverage frames) to match
    the manual notebook's algorithm.

    PROVENANCE: Matches pilot_analysis.eda.01.ipynb lines 642-648.
    """
    if commit_frame is None or dwell_df is None or dwell_df.empty:
        return 0

    junc_dwells = dwell_df[
        (dwell_df["region"] == "junction") & (dwell_df["start_frame"] < commit_frame)
    ]

    total = 0
    for _, r in junc_dwells.iterrows():
        s = int(r["start_frame"])
        e = min(int(r["end_frame"]), commit_frame - 1)
        if e >= s:
            total += (e - s + 1)
    return total
