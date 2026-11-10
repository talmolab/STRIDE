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
Config-driven gait extraction pipeline (Stages 4-6).

Stage 4-5: Keypoint preprocessing + stride detection (via gait_extraction.py)
Stage 6:   Stride filtering (confidence, edge removal, angular velocity)

Inputs per batch: an explicit `metrics_meta_csv` / `yml_dir` in the batch config wins; without
them the stage reads what stage 3 wrote under <output>/<batch>/ (see stride.stages.batch_paths).

Usage:
    python -m stride.stages.run_gait_extraction --config config.yaml --output output
    python -m stride.stages.run_gait_extraction --config config.yaml --output output --batch my_batch
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

from stride.stages.gait_extraction import (
    DEFAULT_PARAMS,
    compute_keypoint_confidence,
    run_gait_extraction,
)
from stride.stages.batch_paths import (
    StageError, batch_name, default_paths, load_config_or_exit, missing_message,
    resolve_input, resolve_yml_dir, select_batches,
)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_params(config: dict) -> dict:
    """Build stride detection params from config, falling back to defaults."""
    sd = config.get("stride_detection", {})
    params = DEFAULT_PARAMS.copy()
    # Map config keys to params keys
    mapping = {
        "min_walk_velocity": "MIN_WALK_VEL",
        "min_paw_velocity": "MIN_PAW_VEL",
        "min_peak_paw_velocity": "MIN_PEAK_PAW_VEL",
        "min_frames_swing": "MIN_FRAMES_SWING",
        "smooth_median_window": "SMOOTH_MED_WIN",
        "smooth_moving_avg_window": "SMOOTH_MOV_WIN",
        "phase_spline_k": "PHASE_SPLINE_K",
        "phase_grid_points": "PHASE_GRID",
        "maze_width_cm": "MAZE_WIDTH_CM",
    }
    for config_key, param_key in mapping.items():
        if config_key in sd:
            params[param_key] = sd[config_key]
    # Override DEFAULT_FPS from batch fps if not in stride_detection
    return params


def filter_strides(stride_csv: Path, confidence_csv: Path, output_csv: Path,
                   config: dict) -> int:
    """Apply stride filtering: confidence, edge removal, angular velocity.

    PROVENANCE: Logic from:
      - 01.filter_strides_by_confidence.py (confidence + edge removal)
      - filter_strides_angular_velocity.py (angular velocity range)
    """
    sf = config.get("stride_filtering", {})
    conf_threshold = sf.get("confidence_threshold", 0.3)
    ang_min = sf.get("angular_velocity_min", -20.0)
    ang_max = sf.get("angular_velocity_max", 20.0)
    remove_edges = sf.get("remove_edge_strides", True)

    df = pd.read_csv(stride_csv)
    n_before = len(df)
    print(f"    Loaded {n_before:,} strides")
    counts = {"loaded": int(n_before)}          # for the run manifest (reporting only)
    stem_pos = list(df.columns).index("stem") if "stem" in df.columns else None

    # Step 1: Confidence filtering
    if confidence_csv.exists():
        df_conf = pd.read_csv(confidence_csv)
        # Compute mean confidence across all keypoints per video
        kp_cols = [c for c in df_conf.columns if c.startswith("avg_")]
        if kp_cols:
            df_conf["mean_confidence"] = df_conf[kp_cols].mean(axis=1)
            # Merge confidence into strides by stem
            if "stem" in df.columns and "stem" in df_conf.columns:
                df = df.merge(df_conf[["stem", "mean_confidence"]], on="stem", how="left")
                # Guard against a silent stem mismatch: unmatched rows get NaN confidence,
                # and NaN > threshold is False, so they would vanish without being counted.
                n_unmatched = int(df["mean_confidence"].isna().sum())
                if len(df) and n_unmatched == len(df):
                    raise RuntimeError(
                        "confidence merge matched 0 stems — stride stems and confidence stems "
                        f"disagree (e.g. stride {df['stem'].iloc[0]!r} vs confidence "
                        f"{df_conf['stem'].iloc[0] if len(df_conf) else '<empty>'!r}). "
                        "This happens when pose .slp files embed a different video name than "
                        "the file on disk (e.g. videos renamed after inference)."
                    )
                if n_unmatched:
                    print(f"    WARNING: {n_unmatched:,} strides have no confidence entry "
                          f"(stem mismatch); they are removed by this filter")
                n_low = int((~(df["mean_confidence"] > conf_threshold)).sum())
                df = df[df["mean_confidence"] > conf_threshold].copy()
                print(f"    Confidence filter (>{conf_threshold}): removed {n_low:,} strides")
                counts["removed_confidence"] = n_low
                counts["no_confidence_entry"] = n_unmatched
                df.drop(columns=["mean_confidence"], inplace=True, errors="ignore")
    else:
        print(f"    No confidence CSV found — skipping confidence filter")

    # Step 2: Edge stride removal
    if remove_edges and "stem" in df.columns and "frame_start" in df.columns:
        n_before_edge = len(df)
        df = df.sort_values(["stem", "frame_start"])
        # Remove first and last stride per video
        df = df.groupby("stem").apply(
            lambda g: g.iloc[1:-1] if len(g) > 2 else g.iloc[0:0],
            include_groups=False,
        )
        # `include_groups=False` drops the grouping column; keep `stem` in the filtered file by
        # putting it back at its original position (same rows, same order, same other columns).
        if "stem" not in df.columns and stem_pos is not None:
            stems = df.index.get_level_values(0) if df.index.nlevels > 1 else None
            df = df.reset_index(drop=True)
            if stems is not None and len(stems) == len(df):
                df.insert(min(stem_pos, len(df.columns)), "stem", list(stems))
        else:
            df = df.reset_index(drop=True)
        print(f"    Edge removal: removed {n_before_edge - len(df):,} strides")
        counts["removed_edge"] = int(n_before_edge - len(df))

    # Step 3: Angular velocity filtering
    ang_col = "angular_velocity_deg_s_mean"
    if ang_col in df.columns:
        n_before_ang = len(df)
        df = df[(df[ang_col] >= ang_min) & (df[ang_col] <= ang_max)].copy()
        print(f"    Angular velocity [{ang_min}, {ang_max}] deg/s: removed {n_before_ang - len(df):,} strides")
        counts["removed_angular_velocity"] = int(n_before_ang - len(df))

    # Step 4: Forward direction filtering
    # Backward strides (mice hesitating/reversing) have fundamentally different
    # kinematics (2x slower, 2.5x more sway, 5x more turning) and are not
    # correctable by sign-flipping. Exclude them from the gait analysis.
    # Detection: median(stride_length_cm) sign gives the forward convention per batch.
    sl_col = "stride_length_cm"
    filter_direction = sf.get("filter_forward_direction", True)
    if filter_direction and sl_col in df.columns and len(df):
        n_before_dir = len(df)
        median_sl = df[sl_col].median()
        forward_sign = np.sign(median_sl)
        if forward_sign != 0:
            backward_mask = (df[sl_col] * forward_sign) < 0
            n_backward = int(backward_mask.sum())
            df = df[~backward_mask].copy()
            direction = "positive" if forward_sign > 0 else "negative"
            print(f"    Forward direction filter (stride_length_cm {direction}): "
                  f"removed {n_backward:,} backward strides ({100*n_backward/n_before_dir:.1f}%)")
            counts["removed_backward"] = n_backward
            counts["forward_sign"] = direction

    df.to_csv(output_csv, index=False)
    print(f"    Final: {len(df):,} strides → {output_csv.name}")
    counts["final"] = int(len(df))

    # Provenance sidecar (best-effort; never breaks a completed run)
    try:
        from stride.run_manifest import write_run_manifest
        write_run_manifest(
            Path(output_csv).parent, "gait_filtering",
            params={"confidence_threshold": conf_threshold, "angular_velocity_min": ang_min,
                    "angular_velocity_max": ang_max, "remove_edge_strides": remove_edges,
                    "filter_forward_direction": filter_direction},
            inputs={"stride_csv": stride_csv, "confidence_csv": confidence_csv},
            outputs=[output_csv], results=counts,
        )
    except Exception:
        pass
    return len(df)


def process_batch(batch_key: str, batch_cfg: dict, config: dict, output_dir: Path):
    """Run Stages 4-6 for one batch. Raises StageError (with the reason) when it cannot run."""
    print(f"\n{'='*60}")
    print(f"Batch: {batch_name(batch_key, batch_cfg)}")
    print(f"{'='*60}")

    batch_out = output_dir / batch_key
    batch_out.mkdir(parents=True, exist_ok=True)

    filtered_csv = batch_out / "gait_per_stride_filtered.csv"

    # Check if finalized and output exists
    if batch_cfg.get("finalized", False):
        existing = batch_cfg.get("stride_csv", "")
        if existing and Path(existing).exists():
            print(f"  Batch is finalized. Using existing: {existing}")
            return
        if filtered_csv.exists():
            print(f"  Batch is finalized. Output exists: {filtered_csv}")
            return
        print(f"  Batch is finalized but no stride CSV found. Attempting extraction...")

    # Inputs: explicit config paths win; otherwise what stage 3 wrote under <output>/<batch>/
    defaults = default_paths(output_dir, batch_key)
    meta_path, source = resolve_input(batch_cfg, "metrics_meta_csv", defaults["metrics_meta_csv"])
    if meta_path is None and defaults["metrics_csv"].exists():
        meta_path, source = defaults["metrics_csv"], "previous stage (no metadata_csv: genotype / sex stay empty)"
    if meta_path is None:
        raise StageError(missing_message(
            "The per-video table for gait (metrics_meta_csv)", batch_cfg, "metrics_meta_csv",
            defaults["metrics_meta_csv"], "stage 3 (run_tmaze)"))
    meta_csv = str(meta_path)
    print(f"  Per-video table: {meta_csv}  [{source}]")
    yml_path = resolve_yml_dir(batch_cfg, output_dir, batch_key)
    print(f"  ROI files: {yml_path if yml_path is not None else 'next to the videos'}")
    yml_dir = str(yml_path) if yml_path is not None else None

    # Build params
    params = build_params(config)
    params["DEFAULT_FPS"] = batch_cfg.get("fps", params["DEFAULT_FPS"])
    # Batch-level calibration override: a ChArUco-verified px/cm constant takes
    # precedence over per-video segment2 derivation (see gait_extraction.py).
    # (The same override the original study used for cohorts with a board-verified scale.)
    if batch_cfg.get("px_per_cm_override"):
        params["PX_PER_CM_OVERRIDE"] = float(batch_cfg["px_per_cm_override"])

    pose_slp_suffix = batch_cfg.get("pose_slp_suffix", ".slp")
    column_map = batch_cfg.get("column_map", {"mouse_id": ["mouse"], "sex": "Gender"})
    n_workers = config.get("gait_extraction", {}).get("n_workers", 16)

    # Stage 4-5: Stride detection
    stride_csv = batch_out / "gait_per_stride.csv"
    if stride_csv.exists():
        print(f"  Stage 4-5: gait_per_stride.csv exists, skipping extraction")
    else:
        print(f"\n  Stage 4-5: Stride detection...")
        run_gait_extraction(
            meta_csv=meta_csv,
            yml_dir=yml_dir,
            output_dir=batch_out,
            params=params,
            pose_slp_suffix=pose_slp_suffix,
            column_map=column_map,
            n_workers=n_workers,
        )

    # Compute confidence CSV
    conf_csv = batch_out / "pose_keypoint_confidence_by_video.csv"
    video_dir = batch_cfg.get("video_dir", "")
    if not conf_csv.exists() and video_dir and Path(video_dir).exists():
        print(f"\n  Computing keypoint confidence...")
        compute_keypoint_confidence(
            video_dir=Path(video_dir),
            pose_slp_suffix=pose_slp_suffix,
            output_path=conf_csv,
        )

    # Stage 6: Stride filtering
    if not stride_csv.exists():
        raise StageError(
            f"no strides were extracted for batch '{batch_key}', so there is nothing to filter "
            f"(see the summary above: every video was skipped or failed; check `pose_slp_suffix`, "
            f"the pose files next to the videos and the video paths in {meta_csv})")

    print(f"\n  Stage 6: Stride filtering...")
    filter_strides(stride_csv, conf_csv, filtered_csv, config)


def main():
    parser = argparse.ArgumentParser(description="Config-driven gait extraction (Stages 4-6)")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config (default: ./config.yaml)")
    parser.add_argument("--output", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    args = parser.parse_args()

    config = load_config_or_exit(args.config)
    output_dir = Path(args.output)

    print("=" * 60)
    print("Gait Extraction Pipeline — Stages 4-6")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    failures = []
    for batch_key, batch_cfg in select_batches(config, args.batch):
        try:
            process_batch(batch_key, batch_cfg, config, output_dir)
        except StageError as exc:
            print(f"  ERROR: {exc}")
            failures.append(f"{batch_key}: {exc}")

    if failures:
        print(f"\nStage 4-6 FAILED for {len(failures)} batch(es):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
