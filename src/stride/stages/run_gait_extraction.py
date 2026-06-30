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

Usage:
    uv run python automated_pipeline/gait/run_gait_extraction.py
    uv run python automated_pipeline/gait/run_gait_extraction.py --batch gopro_sept2025
    uv run python automated_pipeline/gait/run_gait_extraction.py --config automated_pipeline/config.yaml
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


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
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
                n_low = (df["mean_confidence"] <= conf_threshold).sum()
                df = df[df["mean_confidence"] > conf_threshold].copy()
                print(f"    Confidence filter (>{conf_threshold}): removed {n_low:,} strides")
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
        ).reset_index(drop=True)
        print(f"    Edge removal: removed {n_before_edge - len(df):,} strides")

    # Step 3: Angular velocity filtering
    ang_col = "angular_velocity_deg_s_mean"
    if ang_col in df.columns:
        n_before_ang = len(df)
        df = df[(df[ang_col] >= ang_min) & (df[ang_col] <= ang_max)].copy()
        print(f"    Angular velocity [{ang_min}, {ang_max}] deg/s: removed {n_before_ang - len(df):,} strides")

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

    df.to_csv(output_csv, index=False)
    print(f"    Final: {len(df):,} strides → {output_csv.name}")
    return len(df)


def process_batch(batch_key: str, batch_cfg: dict, config: dict, output_dir: Path):
    """Run Stages 4-6 for one batch."""
    print(f"\n{'='*60}")
    print(f"Batch: {batch_cfg['name']}")
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

    # Check required fields
    meta_csv = batch_cfg.get("metrics_meta_csv", "")
    yml_dir = batch_cfg.get("yml_dir", "")
    if not meta_csv or not Path(meta_csv).exists():
        print(f"  ERROR: metrics_meta_csv not found: {meta_csv}")
        return
    if not yml_dir or not Path(yml_dir).exists():
        print(f"  ERROR: yml_dir not found: {yml_dir}")
        return

    # Build params
    params = build_params(config)
    params["DEFAULT_FPS"] = batch_cfg.get("fps", params["DEFAULT_FPS"])

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
        print(f"  ERROR: No gait_per_stride.csv to filter")
        return

    print(f"\n  Stage 6: Stride filtering...")
    filter_strides(stride_csv, conf_csv, filtered_csv, config)


def main():
    parser = argparse.ArgumentParser(description="Config-driven gait extraction (Stages 4-6)")
    parser.add_argument("--config", default="automated_pipeline/config.yaml")
    parser.add_argument("--output", default="automated_pipeline/output")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)

    print("=" * 60)
    print("Gait Extraction Pipeline — Stages 4-6")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for batch_key, batch_cfg in config["batches"].items():
        if args.batch and args.batch != batch_key:
            continue
        process_batch(batch_key, batch_cfg, config, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
