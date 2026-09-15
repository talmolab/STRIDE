# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pyyaml",
#     "sleap-io",
#     "sleap-nn",
# ]
# ///
"""
Config-driven SLEAP inference pipeline (Stages 1-2).

Stage 1:  Distortion check (optional)
Stage 2a: Pose estimation inference (.mp4 → .slp with 15 keypoints)
Stage 2b: ROI segmentation inference (.mp4 → .slp with 23 ROI keypoints)
Stage 2c: SLP to YAML conversion (.slp → .yml polygon files)

Skipped if .slp/.yml files already exist (resumable).
Skipped entirely if batch is marked finalized in config.

Usage:
    python -m stride.stages.run_inference --config config.yaml
    python -m stride.stages.run_inference --config config.yaml --batch gopro_sept2025
    python -m stride.stages.run_inference --config config.yaml --skip-pose  # only ROI + YAML
"""

import argparse
import warnings
from pathlib import Path

import yaml

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_batch(batch_key: str, batch_cfg: dict, config: dict, output_dir: Path,
                  skip_pose: bool = False, skip_roi: bool = False,
                  skip_distortion: bool = False):
    """Run inference stages for one batch."""
    print(f"\n{'='*60}")
    print(f"Batch: {batch_cfg['name']}")
    print(f"{'='*60}")

    if batch_cfg.get("finalized", False):
        print(f"  Batch is finalized. Skipping inference.")
        return

    video_dir = batch_cfg.get("video_dir", "")
    if not video_dir or not Path(video_dir).exists():
        print(f"  ERROR: video_dir not found: {video_dir}")
        return

    video_dir = Path(video_dir)

    # --- Stage 1: Distortion check ---
    if not skip_distortion:
        print(f"\n  Stage 1: Distortion check...")
        try:
            from stride.stages.distortion_check import check_distortion_batch
            result = check_distortion_batch(video_dir)
            needs = result.get("needs_undistortion", False)
            print(f"    Needs undistortion: {needs}")
            if needs:
                print(f"    WARNING: Videos need undistortion before proceeding.")
                print(f"    Use tmaze-undistort pipeline or cv2.undistort().")
        except Exception as e:
            print(f"    Distortion check skipped: {e}")

    # --- Stage 2a: Pose inference ---
    if not skip_pose:
        pose_models = batch_cfg.get("pose_model_paths", [])
        if not pose_models:
            print(f"\n  Stage 2a: No pose_model_paths in config. Skipping pose inference.")
        else:
            # Check if .slp files already exist for most videos
            videos = sorted({v for ext in ("mp4", "MP4") for v in video_dir.rglob(f"*.{ext}")})
            pose_suffix = batch_cfg.get("pose_slp_suffix", ".slp")
            existing = sum(1 for v in videos if v.with_suffix(pose_suffix).exists())

            if existing == len(videos):
                print(f"\n  Stage 2a: All {len(videos)} videos have pose .slp files. Skipping.")
            else:
                print(f"\n  Stage 2a: Pose inference ({existing}/{len(videos)} already done)...")
                try:
                    from stride.stages.pose_inference import run_pose_inference_batch
                    result = run_pose_inference_batch(
                        video_dir=video_dir,
                        model_paths=pose_models,
                        batch_size=16,
                    )
                    print(f"    Done: {result['done']} ok, {result['skipped']} skipped, {result['failed']} failed")
                except Exception as e:
                    print(f"    ERROR: {e}")

    # --- Stage 2b: ROI inference ---
    if not skip_roi:
        roi_models = batch_cfg.get("roi_model_paths", [])
        if not roi_models:
            print(f"\n  Stage 2b: No roi_model_paths in config. Skipping ROI inference.")
        else:
            # Output ROI .slp files go into output_dir/roi_slp/
            roi_slp_dir = output_dir / batch_key / "roi_slp"
            roi_slp_dir.mkdir(parents=True, exist_ok=True)

            videos = sorted({v for ext in ("mp4", "MP4") for v in video_dir.rglob(f"*.{ext}")})
            existing_roi = sum(1 for v in videos
                             if (roi_slp_dir / f"{v.stem}.preds.v2.best1.slp").exists())

            if existing_roi == len(videos):
                print(f"\n  Stage 2b: All {len(videos)} videos have ROI .slp files. Skipping.")
            else:
                print(f"\n  Stage 2b: ROI inference ({existing_roi}/{len(videos)} already done)...")
                try:
                    from stride.stages.roi_inference import run_roi_inference_batch
                    result = run_roi_inference_batch(
                        video_dir=video_dir,
                        output_dir=roi_slp_dir,
                    )
                    print(f"    Done: {result.get('passed', 0)} ok, {result.get('failed', 0)} failed")
                except Exception as e:
                    print(f"    ERROR: {e}")

            # --- Stage 2c: SLP to YAML conversion ---
            yml_dir = batch_cfg.get("yml_dir", "")
            if not yml_dir:
                yml_dir = output_dir / batch_key / "roi_yml"
            yml_dir = Path(yml_dir)
            yml_dir.mkdir(parents=True, exist_ok=True)

            existing_yml = sum(1 for v in videos
                              if (yml_dir / f"{v.stem}.preds.v2.best1.yml").exists())

            if existing_yml == len(videos):
                print(f"\n  Stage 2c: All {len(videos)} videos have .yml files. Skipping.")
            else:
                print(f"\n  Stage 2c: SLP → YAML conversion ({existing_yml}/{len(videos)} already done)...")
                try:
                    from stride.stages.slp_to_yaml import convert_batch
                    result = convert_batch(
                        slp_dir=roi_slp_dir,
                        yml_dir=yml_dir,
                    )
                    print(f"    Done: {result.get('ok', 0)} ok, {result.get('failed', 0)} failed")
                except Exception as e:
                    print(f"    ERROR: {e}")

    print(f"\n  Inference stages complete for {batch_key}.")


def main():
    parser = argparse.ArgumentParser(description="Config-driven SLEAP inference (Stages 1-2)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default="output")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    parser.add_argument("--skip-pose", action="store_true", help="Skip pose inference")
    parser.add_argument("--skip-roi", action="store_true", help="Skip ROI inference")
    parser.add_argument("--skip-distortion", action="store_true", help="Skip distortion check")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SLEAP Inference Pipeline — Stages 1-2")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    for batch_key, batch_cfg in config["batches"].items():
        if args.batch and args.batch != batch_key:
            continue
        process_batch(batch_key, batch_cfg, config, output_dir,
                      skip_pose=args.skip_pose,
                      skip_roi=args.skip_roi,
                      skip_distortion=args.skip_distortion)

    print("\nDone.")


if __name__ == "__main__":
    main()
