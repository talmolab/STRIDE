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

Models: `pose_model_paths` / `roi_model_paths` in the batch (STRIDE ships none). A batch without
them skips that inference step and uses the pose / ROI files that already exist.
Device: --device, else `device:` in the batch, else `inference: {device: ...}` in the config,
else "cuda" for pose (as before) and sleap-nn's own choice for ROI (as before).

Usage:
    python -m stride.stages.run_inference --config config.yaml
    python -m stride.stages.run_inference --config config.yaml --batch my_batch
    python -m stride.stages.run_inference --config config.yaml --skip-pose  # only ROI + YAML
    python -m stride.stages.run_inference --config config.yaml --device cpu
"""

import argparse
import warnings
from pathlib import Path

import yaml

from stride.stages.batch_paths import batch_name, load_config_or_exit, select_batches
from stride.utils.parallel import find_videos

warnings.filterwarnings("ignore")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def process_batch(batch_key: str, batch_cfg: dict, config: dict, output_dir: Path,
                  skip_pose: bool = False, skip_roi: bool = False,
                  skip_distortion: bool = False, device: str | None = None) -> list[str]:
    """Run inference stages for one batch. Returns the list of failures (empty = all good)."""
    failures: list[str] = []
    print(f"\n{'='*60}")
    print(f"Batch: {batch_name(batch_key, batch_cfg)}")
    print(f"{'='*60}")

    if batch_cfg.get("finalized", False):
        print(f"  Batch is finalized. Skipping inference.")
        return failures

    video_dir = batch_cfg.get("video_dir", "")
    if not video_dir or not Path(video_dir).exists():
        print(f"  ERROR: video_dir not found: {video_dir}")
        return [f"video_dir not found: {video_dir!r} (edit the batch in your config)"]

    video_dir = Path(video_dir)
    inf_cfg = config.get("inference", {}) or {}
    # device: CLI > batch > config.inference > default (pose: "cuda"; ROI: chosen by sleap-nn)
    cfg_device = device or batch_cfg.get("device") or inf_cfg.get("device")
    pose_device = cfg_device or "cuda"
    batch_size = int(batch_cfg.get("batch_size") or inf_cfg.get("batch_size") or 16)

    # One recursive, case-insensitive search shared by every step (per-day subfolders, .mp4/.MP4)
    videos = find_videos(video_dir, extensions=(".mp4",))
    print(f"  Videos: {len(videos)} under {video_dir}")
    if not videos:
        return [f"no .mp4 videos found under {video_dir}"]
    manifest = {"n_videos": len(videos), "pose": None, "roi": None, "yml": None, "distortion": None}

    # --- Stage 1: Distortion check ---
    if not skip_distortion:
        print(f"\n  Stage 1: Distortion check...")
        try:
            from stride.stages.distortion_check import check_distortion_batch
            result = check_distortion_batch(video_dir)          # one dict per video
            n_board = sum(1 for r in result if r.get("needs_undistortion") is not None)
            needs = sum(1 for r in result if r.get("needs_undistortion"))
            manifest["distortion"] = {"videos": len(result), "with_board": n_board, "need_undistortion": needs}
            if n_board == 0:
                print(f"    No ChArUco board found in any of {len(result)} videos: distortion not assessed.")
            else:
                print(f"    Board found in {n_board}/{len(result)} videos; needs undistortion: {needs}")
            if needs:
                print(f"    WARNING: {needs} video(s) need undistortion before proceeding.")
                print(f"    Use tmaze-undistort pipeline or cv2.undistort().")
        except Exception as e:
            print(f"    ERROR: distortion check failed: {type(e).__name__}: {e}")
            failures.append(f"distortion check: {type(e).__name__}: {e}")

    # --- Stage 2a: Pose inference ---
    if not skip_pose:
        pose_models = batch_cfg.get("pose_model_paths", [])
        pose_suffix = batch_cfg.get("pose_slp_suffix", ".slp")
        existing = sum(1 for v in videos if v.with_suffix(pose_suffix).exists())
        if not pose_models:
            print(f"\n  Stage 2a: No pose_model_paths in the batch. Skipping pose inference "
                  f"({existing}/{len(videos)} videos already have a pose file).")
            if existing < len(videos):
                print(f"    WARNING: {len(videos) - existing} video(s) have no pose file "
                      f"<video name>{pose_suffix}; the analysis stages will skip them. "
                      f"Set `pose_model_paths` to run pose inference here.")
        elif existing == len(videos):
            print(f"\n  Stage 2a: All {len(videos)} videos have pose .slp files. Skipping.")
        else:
            print(f"\n  Stage 2a: Pose inference ({existing}/{len(videos)} already done, device {pose_device})...")
            try:
                from stride.stages.pose_inference import run_pose_inference_batch
                result = run_pose_inference_batch(
                    video_dir=video_dir,
                    model_paths=pose_models,
                    batch_size=batch_size,
                    device=pose_device,
                )
                print(f"    Done: {result['done']} ok, {result['skipped']} skipped, {result['failed']} failed")
                manifest["pose"] = {k: result[k] for k in ("total", "done", "skipped", "failed")}
                if result["failed"]:
                    why = next((d.get("error") or d.get("reason") for d in result["details"]
                                if d.get("status") not in ("ok", "skip")), "")
                    failures.append(f"pose inference failed for {result['failed']} video(s) (first: {why})")
            except Exception as e:
                print(f"    ERROR: {e}")
                failures.append(f"pose inference: {e}")

    # --- Stage 2b: ROI inference ---
    if not skip_roi:
        roi_models = batch_cfg.get("roi_model_paths", [])
        if not roi_models:
            print(f"\n  Stage 2b: No roi_model_paths in the batch. Skipping ROI inference "
                  f"(existing ROI files are used: `yml_dir`, or <video name>.rois.yml next to the videos).")
        else:
            # Output ROI .slp files go into output_dir/roi_slp/
            roi_slp_dir = output_dir / batch_key / "roi_slp"
            roi_slp_dir.mkdir(parents=True, exist_ok=True)

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
                        model_paths=roi_models,
                        device=cfg_device,
                    )
                    print(f"    Done: {result.get('passed', 0)} ok, {result.get('failed', 0)} failed")
                    manifest["roi"] = {k: result.get(k) for k in ("total", "passed", "skipped", "failed")}
                    if result.get("failed"):
                        why = next((d.get("error") or d.get("reason") for d in result["results"]
                                    if d.get("status") not in ("ok", "skip")), "")
                        failures.append(f"ROI inference failed for {result['failed']} video(s) (first: {why})")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    failures.append(f"ROI inference: {e}")

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
                    print(f"    Done: {result.get('ok', 0)} ok, {result.get('failed', 0)} failed, "
                          f"{result.get('missing_keypoints', 0)} with missing keypoints")
                    manifest["yml"] = dict(result)
                    bad = result.get("failed", 0) + result.get("missing_keypoints", 0)
                    if bad:
                        failures.append(f"ROI SLP -> YAML conversion failed for {bad} file(s)")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    failures.append(f"ROI SLP -> YAML conversion: {e}")

    # Provenance sidecar (best-effort; never breaks a completed run)
    try:
        from stride.run_manifest import write_run_manifest
        write_run_manifest(
            output_dir / batch_key, "inference",
            models=[*(batch_cfg.get("pose_model_paths") or []), *(batch_cfg.get("roi_model_paths") or [])] or None,
            params={"pose_device": pose_device, "roi_device": cfg_device or "auto (chosen by sleap-nn)",
                    "batch_size": batch_size, "skip_pose": skip_pose, "skip_roi": skip_roi,
                    "skip_distortion": skip_distortion,
                    "pose_slp_suffix": batch_cfg.get("pose_slp_suffix", ".slp")},
            inputs={"video_dir": video_dir, "n_videos": len(videos)},
            results={**manifest, "failures": failures or None},
        )
    except Exception:
        pass

    if failures:
        print(f"\n  Inference stages FAILED for {batch_key}:")
        for f in failures:
            print(f"    - {f}")
    else:
        print(f"\n  Inference stages complete for {batch_key}.")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Config-driven SLEAP inference (Stages 1-2)")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config (default: ./config.yaml)")
    parser.add_argument("--output", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    parser.add_argument("--skip-pose", action="store_true", help="Skip pose inference")
    parser.add_argument("--skip-roi", action="store_true", help="Skip ROI inference")
    parser.add_argument("--skip-distortion", action="store_true", help="Skip distortion check")
    parser.add_argument("--device", default=None,
                        help="Inference device (cuda, cuda:0, cpu). Default: `device` in the batch / "
                             "`inference.device` in the config, else cuda for pose")
    args = parser.parse_args()

    config = load_config_or_exit(args.config)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SLEAP Inference Pipeline — Stages 1-2")
    print(f"Config: {args.config}")
    print(f"Output: {output_dir}")
    print("=" * 60)

    all_failures = []
    for batch_key, batch_cfg in select_batches(config, args.batch):
        fails = process_batch(batch_key, batch_cfg, config, output_dir,
                              skip_pose=args.skip_pose,
                              skip_roi=args.skip_roi,
                              skip_distortion=args.skip_distortion,
                              device=args.device)
        all_failures += [f"{batch_key}: {f}" for f in fails]

    if all_failures:
        print(f"\nStage 1-2 FAILED ({len(all_failures)} problem(s)):")
        for f in all_failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
