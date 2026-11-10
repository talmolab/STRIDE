"""
Pose estimation inference using SLEAP models.

Adapted from: 2025-11-04-GoPro_Tmaze_train/01.inference.py

This module runs pose estimation on videos to detect mouse body keypoints
(snout, paws, tail, etc.) for all frames.
"""

from pathlib import Path
from typing import Optional
from datetime import datetime

from stride.config import DEFAULT_POSE_MODELS, check_device, require_sleap_nn, resolve_model_paths
from stride.utils.parallel import find_videos


def pose_output_path(video_path: Path, output_dir: Optional[Path] = None) -> Path:
    """Where the pose file of a video is written (next to the video, or in output_dir)."""
    if output_dir is not None:
        return Path(output_dir) / f"{video_path.stem}.slp"
    return video_path.with_suffix(".slp")


def run_pose_inference_batch(
    video_dir: Path,
    output_dir: Optional[Path] = None,
    model_paths: Optional[list[Path]] = None,
    overwrite: bool = False,
    batch_size: int = 16,
    device: str = "cuda",
    worker_id: int = 0,
    num_workers: int = 1,
) -> dict:
    """
    Run pose inference on all videos in a directory (searched recursively, .mp4 / .MP4).

    For multi-GPU parallelism, run multiple processes with different worker_id values:
        # Terminal 1 - GPU 0
        stride pose-inference -i /videos -m <centroid> -m <centered_instance> --gpu 0 --worker-id 0 --num-workers 2

        # Terminal 2 - GPU 1
        stride pose-inference -i /videos -m <centroid> -m <centered_instance> --gpu 1 --worker-id 1 --num-workers 2

    Args:
        video_dir: Directory containing input videos
        output_dir: Directory for output .slp files (default: same as video)
        model_paths: List of model paths [centroid, centered_instance] (required)
        overwrite: Whether to overwrite existing outputs
        batch_size: Batch size for inference
        device: Device to run inference on (cuda, cuda:0, cuda:1, cpu)
        worker_id: Worker ID for parallel processing (0-indexed)
        num_workers: Total number of parallel workers

    Returns:
        Summary dict with counts

    Raises:
        ValueError: no model given / model path missing
        RuntimeError: sleap-nn not installed, or a CUDA device requested with no GPU visible
    """
    model_paths = resolve_model_paths(model_paths, "pose")

    all_videos = find_videos(video_dir, extensions=(".mp4",))

    # Partition videos across workers - each worker processes every Nth video
    if num_workers > 1:
        videos = [v for idx, v in enumerate(all_videos) if idx % num_workers == worker_id]
        print(f"Worker {worker_id + 1}/{num_workers}: processing {len(videos)}/{len(all_videos)} videos")
    else:
        videos = all_videos

    results = {
        "total": len(videos),
        "total_all": len(all_videos),
        "done": 0,
        "skipped": 0,
        "failed": 0,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "details": [],
    }

    if not all_videos:
        print(f"No .mp4 videos found under {video_dir} (searched recursively, any letter case)")
        return results

    # Only check the runtime / device when there is work to do (a finished folder still resumes).
    todo = [v for v in videos if overwrite or not pose_output_path(v, output_dir).exists()]
    if todo:
        require_sleap_nn()
        check_device(device)

    for i, video_path in enumerate(videos, 1):
        result = run_single_pose_inference(
            video_path=video_path,
            output_dir=output_dir,
            model_paths=model_paths,
            overwrite=overwrite,
            batch_size=batch_size,
            device=device,
        )

        results["details"].append(result)

        if result["status"] == "ok":
            results["done"] += 1
        elif result["status"] == "skip":
            results["skipped"] += 1
        else:
            results["failed"] += 1

        # Print progress (with the reason when a video failed)
        why = result.get("error") or result.get("reason") or ""
        why = f"  ({why})" if why and result["status"] not in ("ok", "skip") else ""
        print(f"[{i}/{len(videos)}] {result['status'].upper():5} {video_path.name}{why}")

    return results


def run_single_pose_inference(
    video_path: Path,
    output_dir: Optional[Path] = None,
    model_paths: Optional[list[str]] = None,
    overwrite: bool = False,
    batch_size: int = 16,
    device: str = "cuda",
) -> dict:
    """
    Run pose inference on a single video.

    Args:
        video_path: Path to input video
        output_dir: Output directory (default: same as video)
        model_paths: List of model paths
        overwrite: Whether to overwrite existing outputs
        batch_size: Batch size for inference
        device: Device for inference

    Returns:
        Result dict with status
    """
    if not model_paths:
        return {
            "status": "error",
            "video": video_path.name,
            "error": "No pose models given",
        }

    # Output path - co-located with video or in output_dir
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = pose_output_path(video_path, output_dir)

    if output_path.exists() and not overwrite:
        return {"status": "skip", "video": video_path.name, "reason": "exists"}

    try:
        from sleap_nn.predict import run_inference

        start_time = datetime.now()

        run_inference(
            data_path=str(video_path),
            model_paths=model_paths,
            output_path=str(output_path),
            batch_size=batch_size,
            queue_maxsize=batch_size * 2,
            device=device,
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if output_path.exists():
            return {
                "status": "ok",
                "video": video_path.name,
                "output": str(output_path),
                "elapsed_sec": elapsed,
            }
        else:
            return {
                "status": "fail",
                "video": video_path.name,
                "reason": "no_output_written",
            }

    except Exception as e:
        return {
            "status": "error",
            "video": video_path.name,
            "error": f"{type(e).__name__}: {e}",
        }


def check_pose_models(model_paths: Optional[list] = None) -> tuple[bool, str]:
    """
    Check if pose models exist.

    Returns:
        Tuple of (success, message)
    """
    try:
        paths = resolve_model_paths(model_paths, "pose")
    except ValueError as e:
        return False, str(e)
    return True, f"Found {len(paths)} model(s)"
