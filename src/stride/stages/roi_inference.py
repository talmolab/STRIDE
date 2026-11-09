"""
ROI keypoint inference using SLEAP models.

Adapted from: 2025-12-08-GoPro-roi-inference/00.run-inference.py

This module runs inference on videos to detect ROI keypoints (23 points
for horizontal T-maze), then selects the best frame from the first N frames.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from stride.config import (
    DEFAULT_ROI_MODELS, STRICT_REQUIRED_23, RELAXED_REQUIRED_20,
    check_device, require_sleap_nn, resolve_model_paths,
)
from stride.utils.parallel import run_parallel, find_videos


def roi_output_path(video_path: Path, output_dir: Path) -> Path:
    """Where the single-frame ROI prediction of a video is written."""
    return Path(output_dir) / f"{video_path.stem}.preds.v2.best1.slp"


def run_roi_inference_batch(
    video_dir: Path,
    output_dir: Path,
    model_paths: Optional[list[Path]] = None,
    overwrite: bool = False,
    frames_to_check: int = 10,
    worker_id: int = 0,
    num_workers: int = 1,
    device: Optional[str] = None,
) -> dict:
    """
    Run ROI inference on all videos in a directory (searched recursively, .mp4 / .MP4).

    For multi-GPU parallelism, run multiple processes with different worker_id values:
        # Terminal 1 - GPU 0
        stride roi-inference -i /videos -o /output -m <centroid> -m <centered_instance> --gpu 0 --worker-id 0 --num-workers 2

        # Terminal 2 - GPU 1
        stride roi-inference -i /videos -o /output -m <centroid> -m <centered_instance> --gpu 1 --worker-id 1 --num-workers 2

    Args:
        video_dir: Directory containing input videos
        output_dir: Directory for output .slp files
        model_paths: List of model paths [centroid, centered_instance] (required)
        overwrite: Whether to overwrite existing outputs
        frames_to_check: Number of frames to check for best ROI
        worker_id: Worker ID for parallel processing (0-indexed)
        num_workers: Total number of parallel workers
        device: Device for inference (cuda, cuda:0, cpu). None = let sleap-nn choose
            (the behaviour before this option existed).

    Returns:
        Summary dict with counts

    Raises:
        ValueError: no model given / model path missing
        RuntimeError: sleap-nn not installed, or a CUDA device requested with no GPU visible
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_paths = resolve_model_paths(model_paths, "ROI")

    all_videos = find_videos(video_dir, extensions=(".mp4",))

    # Partition videos across workers - each worker processes every Nth video
    if num_workers > 1:
        videos = [v for idx, v in enumerate(all_videos) if idx % num_workers == worker_id]
        print(f"Worker {worker_id + 1}/{num_workers}: processing {len(videos)}/{len(all_videos)} videos")
    else:
        videos = all_videos

    if not all_videos:
        print(f"No .mp4 videos found under {video_dir} (searched recursively, any letter case)")

    # Only check the runtime / device when there is work to do (a finished folder still resumes).
    todo = [v for v in videos if overwrite or not roi_output_path(v, output_dir).exists()]
    if todo:
        require_sleap_nn()
        check_device(device)

    results = []
    for i, video_path in enumerate(videos, 1):
        result = run_single_video_inference(
            video_path=video_path,
            output_dir=output_dir,
            model_paths=model_paths,
            overwrite=overwrite,
            frames_to_check=frames_to_check,
            device=device,
        )
        results.append(result)
        status = result.get("status", "error")
        why = result.get("error") or result.get("reason") or ""
        why = f"  ({why})" if why and status not in ("ok", "skip") else ""
        print(f"[{i}/{len(videos)}] {status.upper():5} {video_path.name}{why}")

    # Summarize
    passed = sum(1 for r in results if r.get("status") == "ok")
    failed = sum(1 for r in results if r.get("status") not in ("ok", "skip"))
    skipped = sum(1 for r in results if r.get("status") == "skip")

    return {
        "total": len(videos),
        "total_all": len(all_videos),
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "worker_id": worker_id,
        "num_workers": num_workers,
        "results": results,
    }


def run_single_video_inference(
    video_path: Path,
    output_dir: Path,
    model_paths: list[str],
    overwrite: bool = False,
    frames_to_check: int = 10,
    device: Optional[str] = None,
) -> dict:
    """
    Run ROI inference on a single video.

    Returns:
        Result dict with status
    """
    output_path = roi_output_path(video_path, output_dir)

    if output_path.exists() and not overwrite:
        return {"status": "skip", "video": video_path.name, "reason": "exists"}

    try:
        import sleap_io as sio
        from sleap_nn.predict import run_inference

        # Run inference on first N frames
        tmp_path = output_dir / f"{video_path.stem}.tmp.slp"

        # `device` is forwarded only when the caller asked for one; otherwise sleap-nn picks
        # the device itself, exactly as before the option existed.
        extra = {"device": device} if device else {}
        run_inference(
            data_path=str(video_path),
            model_paths=model_paths,
            output_path=str(tmp_path),
            frames=list(range(frames_to_check)),
            **extra,
        )

        # Load predictions and select best frame
        labels = sio.load_file(str(tmp_path))
        best_idx, best_lf = select_best_frame(labels, max_check=frames_to_check)

        if best_lf is None:
            return {"status": "fail", "video": video_path.name,
                    "reason": f"no_valid_frame ({best_idx}: no frame in the first "
                              f"{frames_to_check} has all required ROI keypoints)"}

        # Save single-frame result
        labels_out = sio.Labels(
            labeled_frames=[best_lf],
            videos=labels.videos,
            skeletons=labels.skeletons if hasattr(labels, "skeletons") else [labels.skeleton],
        )
        # Provenance: keep what sleap-nn recorded and add which models / device / frame were used,
        # so the kept ROI file says how it was produced (best-effort, never fails the video).
        try:
            prov = dict(getattr(labels, "provenance", None) or {})
            prov.update({
                "stride_stage": "roi_inference",
                "stride_roi_model_paths": [str(m) for m in model_paths],
                "stride_device": str(device) if device else "auto (chosen by sleap-nn)",
                "stride_frames_checked": int(frames_to_check),
                "stride_best_frame_index": int(best_idx) if isinstance(best_idx, (int, np.integer)) else str(best_idx),
                "stride_source_video": str(video_path),
            })
            labels_out.provenance = prov
        except Exception:
            pass
        sio.save_file(labels_out, str(output_path))

        # Clean up temp file
        tmp_path.unlink(missing_ok=True)

        return {"status": "ok", "video": video_path.name, "output": str(output_path)}

    except Exception as e:
        return {"status": "error", "video": video_path.name, "error": f"{type(e).__name__}: {e}"}


def select_best_frame(labels, max_check: int = 10):
    """
    Select best frame from predictions using strict (23) then relaxed (20) criteria.

    Returns:
        Tuple of (frame_index, labeled_frame) or (video_name, None) if none valid
    """
    import re

    def _norm(name: str) -> str:
        return re.sub(r"[^a-z0-9]", "", name.lower())

    def get_skeleton_node_names(pred_labels):
        if hasattr(pred_labels, "skeleton") and hasattr(pred_labels.skeleton, "nodes"):
            return [n.name for n in pred_labels.skeleton.nodes]
        try:
            lf0 = pred_labels[0]
            if lf0.instances and hasattr(lf0.instances[0], "skeleton"):
                return [n.name for n in lf0.instances[0].skeleton.nodes]
        except Exception:
            pass
        return []

    skeleton_nodes = get_skeleton_node_names(labels)
    if not skeleton_nodes:
        return "UNKNOWN", None

    exact = {name: i for i, name in enumerate(skeleton_nodes)}
    norm = {_norm(name): i for i, name in enumerate(skeleton_nodes)}

    def lookup_idx(name):
        if name in exact:
            return exact[name]
        return norm.get(_norm(name))

    def get_point_xy(instance, idx):
        if hasattr(instance, "points"):
            pts = instance.points
            if 0 <= idx < len(pts):
                rec = pts[idx]
                if hasattr(rec, "dtype") and hasattr(rec.dtype, "names") and "xy" in rec.dtype.names:
                    xy = rec["xy"]
                    arr = np.asarray(xy, dtype=np.float64).reshape(-1)
                    if arr.size >= 2:
                        return [float(arr[0]), float(arr[1])]
                try:
                    arr = np.asarray(rec, dtype=np.float64).reshape(-1)
                    if arr.size >= 2:
                        return [float(arr[0]), float(arr[1])]
                except Exception:
                    pass
        return [np.nan, np.nan]

    def is_finite_xy(pt):
        arr = np.asarray(pt, dtype=np.float64).reshape(-1)
        return arr.size >= 2 and np.isfinite(arr[:2]).all()

    def eval_frame(lf, required_nodes):
        if not getattr(lf, "instances", None):
            return False, -1.0
        inst = lf.instances[0]
        for name in required_nodes:
            j = lookup_idx(name)
            if j is None:
                return False, -1.0
            xy = get_point_xy(inst, j)
            if not is_finite_xy(xy):
                return False, -1.0
        inst_score = float(getattr(inst, "score", 0.0) or 0.0)
        return True, inst_score

    # Try strict first, then relaxed
    best_strict = None
    best_relaxed = None
    nframes = min(max_check, len(labels))

    for idx in range(nframes):
        lf = labels[idx]

        ok_s, score_s = eval_frame(lf, STRICT_REQUIRED_23)
        if ok_s:
            if best_strict is None or score_s > best_strict[0]:
                best_strict = (score_s, idx, lf)
            continue

        ok_r, score_r = eval_frame(lf, RELAXED_REQUIRED_20)
        if ok_r:
            if best_relaxed is None or score_r > best_relaxed[0]:
                best_relaxed = (score_r, idx, lf)

    if best_strict is not None:
        return best_strict[1], best_strict[2]
    if best_relaxed is not None:
        return best_relaxed[1], best_relaxed[2]

    return "NO_VALID_FRAME", None
