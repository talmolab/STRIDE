"""
ROI keypoint inference using SLEAP models.

This module runs the ROI model on a few frames of each video (23 maze keypoints for the
horizontal T-maze) and keeps the best prediction as a one-frame ``.slp``:

1. slice pass   - 10 frames sampled evenly from the middle of the video (20 %-80 %);
2. full pass    - only if the slice pass finds no usable frame: the whole video;
3. selection    - see :mod:`stride.stages.roi_selection` (every instance of a frame is evaluated,
                  strict 23 nodes before all-16-corners, only the chosen instance is kept).

A video with no usable frame in either pass has something hiding a maze corner for the whole clip;
:mod:`stride.stages.roi_consensus` can then borrow the ROI of the other clips of the same session.
"""

from pathlib import Path
from typing import Optional
import numpy as np

from stride.config import (
    DEFAULT_ROI_MODELS, STRICT_REQUIRED_23, RELAXED_REQUIRED_20,
    check_device, require_sleap_nn, resolve_model_paths,
)
from stride.stages.roi_selection import (
    build_single_frame_labels, get_sample_frames, select_best_frame_v2,
)
from stride.utils.parallel import run_parallel, find_videos

FULL_PASS_BATCH_SIZE = 16   # batch size of the full-video pass (GPU-bound; lower it on a small GPU)


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
    full_pass: bool = True,
    full_batch_size: int = FULL_PASS_BATCH_SIZE,
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
        frames_to_check: Number of frames sampled from the middle of the video in the slice pass
        worker_id: Worker ID for parallel processing (0-indexed)
        num_workers: Total number of parallel workers
        device: Device for inference (cuda, cuda:0, cpu). None = let sleap-nn choose
            (the behaviour before this option existed).
        full_pass: Re-run the whole video when the slice pass finds no usable frame
        full_batch_size: Batch size of that full-video pass

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
            full_pass=full_pass,
            full_batch_size=full_batch_size,
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
    full_pass: bool = True,
    full_batch_size: int = FULL_PASS_BATCH_SIZE,
) -> dict:
    """
    Run ROI inference on a single video: slice pass, then (if needed) the full-video pass.

    Returns:
        Result dict with ``status`` (ok | skip | fail | error); ``pass`` says which pass
        produced the kept frame ("slice" or "full").
    """
    output_path = roi_output_path(video_path, output_dir)

    if output_path.exists() and not overwrite:
        return {"status": "skip", "video": video_path.name, "reason": "exists"}

    try:
        import sleap_io as sio
        from sleap_nn.predict import run_inference

        # `device` is forwarded only when the caller asked for one; otherwise sleap-nn picks
        # the device itself.
        extra = {"device": device} if device else {}

        # 1) slice pass: frames sampled from the middle of the video
        tmp_path = output_dir / f"{video_path.stem}.preds.v2.slice{frames_to_check}.tmp.slp"
        sample_frames = get_sample_frames(video_path, num_frames=frames_to_check)
        run_inference(
            data_path=str(video_path),
            model_paths=model_paths,
            output_path=str(tmp_path),
            frames=sample_frames,
            **extra,
        )
        labels = sio.load_file(str(tmp_path))
        best_idx, best_lf = select_best_frame_v2(labels, max_check=frames_to_check)
        used_pass = "slice"
        tmp_paths = [tmp_path]

        # 2) full pass: the whole video, only when the slice pass found no usable frame
        if best_lf is None and full_pass:
            tmp_full = output_dir / f"{video_path.stem}.preds.v2.full.tmp.slp"
            tmp_paths.append(tmp_full)
            run_inference(
                data_path=str(video_path),
                model_paths=model_paths,
                output_path=str(tmp_full),
                batch_size=full_batch_size,
                **extra,
            )
            labels = sio.load_file(str(tmp_full))
            best_idx, best_lf = select_best_frame_v2(labels, max_check=None)
            used_pass = "full"

        if best_lf is None:
            for t in tmp_paths:
                Path(t).unlink(missing_ok=True)
            why = "no_passing_frame_full" if full_pass else "no_passing_frame"
            return {"status": "fail", "video": video_path.name,
                    "reason": f"{why} (no frame with all 16 maze corners in "
                              f"{'the whole video' if full_pass else f'the {frames_to_check} sampled frames'}: "
                              f"a corner is hidden for the whole clip)"}

        # Save the single-frame result (only the chosen instance)
        labels_out = build_single_frame_labels(labels, best_lf)
        # Provenance: keep what sleap-nn recorded and add which models / device / frame were used,
        # so the kept ROI file says how it was produced (best-effort, never fails the video).
        try:
            prov = dict(getattr(labels, "provenance", None) or {})
            prov.update({
                "stride_stage": "roi_inference",
                "stride_roi_model_paths": [str(m) for m in model_paths],
                "stride_device": str(device) if device else "auto (chosen by sleap-nn)",
                "stride_roi_pass": used_pass,
                "stride_frames_checked": int(frames_to_check) if used_pass == "slice" else int(len(labels)),
                "stride_best_frame_index": int(best_idx) if isinstance(best_idx, (int, np.integer)) else str(best_idx),
                "stride_best_video_frame": int(getattr(best_lf, "frame_idx", -1)),
                "stride_source_video": str(video_path),
            })
            labels_out.provenance = prov
        except Exception:
            pass
        sio.save_file(labels_out, str(output_path))

        # Clean up temp files
        for t in tmp_paths:
            Path(t).unlink(missing_ok=True)

        return {"status": "ok", "video": video_path.name, "output": str(output_path), "pass": used_pass}

    except Exception as e:
        return {"status": "error", "video": video_path.name, "error": f"{type(e).__name__}: {e}"}


def select_best_frame(labels, max_check: int = 10):
    """
    Select the best frame (and instance) of an ROI prediction file.

    Kept for callers of the earlier API; it is :func:`stride.stages.roi_selection.select_best_frame_v2`:
    every predicted instance of a frame is evaluated (not only the first), strict 23 nodes take
    precedence over all-16-corners, and the returned frame keeps only the chosen instance.

    Returns:
        Tuple of (frame_index, labeled_frame) or (video_name, None) if none valid
    """
    return select_best_frame_v2(labels, max_check=max_check)
