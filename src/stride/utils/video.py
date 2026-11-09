"""
Video utility functions.

Provides robust video metadata extraction and frame counting.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import cv2


def read_fps(video_path: Path, labels_video=None, default: float = 120.0) -> float:
    """
    Read FPS from video with multiple fallback methods.

    Args:
        video_path: Path to video file
        labels_video: Optional SLEAP video object with metadata
        default: Default FPS if detection fails

    Returns:
        Video FPS (frames per second)
    """
    # Try SLEAP labels metadata first
    if labels_video is not None:
        for attr in ("fps", "frame_rate", "frame_rate_hz", "frameRate"):
            if hasattr(labels_video, attr):
                try:
                    v = float(getattr(labels_video, attr))
                    if np.isfinite(v) and v > 0:
                        return v
                except Exception:
                    pass

        # Check metadata dict
        meta = getattr(labels_video, "metadata", None) or getattr(labels_video, "info", None)
        if isinstance(meta, dict):
            for k in ("fps", "frame_rate", "frameRate"):
                if k in meta:
                    try:
                        v = float(meta[k])
                        if np.isfinite(v) and v > 0:
                            return v
                    except Exception:
                        pass

    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path))
        v = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if np.isfinite(v) and v > 0:
            return float(v)
    except Exception:
        pass

    return default


def get_frame_count(video_path: Path) -> int:
    """
    Get frame count from a video file.

    Tries SLEAP .slp file first, falls back to OpenCV.

    Args:
        video_path: Path to video file

    Returns:
        Number of frames in video (0 if unable to determine)
    """
    # Try associated .slp file
    slp_path = video_path.with_suffix(".slp")
    if slp_path.exists():
        try:
            import sleap_io as sio
            labels = sio.load_file(str(slp_path))
            video = labels.videos[0]

            if hasattr(video, "frames") and video.frames is not None:
                return int(video.frames)
            if hasattr(video, "shape") and len(video.shape) > 0:
                return int(video.shape[0])
            try:
                return int(len(video))
            except Exception:
                pass
        except Exception:
            pass

    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(str(video_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if n > 0:
            return n
    except Exception:
        pass

    return 0


def count_total_frames(
    video_dir: Path,
    extensions: tuple[str, ...] = (".mp4", ".avi"),
    verbose: bool = True,
) -> tuple[int, int]:
    """
    Count total frames across all videos in a directory.

    Args:
        video_dir: Directory containing video files
        extensions: Video file extensions to include
        verbose: Print progress

    Returns:
        Tuple of (total_frames, num_videos)
    """
    total = 0
    n_videos = 0
    zeros = []

    # Find all video files (recursive, case-insensitive: per-day subfolders, .mp4 / .MP4)
    from stride.utils.parallel import find_videos
    video_files = find_videos(Path(video_dir), extensions=extensions)

    for video_path in sorted(video_files):
        n = get_frame_count(video_path)
        total += n
        n_videos += 1

        if n == 0:
            zeros.append(str(video_path))

        if verbose and n_videos % 100 == 0:
            print(f"...processed {n_videos} videos, running total = {total:,} frames")

    if verbose:
        print(f"Videos scanned: {n_videos}")
        print(f"Total frames: {total:,}")
        if zeros:
            print(f"Warning: {len(zeros)} video(s) returned 0 frames")

    return total, n_videos


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """
    Get video dimensions (width, height).

    Args:
        video_path: Path to video file

    Returns:
        Tuple of (width, height)
    """
    cap = cv2.VideoCapture(str(video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def read_frame(video_path: Path, frame_idx: int = 0) -> Optional[np.ndarray]:
    """
    Read a single frame from a video.

    Args:
        video_path: Path to video file
        frame_idx: Frame index to read

    Returns:
        Frame as numpy array (BGR) or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None
