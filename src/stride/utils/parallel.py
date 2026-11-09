"""
Parallel processing utilities.

Provides a generic framework for batch processing videos.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Any, Optional
import logging


def run_parallel(
    items: list[Any],
    worker_fn: Callable[[Any], dict],
    n_workers: int = 4,
    desc: str = "Processing",
    logger: Optional[logging.Logger] = None,
) -> list[dict]:
    """
    Run a worker function on items in parallel.

    Args:
        items: List of items to process
        worker_fn: Function that takes an item and returns a dict with 'status' key
        n_workers: Number of parallel workers
        desc: Description for progress logging
        logger: Optional logger instance

    Returns:
        List of result dictionaries from worker_fn
    """
    log = logger or logging.getLogger(__name__)
    total = len(items)

    if total == 0:
        log.warning(f"{desc}: No items to process")
        return []

    log.info(f"{desc}: {total} items with {n_workers} workers")

    results = []
    n_done = 0
    counts = {"ok": 0, "error": 0, "skip": 0}

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker_fn, item): item for item in items}

        for future in as_completed(futures):
            n_done += 1
            item = futures[future]

            try:
                result = future.result()
                results.append(result)

                status = result.get("status", "ok")
                if status == "ok":
                    counts["ok"] += 1
                elif status.startswith("skip"):
                    counts["skip"] += 1
                else:
                    counts["error"] += 1

            except Exception as e:
                log.error(f"Exception processing {item}: {e}")
                counts["error"] += 1
                results.append({"status": "error", "item": item, "error": str(e)})

            # Progress logging
            if n_done % 50 == 0 or n_done == total:
                log.info(
                    f"{desc}: {n_done}/{total} "
                    f"(ok={counts['ok']}, skip={counts['skip']}, error={counts['error']})"
                )

    log.info(
        f"{desc} complete: ok={counts['ok']}, skip={counts['skip']}, error={counts['error']}"
    )

    return results


def find_videos(
    video_dir: Path,
    extensions: tuple[str, ...] = (".mp4", ".avi"),
    pattern: Optional[str] = None,
    recursive: bool = True,
) -> list[Path]:
    """
    Find video files under a directory.

    The search is recursive (per-day subfolders are found) and the extension match is
    case-insensitive (``.mp4`` and ``.MP4`` are both found). macOS resource-fork files
    (``._name.mp4``) are ignored. A single video file may be passed instead of a directory.

    Args:
        video_dir: Directory to search (or one video file)
        extensions: Video file extensions, lower case
        pattern: Optional glob pattern (e.g., "Day*_*.mp4"); matched as given, in every subfolder
        recursive: Search subfolders too (default)

    Returns:
        Sorted list of video paths
    """
    video_dir = Path(video_dir)
    if video_dir.is_file():
        return [video_dir]

    if pattern:
        found = video_dir.rglob(pattern) if recursive else video_dir.glob(pattern)
        return sorted({p for p in found if not p.name.startswith("._")})

    exts = {e.lower() for e in extensions}
    found = video_dir.rglob("*") if recursive else video_dir.glob("*")
    return sorted({p for p in found
                   if p.suffix.lower() in exts and not p.name.startswith("._")})
