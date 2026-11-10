"""
Distortion checking using charuco board analysis.

Adapted from: 2025-09-02-Tmaze_undistortion_pipe/02.test_distortion_need.py

This module detects charuco boards in video frames and measures lens distortion
to determine if undistortion is necessary before processing.
"""

from pathlib import Path
from typing import Optional
import cv2
import numpy as np


def detect_charuco_board(
    image: np.ndarray,
    squares_x: int = 14,
    squares_y: int = 9,
    square_length: float = 0.02,
    marker_length: float = 0.015,
    dictionary: int = cv2.aruco.DICT_5X5_250,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Detect charuco board in an image.

    Args:
        image: Input image (grayscale or BGR)
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Length of square side (meters)
        marker_length: Length of marker side (meters)
        dictionary: ArUco dictionary type

    Returns:
        Tuple of (corners, ids, marker_corners)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create charuco board
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )

    # Create CharucoDetector
    charuco_params = cv2.aruco.CharucoParameters()
    detector_params = cv2.aruco.DetectorParameters()
    charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, detector_params)

    # Detect charuco corners
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    if charuco_corners is not None and len(charuco_corners) > 0:
        return charuco_corners, charuco_ids, marker_corners

    return None, None, None


def compute_line_straightness_score(
    corners: np.ndarray,
    ids: np.ndarray,
    squares_x: int,
    squares_y: int,
) -> float:
    """
    Measure how straight the board edges are.

    Distortion causes edge bowing, so higher RMS deviation indicates more distortion.

    Returns:
        RMS deviation from straight lines (pixels) - lower is better
    """
    if corners is None or len(corners) < 4:
        return np.inf

    corners = corners.reshape(-1, 2)
    ids = ids.flatten()

    # Organize corners by row and column
    corner_dict = {}
    for corner, corner_id in zip(corners, ids):
        row = corner_id // (squares_x - 1)
        col = corner_id % (squares_x - 1)
        corner_dict[(row, col)] = corner

    deviations = []

    # Check horizontal lines (rows)
    for row in range(squares_y - 1):
        row_corners = []
        for col in range(squares_x - 1):
            if (row, col) in corner_dict:
                row_corners.append(corner_dict[(row, col)])

        if len(row_corners) >= 3:
            row_corners = np.array(row_corners)
            vx, vy, x0, y0 = cv2.fitLine(
                row_corners.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
            )
            vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

            for pt in row_corners:
                dist = abs(vy * pt[0] - vx * pt[1] + (vx * y0 - vy * x0))
                deviations.append(dist)

    # Check vertical lines (columns)
    for col in range(squares_x - 1):
        col_corners = []
        for row in range(squares_y - 1):
            if (row, col) in corner_dict:
                col_corners.append(corner_dict[(row, col)])

        if len(col_corners) >= 3:
            col_corners = np.array(col_corners)
            vx, vy, x0, y0 = cv2.fitLine(
                col_corners.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
            )
            vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]

            for pt in col_corners:
                dist = abs(vy * pt[0] - vx * pt[1] + (vx * y0 - vy * x0))
                deviations.append(dist)

    if len(deviations) == 0:
        return np.inf

    return np.sqrt(np.mean(np.array(deviations) ** 2))


def compute_corner_spacing_uniformity(
    corners: np.ndarray,
    ids: np.ndarray,
    squares_x: int,
) -> float:
    """
    Measure uniformity of corner spacing.

    Distortion causes non-uniform spacing.

    Returns:
        Coefficient of variation of distances - lower is better (< 0.05 is good)
    """
    if corners is None or len(corners) < 2:
        return np.inf

    corners = corners.reshape(-1, 2)
    ids = ids.flatten()

    corner_dict = {}
    for corner, corner_id in zip(corners, ids):
        row = corner_id // (squares_x - 1)
        col = corner_id % (squares_x - 1)
        corner_dict[(row, col)] = corner

    distances = []

    for row, col in corner_dict.keys():
        # Horizontal distance
        if (row, col + 1) in corner_dict:
            dist = np.linalg.norm(corner_dict[(row, col)] - corner_dict[(row, col + 1)])
            distances.append(dist)
        # Vertical distance
        if (row + 1, col) in corner_dict:
            dist = np.linalg.norm(corner_dict[(row, col)] - corner_dict[(row + 1, col)])
            distances.append(dist)

    if len(distances) == 0:
        return np.inf

    distances = np.array(distances)
    return np.std(distances) / np.mean(distances)


def compute_reprojection_error(
    corners: np.ndarray,
    ids: np.ndarray,
    image_size: tuple[int, int],
    squares_x: int = 14,
    squares_y: int = 9,
    square_length: float = 0.02,
    marker_length: float = 0.015,
    dictionary: int = cv2.aruco.DICT_5X5_250,
) -> float:
    """
    Compute reprojection error after initial calibration.

    High error indicates significant distortion.

    Returns:
        RMS reprojection error in pixels
    """
    if corners is None or len(corners) < 4:
        return np.inf

    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict
    )

    obj_points = board.getChessboardCorners()[ids.flatten()]

    w, h = image_size
    K_init = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_init = np.zeros(5, dtype=np.float64)

    ret, rvec, tvec = cv2.solvePnP(
        obj_points.astype(np.float32),
        corners.astype(np.float32),
        K_init,
        dist_init,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    reprojected, _ = cv2.projectPoints(obj_points, rvec, tvec, K_init, dist_init)
    reprojected = reprojected.reshape(-1, 2)
    corners_2d = corners.reshape(-1, 2)

    errors = np.linalg.norm(corners_2d - reprojected, axis=1)
    return np.sqrt(np.mean(errors ** 2))


def test_video_distortion(
    video_path: Path,
    num_frames: int = 10,
    squares_x: int = 14,
    squares_y: int = 9,
    square_length: float = 0.02,
    marker_length: float = 0.015,
    line_threshold: float = 2.0,
    spacing_threshold: float = 0.05,
    reproj_threshold: float = 5.0,
) -> dict:
    """
    Test if a video needs undistortion by analyzing charuco board.

    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length: Square side length in meters
        marker_length: Marker side length in meters
        line_threshold: Max acceptable line deviation (pixels)
        spacing_threshold: Max acceptable spacing CV
        reproj_threshold: Max acceptable reprojection error (pixels)

    Returns:
        Dictionary with distortion metrics and recommendation
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    line_scores = []
    spacing_scores = []
    reproj_errors = []

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        corners, ids, _ = detect_charuco_board(
            frame, squares_x, squares_y, square_length, marker_length
        )

        if corners is not None:
            line_score = compute_line_straightness_score(corners, ids, squares_x, squares_y)
            spacing_score = compute_corner_spacing_uniformity(corners, ids, squares_x)
            reproj_error = compute_reprojection_error(
                corners, ids, (width, height),
                squares_x, squares_y, square_length, marker_length
            )

            line_scores.append(line_score)
            spacing_scores.append(spacing_score)
            reproj_errors.append(reproj_error)

    cap.release()

    if len(line_scores) == 0:
        return {
            "video": video_path.name,
            "line_straightness": np.inf,
            "spacing_uniformity": np.inf,
            "reprojection_error": np.inf,
            "needs_undistortion": None,
            "frames_detected": 0,
        }

    avg_line = np.mean(line_scores)
    avg_spacing = np.mean(spacing_scores)
    avg_reproj = np.mean(reproj_errors)

    needs_undistortion = (
        avg_line > line_threshold or
        avg_spacing > spacing_threshold or
        avg_reproj > reproj_threshold
    )

    return {
        "video": video_path.name,
        "line_straightness": avg_line,
        "spacing_uniformity": avg_spacing,
        "reprojection_error": avg_reproj,
        "needs_undistortion": needs_undistortion,
        "frames_detected": len(line_scores),
    }


def check_distortion_batch(
    input_path: Path,
    num_frames: int = 10,
    squares_x: int = 14,
    squares_y: int = 9,
    output_file: Optional[Path] = None,
) -> list[dict]:
    """
    Check distortion for a video or directory of videos.

    Args:
        input_path: Path to video file or directory
        num_frames: Number of frames to sample per video
        squares_x: Charuco board squares in X
        squares_y: Charuco board squares in Y
        output_file: Optional path to save results (CSV or JSON)

    Returns:
        List of result dictionaries
    """
    # Recursive, case-insensitive search (per-day subfolders, .mp4 / .MP4 / .avi)
    from stride.utils.parallel import find_videos
    videos = find_videos(Path(input_path), extensions=(".mp4", ".avi"))

    results = []
    for video_path in sorted(videos):
        try:
            result = test_video_distortion(
                video_path,
                num_frames=num_frames,
                squares_x=squares_x,
                squares_y=squares_y,
            )
            results.append(result)
        except Exception as e:
            results.append({
                "video": video_path.name,
                "error": str(e),
                "needs_undistortion": None,
            })

    # Save results if output file specified
    if output_file:
        import json
        import csv

        if output_file.suffix == ".json":
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
        else:  # CSV
            if results:
                with open(output_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)

    return results
