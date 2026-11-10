"""Lab intake (internal): from the camera card to named clips.

    sort  ->  boards (plan + whiteboard crops)  ->  [a person signs BOARD_VERIFICATION.md]  ->  rename  ->  calibrate

The lab's whiteboard-anchored rename procedure as a package. It is NOT part of the public STRIDE pipeline; it
calls the public pipeline (scale calibration: `stride.stages.scale_calibration`). No lab path lives in this code:
folders, camera conventions and the board come from the command line or from a config file (see INTAKE.md).
"""
from .camera import CameraProfile, DJI
from .metadata import read_trials_full, newest_trials_full, day_blocks
from .sort import sort_days
from .plan import build_rename_plan
from .boards import find_whiteboards, make_montage, write_verification_template, fill_verification, signed_off
from .apply import apply_rename, move_to_waste
from .orchestrate import check, auto, boards_days, rename_days, calibrate

__all__ = [
    "CameraProfile", "DJI", "read_trials_full", "newest_trials_full", "day_blocks", "sort_days", "build_rename_plan",
    "find_whiteboards", "make_montage", "write_verification_template", "fill_verification", "signed_off", "apply_rename",
    "move_to_waste", "check", "auto", "boards_days", "rename_days", "calibrate",
]
