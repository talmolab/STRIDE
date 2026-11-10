"""Camera-specific conventions, as one object. The default is the lab's DJI Osmo Action rig.

Everything that depends on how a camera names its files, or on how the rig is mounted, lives here so the
rest of the front-end is camera-agnostic. To use another camera, build a `CameraProfile` (or pass the
matching `--raw-pattern` / `--crop-rotation` options on the command line).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# name of a clip after the rename: Day{N}_{Mouse}_Trial{T}.<ext>  (what the inference/analysis stages parse)
RENAMED_RE = re.compile(r"(?i)^day\d+_.+_trial\d+\.mp4$")


@dataclass(frozen=True)
class CameraProfile:
    """How raw clips are named and how the rig looks.

    raw_pattern   regex for a raw clip name with the named groups `date` (YYYYMMDD) and `seq`
                  (the camera's running clip number); `time` (hhmmss) is optional.
                  Recording order = the lexical order of the raw names (true for DJI names).
    extensions    video suffixes that count as clips, compared case-insensitively.
    crop_rotation how a whiteboard crop is turned so the number reads upright:
                  "ccw90" (lab rig), "cw90", "180" or "none".
    rail_crop     (y0, y1, x0, x1) of the straight corridor in a frame of `frame_size`, used only for the
                  optional 10-cm rail cross-check of the scale; None = no cross-check.
    """

    name: str = "dji"
    raw_pattern: str = r"^DJI_(?P<date>\d{8})(?P<time>\d{6})_(?P<seq>\d{4})_D\.MP4$"
    extensions: tuple = (".mp4",)
    crop_rotation: str = "ccw90"
    rail_crop: Optional[tuple] = (820, 1320, 850, 2500)
    frame_size: Optional[tuple] = (3840, 2160)
    _re: re.Pattern = field(init=False, repr=False, compare=False, default=None)

    def __post_init__(self):
        rx = re.compile(self.raw_pattern)
        for g in ("date", "seq"):
            if g not in rx.groupindex:
                raise ValueError(f"raw_pattern needs a named group '{g}': {self.raw_pattern}")
        if self.crop_rotation not in ("ccw90", "cw90", "180", "none"):
            raise ValueError(f"crop_rotation must be ccw90 | cw90 | 180 | none, not {self.crop_rotation!r}")
        object.__setattr__(self, "_re", rx)

    # ---- raw names -------------------------------------------------------------------------
    def match(self, name: str):
        return self._re.match(name)

    def is_raw(self, name: str) -> bool:
        return self._re.match(name) is not None

    def date_of(self, name: str) -> Optional[str]:
        m = self._re.match(name)
        return m.group("date") if m else None

    def seq_of(self, name: str) -> str:
        """Short label of a clip in reports: the camera's clip number, else the stem."""
        m = self._re.match(name)
        return m.group("seq") if m else Path(name).stem

    # ---- folders ---------------------------------------------------------------------------
    def is_video(self, p: Path) -> bool:
        return p.is_file() and p.suffix.lower() in self.extensions

    def videos(self, folder: Path) -> list:
        """Every clip directly in `folder`, in recording order (lexical by name, as the lab does)."""
        return sorted(p for p in Path(folder).iterdir() if self.is_video(p))

    def raw_videos(self, folder: Path) -> list:
        return [p for p in self.videos(folder) if self.is_raw(p.name)]

    def renamed_videos(self, folder: Path) -> list:
        return [p for p in self.videos(folder) if RENAMED_RE.match(p.name)]


DJI = CameraProfile()


def profile_from_options(raw_pattern: Optional[str] = None, crop_rotation: Optional[str] = None,
                         extensions: Optional[str] = None, rail_crop: Optional[str] = None) -> CameraProfile:
    """Build a profile from command-line options; anything not given keeps the DJI default."""
    kw = {}
    if raw_pattern:
        kw.update(name="custom", raw_pattern=raw_pattern, rail_crop=None, frame_size=None)
    if crop_rotation:
        kw["crop_rotation"] = crop_rotation
    if extensions:
        kw["extensions"] = tuple(e.lower() if e.startswith(".") else "." + e.lower() for e in extensions.split(","))
    if rail_crop is not None:
        if rail_crop.strip().lower() in ("", "none", "off"):
            kw["rail_crop"] = None
        else:
            vals = tuple(int(v) for v in rail_crop.split(","))
            if len(vals) != 4:
                raise ValueError("--rail-crop needs y0,y1,x0,x1")
            kw.update(rail_crop=vals, frame_size=None)
    return CameraProfile(**kw) if kw else DJI
