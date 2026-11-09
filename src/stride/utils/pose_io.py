"""
Pose-file loading shared by the analysis stages.
"""

from pathlib import Path
from typing import Optional


def load_pose_labels(pose_path, video_path: Optional[Path] = None):
    """Load a SLEAP ``.slp`` pose file.

    A ``.slp`` remembers the path of the video it was made from. When that path points to
    another machine or mount that this user cannot read, ``sleap_io.load_file`` raises
    (``PermissionError`` / ``FileNotFoundError``) and the clip used to drop out of the
    analysis. In that case only, the file is loaded without opening the embedded video and
    pointed at the on-disk video (``video_path``). Files that load normally take the
    unchanged path, so their results are unaffected.
    """
    import sleap_io as sio

    try:
        return sio.load_file(str(pose_path))
    except OSError:  # PermissionError / FileNotFoundError on the embedded video path
        labels = sio.load_slp(str(pose_path), open_videos=False)
        if video_path is not None:
            vp = Path(video_path)
            try:
                on_disk = vp.exists()
            except OSError:
                on_disk = False
            if on_disk:
                for v in labels.videos:
                    try:
                        v.replace_filename(str(vp), open=True)
                    except Exception:
                        pass
        return labels
