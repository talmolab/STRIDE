"""
Consensus ROI: recover clips in which no frame shows all maze corners.

A clip that fails both ROI passes (slice and full video) has something - a calibration board, a
box, a hand - hiding a maze corner for the whole clip. Because the maze and the camera do not
move within a recording session, the ROI geometry is the same for every clip of that session, so
it can be borrowed: the consensus ROI is the per-region, per-vertex MEDIAN of every detected ROI
of the session (more robust than copying one clip: it averages out detection jitter).

It is only valid when the table and the camera did not move, so:

1. all donor ROIs must share one region / vertex structure, and the per-vertex spread across
   donors is measured (a large spread means the maze or the camera moved);
2. the consensus is written with a ``# consensus_roi_of: N clips`` provenance line, so borrowed
   ROIs are never used as donors and can always be told apart from detected ones;
3. a QC overlay per recovered clip (the consensus polygons drawn on a real frame of that clip)
   and montage sheets are written for a person to check that the ROI lands on the maze.

Existing ROI files are never overwritten.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import yaml

DRIFT_WARN_PX = 25.0          # static-maze guard: max per-vertex spread across donors (px)
CONSENSUS_TAG = "consensus_roi_of"
PROPAGATED_TAG = "propagated_from"


class ConsensusError(Exception):
    """The consensus ROI cannot be built or is not safe to use; the message says why."""


def is_borrowed_roi(yml_path) -> bool:
    """True when the ROI file was borrowed (consensus / propagated), not detected in the clip."""
    try:
        txt = Path(yml_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return CONSENSUS_TAG in txt or PROPAGATED_TAG in txt


def load_donors(yml_paths: Iterable) -> dict:
    """Donor docs = ROI ymls WITHOUT a consensus / propagated provenance tag."""
    out = {}
    for y in yml_paths:
        y = Path(y)
        if is_borrowed_roi(y):
            continue
        with open(y, encoding="utf-8") as f:
            out[y] = yaml.safe_load(f)
    return out


def consensus(donor_docs: dict):
    """Per-region, per-vertex median across donor docs.

    Returns ``(consensus_doc, spread_max_px, spread_mean_px)``; the spread of a vertex is
    ``hypot(std_x, std_y)`` across donors.
    """
    docs = list(donor_docs.values())
    if not docs:
        raise ConsensusError("no detected ROI to build a consensus from")
    sigs = {tuple((r["name"], len(r["coordinates"])) for r in d["rois"]) for d in docs}
    if len(sigs) != 1:
        raise ConsensusError(f"donor ROI files have {len(sigs)} different region/vertex structures - "
                             f"cannot take a median; inspect the ROI outputs")
    tmpl = copy.deepcopy(docs[0])
    spreads = []
    for ri, roi in enumerate(tmpl["rois"]):
        for vi in range(len(roi["coordinates"])):
            stack = np.array([d["rois"][ri]["coordinates"][vi] for d in docs], float)
            med = np.median(stack, axis=0)
            tmpl["rois"][ri]["coordinates"][vi] = [float(med[0]), float(med[1])]
            spreads.append(np.hypot(*(stack.std(0))))
    return tmpl, float(np.max(spreads)), float(np.mean(spreads))


def consensus_tag(n_donors: int, spread_max: float, spread_mean: float) -> str:
    return (f"{CONSENSUS_TAG}: {n_donors} clips; max per-vertex spread {spread_max:.1f}px, "
            f"mean {spread_mean:.1f}px")


def write_consensus_yml(doc: dict, out_yml, tag: str, qc_dir_name: str = "roi_consensus_qc") -> Path:
    out_yml = Path(out_yml)
    out_yml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yml, "w", encoding="utf-8") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
        f.write(f"\n# {tag}\n# maze static within session; ROI borrowed. QC overlay in {qc_dir_name}/\n")
    return out_yml


def draw_overlay(video, doc: dict, out_png, frac: float = 0.5) -> bool:
    """Draw the ROI polygons on one frame of the clip (at ``frac`` of its length), 1/3 size."""
    import cv2
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frac * n))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        return False
    for roi in doc["rois"]:
        pts = np.array(roi["coordinates"], np.int32).reshape(-1, 1, 2)
        c = roi.get("color", "#ff0000").lstrip("#")
        bgr = (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))
        cv2.polylines(fr, [pts], True, bgr, 3)
        cx, cy = np.array(roi["coordinates"], float).mean(0).astype(int)
        cv2.putText(fr, roi["name"], (int(cx) - 40, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
    h, w = fr.shape[:2]
    small = cv2.resize(fr, (max(1, w // 3), max(1, h // 3)))
    cv2.putText(small, f"CONSENSUS ROI on {Path(video).stem}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    # imencode + write_bytes instead of cv2.imwrite: works with non-ASCII paths on Windows
    ok, buf = cv2.imencode(".png", small)
    if not ok:
        return False
    out_png.write_bytes(buf.tobytes())
    return True


def _imread(path):
    import cv2
    try:
        data = np.frombuffer(Path(path).read_bytes(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def write_montages(qc_dir, stems: list, per: int = 12, cols: int = 3) -> list:
    """Montage sheets (<= ``per`` overlays each) of the QC overlays in ``qc_dir``."""
    import cv2
    qc_dir = Path(qc_dir)
    imgs = [(s, _imread(qc_dir / f"{s}.png")) for s in stems if (qc_dir / f"{s}.png").exists()]
    imgs = [(s, im) for s, im in imgs if im is not None]
    sheets = []
    for si in range(0, len(imgs), per):
        chunk = imgs[si:si + per]
        rows = (len(chunk) + cols - 1) // cols
        cw = max(im.shape[1] for _, im in chunk)
        ch = max(im.shape[0] for _, im in chunk)
        sheet = np.full((rows * ch, cols * cw, 3), 20, np.uint8)
        for i, (_, im) in enumerate(chunk):
            r, c = divmod(i, cols)
            sheet[r * ch:r * ch + im.shape[0], c * cw:c * cw + im.shape[1]] = im
        out = qc_dir / f"MONTAGE_{si // per + 1}.png"
        ok, buf = cv2.imencode(".png", sheet)
        if ok:
            out.write_bytes(buf.tobytes())
            sheets.append(out)
    return sheets


def propagate_consensus(
    missing_videos: list,
    donor_ymls: Iterable,
    out_yml_for: Callable[[Path], Path],
    qc_dir,
    drift_warn: float = DRIFT_WARN_PX,
    on_drift: str = "fail",
    qc_frame: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Write the consensus ROI of one session onto the clips that have none.

    Args:
        missing_videos: videos of the session without an ROI file
        donor_ymls: ROI files of the same session (borrowed ones are ignored)
        out_yml_for: maps a video path to the ROI file to write for it
        qc_dir: folder for the QC overlays, montage sheets and ``consensus_log.txt``
        drift_warn: static-maze guard, max per-vertex spread across donors in px
        on_drift: "fail" = refuse when the guard is exceeded (nothing is written);
                  "warn" = write anyway with a loud warning (a person must check the montage)
        qc_frame: fraction of the clip at which the QC frame is taken

    Returns a dict: ``recovered`` (list of stems), ``n_donors``, ``spread_max``, ``spread_mean``,
    ``static_ok``, ``tag``, ``montages``, ``qc_dir``.

    Raises ConsensusError when there is no donor, the donors disagree in structure, or the
    guard is exceeded with ``on_drift="fail"``.
    """
    missing_videos = [Path(v) for v in missing_videos]
    donors = load_donors(donor_ymls)
    if verbose:
        print(f"{len(donors)} good donor ymls | {len(missing_videos)} missing")
    if not missing_videos:
        return {"recovered": [], "n_donors": len(donors), "spread_max": None, "spread_mean": None,
                "static_ok": True, "tag": "", "montages": [], "qc_dir": str(qc_dir)}
    if not donors:
        raise ConsensusError("no detected ROI in this session - cannot build a consensus ROI")

    cons, spread_max, spread_mean = consensus(donors)
    tag = consensus_tag(len(donors), spread_max, spread_mean)
    static_ok = spread_max <= drift_warn
    if verbose:
        flag = ("  (static-maze OK)" if static_ok else
                "  <-- WARNING: large spread, maze may have MOVED; do NOT trust consensus")
        print(f"consensus ROI: {tag}{flag}")
    if not static_ok and on_drift == "fail":
        raise ConsensusError(
            f"consensus ROI refused: max per-vertex spread across the {len(donors)} detected ROIs is "
            f"{spread_max:.1f} px (> {drift_warn:g} px), so the maze or the camera may have moved "
            f"within the session")

    qc_dir = Path(qc_dir)
    recovered, log = [], []
    for v in missing_videos:
        tyml = Path(out_yml_for(v))
        if tyml.exists():           # never overwrite an existing ROI
            continue
        write_consensus_yml(cons, tyml, tag, qc_dir_name=qc_dir.name)
        drew = draw_overlay(v, cons, qc_dir / f"{v.stem}.png", qc_frame)
        recovered.append(v.stem)
        log.append(f"{v.stem}  QC:{'ok' if drew else 'NO_FRAME'}")
        if verbose:
            print(f"  {v.stem}  <- consensus   QC:{'ok' if drew else 'NO_FRAME'}")

    montages = write_montages(qc_dir, recovered) if recovered else []
    if recovered:
        qc_dir.mkdir(parents=True, exist_ok=True)
        (qc_dir / "consensus_log.txt").write_text(tag + "\n\n" + "\n".join(log) + "\n", encoding="utf-8")
        if verbose:
            for m in montages:
                print(f"QC montage -> {m}")
            print(f"\nrecovered {len(recovered)} clips via consensus ROI. QC -> {qc_dir}")
            print("MANUAL CHECK: open the montage(s) and confirm each consensus ROI lands on the maze.")
    return {"recovered": recovered, "n_donors": len(donors), "spread_max": spread_max,
            "spread_mean": spread_mean, "static_ok": static_ok, "tag": tag,
            "montages": [str(m) for m in montages], "qc_dir": str(qc_dir)}


def propagate_roi_in_folder(video_dir, drift_warn: float = DRIFT_WARN_PX, on_drift: str = "warn",
                            qc_frame: float = 0.5, qc_out=None) -> dict:
    """The lab's per-day procedure on one folder: ROI files sit next to the videos as
    ``<video stem>.rois.yml``; every video of the folder without one gets the folder's consensus.
    """
    from stride.utils.parallel import find_videos
    vid = Path(video_dir)
    vids = find_videos(vid, extensions=(".mp4",), recursive=False)
    yml_of = lambda v: v.with_suffix("").with_suffix(".rois.yml")  # noqa: E731
    missing = [v for v in vids if not yml_of(v).exists()]
    donors = sorted(vid.glob("*.rois.yml"))
    print(f"{len(vids)} videos | ", end="")
    qc = Path(qc_out) if qc_out else (vid / "roi_consensus_qc")
    return propagate_consensus(missing, donors, yml_of, qc, drift_warn=drift_warn, on_drift=on_drift,
                               qc_frame=qc_frame)
