"""Tests of the scale calibration (`stride calibrate`, stride.stages.scale_calibration) on drawn clips.

The clips are generated into a temporary folder by tests/mock/make_mock_charuco.py; the ChArUco board is drawn
at a planted px/cm, so every test has a known answer.
    pytest tests/test_scale_calibration.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).parent / "mock"))
import make_mock_charuco as mock  # noqa: E402

from stride.stages.scale_calibration import (BoardSpec, CalibrationError, calibrate_px_per_cm, calibrate_scale,  # noqa: E402
                                             find_videos, limit_clips, make_board, measure_clip, pick_clips, summarize)


@pytest.fixture(scope="session")
def clips_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mock_charuco")
    for name in mock.SCENARIOS:
        mock.make_scenario(name, root)
    return root


def planted(root: Path) -> float:
    return json.load(open(root / "truth.json"))["px_per_cm"]


def quiet(_msg):
    pass


def cli(*args):
    r = subprocess.run([sys.executable, "-m", "stride.stages.scale_calibration", *map(str, args)], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_a_folder_of_day_folders_gives_the_planted_scale(clips_root, tmp_path):
    root = clips_root / "days"
    r = calibrate_scale(root, tmp_path / "cal", workers=3, log=quiet)
    assert abs(r["px_per_cm"] - planted(root)) / planted(root) < 0.005
    assert r["n_clips"] == 6 and r["n_usable"] == 6 and r["spread_pct"] < 1.0       # first clip of each day + 3 more
    assert {Path(c["clip"]).name for c in r["table"]} >= {"Day1_301_Trial1.MP4", "Day2_301_Trial1.MP4", "Day3_301_Trial1.MP4"}
    out = tmp_path / "cal"
    assert float((out / "px_per_cm.txt").read_text()) == r["px_per_cm"]
    rows = list(csv.DictReader(open(out / "calibration_readings.csv")))
    assert len(rows) == 6 and all(x["usable"] == "True" and abs(float(x["px_per_cm"]) - 26.0) < 0.13 for x in rows)
    assert len(list(out.glob("*_checkerboard.png"))) == 6 and len(list(out.glob("Day*.json"))) == 6
    assert "WROTE" in (out / "calibration_summary.txt").read_text() and (out / "calibrate_run_manifest.yaml").exists()


def test_default_output_folder_and_a_flat_folder(clips_root, tmp_path):
    import shutil
    root = tmp_path / "flat"
    shutil.copytree(clips_root / "flat", root)
    r = calibrate_scale(root, log=quiet)
    assert r["out_dir"] == root / "calibration" and (root / "calibration" / "px_per_cm.txt").exists()
    assert abs(r["px_per_cm"] - 23.5) / 23.5 < 0.005 and r["n_clips"] == 5
    assert calibrate_scale(root, log=quiet)["n_clips"] == 5                           # its own output folder holds no clips


def test_an_explicit_list_of_clips(clips_root, tmp_path):
    files = sorted((clips_root / "flat").glob("*.mp4"))[:3]
    r = calibrate_scale(files, tmp_path / "cal", log=quiet)
    assert r["n_clips"] == 3 and abs(r["px_per_cm"] - 23.5) / 23.5 < 0.005
    with pytest.raises(CalibrationError, match="clip not found"):
        calibrate_scale([tmp_path / "nope.mp4"], tmp_path / "cal2", log=quiet)
    with pytest.raises(CalibrationError, match="no video files"):
        calibrate_scale(tmp_path, tmp_path / "cal3", log=quiet)


def test_no_board_in_view_is_a_clear_error(clips_root, tmp_path):
    with pytest.raises(CalibrationError, match="usable ChArUco reading") as e:
        calibrate_scale(clips_root / "no_board", tmp_path / "cal", log=quiet)
    assert "board in view" in str(e.value)
    out = tmp_path / "cal"
    assert not (out / "px_per_cm.txt").exists()
    assert "NOT WRITTEN" in (out / "calibration_summary.txt").read_text()
    assert [r["usable"] for r in csv.DictReader(open(out / "calibration_readings.csv"))] == ["False"] * 3


def test_clips_without_the_board_are_listed_and_the_rest_decides(clips_root, tmp_path):
    r = calibrate_scale(clips_root / "partly", tmp_path / "cal", log=quiet)
    assert r["n_usable"] == 3 and r["n_clips"] == 5 and abs(r["px_per_cm"] - 26.0) / 26.0 < 0.005
    assert sum(1 for c in r["table"] if not c["usable"]) == 2
    assert "no board found in 2 clip(s)" in (tmp_path / "cal" / "calibration_summary.txt").read_text()
    with pytest.raises(CalibrationError, match="need >= 4"):
        calibrate_scale(clips_root / "partly", tmp_path / "cal2", min_readings=4, log=quiet)


def test_readings_that_disagree_are_refused(clips_root, tmp_path):
    with pytest.raises(CalibrationError, match="spread"):
        calibrate_scale(clips_root / "moved", tmp_path / "cal", log=quiet)
    assert not (tmp_path / "cal" / "px_per_cm.txt").exists()
    assert calibrate_scale(clips_root / "moved", tmp_path / "cal2", max_spread_pct=10.0, log=quiet)["px_per_cm"] == pytest.approx(26.5, abs=0.15)


def test_another_board_needs_its_specification(clips_root, tmp_path):
    root = clips_root / "other_board"
    board = BoardSpec(7, 5, 30.0, 22.0, "DICT_4X4_50")
    r = calibrate_scale(root, tmp_path / "cal", board=board, log=quiet)
    assert abs(r["px_per_cm"] - planted(root)) / planted(root) < 0.005
    with pytest.raises(CalibrationError, match="board specification"):
        calibrate_scale(root, tmp_path / "cal2", log=quiet)                           # the default 14x9 board is not there
    for bad, why in ((BoardSpec(dictionary="DICT_NOPE"), "unknown ArUco dictionary"), (BoardSpec(14, 9, 10.0, 15.0), "impossible board")):
        with pytest.raises(CalibrationError, match=why):
            calibrate_scale(root, tmp_path / "cal3", board=bad, log=quiet)


def test_clips_are_found_recursively_and_waste_is_skipped(clips_root):
    groups = find_videos(clips_root / "nested")
    names = [p.name for g in groups for p in g]
    assert sorted(names) == ["take_1.mp4", "take_2.MP4", "take_3.MP4"]                 # not waste/, not _private/
    groups = find_videos(clips_root / "days")
    assert [g[0].parent.name for g in groups] == ["2027-01-01-DAY1", "2027-01-02-DAY2", "2027-01-03-DAY3"]
    assert all(len(g) == 4 for g in groups)                                            # .MP4 and .mp4 both


def test_clip_choice_rule():
    days = [[f"d{d}_{i:02d}" for i in range(20)] for d in range(8)]
    assert pick_clips(days) == [d[0] for d in days]                                    # one per day when there are >= 6 days
    two = [[f"a{i:02d}" for i in range(20)], [f"b{i:02d}" for i in range(20)]]
    assert pick_clips(two) == ["a00", "b00", "a10", "b10", "a17", "b17"]               # then the 11th, 18th clip of each
    short = [["a0", "a1", "a2"]]
    assert pick_clips(short) == ["a0", "a2", "a1"] and pick_clips([]) == []


def test_one_clip_reading_and_the_summary_rule(clips_root, tmp_path):
    clip = sorted((clips_root / "flat").glob("*.mp4"))[0]
    m = measure_clip(clip, tmp_path)
    assert set(m["charuco"]) == {"t", "n", "nn_px", "scale_h_px", "reproj_px", "aniso", "ux", "uy", "pxcm_nn", "pxcm_H"}
    assert m["charuco"]["n"] == 104 and abs(m["charuco"]["pxcm_H"] - 23.5) < 0.1 and m["rail_best"] is None
    assert (tmp_path / f"{clip.stem}.json").exists() and (tmp_path / f"{clip.stem}_checkerboard.png").exists()

    def rd(v):
        return {"stem": f"c{v}", "charuco": {"pxcm_H": v, "n": 60, "aniso": 0.01, "reproj_px": 0.2}}
    s = summarize([rd(26.04), rd(26.05), rd(26.02), rd(26.00), rd(26.02)])
    assert s["value"] == 26.02 and s["n_usable"] == 5 and s["spread_pct"] == pytest.approx(0.192, abs=0.01)   # median, 2 decimals
    assert summarize([rd(26.0), rd(26.1)])["value"] is None                            # fewer than 3 readings
    assert summarize([rd(26.0), rd(26.1), rd(26.4)])["value"] is None                  # spread >= 1 %


def test_command_line(clips_root, tmp_path):
    code, out = cli(clips_root / "days", "--out", tmp_path / "cal", "--workers", "2")
    assert code == 0 and "px_per_cm = 26.00" in out and (tmp_path / "cal" / "px_per_cm.txt").read_text().strip() == "26.00"
    files = sorted((clips_root / "flat").glob("*.mp4"))[:3]
    code, out = cli(*files, "--out", tmp_path / "cal_files")
    assert code == 0 and "px_per_cm = 23.50" in out
    code, out = cli(clips_root / "other_board", "--out", tmp_path / "cal_b", "--board-squares", "7x5", "--square-mm", "30",
                    "--marker-mm", "22", "--dictionary", "DICT_4X4_50")
    assert code == 0 and "px_per_cm = 18.00" in out
    for args, why in (((clips_root / "no_board",), "usable ChArUco reading"), ((clips_root / "moved",), "spread"),
                      ((tmp_path / "missing",), "not found"), ((clips_root / "days", "--board-squares", "14"), "14x9"),
                      ((clips_root / "days", "--dictionary", "DICT_NOPE"), "unknown ArUco dictionary"),
                      ((clips_root / "days", "--rail-crop", "1,2"), "y0,y1,x0,x1")):
        code, out = cli(*args, "--out", tmp_path / "cal_err")
        assert code == 1 and "ERROR:" in out and why in out, (args, out)


def test_the_module_stands_alone():
    """It must be usable without the lab intake layer and without the rest of the pipeline's stages."""
    code = ("import sys; import stride.stages.scale_calibration as m; "
            "bad = [k for k in sys.modules if k.startswith(('stride_lab', 'stride.prepare')) or k in "
            "('stride.stages.gait_extraction', 'stride.stages.decision_analysis', 'stride.cli')]; "
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    src = Path(sys.modules["stride.stages.scale_calibration"].__file__).read_text()
    assert "stride_lab" not in src and "from ." not in src and "import sleap" not in src


def test_config_driven_stage_is_opt_in(clips_root, tmp_path):
    """python -m stride.stages.run_calibration: only batches with a `calibrate:` entry are measured."""
    from stride.stages.run_calibration import read_batch_px_per_cm

    def run(cfg_text, *extra):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(cfg_text)
        r = subprocess.run([sys.executable, "-m", "stride.stages.run_calibration", "--config", str(cfg), "--output", str(tmp_path / "out"), *extra],
                           capture_output=True, text=True)
        return r.returncode, r.stdout + r.stderr

    code, out = run(f"batches:\n  plain:\n    video_dir: {clips_root / 'no_board'}\n  with_board:\n    video_dir: {clips_root / 'days'}\n"
                    "    calibrate: true\n    px_per_cm_override: 30.0\n")
    assert code == 0 and "[plain] no `calibrate:` entry" in out and "WARNING: the batch pins px_per_cm_override = 30.0" in out
    assert read_batch_px_per_cm(tmp_path / "out", "with_board") == pytest.approx(26.0, abs=0.13)
    assert read_batch_px_per_cm(tmp_path / "out", "plain") is None
    code, out = run(f"batches:\n  small:\n    video_dir: {clips_root / 'other_board'}\n    calibrate:\n      board_squares: 7x5\n"
                    "      square_mm: 30\n      marker_mm: 22\n      dictionary: DICT_4X4_50\n")
    assert code == 0 and read_batch_px_per_cm(tmp_path / "out", "small") == pytest.approx(18.0, abs=0.09)
    code, out = run(f"batches:\n  bad:\n    video_dir: {clips_root / 'no_board'}\n    calibrate: true\n")
    assert code == 1 and "ERROR:" in out and "FAILED for: bad" in out
    code, out = run(f"batches:\n  x:\n    video_dir: {clips_root / 'days'}\n    calibrate: {{clipz: 3}}\n")
    assert code == 1 and "unknown key" in out
    code, out = run("batches: {}\n", "--batch", "nope")
    assert code == 1 and "is not in" in out


def test_public_entry_point_returns_ok_and_reason_and_never_raises(clips_root, tmp_path):
    r = calibrate_px_per_cm(clips_root / "days", tmp_path / "cal", log=quiet)
    assert {"px_per_cm", "spread_pct", "n_clips_used", "per_clip", "method", "ok", "reason"} <= set(r)
    assert r["ok"] and r["reason"] == "" and r["method"] == "charuco_homography_median"
    assert abs(r["px_per_cm"] - 26.0) / 26.0 < 0.005 and r["n_clips_used"] == 6 and len(r["per_clip"]) == 6
    assert {"clip", "px_per_cm", "usable", "corners"} <= set(r["per_clip"][0])
    bad = calibrate_px_per_cm(clips_root / "no_board", tmp_path / "cal_no", log=quiet)
    assert not bad["ok"] and bad["px_per_cm"] is None and "usable ChArUco reading" in bad["reason"] and len(bad["per_clip"]) == 3
    for r in (calibrate_px_per_cm(tmp_path / "nowhere", log=quiet), calibrate_px_per_cm([], log=quiet),
              calibrate_px_per_cm(clips_root / "days", tmp_path / "x", dictionary="DICT_NOPE", log=quiet),
              calibrate_px_per_cm(clips_root / "days", tmp_path / "x", board=42, log=quiet)):
        assert not r["ok"] and r["reason"] and r["px_per_cm"] is None
    assert not (tmp_path / "x").exists()                                      # refused before anything was written
    moved = calibrate_px_per_cm(clips_root / "moved", tmp_path / "cal_moved", log=quiet)
    assert not moved["ok"] and "spread" in moved["reason"] and moved["spread_pct"] > 1.0 and moved["n_clips_used"] == 4


def test_every_setting_is_a_plain_keyword_so_a_yaml_block_can_drive_it(clips_root, tmp_path):
    import yaml
    block = yaml.safe_load("""
    squares_x: 7
    squares_y: 5
    square_mm: 30
    marker_mm: 22
    dictionary: DICT_4X4_50
    max_clips: 2
    min_readings: 2
    frames_per_clip: 4
    workers: 2
    """)
    r = calibrate_px_per_cm(str(clips_root / "other_board"), str(tmp_path / "cal"), log=quiet, **block)
    assert r["ok"] and r["n_clips"] == 2 and abs(r["px_per_cm"] - 18.0) / 18.0 < 0.005 and "7x5" in r["board"]
    assert make_board({"squares": "7x5", "square_mm": 30}) == BoardSpec(7, 5, 30.0, 15.0, "DICT_5X5_250")
    assert make_board("14x9") == make_board(None) == make_board(BoardSpec()) == BoardSpec()
    for bad in ({"squares": "14"}, {"colour": "red"}, {"square_mm": "wide"}):
        with pytest.raises(CalibrationError):
            make_board(bad)
    assert limit_clips(list(range(12)), 5) == [0, 3, 6, 8, 11] and limit_clips([1, 2, 3], None) == [1, 2, 3]
    r = calibrate_px_per_cm(clips_root / "days", tmp_path / "cal3", max_clips=3, log=quiet)
    assert r["ok"] and r["n_clips"] == 3


def test_paths_with_spaces_and_non_ascii_names(clips_root, tmp_path):
    import shutil
    root = tmp_path / "my cohort (été) 2027" / "day 1"
    root.mkdir(parents=True)
    for i, c in enumerate(sorted((clips_root / "flat").glob("*.mp4"))[:3]):
        shutil.copy(c, root / f"clip number {i + 1}.MP4")
    r = calibrate_px_per_cm(root.parent, log=quiet)
    assert r["ok"] and r["out_dir"] == root.parent / "calibration" and abs(r["px_per_cm"] - 23.5) / 23.5 < 0.005
    assert len(list(r["out_dir"].glob("*_checkerboard.png"))) == 3 and (r["out_dir"] / "px_per_cm.txt").read_text(encoding="utf-8").strip() == "23.50"
    code, out = cli(root.parent, "--out", tmp_path / "out with space")
    assert code == 0 and "px_per_cm = 23.50" in out


def test_a_refused_run_does_not_leave_an_older_value(clips_root, tmp_path):
    out = tmp_path / "cal"
    assert calibrate_px_per_cm(clips_root / "days", out, log=quiet)["ok"] and (out / "px_per_cm.txt").exists()
    assert not calibrate_px_per_cm(clips_root / "no_board", out, log=quiet)["ok"] and not (out / "px_per_cm.txt").exists()


def test_optional_rail_cross_check_is_reported_not_used(clips_root, tmp_path):
    """The drawn corridor is 140 px wide; the cross-check reports it and the ChArUco value is unchanged."""
    clip = tmp_path / "long clip.mp4"                                         # the cross-check reads frames at 1, 3, 6, 9 s
    import cv2
    frame = mock.scene(23.5)[0]
    cv2.rectangle(frame, (150, 640), (1450, 780), (205, 205, 205), -1)        # a bright corridor, 140 px between its rails
    mock.write_clip(clip, frame, n_frames=45)
    m = measure_clip(clip, tmp_path, rail_crop=(560, 860, 100, 1500))
    assert m["rail_best"] is not None and abs(m["rail_best"]["width_px"] - 140) <= 3 and (tmp_path / f"{clip.stem}_rail.png").exists()
    assert measure_clip(clip, None, rail_crop=(560, 860, 100, 1500), rail_frame_size=(3840, 2160))["rail_best"] is None   # other frame size: skipped
    clips = [clip] + sorted((clips_root / "flat").glob("*.mp4"))[:2]
    a = calibrate_px_per_cm(clips, tmp_path / "a", log=quiet)
    b = calibrate_px_per_cm(clips, tmp_path / "b", rail_crop=(560, 860, 100, 1500), log=quiet)
    assert a["ok"] and a["px_per_cm"] == b["px_per_cm"]
    assert "rail cross-check" in (tmp_path / "b" / "calibration_summary.txt").read_text(encoding="utf-8")
