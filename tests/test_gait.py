"""Tests of `stride gait` (the gait-only pipeline) on synthetic runway clips.

No animal video, no model, no GPU, no maze ROI: a kinematic mouse walks a straight line
(tests/mock/make_mock_gait.py). What is asserted:

  * px/cm given: every table written, planted stride values recovered, metadata columns carried;
  * no metadata: same numbers, id columns empty;
  * px/cm from a ChArUco board drawn into the videos; no scale at all -> refusal with the reason;
  * keypoint mapping for a skeleton with other node names (same numbers as the standard names);
    a missing required keypoint -> refusal naming it and listing the pose file's nodes;
    one landmark per hind paw; a skeleton without a tail tip (tail-tip columns empty, rest equal);
  * config file + command-line override, steps switched off, `stride init-config gait`;
  * pose files in another folder / with another suffix; videos without a pose file are left out;
  * a second run does nothing; the pipeline runs with the `spawn` start method;
  * a vertical-walking clip gives a warning and a completed run;
  * on the T-maze mock `core` scenario the tables equal those of the existing path.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
STRIDE = [PY, "-m", "stride.cli"]
TABLES = ["gait_per_stride.csv", "gait_per_step.csv", "gait_per_video.csv", "gait_per_stride_filtered.csv",
          "pose_keypoint_confidence_by_video.csv"]
ID_COLS = ["video_path", "stem", "day", "mouse_id", "sex", "genotype"]
TAILTIP_COLS = ["tailtip_lat_disp_norm", "tailtip_lat_phase_pct"]


def run(cmd, cwd=None, timeout=900):
    env = dict(os.environ, PYTHONWARNINGS="ignore", MPLBACKEND="Agg", COLUMNS="200", NO_COLOR="1")
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + "\n" + p.stderr


@pytest.fixture(scope="session")
def mock(tmp_path_factory) -> dict:
    """name -> scenario.json of the gait mock (generated once, or reused via STRIDE_MOCK_GAIT_DATA)."""
    reuse = os.environ.get("STRIDE_MOCK_GAIT_DATA")
    if reuse and (Path(reuse) / "runway" / "scenario.json").exists():
        root = Path(reuse).resolve()
    else:
        root = tmp_path_factory.mktemp("stride_mock_gait")
        code, out = run([PY, REPO / "tests" / "mock" / "make_mock_gait.py", "--out", root], timeout=1800)
        assert code == 0, out[-2000:]
    return {p.parent.name: json.loads(p.read_text()) for p in root.glob("*/scenario.json")}


@pytest.fixture(scope="session")
def runway_out(mock, tmp_path_factory) -> Path:
    """`stride gait` on the runway scenario with the scale given and the metadata file."""
    out = tmp_path_factory.mktemp("gait_runway") / "out"
    code, text = run(STRIDE + ["gait", mock["runway"]["video_dir"], "--px-per-cm", 24, "-m", mock["runway"]["metadata_csv"],
                               "-o", out, "--workers", 2])
    assert code == 0, text[-3000:]
    return out


def numeric_equal(a: pd.DataFrame, b: pd.DataFrame, keys: list, skip=()) -> list:
    """Columns (present in both, not keys / skip) whose values differ after joining on `keys`."""
    m = a.merge(b, on=keys, how="outer", suffixes=("__a", "__b"), indicator=True)
    assert (m["_merge"] == "both").all(), m["_merge"].value_counts().to_dict()
    bad = []
    for c in [c for c in a.columns if c in b.columns and c not in keys and c not in skip]:
        x, y = m[c + "__a"], m[c + "__b"]
        xn, yn = pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")
        if xn.notna().sum() == x.notna().sum() and yn.notna().sum() == y.notna().sum():
            same = (xn.isna() & yn.isna()) | (xn.astype(float) == yn.astype(float))
        else:
            same = (x.isna() & y.isna()) | (x.astype(str) == y.astype(str))
        if not same.all():
            bad.append(c)
    return bad


# --------------------------------------------------------------------------------------------
# the supported cases
# --------------------------------------------------------------------------------------------
def test_gait_help_and_init_config(tmp_path):
    code, text = run(STRIDE + ["gait", "--help"])
    assert code == 0 and "STRIDE Gait" in text and "--px-per-cm" in text and "--keypoint" in text
    cfg = tmp_path / "my gait.yaml"
    code, text = run(STRIDE + ["init-config", "gait", "-o", cfg])
    assert code == 0 and cfg.exists()
    from stride.stages.gait_pipeline import DEFAULTS, STEPS, load_settings
    d = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert set(d) <= set(DEFAULTS) and set(d["steps"]) == set(STEPS) and all(d["steps"].values())
    assert load_settings(cfg)["scale"]["px_per_cm"] is None
    code, text = run(STRIDE + ["init-config", "gait", "-o", cfg])           # no silent overwrite
    assert code == 2 and "already exists" in text
    (tmp_path / "bad.yaml").write_text("videos: x\nouptut: y\n", encoding="utf-8")
    code, text = run(STRIDE + ["gait", "--config", tmp_path / "bad.yaml"])
    assert code == 2 and "unknown setting" in text and "ouptut" in text
    (tmp_path / "bad2.yaml").write_text("videos: x\noutput: y\nsteps: {decisions: true}\n", encoding="utf-8")
    code, text = run(STRIDE + ["gait", "--config", tmp_path / "bad2.yaml"])
    assert code == 2 and "unknown step" in text


def test_gait_px_per_cm_given(mock, runway_out):
    g = runway_out / "gait"
    for name in TABLES:
        assert (g / name).exists(), name
    for name in ("SUMMARY.md", "gait_run.log", "stride_gait_run_manifest.yaml", "gait_state.json"):
        assert (runway_out / name).exists(), name
    assert not (runway_out / "decisions").exists() and not (runway_out / "roi").exists() and not (runway_out / "stats").exists()
    summary = (runway_out / "SUMMARY.md").read_text(encoding="utf-8")
    assert "gait only" in summary and "no trial phase and no maze region" in summary and "given by the user" in summary
    man = yaml.safe_load((runway_out / "stride_gait_run_manifest.yaml").read_text(encoding="utf-8"))
    assert man["params"]["scale"]["px_per_cm"] == 24 and man["results"]["ok"] is True

    s = pd.read_csv(g / "gait_per_stride.csv")
    truth = pd.read_csv(mock["runway"]["truth_csv"]).set_index("stem")
    assert set(s["stem"]) == set(truth.index) and len(s) >= 6 * 5
    assert np.allclose(1.0 / s["px_per_cm"], 24.0) and (s["fps"] > 119).all()
    for stem, grp in s.groupby("stem"):
        t = truth.loc[stem]
        # stride length is body-relative: the planted ground stride x duty factor
        assert abs(grp["stride_length_euclid_cm"].median() / t["planted_body_frame_excursion_cm"] - 1) < 0.10, stem
        assert abs(grp["duration_ms"].median() - t["planted_stride_period_ms"]) <= 1000.0 / 120 + 1.0, stem
        assert abs(grp["stride_speed_cm_s_mean"].median() / t["planted_speed_cm_s"] - 1) < 0.10, stem
    f = pd.read_csv(g / "gait_per_stride_filtered.csv")
    assert 0 < len(f) < len(s) and "stem" in f.columns

    # metadata: animal -> mouse_id, sex -> sex, every other column added at the end of the tables
    by = s.groupby("stem").first()
    assert by.loc["runway A1", "mouse_id"] == "A1" and by.loc["runway B1", "sex"] == "M"
    assert by.loc["runway A1", "group"] == "control" and float(by.loc["runway A2", "weight_g"]) == 22.5
    last = mock["runway"]["no_metadata_stem"]                                # the clip without a metadata row
    assert pd.isna(by.loc[last, "mouse_id"]) and pd.isna(by.loc[last, "group"])
    for name in ("gait_per_step.csv", "gait_per_video.csv", "gait_per_stride_filtered.csv"):
        cols = list(pd.read_csv(g / name, nrows=1).columns)
        assert cols[-3:] == ["animal", "group", "weight_g"], (name, cols[-5:])
    v = pd.read_csv(g / "gait_per_video.csv")
    assert v[["choice_LR", "reward", "correct_bin", "probes_L"]].isna().all().all()      # no trial information


def test_gait_no_metadata_same_numbers(mock, runway_out, tmp_path):
    out = tmp_path / "out"
    code, text = run(STRIDE + ["gait", mock["runway"]["video_dir"], "--px-per-cm", 24, "-o", out, "--workers", 2])
    assert code == 0 and "[metadata] none given" in text, text[-3000:]
    extra = ["animal", "group", "weight_g"]
    for name, keys in (("gait_per_stride.csv", ["stem", "stride_id"]), ("gait_per_step.csv", ["stem", "step_id", "paw_side", "frame_a"]),
                       ("gait_per_stride_filtered.csv", ["stem", "frame_start", "frame_end"]),
                       ("pose_keypoint_confidence_by_video.csv", ["stem"])):
        a, b = pd.read_csv(runway_out / "gait" / name), pd.read_csv(out / "gait" / name)
        assert not [c for c in extra if c in b.columns]
        assert numeric_equal(a.drop(columns=[c for c in extra if c in a.columns]), b, keys,
                             skip=("mouse_id", "sex", "genotype")) == [], name
    s = pd.read_csv(out / "gait" / "gait_per_stride.csv")
    assert s["mouse_id"].isna().all() and s["sex"].isna().all() and s["genotype"].isna().all()


def test_gait_scale_from_charuco_board(mock, tmp_path):
    b = mock["board"]
    out = tmp_path / "out"
    code, text = run(STRIDE + ["gait", b["video_dir"], "-o", out, "--workers", 2])
    assert code == 0 and "from the ChArUco board in view" in text, text[-3000:]
    scale = json.loads((out / "scale" / "px_per_cm.json").read_text())
    assert scale["source"].startswith("ChArUco") and abs(scale["px_per_cm"] / b["board_px_per_cm"] - 1) < 0.005
    params = json.loads((out / "gait" / "run_params.json").read_text())
    assert params["PX_PER_CM_OVERRIDE"] == scale["px_per_cm"]
    assert len(pd.read_csv(out / "gait" / "gait_per_stride.csv")) > 0
    # a value given by the user wins over the board
    out2 = tmp_path / "out2"
    code, text = run(STRIDE + ["gait", b["video_dir"], "--px-per-cm", 20, "-o", out2, "--workers", 2])
    assert code == 0 and json.loads((out2 / "scale" / "px_per_cm.json").read_text())["source"] == "given by the user"


def test_gait_no_scale_is_refused(mock, tmp_path):
    code, text = run(STRIDE + ["gait", mock["runway"]["video_dir"], "-o", tmp_path / "a", "--workers", 2])
    assert code == 1 and "no scale" in text and "--px-per-cm" in text, text[-2000:]
    assert not (tmp_path / "a" / "gait" / "gait_per_stride.csv").exists()
    code, text = run(STRIDE + ["gait", mock["runway"]["video_dir"], "-o", tmp_path / "b", "--skip", "calibrate"])
    assert code == 1 and "no scale" in text and "calibrate" in text
    code, text = run(STRIDE + ["gait", mock["runway"]["video_dir"], "-o", tmp_path / "c", "--px-per-cm", "-3"])
    assert code == 1 and "must be positive" in text


def test_gait_keypoint_mapping(mock, runway_out, tmp_path):
    r = mock["renamed"]
    # without the mapping: refused, naming what is missing and what the pose file has
    code, text = run(STRIDE + ["gait", r["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "none"])
    assert code == 1 and "lacks keypoint(s)" in text and "'snout'" in text and "tail_root" in text and "--keypoint" in text
    # mapping in the config file
    cfg = tmp_path / "gait.yaml"
    cfg.write_text(yaml.safe_dump({"videos": r["video_dir"], "output": str(tmp_path / "out"), "workers": 2,
                                   "scale": {"px_per_cm": 24}, "keypoints": r["keypoints"]}), encoding="utf-8")
    code, text = run(STRIDE + ["gait", "--config", cfg])
    assert code == 0, text[-3000:]
    ref = pd.read_csv(runway_out / "gait" / "gait_per_stride.csv")
    ref = ref[ref["stem"].isin(r["stems"])].drop(columns=["animal", "group", "weight_g"])
    new = pd.read_csv(tmp_path / "out" / "gait" / "gait_per_stride.csv")
    assert len(new) == len(ref) > 0
    # same coordinates under other node names -> the same numbers (node order differs, so the picked
    # landmark index columns are not compared)
    assert numeric_equal(ref, new, ["stem", "stride_id"], skip=("video_path", "mouse_id", "sex", "genotype",
                                                                "hindL_ind", "hindR_ind")) == []
    conf = pd.read_csv(tmp_path / "out" / "gait" / "pose_keypoint_confidence_by_video.csv")
    assert "avg_snout" in conf.columns and "avg_jaw" in conf.columns and (conf["avg_snout"] > 0.8).all()
    # mapping on the command line; wrong names are refused
    args = [x for k, v in r["keypoints"].items() for x in ("--keypoint", f"{k}={v}")]
    code, text = run(STRIDE + ["gait", r["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "cli", "--workers", 2] + args)
    assert code == 0, text[-3000:]
    code, text = run(STRIDE + ["gait", r["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "x1", "--keypoint", "nose=nozzle"])
    assert code == 1 and "not STRIDE keypoint" in text
    code, text = run(STRIDE + ["gait", r["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "x2", "--keypoint", "beak=snout"])
    assert code == 1 and "not in the pose file" in text
    code, text = run(STRIDE + ["gait", r["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "x3", "--keypoint", "nose"])
    assert code == 2 and "THEIR_NAME=STRIDE_NAME" in text


def test_gait_missing_required_keypoint_is_refused(mock, tmp_path):
    code, text = run(STRIDE + ["gait", mock["no_tailbase"]["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "out"])
    assert code == 1 and "lacks keypoint(s)" in text and "['tailbase']" in text and "'hindpawL1'" in text, text[-2000:]
    assert not (tmp_path / "out" / "gait" / "gait_per_stride.csv").exists()


def test_gait_one_landmark_per_hind_paw(mock, tmp_path):
    o = mock["one_landmark"]
    args = [x for k, v in o["keypoints"].items() for x in ("--keypoint", f"{k}={v}")]
    code, text = run(STRIDE + ["gait", o["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "out", "--workers", 2] + args)
    assert code == 0 and "one left hind paw landmark" in text and "one right hind paw landmark" in text, text[-3000:]
    s = pd.read_csv(tmp_path / "out" / "gait" / "gait_per_stride.csv")
    assert set(s["stem"]) == set(o["stems"]) and (s.groupby("stem").size() >= 5).all()
    assert set(s["hindL_ind"]) == {3} and set(s["hindR_ind"]) == {4}             # the only landmarks there are
    assert len(pd.read_csv(tmp_path / "out" / "gait" / "gait_per_stride_filtered.csv")) > 0
    conf = pd.read_csv(tmp_path / "out" / "gait" / "pose_keypoint_confidence_by_video.csv")
    assert [c for c in conf.columns if c.startswith("avg_")] == ["avg_snout", "avg_tailbase", "avg_tailtip",
                                                                  "avg_hindpawL2", "avg_hindpawR2"]


def test_gait_no_tail_tip(mock, runway_out, tmp_path):
    n = mock["no_tailtip"]
    code, text = run(STRIDE + ["gait", n["video_dir"], "--px-per-cm", 24, "-o", tmp_path / "out", "--workers", 2])
    assert code == 0 and "the tail-tip columns stay empty" in text, text[-3000:]
    new = pd.read_csv(tmp_path / "out" / "gait" / "gait_per_stride.csv")
    assert new[TAILTIP_COLS].isna().all().all()
    ref = pd.read_csv(runway_out / "gait" / "gait_per_stride.csv")
    ref = ref[ref["stem"].isin(n["stems"])].drop(columns=["animal", "group", "weight_g"])
    assert len(new) == len(ref) > 0
    assert numeric_equal(ref, new, ["stem", "stride_id"], skip=("video_path", "mouse_id", "sex", "genotype",
                                                                "hindL_ind", "hindR_ind", "nans_in_stride_TF",
                                                                *TAILTIP_COLS)) == []


def test_gait_config_then_cli_and_steps(mock, tmp_path):
    cfg = tmp_path / "gait.yaml"
    cfg.write_text(yaml.safe_dump({"videos": mock["runway"]["video_dir"], "output": str(tmp_path / "from_cfg"),
                                   "workers": 2, "scale": {"px_per_cm": 10},
                                   "steps": {"stride_filtering": False}}), encoding="utf-8")
    code, text = run(STRIDE + ["gait", "--config", cfg, "--px-per-cm", 24, "-o", tmp_path / "out"])
    assert code == 0 and "[stride_filtering] off" in text, text[-3000:]
    assert not (tmp_path / "from_cfg").exists()
    assert json.loads((tmp_path / "out" / "gait" / "run_params.json").read_text())["PX_PER_CM_OVERRIDE"] == 24
    assert (tmp_path / "out" / "gait" / "gait_per_stride.csv").exists()
    assert not (tmp_path / "out" / "gait" / "gait_per_stride_filtered.csv").exists()
    code, text = run(STRIDE + ["gait", "--config", cfg, "--px-per-cm", 24, "-o", tmp_path / "out", "--enable", "stride_filtering"])
    assert code == 0 and (tmp_path / "out" / "gait" / "gait_per_stride_filtered.csv").exists()
    code, text = run(STRIDE + ["gait", "--config", cfg, "--px-per-cm", 24, "-o", tmp_path / "stop", "--stop-after", "pose_inference"])
    assert code == 0 and "stopped after pose_inference" in text and not (tmp_path / "stop" / "gait").exists()
    code, text = run(STRIDE + ["gait", "--config", cfg, "--px-per-cm", 24, "-o", tmp_path / "noext", "--skip", "gait_extraction",
                               "--enable", "stride_filtering"])
    assert code == 0 and "[gait_extraction] off" in text and "left out: gait_per_stride.csv does not exist" in text


def test_gait_pose_dir_suffix_and_missing_pose(mock, tmp_path):
    src = Path(mock["runway"]["video_dir"])
    vids, poses = tmp_path / "videos here", tmp_path / "poses"
    vids.mkdir(), poses.mkdir()
    stems = mock["runway"]["stems"]
    for v in sorted(src.iterdir()):
        if v.suffix.lower() == ".mp4":
            shutil.copy(v, vids / v.name)
            if v.stem != stems[0]:                                    # one video has no pose file
                shutil.copy(v.with_suffix(".slp"), poses / f"{v.stem}.pose.slp")
    cmd = STRIDE + ["gait", vids, "--pose-dir", poses, "--pose-suffix", ".pose.slp", "--px-per-cm", 24, "--workers", 2]
    code, text = run(cmd + ["-o", tmp_path / "out"])
    assert code == 0 and "no pose model given: 5 videos have a pose file, 1 do not" in text, text[-3000:]
    assert "1 of 6 video(s) were left out" in text and "no pose file" in text
    s = pd.read_csv(tmp_path / "out" / "gait" / "gait_per_stride.csv")
    assert set(s["stem"]) == set(stems[1:])
    code, text = run(cmd + ["-o", tmp_path / "strict", "--strict"])
    assert code == 1 and "left out (--strict)" in text.replace("\n", " ")
    code, text = run(STRIDE + ["gait", vids, "--px-per-cm", 24, "-o", tmp_path / "nopose"])     # no pose file found at all
    assert code == 1 and "none of the 6 videos has a pose file" in text and "--pose-model" in text


def test_gait_resume_does_nothing(mock, tmp_path):
    cmd = STRIDE + ["gait", mock["runway"]["video_dir"], "--px-per-cm", 24, "-m", mock["runway"]["metadata_csv"],
                    "-o", tmp_path / "out", "--workers", 1]
    code, text = run(cmd)
    assert code == 0, text[-3000:]
    files = [tmp_path / "out" / "gait" / n for n in TABLES]
    before = [(f.stat().st_mtime_ns, f.read_bytes()) for f in files]
    code, text = run(cmd)
    assert code == 0, text[-3000:]
    for step in ("gait_extraction", "keypoint_confidence", "stride_filtering"):
        assert f"[{step}] already done" in text, text[-2000:]
    assert before == [(f.stat().st_mtime_ns, f.read_bytes()) for f in files]
    code, text = run(cmd + ["--force"])
    assert code == 0 and "[gait_extraction] 6 videos" in text


def test_gait_spawn_start_method(mock, tmp_path):
    """Windows and macOS start worker processes with `spawn`: the workers must be importable."""
    code, text = run([PY, "-c", (
        "import multiprocessing as mp, sys\n"
        "if __name__ == '__main__':\n"
        "    mp.set_start_method('spawn', force=True)\n"
        "    from stride.stages.gait_pipeline import run_gait_pipeline\n"
        "    r = run_gait_pipeline(videos=sys.argv[1], output=sys.argv[2], workers=2, scale={'px_per_cm': 24},\n"
        "                          keypoints={'hind_left': 'hindpawL2', 'hind_right': 'hindpawR2'})\n"
        "    print('START_METHOD', mp.get_start_method(), 'OK' if r['ok'] else 'FAILED')\n"),
        mock["one_landmark"]["video_dir"], tmp_path / "out spawn"])
    assert code == 0 and "START_METHOD spawn OK" in text, text[-3000:]
    assert len(pd.read_csv(tmp_path / "out spawn" / "gait" / "gait_per_stride_filtered.csv")) > 0


def test_gait_vertical_walking_warns(mock, tmp_path):
    v = mock["vertical"]
    code, text = run(STRIDE + ["gait", v["video_dir"], "--px-per-cm", v["px_per_cm"], "-o", tmp_path / "out", "--workers", 2])
    assert code == 0, text[-3000:]
    flat = " ".join(text.split())
    assert "only 0 % of the walking frames have the body axis within 45 degrees of the image x-axis" in flat
    assert "below 20 px/cm" in flat                                    # the stage's own scale warning is passed on
    summary = (tmp_path / "out" / "SUMMARY.md").read_text(encoding="utf-8")
    assert "## Warnings" in summary and "left-right in the image" in summary
    assert len(pd.read_csv(tmp_path / "out" / "gait" / "gait_per_stride.csv")) > 0
    man = yaml.safe_load((tmp_path / "out" / "stride_gait_run_manifest.yaml").read_text(encoding="utf-8"))
    assert man["results"]["stages"]["gait_extraction"]["share_along_image_x"] == 0.0


def test_gait_refusals(mock, tmp_path):
    vd = mock["runway"]["video_dir"]
    code, text = run(STRIDE + ["gait", "-o", tmp_path / "a", "--px-per-cm", 24])
    assert code == 2 and "no videos given" in text
    code, text = run(STRIDE + ["gait", vd, "--px-per-cm", 24])
    assert code == 2 and "no output folder given" in text
    code, text = run(STRIDE + ["gait", vd, "--px-per-cm", 24, "-o", tmp_path / "b", "--skip", "decisions"])
    assert code == 2 and "unknown step" in text
    code, text = run(STRIDE + ["gait", tmp_path / "nowhere", "--px-per-cm", 24, "-o", tmp_path / "c"])
    assert code == 1 and "does not exist" in text
    (tmp_path / "empty").mkdir()
    code, text = run(STRIDE + ["gait", tmp_path / "empty", "--px-per-cm", 24, "-o", tmp_path / "d"])
    assert code == 1 and "no .mp4 videos found" in text
    (tmp_path / "meta.csv").write_text("animal,sex\nA1,F\n", encoding="utf-8")
    code, text = run(STRIDE + ["gait", vd, "--px-per-cm", 24, "-o", tmp_path / "e", "-m", tmp_path / "meta.csv"])
    assert code == 1 and "needs a `video` column" in text
    code, text = run(STRIDE + ["gait", vd, "--px-per-cm", 24, "-o", tmp_path / "f", "--pose-dir", tmp_path / "nope"])
    assert code == 2 and "pose_dir: folder not found" in text


# --------------------------------------------------------------------------------------------
# same numbers as the existing path
# --------------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def core(tmp_path_factory) -> dict:
    reuse = os.environ.get("STRIDE_MOCK_DATA")
    if reuse and (Path(reuse) / "core" / "scenario.json").exists():
        root = Path(reuse).resolve()
    else:
        root = tmp_path_factory.mktemp("stride_mock_core_for_gait")
        code, out = run([PY, REPO / "tests" / "mock" / "make_mock_dataset.py", "--out", root, "--only", "core"], timeout=1800)
        assert code == 0, out[-2000:]
    return json.loads((root / "core" / "scenario.json").read_text())


def test_gait_equals_existing_path_on_tmaze_mock(core, tmp_path):
    """T-maze mock `core`: `stride gait` (no ROI used) vs the gait tables of the T-maze path, same px/cm."""
    ref = tmp_path / "tmaze"
    code, text = run(STRIDE + ["tmaze", core["video_dir"], "--roi-dir", core["yml_dir"], "-m", core["meta_trials_csv"],
                               "-m", core["metadata_csv"], "-o", ref, "--skip-inference", "--px-per-cm", 16, "--workers", 2])
    assert code == 0, text[-3000:]
    mm = pd.read_csv(ref / "gait" / "metrics_meta.csv", dtype=str, keep_default_na=False)
    pd.DataFrame({"video": mm["video_path"].map(lambda p: Path(p).name), "sex": mm["Gender"],
                  "genotype": mm["Genotype"]}).to_csv(tmp_path / "by_video.csv", index=False)
    out = tmp_path / "gait"
    code, text = run(STRIDE + ["gait", core["video_dir"], "--px-per-cm", 16, "-m", tmp_path / "by_video.csv", "-o", out, "--workers", 2])
    assert code == 0, text[-3000:]
    for name, keys in (("gait_per_stride.csv", ["video_path", "stride_id"]),
                       ("gait_per_step.csv", ["video_path", "step_id", "paw_side", "frame_a"]),
                       ("gait_per_stride_filtered.csv", ["video_path", "frame_start", "frame_end"]),
                       ("pose_keypoint_confidence_by_video.csv", ["video_path"])):
        a, b = pd.read_csv(ref / "gait" / name), pd.read_csv(out / "gait" / name)
        assert len(a) == len(b) > 0 and list(a.columns) == list(b.columns), name
        assert numeric_equal(a, b, keys) == [], name
    # the per-video table: everything except the trial columns, which come from the decision stage
    a, b = pd.read_csv(ref / "gait" / "gait_per_video.csv"), pd.read_csv(out / "gait" / "gait_per_video.csv")
    trial_cols = ["choice_LR", "latency_choice_ms", "stem_latency_ms", "junction_explore_ms", "reward", "correct_TF",
                  "correct_bin", "probes_L", "probes_R"]
    assert list(a.columns) == list(b.columns) and numeric_equal(a, b, ["video_path"], skip=trial_cols) == []
    assert b[trial_cols].isna().all().all()
