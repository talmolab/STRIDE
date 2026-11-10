"""Tests of `stride tmaze` (the automated T-maze pipeline) on the synthetic mock set.

No animal video, no model, no GPU: the two inference steps are switched off (`--skip-inference`)
and the pose / ROI files of the mock set are used. What is asserted:

  * lab-pattern names with metadata: planted choices recovered, every table written, no statistics;
  * arbitrary file names (with spaces) without metadata: correctness empty, everything else equal;
  * metadata keyed by file name; config file + command-line override; `stride init-config`;
  * a clip without an ROI gets the consensus ROI of its folder; one that cannot is left out with the reason;
  * px/cm precedence: user value > ChArUco measurement (stubbed) > ROI corridor width;
  * a second run does nothing; every refusal exits non-zero and prints the reason;
  * the pipeline runs with the multiprocessing start method `spawn` (Windows / macOS default);
  * the decision-rules port and the best-instance ROI selection (unit tests).
"""
from __future__ import annotations

import csv
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
STRIDE = [PY, "-m", "stride.cli"]
CHOICE_COLS = ["commit_side", "commit_frame", "ok", "enter_seg1_frame", "enter_seg4_frame", "enter_junction_frame",
               "enter_seg1_ms", "enter_seg4_ms", "enter_junction_ms", "commit_ms",
               "junction_explore_frames_precommit", "junction_explore_ms_precommit",
               "probes_L", "probes_R", "probe_frames_L", "probe_frames_R",
               "n_reversals", "reversal_frames", "reversal_ms", "coverage_thr_rule"]


def run(cmd, cwd=None, timeout=900):
    env = dict(os.environ, PYTHONWARNINGS="ignore", MPLBACKEND="Agg", COLUMNS="200", NO_COLOR="1")
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + "\n" + p.stderr


@pytest.fixture(scope="session")
def core(tmp_path_factory) -> dict:
    """scenario.json of the mock `core` scenario (generated once, or reused via STRIDE_MOCK_DATA)."""
    reuse = os.environ.get("STRIDE_MOCK_DATA")
    if reuse and (Path(reuse) / "core" / "scenario.json").exists():
        root = Path(reuse).resolve()
    else:
        root = tmp_path_factory.mktemp("stride_mock_tmaze")
        code, out = run([PY, REPO / "tests" / "mock" / "make_mock_dataset.py", "--out", root, "--only", "core"],
                        timeout=1800)
        assert code == 0, out[-2000:]
    return json.loads((root / "core" / "scenario.json").read_text())


@pytest.fixture(scope="session")
def core_tmaze(core, tmp_path_factory) -> Path:
    """`stride tmaze` on the core scenario: lab-pattern names, metadata, existing pose + ROI files."""
    out = tmp_path_factory.mktemp("tmaze_core") / "out"
    code, text = run(STRIDE + ["tmaze", core["video_dir"], "--roi-dir", core["yml_dir"],
                               "-m", core["meta_trials_csv"], "-m", core["metadata_csv"],
                               "-o", out, "--skip-inference", "--workers", 2])
    assert code == 0, text[-3000:]
    return out


def copy_clips(core: dict, dst: Path, stems: list, rename=None, with_roi=True) -> dict:
    """Copy a few core clips (video + pose + ROI next to it) into ``dst``; returns old stem -> new path."""
    dst.mkdir(parents=True, exist_ok=True)
    out = {}
    for i, stem in enumerate(stems):
        new = rename(i, stem) if rename else stem
        v = Path(core["video_dir"]) / f"{stem}.mp4"
        shutil.copy(v, dst / f"{new}.mp4")
        shutil.copy(v.with_suffix(".slp"), dst / f"{new}.slp")
        if with_roi is True or (callable(with_roi) and with_roi(i, stem)):
            shutil.copy(Path(core["yml_dir"]) / f"{stem}.preds.v2.best1.yml", dst / f"{new}.rois.yml")
        out[stem] = dst / f"{new}.mp4"
    return out


def _draw_board_into(mk, videos: list, pxcm: float = 16.0) -> float:
    """Rewrite the (blank) mock videos with a ChArUco board in view; returns the exact px/cm drawn."""
    import cv2
    patch, exact = mk.charuco_patch(pxcm)
    for v in videos:
        cap = cv2.VideoCapture(str(v))
        fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        frame = np.full((h, w, 3), 90, np.uint8)
        ph, pw = patch.shape[:2]
        frame[20:20 + ph, 20:20 + pw] = patch[..., None] if patch.ndim == 2 else patch
        tmp = v.with_name(v.stem + ".tmp.mp4")
        wr = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for _ in range(n):
            wr.write(frame)
        wr.release()
        tmp.replace(v)
    return exact


def first_stems(core: dict, n: int) -> list:
    return sorted(p.stem for p in Path(core["video_dir"]).glob("*.mp4"))[:n]


# --------------------------------------------------------------------------------------------
# the supported cases
# --------------------------------------------------------------------------------------------
def test_tmaze_help_and_init_config(tmp_path):
    for sub in (["tmaze"], ["init-config"], ["apply-decision-rules"], ["propagate-roi"]):
        code, out = run(STRIDE + sub + ["--help"])
        assert code == 0 and "Usage" in out
    cfg = tmp_path / "my config.yaml"
    code, out = run(STRIDE + ["init-config", "tmaze", "-o", cfg])
    assert code == 0 and cfg.exists()
    from stride.stages.tmaze_pipeline import DEFAULTS, STEPS, load_settings
    doc = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert set(doc) <= set(DEFAULTS) and set(doc["steps"]) == set(STEPS)
    assert doc["steps"]["mixed_models"] is False and doc["steps"]["classifier"] is False
    assert all(doc["steps"][s] for s in STEPS if s not in ("mixed_models", "classifier"))
    load_settings(cfg)                                    # the template is a valid config
    code, out = run(STRIDE + ["init-config", "tmaze", "-o", cfg])
    assert code != 0 and "already exists" in out


def test_tmaze_lab_names_with_metadata(core, core_tmaze):
    out = core_tmaze
    for rel in ("decisions/decisions.csv", "decisions/decisions_clean.csv", "decisions/events.csv",
                "decisions/metrics.csv", "gait/gait_per_stride.csv", "gait/gait_per_step.csv",
                "gait/gait_per_video.csv", "gait/gait_per_stride_filtered.csv",
                "gait/pose_keypoint_confidence_by_video.csv", "SUMMARY.md", "tmaze_run_manifest.yaml",
                "decisions/decisions_run_manifest.yaml", "gait/gait_run_manifest.yaml", "scale/px_per_cm.json",
                "roi/roi_sources.csv", "tmaze_run.log"):
        assert (out / rel).exists(), rel
    truth = pd.read_csv(core["truth_csv"])
    dec = pd.read_csv(out / "decisions" / "decisions.csv")
    m = truth.merge(dec, on="stem")
    assert len(m) == len(truth) == len(dec)
    assert (m["planted_choice"].fillna("") == m["entry_side"].fillna("")).all()
    assert (m["expected_correct"] == m["correct_bin"]).all()
    # no genotype statistics unless asked for
    assert not (out / "stats").exists()
    assert not list(out.rglob("lmm_results*")) and not list(out.rglob("*classif*"))
    man = yaml.safe_load((out / "tmaze_run_manifest.yaml").read_text(encoding="utf-8"))
    assert man["params"]["steps"]["pose_inference"] is False and man["params"]["workers"] == 2
    assert man["results"]["roi_sources"] == {"detected": len(truth)}


def test_tmaze_same_numbers_as_stage_entry_points(core, core_tmaze, tmp_path):
    """The command adds no computation: its tables equal those of the stage functions called directly."""
    from stride.stages.decision_analysis import run_decision_analysis
    ref = tmp_path / "ref"
    run_decision_analysis(video_dir=Path(core["video_dir"]), yml_dir=Path(core["yml_dir"]),
                          meta_csv=Path(core["meta_trials_csv"]), output_dir=ref, n_workers=2)
    for name in ("decisions.csv", "events.csv", "metrics.csv"):
        a, b = pd.read_csv(ref / name), pd.read_csv(core_tmaze / "decisions" / name)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_tmaze_arbitrary_names_no_metadata_and_by_filename(core, core_tmaze, tmp_path):
    stems = first_stems(core, 6)
    vids = copy_clips(core, tmp_path / "my videos" / "session one", stems, rename=lambda i, s: f"clip {i + 1} (cam A)")
    new_stem = {s: p.stem for s, p in vids.items()}
    ref = pd.read_csv(core_tmaze / "decisions" / "events.csv").set_index("stem")
    ref_dec = pd.read_csv(core_tmaze / "decisions" / "decisions.csv").set_index("stem")

    # (a) no metadata: correctness empty, everything else as with metadata
    out = tmp_path / "out a"
    code, text = run(STRIDE + ["tmaze", tmp_path / "my videos", "-o", out, "--skip-inference", "--workers", 2])
    assert code == 0, text[-3000:]
    assert "correctness" in text and "empty" in text
    ev = pd.read_csv(out / "decisions" / "events.csv").set_index("stem")
    dec = pd.read_csv(out / "decisions" / "decisions.csv").set_index("stem")
    met = pd.read_csv(out / "decisions" / "metrics.csv")
    clean = pd.read_csv(out / "decisions" / "decisions_clean.csv")
    assert len(ev) == 6
    assert dec["correct_TF"].isna().all() and dec["correct_bin"].isna().all()
    assert met["correct_TF"].isna().all() and met["correct_bin"].isna().all() and clean["correct_bin"].isna().all()
    assert dec["day"].isna().all() and dec["mouse"].isna().all() and dec["trial"].isna().all()
    for old, new in new_stem.items():
        for c in CHOICE_COLS:
            a, b = ref.loc[old, c], ev.loc[new, c]
            assert (pd.isna(a) and pd.isna(b)) or a == b, (old, c, a, b)
        assert ref_dec.loc[old, "entry_side"] == dec.loc[new, "entry_side"]
    g = pd.read_csv(out / "gait" / "gait_per_stride.csv")
    assert len(g) > 0 and set(g["stem"]) <= set(new_stem.values())

    # (b) metadata keyed by file name
    truth = pd.read_csv(core["truth_csv"]).set_index("stem")
    meta = tmp_path / "meta by file.csv"
    with open(meta, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Video", "Reward", "Mouse", "Day", "Trial", "Genotype", "Sex"])
        for old, p in vids.items():
            t = truth.loc[old]
            w.writerow([p.name, t["reward"], t["mouse"], t["day"], t["trial"], t["genotype"], t["sex"]])
    out = tmp_path / "out b"
    code, text = run(STRIDE + ["tmaze", tmp_path / "my videos", "-o", out, "-m", meta, "--skip-inference", "--workers", 2])
    assert code == 0, text[-3000:]
    dec = pd.read_csv(out / "decisions" / "decisions.csv").set_index("stem")
    for old, new in new_stem.items():
        assert dec.loc[new, "correct_bin"] == ref_dec.loc[old, "correct_bin"]
        assert str(dec.loc[new, "mouse"]) == str(ref_dec.loc[old, "mouse"]) and dec.loc[new, "day"] == ref_dec.loc[old, "day"]
    g = pd.read_csv(out / "gait" / "gait_per_stride.csv")
    assert set(g["genotype"].dropna()) <= {"WT", "Tau", "APP"} and g["genotype"].notna().all()


def test_tmaze_config_then_cli(core, tmp_path):
    stems = first_stems(core, 3)
    copy_clips(core, tmp_path / "v", stems)
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({
        "videos": str(tmp_path / "v"), "output": str(tmp_path / "from config"), "metadata": core["meta_trials_csv"],
        "workers": 1, "scale": {"px_per_cm": 16.0},
        "steps": {"pose_inference": False, "roi_inference": False, "distortion_check": False, "gait": False},
    }), encoding="utf-8")
    code, text = run(STRIDE + ["tmaze", "--config", cfg])
    assert code == 0, text[-3000:]
    assert (tmp_path / "from config" / "decisions" / "decisions.csv").exists()
    assert not (tmp_path / "from config" / "gait").exists()            # step switched off in the config
    assert "[gait] off" in text and "[distortion_check] off" in text
    # command line overrides the config: other output folder, gait back on, other px/cm
    code, text = run(STRIDE + ["tmaze", "--config", cfg, "-o", tmp_path / "from cli", "--enable", "gait",
                               "--px-per-cm", 20])
    assert code == 0, text[-3000:]
    params = json.loads((tmp_path / "from cli" / "gait" / "run_params.json").read_text())
    assert params["PX_PER_CM_OVERRIDE"] == 20.0
    man = yaml.safe_load((tmp_path / "from cli" / "tmaze_run_manifest.yaml").read_text(encoding="utf-8"))
    assert man["params"]["scale"]["px_per_cm"] == 20.0 and man["params"]["steps"]["gait"] is True
    assert man["params"]["config_file"] == str(cfg)


def test_tmaze_stop_after_and_step_off_message(core, tmp_path):
    copy_clips(core, tmp_path / "v", first_stems(core, 2))
    code, text = run(STRIDE + ["tmaze", tmp_path / "v", "-o", tmp_path / "o1", "--skip-inference",
                               "--stop-after", "decisions"])
    assert code == 0 and "stopped after decisions" in text
    assert (tmp_path / "o1" / "decisions" / "decisions.csv").exists() and not (tmp_path / "o1" / "gait").exists()
    # decisions off: the rules step says why it is left out; gait still runs from the pose files
    code, text = run(STRIDE + ["tmaze", tmp_path / "v", "-o", tmp_path / "o2", "--skip-inference", "--skip", "decisions"])
    assert code == 0, text[-3000:]
    assert "[decision_rules] left out" in text and "Traceback" not in text
    assert (tmp_path / "o2" / "gait" / "gait_per_stride.csv").exists()


def test_tmaze_consensus_roi(core, tmp_path):
    stems = first_stems(core, 8)
    # session A: 5 clips, one without an ROI -> consensus of the other four
    copy_clips(core, tmp_path / "v" / "session A", stems[:5], with_roi=lambda i, s: i != 2)
    # session B: one clip, no ROI anywhere -> cannot be recovered
    copy_clips(core, tmp_path / "v" / "session B", stems[5:6], with_roi=False)
    out = tmp_path / "out"
    code, text = run(STRIDE + ["tmaze", tmp_path / "v", "-o", out, "--skip-inference", "--workers", 2])
    assert code == 0, text[-3000:]
    src = pd.read_csv(out / "roi" / "roi_sources.csv").set_index("stem")
    assert src.loc[stems[2], "roi_source"] == "consensus" and "consensus_roi_of: 4 clips" in src.loc[stems[2], "note"]
    assert src.loc[stems[5], "roi_source"] == "missing"
    assert (src.loc[[stems[0], stems[1], stems[3], stems[4]], "roi_source"] == "detected").all()
    yml = out / "roi" / f"{stems[2]}.rois.yml"
    assert "consensus_roi_of: 4 clips" in yml.read_text(encoding="utf-8")
    assert (out / "roi" / "roi_consensus_qc" / "session A" / f"{stems[2]}.png").exists()
    assert (out / "roi" / "roi_consensus_qc" / "session A" / "MONTAGE_1.png").exists()
    dec = pd.read_csv(out / "decisions" / "decisions.csv")
    assert stems[2] in set(dec["stem"]) and stems[5] not in set(dec["stem"]) and len(dec) == 5
    summ = (out / "SUMMARY.md").read_text(encoding="utf-8")
    assert f"{stems[5]}.mp4" in summ and "no ROI" in summ and "consensus" in summ
    man = yaml.safe_load((out / "tmaze_run_manifest.yaml").read_text(encoding="utf-8"))
    assert man["results"]["roi_sources"] == {"detected": 4, "consensus": 1, "missing": 1}
    # --strict turns a left-out video into an error exit
    code, text = run(STRIDE + ["tmaze", tmp_path / "v", "-o", out, "--skip-inference", "--strict"])
    assert code != 0 and "left out" in text

    # static-maze guard: a donor whose maze sits elsewhere -> the consensus is refused, with the reason
    d = tmp_path / "moved" / "s"
    copy_clips(core, d, stems[:4], with_roi=lambda i, s: i != 0)
    doc = yaml.safe_load((d / f"{stems[1]}.rois.yml").read_text())
    for r in doc["rois"]:
        r["coordinates"] = [[x + 300.0, y] for x, y in r["coordinates"]]
    (d / f"{stems[1]}.rois.yml").write_text(yaml.safe_dump(doc, sort_keys=False))
    code, text = run(STRIDE + ["tmaze", tmp_path / "moved", "-o", tmp_path / "out moved", "--skip-inference"])
    assert code == 0, text[-3000:]
    assert "consensus ROI refused" in text and "may have moved" in text
    src = pd.read_csv(tmp_path / "out moved" / "roi" / "roi_sources.csv").set_index("stem")
    assert src.loc[stems[0], "roi_source"] == "missing"


def test_tmaze_scale_precedence(core, tmp_path, monkeypatch):
    from stride.stages.tmaze_pipeline import load_settings, run_tmaze_pipeline
    copy_clips(core, tmp_path / "v", first_stems(core, 2))
    base = {"videos": str(tmp_path / "v"), "workers": 1,
            "steps": {"pose_inference": False, "roi_inference": False, "distortion_check": False}}

    def go(name, **extra):
        s = load_settings(None, {**base, "output": str(tmp_path / name), **extra})
        r = run_tmaze_pipeline(s)
        assert r["ok"], r
        scale = json.loads((tmp_path / name / "scale" / "px_per_cm.json").read_text())
        params = json.loads((tmp_path / name / "gait" / "run_params.json").read_text())
        return scale, params

    # (c) nothing given and no board in the (blank) mock videos: per video from the ROI corridor width
    scale, params = go("roi")
    assert scale["px_per_cm"] is None and scale["source"].startswith("ROI corridor width")
    assert scale["charuco_reason"]                                      # the real module said why
    assert "PX_PER_CM_OVERRIDE" not in params
    import stride.stages.scale_calibration as real_calibration
    px = pd.read_csv(tmp_path / "roi" / "gait" / "gait_per_stride.csv")["px_per_cm"]     # column holds cm per px
    assert np.allclose(1.0 / px, 16.0, rtol=0.02)

    # (b) ChArUco measurement available (stub of the calibration contract)
    calls = {}

    def calibrate_px_per_cm(videos, out_dir, board=None, max_clips=None, **kw):
        calls.update(n=len(videos), board=board, max_clips=max_clips)
        return {"ok": True, "px_per_cm": 24.5, "spread_pct": 0.2, "n_clips_used": 2, "n_clips": 2,
                "per_clip": [{"clip": "a", "px_per_cm": 24.5}], "method": "stub", "reason": ""}

    stub = types.ModuleType("stride.stages.scale_calibration")
    stub.calibrate_px_per_cm = calibrate_px_per_cm
    stub.pick_clips = real_calibration.pick_clips
    monkeypatch.setitem(sys.modules, "stride.stages.scale_calibration", stub)
    scale, params = go("charuco", scale={"charuco": {"squares_x": 7, "square_mm": 25, "max_clips": 4}})
    assert scale["px_per_cm"] == 24.5 and scale["source"].startswith("ChArUco")
    assert params["PX_PER_CM_OVERRIDE"] == 24.5
    assert calls == {"n": 2, "board": {"squares_x": 7, "square_mm": 25}, "max_clips": 4}

    # (a) the user's value wins over the ChArUco measurement
    calls.clear()
    scale, params = go("user", scale={"px_per_cm": 30.0})
    assert scale["px_per_cm"] == 30.0 and scale["source"] == "given by the user" and not calls
    assert params["PX_PER_CM_OVERRIDE"] == 30.0

    # ChArUco module present but no board found: falls through to the ROI, with the reason
    stub.calibrate_px_per_cm = lambda *a, **k: {"ok": False, "px_per_cm": None, "reason": "no board in 2 clips"}
    scale, params = go("noboard")
    assert scale["px_per_cm"] is None and scale["charuco_reason"] == "no board in 2 clips"

    # calibrate switched off: the module is not called
    stub.calibrate_px_per_cm = lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called"))
    scale, params = go("off", steps={"calibrate": False})
    assert scale["px_per_cm"] is None

    # the module cannot be imported at all: the run still completes on the ROI width
    monkeypatch.setitem(sys.modules, "stride.stages.scale_calibration", None)     # import raises ImportError
    scale, params = go("nomodule")
    assert scale["px_per_cm"] is None and "could not be loaded" in scale["charuco_reason"]


def test_tmaze_scale_from_charuco_in_view(core, tmp_path):
    """The real calibration module inside `stride tmaze`: a board drawn into the videos at a known px/cm."""
    sys.path.insert(0, str(REPO / "tests" / "mock"))
    try:
        import make_mock_charuco as mk
    finally:
        sys.path.pop(0)
    from stride.stages.tmaze_pipeline import load_settings, run_tmaze_pipeline
    stems = first_stems(core, 3)
    vids = copy_clips(core, tmp_path / "v", stems)
    makers = [n for n in dir(mk) if not n.startswith("_")]
    assert makers, "make_mock_charuco has no public helper"
    planted = _draw_board_into(mk, list(vids.values()))
    s = load_settings(None, {"videos": str(tmp_path / "v"), "output": str(tmp_path / "out"), "workers": 2,
                             "scale": {"charuco": {"min_clips": 3}},
                             "steps": {"pose_inference": False, "roi_inference": False, "distortion_check": False}})
    r = run_tmaze_pipeline(s)
    assert r["ok"], r
    scale = json.loads((tmp_path / "out" / "scale" / "px_per_cm.json").read_text())
    assert scale["source"].startswith("ChArUco") and abs(scale["px_per_cm"] / planted - 1) < 0.005
    params = json.loads((tmp_path / "out" / "gait" / "run_params.json").read_text())
    assert params["PX_PER_CM_OVERRIDE"] == scale["px_per_cm"]
    assert (tmp_path / "out" / "scale" / "px_per_cm.txt").exists()
    # the videos to measure can be named explicitly; a missing one stops the run with the reason
    s2 = load_settings(None, {**{k: s[k] for k in ("videos", "workers", "steps")}, "output": str(tmp_path / "out2"),
                              "scale": {"charuco": {"clips": [str(v) for v in vids.values()]}}})
    r = run_tmaze_pipeline(s2)
    assert r["ok"] and json.loads((tmp_path / "out2" / "scale" / "px_per_cm.json").read_text())["px_per_cm"] == scale["px_per_cm"]
    s3 = load_settings(None, {**{k: s[k] for k in ("videos", "workers", "steps")}, "output": str(tmp_path / "out3"),
                              "scale": {"charuco": {"clips": [str(tmp_path / "nope.mp4")]}}})
    r = run_tmaze_pipeline(s3)
    assert not r["ok"] and "video not found" in r["error"]


def test_tmaze_resume_does_nothing(core, tmp_path):
    copy_clips(core, tmp_path / "v", first_stems(core, 3))
    cmd = STRIDE + ["tmaze", tmp_path / "v", "-o", tmp_path / "out", "--skip-inference", "--workers", 1]
    code, text = run(cmd)
    assert code == 0, text[-3000:]
    files = [tmp_path / "out" / "decisions" / "decisions.csv", tmp_path / "out" / "decisions" / "decisions_clean.csv",
             tmp_path / "out" / "gait" / "gait_per_stride.csv", tmp_path / "out" / "gait" / "gait_per_stride_filtered.csv"]
    before = [(f.stat().st_mtime_ns, f.read_bytes()) for f in files]
    code, text = run(cmd)
    assert code == 0, text[-3000:]
    for step in ("decisions", "decision_rules", "gait"):
        assert f"[{step}] already done" in text, text[-2000:]
    assert before == [(f.stat().st_mtime_ns, f.read_bytes()) for f in files]
    code, text = run(cmd + ["--force"])                                 # --force redoes the work
    assert code == 0 and "[decisions] 3 of 3 videos analysed" in text


def test_tmaze_spawn_start_method(core, tmp_path):
    """Windows and macOS start worker processes with `spawn`: the workers must be importable."""
    code, text = run([PY, "-c", (
        "import multiprocessing as mp, sys\n"
        "if __name__ == '__main__':\n"
        "    mp.set_start_method('spawn', force=True)\n"
        "    from stride.stages.tmaze_pipeline import load_settings, run_tmaze_pipeline\n"
        "    s = load_settings(None, {'videos': sys.argv[1], 'roi_dir': sys.argv[2], 'metadata': sys.argv[3],\n"
        "                             'output': sys.argv[4], 'workers': 2,\n"
        "                             'steps': {'pose_inference': False, 'roi_inference': False}})\n"
        "    r = run_tmaze_pipeline(s)\n"
        "    print('START_METHOD', mp.get_start_method(), 'OK' if r['ok'] else 'FAILED')\n"),
        core["video_dir"], core["yml_dir"], core["meta_trials_csv"], tmp_path / "out spawn"])
    assert code == 0 and "START_METHOD spawn OK" in text, text[-3000:]
    dec = pd.read_csv(tmp_path / "out spawn" / "decisions" / "decisions.csv")
    assert len(dec) == core["n_clips"]
    assert len(pd.read_csv(tmp_path / "out spawn" / "gait" / "gait_per_stride_filtered.csv")) > 0


def test_tmaze_optional_statistics(core, core_tmaze):
    """mixed_models / classifier are off by default; switched on they run the existing stages."""
    cmd = STRIDE + ["tmaze", core["video_dir"], "--roi-dir", core["yml_dir"], "-m", core["meta_trials_csv"],
                    "-m", core["metadata_csv"], "-o", core_tmaze, "--skip-inference", "--workers", 2]
    cfg = core_tmaze.parent / "stats.yaml"
    cfg.write_text(yaml.safe_dump({"stats_config": str(REPO / "config.example.yaml")}), encoding="utf-8")
    code, text = run(cmd + ["--config", cfg, "--enable", "mixed_models"])
    assert code == 0, text[-3000:]
    assert "[gait] already done" in text                      # nothing upstream is repeated
    assert (core_tmaze / "stats" / "lmm_results_all.csv").exists()
    assert not (core_tmaze / "stats" / "regression_feature_ranking.csv").exists()   # classifier still off


# --------------------------------------------------------------------------------------------
# refusals: non-zero exit and the reason
# --------------------------------------------------------------------------------------------
def test_tmaze_refusals(core, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    v = tmp_path / "v"
    copy_clips(core, v, first_stems(core, 2))
    nopose = tmp_path / "nopose"
    copy_clips(core, nopose, first_stems(core, 2))
    for f in nopose.glob("*.slp"):
        f.unlink()
    dup = tmp_path / "dup"
    copy_clips(core, dup / "a", first_stems(core, 1))
    copy_clips(core, dup / "b", first_stems(core, 1))
    bad_cfg = tmp_path / "bad.yaml"
    bad_cfg.write_text("videoz: /x\n", encoding="utf-8")
    bad_step = tmp_path / "badstep.yaml"
    bad_step.write_text("steps: {gaits: false}\n", encoding="utf-8")
    cases = [
        (["tmaze", "-o", tmp_path / "o"], "no videos given"),
        (["tmaze", v], "no output folder given"),
        (["tmaze", empty, "-o", tmp_path / "o"], "no .mp4 videos found"),
        (["tmaze", tmp_path / "missing folder", "-o", tmp_path / "o"], "does not exist"),
        (["tmaze", dup, "-o", tmp_path / "o", "--skip-inference"], "used by more than one video"),
        (["tmaze", nopose, "-o", tmp_path / "o"], "--pose-model"),                      # inference on, no model
        (["tmaze", nopose, "-o", tmp_path / "o", "--pose-model", tmp_path / "no such model"], "not found"),
        (["tmaze", nopose, "-o", tmp_path / "o", "--skip-inference"], "no video has both a pose file and an ROI file"),
        (["tmaze", v, "-o", tmp_path / "o", "--skip", "gaits"], "unknown step"),
        (["tmaze", v, "-o", tmp_path / "o", "--stop-after", "nowhere"], "unknown step"),
        (["tmaze", "--config", tmp_path / "nope.yaml"], "config file not found"),
        (["tmaze", "--config", bad_cfg], "unknown setting"),
        (["tmaze", "--config", bad_step], "unknown step"),
        (["tmaze", v, "-o", tmp_path / "o", "-m", tmp_path / "nometa.csv", "--skip-inference"], "metadata file not found"),
        (["tmaze", v, "-o", tmp_path / "o", "--roi-dir", tmp_path / "noroi"], "folder not found"),
        (["tmaze", v, "-o", tmp_path / "o", "--skip-inference", "--enable", "mixed_models"], "stats_config"),
        (["tmaze", v, "-o", tmp_path / "o", "--skip-inference", "--px-per-cm", "-3"], "must be positive"),
        (["apply-decision-rules", tmp_path / "nofile.csv"], "not found"),
        (["propagate-roi", tmp_path / "nofolder"], "not found"),
    ]
    for args, reason in cases:
        code, text = run(STRIDE + args)
        assert code != 0, (args, text[-800:])
        assert reason in " ".join(text.split()), (args, reason, text[-1500:])
        assert "Traceback" not in text, (args, text[-1500:])


# --------------------------------------------------------------------------------------------
# unit tests of the two ported pieces
# --------------------------------------------------------------------------------------------
def test_decision_rules_port(tmp_path):
    from stride.stages.decision_rules import apply_decision_rules
    cols = ["video_path", "stem", "day", "mouse", "trial", "reward", "fps", "ok", "reason", "gate_frame",
            "gate_reason", "entry_side", "entry_frame", "entry_ms", "entry_threshold", "correct_TF", "correct_bin"]
    strict, soft = ">=95% for ≥3f", ">=60% for ≥3f"
    rows = [
        ["a.mp4", "a", "DAY1", "1", 1, "L", 120.0, True, "", 60, "g", "L", 500, 4166.7, strict, "T", 1],     # stays
        ["b.mp4", "b", "DAY1", "1", 2, "L", 120.0, True, "", 60, "g", "R", 500, 4166.7, soft, "F", 0],       # soft
        ["c.mp4", "c", "DAY1", "1", 3, "L", 120.0, True, "", 60, "g", "L", 20000, 166666.7, strict, "T", 1], # late
        ["d.mp4", "d", "DAY1", "1", 4, "L", 120.0, True, "", 60, "g", "L", 20000, 166666.7, soft, "T", 1],   # soft + late
        ["e.mp4", "e", "DAY1", "1", 5, "L", 120.0, False, "no_arm_dwell_after_gate", 60, "g", "", "", "", "none", "F", 0],
        ["f.mp4", "f", "", "", "", "", 120.0, True, "", 60, "g", "L", 500, 4166.7, soft, "", ""],             # reward unknown
    ]
    raw = tmp_path / "decisions.csv"
    pd.DataFrame(rows, columns=cols).to_csv(raw, index=False)
    before = raw.read_bytes()
    r = apply_decision_rules(raw, verbose=False)
    assert raw.read_bytes() == before                                   # the raw file is untouched
    assert (r["trials"], r["soft"], r["late"], r["decisions"], r["no_decision"]) == (6, 3, 1, 1, 5)
    c = pd.read_csv(tmp_path / "decisions_clean.csv", keep_default_na=False, dtype=str).set_index("stem")
    assert list(c.columns) == cols[:1] + cols[2:]
    assert c.loc["a"].to_dict() == pd.read_csv(raw, keep_default_na=False, dtype=str).set_index("stem").loc["a"].to_dict()
    assert c.loc["b", ["ok", "reason", "entry_side", "entry_frame", "entry_ms", "entry_threshold", "correct_TF", "correct_bin"]].tolist() == \
        ["False", "soft_entry_no_commit", "", "", "", "none", "F", "0"]
    assert c.loc["c", "reason"] == "entry_after_window" and c.loc["d", "reason"] == "soft_entry_no_commit"
    assert c.loc["e", "reason"] == "no_arm_dwell_after_gate"
    assert c.loc["f", "ok"] == "False" and c.loc["f", "correct_TF"] == "" and c.loc["f", "correct_bin"] == ""
    r = apply_decision_rules(raw, window_s=1000, out_csv=tmp_path / "w.csv", verbose=False)
    assert r["late"] == 0 and r["soft"] == 3
    with pytest.raises(ValueError):
        (tmp_path / "x.csv").write_text("a,b\n1,2\n")
        apply_decision_rules(tmp_path / "x.csv", verbose=False)


def _roi_labels(frames):
    """Synthetic ROI predictions: ``frames`` = list of lists of (points 23x2, point_scores 23, instance_score)."""
    import sleap_io as sio
    from stride.config import STRICT_REQUIRED_23
    skel = sio.Skeleton(nodes=list(STRICT_REQUIRED_23))
    video = sio.Video(filename="synthetic.mp4", open_backend=False)
    lfs = []
    for fi, insts in enumerate(frames):
        made = [sio.PredictedInstance.from_numpy(points_data=np.asarray(p, float), skeleton=skel,
                                                point_scores=np.asarray(ps, float), score=float(sc))
                for p, ps, sc in insts]
        lfs.append(sio.LabeledFrame(video=video, frame_idx=fi * 10, instances=made))
    return sio.Labels(labeled_frames=lfs, videos=[video], skeletons=[skel])


def test_roi_best_instance_selection():
    from stride.config import STRICT_REQUIRED_23
    from stride.stages.roi_selection import REQUIRED_CORNERS_16, select_best_frame_v2
    rng = np.random.default_rng(0)
    names = list(STRICT_REQUIRED_23)
    full = rng.uniform(100, 900, (23, 2))

    def inst(missing=(), point_score=0.8, score=0.75, shift=0.0):
        p = full + shift
        for n in missing:
            p[names.index(n)] = np.nan
        return p, np.full(23, point_score), score

    spurious = inst(missing=["segment1.top_left", "segment1.bottom_left", "segment2.top_left"], point_score=0.3,
                    score=0.27, shift=40.0)
    # frame 0: the spurious instance comes FIRST, the real maze second -> the real one must be chosen
    # frame 1: only the spurious instance; frame 2: real maze with a lower point score
    labels = _roi_labels([[spurious, inst()], [spurious], [inst(point_score=0.6)]])
    idx, lf = select_best_frame_v2(labels, max_check=10)
    assert idx == 0 and len(lf.instances) == 1
    assert np.allclose(lf.instances[0].numpy(), full)

    # the first-instance rule would have rejected frames 0 and 1 (the old selector's behaviour)
    first_only = _roi_labels([[spurious, inst()], [spurious]])
    first_only.labeled_frames[0].instances = first_only.labeled_frames[0].instances[:1]
    assert select_best_frame_v2(first_only, max_check=10)[1] is None

    # strict (23 nodes) beats corners-only even when the corners-only frame scores higher
    corners_only = inst(missing=["junction.centroid"], point_score=0.99, score=0.99)
    labels = _roi_labels([[corners_only], [inst(point_score=0.5, score=0.5)]])
    idx, lf = select_best_frame_v2(labels, max_check=10)
    assert idx == 1
    # corners-only is accepted when no frame is strict; a missing corner is never accepted
    idx, lf = select_best_frame_v2(_roi_labels([[corners_only]]), max_check=10)
    assert idx == 0 and lf is not None
    assert all(n in names for n in REQUIRED_CORNERS_16)
    idx, lf = select_best_frame_v2(_roi_labels([[inst(missing=["arm_left.bottom_left"])]]), max_check=10)
    assert lf is None
    # max_check limits the frames looked at; None looks at all (the full-video pass)
    labels = _roi_labels([[spurious], [spurious], [inst()]])
    assert select_best_frame_v2(labels, max_check=2)[1] is None
    assert select_best_frame_v2(labels, max_check=None)[0] == 2


def test_roi_sample_frames_and_conversion(core, tmp_path):
    from stride.stages.roi_selection import get_sample_frames
    from stride.stages.slp_to_yaml import slp_to_roi_yaml
    v = sorted(Path(core["video_dir"]).glob("*.mp4"))[0]
    frames = get_sample_frames(v)
    n = pd.read_csv(core["truth_csv"]).set_index("stem").loc[v.stem, "n_frames"]
    assert len(frames) == 10 and frames[0] == int(n * 0.2) and frames[-1] < int(n * 0.8)
    assert get_sample_frames(tmp_path / "not a video.mp4") == list(range(10))
    # an ROI prediction converts to the same polygons the mock set ships
    slp = sorted(Path(core["roi_slp_dir"]).glob("*.slp"))[0]
    out = tmp_path / "roi.yml"
    slp_to_roi_yaml(str(slp), str(out))
    made = yaml.safe_load(out.read_text())
    ref = yaml.safe_load((Path(core["yml_dir"]) / (slp.stem + ".yml")).read_text())
    assert [r["name"] for r in made["rois"]] == [r["name"] for r in ref["rois"]]
    assert np.allclose([r["coordinates"] for r in made["rois"]], [r["coordinates"] for r in ref["rois"]])


def test_consensus_port(core, tmp_path):
    from stride.stages.roi_consensus import ConsensusError, consensus, load_donors, propagate_consensus
    ymls = sorted(Path(core["yml_dir"]).glob("*.yml"))[:3]
    docs = load_donors(ymls)
    # jitter the donors: the consensus is the per-vertex median, the spread is hypot(std_x, std_y)
    for k, d in enumerate(docs.values()):
        for r in d["rois"]:
            r["coordinates"] = [[x + (k - 1) * 2.0, y + (k - 1) * 1.0] for x, y in r["coordinates"]]
    ref = yaml.safe_load(ymls[0].read_text())
    cons, smax, smean = consensus(docs)
    assert np.allclose([r["coordinates"] for r in cons["rois"]], [r["coordinates"] for r in ref["rois"]])
    expected = float(np.hypot(np.std([-2.0, 0.0, 2.0]), np.std([-1.0, 0.0, 1.0])))
    assert abs(smax - expected) < 1e-9 and abs(smean - expected) < 1e-9
    with pytest.raises(ConsensusError):
        propagate_consensus([tmp_path / "x.mp4"], [], lambda v: tmp_path / "x.rois.yml", tmp_path / "qc", verbose=False)
