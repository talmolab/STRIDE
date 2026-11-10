"""Regression tests on the synthetic T-maze mock set (tests/mock/make_mock_dataset.py).

The mock set has no animal video and needs no model or GPU: each clip is a generated pose file,
a blank video and an ROI file, with the answers the generator planted in ``truth.csv``.

    pytest tests                         # generates what it needs into a temp folder (~1-2 min)
    STRIDE_MOCK_DATA=tests/mock/_data pytest tests     # reuse an already generated set

What is asserted
  * every `stride` subcommand and every `run_all.py` path exits 0 on the supported cases;
  * broken inputs exit non-zero AND print the reason;
  * the planted arm choices are recovered; a snout tracking dropout inside a probe gives 1 probe;
  * every stage leaves its run manifest; the stage outputs are chained through --output.

Nothing here checks a metric definition: these tests are about running and logging.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
STRIDE = [PY, "-m", "stride.cli"]
SCENARIOS = ["core", "batch_wt_tau", "behaviour", "layout_per_day_colocated", "layout_custom_pose_suffix",
             "pose_snout_dropout", "meta_missing_rows"]


def run(cmd, cwd=None, timeout=900):
    env = dict(os.environ, PYTHONWARNINGS="ignore", MPLBACKEND="Agg", COLUMNS="200", NO_COLOR="1")
    p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout + "\n" + p.stderr


@pytest.fixture(scope="session")
def mock(tmp_path_factory) -> dict:
    """Scenario name -> scenario.json content. Generated once per session (or reused)."""
    reuse = os.environ.get("STRIDE_MOCK_DATA")
    if reuse and all((Path(reuse) / n / "scenario.json").exists() for n in SCENARIOS):
        root = Path(reuse).resolve()
    else:
        root = tmp_path_factory.mktemp("stride_mock")
        code, out = run([PY, REPO / "tests" / "mock" / "make_mock_dataset.py", "--out", root,
                         "--only", ",".join(SCENARIOS)], timeout=1800)
        assert code == 0, out[-2000:]
    return {n: json.loads((root / n / "scenario.json").read_text()) for n in SCENARIOS}


@pytest.fixture(scope="session")
def core_run(mock, tmp_path_factory) -> Path:
    """`python run_all.py --config <README-style config> --batch <key>` on the core scenario."""
    out = tmp_path_factory.mktemp("core_run") / "out"
    code, text = run([PY, REPO / "run_all.py", "--config", mock["core"]["config_readme"], "--output", out,
                      "--batch", mock["core"]["batch_key"]])
    assert code == 0, text[-3000:]
    assert "All stages completed successfully." in text
    return out


# --------------------------------------------------------------------------------------------
# launch: help everywhere
# --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("sub", [[], ["run"], ["pose-inference"], ["check-distortion"], ["roi-inference"],
                                 ["convert-roi"], ["analyze-decisions"], ["analyze-gait"], ["count-frames"],
                                 ["status"]])
def test_cli_help(sub):
    code, out = run(STRIDE + sub + ["--help"])
    assert code == 0 and "Usage" in out


@pytest.mark.parametrize("mod", ["run_inference", "run_tmaze", "run_gait_extraction", "gait_lmm",
                                 "classification", "safety_checks"])
def test_stage_module_help(mod):
    code, out = run([PY, "-m", f"stride.stages.{mod}", "--help"])
    assert code == 0 and "usage" in out.lower()


def test_run_all_help():
    code, out = run([PY, REPO / "run_all.py", "--help"])
    assert code == 0 and "--batch" in out and "--stages" in out


# --------------------------------------------------------------------------------------------
# the full pipeline, README-style config (only the fields a user provides)
# --------------------------------------------------------------------------------------------
def test_run_all_recovers_planted_decisions(mock, core_run):
    b = core_run / mock["core"]["batch_key"]
    dec = pd.read_csv(b / "decisions" / "decisions.csv")
    truth = pd.read_csv(mock["core"]["truth_csv"])
    m = truth.merge(dec, on="stem", suffixes=("_t", ""))
    assert len(m) == len(truth) == 52
    ent = m[m.planted_choice.notna() & (m.planted_choice != "")]
    assert (ent.entry_side == ent.planted_choice).all()
    assert (ent.correct_bin.astype(float) == ent.expected_correct.astype(float)).all()


def test_run_all_chains_stage_outputs(mock, core_run):
    b = core_run / mock["core"]["batch_key"]
    for rel in ("decisions/decisions.csv", "decisions/events.csv", "decisions/metrics.csv", "metrics_meta.csv",
                "learning_cycle_1.csv", "gait_per_stride.csv", "gait_per_step.csv", "gait_per_video.csv",
                "gait_per_stride_filtered.csv", "pose_keypoint_confidence_by_video.csv"):
        assert (b / rel).exists(), rel
    for rel in ("lmm_results_all.csv", "lmm_fit_log.csv", "heatmap_M1_mock_core.png",
                "regression_feature_ranking.csv", "regression_roc_curves.png"):
        assert (core_run / rel).exists(), rel
    raw, filt = pd.read_csv(b / "gait_per_stride.csv"), pd.read_csv(b / "gait_per_stride_filtered.csv")
    assert 0 < len(filt) < len(raw)
    assert list(filt.columns) == list(raw.columns)          # the filtered file keeps `stem`
    assert set(filt.stem) <= set(raw.stem)


def test_run_all_writes_every_manifest(mock, core_run):
    b = core_run / mock["core"]["batch_key"]
    expected = [b / "inference_run_manifest.yaml", b / "decisions" / "decisions_run_manifest.yaml",
                b / "tmaze_run_manifest.yaml", b / "gait_run_manifest.yaml", b / "gait_filtering_run_manifest.yaml",
                core_run / "lmm_run_manifest.yaml", core_run / "classification_run_manifest.yaml"]
    for p in expected:
        assert p.exists(), p
        d = yaml.safe_load(p.read_text())
        assert d["stage"] and d["timestamp_utc"] and "git" in d and "environment" in d
    code, out = run(STRIDE + ["status", "--output", core_run])
    assert code == 0
    for stage in ("decisions", "gait", "gait_filtering", "lmm", "classification"):
        assert stage in out


def test_lmm_table_has_fdr_and_fit_log(core_run):
    res = pd.read_csv(core_run / "lmm_results_all.csv")
    assert {"Tau_pval", "Tau_fdr", "APP_pval", "APP_fdr"} <= set(res.columns)
    assert ((res.Tau_fdr >= res.Tau_pval - 1e-12) | res.Tau_pval.isna()).all()
    log = pd.read_csv(core_run / "lmm_fit_log.csv")
    assert (log.status == "ok").sum() == len(res)
    # the mock plants a shorter stride in Tau: the pipeline must run far enough to see it
    row = res[(res.model == "M1") & (res.variable == "stride_length_euclid_cm")].iloc[0]
    assert row.Tau_coef < 0 and row.Tau_pval < 0.05


def test_two_genotype_batch_runs_the_mixed_models(mock, tmp_path):
    """A WT + Tau batch (no APP mice) must fit; the unused level used to make every fit singular."""
    out = tmp_path / "out"
    code, text = run([PY, REPO / "run_all.py", "--config", mock["batch_wt_tau"]["config_readme"], "--output", out,
                      "--stages", "3,4-6,7-9"])
    assert code == 0, text[-3000:]
    res = pd.read_csv(out / "lmm_results_all.csv")
    assert len(res) > 0 and "Tau_pval" in res.columns and res.Tau_pval.notna().any()


def test_stages_one_at_a_time(mock, tmp_path):
    cfg, key, out = mock["core"]["config_readme"], mock["core"]["batch_key"], tmp_path / "out"
    for st in ("1-2", "3", "4-6", "7-9", "10"):
        code, text = run([PY, REPO / "run_all.py", "--config", cfg, "--output", out, "--stages", st, "--batch", key])
        assert code == 0, f"stage {st}: {text[-2500:]}"


# --------------------------------------------------------------------------------------------
# single commands
# --------------------------------------------------------------------------------------------
def test_analyze_decisions_cli(mock, tmp_path):
    c = mock["core"]
    out = tmp_path / "dec"
    code, text = run(STRIDE + ["analyze-decisions", "--videos", c["video_dir"], "--yml", c["yml_dir"],
                               "--meta", c["meta_trials_csv"], "--output", out, "-w", "4"])
    assert code == 0, text[-2000:]
    assert len(pd.read_csv(out / "decisions.csv")) == 52
    assert (out / "decisions_run_manifest.yaml").exists()


def test_per_day_folders_upper_case_and_colocated_roi(mock, tmp_path):
    s = mock["layout_per_day_colocated"]
    code, text = run(STRIDE + ["analyze-decisions", "--videos", s["video_dir"], "--meta", s["meta_trials_csv"],
                               "--output", tmp_path / "dec"])          # no --yml: ROI files next to the videos
    assert code == 0, text[-2000:]
    assert len(pd.read_csv(tmp_path / "dec" / "decisions.csv")) == 4
    code, text = run(STRIDE + ["count-frames", "-i", s["video_dir"]])
    assert code == 0 and "Videos: 4" in text
    code, text = run(STRIDE + ["check-distortion", "-i", s["video_dir"], "-n", "2"])
    assert code == 0 and text.count("NO BOARD") == 4
    code, text = run([PY, "-m", "stride.stages.safety_checks", "--roi-coverage", s["video_dir"], "--expect-count", "4"])
    assert code == 0 and "all 4 videos" in text


def test_roi_coverage_accepts_a_separate_yml_folder(mock):
    c = mock["core"]
    code, text = run([PY, "-m", "stride.stages.safety_checks", "--roi-coverage", c["video_dir"], "--yml-dir", c["yml_dir"]])
    assert code == 0 and "all 52 videos" in text
    code, text = run([PY, "-m", "stride.stages.safety_checks", "--roi-coverage", c["video_dir"]])
    assert code == 2 and "--yml-dir" in text


def test_custom_pose_suffix(mock, tmp_path):
    s = mock["layout_custom_pose_suffix"]
    base = STRIDE + ["analyze-decisions", "--videos", s["video_dir"], "--meta", s["meta_trials_csv"]]
    yml = [] if s["roi_mode"] == "colocated" else ["--yml", s["yml_dir"]]
    code, text = run(base + yml + ["--output", tmp_path / "a", "--pose-suffix", s["pose_suffix"]])
    assert code == 0, text[-2000:]
    assert len(pd.read_csv(tmp_path / "a" / "decisions.csv")) == s["n_clips"]
    # without the suffix nothing can be analysed: non-zero exit and the reason
    code, text = run(base + yml + ["--output", tmp_path / "b"])
    assert code != 0 and "no pose file" in text
    # the config-driven path takes the suffix from the batch
    out = tmp_path / "out"
    code, text = run([PY, REPO / "run_all.py", "--config", s["config_readme"], "--output", out, "--stages", "3,4-6"])
    assert code == 0, text[-2500:]
    assert len(pd.read_csv(out / s["batch_key"] / "decisions" / "decisions.csv")) == s["n_clips"]
    assert (out / s["batch_key"] / "gait_per_stride.csv").exists()


def test_snout_dropout_inside_a_probe_counts_one_probe(mock, tmp_path):
    s = mock["pose_snout_dropout"]
    yml = [] if s["roi_mode"] == "colocated" else ["--yml", s["yml_dir"]]
    code, text = run(STRIDE + ["analyze-decisions", "--videos", s["video_dir"], "--meta", s["meta_trials_csv"],
                               "--output", tmp_path / "dec"] + yml)
    assert code == 0, text[-2000:]
    ev = pd.read_csv(tmp_path / "dec" / "events.csv").merge(pd.read_csv(s["truth_csv"]), on="stem")
    assert len(ev) == 2
    assert (ev.probes_R == ev.planted_probes_R).all() and (ev.probes_R == 1).all()
    assert (ev.probes_L == ev.planted_probes_L).all()


def test_metadata_mismatches_are_reported_not_rescored(mock, tmp_path):
    s = mock["meta_missing_rows"]
    yml = [] if s["roi_mode"] == "colocated" else ["--yml", s["yml_dir"]]
    code, text = run(STRIDE + ["analyze-decisions", "--videos", s["video_dir"], "--meta", s["meta_trials_csv"],
                               "--output", tmp_path / "dec"] + yml)
    assert code == 0, text[-2000:]
    assert "WARNING" in text and "no L/R reward row" in text
    dec = pd.read_csv(tmp_path / "dec" / "decisions.csv")
    no_reward = dec[dec.reward.isna()]
    assert len(no_reward) > 0 and (no_reward.correct_bin == 0).all()     # scoring itself is unchanged


def test_stride_run_on_existing_pose_and_roi(mock, tmp_path):
    c = mock["core"]
    out = tmp_path / "out"
    code, text = run(STRIDE + ["run", "-i", c["video_dir"], "-o", out, "-m", c["meta_trials_csv"], "--yml", c["yml_dir"],
                               "--skip-distortion", "--skip-pose", "--skip-roi", "-w", "4"])
    assert code == 0, text[-2000:]
    assert len(pd.read_csv(out / "decisions" / "decisions.csv")) == 52
    assert (out / "pipeline_run_manifest.yaml").exists()


def test_library_functions_write_manifests_and_keep_stem(mock, tmp_path):
    from stride.stages.decision_analysis import run_decision_analysis
    from stride.stages.gait_extraction import compute_keypoint_confidence, run_gait_extraction
    from stride.stages.run_gait_extraction import build_params, filter_strides

    s = mock["behaviour"]
    cfg = yaml.safe_load(Path(s["config_readme"]).read_text())
    yml_dir = None if s["roi_mode"] == "colocated" else Path(s["yml_dir"])
    dec = tmp_path / "decisions"
    res = run_decision_analysis(video_dir=Path(s["video_dir"]), yml_dir=yml_dir, meta_csv=Path(s["meta_trials_csv"]),
                                output_dir=dec, n_workers=1)
    assert res["ok"] == res["total"] == s["n_clips"]
    assert (dec / "decisions_run_manifest.yaml").exists()

    gait = tmp_path / "gait"
    gait.mkdir()
    me = pd.read_csv(dec / "metrics.csv", dtype={"mouse": str})
    me["Gender"], me["Genotype"] = "F", "WT"
    me.to_csv(gait / "metrics_meta.csv", index=False)
    params = build_params(cfg)
    r = run_gait_extraction(meta_csv=gait / "metrics_meta.csv", yml_dir=yml_dir, output_dir=gait, params=params,
                            column_map={"mouse_id": ["mouse"], "mouse_id_sep": "", "sex": "Gender"}, n_workers=2)
    assert r["ok"] == s["n_clips"] and r["errors"] == 0
    assert (gait / "gait_run_manifest.yaml").exists()
    conf = gait / "pose_keypoint_confidence_by_video.csv"
    compute_keypoint_confidence(video_dir=Path(s["video_dir"]), pose_slp_suffix=".slp", output_path=conf)
    n = filter_strides(gait / "gait_per_stride.csv", conf, gait / "gait_per_stride_filtered.csv", cfg)
    assert (gait / "gait_filtering_run_manifest.yaml").exists()
    filt = pd.read_csv(gait / "gait_per_stride_filtered.csv")
    assert len(filt) == n and "stem" in filt.columns


# --------------------------------------------------------------------------------------------
# broken inputs: non-zero exit and the reason
# --------------------------------------------------------------------------------------------
def test_inference_without_models_says_what_to_pass(mock, tmp_path):
    v = mock["behaviour"]["video_dir"]
    code, text = run(STRIDE + ["pose-inference", "-i", v])
    assert code != 0 and "No pose model given" in text and "--model" in text and "Traceback" not in text
    code, text = run(STRIDE + ["roi-inference", "-i", v, "-o", tmp_path / "roi"])
    assert code != 0 and "No ROI model given" in text and "Traceback" not in text
    code, text = run(STRIDE + ["run", "-i", v, "-o", tmp_path / "out"])
    assert code != 0 and "--pose-model" in text and "Traceback" not in text


def test_inference_without_sleap_nn_gives_one_clear_line(mock, tmp_path):
    import importlib.util
    if importlib.util.find_spec("sleap_nn") is not None:
        pytest.skip("sleap-nn is installed in this environment")
    model = tmp_path / "dummy_model"
    model.mkdir()
    v = mock["behaviour"]["video_dir"]
    code, text = run(STRIDE + ["pose-inference", "-i", v, "-m", model, "-o", tmp_path / "pose", "--device", "cpu"])
    assert code != 0 and "sleap-nn is not installed" in text and "Traceback" not in text
    code, text = run(STRIDE + ["roi-inference", "-i", v, "-m", model, "-o", tmp_path / "roi"])
    assert code != 0 and "sleap-nn is not installed" in text and "Traceback" not in text


def test_missing_config_and_missing_inputs(mock, tmp_path):
    code, text = run([PY, REPO / "run_all.py", "--config", tmp_path / "nope.yaml"])
    assert code != 0 and "config file not found" in text
    for mod in ("run_inference", "run_tmaze", "run_gait_extraction", "gait_lmm", "classification"):
        code, text = run([PY, "-m", f"stride.stages.{mod}"], cwd=tmp_path)       # no ./config.yaml here
        assert code != 0 and "config file not found" in text and "Traceback" not in text, mod
    cfg = mock["core"]["config_readme"]
    empty = tmp_path / "empty_out"
    code, text = run([PY, "-m", "stride.stages.run_gait_extraction", "--config", cfg, "--output", empty])
    assert code == 1 and "stage 3" in text and "Traceback" not in text
    code, text = run([PY, "-m", "stride.stages.gait_lmm", "--config", cfg, "--output", empty])
    assert code == 1 and "stage 4-6" in text and "Traceback" not in text
    code, text = run([PY, "-m", "stride.stages.classification", "--config", cfg, "--output", empty])
    assert code == 1 and "Stage 10 FAILED" in text and "Traceback" not in text
    code, text = run([PY, REPO / "run_all.py", "--config", cfg, "--output", empty, "--stages", "4-6,7-9"])
    assert code != 0 and "Stage 4-6 failed" in text and "7-9" not in text.split("PIPELINE SUMMARY")[-1]


def test_empty_folder_and_unknown_inputs(tmp_path):
    (tmp_path / "v").mkdir()
    code, text = run(STRIDE + ["count-frames", "-i", tmp_path / "v"])
    assert code != 0 and "no .mp4" in text
    code, text = run(STRIDE + ["check-distortion", "-i", tmp_path / "v"])
    assert code != 0 and "no .mp4" in text
    code, text = run([PY, "-m", "stride.stages.safety_checks", "--gait", tmp_path / "nope.csv"])
    assert code == 2 and "not found" in text and "Traceback" not in text
    code, text = run(STRIDE + ["status", "--output", tmp_path / "v"])
    assert code != 0 and "No run manifests" in text


# --------------------------------------------------------------------------------------------
# small unit checks of the shared plumbing
# --------------------------------------------------------------------------------------------
def test_find_videos_is_recursive_and_case_insensitive(tmp_path):
    from stride.utils.parallel import find_videos
    for rel in ("a.mp4", "DAY1/b.MP4", "DAY2/sub/c.Mp4", "DAY2/._c.mp4", "notes.txt", "d.avi"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"")
    assert [p.name for p in find_videos(tmp_path, extensions=(".mp4",))] == ["b.MP4", "c.Mp4", "a.mp4"]
    assert len(find_videos(tmp_path)) == 4                       # + the .avi
    assert find_videos(tmp_path / "a.mp4") == [tmp_path / "a.mp4"]


def test_manifest_fingerprints_the_checkpoint_not_the_folder(tmp_path):
    from stride.run_manifest import write_run_manifest
    model = tmp_path / "pose_model" / "centroid"
    model.mkdir(parents=True)
    (model / "best.ckpt").write_bytes(b"x" * 12345)
    (model / "training_config.yaml").write_text("a: 1\n")
    out = write_run_manifest(tmp_path / "out", "pose_inference", models=[model])
    m = yaml.safe_load(out.read_text())["models"][0]
    assert m["exists"] and m["size_bytes"] == 12345 + 5
    assert m["files"][0] == {"name": "best.ckpt", "size_bytes": 12345, "modified_utc": m["files"][0]["modified_utc"]}


def test_example_config_has_every_section_the_stages_read():
    cfg = yaml.safe_load((REPO / "config.example.yaml").read_text())
    for section in ("batches", "gait_extraction", "stride_detection", "stride_filtering", "metrics", "covariates",
                    "models", "statistics", "classification"):
        assert section in cfg, section
    assert "/home/" not in (REPO / "config.example.yaml").read_text()
    assert len(cfg["metrics"]) == 14 and cfg["covariates"]["speed"] == "stride_speed_cm_s_mean"
