"""Tests of `stride-lab` end to end on synthetic raw cohorts (lab/tests/make_mock_cohort.py).

Everything is generated into a temporary folder; no animal video, no lab paths. Inference is not run: the cohorts
carry synthetic pose and ROI files (from the public mock generator), as an earlier inference would have left them.
    pip install -e . && pip install -e lab/ && pytest lab/tests
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
pytest.importorskip("sleap_io")
LAB = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(LAB), str(Path(__file__).parent)]
import make_mock_cohort  # noqa: E402

if not (make_mock_cohort.REPO / "tests" / "mock" / "make_mock_dataset.py").exists():
    pytest.skip("the public mock generator (tests/mock/make_mock_dataset.py) is not available", allow_module_level=True)

from stride_lab import calibration as cal  # noqa: E402
from stride_lab.cohort import load_cohort_config  # noqa: E402
from stride_lab.intake._log import IntakeError  # noqa: E402
from stride_lab.pipeline import build_analysis_metadata  # noqa: E402


def lab(*args, expect=None):
    r = subprocess.run([sys.executable, "-m", "stride_lab", *map(str, args)], capture_output=True, text=True)
    text = r.stdout + r.stderr
    if expect is not None:
        assert r.returncode == expect, f"exit {r.returncode}, expected {expect}:\n{text[-3000:]}"
    return r.returncode, text


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="session")
def clip_cache(tmp_path_factory):
    return tmp_path_factory.mktemp("clip_cache")


@pytest.fixture
def make(tmp_path, clip_cache):
    """make(**cohort options, config={extra config}) -> (config path, cohort folder, output folder)."""
    def _make(config=None, **kw):
        root = tmp_path / "cohort"
        make_mock_cohort.make_cohort(root, cache_dir=clip_cache, **kw)
        cfg = dict(cohort="mock", raw_dir=str(root), output_dir=str(tmp_path / "out"), verifier="pytest",
                   analysis=dict(workers=2), inference=dict(skip=True))
        for k, v in (config or {}).items():
            cfg[k] = {**cfg[k], **v} if isinstance(v, dict) and isinstance(cfg.get(k), dict) else v
        import yaml
        p = tmp_path / "cohort.yaml"
        p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return p, root, tmp_path / "out"
    return _make


def to_signed(cfg, days="1-3"):
    """First run stops at the whiteboard gate (exit 2, nothing renamed); then the boards are signed off."""
    rc, text = lab("run", "--config", cfg, "--apply", expect=2)
    assert "whiteboard sign-off needed" in text
    lab("verify", "--config", cfg, "--days", days, "--verifier", "pytest", expect=0)


# ------------------------------------------------------------------------------------------------- end to end
def test_normal_cohort_end_to_end(make):
    cfg, root, out = make()
    truth = rows(root / "truth.csv")
    rc, text = lab("run", "--config", cfg, "--apply", expect=2)                  # stops at the gate
    assert not list(root.glob("*-DAY*/Day*_*.MP4")), "nothing may be renamed before the sign-off"
    assert not (out / "decisions").exists()
    lab("verify", "--config", cfg, "--days", "1-3", "--verifier", "pytest", expect=0)
    rc, text = lab("run", "--config", cfg, "--apply", expect=0)                  # resumes and finishes
    # names: the lab pattern, exactly the planted ones, sidecars carried along
    final = {p.name for p in root.glob("*-DAY*/*.MP4")}
    assert final == {t["final_name"] for t in truth}
    assert not list(root.glob("*-DAY*/DJI_*")) and not list(root.glob("DJI_*"))
    for t in truth:
        d = next(root.glob(f"*-DAY{t['day']}"))
        assert (d / t["final_name"]).with_suffix(".slp").exists() and (d / t["final_name"]).with_suffix(".rois.yml").exists()
    # tables: the planted arm choices come back, correctness follows the export's reward
    dec = {r["stem"]: r for r in rows(out / "decisions" / "decisions_clean.csv")}
    assert len(dec) == len(truth)
    for t in truth:
        r = dec[Path(t["final_name"]).stem]
        assert r["entry_side"] == t["planted_choice"] and r["reward"] == t["reward"]
        assert int(float(r["correct_bin"])) == int(t["expected_correct"])
    gait = rows(out / "gait" / "gait_per_stride_filtered.csv")
    assert gait and {r["sex"] for r in gait} == {"F", "M"} and {r["genotype"] for r in gait} == {"WT"}
    # scale: measured per day, cohort value = the planted one
    planted = json.loads((root / "truth.json").read_text())["planted_px_per_cm"]["1"]
    days = rows(root / "_intake" / "calibration" / "per_day_px_per_cm.csv")
    assert [d["status"] for d in days] == ["ok", "ok", "ok"] and all(int(d["clips_used"]) == 3 for d in days)
    assert abs(float((root / "_intake" / "calibration" / "px_per_cm.txt").read_text()) - planted) / planted < 0.005
    assert abs(float(gait[0]["px_per_cm"]) - 1 / 26.0) < 1e-4                      # the gait stage used the cohort value
    # reports, log, manifests
    for f in ("COHORT_SUMMARY.md", "DATA_QUALITY.md", "cohort_summary.csv", "roi_position_per_day.csv", "metadata_for_stride.csv",
              "lab_pipeline_run_manifest.yaml", "tmaze_run_manifest.yaml"):
        assert (out / f).exists(), f
    assert (root / "_intake" / "calibration" / "lab_calibration_run_manifest.yaml").exists()
    log = (root / "_intake" / "intake.log").read_text()
    for stage in ("[sort]", "[boards]", "[verify]", "[rename]", "[calibrate]", "[pipeline]"):
        assert stage in log
    summ = {r["day"]: r for r in rows(out / "cohort_summary.csv")}
    assert all(summ[d]["trials_analysed"] == "4" and summ[d]["accuracy"] == "0.75" for d in ("1", "2", "3"))
    # no genotype statistics are run by the lab pipeline
    assert not list(out.rglob("lmm_results*")) and not list(out.rglob("regression_*"))
    # safe to re-run: nothing is redone, same answer
    before = (out / "decisions" / "decisions_clean.csv").read_bytes()
    rc, text = lab("run", "--config", cfg, "--apply", expect=0)
    assert "already done" in text and (out / "decisions" / "decisions_clean.csv").read_bytes() == before


def test_day_off_by_less_than_stop_level_warns_and_continues(make):
    cfg, root, out = make(day_pxcm={2: 26.18})                                   # +0.7 %
    to_signed(cfg)
    rc, text = lab("run", "--config", cfg, "--apply", expect=0)
    days = {d["day"]: d for d in rows(root / "_intake" / "calibration" / "per_day_px_per_cm.csv")}
    assert days["2"]["status"] == "warn" and days["2"]["use"] == "cohort" and 0.5 < float(days["2"]["deviation_pct"]) < 1.0
    assert "above the warning level" in (out / "DATA_QUALITY.md").read_text()
    assert len(rows(out / "decisions" / "decisions_clean.csv")) == 12              # the day is processed with the cohort value


def test_day_off_by_more_than_stop_level_is_held_until_a_decision(make):
    cfg, root, out = make(day_pxcm={3: 26.39})                                   # +1.5 %
    to_signed(cfg)
    rc, text = lab("run", "--config", cfg, "--apply", expect=2)
    assert "held by the scale check" in text
    assert {r["day"] for r in rows(out / "decisions" / "decisions_clean.csv")} == {"DAY1", "DAY2"}   # the others finished
    assert "Decision needed" in (root / "_intake" / "calibration" / "CALIBRATION.md").read_text()
    assert not (out / "own_scale").exists()
    # a decision needs a name; a recorded one is appended to the decisions file and logged
    rc, text = lab("calibrate", "--config", cfg, "--decide", "3=accept_own", "--by", "pytest", "--note", "board nudged", expect=0)
    assert rows(root / "_intake" / "calibration" / "day_decisions.csv")[-1]["decision"] == "accept_own"
    assert "decision recorded: accept_own" in (root / "_intake" / "intake.log").read_text()
    lab("run", "--config", cfg, "--apply", expect=0)
    own = rows(out / "own_scale" / "DAY3" / "gait" / "gait_per_stride.csv")
    assert own and abs(1 / float(own[0]["px_per_cm"]) - 26.38) < 0.01              # its own value
    assert {r["day"] for r in rows(out / "decisions" / "decisions_clean.csv")} == {"DAY1", "DAY2"}
    # accept with the cohort value: the day joins the cohort tables
    lab("calibrate", "--config", cfg, "--decide", "3=accept_cohort", "--by", "pytest", expect=0)
    lab("run", "--config", cfg, "--apply", expect=0)
    assert {r["day"] for r in rows(out / "decisions" / "decisions_clean.csv")} == {"DAY1", "DAY2", "DAY3"}
    # exclude: the day is left out
    lab("calibrate", "--config", cfg, "--decide", "3=exclude", "--by", "pytest", expect=0)
    lab("run", "--config", cfg, "--apply", expect=0)
    assert {r["day"] for r in rows(out / "decisions" / "decisions_clean.csv")} == {"DAY1", "DAY2"}
    summ = {r["day"]: r for r in rows(out / "cohort_summary.csv")}
    assert summ["3"]["scale_use"] == "exclude"
    assert "exclude (recorded decision" in (out / "DATA_QUALITY.md").read_text()


def test_reference_px_per_cm_off_is_a_warning_only(make):
    cfg, root, out = make(config=dict(calibration=dict(reference_px_per_cm=26.5)))
    to_signed(cfg)
    rc, text = lab("run", "--config", cfg, "--apply", expect=0)
    assert "from the reference 26.5" in text and "warning only" in text
    assert len(rows(out / "decisions" / "decisions_clean.csv")) == 12
    assert "from the reference" in (out / "DATA_QUALITY.md").read_text()


def test_maze_moved_on_one_day_gives_a_position_warning(make):
    cfg, root, out = make(maze_shift={2: (40, 0)})
    to_signed(cfg)
    rc, text = lab("run", "--config", cfg, "--apply", expect=0)
    pos = {r["day"]: r for r in rows(out / "roi_position_per_day.csv")}
    assert pos["2"]["status"] == "warn" and abs(float(pos["2"]["corner_drift_max_px"]) - 40) < 1 and pos["3"]["status"] == "ok"
    assert "maze position" in (out / "DATA_QUALITY.md").read_text()
    assert len(rows(out / "decisions" / "decisions_clean.csv")) == 12              # report only


def test_incremental_days(make):
    cfg, root, out = make()
    rc, text = lab("run", "--config", cfg, "--apply", "--days", "1", expect=2)   # sorts everything, plans day 1 only
    lab("verify", "--config", cfg, "--days", "1", "--verifier", "pytest", expect=0)
    lab("run", "--config", cfg, "--apply", "--days", "1", expect=0)
    assert {r["day"] for r in rows(out / "decisions" / "decisions_clean.csv")} == {"DAY1"}
    assert list(root.glob("*-DAY2/DJI_*.MP4")), "day 2 must be untouched"
    # the whole cohort cannot run while days still hold camera-named clips: refused with the reason
    rc, text = lab("pipeline", "--config", cfg, expect=1)
    assert "still hold camera-named clips" in text
    # the remaining days later
    lab("run", "--config", cfg, "--apply", expect=2)
    lab("verify", "--config", cfg, "--days", "2-3", "--verifier", "pytest", expect=0)
    lab("run", "--config", cfg, "--apply", expect=0)
    assert {r["day"] for r in rows(out / "decisions" / "decisions_clean.csv")} == {"DAY1", "DAY2", "DAY3"}


def test_dry_run_moves_nothing(make):
    cfg, root, out = make()
    rc, text = lab("run", "--config", cfg, expect=0)
    assert "DRY RUN" in text and len(list(root.glob("DJI_*.MP4"))) == 12 and not list(root.glob("*-DAY*"))


# ------------------------------------------------------------------------------------------------- refusals
def test_refusals_exit_non_zero_with_the_reason(make, tmp_path):
    cfg, root, out = make()
    rc, text = lab("pipeline", expect=1)
    assert "needs the cohort config" in text
    bad = tmp_path / "bad.yaml"
    bad.write_text(f"raw_dir: {root}\nnot_a_key: 1\n")
    rc, text = lab("check", "--config", bad, expect=1)
    assert "unknown key" in text and "not_a_key" in text
    rc, text = lab("check", "--config", tmp_path / "missing.yaml", expect=1)
    assert "config file not found" in text
    noout = tmp_path / "noout.yaml"
    noout.write_text(f"raw_dir: {root}\n")
    to_signed(cfg)
    lab("run", "--config", cfg, "--apply", expect=0)
    rc, text = lab("pipeline", "--config", noout, expect=1)
    assert "output_dir" in text
    rc, text = lab("calibrate", "--config", cfg, "--decide", "3=maybe", "--by", "x", expect=1)
    assert "decision must be one of" in text
    rc, text = lab("calibrate", "--config", cfg, "--decide", "9=exclude", "--by", "x", expect=1)
    assert "DAY9 is not in the calibration table" in text


def test_decision_needs_a_name(make):
    cfg, root, out = make(config=dict(verifier=""), day_pxcm={3: 26.39})
    to_signed(cfg)
    lab("run", "--config", cfg, "--apply", expect=2)
    rc, text = lab("calibrate", "--config", cfg, "--decide", "3=exclude", expect=1)
    assert "needs --by" in text


def test_no_board_anywhere_refuses_unless_a_value_is_given(make):
    cfg, root, out = make(day_pxcm={1: None, 2: None, 3: None})
    to_signed(cfg)
    rc, text = lab("run", "--config", cfg, "--apply", expect=1)
    assert "no ChArUco reading on any day" in text
    import yaml
    c = yaml.safe_load(Path(cfg).read_text())
    c["calibration"] = dict(px_per_cm=26.0)
    Path(cfg).write_text(yaml.safe_dump(c))
    rc, text = lab("run", "--config", cfg, "--apply", expect=0)                  # explicit value: runs, days reported as unmeasured
    assert "could not be checked" in (out / "DATA_QUALITY.md").read_text()
    assert len(rows(out / "decisions" / "decisions_clean.csv")) == 12


def test_placeholder_ids_refused_then_allowed(make):
    cfg, root, out = make(placeholder_ids=True)
    rc, text = lab("run", "--config", cfg, "--apply", expect=1)
    assert "placeholder" in text or "1-2 digit" in text
    rc, text = lab("run", "--config", cfg, "--apply", "--allow-placeholders", expect=2)
    lab("verify", "--config", cfg, "--days", "1-3", "--verifier", "pytest", expect=0)
    rc, text = lab("run", "--config", cfg, "--apply", expect=1)                  # refused again on every run without the override
    assert "placeholder" in text or "1-2 digit" in text
    lab("run", "--config", cfg, "--apply", "--allow-placeholders", expect=0)
    assert {r["mouse"] for r in rows(out / "decisions" / "decisions_clean.csv")} == {"1", "2"}


# --------------------------------------------------------------------------------------------- inference plan
def test_inference_plan_spreads_workers_over_the_devices(make, tmp_path):
    for m in ("pose_a", "pose_b", "roi_a", "roi_b"):
        (tmp_path / m).mkdir()
    cfg, root, out = make(config=dict(models=dict(pose=[str(tmp_path / "pose_a"), str(tmp_path / "pose_b")],
                                                  roi=[str(tmp_path / "roi_a"), str(tmp_path / "roi_b")]),
                                      inference=dict(skip=False, devices=["cuda:0", "cuda:1"], workers_per_device=2, sidecars="output")))
    to_signed(cfg)
    lab("rename", "--config", cfg, "--apply", expect=0)
    for p in list(root.glob("*-DAY2/*.slp")) + list(root.glob("*-DAY2/*.rois.yml")):   # day 2 has not been through inference
        p.unlink()
    rc, text = lab("pipeline", "--config", cfg, "--plan", expect=0)
    lines = [ln for ln in text.splitlines() if "stride.cli" in ln]
    pose = [ln for ln in lines if "pose-inference" in ln]
    roi = [ln for ln in lines if "roi-inference" in ln]
    assert len(pose) == 4 and len(roi) == 4 and all("DAY2" in ln for ln in lines)
    assert [ln.split("--device ")[1].split()[0] for ln in pose] == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]
    assert [ln.split("--worker-id ")[1].split()[0] for ln in pose] == ["0", "1", "2", "3"]
    assert all("--num-workers 4" in ln and str(tmp_path / "pose_a") in ln and str(tmp_path / "pose_b") in ln for ln in pose)
    assert all(str(out / "roi") in ln and str(tmp_path / "roi_a") in ln for ln in roi)
    assert not (out / "decisions").exists(), "a plan runs nothing"


# -------------------------------------------------------------------------------------------------- units
def test_pick_day_clips_spreads_over_the_day():
    clips = [f"c{i}" for i in range(9)]
    assert cal.pick_day_clips(clips, 3) == ["c0", "c4", "c8"]
    assert cal.pick_day_clips(clips, 1) == ["c0"]
    assert cal.pick_day_clips(clips[:2], 3) == ["c0", "c1"]
    assert cal.pick_day_clips([], 3) == []


def test_analysis_metadata_follows_the_export(tmp_path):
    tf = tmp_path / "x_trials_full.csv"
    tf.write_text("day,date,phase,mouse,genotype,sex,reward,T1\n"
                  "DAY1,2027-01-01,acq,16801,Tau,F,L,L\n"
                  "DAY1,2027-01-01,acq,16801,Tau,F,R,L\n"          # a second row for the same (day, mouse): the first wins
                  "day2,2027-01-02,acq,16801,,,r,L\n"
                  "DAY2,2027-01-02,acq,16802,WT,M,,L\n")           # no reward: no row
    m = build_analysis_metadata(tf, tmp_path / "m.csv")
    got = rows(tmp_path / "m.csv")
    assert got == [dict(day="DAY1", mouse="16801", reward="L", sex="F", genotype="Tau", trials_in_export="1"),
                   dict(day="DAY2", mouse="16801", reward="R", sex="F", genotype="Tau", trials_in_export="1")]
    assert build_analysis_metadata(tf, tmp_path / "m2.csv", genotype="KO")["rows"][0]["genotype"] == "KO"
    tf.write_text("day,date,mouse,reward\nDAY1,2027-01-01,1,L\nDAY1,2027-01-01,12,R\n")
    with pytest.raises(IntakeError, match="placeholder"):
        build_analysis_metadata(tf, tmp_path / "m3.csv")
    assert build_analysis_metadata(tf, tmp_path / "m3.csv", allow_placeholders=True)["n_rows"] == 2


def test_cohort_config_checks_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("raw_dir: /x\ncalibration: {warn_pct: 2, stop_pct: 1}\n")
    with pytest.raises(IntakeError, match="warn_pct <= stop_pct"):
        load_cohort_config(p)
    p.write_text("raw_dir: /x\ncohort_dir_typo: 1\n")
    with pytest.raises(IntakeError, match="unknown key"):
        load_cohort_config(p)
    p.write_text("cohort_dir: /x\nworkers: 6\n")                      # the older intake config still loads
    c = load_cohort_config(p)
    assert str(c["raw_dir"]) == "/x" and c["analysis"]["workers"] == 6 and c["calibration"]["warn_pct"] == 0.5
