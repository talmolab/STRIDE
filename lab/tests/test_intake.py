"""Tests of the lab intake layer (stride_lab.intake) on synthetic raw-camera cohorts (lab/tests/make_mock_raw.py).

Everything is generated into a temporary folder; no animal video, no lab paths.
    pytest lab/tests/test_intake.py
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")
LAB = Path(__file__).resolve().parents[1]                       # the lab/ folder: holds the stride_lab package
sys.path[:0] = [str(LAB), str(Path(__file__).parent)]
import make_mock_raw  # noqa: E402

from stride_lab.intake import (apply_rename, build_rename_plan, move_to_waste, signed_off, sort_days)  # noqa: E402
from stride_lab.intake._log import IntakeError, StageLog  # noqa: E402
from stride_lab.intake.boards import fill_verification  # noqa: E402
from stride_lab.intake.camera import profile_from_options  # noqa: E402
from stride_lab.intake.orchestrate import auto, boards_days, calibrate, check, day_dirs, rename_days  # noqa: E402


@pytest.fixture(scope="session")
def raw_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("mock_raw")
    for name in make_mock_raw.SCENARIOS:
        make_mock_raw.make_scenario(name, root)
    return root


@pytest.fixture
def cohort(raw_root, tmp_path):
    def _copy(name):
        dst = tmp_path / name
        shutil.copytree(raw_root / name, dst)
        return dst
    return _copy


def quiet(stage="test"):
    return StageLog(stage, quiet=True)


def truth(root: Path) -> dict:
    """raw name -> the name the clip must end up with ('' for an extra take)."""
    return {r["raw_name"]: r["true_new_name"] for r in csv.DictReader(open(root / "truth.csv"))}


def names_on_disk(root: Path) -> set:
    return {p.name for d in day_dirs(root).values() for p in d.iterdir() if p.suffix.lower() == ".mp4"}


def cli(*args):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(LAB), os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-m", "stride_lab.intake", *map(str, args)], capture_output=True, text=True, env=env)
    return r.returncode, r.stdout + r.stderr


def sorted_cohort(cohort, name, **kw):
    root = cohort(name)
    sort_days(root, apply=True, log=quiet(), **kw)
    return root


# ------------------------------------------------------------------------------------------------------ sort
def test_sort_is_a_dry_run_by_default(cohort):
    root = cohort("base")
    before = sorted(p.name for p in root.iterdir())
    r = sort_days(root, log=quiet())
    assert not r["applied"] and r["moved"] == 0 and not r["mismatches"]
    assert sorted(p.name for p in root.iterdir()) == before
    assert [(x["day"], x["n_clips"], x["n_trials"]) for x in r["rows"]] == [("DAY1", 9, 9), ("DAY2", 9, 9)]


def test_sort_apply_builds_day_folders_with_metadata(cohort):
    root = cohort("base")
    r = sort_days(root, apply=True, log=quiet())
    assert r["applied"] and r["moved"] == 18
    days = day_dirs(root)
    assert [d.name for d in days.values()] == ["2027-01-05-DAY1", "2027-01-06-DAY2"]
    for d in days.values():
        assert len([p for p in d.iterdir() if p.suffix == ".MP4"]) == 9
        assert list((d / "metadata").glob("*_trials_full.csv"))
    assert len(list(csv.DictReader(open(root / "sort_log.csv")))) == 18
    assert (root / "SORT_REPORT.md").exists() and (root / "intake_sort_run_manifest.yaml").exists()
    assert not [p for p in root.iterdir() if p.suffix == ".MP4"]          # nothing left flat


@pytest.mark.parametrize("name,delta", [("extra_clip", "+1"), ("missing_clip", "-1")])
def test_sort_refuses_a_day_that_does_not_reconcile(cohort, name, delta):
    root = cohort(name)
    with pytest.raises(IntakeError, match="does not reconcile"):
        sort_days(root, apply=True, log=quiet())
    assert not day_dirs(root)                                              # refused before anything moved
    code, out = cli("sort", "--cohort-dir", root, "--apply")
    assert code == 1 and f"delta {delta}" in out and "ERROR:" in out
    assert sort_days(root, apply=True, allow_mismatch=True, log=quiet())["applied"]


def test_sort_needs_the_recorder_export(cohort):
    root = cohort("base")
    for c in root.glob("*_trials_full.csv"):
        c.unlink()
    code, out = cli("sort", "--cohort-dir", root)
    assert code == 1 and "trials_full" in out


# ------------------------------------------------------------------------------------------------------ plan
def test_plan_reproduces_the_planted_names(cohort):
    root = sorted_cohort(cohort, "base")
    want = truth(root)
    for n, d in day_dirs(root).items():
        meta = next((d / "metadata").glob("*_trials_full.csv"))
        p = build_rename_plan(meta, f"DAY{n}", n, d, root / "_intake" / f"day{n}", log=quiet())
        assert p["perfect"] and p["delta"] == 0 and not p["leftover"]
        plan = {r["old_name"]: r["new_name"] for r in csv.DictReader(open(p["plan_csv"]))}
        assert plan == {k: v for k, v in want.items() if k in plan} and len(plan) == 9
        assert {p_.name for p_ in d.iterdir() if p_.suffix == ".MP4"} == set(plan)   # the plan never renames


def test_extra_clip_is_a_mismatch_until_it_goes_to_waste(cohort):
    root = sorted_cohort(cohort, "extra_clip", allow_mismatch=True)
    d = day_dirs(root)[1]
    meta = next((d / "metadata").glob("*_trials_full.csv"))
    D = root / "_intake" / "day1"
    p = build_rename_plan(meta, "DAY1", 1, d, D, log=quiet())
    assert not p["perfect"] and p["delta"] == 1 and len(p["leftover"]) == 1
    assert "MISMATCH" in (D / "reconciliation_report.txt").read_text()
    extra = [r["raw_name"] for r in csv.DictReader(open(root / "truth.csv")) if r["kind"] == "extra"]
    (d / (Path(extra[0]).stem + ".slp")).write_text("pose")                  # a sidecar goes with it
    move_to_waste(root, d, extra, reason="aborted short take", log=quiet())
    assert (root / "waste" / d.name / extra[0]).exists() and (root / "waste" / d.name / (Path(extra[0]).stem + ".slp")).exists()
    assert len(list(csv.DictReader(open(root / "waste" / "waste_log.csv")))) == 2
    p = build_rename_plan(meta, "DAY1", 1, d, D, log=quiet())
    plan = {r["old_name"]: r["new_name"] for r in csv.DictReader(open(p["plan_csv"]))}
    assert p["perfect"] and plan == {k: v for k, v in truth(root).items() if v}


def test_missing_clip_is_reported(cohort):
    root = sorted_cohort(cohort, "missing_clip", allow_mismatch=True)
    b = boards_days(root, log=quiet())[1]
    assert b["status"] == "pending" and not b["perfect"]
    code, out = cli("check", "--cohort-dir", root)
    assert code == 0 and "PERSON" in out                                     # a plan exists; a person must look


# ---------------------------------------------------------------------------------------------------- boards
def test_boards_finds_the_whiteboard_at_each_block_start(cohort):
    root = sorted_cohort(cohort, "base")
    b = boards_days(root, workers=2, log=quiet())
    assert b[1]["candidates"] == 3 and b[1]["crops"] == 3
    assert b[2]["candidates"] == 3 and b[2]["crops"] == 2                    # day 2: the first mouse has no board
    for n in (1, 2):
        D = root / "_intake" / f"day{n}"
        assert (D / "wb_montage.png").exists() and (D / "wb" / "whiteboard_sweep.csv").exists()
        md = (D / "BOARD_VERIFICATION.md").read_text()
        assert "VERDICT: PENDING" in md and md.count("| ____ |") == 3 and not signed_off(D / "BOARD_VERIFICATION.md")
        assert (D / "intake_boards_run_manifest.yaml").exists()
    assert names_on_disk(root) == set(truth(root))                           # still the camera names
    sweep = {r["clip"]: float(r["score"]) for r in csv.DictReader(open(root / "_intake/day2/wb/whiteboard_sweep.csv"))}
    first_clip_day2 = sorted(sweep)[0]
    assert sweep[first_clip_day2] == 0 and all(v > 0 for k, v in sweep.items() if k != first_clip_day2)


def test_a_board_shown_on_the_second_trial_is_searched_too(cohort):
    root = sorted_cohort(cohort, "board_second_trial")
    b = boards_days(root, log=quiet())[1]
    assert b["candidates"] == 5                              # 3 block starts + every clip of the mouse whose note names the board
    rows = {r["raw_name"]: r for r in csv.DictReader(open(root / "truth.csv"))}
    t1 = next(k for k, r in rows.items() if r["mouse"] == "302" and r["trial"] == "1")
    t2 = next(k for k, r in rows.items() if r["mouse"] == "302" and r["trial"] == "2")
    wb = root / "_intake" / "day1" / "wb"
    assert (wb / f"{Path(t2).stem}_wb.png").exists() and not (wb / f"{Path(t1).stem}_wb.png").exists()


def test_placeholder_ids_are_refused_unless_allowed(cohort):
    root = sorted_cohort(cohort, "placeholder_ids")
    with pytest.raises(IntakeError, match="placeholders"):
        boards_days(root, log=quiet())
    code, out = cli("boards", "--cohort-dir", root)
    assert code == 1 and "--allow-placeholders" in out
    assert boards_days(root, allow_placeholders=True, log=quiet())[1]["perfect"]


# ---------------------------------------------------------------------------------------------------- rename
def _signed_day1(cohort, name="base"):
    root = sorted_cohort(cohort, name)
    boards_days(root, days="1", log=quiet())
    return root, day_dirs(root)[1], root / "_intake" / "day1"


def test_rename_refuses_without_the_sign_off(cohort):
    root, d, D = _signed_day1(cohort)
    with pytest.raises(IntakeError, match="VERDICT: APPLY"):
        rename_days(root, days="1", apply=True, log=quiet())
    code, out = cli("rename", "--cohort-dir", root, "--days", "1", "--apply")
    assert code == 1 and "signed off" in out
    assert {p.name for p in d.iterdir() if p.suffix == ".MP4"} <= set(truth(root))


def test_rename_dry_run_then_apply_moves_sidecars_and_logs(cohort):
    root, d, D = _signed_day1(cohort)
    want = {k: v for k, v in truth(root).items() if (d / k).exists()}
    first = sorted(want)[0]
    (d / (Path(first).stem + ".slp")).write_text("pose")
    (d / (Path(first).stem + ".rois.yml")).write_text("rois")
    fill_verification(D / "BOARD_VERIFICATION.md", verifier="pytest", log=quiet())
    assert signed_off(D / "BOARD_VERIFICATION.md")
    r = rename_days(root, days="1", log=quiet())[1]                           # dry run
    assert not r["applied"] and {p.name for p in d.iterdir() if p.suffix == ".MP4"} == set(want)
    r = rename_days(root, days="1", apply=True, log=quiet())[1]
    assert r["applied"] and r["n"] == 9
    assert {p.name for p in d.iterdir() if p.suffix == ".MP4"} == set(want.values())
    new_stem = Path(want[first]).stem
    assert (d / f"{new_stem}.slp").read_text() == "pose" and (d / f"{new_stem}.rois.yml").read_text() == "rois"
    assert not list(d.glob(".__rntmp_*"))                                     # no temporary names left behind
    log = {r_["old_name"]: r_["new_name"] for r_ in csv.DictReader(open(D / "rename_log.csv"))}
    assert log == want and (D / "rename_plan.applied.csv").exists() and (D / "intake_rename_run_manifest.yaml").exists()
    assert rename_days(root, days="1", apply=True, log=quiet())[1]["status"] == "renamed"     # re-run: skipped
    with pytest.raises(IntakeError, match="looks renamed"):                 # the low-level apply refuses a renamed day
        apply_rename(D / "rename_plan.csv", d, apply=True, log=quiet())


def test_rename_refuses_collisions_and_missing_sources_before_touching_anything(cohort):
    root, d, D = _signed_day1(cohort)
    fill_verification(D / "BOARD_VERIFICATION.md", verifier="pytest", log=quiet())
    plan = D / "rename_plan.csv"
    rows = list(csv.DictReader(open(plan)))
    before = sorted(p.name for p in d.iterdir())

    def write(rs):
        with open(plan, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rs)

    write([{**rows[0]}, {**rows[1], "new_name": rows[0]["new_name"]}] + rows[2:])
    with pytest.raises(IntakeError, match="collision"):
        apply_rename(plan, d, apply=True, log=quiet())
    write([{**rows[0], "old_name": "DJI_19990101000000_0001_D.MP4"}] + rows[1:])
    with pytest.raises(IntakeError, match="missing source"):
        apply_rename(plan, d, apply=True, log=quiet())
    write(rows)
    (d / rows[0]["new_name"]).write_bytes(b"x")                               # a file already sits on a target name
    with pytest.raises(IntakeError):
        apply_rename(plan, d, apply=True, force=True, log=quiet())
    (d / rows[0]["new_name"]).unlink()
    assert sorted(p.name for p in d.iterdir()) == before                     # every refusal came before any move


def test_run_order_differs_from_metadata_only_the_boards_tell(cohort):
    root, d, D = _signed_day1(cohort, "swapped_order")
    want = truth(root)
    provisional = {r["old_name"]: r["new_name"] for r in csv.DictReader(open(D / "rename_plan.csv"))}
    assert provisional != want                                                # counts agree, names are WRONG
    code, out = cli("verify", "--cohort-dir", root, "--days", "1", "--reads", "1=302,2=301,3=303")
    assert code == 1 and "re-bind" in out and not signed_off(D / "BOARD_VERIFICATION.md")
    code, out = cli("verify", "--cohort-dir", root, "--days", "1", "--reads", "1=302,2=301,3=303", "--rebind", "--verifier", "pytest")
    assert code == 0, out
    assert {r["old_name"]: r["new_name"] for r in csv.DictReader(open(D / "rename_plan.csv"))} == want
    assert (D / "rename_plan.provisional.csv").exists() and signed_off(D / "BOARD_VERIFICATION.md")
    rename_days(root, days="1", apply=True, log=quiet())
    assert {p.name for p in d.iterdir() if p.suffix == ".MP4"} == set(want.values())


def test_rebind_refuses_an_impossible_read(cohort):
    root, d, D = _signed_day1(cohort, "swapped_order")
    for reads, why in (("1=302,2=301", "every block"), ("1=302,2=302,3=303", "read on blocks"), ("1=999,2=301,3=303", "not a mouse")):
        code, out = cli("verify", "--cohort-dir", root, "--days", "1", "--reads", reads, "--rebind")
        assert code == 1 and why in out
    assert not signed_off(D / "BOARD_VERIFICATION.md")


def test_a_signed_day_whose_clips_changed_is_refused(cohort):
    root, d, D = _signed_day1(cohort)
    fill_verification(D / "BOARD_VERIFICATION.md", verifier="pytest", log=quiet())
    stray = d / "DJI_20270105235959_0099_D.MP4"
    shutil.copy2(sorted(p for p in d.iterdir() if p.suffix == ".MP4")[0], stray)
    with pytest.raises(IntakeError, match="no longer the ones in the plan"):
        boards_days(root, days="1", log=quiet())


# ------------------------------------------------------------------------------------------------- calibrate
def test_calibration_recovers_the_planted_scale(cohort):
    root = sorted_cohort(cohort, "base")
    planted = json.load(open(root / "truth.json"))["px_per_cm"]
    r = calibrate(root, workers=2, log=quiet())
    assert abs(r["px_per_cm"] - planted) / planted < 0.005
    cal = root / "_intake" / "calibration"
    assert float((cal / "px_per_cm.txt").read_text()) == r["px_per_cm"]
    assert r["ok"] and r["n_clips_used"] >= 3 and r["spread_pct"] < 1.0
    assert list(cal.glob("*.json")) and list(cal.glob("*_checkerboard.png")) and (cal / "calibration_summary.txt").exists()
    assert (cal / "calibrate_run_manifest.yaml").exists()
    code, out = cli("calibrate", "--cohort-dir", root)
    assert code == 0 and f"px_per_cm = {planted:.2f}" in out, out


def test_calibration_fails_clearly_without_a_board_or_with_the_wrong_board(cohort):
    cam = profile_from_options(raw_pattern=make_mock_raw.SCENARIOS["other_camera"]["raw_pattern"], crop_rotation="none")
    root = cohort("other_camera")
    sort_days(root, apply=True, camera=cam, log=quiet())
    with pytest.raises(IntakeError, match="usable ChArUco reading"):
        calibrate(root, camera=cam, log=quiet())
    assert not (root / "_intake" / "calibration" / "px_per_cm.txt").exists()
    assert "NOT WRITTEN" in (root / "_intake" / "calibration" / "calibration_summary.txt").read_text()
    base = sorted_cohort(cohort, "base")
    code, out = cli("calibrate", "--cohort-dir", base, "--board-squares", "7x5", "--dictionary", "DICT_4X4_50")
    assert code == 1 and "ERROR:" in out and "usable ChArUco reading" in out
    code, out = cli("calibrate", "--cohort-dir", base, "--dictionary", "DICT_NOPE")
    assert code == 1 and "unknown ArUco dictionary" in out


# ----------------------------------------------------------------------------------------------- other camera
def test_another_camera_naming_scheme(cohort):
    spec = make_mock_raw.SCENARIOS["other_camera"]
    root = cohort("other_camera")
    opts = ["--raw-pattern", spec["raw_pattern"], "--crop-rotation", "none"]
    assert cli("sort", "--cohort-dir", root, "--apply", *opts)[0] == 0
    assert cli("boards", "--cohort-dir", root, *opts)[0] == 0
    assert cli("verify", "--cohort-dir", root, "--days", "1", "--verifier", "pytest")[0] == 0
    code, out = cli("rename", "--cohort-dir", root, "--apply", *opts)
    assert code == 0, out
    assert names_on_disk(root) == set(truth(root).values())                  # Day1_301_Trial1.mp4: the source extension is kept
    code, out = cli("sort", "--cohort-dir", cohort("base"), "--raw-pattern", "^no_named_groups$")
    assert code == 1 and "named group" in out


# ------------------------------------------------------------------------------------------------------ auto
def test_auto_is_a_dry_run_without_apply(cohort):
    root = cohort("base")
    before = sorted(p.name for p in root.iterdir())
    code, out = cli("auto", "--cohort-dir", root)
    assert code == 0 and "DRY RUN" in out
    assert sorted(p.name for p in root.iterdir() if p.name != "_intake") == before


def test_auto_stops_at_the_sign_off_and_resumes(cohort):
    root = cohort("base")
    want = truth(root)
    code, out = cli("auto", "--cohort-dir", root, "--apply", "--workers", "2")
    assert code == 2 and "WAITING" in out and "STOP" in out
    assert names_on_disk(root) == set(want)                                  # sorted into days, NOT renamed
    assert check(root, quiet=True)["days"][0]["next"].startswith("PERSON")
    code, out = cli("auto", "--cohort-dir", root, "--apply")                 # still waiting: nothing changes
    assert code == 2 and names_on_disk(root) == set(want)
    assert cli("verify", "--cohort-dir", root, "--days", "1-2", "--verifier", "pytest")[0] == 0
    code, out = cli("auto", "--cohort-dir", root, "--apply", "--workers", "2")
    assert code == 0 and "INTAKE COMPLETE" in out, out
    assert names_on_disk(root) == set(want.values())
    planted = json.load(open(root / "truth.json"))["px_per_cm"]
    assert abs(float((root / "_intake/calibration/px_per_cm.txt").read_text()) - planted) / planted < 0.005
    st = check(root, quiet=True)
    assert all(d["next"] == "ready for inference" for d in st["days"]) and st["px_per_cm"]
    code, out = cli("auto", "--cohort-dir", root, "--apply")                 # re-run: every stage skips
    assert code == 0 and "already renamed" in out and "calibration present" in out
    log = (root / "_intake" / "intake.log").read_text()
    assert all(f"[{s}]" in log for s in ("sort", "boards", "verify", "rename", "calibrate", "auto"))
    assert (root / "_intake" / "intake_auto_run_manifest.yaml").exists()


def test_fixed_run_order_skip_is_explicit_and_loud(cohort):
    root = cohort("base")                                                     # boards at every block start but day 2's first
    code, out = cli("auto", "--cohort-dir", root, "--apply", "--fixed-run-order", "--skip-calibration")
    assert code == 0 and "FIXED-RUN-ORDER SKIP" in out and "NOT read" in out, out
    assert names_on_disk(root) == set(truth(root).values())
    md = (root / "_intake" / "day1" / "BOARD_VERIFICATION.md").read_text()
    assert "NUMBERS NOT READ" in md and "not read" in md and signed_off(root / "_intake" / "day1" / "BOARD_VERIFICATION.md")
    root = cohort("board_second_trial")                                       # block 2 starts without a board: no skip
    code, out = cli("auto", "--cohort-dir", root, "--apply", "--fixed-run-order")
    assert code == 2 and "skip NOT possible" in out
    root = cohort("swapped_order")                                            # the skip's blind spot, stated in the docs:
    code, out = cli("auto", "--cohort-dir", root, "--apply", "--fixed-run-order", "--skip-calibration")
    applied = {r["old_name"]: r["new_name"] for r in csv.DictReader(open(root / "_intake" / "day1" / "rename_log.csv"))}
    assert code == 0 and applied != truth(root)                              # counts + presence cannot see a swap


def test_check_on_a_missing_folder_fails(tmp_path):
    code, out = cli("check", "--cohort-dir", tmp_path / "nope")
    assert code == 1 and "not found" in out


# ---------------------------------------------------------------------------------------------------- config
def test_config_file_supplies_the_defaults(cohort, tmp_path):
    root = cohort("other_camera")
    spec = make_mock_raw.SCENARIOS["other_camera"]
    cfg = tmp_path / "intake.yaml"
    cfg.write_text(f"cohort_dir: {root}\nrecords_dir: {tmp_path / 'records'}\nverifier: pytest\n"
                   f"camera:\n  raw_pattern: '{spec['raw_pattern']}'\n  crop_rotation: none\n")
    assert cli("--config", cfg, "sort", "--apply")[0] == 0
    assert cli("--config", cfg, "boards")[0] == 0
    assert (tmp_path / "records" / "day1" / "wb_montage.png").exists() and not (root / "_intake").exists()
    assert cli("--config", cfg, "verify", "--days", "1")[0] == 0
    assert "Verifier: pytest" in (tmp_path / "records" / "day1" / "BOARD_VERIFICATION.md").read_text()
    assert cli("--config", cfg, "rename", "--apply")[0] == 0
    assert names_on_disk(root) == set(truth(root).values())
    bad = tmp_path / "bad.yaml"
    bad.write_text("cohort_dir: x\nmodel_dir: /somewhere\n")
    code, out = cli("--config", bad, "check")
    assert code == 1 and "unknown key" in out
