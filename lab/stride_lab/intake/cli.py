"""Lab intake command line: `python -m stride_lab.intake <command> ...`

    check | sort | boards | verify | rename | waste | calibrate | auto

Attachable to another click command line with `main.add_command(intake)`.
`--config FILE` (before the command) supplies defaults for every option from a YAML file.

Exit codes: 0 = done; 1 = a stage refused or failed (the reason is printed); 2 = `auto` stopped at the
whiteboard sign-off (nothing is wrong: a person has to read the boards, then re-run).
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import click
from stride.stages.scale_calibration import CalibrationError, parse_board

from ._log import IntakeError, StageLog
from .camera import profile_from_options


def _camera_options(f):
    f = click.option("--raw-pattern", default=None, help="Regex of a raw clip name with named groups `date` (YYYYMMDD) and `seq`. "
                     "Default: DJI names, DJI_YYYYMMDDhhmmss_NNNN_D.MP4.")(f)
    f = click.option("--extensions", default=None, help="Comma-separated video suffixes (case-insensitive). Default: mp4.")(f)
    f = click.option("--crop-rotation", type=click.Choice(["ccw90", "cw90", "180", "none"]), default=None,
                     help="How whiteboard crops are turned so the number reads upright. Default: ccw90.")(f)
    return f


def _cohort_options(f):
    f = click.option("--cohort-dir", required=True, type=click.Path(file_okay=False, path_type=Path),
                     help="The cohort folder: raw clips + the recorder export, or its <DATE>-DAY{N} folders.")(f)
    f = click.option("--records-dir", default=None, type=click.Path(file_okay=False, path_type=Path),
                     help="Where plans, crops, sign-offs, logs and the calibration go. Default: <cohort-dir>/_intake.")(f)
    return f


def _days_option(f):
    return click.option("--days", default=None, help='Days to work on: "1,2" | "1-3" | "DAY1" | "2026-10-14-DAY1". '
                        "Default: every DAY >= 1 folder on disk.")(f)


def _board_options(f):
    f = click.option("--board-squares", default="14x9", show_default=True, help="ChArUco squares, X x Y.")(f)
    f = click.option("--square-mm", default=20.0, show_default=True, help="Square side, mm.")(f)
    f = click.option("--marker-mm", default=15.0, show_default=True, help="Marker side, mm.")(f)
    f = click.option("--dictionary", default="DICT_5X5_250", show_default=True, help="ArUco dictionary name.")(f)
    return f


def _guard(f):
    """Turn a refusal into exit code 1 with the reason on the last line."""
    @functools.wraps(f)
    def wrapper(*a, **k):
        try:
            return f(*a, **k)
        except (IntakeError, CalibrationError) as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
    return wrapper


def _camera(raw_pattern, extensions, crop_rotation, rail_crop=None):
    try:
        return profile_from_options(raw_pattern, crop_rotation, extensions, rail_crop)
    except Exception as e:  # bad regex / option value
        raise IntakeError(f"camera options: {e}")


@click.group()
@click.option("--config", "config_file", default=None, type=click.Path(dir_okay=False, path_type=Path),
              help="YAML file with defaults for the options below (folders, camera, board, ...).")
@click.pass_context
def intake(ctx, config_file):
    """Lab intake: raw clips + recorder metadata -> day folders, whiteboard-verified names, px/cm."""
    if config_file is not None:
        from .config import load_defaults
        try:
            d = load_defaults(config_file)
        except IntakeError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
        ctx.default_map = {name: dict(d) for name in ctx.command.commands}


@intake.command("check")
@_cohort_options
@_camera_options
@_guard
def check_cmd(cohort_dir, records_dir, raw_pattern, extensions, crop_rotation):
    """Report what exists per day and the next step. Writes nothing."""
    from .orchestrate import check
    r = check(cohort_dir, records_dir, camera=_camera(raw_pattern, extensions, crop_rotation))
    if not r["exists"]:
        raise IntakeError(f"cohort folder not found: {cohort_dir}")


@intake.command("sort")
@_cohort_options
@click.option("--trials-full", default=None, type=click.Path(dir_okay=False, path_type=Path), help="The recorder export. Default: newest *_trials_full.csv in the cohort folder.")
@click.option("--apply", is_flag=True, help="Move the clips (default: dry run).")
@click.option("--allow-mismatch", is_flag=True, help="Sort even if a day's clip count differs from the metadata. Never use blindly.")
@_camera_options
@_guard
def sort_cmd(cohort_dir, records_dir, trials_full, apply, allow_mismatch, raw_pattern, extensions, crop_rotation):
    """Flat raw clips -> <DATE>-DAY{N}/ folders (metadata copied into each day)."""
    from .orchestrate import records_dir_of
    from .sort import sort_days
    log = StageLog("sort", records_dir_of(cohort_dir, records_dir) / "intake.log")
    sort_days(cohort_dir, trials_full, apply=apply, allow_mismatch=allow_mismatch, camera=_camera(raw_pattern, extensions, crop_rotation), log=log)


@intake.command("boards")
@_cohort_options
@_days_option
@click.option("--allow-placeholders", is_flag=True, help="Accept 1-2 digit mouse ids (the recorder's placeholders) as real ids.")
@click.option("--sweep-all", is_flag=True, help="Search EVERY clip of the day for a whiteboard (slow; for days that do not reconcile).")
@click.option("--whole-clip", is_flag=True, help="Search the whole clip instead of its first 12 s.")
@click.option("--include-day0", is_flag=True, help="Also plan DAY0 (preference phase).")
@click.option("--workers", default=1, show_default=True, help="Clips searched in parallel.")
@click.option("--force", is_flag=True, help="Rebuild the plan and crops even for a renamed or signed-off day.")
@_camera_options
@_guard
def boards_cmd(cohort_dir, records_dir, days, allow_placeholders, sweep_all, whole_clip, include_day0, workers, force,
               raw_pattern, extensions, crop_rotation):
    """Provisional rename plan + whiteboard crops + montage + BOARD_VERIFICATION.md per day. Never renames."""
    from .orchestrate import boards_days
    boards_days(cohort_dir, records_dir, days, allow_placeholders=allow_placeholders, sweep_all=sweep_all, whole_clip=whole_clip,
                include_day0=include_day0, camera=_camera(raw_pattern, extensions, crop_rotation), workers=workers, force=force)


@intake.command("verify")
@_cohort_options
@click.option("--days", required=True, help="The day(s) being signed off, e.g. 3 or 1-5.")
@click.option("--reads", default="", help="Numbers read off the boards, 'block=number,...' e.g. 1=301,2=305. Blocks not listed are recorded as read == expected.")
@click.option("--rebind", is_flag=True, help="Re-bind the plan to the mice that were READ (needs a read for every block).")
@click.option("--verdict", default="APPLY", show_default=True, help="APPLY, or 'FIX <what>'.")
@click.option("--verifier", default="", help="Who read the boards (recorded in the file).")
@click.option("--note", default="", help="Free text recorded with the verdict.")
@click.option("--fixed-run-order", is_flag=True, help="Sign off WITHOUT reading the numbers: only for cohorts whose mice are always run "
              "in the metadata order; needs clean counts and a board at every block start. Logged loudly.")
@_guard
def verify_cmd(cohort_dir, records_dir, days, reads, rebind, verdict, verifier, note, fixed_run_order):
    """Record the board reads and the verdict in BOARD_VERIFICATION.md (the gate before `rename --apply`)."""
    from .boards import fill_verification, rebind_plan, write_verification_template
    from .metadata import trials_full_for_day
    from .orchestrate import records_dir_of, _need_days, _block_starts_without_crop
    rec = records_dir_of(cohort_dir, records_dir)
    log = StageLog("verify", rec / "intake.log")
    try:
        rd = {int(k): v.strip() for k, v in (x.split("=") for x in reads.split(",") if x.strip())}
    except ValueError:
        raise IntakeError(f"--reads must look like 1=301,2=305 (got {reads!r})")
    dd = _need_days(cohort_dir, days, log)
    if (rd or rebind) and len(dd) != 1:
        raise IntakeError("--reads / --rebind apply to ONE day: name a single day with --days")
    for n, d in dd.items():
        D = rec / f"day{n}"
        vmd = D / "BOARD_VERIFICATION.md"
        if rebind:
            rebind_plan(D / "rename_plan.csv", trials_full_for_day(d, cohort_dir), f"DAY{n}", n, rd, log=log)
            if vmd.exists():                                       # the table is rebuilt from the re-bound plan
                vmd.unlink()
            write_verification_template(D / "rename_plan.csv", vmd, f"{Path(cohort_dir).name} — DAY{n} ({d.name})", True, n)
            note_n = (note + " " if note else "") + "Plan re-bound from the board reads (rename_plan.provisional.csv = the metadata-order plan)."
            fill_verification(vmd, reads=rd, verdict=verdict, note=note_n, verifier=verifier, log=log)
        elif fixed_run_order:
            rep = D / "reconciliation_report.txt"
            perfect = rep.exists() and "PERFECT MATCH" in rep.read_text()
            missing = _block_starts_without_crop(D) if (D / "rename_plan.csv").exists() else [0]
            if not perfect or any(k != 1 for k in missing):
                raise IntakeError(f"DAY{n}: the fixed-run-order skip needs clean counts and a whiteboard at every block start "
                                  f"(counts {'ok' if perfect else 'do NOT reconcile'}; block starts without a board: {missing}) — read the boards")
            log(f"DAY{n}: !! FIXED-RUN-ORDER SKIP !! the board NUMBERS are NOT read; valid ONLY if the mice are always run in the metadata order")
            fill_verification(vmd, verdict=verdict, note=note, verifier=verifier, numbers_read=False, log=log)
        else:
            fill_verification(vmd, reads=rd, verdict=verdict, note=note, verifier=verifier, log=log)


@intake.command("rename")
@_cohort_options
@_days_option
@click.option("--apply", is_flag=True, help="Rename (default: dry run that validates the plan and the sign-off).")
@click.option("--force", is_flag=True, help="Proceed although the day already holds renamed clips.")
@_camera_options
@_guard
def rename_cmd(cohort_dir, records_dir, days, apply, force, raw_pattern, extensions, crop_rotation):
    """Apply the signed-off plan: two-phase atomic rename, sidecars moved along, rename_log.csv written."""
    from .orchestrate import rename_days
    rename_days(cohort_dir, records_dir, days, apply=apply, force=force, camera=_camera(raw_pattern, extensions, crop_rotation))


@intake.command("waste")
@_cohort_options
@click.option("--day", "day", required=True, help="The day the clip is in, e.g. 6.")
@click.option("--reason", default="", help="Why (recorded in waste_log.csv).")
@click.argument("clips", nargs=-1, required=True)
@_guard
def waste_cmd(cohort_dir, records_dir, day, reason, clips):
    """Move extra / aborted CLIPS (and sidecars) out of a day into <cohort>/waste/<day folder>/."""
    from .apply import move_to_waste
    from .orchestrate import records_dir_of, _need_days
    log = StageLog("waste", records_dir_of(cohort_dir, records_dir) / "intake.log")
    dd = _need_days(cohort_dir, day, log, include_day0=True)
    if len(dd) != 1:
        raise IntakeError("--day must name one day")
    move_to_waste(cohort_dir, next(iter(dd.values())), clips, reason=reason, log=log)


@intake.command("calibrate")
@_cohort_options
@_days_option
@click.option("--clip", "clips", multiple=True, type=click.Path(dir_okay=False, path_type=Path),
              help="Measure these clips instead of the first clip of each day (repeatable).")
@_board_options
@click.option("--rail-crop", default=None, help="y0,y1,x0,x1 of the straight corridor for the 10-cm rail cross-check, or 'none'. "
              "Default: the DJI profile's region, used only on 3840x2160 frames.")
@click.option("--workers", default=1, show_default=True, help="Clips measured in parallel.")
@_camera_options
@_guard
def calibrate_cmd(cohort_dir, records_dir, days, clips, board_squares, square_mm, marker_mm, dictionary, rail_crop, workers,
                  raw_pattern, extensions, crop_rotation):
    """px/cm of the cohort (the public `stride calibrate`, on the first clip of each day) -> <records>/calibration/px_per_cm.txt."""
    from .orchestrate import calibrate
    r = calibrate(cohort_dir, records_dir, days, clips=list(clips) or None, board=parse_board(board_squares, square_mm, marker_mm, dictionary),
                  camera=_camera(raw_pattern, extensions, crop_rotation, rail_crop), workers=workers)
    click.echo(f"px_per_cm = {r['px_per_cm']:.2f}")


@intake.command("auto")
@_cohort_options
@_days_option
@click.option("--apply", is_flag=True, help="Actually sort and rename (default: dry run of both).")
@click.option("--fixed-run-order", is_flag=True, help="Sign the boards off WITHOUT reading the numbers when the counts reconcile and a "
              "board is found at every block start. Only for cohorts always run in the metadata order. Logged loudly.")
@click.option("--verifier", default="", help="Recorded with an automatic sign-off.")
@click.option("--allow-placeholders", is_flag=True, help="Accept 1-2 digit mouse ids as real ids.")
@click.option("--allow-mismatch", is_flag=True, help="Sort although a day's clip count differs from the metadata.")
@click.option("--sweep-all", is_flag=True, help="Search every clip for whiteboards.")
@click.option("--whole-clip", is_flag=True, help="Search whole clips instead of the first 12 s.")
@click.option("--skip-calibration", is_flag=True, help="Stop after the rename.")
@_board_options
@click.option("--workers", default=1, show_default=True)
@_camera_options
@_guard
def auto_cmd(cohort_dir, records_dir, days, apply, fixed_run_order, verifier, allow_placeholders, allow_mismatch, sweep_all,
             whole_clip, skip_calibration, board_squares, square_mm, marker_mm, dictionary, workers, raw_pattern, extensions,
             crop_rotation):
    """sort -> boards -> STOP until the boards are signed off -> rename -> calibrate. Safe to re-run."""
    from .orchestrate import auto
    r = auto(cohort_dir, records_dir, days, apply=apply, fixed_run_order=fixed_run_order, verifier=verifier,
             allow_placeholders=allow_placeholders, allow_mismatch=allow_mismatch, sweep_all=sweep_all, whole_clip=whole_clip,
             board=parse_board(board_squares, square_mm, marker_mm, dictionary), camera=_camera(raw_pattern, extensions, crop_rotation),
             workers=workers, skip_calibration=skip_calibration)
    if r["status"] == "waiting_for_signoff":
        click.echo(f"WAITING: whiteboard sign-off needed for day(s) {', '.join(map(str, r['pending_days']))}", err=True)
        sys.exit(2)
    if r["status"] == "dry_run":
        click.echo("DRY RUN: nothing was moved or renamed (add --apply)")


if __name__ == "__main__":
    intake()
