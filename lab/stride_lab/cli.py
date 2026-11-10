"""`stride-lab`: the lab's own pipeline, from the camera card to the STRIDE tables.

    stride-lab run --config cohort.yaml [--apply] [--days 1-3]

    check      what exists per day and the next step (writes nothing)
    sort       flat raw clips -> <DATE>-DAY{N}/ folders
    boards     rename plan + whiteboard crops + montage + BOARD_VERIFICATION.md per day (never renames)
    verify     record the board reads and the verdict (the sign-off)
    rename     apply a signed-off plan (dry run unless --apply)
    waste      move extra / aborted clips out of a day
    calibrate  px/cm on every recording day + the camera-moved check; --decide records a decision for a held day
    pipeline   scale check -> inference -> the public STRIDE T-maze pipeline -> gate -> summary
    summary    rewrite COHORT_SUMMARY.md and DATA_QUALITY.md from what is on disk
    auto       sort -> boards -> STOP for the sign-off -> rename -> calibrate
    run        auto, then pipeline

`--config FILE` (anywhere on the line) is the cohort config; a command-line option always wins over the file.

Exit codes: 0 = done; 1 = a stage refused or failed (the reason is printed); 2 = stopped for a person: the
whiteboard sign-off, or a recording day held by the scale check. Nothing is wrong: decide, then re-run.
"""
from __future__ import annotations

import functools
import sys
from pathlib import Path

import click
from stride.stages.scale_calibration import CalibrationError

from .intake import cli as _icli
from .intake._log import IntakeError, StageLog


class LabGroup(click.Group):
    """Lets `--config FILE` be written after the command name as well as before it."""

    def parse_args(self, ctx, args):
        args = list(args)
        for i, a in enumerate(args):
            if a == "--config" and i + 1 < len(args):
                args = [a, args[i + 1]] + args[:i] + args[i + 2:]
                break
            if a.startswith("--config="):
                args = [a] + args[:i] + args[i + 1:]
                break
        return super().parse_args(ctx, args)


def _guard(f):
    @functools.wraps(f)
    def wrapper(*a, **k):
        try:
            return f(*a, **k)
        except (IntakeError, CalibrationError) as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
    return wrapper


def _intake_defaults(cfg: dict) -> dict:
    """Cohort config -> the option defaults of the intake commands."""
    cam, b = cfg.get("camera") or {}, cfg.get("board") or {}
    d = dict(cohort_dir=str(cfg["raw_dir"]), records_dir=str(cfg["records_dir"]), workers=int(cfg["analysis"]["workers"]),
             verifier=cfg.get("verifier") or "", allow_placeholders=bool(cfg.get("allow_placeholders")),
             allow_mismatch=bool(cfg.get("allow_mismatch")), fixed_run_order=bool(cfg.get("fixed_run_order")),
             board_squares=b.get("squares", "14x9"), square_mm=b.get("square_mm", 20.0), marker_mm=b.get("marker_mm", 15.0),
             dictionary=b.get("dictionary", "DICT_5X5_250"))
    if cfg.get("days"):
        d["days"] = str(cfg["days"])
    for k in ("raw_pattern", "crop_rotation"):
        if cam.get(k):
            d[k] = cam[k]
    if cam.get("extensions"):
        d["extensions"] = cam["extensions"] if isinstance(cam["extensions"], str) else ",".join(cam["extensions"])
    if cam.get("rail_crop") is not None:
        d["rail_crop"] = cam["rail_crop"] if isinstance(cam["rail_crop"], str) else ",".join(str(x) for x in cam["rail_crop"])
    return d


def _cfg(ctx) -> dict:
    cfg = ctx.obj
    if not cfg:
        raise IntakeError("this command needs the cohort config: stride-lab <command> --config cohort.yaml")
    return cfg


@click.group(cls=LabGroup)
@click.option("--config", "config_file", default=None, type=click.Path(dir_okay=False, path_type=Path),
              help="The cohort config (YAML): folders, metadata, models, board, thresholds.")
@click.pass_context
def main(ctx, config_file):
    """STRIDE Lab: raw clips + recorder metadata -> named clips -> scale check -> the STRIDE T-maze pipeline."""
    ctx.obj = None
    if config_file is not None:
        from .cohort import load_cohort_config
        try:
            ctx.obj = load_cohort_config(config_file)
        except IntakeError as e:
            click.echo(f"ERROR: {e}", err=True)
            sys.exit(1)
        ctx.default_map = {name: _intake_defaults(ctx.obj) for name in ctx.command.commands}


# the intake stages are the validated commands of stride_lab.intake, unchanged
for _name in ("check", "sort", "boards", "verify", "rename", "waste"):
    main.add_command(_icli.intake.commands[_name], _name)

_days = click.option("--days", default=None, help='Days to work on: "1,2" | "1-3" | "DAY1" | "2026-10-14-DAY1". Default: every DAY >= 1.')


@main.command("calibrate")
@_days
@click.option("--decide", "decide", multiple=True, metavar="N=DECISION",
              help="Record a person's decision for a held day: accept_own | accept_cohort | exclude | clear (repeatable).")
@click.option("--by", default="", help="Who decided (recorded).")
@click.option("--note", default="", help="Free text recorded with the decision.")
@click.option("--force", is_flag=True, help="Measure the clips again instead of using the stored readings.")
@click.pass_context
@_guard
def calibrate_cmd(ctx, days, decide, by, note, force):
    """px/cm on every recording day (ChArUco in view) and the camera-moved check."""
    from . import calibration as cal
    from .pipeline import run_calibration
    cfg = _cfg(ctx)
    log = StageLog("calibrate", cfg["records_dir"] / "intake.log")
    if decide and cal.load_state(cfg["raw_dir"], cfg["records_dir"]) is None:
        run_calibration(cfg, days or cfg.get("days"), log=log)
    for item in decide:
        try:
            n, dec = item.split("=", 1)
            n = int(str(n).upper().replace("DAY", ""))
        except ValueError:
            raise IntakeError(f"--decide must look like 4=accept_own (got {item!r})")
        cal.record_decision(cfg["raw_dir"], cfg["records_dir"], n, dec.strip(), by=by or cfg.get("verifier") or "", note=note, log=log)
    state = run_calibration(cfg, None if decide else (days or cfg.get("days")), force=force, log=log)
    click.echo(f"px_per_cm = {state['used_px_per_cm']:.2f}   ({state['source']}); report: "
               f"{cal.cal_dir(cfg['raw_dir'], cfg['records_dir']) / 'CALIBRATION.md'}")
    if state["hold_days"]:
        click.echo(f"WAITING: day(s) {', '.join(map(str, state['hold_days']))} are held by the scale check — record a decision "
                   "(--decide N=accept_own|accept_cohort|exclude --by NAME)", err=True)
        sys.exit(2)


@main.command("pipeline")
@_days
@click.option("--skip-inference", is_flag=True, default=None, help="Analyse only clips that already have pose and ROI files.")
@click.option("--plan", "plan_only", is_flag=True, help="Print the inference worker commands and stop (nothing is run).")
@click.pass_context
@_guard
def pipeline_cmd(ctx, days, skip_inference, plan_only):
    """Scale check -> pose + ROI inference -> the public STRIDE T-maze pipeline -> gait gate -> cohort summary."""
    from .pipeline import pipeline
    cfg = _cfg(ctx)
    r = pipeline(cfg, days or cfg.get("days"), skip_inference=skip_inference, plan_only=plan_only)
    _exit_for(r)


@main.command("summary")
@click.pass_context
@_guard
def summary_cmd(ctx):
    """Rewrite COHORT_SUMMARY.md, cohort_summary.csv and DATA_QUALITY.md from what is on disk."""
    from .pipeline import write_summary
    cfg = _cfg(ctx)
    write_summary(cfg, log=StageLog("summary", cfg["records_dir"] / "intake.log"))


def _intake_auto(cfg, days, apply, fixed_run_order, allow_placeholders, allow_mismatch, sweep_all, whole_clip, log):
    from .cohort import board_of, camera_of
    from .intake.orchestrate import auto
    return auto(cfg["raw_dir"], cfg["records_dir"], days, apply=apply,
                fixed_run_order=fixed_run_order or bool(cfg.get("fixed_run_order")), verifier=cfg.get("verifier") or "",
                allow_placeholders=allow_placeholders or bool(cfg.get("allow_placeholders")),
                allow_mismatch=allow_mismatch or bool(cfg.get("allow_mismatch")), sweep_all=sweep_all, whole_clip=whole_clip,
                board=board_of(cfg), camera=camera_of(cfg), workers=int(cfg["analysis"]["workers"]), skip_calibration=True, log=log)


def _auto_options(f):
    f = click.option("--whole-clip", is_flag=True, help="Search whole clips for whiteboards instead of the first 12 s.")(f)
    f = click.option("--sweep-all", is_flag=True, help="Search every clip for whiteboards.")(f)
    f = click.option("--allow-mismatch", is_flag=True, help="Sort although a day's clip count differs from the metadata.")(f)
    f = click.option("--allow-placeholders", is_flag=True, help="Accept 1-2 digit mouse ids as real ids.")(f)
    f = click.option("--fixed-run-order", is_flag=True, help="Sign the boards off WITHOUT reading the numbers when the counts "
                     "reconcile and a board is found at every block start. Only for cohorts always run in the metadata order.")(f)
    f = click.option("--apply", is_flag=True, help="Actually sort and rename (default: dry run of both).")(f)
    return _days(f)


def _exit_for(r: dict) -> None:
    if r["status"] == "waiting_for_signoff":
        click.echo(f"WAITING: whiteboard sign-off needed for day(s) {', '.join(map(str, r['pending_days']))}", err=True)
        sys.exit(2)
    if r["status"] == "held":
        click.echo(f"WAITING: day(s) {', '.join(map(str, r['held']))} are held by the scale check — record a decision "
                   "(stride-lab calibrate --decide N=accept_own|accept_cohort|exclude --by NAME), then re-run", err=True)
        sys.exit(2)
    if r["status"] == "dry_run":
        click.echo("DRY RUN: nothing was moved or renamed (add --apply)")


@main.command("auto")
@_auto_options
@click.pass_context
@_guard
def auto_cmd(ctx, days, apply, fixed_run_order, allow_placeholders, allow_mismatch, sweep_all, whole_clip):
    """sort -> boards -> STOP until the boards are signed off -> rename -> calibrate. Safe to re-run."""
    from .pipeline import run_calibration
    cfg = _cfg(ctx)
    log = StageLog("auto", cfg["records_dir"] / "intake.log")
    r = _intake_auto(cfg, days or cfg.get("days"), apply, fixed_run_order, allow_placeholders, allow_mismatch, sweep_all, whole_clip, log)
    if r["status"] != "complete":
        _exit_for(r)
        return
    log.stage = "calibrate"
    state = run_calibration(cfg, days or cfg.get("days"), log=log)
    _exit_for(dict(status="held" if state["hold_days"] else "complete", held=state["hold_days"]))


@main.command("run")
@_auto_options
@click.option("--skip-inference", is_flag=True, default=None, help="Analyse only clips that already have pose and ROI files.")
@click.pass_context
@_guard
def run_cmd(ctx, days, apply, fixed_run_order, allow_placeholders, allow_mismatch, sweep_all, whole_clip, skip_inference):
    """The whole flow: intake (sort, whiteboard check, rename), scale check, inference, analysis, summary."""
    from .pipeline import pipeline
    cfg = _cfg(ctx)
    if allow_placeholders:
        cfg["allow_placeholders"] = True
    log = StageLog("auto", cfg["records_dir"] / "intake.log")
    r = _intake_auto(cfg, days or cfg.get("days"), apply, fixed_run_order, allow_placeholders, allow_mismatch, sweep_all, whole_clip, log)
    if r["status"] != "complete":
        _exit_for(r)
        return
    log.stage = "pipeline"
    _exit_for(pipeline(cfg, days or cfg.get("days"), skip_inference=skip_inference, log=log))


if __name__ == "__main__":
    main()
