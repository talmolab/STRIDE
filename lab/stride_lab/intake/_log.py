"""One-line-per-event logging for the front-end stages, plus the best-effort run manifest."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Optional


CMD = "stride-lab"                      # how the commands are spelled in messages


class IntakeError(RuntimeError):
    """A stage refused or failed. The message is the reason; the command exits non-zero with it."""


class StageLog:
    """Prints `[time] [stage] message` and appends the same line to a log file (if given)."""

    def __init__(self, stage: str, log_file: Optional[Path] = None, quiet: bool = False):
        self.stage, self.log_file, self.quiet = stage, (Path(log_file) if log_file else None), quiet
        self.lines = []

    def __call__(self, msg: str) -> None:
        line = f"[{_dt.datetime.now():%Y-%m-%d %H:%M:%S}] [{self.stage}] {msg}"
        self.lines.append(line)
        if not self.quiet:
            print(line, flush=True)
        if self.log_file is not None:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_file, "a") as f:
                    f.write(line + "\n")
            except OSError:
                pass

    def warn(self, msg: str) -> None:
        self(f"WARN {msg}")

    def refuse(self, msg: str):
        self(f"REFUSED: {msg}")
        raise IntakeError(msg)


def write_manifest(out_dir, stage: str, **kw):
    """`<out_dir>/<stage>_run_manifest.yaml` through STRIDE's own writer. Never raises."""
    try:
        from stride.run_manifest import write_run_manifest
        return write_run_manifest(out_dir, stage, **kw)
    except Exception:  # provenance must never break a finished stage
        return None
