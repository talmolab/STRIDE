"""The recorder export (`*_trials_full.csv`): one row per mouse per day.

Columns used: `day` (DAY0, DAY1, ...), `date` (YYYY-MM-DD), `mouse`, `T1`..`T9` (the reward side of each
trial that was run; the number of non-empty cells is that mouse's number of trials = clips) and `notes`.
Row order within a day is the order the mice are expected to be run in.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from ._log import IntakeError

TRIAL_COLS = [f"T{i}" for i in range(1, 10)]
PLACEHOLDER_ID = re.compile(r"\d{1,2}")   # recorder placeholders 1..16 before the real ids are typed in


def _rows(path: Path) -> list:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and not {"day", "mouse"}.issubset(rows[0].keys()):
        raise IntakeError(f"{path}: not a trials_full export (needs the columns day, mouse, T1..T9; has {list(rows[0])[:8]}...)")
    return rows


def newest_trials_full(folder: Path) -> Optional[Path]:
    """The newest export in a folder = the last `*_trials_full.csv` by name (names start with the export date)."""
    c = sorted(Path(folder).glob("*_trials_full.csv")) if Path(folder).is_dir() else []
    return c[-1] if c else None


def trials_full_for_day(day_dir: Path, cohort_dir: Path) -> Optional[Path]:
    """A day's metadata: `<day>/metadata/` first, the cohort root second (as the lab orchestrator does)."""
    return newest_trials_full(Path(day_dir) / "metadata") or newest_trials_full(cohort_dir)


def read_trials_full(path: Path):
    """-> (day_date {DAY: date}, trials {DAY: n}, mice {DAY: [(mouse, n), ...]}), mice with 0 trials left out."""
    day_date, trials, mice = {}, defaultdict(int), defaultdict(list)
    for r in _rows(path):
        day, date = (r.get("day") or "").strip(), (r.get("date") or "").strip()
        if not day or not date:
            continue
        n = sum(1 for k in TRIAL_COLS if (r.get(k) or "").strip())
        day_date.setdefault(day, date)
        if n:
            trials[day] += n
            mice[day].append(((r.get("mouse") or "").strip(), n))
    return day_date, trials, mice


def day_blocks(path: Path, day: str):
    """-> ([(mouse, n_trials, directions, notes)] in row order, [(mouse, notes)] for the mice with no trial)."""
    day_rows = [r for r in _rows(path) if (r.get("day") or "").strip().upper() == day.upper()]
    if not day_rows:
        raise IntakeError(f"no rows for day={day} in {path}")
    blocks, excluded = [], []
    for r in day_rows:
        dirs = [(r.get(c) or "").strip() for c in TRIAL_COLS if (r.get(c) or "").strip()]   # one entry per trial run
        mouse, notes = (r.get("mouse") or "").strip(), (r.get("notes") or "").strip()
        (blocks if dirs else excluded).append((mouse, len(dirs), dirs, notes) if dirs else (mouse, notes))
    return blocks, excluded


def placeholder_ids(blocks) -> bool:
    """True when every mouse id of the day is a 1-2 digit number (the recorder's placeholders)."""
    ids = [b[0] for b in blocks if b[0]]
    return bool(ids) and all(PLACEHOLDER_ID.fullmatch(m) for m in ids)
