"""
Protocol rules applied to the raw arm-entry decisions (``decisions.csv`` -> ``decisions_clean.csv``).

The raw decision stage over-calls an arm entry in two situations that a person scoring the
trial records as "no decision":

  R1  no full commit   the entry only cleared the soft 60 % body-in-arm fallback, not the strict
                       95 % (the mouse put 60-95 % of its body into the arm and withdrew)
  R2  late entry       the entry happened after the decision window (default 150 s = the 2:30
                       trial of the T-maze protocol; ``entry_ms > window``)

A trial stays a decision only if it cleared the strict threshold AND entered within the window.
Reclassified trials get ``ok = False``, ``reason = soft_entry_no_commit | entry_after_window`` and
empty entry columns. The raw ``decisions.csv`` is never modified.

The function works on the CSV text (no type conversion), so every untouched cell is written back
exactly as it was read.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

DEFAULT_WINDOW_S = 150.0
CLEAN_NAME = "decisions_clean.csv"


def apply_decision_rules(decisions_csv, window_s: float = DEFAULT_WINDOW_S,
                         out_csv: Optional[Path] = None, verbose: bool = True) -> dict:
    """Write ``decisions_clean.csv`` next to ``decisions_csv`` (or to ``out_csv``).

    Returns a summary dict: ``trials``, ``soft``, ``late``, ``reclassified``, ``decisions``,
    ``no_decision``, ``output``.
    """
    decisions_csv = Path(decisions_csv)
    if not decisions_csv.exists():
        raise FileNotFoundError(f"decisions file not found: {decisions_csv}")
    win_ms = float(window_s) * 1000.0

    with open(decisions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    need = {"ok", "entry_threshold", "entry_ms"}
    if not need.issubset(fieldnames):
        raise ValueError(f"{decisions_csv.name} is not a STRIDE decisions file: missing column(s) "
                         f"{sorted(need - set(fieldnames))}")

    n_soft = n_late = 0
    for r in rows:
        if r.get("ok") != "True":
            continue
        soft = "60%" in r.get("entry_threshold", "")
        try:
            late = float(r["entry_ms"]) > win_ms
        except (ValueError, KeyError):
            late = False
        if soft or late:
            r["ok"] = "False"
            r["reason"] = "soft_entry_no_commit" if soft else "entry_after_window"
            r["entry_side"] = ""
            r["entry_frame"] = ""
            r["entry_ms"] = ""
            r["entry_threshold"] = "none"
            # A trial whose reward is unknown has empty correctness columns; they stay empty.
            unknown_reward = r.get("correct_TF", "") == ""
            r["correct_TF"] = "" if unknown_reward else "F"
            r["correct_bin"] = "" if unknown_reward else "0"
            n_soft += soft
            n_late += (late and not soft)

    out = Path(out_csv) if out_csv else decisions_csv.parent / CLEAN_NAME
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    dec = sum(1 for r in rows if r["ok"] == "True")
    summary = {
        "trials": len(rows), "soft": int(n_soft), "late": int(n_late),
        "reclassified": int(n_soft + n_late), "decisions": dec,
        "no_decision": len(rows) - dec, "window_s": float(window_s), "output": str(out),
    }
    if verbose:
        print(f"reclassified to U: {n_soft} soft(no-commit) + {n_late} late(>{window_s:.0f}s) "
              f"= {n_soft + n_late} of {len(rows)} trials")
        print(f"decisions now: {dec} | U(no-decision): {len(rows) - dec}  -> {out}")
    return summary
