"""
Shared plumbing for the config-driven stage scripts (run_inference, run_tmaze,
run_gait_extraction, gait_lmm, classification).

* loading the config with a clear message when it is missing,
* selecting the batch(es) to process,
* where each stage finds its inputs: an explicit path in the batch config always wins; when the
  batch gives none, the stage reads what the previous stage wrote under ``<output>/<batch>/``.

Layout written by the stages under ``<output>/<batch>/``::

    roi_slp/                       stage 2b  ROI predictions (.preds.v2.best1.slp)
    roi_yml/                       stage 2c  ROI polygons    (.preds.v2.best1.yml)
    decisions/decisions.csv        stage 3
    decisions/events.csv           stage 3
    decisions/metrics.csv          stage 3
    metrics_meta.csv               stage 3b  metrics + Gender / Genotype
    learning_cycle_1.csv, _2.csv   stage 3c
    gait_per_stride.csv ...        stage 4-5
    gait_per_stride_filtered.csv   stage 6

Nothing here computes anything.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import yaml


class StageError(Exception):
    """A stage cannot run or produced nothing; the message says why (printed, exit code 1)."""


def exists(path) -> bool:
    """``Path.exists()`` that returns False instead of raising on an unreadable path."""
    try:
        return bool(path) and Path(path).exists()
    except OSError:
        return False


def load_config_or_exit(config_path) -> dict:
    """Read the pipeline config; exit with one clear line when it is missing or empty."""
    p = Path(config_path)
    if not exists(p):
        print(f"ERROR: config file not found: {p}\n"
              f"       Pass --config <your config.yaml> (start from config.example.yaml).")
        sys.exit(2)
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or not isinstance(cfg.get("batches"), dict) or not cfg["batches"]:
        print(f"ERROR: {p} has no `batches:` section with at least one batch.")
        sys.exit(2)
    return cfg


def select_batches(config: dict, batch: Optional[str]) -> list[tuple[str, dict]]:
    """The (key, config) pairs to process; exits when --batch names an unknown batch."""
    batches = config["batches"]
    if batch:
        if batch not in batches:
            print(f"ERROR: batch '{batch}' is not in the config. Batches defined: {', '.join(batches)}")
            sys.exit(2)
        return [(batch, batches[batch])]
    return list(batches.items())


def batch_name(batch_key: str, batch_cfg: dict) -> str:
    return str(batch_cfg.get("name") or batch_key)


def batch_out(output_dir, batch_key: str) -> Path:
    return Path(output_dir) / batch_key


def default_paths(output_dir, batch_key: str) -> dict:
    """Where the stages write under <output>/<batch>/ (see the module docstring)."""
    bo = batch_out(output_dir, batch_key)
    return {
        "roi_slp_dir": bo / "roi_slp",
        "yml_dir": bo / "roi_yml",
        "decisions_csv": bo / "decisions" / "decisions.csv",
        "events_csv": bo / "decisions" / "events.csv",
        "metrics_csv": bo / "decisions" / "metrics.csv",
        "metrics_meta_csv": bo / "metrics_meta.csv",
        "stride_raw_csv": bo / "gait_per_stride.csv",
        "stride_csv": bo / "gait_per_stride_filtered.csv",
    }


def resolve_input(batch_cfg: dict, key: str, default: Optional[Path]) -> tuple[Optional[Path], str]:
    """Path of one stage input and where it came from.

    Returns ``(path, source)`` with source "config" (explicit, existing path in the batch),
    "previous stage" (the default under <output>/<batch>/) or "" when neither exists.
    """
    explicit = batch_cfg.get(key)
    if explicit and exists(explicit):
        return Path(explicit), "config"
    if default is not None and exists(default):
        return Path(default), "previous stage"
    return None, ""


def resolve_yml_dir(batch_cfg: dict, output_dir, batch_key: str) -> Optional[Path]:
    """ROI yml folder of a batch.

    explicit ``yml_dir`` (must exist) > ``<output>/<batch>/roi_yml`` when stage 2c wrote it >
    None, which means "ROI files sit next to the videos" (``<stem>.rois.yml`` etc.).
    """
    explicit = batch_cfg.get("yml_dir")
    if explicit:
        if not exists(explicit):
            raise StageError(f"yml_dir not found: {explicit} (leave `yml_dir` out of the batch when "
                             f"the ROI files sit next to the videos)")
        return Path(explicit)
    d = default_paths(output_dir, batch_key)["yml_dir"]
    return d if exists(d) and any(d.glob("*.yml")) else None


def missing_message(what: str, batch_cfg: dict, key: str, default: Optional[Path], made_by: str) -> str:
    """One line saying which input is missing, where it was looked for and which stage makes it."""
    explicit = batch_cfg.get(key)
    looked = []
    if explicit:
        looked.append(f"`{key}: {explicit}` (from the config)")
    if default is not None:
        looked.append(f"{default}")
    return (f"{what} not found. Looked for {' and '.join(looked)}. "
            f"It is written by {made_by}; run that stage first or set `{key}` in the batch.")
