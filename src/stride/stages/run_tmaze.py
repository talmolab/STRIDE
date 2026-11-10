# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "pyyaml",
# ]
# ///
"""
Config-driven T-maze behavioral analysis (Stage 3) + post-processing.

For finalized batches: loads existing decisions/events/metrics CSVs.
For new batches: calls stride.stages.decision_analysis,
then runs learning cycle splitting and metadata enrichment.

PROVENANCE:
  Decision analysis: delegates to stride.stages.decision_analysis
  Learning cycle splitting and metadata enrichment: ported from the analysis notebooks of the
  original study

Usage:
    python -m stride.stages.run_tmaze --config config.yaml --output output
    python -m stride.stages.run_tmaze --config config.yaml --output output --batch my_batch
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


from stride.stages.batch_paths import (
    StageError, batch_name, load_config_or_exit, resolve_yml_dir, select_batches,
)


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# STAGE 3: Decision analysis (stride.stages.decision_analysis)
# ============================================================================

def run_decisions(batch_cfg: dict, output_dir: Path, yml_dir: Path | None = None) -> dict[str, Path]:
    """Run decision analysis for a non-finalized batch.

    `yml_dir` None means the ROI files sit next to the videos.
    Returns dict with paths to decisions.csv, events.csv, metrics.csv.
    Raises StageError (with the reason) when an input is missing or no video could be analysed.
    """
    from stride.stages.decision_analysis import run_decision_analysis
    from stride.config import PipelineConfig

    for key in ("video_dir", "meta_trials_csv"):
        if not batch_cfg.get(key):
            raise StageError(f"the batch has no `{key}` (required for stage 3)")
    video_dir = Path(batch_cfg["video_dir"])
    meta_csv = Path(batch_cfg["meta_trials_csv"])

    if not video_dir.exists():
        raise StageError(f"video_dir not found: {video_dir} (edit the batch in your config)")
    if not meta_csv.exists():
        raise StageError(f"meta_trials_csv not found: {meta_csv}")

    decisions_dir = output_dir / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running decision analysis...")
    print(f"    Videos: {video_dir}")
    print(f"    ROI YAMLs: {yml_dir if yml_dir is not None else 'next to the videos'}")
    print(f"    Meta: {meta_csv}")

    result = run_decision_analysis(
        video_dir=video_dir,
        yml_dir=yml_dir,
        meta_csv=meta_csv,
        output_dir=decisions_dir,
        n_workers=4,
        pose_slp_suffix=batch_cfg.get("pose_slp_suffix", ".slp"),
    )

    print(f"  Decision analysis: {result.get('ok', 0)} ok, {result.get('skipped', 0)} skipped")
    if result.get("ok", 0) == 0:
        why = "; ".join(f"{n} x {r}" for r, n in (result.get("skip_reasons") or {}).items()) or "no videos found"
        raise StageError(f"no video could be analysed in {video_dir} ({why})")

    return {
        "decisions_csv": decisions_dir / "decisions.csv",
        "events_csv": decisions_dir / "events.csv",
        "metrics_csv": decisions_dir / "metrics.csv",
    }


# ============================================================================
# STAGE 3b: Metadata enrichment
# PROVENANCE: analysis notebooks of the original study
# ============================================================================

def enrich_with_metadata(metrics_csv: Path, metadata_csv: Path,
                         column_map: dict, output_csv: Path) -> Path:
    """Merge genotype/sex from external metadata CSV into metrics.

    Args:
        metrics_csv: Path to metrics.csv (or metrics.v2.csv)
        metadata_csv: Path to T-maze-metadata.csv (columns: Mouse, Gender, Genotype)
        column_map: Batch-specific column mapping
        output_csv: Where to write enriched CSV
    """
    data = pd.read_csv(metrics_csv)
    metadata = pd.read_csv(metadata_csv)

    # Determine mouse ID column in metrics
    id_cols = column_map.get("mouse_id", ["mouse"])
    if isinstance(id_cols, str):
        id_cols = [id_cols]

    # Standardize metadata mouse column
    if "Mouse" in metadata.columns:
        metadata = metadata.rename(columns={"Mouse": "mouse"})

    # For single-column ID, merge directly
    if len(id_cols) == 1:
        merge_col = id_cols[0]
        if merge_col not in data.columns:
            print(f"    WARNING: merge column '{merge_col}' not in metrics")
            return metrics_csv
        # Ensure compatible types
        data[merge_col] = data[merge_col].astype(str)
        metadata["mouse"] = metadata["mouse"].astype(str)
        merge_cols = ["mouse"]
        if "Gender" in metadata.columns:
            merge_cols.append("Gender")
        if "Genotype" in metadata.columns:
            merge_cols.append("Genotype")
        data = data.merge(
            metadata[merge_cols].rename(columns={"mouse": merge_col}),
            how="left", on=merge_col,
        )
    else:
        # Composite ID: build MouseID in both dataframes and merge
        sep = column_map.get("mouse_id_sep", "_")
        data["_merge_id"] = data[id_cols[0]].astype(str)
        for c in id_cols[1:]:
            data["_merge_id"] = data["_merge_id"] + sep + data[c].astype(str)
        metadata["_merge_id"] = metadata["mouse"].astype(str)
        merge_cols = ["_merge_id"]
        if "Gender" in metadata.columns:
            merge_cols.append("Gender")
        if "Genotype" in metadata.columns:
            merge_cols.append("Genotype")
        data = data.merge(metadata[merge_cols], how="left", on="_merge_id")
        data.drop(columns=["_merge_id"], inplace=True)

    data.to_csv(output_csv, index=False)
    print(f"    Enriched with metadata: {len(data)} rows → {output_csv.name}")
    return output_csv


# ============================================================================
# STAGE 3c: Learning cycle splitting
# PROVENANCE: analysis notebooks of the original study
# ============================================================================

def split_learning_cycles(metrics_csv: Path, output_dir: Path,
                          mouse_id_col: str = "mouseID") -> tuple[Path, Path]:
    """Detect reward reversals and split data into learning cycles 1 and 2.

    Logic:
    1. Extract day number from "DAY1", "DAY2", etc.
    2. Detect when reward side changes per mouse
    3. learn_1 = first reversal day per mouse
    4. Cycle 1: all trials before learn_1 (or all trials if no reversal)
    5. Cycle 2: all trials from learn_1 onward

    Args:
        metrics_csv: Path to metrics CSV with columns: day, mouseID/mouse, reward
        output_dir: Where to write learning_cycle_1.csv and learning_cycle_2.csv
        mouse_id_col: Column name for mouse ID

    Returns:
        Tuple of (path to cycle 1 CSV, path to cycle 2 CSV)
    """
    df = pd.read_csv(metrics_csv)

    # Build mouse ID column if not present
    if mouse_id_col not in df.columns:
        if "mouse" in df.columns:
            mouse_id_col = "mouse"
        elif "MouseID" in df.columns:
            mouse_id_col = "MouseID"
        elif "mouse_id" in df.columns:
            mouse_id_col = "mouse_id"
        elif "G" in df.columns and "number" in df.columns:
            # Basler batch: composite ID from G + number
            df["mouseID"] = df["G"].astype(str) + "_" + df["number"].astype(str)
            mouse_id_col = "mouseID"
        else:
            print(f"    WARNING: No mouse ID column found. Skipping learning cycle split.")
            return None, None

    df["mouseID"] = df[mouse_id_col].astype(str)

    # Extract day number
    df["day_num"] = df["day"].str.extract(r"(\d+)").astype(int)
    df_sorted = df.sort_values(["mouseID", "day_num"])

    # Detect reward side changes
    df_sorted["reward_changed"] = df_sorted.groupby("mouseID")["reward"].transform(
        lambda x: x != x.shift(1)
    )

    # Get first change day per mouse
    changes = df_sorted[
        df_sorted["reward_changed"] & (df_sorted.groupby("mouseID").cumcount() > 0)
    ]
    change_summary = changes.groupby("mouseID")["day"].apply(list).reset_index()
    change_summary["num_changes"] = change_summary["day"].apply(len)
    change_summary["learn_1"] = change_summary["day"].apply(lambda x: x[0] if len(x) > 0 else None)

    # Merge learn_1 back
    df_with = df_sorted.merge(change_summary[["mouseID", "learn_1"]], on="mouseID", how="left")
    df_with["learn_1_num"] = df_with["learn_1"].str.extract(r"(\d+)").astype(float)

    # Split
    cycle1 = df_with[
        (df_with["day_num"] >= 1) &
        ((df_with["day_num"] < df_with["learn_1_num"]) | (df_with["learn_1_num"].isna()))
    ].copy()
    cycle1["learning_cycle"] = 1

    cycle2 = df_with[
        (df_with["day_num"] >= df_with["learn_1_num"]) &
        (df_with["learn_1_num"].notna())
    ].copy()
    cycle2["learning_cycle"] = 2

    out1 = output_dir / "learning_cycle_1.csv"
    out2 = output_dir / "learning_cycle_2.csv"
    cycle1.to_csv(out1, index=False)
    cycle2.to_csv(out2, index=False)

    print(f"    Learning cycle 1: {len(cycle1)} trials → {out1.name}")
    print(f"    Learning cycle 2: {len(cycle2)} trials → {out2.name}")

    return out1, out2


# ============================================================================
# MAIN
# ============================================================================

def process_batch(batch_key: str, batch_cfg: dict, config: dict, output_dir: Path):
    """Process one batch: decisions → metadata enrichment → learning cycle split."""
    print(f"\n{'='*60}")
    print(f"Batch: {batch_name(batch_key, batch_cfg)}")
    print(f"{'='*60}")

    batch_out = output_dir / batch_key
    batch_out.mkdir(parents=True, exist_ok=True)

    # --- Stage 3: Decision analysis ---
    decisions_path = batch_cfg.get("decisions_csv", "")
    events_path = batch_cfg.get("events_csv", "")

    if batch_cfg.get("finalized", False) and decisions_path and Path(decisions_path).exists():
        print(f"  Stage 3: Finalized. Loading existing CSVs.")
        decisions = pd.read_csv(decisions_path)
        print(f"    decisions.csv: {len(decisions)} trials")
        if events_path and Path(events_path).exists():
            events = pd.read_csv(events_path)
            print(f"    events.csv: {len(events)} event rows")
    else:
        print(f"  Stage 3: Running decision analysis...")
        video_dir = batch_cfg.get("video_dir")
        if not video_dir or not Path(video_dir).exists():
            raise StageError(f"video_dir not found: {video_dir!r}. Edit the batch in your config "
                             f"(config.example.yaml in the repository is the template to copy).")
        paths = run_decisions(batch_cfg, batch_out, resolve_yml_dir(batch_cfg, output_dir, batch_key))
        decisions_path = str(paths["decisions_csv"])
        events_path = str(paths["events_csv"])

    # --- Stage 3b: Metadata enrichment ---
    metrics_meta_csv = batch_cfg.get("metrics_meta_csv", "")
    metadata_csv = batch_cfg.get("metadata_csv", "")
    column_map = batch_cfg.get("column_map", {"mouse_id": ["mouse"], "sex": "Gender"})

    if metrics_meta_csv and Path(metrics_meta_csv).exists():
        print(f"\n  Stage 3b: metrics_meta already exists: {Path(metrics_meta_csv).name}")
        enriched_csv = Path(metrics_meta_csv)
    elif metadata_csv and Path(metadata_csv).exists():
        # Find a metrics CSV to enrich
        metrics_candidates = [
            batch_out / "decisions" / "metrics.csv",
        ]
        metrics_src = None
        for mc in metrics_candidates:
            if mc.exists():
                metrics_src = mc
                break
        if metrics_src:
            print(f"\n  Stage 3b: Enriching metrics with metadata...")
            enriched_csv = batch_out / "metrics_meta.csv"
            enrich_with_metadata(metrics_src, Path(metadata_csv), column_map, enriched_csv)
        else:
            print(f"\n  Stage 3b: No metrics CSV to enrich. Skipping.")
            enriched_csv = None
    else:
        print(f"\n  Stage 3b: No metadata_csv configured. Skipping.")
        enriched_csv = None

    # --- Stage 3c: Learning cycle splitting ---
    if enriched_csv and enriched_csv.exists():
        print(f"\n  Stage 3c: Splitting learning cycles...")
        lc1_path = batch_out / "learning_cycle_1.csv"
        if lc1_path.exists():
            print(f"    learning_cycle_1.csv already exists. Skipping.")
        else:
            split_learning_cycles(enriched_csv, batch_out)
    elif metrics_meta_csv and Path(metrics_meta_csv).exists():
        # Already have enriched data from finalized batch
        lc_dir = Path(metrics_meta_csv).parent
        lc1 = lc_dir / "learning_cycle_1.csv"
        if lc1.exists():
            print(f"\n  Stage 3c: learning_cycle_1.csv already exists at {lc_dir.name}/")
        else:
            print(f"\n  Stage 3c: Splitting learning cycles from existing metrics_meta...")
            split_learning_cycles(Path(metrics_meta_csv), batch_out)
    else:
        print(f"\n  Stage 3c: No enriched metrics available. Skipping learning cycle split.")

    # Provenance sidecar for the stage-3 outputs of this batch (best-effort)
    try:
        from stride.run_manifest import write_run_manifest
        write_run_manifest(
            batch_out, "tmaze",
            inputs={"video_dir": batch_cfg.get("video_dir"), "yml_dir": batch_cfg.get("yml_dir"),
                    "meta_trials_csv": batch_cfg.get("meta_trials_csv"),
                    "metadata_csv": batch_cfg.get("metadata_csv"),
                    "finalized": bool(batch_cfg.get("finalized", False))},
            params={"pose_slp_suffix": batch_cfg.get("pose_slp_suffix", ".slp"), "column_map": column_map},
            outputs=[p for p in (decisions_path, events_path, batch_out / "metrics_meta.csv",
                                 batch_out / "learning_cycle_1.csv", batch_out / "learning_cycle_2.csv") if p],
        )
    except Exception:
        pass

    # Summary
    print(f"\n  Batch {batch_key} complete.")
    if decisions_path and Path(decisions_path).exists():
        d = pd.read_csv(decisions_path)
        if "correct_TF" in d.columns:
            ok = d[d.get("ok", True) == True] if "ok" in d.columns else d
            if "correct_bin" in ok.columns:
                accuracy = ok["correct_bin"].astype(float).mean() * 100
                print(f"    Overall accuracy: {accuracy:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Config-driven T-maze behavioral analysis")
    parser.add_argument("--config", default="config.yaml", help="Pipeline config (default: ./config.yaml)")
    parser.add_argument("--output", default="output", help="Output directory (default: ./output)")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    args = parser.parse_args()

    config = load_config_or_exit(args.config)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("T-Maze Behavioral Analysis Pipeline — Stage 3")
    print(f"Config: {args.config}")
    print("=" * 60)

    failures = []
    for batch_key, batch_cfg in select_batches(config, args.batch):
        try:
            process_batch(batch_key, batch_cfg, config, out_dir)
        except StageError as exc:
            print(f"  ERROR: {exc}")
            failures.append(f"{batch_key}: {exc}")

    if failures:
        print(f"\nStage 3 FAILED for {len(failures)} batch(es):")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\nDone.")


if __name__ == "__main__":
    main()
