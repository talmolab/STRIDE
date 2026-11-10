# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""
Master orchestrator: runs the full T-maze pipeline from raw data to figures.

Stages:
  1-2:   run_inference.py            — SLEAP pose + ROI inference, SLP→YAML
  3:     run_tmaze.py                — decisions, events, metrics, learning cycles
  4-6:   run_gait_extraction.py      — stride detection + filtering
  7-9:   gait_lmm.py                 — trial tagging, LMM, heatmaps
  10:    classification.py            — WT vs Tau classification

Each stage checks for output file existence before running (resumable).

Usage (from the repo root):
    python run_all.py                      # uses ./config.yaml, writes ./output/
    python run_all.py --batch gopro_sept2025
    python run_all.py --stages 7-9         # only LMM
    python run_all.py --stages 3,7-9       # skip inference, run decisions + LMM
    python run_all.py --stages 1-2,3,4-6   # full from scratch through filtering

Each stage reads what the previous one wrote under <output>/<batch>/ unless the batch config
names an explicit path. A stage that fails stops the run with a non-zero exit code.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parent
SRC_STAGES = PIPELINE_DIR / "src" / "stride" / "stages"
STAGES = {
    "1-2": {
        "name": "SLEAP Inference + SLP→YAML",
        "script": SRC_STAGES / "run_inference.py",
    },
    "3": {
        "name": "T-Maze Behavioral Analysis",
        "script": SRC_STAGES / "run_tmaze.py",
    },
    "4-6": {
        "name": "Gait Extraction & Filtering",
        "script": SRC_STAGES / "run_gait_extraction.py",
    },
    "7-9": {
        "name": "Gait LMM Analysis",
        "script": SRC_STAGES / "gait_lmm.py",
    },
    "10": {
        "name": "WT vs Tau Classification",
        "script": SRC_STAGES / "classification.py",
    },
}


def run_stage(stage_key: str, stage_info: dict, config: str, output: str,
              batch: str | None = None, device: str | None = None) -> bool:
    """Run a single pipeline stage via subprocess."""
    script = stage_info["script"]
    name = stage_info["name"]

    print(f"\n{'#' * 60}")
    print(f"# Stage {stage_key}: {name}")
    print(f"# Script: {script.name}")
    print(f"{'#' * 60}")

    if not script.exists():
        print(f"  ERROR: Script not found: {script}")
        return False

    # Resolve to absolute paths so stage subprocesses are CWD-independent.
    # (A relative --config previously broke: stages ran with cwd set to the
    # PARENT of the repo — a relic of the pre-package `automated_pipeline/`
    # layout — so `config.yaml` raised FileNotFoundError even when the user
    # ran from the repo root, exactly as the README instructs.)
    cmd = [sys.executable, str(script),
           "--config", str(Path(config).resolve()),
           "--output", str(Path(output).resolve())]
    if batch and stage_key != "10":  # the classifier pools all batches; every other stage takes --batch
        cmd.extend(["--batch", batch])
    if device and stage_key == "1-2":  # inference device (stages 1-2 only)
        cmd.extend(["--device", device])

    t0 = time.time()
    result = subprocess.run(cmd)  # inherit the caller's CWD
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED (exit code {result.returncode}) after {elapsed:.1f}s — the reason is printed above")
        return False

    print(f"\n  Completed in {elapsed:.1f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="T-Maze Pipeline Orchestrator")
    parser.add_argument("--config", default=str(PIPELINE_DIR / "config.yaml"),
                        help="Pipeline config (default: config.yaml next to this script)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: ./output, resolved from CWD)")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated stage keys to run, from: 1-2, 3, 4-6, 7-9, 10 "
                             "(e.g. '3,4-6,7-9'). Default: all")
    parser.add_argument("--device", default=None,
                        help="Inference device for stages 1-2 (cuda, cuda:0, cpu). Default: from the config, else cuda")
    args = parser.parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: config file not found: {args.config}\n"
              f"       Pass --config <your config.yaml> (start from config.example.yaml).")
        sys.exit(2)

    print("=" * 60)
    print("T-MAZE AUTOMATED PIPELINE — FULL RUN")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    if args.batch:
        print(f"Batch:  {args.batch}")
    print("=" * 60)

    # Determine which stages to run
    if args.stages:
        stage_keys = [s.strip() for s in args.stages.split(",")]
    else:
        stage_keys = list(STAGES.keys())

    # Validate
    for sk in stage_keys:
        if sk not in STAGES:
            print(f"ERROR: Unknown stage '{sk}'. Available: {', '.join(STAGES.keys())}")
            sys.exit(1)

    t_total = time.time()
    results = {}

    for sk in stage_keys:
        ok = run_stage(sk, STAGES[sk], args.config, args.output, args.batch, args.device)
        results[sk] = ok
        if not ok:
            print(f"\nStage {sk} failed. Stopping pipeline.")
            break

    # Summary
    elapsed_total = time.time() - t_total
    print(f"\n{'=' * 60}")
    print("PIPELINE SUMMARY")
    print(f"{'=' * 60}")
    for sk, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  Stage {sk:5s} ({STAGES[sk]['name']:35s}): {status}")
    print(f"\nTotal time: {elapsed_total:.1f}s")

    if all(results.values()):
        print("\nAll stages completed successfully.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
