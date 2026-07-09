# STRIDE / T-maze pipeline — RUNBOOK (run steps in order; do not self-select)

**Purpose.** A deterministic, gated procedure for taking one recording day from raw video to gait.
It exists because steps were being reordered/skipped by judgement and a silent calibration bug reached
the figures. **Follow the steps in order. Each step has a GATE that must pass before the next step, and
an explicit SKIP-IF that is the *only* condition under which it may be skipped.** Do not invent steps,
do not substitute ad-hoc scripts for the named tools, and do not "self-pick" which steps to run.

## Operating rules (read first)
1. **Run steps 0→7 in order.** Never jump ahead of a failed or unrun GATE.
2. **A GATE is mandatory.** If a GATE fails, **STOP and report** — do not work around it silently.
3. **SKIP-IF is the only license to skip.** If its condition isn't provably met, run the step.
4. **Use the named tool.** If a step names a script, use it. Do not hand-roll a replacement (that is
   how the ROI consensus fallback got reinvented three times and the calibration bug got missed).
5. **The automatic gates are `stride.stages.safety_checks`.** Run them; a nonzero exit means STOP.
6. When unsure whether a SKIP-IF holds, **do not skip** — running a satisfied step is cheap; skipping
   an unsatisfied one corrupts the cohort.

Paths below use `$PROJ` = the cohort project (e.g. `2026-06-15-T-maze-analysis`), `$VID` = the day's
video dir, `$PY` = the sleap-nn python. Front-end scripts (rename/ROI/pose) live in `$PROJ`; decisions
and gait are STRIDE stages.

---

## Step 0 — Preflight
- **Do:** confirm `$VID/*.MP4` present; `$VID/metadata/*_trials_full.csv` present; delete `.DS_Store`;
  count videos vs `sum(metadata trial counts)`.
- **✅ GATE:** `#videos == sum(trial counts)` (a clean day = 16 mice × 8 = 128, allow camera gaps in
  DJI sequence numbering but not in the count).
- **⏭ SKIP-IF:** never (always run).
- **🚫 NEVER:** proceed if the count delta ≠ 0 without resolving extras/aborts first (RENAME_SOP gotcha).

## Step 1 — Rename (whiteboard-anchored) — `$PROJ/rename/`
- **Do:** `build_rename_match.py` → provisional plan; then confirm a **whiteboard is present at each of
  the 16 block-starts** (montage of block-start frames); then two-phase atomic rename + `rename_log.csv`.
- **✅ GATE (both):** (a) count reconciles per Step 0; (b) a board is **present** at all 16 block-starts.
- **⏭ SKIP-IF (number-reading only):** you may skip *reading the board numbers* when every block-start
  shows a board AND counts reconcile — presence + clean counts fix the block structure. **Read the
  numbers only if a board is missing/ambiguous at an expected position** (indicates an extra/aborted
  clip or a run-order change; then the number is the only ground truth).
- **🚫 NEVER:** apply the rename without the presence GATE; never trust metadata row-order over a board
  when a board is missing; never auto-apply on a day whose counts don't reconcile.

## Step 2 — ROI inference — `$PROJ/roi_inference/`
- **Do:** `launch-roi.sh` (10-frame slice pass) → if failures, `launch-roi-full.sh` (full-frame pass)
  → if any clip still fails both, `propagate_roi.py` (consensus-ROI fallback) → `slp_to_yaml.py`.
- **✅ GATE:** `safety_checks --roi-coverage $VID` returns OK (every video has a `.rois.yml`).
  If `propagate_roi.py` ran, its static-maze guard passed AND you opened `roi_consensus_qc/MONTAGE_*.png`
  and confirmed each borrowed ROI lands on the maze.
- **⏭ SKIP-IF:** skip `launch-roi-full.sh` only if the slice pass already produced 100% ROI; skip
  `propagate_roi.py` only if the full pass produced 100% ROI.
- **🚫 NEVER:** hand-roll ROI borrowing — use `propagate_roi.py` (it is the documented consensus median
  + QC). Never proceed to pose/decisions with <100% `.rois.yml`.

## Step 3 — Pose inference (v5) — `$PROJ/pose_inference/`
- **Do:** `launch-pose.sh` (writes `<stem>.slp` next to each MP4).
- **✅ GATE:** `#.slp == #.MP4`; spot-check coverage ≈100%, mean nodes ≈14/15.
- **⏭ SKIP-IF:** a `<stem>.slp` already exists for every video (the launcher skips existing — safe resume).
- **🚫 NEVER:** use `finetune_final` — the corrected model is **`finetune_final_v5`** (L/R paw fix).

## Step 4 — CALIBRATION GATE (mandatory — this is the one that was missing)
- **Do:** verify px/cm BEFORE trusting any cm metric. Either (a) an independent ChArUco board reading
  for this camera/session, or (b) run gait once and check the distribution.
- **✅ GATE:** `safety_checks --gait <gait_per_stride.csv>` returns OK — px/cm **uniform** (spread <1.4×)
  and in range, and body_length_cm median plausible (4–11 cm). A bimodal/2× px/cm or an implausible
  body length is a **FAIL**.
- **⏭ SKIP-IF:** never — always verify calibration for a new camera/cohort.
- **🚫 NEVER:** accept the per-video `segment2` px/cm blindly. If the gate FAILS, pin the verified
  constant via `params["PX_PER_CM_OVERRIDE"]` (from `output/calibration/calibration_manifest.*`) and
  re-run. Record the constant in the manifest.

## Step 5 — Decisions — `$PROJ/analysis/run_stride_<cohort>_decisions.py` → `apply_decision_rules.py`
- **✅ GATE:** result reports `skipped == 0` (every video matched meta by day+mouse); `decisions_clean.csv` written.
- **⏭ SKIP-IF:** `decisions_clean.csv` already exists AND pose/ROI unchanged since.
- **🚫 NEVER:** proceed if videos were skipped (means a rename/meta mismatch — fix the join first).

## Step 6 — Gait — `$PROJ/analysis/run_stride_<cohort>_gait.py`
- **Do:** run with `PX_PER_CM_OVERRIDE` set to the verified constant (Step 4).
- **✅ GATE:** re-run `safety_checks --gait gait_per_stride_filtered.csv` → OK (px/cm now uniform =
  verified constant; body length plausible).
- **⏭ SKIP-IF:** filtered gait already exists AND calibration/pose unchanged.
- **🚫 NEVER:** ship gait whose calibration gate is not green.

## Step 7 — Data-quality log
- **Do:** append to `DATA_QUALITY_FLAGS.md` anything abnormal this day: propagated-ROI clips + reason
  (clutter in maze), any calibration override used, fps anomalies, excluded mice.
- **✅ GATE:** none (bookkeeping) — but do not skip; downstream analysis reads this file.

---

## Quick per-day sequence (for reference — still follow the GATES above)
```bash
# 0 preflight; 1 rename (board presence); 2 ROI:
ROI_VIDEOS=$VID ROI_OUT=$OUT bash roi_inference/launch-roi.sh
# (full pass only if failures) → propagate_roi.py only if full pass <100% → slp_to_yaml.py
$PY -m stride.stages.safety_checks --roi-coverage $VID || { echo STOP; exit 1; }
# 3 pose:
POSE_VIDEOS=$VID bash pose_inference/launch-pose.sh
# 4 CALIBRATION GATE + 5 decisions + 6 gait:
$PY analysis/run_stride_<cohort>_decisions.py && $PY analysis/apply_decision_rules.py <decisions.csv>
$PY analysis/run_stride_<cohort>_gait.py
$PY -m stride.stages.safety_checks --gait <gait_per_stride_filtered.csv> --strict || { echo STOP; exit 1; }
# 7 log to DATA_QUALITY_FLAGS.md
```
