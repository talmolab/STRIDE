# STRIDE release plan (v1.0.0)

Living plan for taking STRIDE from a lab-internal pipeline to a public, user-ready release.
Update this file whenever a step lands or a decision is made, so a fresh session can pick up from here.
Last updated: 2026-11-10.

## 1. Goal and ground rules

**Goal.** People who cite the paper can run STRIDE on their own recordings, smoothly and with correct logs.
The release is **STRIDE 1.0.0**.

**Ground rules (owner's decisions, 2026-11-09).**
1. The rules are not being improved. No metric definition, threshold, filter or statistical recipe changes.
   The work is: correct the code and logic so data runs smoothly and is logged correctly.
2. The one approved bug fix that changes numbers is the snout-probe gap fill (see section 5).
3. Anything the lab already does outside the package is brought in as a faithful port, not a redesign.
4. Findings about definitions are documented exactly as computed, not changed (section 6).
5. Every change is checked against "numbers unchanged" gates (section 4) before it is proposed.

**Workflow.** Work on branch `next`; never push to `main`. One pull request per step. Commits are authored by the
repository owner only, with no co-author or session trailers, and PR descriptions carry no generated-by footer.
Nothing is committed or pushed without the owner asking.

## 2. What STRIDE 1.0.0 offers

Three functions for three audiences, one repository.

| Function | For | Goes in | Comes out | Command |
|---|---|---|---|---|
| **STRIDE T-maze** | labs with their own trained pose and ROI models | mp4 files, two model paths | per-trial and per-stride CSVs, run logs | `stride tmaze` |
| **STRIDE Gait** | pose tracking only, no maze ROI | mp4 + pose files (or a pose model), px/cm from ChArUco or typed in | per-stride and per-video gait CSVs | `stride gait` |
| **STRIDE Lab** | the authors' own data intake | raw camera mp4 + recorder metadata | sort, whiteboard check, rename, then the T-maze pipeline with the lab's final models | `stride-lab` |

- ChArUco px/cm calibration is part of the public pipelines (`stride calibrate`). Rename is not offered publicly.
- The genotype statistics (mixed models, classifier) stay available as separate, optional steps; default off.
- Scale precedence: a value the user gives, then the in-view ChArUco measurement, then the corridor width from the ROI.

**Repository layout (target).**

```
src/stride/     public package: tmaze, gait, calibrate, the stages
lab/            lab intake package (sort, whiteboard check, rename); paths and models come from a config outside the repo
tests/          mock data generators and tests for all three functions
docs/           tmaze.md, gait.md, lab-workflow.md
README.md       the two public pipelines only
PLAN.md         this file
```

**Requirements from the software team (2026-11-09).**
- Command line works on Windows, macOS and Linux (pathlib, no shell-outs, spawn-safe workers, no symlinks).
- Config then CLI: one YAML holds the mp4 location, model paths, output folder and options; CLI flags override it;
  the pure-CLI form also works. `stride init-config` writes a commented template.
- Every step can be switched on or off in the config. Inference and analysis steps default on; genotype statistics
  default off.
- The documentation site (https://talmolab.github.io/STRIDE-docs/) covers all of these steps.
- Version 1.0.0 at release.
- Possibly a text user interface (TUI) on top of the CLI (section 7).

## 3. Steps and status

| Step | Content | Status |
|---|---|---|
| 1 | Core fixes: launching, file discovery, stage chaining, honest exits, run manifests and warnings, snout fix, mock test set | merged 2026-11-10 (PR #9) |
| 2 | ChArUco calibration (`stride calibrate`) + `stride tmaze` (any file name, optional metadata, decision-rules step, ROI best-instance + consensus fallback, config-then-CLI, step switches) | built and checked; PR opened 2026-11-10 |
| 3 | `stride gait` (start from a folder of videos + pose files, px/cm without ROI, keypoint-name mapping, orientation warning) | not started |
| 4 | `lab/` intake (`stride-lab`), checked by reproducing a real cohort's canonical outputs | intake code built and checked against four real days' rename logs (side branch `step2-rename-calib`, `lab/`); packaging and integration with `stride tmaze` not started |
| 5 | Release: v1.0.0 version bump and tag, docs site, README restructure, CI on three operating systems, repository hygiene (lab paths out of config.yaml / RUNBOOK.md / docstrings), model and example-data deposit, software DOI, visibility switch | not started |

Step 2 notes.
- GPU launch checked on a 24-frame real clip: pose and ROI ran on the requested GPU for `cuda:0` and `cuda:1`.
- OpenCV is pinned to `>=4.8,<4.14`: 4.14 and 5.x read the same ChArUco board about 0.1 % lower than 4.13, which is
  what the lab's recorded px/cm values were measured with.
- The automatic choice of calibration clips does not reproduce the lab's hand-picked clips (26.05 against 26.08 on one
  cohort); `scale.charuco.clips` names clips explicitly. Open: whether the lab pipeline pins its recorded values or
  names its clips.
- ROI selection now follows the lab's own scripts (frames from the middle of the video, every instance evaluated, all
  16 corners required, full-video pass, then the folder's consensus ROI). This changes what `stride roi-inference`,
  `stride run` and `run_inference` produce compared with the first release.
- `stride tmaze` skips folders named `waste`; a run with left-out videos exits 0 and lists them (`--strict` makes it
  an error); the consensus ROI is refused above 25 px drift (`roi.drift_max_px`).
- Not yet run on Windows or macOS.

## 4. How changes are checked

- **Mock test set** (`tests/mock/`): seeded synthetic T-maze data (25 scenarios, 112 clips) with planted truth; a runner
  for every entry point; `pytest tests`. Generated data is git-ignored.
- **Numbers-unchanged gates** before every PR:
  1. every mock scenario that worked before gives identical output files on all pre-existing columns;
  2. one real cohort, run exactly as the lab drivers do, gives identical decisions and gait files;
  3. the full mock runner and the inference launch test are re-run for before/after counts.
- **Step 2 adds:** `stride tmaze --skip-inference` on a real cohort must reproduce the canonical decisions,
  cleaned decisions and gait files; the decision-rules port must equal the lab's cleaned files on two more cohorts;
  the ROI port must reproduce the lab's ROI files; calibration must reproduce the lab's recorded px/cm values;
  the rename port must reproduce the lab's rename logs.

Baseline on the original code (122 mock runs): 54 work, 19 wrong or misleading, 39 crash, 10 silently wrong.
After step 1: 108 / 13 / 0 / 1. Inference launch on CPU: 8 of 24, then 24 of 24.

## 5. Decisions and trace (2026-11-09)

- **Snout probes.** The notebook that produced the first cohort's probe counts gap-filled the snout; the package
  stored the raw snout from its first release, which split one probe at every tracking dropout. Fixed (gap fill,
  `max_gap=7`). The decision stage was re-run for the cohorts that carried raw-snout counts, into side folders
  (`decisions_snoutfix.v1/`); only the four probe columns change. Switching the lab's registry to those folders and
  recomputing the probe figures is the owner's open decision.
- **Body-frame rotation.** The rotation in `normalize_to_egocentric` is exact only when the animal heads along the
  image x-axis. A corrected sign was tested and withdrawn: with body yaw, both signs misread a planted track width by
  a similar amount in opposite directions, and step width / Euclidean stride length do not depend on the maze angle.
  All lab cohorts were recorded with the stem left to right. Kept as is; the pipeline warns when the ROI stem is not
  left to right.
- **No ROI-direction feature.** Reading the direction from the ROI would be a new rule; not built.
- **Audience and scope.** Three functions (section 2); rename stays internal; calibration is public.
- **One PR per step.**

## 6. Documented as computed (not changed)

To be written into the docs exactly as the code computes them:
`n_reversals` (segment order as coded: at least 1 on a straight run); the gate (in practice 0.5 s after video start);
`step_length_cm` depends on which paw landmark was auto-picked per side; `duration_ms = (s1 - s0) / fps`;
stride length is body-relative; the "forward" filter keeps strides along the stem axis; step width and step length are
body-relative and sensitive to body yaw; the classifier ranks features outside the cross-validation folds; a trial
with no reward row is scored as not correct in the existing entry points.

## 7. Backlog and ideas

- **TUI** for the CLI (e.g. a guided screen that fills the config, shows per-step status and runs it). After the CLI
  and config schema are stable.
- **Docs site**: pages for T-maze, Gait, calibration, config reference, outputs glossary, troubleshooting, lab workflow.
- **CI**: pytest on Linux, macOS and Windows for every PR.
- **Production logic still outside the package after step 2** (if any): list here when step 2 reports.
- **Release hygiene**: generic example config only; lab config, RUNBOOK and internal paths moved out; decide whether
  the public history is squashed; author e-mail in `pyproject.toml`.
- **Models and data**: deposit the trained pose and ROI models with their training-library versions, plus a small
  example clip, and add the DOIs to the README.

## 8. Working copies and pointers (lab machine; remove this section before the repository goes public)

- `/home/exx/leo-local/STRIDE` — branch `next`. Untracked working reports kept out of git:
  `CHANGES_NEXT.md`, `tests/mock/BASELINE_REPORT.md`, `tests/mock/AFTER_FIX_REPORT.md`.
- `/home/exx/leo-local/STRIDE-tmaze` — branch `step2-tmaze` (from `next`): `stride tmaze`.
- `/home/exx/leo-local/STRIDE-step2` — branch `step2-rename-calib` (from `main`, new files only): calibration + lab intake.
- `/home/exx/vast/leo/stride` — the copy the lab's production runs import. Never edit it during a run; it is two
  commits behind `main` and carries the snout fix as an uncommitted edit. Point production at a released tag once
  v1.0.0 exists.
- Lab-side records: `Tmaze-core/EVENTS.md` (entries of 2026-11-09), the to-do list `stride-public-release`,
  `2026-06-15-T-maze-analysis/output/snoutfix_rerun/` and `output/rotation_fix_comparison/`.
