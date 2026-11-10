# STRIDE Lab: the authors' own workflow (`stride-lab`)

**This is not needed to use STRIDE.** It is how the authors take a recording session from the camera card to the
STRIDE tables: the clips are renamed after a whiteboard check, the scale is checked on every recording day, and then
the public T-maze pipeline (`stride tmaze`, [docs/tmaze.md](tmaze.md)) runs with the lab's finalized models. No
analysis code lives in the lab layer; every number comes from the public package.

```
raw clips + recorder export
   -> sort into day folders -> whiteboard finder -> [PERSON: sign-off] -> rename
   -> px/cm on every day (camera-moved check) -> [PERSON: only if a day is flagged]
   -> pose + ROI inference -> ROI files, consensus ROI -> decisions -> decision rules -> gait -> gate -> summary
```

## Install

```bash
pip install -e .          # the public package (repository root)
pip install -e lab/       # adds the `stride-lab` command
pytest                    # public tests + lab tests (the lab tests are left out when the lab layer is not installed)
```

## What goes in

1. **The clips**, all in one cohort folder with the names the camera gave them (default: DJI names,
   `DJI_YYYYMMDDhhmmss_NNNN_D.MP4`; recording order = name order; one clip = one trial).
2. **The recorder export** `<date>_<cohort>_trials_full.csv` in the same folder: one row per mouse per day with
   `day`, `date`, `mouse`, `sex`, `genotype`, `reward` and one `T1..T9` cell per trial that was run.
3. **A cohort config** (YAML, kept outside the repository): folders, the finalized pose and ROI models, the board,
   the thresholds. Template: [`lab/cohort.example.yaml`](../lab/cohort.example.yaml).

The naming system is fixed: every clip ends up as `Day{N}_{Mouse}_Trial{T}.MP4`.

## The one command

```bash
stride-lab run --config my_cohort.yaml            # dry run of the intake: shows what would happen
stride-lab run --config my_cohort.yaml --apply    # sorts, finds the whiteboards, then STOPS (exit code 2)
#   read <records>/day<N>/wb_montage.png, then sign off:
stride-lab verify --config my_cohort.yaml --days 1-12 --verifier "your name"
stride-lab run --config my_cohort.yaml --apply    # renames, checks the scale, runs inference and the analysis
```

`run` is safe to repeat: each stage skips what is done. `--days 1-3` limits the intake, the scale measurement and
the inference to those days; the analysis always covers every day of the cohort that is ready.

| command | what it does |
|---|---|
| `check` | per day: clips, renamed or not, plan, sign-off, next step. Writes nothing |
| `sort [--apply]` | flat clips -> `<DATE>-DAY{N}/`; refuses unless every day has as many clips as metadata trials |
| `boards` | provisional rename plan, whiteboard crop of each block start, montage, sign-off table. Never renames |
| `verify --days N` | records the board reads and `VERDICT: APPLY` (the sign-off) |
| `rename [--apply]` | validates plan + sign-off; two-phase atomic rename; pose / ROI sidecars move with their clip; `rename_log.csv` |
| `waste --day N CLIP` | takes an extra or aborted clip out of a day |
| `calibrate` | px/cm on every recording day and the camera-moved check; `--decide` records a decision for a held day |
| `pipeline` | scale check -> inference -> the public T-maze pipeline -> gait gate -> summary (`--plan` prints the inference workers and stops) |
| `summary` | rewrites the cohort summary and the data-quality notes from what is on disk |
| `auto` | sort -> boards -> stop until signed -> rename -> calibrate |
| `run` | `auto`, then `pipeline` |

Exit codes: `0` done; `1` a stage refused or failed (the reason is printed); `2` stopped for a person (sign-off, or
a day held by the scale check). The intake stages are described in detail in [`lab/INTAKE.md`](../lab/INTAKE.md).

## The two decisions a person makes

**1. The whiteboard sign-off.** The whiteboard held up at the start of each mouse's block is the only anchor that
ties a block of clips to a mouse. `boards` writes one montage per day; a person reads every number and signs off
(`verify`). Nothing is renamed before that. For a cohort whose mice are always run in the metadata order,
`--fixed-run-order` signs off without reading the numbers when the counts reconcile and a board is found at every
block start; it is logged loudly and cannot see two mice run in swapped order.

**2. A recording day flagged by the scale check.** See the next section. The run reports the day, finishes the other
days, and exits with code 2 until the decision is recorded:

```bash
stride-lab calibrate --config my_cohort.yaml --decide 7=accept_own --by "name" --note "board was re-taped"
#   accept_own      process the day on its own, with its own px/cm  (tables in <output>/own_scale/DAY7/)
#   accept_cohort   process the day with the cohort value           (it joins the cohort tables)
#   exclude         leave the day out
#   clear           remove an earlier decision
```

## How px/cm is monitored

The scale is not a pinned number. On every recording day `clips_per_day` clips (default 3, spread over the day,
chosen automatically) are read with the public ChArUco measurement ([docs/CALIBRATION.md](CALIBRATION.md)).

- **Day value** = median of the day's clip readings. **Cohort value** = median of the day values, two decimals.
- The cohort value is what the gait stage uses for every day that agrees with it.
- A day more than `warn_pct` (0.5 %) from the cohort median: **warning**, processed with the cohort value.
- A day more than `stop_pct` (1 %) from it: **held** until a decision is recorded (the camera or the board moved).
- The cohort value is compared with `reference_px_per_cm` (the previous cohort on the same rig): warning only.
- A day with no board reading cannot be checked: warning (or held, with `on_unmeasured_day: stop`).
- `calibration.px_per_cm` in the config is an explicit value for the analysis; the days are still measured and
  compared with it.
- As a second check the maze's position in the image is read from each day's ROI files: the largest distance of a
  maze corner from where it was on the first day, and the stem direction. Above `drift_warn_px` (25 px) or
  `stem_angle_warn_deg` (1 degree): warning. This never changes an output.

The cohort value is recomputed whenever a day is added, and the gait tables are then recomputed for the whole cohort.

Files under `<records>/calibration/`: `per_day_px_per_cm.csv`, `px_per_cm.txt`, `CALIBRATION.md` (the report),
`day_decisions.csv` (append-only), `day<N>/<clip>_checkerboard.png` (QC images).

## Inference

`inference.devices` x `inference.workers_per_device` workers run per day through the public commands
(`stride pose-inference`, `stride roi-inference` with `--worker-id / --num-workers`), pose first, then ROI; a worker
that fails stops the run with its log file named (`<records>/inference/`). The public pipeline then writes the ROI
regions files, borrows the day's consensus ROI for clips with a hidden maze corner, and infers anything the workers
left. Days that still hold camera-named clips are refused. With `inference.sidecars: videos` the pose and ROI files
end up next to the clips (the lab's convention); with `output` only under the output folder.

## What comes out

```
<output_dir>/
  decisions/  decisions.csv  decisions_clean.csv  events.csv  metrics.csv        (docs/tmaze.md)
  gait/       gait_per_stride.csv  gait_per_step.csv  gait_per_video.csv  gait_per_stride_filtered.csv ...
  roi/  pose/  scale/  SUMMARY.md  tmaze_run.log                                 the public pipeline's own outputs
  own_scale/DAY<N>/...            a day accepted with its own px/cm (a separate run with the same layout)
  metadata_for_stride.csv         day, mouse, reward, sex, genotype built from the recorder export
  COHORT_SUMMARY.md  cohort_summary.csv     per day: clips, pose / ROI files, trials, decisions, accuracy, strides, px/cm
  DATA_QUALITY.md                 what a person should look at
  roi_position_per_day.csv
<records_dir>/                    (default <raw_dir>/_intake)
  intake.log                      one line per stage event, every stage
  day<N>/  rename_plan.csv  wb/  wb_montage.png  BOARD_VERIFICATION.md  rename_log.csv
  calibration/  inference/
```

`COHORT_SUMMARY.md` reports counts and the per-day accuracy (share of the decided trials that entered the rewarded
arm). It computes no learning criterion. `DATA_QUALITY.md` lists flagged days and recorded decisions, maze-position
warnings, clips that use the consensus ROI, clips taken out as waste, clips the analysis left out, metadata that
does not match, and days that are not ready.

After the gait stage the lab's gait calibration gate runs
(`python -m stride.stages.safety_checks --gait <filtered> --expect-pxcm <value>`); a failure stops the run.

Every stage writes a run manifest (`*_run_manifest.yaml`) next to its outputs.

## What is specific to the DJI rig

Set in the config's `camera:` block: the raw clip name pattern (`DJI_YYYYMMDDhhmmss_NNNN_D.MP4`), the turn applied to
whiteboard crops (90 degrees counter-clockwise), and the corridor region used for the optional rail cross-check of
the scale (3840x2160 frames). The earlier GoPro batches were renamed with a different procedure that is not part of
this package.
