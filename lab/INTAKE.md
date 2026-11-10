# Lab intake (internal): sort, whiteboard check, rename

**Internal lab layer — not part of the public STRIDE pipeline.** This page is the reference for the intake stages of
`stride-lab` (the first half of `docs/lab-workflow.md`): a camera card becomes one folder per recording day, every
clip named `Day{N}_{Mouse}_Trial{T}.MP4` with the mouse identity verified against the whiteboard.

The code lives in `lab/stride_lab/intake/` and imports the public package (`stride.stages.scale_calibration`,
`stride.run_manifest`); the public package never imports it. No lab folder, model or cohort name is written into the
code: folders, camera conventions and the board come from the cohort config (`lab/cohort.example.yaml`) or the
command line.

```
sort  ->  boards  ->  [a person reads the whiteboards and signs off]  ->  rename  ->  calibrate
```

```bash
pip install -e . && pip install -e lab/                 # the public package, then the lab layer
I="stride-lab --config my_cohort.yaml"
$I check                    # per day: what exists, what is next (writes nothing)
$I auto                     # dry run: shows what would happen
$I auto --apply             # sorts, finds the boards, then STOPS (exit code 2)
#   ... read <records>/day<N>/wb_montage.png, then:
$I verify --days 1-12 --verifier "your name"
$I auto --apply             # renames, then measures px/cm on every day (docs/lab-workflow.md)
```

Without `--config`, pass `--cohort-dir` (and the other options) on each command.

## Inputs

1. **The clips**, all in the cohort folder with the names the camera gave them. Recording order must be the lexical
   order of the names. Default: DJI names, `DJI_YYYYMMDDhhmmss_NNNN_D.MP4`. One clip = one trial.
2. **The recorder export `<date>_<cohort>_trials_full.csv`** in the same folder — one row per mouse per day:

   | column | meaning |
   |---|---|
   | `day` | `DAY0`, `DAY1`, ... (`DAY0` = preference phase) |
   | `date` | `YYYY-MM-DD`; must equal the date in the clip names of that day |
   | `mouse` | the id written on the whiteboard |
   | `T1` ... `T9` | one cell per trial that was run. **The number of non-empty cells = the number of clips of that mouse that day.** |
   | `notes` | free text; if it mentions the board ("Board show on second trial"), all clips of that mouse are searched |

   Row order within a day = the order the mice are expected to be run in. The newest export by name is used.
3. **The whiteboard convention.** At the start of each mouse's first trial a whiteboard with the mouse id is held in
   view for a moment. It is the **only** anchor that ties a block of clips to a mouse: counts can agree by coincidence,
   and the mice may have been run in another order than the metadata rows.
4. **A ChArUco board in view** for the scale — see `docs/CALIBRATION.md`.

## Stages

| stage | command | what it does | writes |
|---|---|---|---|
| check | `check` | per-day status and next step | nothing |
| sort | `sort [--apply]` | flat clips -> `<DATE>-DAY{N}/`; the export CSVs are copied into each day's `metadata/`. Refuses unless every day has exactly as many clips as metadata trials (`--allow-mismatch` overrides — never blindly). | `sort_log.csv`, `SORT_REPORT.md` |
| boards | `boards [--days ...]` | per day: a **provisional** plan (clips in recording order cut into blocks by the metadata trial counts), the whiteboard crop of each block start, one montage, the sign-off table. Never renames. | `<records>/day<N>/rename_plan.csv`, `reconciliation_report.txt`, `wb/*_wb.png`, `wb_montage.png`, `BOARD_VERIFICATION.md` |
| sign-off | `verify --days N ...` (or edit the file) | records what was read off the boards, sets `VERDICT: APPLY` | `BOARD_VERIFICATION.md` |
| rename | `rename [--days ...] [--apply]` | validates plan + sign-off; with `--apply` a two-phase atomic rename; sidecars (`.slp`, `.rois.yml`, `.pred.slp`) move with their clip | `rename_log.csv` (the rename is reversible from it), `rename_plan.applied.csv` |
| calibrate | `calibrate` | px/cm on every recording day with the public ChArUco measurement, and the camera-moved check (docs/lab-workflow.md) | `<records>/calibration/` |
| waste | `waste --day N CLIP...` | takes an extra / aborted clip (and its sidecars) out of a day | `<cohort>/waste/<day folder>/`, `waste/waste_log.csv` |
| auto | `auto [--apply]` | sort -> boards -> stop until signed -> rename -> calibrate; safe to re-run, each stage skips what is done | all of the above |

Everything is a dry run unless `--apply` is given. Every stage appends one line per event to `<records>/intake.log`
and writes an `intake_<stage>_run_manifest.yaml` next to its outputs. `<records>` defaults to `<cohort>/_intake`.

Exit codes: `0` done · `1` a stage refused or failed (reason on the last line) · `2` `auto` stopped at the sign-off.

## The sign-off gate

`boards` stops with `VERDICT: PENDING`. Open `wb_montage.png`: each tile is the whiteboard found in one block-start
clip, captioned with the clip number and the mouse the provisional plan expects. Read **every** number.

- All numbers equal the expected mice -> `verify --days 3 --verifier "name"`.
- A block shows another mouse (the run order differed from the metadata) ->
  `verify --days 3 --reads 1=302,2=301,3=303 --rebind`. The plan is re-bound to the mice that were read; refused unless
  every block has a read, each mouse is used once and the block has as many clips as that mouse has trials. The
  metadata-order plan is kept as `rename_plan.provisional.csv`.
- A block is one clip too long -> an aborted take hides in it, usually the shortest clip. Decide which, move it out
  with `waste`, run `boards` again.
- No board in a tile -> look at the 3-5 clips around the block start (`--sweep-all` searches every clip of the day,
  `--whole-clip` searches past the first 12 s).

`rename --apply` refuses until `BOARD_VERIFICATION.md` has a line starting with `VERDICT: APPLY`. If the clips of a
day change after the sign-off, the stage refuses and says so.

**The one documented shortcut.** For a cohort whose mice are *always* run in the metadata order the numbers need not
be read: `--fixed-run-order` (on `verify` or `auto`) signs off when the counts reconcile exactly and a board was found
at every block start (the first mouse of a day may have none). It is logged loudly and recorded as "NUMBERS NOT READ".
It cannot see two mice run in swapped order — if the run order is not guaranteed, read the numbers.

## How the whiteboard is found

The lab's transient detector: frames sampled every 12, scaled to 960 x 540; the background is the median of about
nine sampled frames; only brightness that is new against the background is kept (threshold 45), so the lit floor
panel and a taped ChArUco board disappear; a candidate region covers at least 1500 px, fills at least half of its
bounding box and has a Canny edge density of at most 10 % (rejects glare and grids). The best region of the clip is
saved, turned upright (`crop_rotation`). The detector does not read the number.

## Layout after intake

```
<cohort>/
  2027-01-05-DAY1/
    Day1_301_Trial1.MP4 ...          <- what pose / ROI inference and the analysis stages read
    metadata/<export CSVs>
  waste/2027-01-05-DAY1/...          <- extra clips taken out (waste_log.csv)
  sort_log.csv  SORT_REPORT.md
<records>/                           (default <cohort>/_intake)
  intake.log
  day1/ rename_plan.csv  wb/  wb_montage.png  BOARD_VERIFICATION.md  rename_log.csv ...
  calibration/ px_per_cm.txt  per_day_px_per_cm.csv  CALIBRATION.md  day_decisions.csv  day<N>/<clip>_checkerboard.png
```

The analysis needs a `day, mouse, reward` table: `stride-lab pipeline` builds it from the same recorder export
(`<output>/metadata_for_stride.csv`).

## Rig-specific settings (config `camera:` / options)

| setting | default (DJI profile) | option |
|---|---|---|
| raw clip name | `DJI_YYYYMMDDhhmmss_NNNN_D.MP4` | `raw_pattern` — a regex with the named groups `date` (YYYYMMDD) and `seq` |
| clip suffix | `.mp4`, any case | `extensions` |
| whiteboard crop turn | 90 degrees counter-clockwise | `crop_rotation: ccw90 | cw90 | 180 | none` |
| rail cross-check region | `820,1320,850,2500` on 3840x2160 frames | `rail_crop` (or `none`) |

A renamed clip keeps the suffix of its source (`.MP4` stays `.MP4`).

## Known limits

- **A person reads the boards.** The tool finds and shows them.
- **1-2 digit mouse ids are refused by default** (`--allow-placeholders` / `allow_placeholders: true`): that is what
  the recorder writes before real ids are typed in.
- **DAY0** is sorted but not planned or renamed unless `boards --include-day0` is used.
- **Detector thresholds** are those of the lab rig. A board that stays in view for the whole searched window is part
  of the background and is not found.
- **GoPro batches** were renamed with an older, different procedure (manual screening of GX-named files and a later
  correction table); it is not ported. A GoPro cohort would need a `raw_pattern` that carries the recording date and
  names whose lexical order is the recording order.
- Mouse ids must not contain `_Trial` and should not contain spaces.
- With `--workers > 1` the clips are read in threads; the result does not depend on the number of workers.
