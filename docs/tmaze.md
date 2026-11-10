# STRIDE T-maze: the automated pipeline (`stride tmaze`)

Videos of the T-maze task and your two trained SLEAP models go in; per-trial and per-stride CSV
tables come out. One command, no file renaming, no genotype statistics.

```
discover -> distortion check -> scale (px/cm) -> pose inference -> ROI inference (+ consensus ROI)
         -> decisions -> decision rules -> gait
```

The command adds no computation of its own: every step calls the same stage function the rest of
the package uses, so the tables are the same whichever way the stages are started.

## What you need

| | |
|---|---|
| Videos | `.mp4` (any letter case), any file names, in one folder or many. Names must be unique: every table is keyed by the file name. |
| Pose model | A trained SLEAP top-down model (one or two folders: centroid, centered instance) with the 15-node mouse skeleton: `snout, mouth, forepawR2, forepawR1, forepawL1, forepawL2, hindpawR2, hindpawR1, hindpawL2, hindpawL1, tailbase, tail1, tail2, tail3, tailtip`. The decision step needs `snout, forepawL2, forepawR2, hindpawL2, hindpawR2, tailbase`; the gait step needs `snout, tailbase, tailtip` and the four hind-paw nodes. |
| ROI model | A trained SLEAP top-down model with the 23-node maze skeleton: 16 corners (`arm_right.top_left/top_right`, `junction.top_left/top_right/bottom_left/bottom_right`, `arm_left.bottom_left/bottom_right`, `segment1..4.top_left/bottom_left`) and 7 centroids (`<region>.centroid`). |
| Runtime | The analysis runs on Windows, macOS and Linux with `pip install -e .`. The two inference steps additionally need [sleap-nn](https://github.com/talmolab/sleap-nn) and, in practice, a GPU. |
| Recording | Camera below the transparent maze floor, maze stem running left to right in the image, at least 100 fps and 20 px/cm for the gait tables (decisions work from 30 fps). |

## Run it

```bash
# everything on the command line
stride tmaze /data/videos \
    --pose-model /models/pose/centroid --pose-model /models/pose/centered_instance \
    --roi-model  /models/roi/centroid  --roi-model  /models/roi/centered_instance \
    --out /data/stride_out

# or from a config file (any option on the command line overrides the file)
stride init-config tmaze -o my_tmaze.yaml      # writes a commented template
stride tmaze --config my_tmaze.yaml
stride tmaze --config my_tmaze.yaml --out /somewhere/else --workers 8

# pose and ROI files already exist (next to the videos, or in --pose-dir / --roi-dir)
stride tmaze /data/videos --out /data/stride_out --skip-inference
```

`VIDEOS` may be a folder (searched recursively), a single file, a pattern such as
`"/data/2026-*/*.mp4"`, or several of these. Folders named `waste` are skipped
(`exclude_dirs` in the config).

### Steps

Each step can be switched off (`--skip STEP`, or `steps:` in the config) and the run can end early
(`--stop-after STEP`).

| step | what it does | default |
|---|---|---|
| `distortion_check` | Samples 6 videos and tests a ChArUco board in view for lens distortion. Report only (`scale/distortion_check.csv`); STRIDE does not undistort. | on |
| `calibrate` | px/cm from the ChArUco board in view (see *Scale*). | on |
| `pose_inference` | Runs your pose model on every video that has no pose file yet. | on |
| `roi_inference` | Runs your ROI model on every video that has no ROI file yet: 10 frames from the middle of the video, the whole video if none of them shows the full maze, then keeps the best frame. | on |
| `roi_consensus` | A clip in which a maze corner is hidden for its whole length gets the consensus ROI of the other clips in the same folder (see below). | on |
| `decisions` | Arm choice, entry time, latencies, snout probes, junction exploration per trial. | on |
| `decision_rules` | Writes `decisions_clean.csv`: a trial stays a decision only if the entry cleared the strict 95 % body-in-arm threshold and happened within the decision window (150 s). | on |
| `gait` | Stride detection, per-stride metrics, the stride filters. | on |
| `mixed_models`, `classifier` | The genotype statistics (existing stages, unchanged). Need `stats_config` and a `genotype` column in the metadata. | **off** |

A step that is off never causes a traceback: later steps use the files that exist and say which
videos they leave out and why. `--skip-inference` switches both inference steps off.

### Running again

A second run does not repeat finished work: inference is skipped per video when its pose / ROI
file exists, and the analysis steps are skipped when their inputs and settings are unchanged
(`tmaze_state.json`). `--force` redoes the analysis steps. To redo inference for a video, delete
its pose / ROI file from the output folder.

### Exit code

`0` when the run completed, also when some videos were left out (they are listed, with the reason,
on screen and in `SUMMARY.md`). `--strict` turns any left-out video into exit code 1. Anything
that stops the run exits non-zero with one line saying why.

## Metadata (optional)

Without metadata the arm choice, the timing and the probe columns are written and the
correctness columns stay **empty**: STRIDE cannot know which arm was rewarded.

A metadata CSV (several may be given) has trial rows and/or mouse rows. Column names are
case-insensitive.

| rows | key columns | value columns |
|---|---|---|
| trial rows | `video` (file name, with or without extension) **or** `day` + `mouse` (+ `trial`) | `reward` (`L` / `R`); optional `day`, `mouse`, `trial`, `sex`, `genotype` |
| mouse rows | `mouse` | `sex` (or `Gender`), `genotype` |

`day` + `mouse` rows are matched to videos named `Day<N>_<mouse>_Trial<T>` (the day may be
written `Day1`, `DAY1` or `1`); for any other file name use the `video` column. A video without a
matching row keeps empty correctness; the run says how many.

## Scale (px/cm)

The gait tables are in centimetres. The scale is taken, in this order, from

1. `--px-per-cm` / `scale.px_per_cm`: a value you measured yourself;
2. the ChArUco board visible in the videos (`calibrate` step): the first video of every folder is
   measured, at least 6 in total, and the value is accepted when at least 3 clips give a reading
   and they agree within 1 %. Board geometry: `scale.charuco` or the `--board-*` options (default
   14 x 9 squares, 20 mm squares, 15 mm markers, `DICT_5X5_250`). The same measurement is
   available on its own as `stride calibrate` ([`CALIBRATION.md`](CALIBRATION.md)). To measure
   particular videos instead, list them under `scale.charuco.clips` in the config;
3. the corridor width of each video's ROI (`scale.maze_width_cm`, default 10 cm).

The value and where it came from are written to `scale/px_per_cm.json`, to `SUMMARY.md` and to the
run manifest; the ChArUco readings per clip and the QC images are in `scale/`. The decision tables
do not depend on the scale. The package pins `opencv-python < 4.14`: from 4.14 on the ChArUco
detector reads the same board about 0.1 % lower.

## Consensus ROI

When no frame of a clip shows all 16 maze corners, something (a calibration board, a hand)
covered a corner for the whole clip. The maze and the camera do not move within a session, so the
clip gets the per-vertex median of the ROIs detected in the other clips of the same folder
(`roi.session: all` treats the whole run as one session). Safeguards:

* the consensus is refused, and the clip left out, when the detected ROIs of the session differ
  by more than `roi.drift_max_px` (25 px) at any vertex: the maze or the camera moved;
* borrowed ROI files carry a `# consensus_roi_of: N clips` line and are never used as donors;
* `roi/roi_sources.csv` lists, per video, whether its ROI was `detected`, `consensus` or `missing`;
* `roi/roi_consensus_qc/` holds an overlay per recovered clip and montage sheets. **Open them**
  and confirm the ROI lands on the maze.

## Output folder

```
SUMMARY.md                     what ran, counts per step, px/cm and its source, warnings, videos left out
tmaze_run.log                  everything printed during the run(s)
tmaze_run_manifest.yaml        resolved settings (config + command line), models, code version, results
scale/px_per_cm.json           scale and its source      scale/distortion_check.csv
scale/px_per_cm.txt, calibration_summary.txt, calibration_readings.csv, *_checkerboard.png   the ChArUco measurement
pose/<video>.slp               pose files made by this run (videos that had none)
roi/<video>.rois.yml           ROI polygons (7 regions)  roi/<video>.preds.v2.best1.slp  the kept ROI prediction
roi/roi_sources.csv            detected | consensus | missing, per video
decisions/decisions.csv        one row per trial: arm entered, entry frame / ms, threshold, correctness
decisions/decisions_clean.csv  the same after the two decision rules (use this one for choices)
decisions/events.csv           region entry times, commit, junction exploration, snout probes, reversals
decisions/metrics.csv          per-trial summary: choice, latencies, junction exploration, probes, correctness
gait/gait_per_stride.csv       one row per stride: timing, speed, duty factor, step / stride length, sway, phase
gait/gait_per_stride_filtered.csv   after the confidence, edge, angular-velocity and direction filters
gait/gait_per_step.csv         one row per step             gait/gait_per_video.csv   per-video summaries
gait/pose_keypoint_confidence_by_video.csv   mean keypoint confidence per video (used by the confidence filter)
```

Column notes: `correct_TF` / `correct_bin` are empty when the reward is unknown; `day`, `mouse`,
`trial` are empty for file names that are not `Day<N>_<mouse>_Trial<T>` and have no metadata row;
the `px_per_cm` column of the gait tables stores **cm per pixel** (historical name).

## Warnings you may see

| message | meaning |
|---|---|
| `N video(s) were left out` | no pose file, no ROI, or an error in that clip; the reason is listed per video |
| `consensus (borrowed) ROI` | check the montage under `roi/roi_consensus_qc/` |
| `scale below 20 px/cm`, `fps < 100` | the recording is too coarse / too slow for the gait metrics; decisions are unaffected |
| `stem is not left to right` | the gait metrics assume the maze stem runs left to right in the image |
| `px/cm ... sentinel 1.0` | the scale of a clip could not be derived; its centimetre values are invalid |
| `may need undistortion` | the sampled ChArUco board shows lens distortion; STRIDE does not correct it |

## Limits

* The maze stem must run left to right in the image; the gait step warns otherwise.
* Gait needs paws visible from below, at least 100 fps and 20 px/cm.
* The pose and ROI skeletons must use the node names listed above.
* File names must be unique across the folders of one run.
* The two decision rules assume the 150 s trial of the published protocol; change
  `decision_rules.window_s` for another trial length.
