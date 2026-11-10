# STRIDE Gait: the gait-only pipeline (`stride gait`)

Videos with pose tracking go in; per-stride, per-step and per-video gait tables come out. No maze
and no ROI model are needed: you bring the pose tracking, STRIDE gets the scale (px/cm) and does
the gait analysis.

```
discover -> scale (px/cm) -> pose inference (only with a pose model) -> stride detection
         -> keypoint confidence -> stride filtering
```

The command adds no computation of its own: every step calls the same stage function the rest of
the package uses (`run_gait_extraction`, `compute_keypoint_confidence`, `filter_strides`), so the
tables are the same whichever way the stages are started.

## What you need

| | |
|---|---|
| Videos | `.mp4` (any letter case), any file names, in one folder or many. Names must be unique: every table is keyed by the file name. |
| Pose | Either a pose file per video (`<video name>.slp`, next to the video or in `--pose-dir`; another ending with `--pose-suffix`), or your trained SLEAP top-down pose model (`--pose-model`, one or two folders) to make them. One animal per video. |
| Keypoints | `snout`, `tailbase`, and at least one landmark per hind paw (`hindpawL1` / `hindpawL2`, `hindpawR1` / `hindpawR2`); `tailtip` is optional. Other node names are fine: map them (see [Keypoints](#keypoints)). |
| Scale | px/cm: a value you measured (`--px-per-cm`), or a ChArUco board in view. There is no other source: without one of the two the run stops. |
| Recording | The paws must be visible (camera below a transparent floor). The animal should walk left-right in the image, at 100 fps or more and 20 px/cm or more; the run warns otherwise (see [Warnings](#warnings-you-may-see)). |
| Runtime | The analysis runs on Windows, macOS and Linux with `pip install -e .`. Pose inference additionally needs [sleap-nn](https://github.com/talmolab/sleap-nn) and, in practice, a GPU. |

## Run it

```bash
# pose files exist, scale measured by you
stride gait /data/videos --px-per-cm 26.1 --out /data/gait_out

# pose files elsewhere, scale from the ChArUco board in view
stride gait /data/videos --pose-dir /data/pose --out /data/gait_out

# no pose files yet: your pose model makes them first
stride gait /data/videos --pose-model /models/pose/centroid --pose-model /models/pose/centered_instance \
    --px-per-cm 26.1 --out /data/gait_out

# or from a config file (any option on the command line overrides the file)
stride init-config gait -o my_gait.yaml        # writes a commented template
stride gait --config my_gait.yaml
stride gait --config my_gait.yaml --out /somewhere/else --workers 8
```

`VIDEOS` is a folder (searched recursively), one video, or a pattern such as `"D:/rec/*/*.mp4"`;
several may be given. Folders named `waste` are skipped (`exclude_dirs` in the config).

### Steps

| step | what it does | switch off with |
|---|---|---|
| discover | finds the videos, reads the metadata file | (always runs) |
| `calibrate` | px/cm from the ChArUco board in view; not used when `--px-per-cm` is given | `--skip calibrate` |
| `pose_inference` | runs your pose model on the videos that have no pose file; without a model those videos are left out and listed | `--skip pose_inference` |
| (keypoints) | checks the first pose file for the keypoints the analysis needs | (always runs) |
| `gait_extraction` | stride detection and the per-stride, per-step and per-video tables | `--skip gait_extraction` |
| `keypoint_confidence` | mean keypoint confidence per video (used by the confidence filter) | `--skip keypoint_confidence` |
| `stride_filtering` | writes `gait_per_stride_filtered.csv` | `--skip stride_filtering` |

In the config the same switches are the `steps:` block. `--stop-after STEP` ends the run after a
step. A second run does not repeat finished work (`--force` redoes it). Exit code: 0 when the run
completed, 1 when it stopped (the reason is printed and written to `SUMMARY.md`), 2 for a wrong
option. Videos that had to be left out do not change the exit code unless `--strict` is given.

## Keypoints

The gait stage reads these keypoints by name:

| STRIDE name | needed | used for |
|---|---|---|
| `snout` | yes | body axis, body length, snout sway |
| `tailbase` | yes | body position and speed, body axis, origin of the body frame |
| `hindpawL1`, `hindpawL2` | at least one | left hind paw (swings, strides, step and stride measures) |
| `hindpawR1`, `hindpawR2` | at least one | right hind paw |
| `tailtip` | no | tail-tip sway; without it the two tail-tip columns stay empty |

If your skeleton uses other names, map them in the config (`their name: STRIDE name`) or on the
command line:

```yaml
keypoints:
  nose: snout
  tail_root: tailbase
  tail_end: tailtip
  left_hindpaw: hindpawL1
  right_hindpaw: hindpawR1
```

```bash
stride gait /data/videos --px-per-cm 26.1 -o out --keypoint nose=snout --keypoint tail_root=tailbase \
    --keypoint left_hindpaw=hindpawL1 --keypoint right_hindpaw=hindpawR1
```

- With two landmarks per hind paw, one is picked per side and per video (see `hindL_ind` below).
  With one landmark per hind paw that landmark is used.
- The check runs on the first pose file before any analysis. If a needed keypoint is missing the
  run stops and prints what is missing and which nodes the pose file has.
- The 15-node STRIDE skeleton is `snout, mouth, forepawR2, forepawR1, forepawL1, forepawL2,
  hindpawR2, hindpawR1, hindpawL2, hindpawL1, tailbase, tail1, tail2, tail3, tailtip`. For any
  other skeleton the per-video mean keypoint confidence (used by the confidence filter) is taken
  over the nodes that skeleton has.

## Scale (px/cm)

1. `--px-per-cm VALUE` (or `scale: px_per_cm:` in the config): used as given.
2. Otherwise the ChArUco board in view: the first video of every folder is measured (at least
   `min_clips`), the median is used when at least 3 clips agree within 1 %. Describe a board other
   than the default (14 x 9 squares, 20 mm squares, 15 mm markers, `DICT_5X5_250`) with
   `--board-*` or `scale: charuco:`. `scale: charuco: clips:` names the videos to measure. The
   measurement is also available on its own: `stride calibrate` ([CALIBRATION.md](CALIBRATION.md)).
3. Otherwise the run stops with the reason. One value is used for every video of the run.

`scale/px_per_cm.json` records the value and where it came from.

## Metadata (optional)

A CSV with a `video` column holding the file name (with or without folder and extension) and any
other columns:

```csv
video,animal,sex,group,weight_g
runway A1.mp4,A1,F,control,21.5
```

- An animal id (`mouse_id`, `mouse`, `animal`, `animal_id`, `subject` or `id`) fills `mouse_id`;
  `sex` / `gender` fills `sex`; `genotype` fills `genotype`; `day` fills `day`.
- Every other column (and the id column under its own name) is added at the end of
  `gait_per_stride.csv`, `gait_per_step.csv`, `gait_per_video.csv` and `gait_per_stride_filtered.csv`.
- A file named `Day<N>_<animal>_Trial<T>.mp4` gets `day` and `mouse_id` from its name.
- Without metadata (or without a row for a video) these columns stay empty.

## Output folder

```
SUMMARY.md                       what ran, counts, px/cm and its source, warnings, videos left out and why
gait_run.log                     everything printed
stride_gait_run_manifest.yaml    resolved settings, model, code version, results
gait_state.json                  what is finished (for the next run)
scale/px_per_cm.json             the scale and its source (+ ChArUco files when the board was measured)
pose/<video name>.slp            only for videos whose pose file was made by this run
gait/gait_per_stride.csv         one row per stride
gait/gait_per_stride_filtered.csv   the strides that pass the filters
gait/gait_per_step.csv           one row per hind-paw swing
gait/gait_per_video.csv          one row per video
gait/pose_keypoint_confidence_by_video.csv
gait/videos.csv, run_params.json, *_run_manifest.yaml
```

No trial phase and no maze region is attached to a stride in this mode, and the trial columns of
`gait_per_video.csv` (`choice_LR`, `latency_choice_ms`, `stem_latency_ms`, `junction_explore_ms`,
`reward`, `correct_TF`, `correct_bin`, `probes_L`, `probes_R`) are empty.

## How the tables are computed

Coordinates are divided by px/cm; missing values are filled per coordinate (nearest neighbour,
extrapolated at the ends); a median filter (3 frames) and a moving average (5 frames) follow. The
speed of a keypoint is its frame-to-frame displacement times the frame rate. The body frame of a
frame has the tail base at the origin and is rotated by the angle of the tail base -> snout axis in
the image; the rotation puts the snout on the +x axis when the animal heads along the image
x-axis.

- **Walking**: tail-base speed of at least 5 cm/s.
- **Swing** of a hind paw: consecutive walking frames with paw speed of at least 8 cm/s, at least
  3 frames long, with a peak paw speed of at least 15 cm/s that is not below the peak tail-base
  speed in the same frames.
- **Stride**: from the frame after a left hind-paw swing ends (`frame_start`) to the last frame of
  the next left swing (`frame_end`); kept when a right swing ends inside it.

These values are the defaults of `stride_detection:` in the config.

### `gait_per_stride.csv`

| column | computed as |
|---|---|
| `video_path`, `stem` | the video and its file name without extension |
| `day`, `mouse_id`, `sex`, `genotype` | from the file name / metadata; empty when unknown |
| `stride_id` | index of the stride within the video, from 0 |
| `frame_start`, `frame_end` | `s0`, `s1` as defined above |
| `t_start_ms`, `t_end_ms` | `s0 / fps` and `(s1 - 1) / fps`, in ms |
| `duration_ms` | `(s1 - s0) / fps`, in ms |
| `fps` | frame rate read from the video (`--fps` only when it cannot be read) |
| `px_per_cm` | **cm per pixel** (1 / scale), despite the name |
| `body_length_cm` | mean snout - tail base distance over the whole video |
| `hindL_ind`, `hindR_ind` | node index of the hind-paw landmark used on each side. With two landmarks per side the one with the higher (share of walking frames with a speed) x (median of its 10 highest speeds while walking) is taken, per video and per side |
| `angular_velocity_deg_s_mean` | mean over the stride frames of the rate of change of the body-axis angle, deg/s |
| `stride_speed_cm_s_mean` | mean tail-base speed over the stride frames |
| `limb_duty_factor_mean` | mean of the left and right duty factor; a paw's duty factor is the share of the stride frames in which it is not in a swing |
| `temporal_symmetry` | (left duty - right duty) / (left duty + right duty) |
| `step_length_cm` | body-frame x of the right paw at the right strike inside the stride, minus body-frame x of the left paw at the left strike before the stride (each in the body frame of its own frame). It depends on which landmark was picked on each side |
| `step_width_cm` | distance of the right paw at that right strike from the segment between the left paw at toe-off and at strike of the left swing before the stride, in body-frame coordinates |
| `stride_length_cm` | body-frame x of the left paw at the strike that ends the stride, minus its body-frame x at toe-off of that swing. This is the paw's excursion relative to the body, not the distance covered on the ground, and it is signed |
| `stride_length_euclid_cm` | length of that body-frame displacement (unsigned) |
| `nose_lat_disp_norm`, `tailbase_lat_disp_norm`, `tailtip_lat_disp_norm` | range (max - min) over the stride frames of the keypoint's offset perpendicular to the straight line from the tail base at `s0` to the tail base at `s1 - 1`, divided by `body_length_cm` |
| `nose_lat_phase_pct`, `tailbase_lat_phase_pct`, `tailtip_lat_phase_pct` | where in the stride (0-100 %) that perpendicular offset is largest (cubic spline through the frames, 101 points) |
| `valid_stride_TF`, `has_R_end_inside_TF` | a right swing ends inside the stride (always true for the rows written) |
| `nans_in_stride_TF` | any keypoint coordinate of the stride frames is not finite after the filling step (true for every stride when a keypoint was never tracked or the skeleton has no tail tip) |
| `comment` | empty, or why the stride is not valid |

### `gait_per_stride_filtered.csv`

The same columns, for the strides that pass, in this order:

1. **Confidence**: the video's mean keypoint confidence (mean of the `avg_*` columns of the
   confidence table) is above 0.3.
2. **Edges**: the first and the last stride of every video are removed (a video with one or two
   strides keeps none).
3. **Turning**: `angular_velocity_deg_s_mean` within -20 to +20 deg/s.
4. **Forward direction**: the sign of the median `stride_length_cm` of the remaining strides of the
   run is taken as forward; strides with the other sign are removed. With the body frame as
   defined above, `stride_length_cm` is positive while the body axis is within 45 degrees of the
   image x-axis, so this filter keeps strides along the image x-axis and removes backward steps
   and strides across it.

The thresholds are the defaults of `stride_filtering:` in the config.

### `gait_per_step.csv`

One row per hind-paw swing: `step_id`, `paw_side` (L / R), `frame_a`, `frame_b` (first and last
swing frame), `t_a_ms`, `t_b_ms`, `dur_ms` = `(frame_b - frame_a + 1) / fps`, `peak_speed_cm_s`
(paw), `peak_tailbase_speed_cm_s`, and `within_stride_id` (the stride in which the swing ends, -1
when none).

### `gait_per_video.csv`

One row per video: the id columns, the (empty) trial columns, `hindL_ind`, `hindR_ind`,
`n_strides`, `n_valid_strides`, `frac_valid`, and `<column>_mean`, `<column>_sd`, `<column>_n`
over the video's strides for `stride_speed_cm_s_mean`, `angular_velocity_deg_s_mean`,
`limb_duty_factor_mean`, `temporal_symmetry`, `step_length_cm`, `step_width_cm`,
`stride_length_cm`, the three `*_lat_disp_norm` and the three `*_lat_phase_pct` columns. These
are over all strides of the video, before the filters.

### `pose_keypoint_confidence_by_video.csv`

One row per video: `frames_total`, `frames_kept`, and `avg_<keypoint>` = the mean score of that
keypoint over the frames that have an instance (0 when the keypoint has no score).

## Warnings you may see

Warnings never change a value. They are printed, collected in `SUMMARY.md` and in the manifest.

| warning | meaning |
|---|---|
| body axis within 45 degrees of the image x-axis in less than half of the walking frames | `stride_length_cm`, `step_length_cm` and the forward-direction filter assume the animal walks left-right in the image. Rotate the camera or the videos so that the runway is horizontal. The share is always printed (`[direction]`). |
| fps below 100 | stride detection works in frames; timing and stride length are not comparable with recordings at 100 fps or more |
| scale below 20 px/cm | paw tracking at this resolution is too coarse for the gait measures |
| no stride was found in any video | the clips are too short or the animal does not walk |
| N video(s) were left out | no pose file and no pose model, or the pose file could not be analysed; `SUMMARY.md` lists each with its reason |

## Limits

- One scale for the whole run. Recordings with different scales are run separately.
- One animal per video; the first instance of each frame is used.
- The step and stride measures are relative to the body and assume walking along the image x-axis.
- No trial phase, no maze region, no decision table: those need the maze ROI (`stride tmaze`).
- The genotype statistics are not part of this command.
