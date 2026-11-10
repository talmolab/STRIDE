# Scale calibration — px/cm from a ChArUco board

Gait distances are reported in centimetres, so STRIDE needs to know how many pixels one centimetre is in your
recordings. `stride calibrate` measures it from a ChArUco board lying in view of the camera and writes one value for
the recording set-up.

```bash
stride calibrate /data/my_cohort
# px_per_cm = 26.02   (median of 6 clips, spread 0.18 %)   -> /data/my_cohort/calibration/px_per_cm.txt
```

Put that number in the batch as `px_per_cm_override` (config) — the gait stage then uses it for every clip instead of
deriving a scale per video from the maze outline.

It is a stand-alone tool: it needs only the videos. It works before or after any other stage.

## What you need

- **A ChArUco board, printed and measured.** Default: 14 x 9 squares, 20 mm squares, 15 mm markers, dictionary
  `DICT_5X5_250`. Measure one square on the print with a ruler — the square size is what turns pixels into centimetres.
- **The board flat in the plane the animal walks on, facing the camera,** fully visible and unobstructed.
- **The board in view during the first 10 s of at least three clips** (it can stay in view in every clip). Clips
  spread over the recording days show that the camera did not move.

## Usage

```bash
stride calibrate FOLDER                      # clips are found recursively; the first clip of each sub-folder is measured
stride calibrate clip1.mp4 clip2.mp4 clip3.mp4   # or name the clips yourself
stride calibrate FOLDER --out results/calibration --workers 4
stride calibrate FOLDER --board-squares 7x5 --square-mm 30 --marker-mm 22 --dictionary DICT_4X4_50
```

| option | default | meaning |
|---|---|---|
| `--out` | `FOLDER/calibration` | output folder |
| `--board-squares`, `--square-mm`, `--marker-mm`, `--dictionary` | `14x9`, `20`, `15`, `DICT_5X5_250` | the printed board |
| `--clips` | 6 | for a folder: measure at least this many clips |
| `--max-clips` | none | measure at most this many clips, spread over the folders |
| `--frames-per-clip` | 67 (one every 0.15 s) | frames examined per clip over the first 10 s |
| `--min-readings` | 3 | usable clip readings needed to accept a value |
| `--max-spread` | 1.0 | largest accepted spread of the readings, in % |
| `--rail-crop y0,y1,x0,x1` | off | optional cross-check on a straight corridor of 10 cm (reported, never used) |
| `--workers` | 1 | clips measured in parallel |

With a folder, clips are found in every sub-folder, `.mp4` or `.MP4`. Folders named `waste`, and folders whose name
starts with `.` or `_`, are skipped. One clip per sub-folder (recording day) is measured — the first by name — and, if
there are fewer than six sub-folders, more clips of the same folders are added.

## From Python, or from a config block

```python
from stride.stages.scale_calibration import calibrate_px_per_cm

r = calibrate_px_per_cm("/data/my_cohort", out_dir="/data/my_cohort/calibration")
if r["ok"]:
    print(r["px_per_cm"], r["spread_pct"], r["n_clips_used"])
else:
    print(r["reason"])
```

It does not raise when a calibration cannot be accepted; it returns:

| key | meaning |
|---|---|
| `ok` | `True` when a value was accepted and written |
| `px_per_cm` | the accepted value, else `None` |
| `spread_pct` | (max - min) / median of the usable readings, in % |
| `n_clips_used` | clips that gave a reading (`n_clips` = clips examined) |
| `per_clip` | one dict per clip: `clip`, `px_per_cm`, `usable`, `corners`, `anisotropy`, `reprojection_px`, `frame_time_s`, ... |
| `method` | `"charuco_homography_median"` |
| `reason` | empty when `ok`, else why no value was accepted |
| `out_dir`, `px_per_cm_txt`, `summary_txt`, `readings_csv`, `median`, `board` | where things went, and details |

Every setting is a plain keyword argument, so a YAML block can be passed straight in:

```yaml
calibrate:
  squares_x: 14
  squares_y: 9
  square_mm: 20
  marker_mm: 15
  dictionary: DICT_5X5_250
  max_clips: 12          # optional cap; clips stay spread over the folders
  min_clips: 6           # for a folder: measure at least this many
  frames_per_clip: 67    # frames examined per clip over the first 10 s (default: one every 0.15 s)
  min_readings: 3
  max_spread_pct: 1.0
  workers: 4
```

```python
r = calibrate_px_per_cm(video_dir, out_dir, **config["calibrate"])
```

In a pipeline config the same block sits under a batch (`calibrate: true` uses the defaults); stage 0
(`python -m stride.stages.run_calibration --config config.yaml --output output`) then writes
`output/<batch>/calibration/px_per_cm.txt`. Batches without a `calibrate:` entry are skipped.

`calibrate_scale(...)` is the same call for code that prefers an exception (`CalibrationError`) to an `ok` flag.

## Platforms

Windows, macOS and Linux: paths are handled with `pathlib` (spaces and non-ASCII names are fine), no shell command is
run, no link is created, text files are written as UTF-8, and video suffixes are matched case-insensitively.
It needs `opencv-python >= 4.8` (for the ChArUco detector) and NumPy.

## What it writes

| file | content |
|---|---|
| `px_per_cm.txt` | the accepted value, two decimals (only when accepted) |
| `calibration_summary.txt` | the readings, the median, the spread, and why a value was not written |
| `calibration_readings.csv` | one row per clip: px/cm, usable or not, corners found, anisotropy, reprojection error, frame time |
| `<clip>_checkerboard.png` | the frame used for that clip with the detected corners drawn — look at these |
| `<clip>.json` | the full reading of that clip |
| `calibrate_run_manifest.yaml` | how the value was produced (board, clips, versions) |

## How the value is computed

1. **Per clip.** Frames are read every 0.15 s over the first 10 s. In each, the ChArUco corners are detected (at least
   8 are needed) and a homography from the board's own grid to the image is fitted (RANSAC, 3 px). The scale of the
   frame is the pixel length of one square at the centre of the detected corners, averaged over the board's two axes.
   The frame with the best score — many corners, low reprojection error, low anisotropy — gives the clip's reading:
   px/cm = pixels per square / square size in cm.
2. **For the set-up.** The value is the **median** of the clip readings, rounded to two decimals. It is accepted only
   with at least 3 usable readings whose spread, (max - min) / median, is below 1 %.

Anisotropy is the difference between the two axes of the board; a few percent means the board is tilted against the
image plane.

## When it refuses (exit code 1, the reason is printed)

| message | what to check |
|---|---|
| `only N usable ChArUco reading(s)` | Is the board in view in the first 10 s? Flat and unobstructed? Do `--board-squares`, the sizes and `--dictionary` match the print? The summary lists the clips in which no board was found. |
| `the readings spread by X %` | The board or the camera moved between clips, or one reading is bad. Compare the `_checkerboard.png` images and the table. |
| `no video files found` / `not found` | the path |
| `unknown ArUco dictionary` / `impossible board` | the board options |
| `this OpenCV build has no ChArUco detector` | install `opencv-python >= 4.8` |

Nothing is written to `px_per_cm.txt` in these cases; the summary, the table and the images are still written.

## Limits

- One value per set-up: it assumes the camera, the lens setting and the distance to the floor did not change. A new
  camera position needs a new calibration.
- The board must lie in the plane of the paws. A board held above the floor reads too large a scale.
- It measures scale, not lens distortion — for that, see `stride check-distortion`.
- The optional rail cross-check assumes a straight, roughly horizontal 10-cm corridor inside the crop.
