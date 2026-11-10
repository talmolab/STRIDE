# STRIDE

**Stride-level Tracking and Integrated Decision Extraction**

**[Documentation](https://talmolab.github.io/STRIDE-docs/)** · **[Project page](https://talmolab.github.io/STRIDE-page/)**

STRIDE analyses mice performing the T-maze rewarded-alternation task from a single bottom-up video
(camera below a transparent maze floor) using [SLEAP](https://sleap.ai) pose estimation. From that
one video it extracts the trial-level cognitive measures (arm choice, correctness, latencies, snout
probes, junction exploration) and stride-level gait measures (stride and step geometry, timing,
duty factor, lateral body sway and its phase), with no gait-specific hardware. The study it was
built for, and its results, are described in the paper and on the
[project page](https://talmolab.github.io/STRIDE-page/).

## Which STRIDE do I need?

| | For | Goes in | Comes out | Command |
|---|---|---|---|---|
| **STRIDE T-maze** | T-maze recordings, with your own trained pose and maze-ROI models | videos, two model paths | per-trial and per-stride CSV tables, a summary, run logs | `stride tmaze` |
| **STRIDE Gait** | any recording with pose tracking, no maze needed | videos + pose files (or a pose model), and the scale (px/cm) | per-stride, per-step and per-video gait tables | `stride gait` |
| **STRIDE Lab** | the authors' own intake workflow (not needed to use STRIDE) | raw camera clips + recorder metadata | renamed clips, a per-day scale check, then STRIDE T-maze | `stride-lab`, see [docs/lab-workflow.md](docs/lab-workflow.md) |

Pages: [STRIDE T-maze](docs/tmaze.md) · [STRIDE Gait](docs/gait.md) ·
[Scale calibration](docs/CALIBRATION.md) · [docs index](docs/README.md)

## Install

```bash
git clone https://github.com/talmolab/STRIDE.git
cd STRIDE
pip install -e .            # or: uv pip install -e .
pip install -e ".[dev]"     # adds pytest, needed for the mock-data quick starts below
```

- **Python 3.12 or newer.**
- **Inference runtime.** STRIDE runs your SLEAP models through
  [sleap-nn](https://github.com/talmolab/sleap-nn) (PyTorch; a GPU in practice). Install it
  separately, per its own instructions. It is needed only when STRIDE runs the models; the analysis
  of existing pose and ROI files does not use it. `sleap-io` (installed with STRIDE) only reads and
  writes `.slp` files.
- **OpenCV is pinned to `>=4.8,<4.14`.** OpenCV 4.14 and 5.x detect the ChArUco board slightly
  differently and read the same board about 0.1 % lower.
- **Operating systems.** The code is written to run on Windows, macOS and Linux (no shell commands,
  no links, UTF-8 files, workers that start under `spawn`). So far it has been run on Linux only,
  including the inference steps on NVIDIA GPUs. The test suite is set up to run on all three systems
  for every pull request (`.github/workflows/tests.yml`); until those runs are green, treat Windows
  and macOS as untested.

## Quick start

### STRIDE T-maze

```bash
stride tmaze /data/videos \
    --pose-model /models/pose/centroid --pose-model /models/pose/centered_instance \
    --roi-model  /models/roi/centroid  --roi-model  /models/roi/centered_instance \
    --out /data/stride_out
```

or with the settings in a file (anything on the command line overrides the file):

```bash
stride init-config tmaze -o my_tmaze.yaml     # commented template: videos, models, output, steps ...
stride tmaze --config my_tmaze.yaml
```

Try it on mock data (no real videos, models or GPU needed; the two inference steps are skipped
because the mock set comes with its pose and ROI files):

```bash
python tests/mock/make_mock_dataset.py --out tests/mock/_data --only core
stride tmaze tests/mock/_data/core/videos --roi-dir tests/mock/_data/core/roi_yml \
    --metadata tests/mock/_data/core/meta_trials.csv --metadata tests/mock/_data/core/metadata.csv \
    --out mock_output/tmaze --skip-inference
```

The mock maze is drawn at 16 px/cm, so the gait step prints its "below 20 px/cm" warning.

### STRIDE Gait

```bash
# pose files (<video name>.slp) next to the videos or in --pose-dir; scale measured by you
stride gait /data/videos --px-per-cm 26.1 --out /data/gait_out

# no pose files yet: your trained pose model makes them; scale from the ChArUco board in view
stride gait /data/videos \
    --pose-model /models/pose/centroid --pose-model /models/pose/centered_instance \
    --out /data/gait_out

# or from a config file
stride init-config gait -o my_gait.yaml
stride gait --config my_gait.yaml
```

Try it on mock data (a synthetic mouse walking a straight line, with its pose files):

```bash
python tests/mock/make_mock_gait.py --out tests/mock/_data_gait --only runway
stride gait tests/mock/_data_gait/runway/videos --px-per-cm 24 \
    --metadata tests/mock/_data_gait/runway/metadata.csv --out mock_output/gait
```

### Scale only

```bash
stride calibrate /data/videos --out /data/calibration     # px/cm from the ChArUco board in view
```

## How STRIDE is designed

**One command, or one file.** Each pipeline runs from the command line alone, or from one commented
YAML file (`stride init-config tmaze|gait` writes the template). The file holds the video location,
the model paths, the output folder and every option; a flag on the command line overrides the file.
The settings a run actually used (file plus overrides) are written into its manifest.

**Every step can be switched on or off.** `steps:` in the config, or `--skip STEP` / `--enable STEP`.
`--skip-inference` analyses videos that already have pose and ROI files; `--stop-after STEP` ends early.

| Pipeline | Steps, in order | Default |
|---|---|---|
| `stride tmaze` | `distortion_check`, `calibrate`, `pose_inference`, `roi_inference`, `roi_consensus`, `decisions`, `decision_rules`, `gait` | on |
| | `mixed_models`, `classifier` (genotype statistics) | off |
| `stride gait` | `calibrate`, `pose_inference`, `gait_extraction`, `keypoint_confidence`, `stride_filtering` | on |

**Runs resume.** Inference is skipped for a video whose pose or ROI file exists; analysis steps are
skipped when their inputs and settings have not changed. `--force` redoes the analysis.

**Scale (px/cm) has a fixed order of precedence**, and the value used and its source are logged:

1. a value you give (`--px-per-cm`);
2. the ChArUco board in view of the camera ([docs/CALIBRATION.md](docs/CALIBRATION.md));
3. T-maze only: the corridor width of each video's ROI (`--maze-width-cm`, default 10).

`stride gait` has no ROI to fall back on: without 1 or 2 it stops and says so.

**Your files, your names.** Any video file names work (they must be unique within a run). Metadata is
optional: without a reward per trial the choice, timing and probe columns are written and the
correctness columns stay empty. A metadata CSV is keyed by video file name (or by day and mouse); in
`stride gait` its extra columns (animal, sex, group, ...) are carried into the tables.

**What your models must provide.**

- *Pose model* (SLEAP top-down): the 15-node mouse skeleton `snout, mouth, forepawR2, forepawR1,
  forepawL1, forepawL2, hindpawR2, hindpawR1, hindpawL2, hindpawL1, tailbase, tail1, tail2, tail3,
  tailtip`. `stride gait` also accepts other skeletons: map their node names with `keypoints:`; it
  needs the snout, the tail base and at least one landmark per hind paw (the tail tip is optional).
- *ROI model* (SLEAP top-down, T-maze only): the 23-node maze skeleton, 16 corners and 7 centroids of
  the regions `segment1..4`, `junction`, `arm_left`, `arm_right`. The node names are listed in
  [docs/tmaze.md](docs/tmaze.md#what-you-need).

A clip in which a maze corner is hidden gets the consensus ROI of the other clips in its folder, is
marked as such, and is shown in a montage for a visual check.

**Honest exits and warnings.** Anything that stops a run exits non-zero with one line saying why.
Videos that had to be left out (no pose file, no ROI, an error in that clip) are counted and listed
with the reason; `--strict` turns that into an error. Warnings never change a value: a recording
below 100 fps or 20 px/cm, a maze stem that does not run left to right, a scale that could not be
derived for a clip, metadata rows that match no video.

**What every run writes**, next to the tables:

| File | Purpose |
|---|---|
| `SUMMARY.md` | what ran, counts per step, px/cm and where it came from, warnings, videos left out and why |
| `tmaze_run.log` / `gait_run.log` | everything printed during the run |
| `*_run_manifest.yaml` | the resolved settings, the models (fingerprinted by their checkpoint files), the code version and git commit, package versions, inputs and outputs with row counts |

Each stage also writes its own manifest, including when its functions are called from Python;
`stride status --output <dir>` lists them. An annotated example is in
[docs/example_run_manifest.yaml](docs/example_run_manifest.yaml).

**Optional statistics.** The genotype comparison of the paper (linear mixed models per gait metric,
and a WT-vs-Tau classifier) is not part of the automated run. Switch it on in `stride tmaze`
(`steps: {mixed_models: true, classifier: true}` plus `stats_config`), or use the stage-by-stage
runner with [`config.example.yaml`](config.example.yaml), the one configuration file for the
statistics and for batch runs:

```bash
cp config.example.yaml my_config.yaml         # edit the `batches:` section
python run_all.py --config my_config.yaml --output output
python run_all.py --config my_config.yaml --output output --stages 3,4-6,7-9    # selected stages
```

The stage-by-stage runner on the mock data (about a minute; it expects videos named
`Day<N>_<mouse>_Trial<T>` and a reward table, as in the original study):

```bash
python tests/mock/make_mock_dataset.py --out tests/mock/_data --only core
python run_all.py --config tests/mock/_data/core/config.readme.yaml --output mock_output
stride status --output mock_output          # what was run, when, with which code
```

## Outputs

| File | One row per | Described in |
|---|---|---|
| `decisions/decisions.csv` | trial: arm entered, entry frame and time, threshold cleared, correctness | [docs/tmaze.md](docs/tmaze.md#output-folder) |
| `decisions/decisions_clean.csv` | trial, after the two decision rules (use this one for choices) | [docs/tmaze.md](docs/tmaze.md#steps) |
| `decisions/events.csv`, `decisions/metrics.csv` | trial: region entry times, junction exploration, snout probes, latencies | [docs/tmaze.md](docs/tmaze.md#output-folder) |
| `gait/gait_per_stride.csv` | stride: timing, speed, duty factor, step and stride length, sway, phase | [docs/gait.md](docs/gait.md#gait_per_stridecsv) |
| `gait/gait_per_stride_filtered.csv` | stride that passes the confidence, edge, angular-velocity and direction filters | [docs/gait.md](docs/gait.md#gait_per_stride_filteredcsv) |
| `gait/gait_per_step.csv`, `gait/gait_per_video.csv` | hind-paw swing; video | [docs/gait.md](docs/gait.md#gait_per_stepcsv) |
| `gait/pose_keypoint_confidence_by_video.csv` | video: mean keypoint confidence | [docs/gait.md](docs/gait.md#pose_keypoint_confidence_by_videocsv) |
| `scale/`, `roi/`, `pose/` | the scale and its source; the ROI of every video and whether it was detected or borrowed; pose files made by the run | [docs/tmaze.md](docs/tmaze.md#output-folder) |

How each number is computed, column by column and exactly as the code computes it:
[docs/gait.md, "How the tables are computed"](docs/gait.md#how-the-tables-are-computed). One naming
quirk to know: the `px_per_cm` column of the gait tables stores centimetres per pixel.

## What a recording needs

- Camera below a transparent floor, so the paws are visible.
- **At least 100 fps and 20 px/cm** for the gait tables; arm choices work from 30 fps.
- **The maze stem (or the walking direction) running left to right in the image.** The step and stride
  measures assume it; STRIDE warns when a recording does not meet it and does not correct for it.
- A ChArUco board in view if STRIDE is to measure the scale.

The [rig checker](https://talmolab.github.io/STRIDE-page/tools/rig-checker/) on the project page
reads a video in the browser and reports whether a set-up meets these.

## Models and data

STRIDE ships no trained models. Point it at your own SLEAP pose and ROI models (`--pose-model`,
`--roi-model`, or the config).

> Trained model weights of the original study: *to be deposited (DOI to be added).*
>
> The raw T-maze videos and derived tables of the original study: *repository and DOI to be added.*
> An example clip with a matching config will be provided with the release.

## Tests

```bash
pip install -e ".[dev]"
pytest tests                    # synthetic data only; no videos, models or GPU needed
```

## Citation

If you use STRIDE, please cite:

> Li Y, Ling L, Chou R, Ketterer-Sykes EB, Andrews NA, Han C, Liao C, Lee CT, Lee KF\*, Pereira TD\*.
> Dual cognitive and motor phenotyping for early detection of Alzheimer's Disease progression in
> mouse models. In preparation.

```bibtex
@unpublished{li2026dual,
  author = {Li, Yipeng and Ling, L. and Chou, R. and Ketterer-Sykes, E. B. and Andrews, N. A. and Han, C. and Liao, C. and Lee, C. T. and Lee, K. F. and Pereira, Talmo D.},
  title  = {Dual cognitive and motor phenotyping for early detection of Alzheimer's Disease progression in mouse models},
  note   = {In preparation},
  year   = {2026}
}
```

For the software itself see [`CITATION.cff`](CITATION.cff) (a software DOI will be added when the
release is archived). Changes between versions: [`CHANGELOG.md`](CHANGELOG.md).

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Built on [SLEAP](https://sleap.ai) for pose estimation. The gait measures follow
[Sheppard et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110231) and
[Bellardita & Kiehn (2015)](https://doi.org/10.1016/j.cub.2015.04.005).
