# STRIDE

**Stride-level Tracking and Integrated Decision Extraction**

STRIDE is an automated pipeline for dual cognitive and motor phenotyping of mice performing the T-maze rewarded-alternation task, using [SLEAP](https://sleap.ai) pose estimation. From a single **bottom-up video** (camera mounted beneath a transparent maze floor), it extracts **30 behavioral metrics** — 11 T-maze cognitive, 16 stride-level gait, and 3 region-specific gait measures — versus the 4 obtainable by conventional manual scoring, and requires **no specialized gait hardware** (no treadmills, force plates, or pressure-sensitive walkways).

The central finding: **stride-level gait deficits emerge in tau (hMAPT) mice before T-maze choice accuracy diverges from wild type**, suggesting gait as an early, model-specific behavioral marker of tauopathy. (In these cohorts, App^NL-F mice were behaviorally indistinguishable from wild type.)

---

## What STRIDE Does

```
Raw video (.mp4) + SLEAP models
    |
    v
[1] Camera calibration & distortion check
[2] SLEAP top-down inference — centroid + centered-instance (15-keypoint pose + 7-region ROI segmentation)
[3] T-maze behavioral analysis (arm choice, snout probes, junction exploration)
[4] Keypoint preprocessing (interpolation, smoothing, velocity)
[5] Stride detection (swing phases, foot-strikes, stride segmentation)
[6] Stride filtering (confidence, edge removal, angular velocity)
[7] Trial phase tagging (task vs return locomotion)
[8] Derived metrics (step frequency, regional assignment)
[9] Linear mixed model analysis (genotype effects, FDR correction)
[10] WT vs Tau classification (logistic regression, ROC-AUC)
```

## Pipeline Overview

### Cognitive Phenotyping (T-maze)
- Automated arm entry detection with depth-based validation
- Snout probe counting (geometric snout-in-polygon detection)
- Junction exploration timing
- Learning curve analysis with reward reversal detection
- Choice latency and stem traversal metrics

### Motor Phenotyping (Gait)
- 16 stride-level gait metrics (14 enter the mixed-model analysis) following [Sheppard et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110231) and [Bellardita & Kiehn (2015)](https://doi.org/10.1016/j.cub.2015.04.005)
- Spatial: stride length, step length, step width
- Temporal: stride duration, step frequency, limb duty factor, temporal symmetry
- Body dynamics: lateral displacement of snout, tailbase, tailtip (normalized)
- Phase: timing of maximum lateral displacement within stride cycle
- Angular velocity for straight-line walking selection

### Combined Analysis
- Linear mixed models (M1: all effects, M2: speed-corrected direct effects)
- FDR-corrected significance testing across 14 gait metrics
- All three genotypes (WT, App^NL-F, Tau) enter the mixed models; the classifier is a **binary WT-vs-Tau** logistic regression (cognitive + motor features), as App^NL-F mice were behaviorally indistinguishable from WT
- Regional gait analysis (stem, junction, arm)

## Installation

```bash
# Clone
git clone https://github.com/talmolab/STRIDE.git
cd STRIDE

# Install (requires Python >= 3.12)
pip install -e .

# Or with uv
uv pip install -e .

# With the test tools (pytest)
pip install -e ".[dev]"
```

> **Inference runtime (stages 1–2).** SLEAP pose/ROI inference runs on a **separate GPU environment**
> with the [sleap-nn](https://github.com/talmolab/sleap-nn) runtime (PyTorch + a CUDA GPU); install it
> per its own instructions. The `pip install` above provides the STRIDE analysis package (stages 3–10).
> `sleap-io` (a dependency here) handles `.slp` file I/O only — it is not an inference engine.

## STRIDE T-maze (one command)

`stride tmaze` runs the whole T-maze pipeline on a folder of videos: it finds the videos, measures
the scale, runs your pose model and your maze-ROI model, and writes the per-trial and per-stride
tables. No file renaming and no genotype statistics; any file names work. Full description:
[`docs/tmaze.md`](docs/tmaze.md).

```bash
stride tmaze /data/videos \
    --pose-model /models/pose/centroid --pose-model /models/pose/centered_instance \
    --roi-model  /models/roi/centroid  --roi-model  /models/roi/centered_instance \
    --out /data/stride_out
```

Or put the settings in a file; anything given on the command line overrides it:

```bash
stride init-config tmaze -o my_tmaze.yaml     # commented template: videos, models, output, steps ...
stride tmaze --config my_tmaze.yaml
```

Try it on the mock data (no videos, models or GPU needed; the two inference steps are skipped
because the mock set ships its pose and ROI files):

```bash
pip install -e ".[dev]"
python tests/mock/make_mock_dataset.py --out tests/mock/_data --only core
stride tmaze tests/mock/_data/core/videos --roi-dir tests/mock/_data/core/roi_yml \
    --metadata tests/mock/_data/core/meta_trials.csv --metadata tests/mock/_data/core/metadata.csv \
    --out mock_output/tmaze --skip-inference
```

`mock_output/tmaze/` then holds:

| | |
|---|---|
| `SUMMARY.md` | what ran, counts per step, px/cm and where it came from, warnings, videos left out and why |
| `decisions/decisions.csv`, `decisions_clean.csv`, `events.csv`, `metrics.csv` | per trial: arm choice, entry time, latencies, snout probes, junction exploration (`decisions_clean.csv` = after the two protocol rules) |
| `gait/gait_per_stride.csv`, `gait_per_stride_filtered.csv`, `gait_per_step.csv`, `gait_per_video.csv` | per stride / step / video gait tables |
| `scale/`, `roi/`, `pose/` | the scale and its source, the ROI of every video and whether it was detected or borrowed, pose files made by the run |
| `tmaze_run_manifest.yaml`, `tmaze_run.log` | resolved settings, models, code version; everything printed |

Notes:

- **Metadata is optional.** Without it the choice, timing and probe columns are written and the
  correctness columns stay empty. A metadata CSV has a `video` column (file name) or `day` + `mouse`,
  plus `reward` (L/R); optional `sex`, `genotype`.
- **Scale**: a value you give (`--px-per-cm`) wins; otherwise the ChArUco board in view (also available
  on its own: `stride calibrate`, see [`docs/CALIBRATION.md`](docs/CALIBRATION.md)); otherwise the
  corridor width of each video's ROI (`--maze-width-cm`, default 10).
- **Every step can be switched off** (`--skip STEP`, or `steps:` in the config); `--skip-inference`
  analyses videos that already have pose and ROI files. A second run does not repeat finished work.
- **A clip with a hidden maze corner** gets the consensus ROI of the other clips in its folder; check
  the montage under `roi/roi_consensus_qc/`.
- The mock batch is drawn at 16 px/cm, so the gait step prints its "below 20 px/cm" warning.

## Quick Start

### Try it on the mock data (no videos, models or GPU needed)

The repository can generate a small synthetic T-maze batch (blank videos, generated pose and ROI
files, known answers) that runs through every stage in about a minute:

```bash
pip install -e ".[dev]"
python tests/mock/make_mock_dataset.py --out tests/mock/_data --only core
python run_all.py --config tests/mock/_data/core/config.readme.yaml --output mock_output
stride status --output mock_output          # what was run, when, with which code
pytest tests                                # the regression tests use the same generator
```

Stage 1-2 reports that the batch has no model paths and uses the generated pose / ROI files. The mock
batch is drawn at 16 px/cm, so the gait stage prints its "below 20 px/cm" warning — expected here.
`mock_output/mock_core/` then holds the decisions, strides and run manifests; `mock_output/` the
mixed-model table, heatmaps and classifier figures.

### Full pipeline (config-driven)

```bash
# Copy config.example.yaml, edit the batch (videos, metadata, models), then:
python run_all.py --config my_config.yaml --output output
python run_all.py --config my_config.yaml --output output --batch my_batch      # one batch
python run_all.py --config my_config.yaml --output output --stages 3,4-6,7-9    # selected stages
```

Each stage reads what the previous one wrote under `<output>/<batch>/`, so a batch needs only the
inputs you provide (videos, metadata, models). A stage that fails stops the run with a non-zero exit
code and prints the reason. `config.yaml` in this repository is the configuration of the paper's own
cohorts (lab paths); start from [`config.example.yaml`](config.example.yaml).

### Individual stages

```bash
# T-maze behavioral analysis -> decisions/{decisions,events,metrics}.csv
stride analyze-decisions --videos /path/to/videos --yml /path/to/roi_yml --meta meta.csv --output decisions

# The config-driven stages, one at a time (same --config / --output as run_all.py)
python -m stride.stages.run_tmaze           --config my_config.yaml --output output   # stage 3
python -m stride.stages.run_gait_extraction --config my_config.yaml --output output   # stages 4-6
python -m stride.stages.gait_lmm            --config my_config.yaml --output output   # stages 7-9
python -m stride.stages.classification      --config my_config.yaml --output output   # stage 10
```

### Adding a new batch

1. Add a batch entry to your config with video paths, model paths, and metadata
2. Run `python run_all.py --config my_config.yaml --output output --batch your_batch_name`

No code changes required — all batch-specific parameters are config-driven.

### What the pipeline expects from a recording

- Video files named `Day<N>_<mouse>_Trial<T>.mp4` (any letter case; per-day subfolders are fine), with the
  pose file next to each video (`<video name>.slp`, or the batch's `pose_slp_suffix`).
- One ROI file per video: written by stage 2 (`<output>/<batch>/roi_yml/`), in the batch's `yml_dir`, or
  next to the video as `<video name>.rois.yml`.
- `meta_trials_csv` with columns `day,mouse,reward` (day written like `Day1`; reward `L` or `R`) and
  `metadata_csv` with `Mouse,Gender,Genotype`. Videos without a reward row and metadata rows without a
  video are listed as warnings.
- For gait: at least 100 fps and 20 px/cm, and the maze stem running left to right in the image (the
  orientation of the paper's recordings). The gait stage warns when a recording does not meet these; it
  does not correct for them.

## Configuration

All parameters are centralized in one YAML file ([`config.example.yaml`](config.example.yaml) is a
commented template with every key the pipeline reads):

```yaml
batches:
  my_batch:
    name: "My Batch"
    video_dir: "/path/to/videos/"
    yml_dir: "/path/to/roi_yml/"
    meta_trials_csv: "/path/to/meta_trials.csv"
    metadata_csv: "/path/to/genotype_metadata.csv"
    fps: 120
    column_map:
      mouse_id: ["mouse"]
      sex: "Gender"

stride_detection:
  min_walk_velocity: 5.0      # cm/s
  min_paw_velocity: 8.0       # cm/s
  min_peak_paw_velocity: 15.0 # cm/s

stride_filtering:
  confidence_threshold: 0.3
  angular_velocity_min: -20.0 # deg/s
  angular_velocity_max: 20.0  # deg/s

models:            # z-scored metric ~ Genotype [+ Sex] + z-scored covariates, random intercept per mouse
  M1:
    covariates: [body_length_cm]
  M2:
    covariates: [stride_speed_cm_s_mean, body_length_cm]
```

## Project Structure

```
stride/
  config.example.yaml      # Commented configuration template (start here)
  config.yaml              # Configuration of the paper's cohorts (lab paths)
  run_all.py               # Orchestrator (runs all stages)
  tests/                   # pytest suite + tests/mock/ (synthetic T-maze generator and runners)
  src/stride/
    cli.py                 # Click CLI for individual stages
    config.py              # PipelineConfig dataclass
    stages/
      distortion_check.py  # [1] Camera distortion assessment
      pose_inference.py    # [2] SLEAP pose estimation
      roi_inference.py     # [2] ROI region segmentation
      slp_to_yaml.py       # [2] SLP to polygon YAML conversion
      decision_analysis.py # [3] T-maze arm entry decisions
      gait_extraction.py   # [4-5] Keypoint preprocessing + stride detection
      gait_filtering.py    # [6] Stride filtering
      gait_lmm.py          # [7-9] Trial tagging + LMM analysis
      classification.py    # [10] WT vs Tau logistic regression
      run_inference.py     # [1-2] Config-driven inference runner (distortion + pose + ROI + yaml)
      run_tmaze.py         # Config-driven T-maze runner
      run_gait_extraction.py # Config-driven gait extraction runner
      batch_paths.py       # Where each stage finds its inputs under <output>/<batch>/
      safety_checks.py     # Gate checks (calibration / ROI coverage / fps / body-length)
    run_manifest.py        # Provenance sidecar written by every stage
    utils/
      checkpoint.py        # Legacy checkpoint-file reader
      parallel.py          # Parallel processing + video discovery (recursive, any letter case)
      pose_io.py           # Pose-file loading (handles .slp files made on another machine)
      roi.py               # ROI-file helpers
      video.py             # Video I/O utilities
```

## Key Outputs

| Stage | Output | Description |
|-------|--------|-------------|
| 3 | `decisions.csv` | Per-trial arm choice, correctness, entry timing |
| 3 | `events.csv` | Detailed timing: segment entries, probes, junction exploration |
| 3 | `learning_cycle_1.csv` | Trials before reward reversal |
| 5 | `gait_per_stride.csv` | Per-stride metrics (14 gait variables) |
| 6 | `gait_per_stride_filtered.csv` | Quality-filtered strides |
| 9 | `lmm_results_all.csv` | M1/M2 LMM coefficients, p-values, FDR (`<genotype>_fdr`: Benjamini-Hochberg within batch × model) |
| 9 | `lmm_fit_log.csv` | Every metric × model: fitted or not, and why not |
| 9 | `heatmap_*.png` | Dual-panel heatmaps (-log10(p) + effect sizes) |
| 10 | `regression_roc_curves.png` | ROC curves for T-maze, Gait, Combined |
| all | `<stage>_run_manifest.yaml` | **Provenance sidecar** — see below |

### Run manifest (provenance)

Every pipeline stage writes a `<stage>_run_manifest.yaml` next to its outputs recording **how** they
were produced, so a run stays reproducible and auditable long after the fact (`inference`, `decisions`,
`tmaze`, `gait`, `gait_filtering`, `lmm`, `classification`; the same files are written when the stage
functions are called from Python). `stride status --output <dir>` lists the manifests of a run.

- `timestamp_utc`, `stride_version`, and the exact **git** commit/branch (plus a `dirty` flag)
- the **models** used (paths + size/mtime fingerprint of the checkpoint and config files inside each model folder)
- the **parameters** and the **inputs**/**outputs** (with CSV row counts)
- the **environment** — Python + key package versions (`sleap-nn`, `sleap-io`, `numpy`, …), host, command

```yaml
stage: decisions
timestamp_utc: '2026-08-07T17:06:46+00:00'
stride_version: 0.1.0
git: {commit_short: 46074bc85314, branch: main, dirty: false}
models: [{path: .../finetune_final_v5/centroid/best.ckpt, size_bytes: 1507696075}]
environment: {python: 3.13.12, packages: {sleap-nn: 0.2.0, ...}}
results: {ok: 1330, total: 1476}
```

A complete, annotated example (every field, one per line) is checked in at
[`docs/example_run_manifest.yaml`](docs/example_run_manifest.yaml) — real manifests are
generated at runtime next to each stage's outputs and are not committed.

Programmatic (non-CLI) callers can emit one too:

```python
from stride.run_manifest import write_run_manifest
write_run_manifest(output_dir, stage="gait", models=[...], params={...}, outputs=[...])
```

## Models

STRIDE requires two trained SLEAP models that are **not bundled in this repository**:
- **Pose** — 15-keypoint top-down model (centroid + centered-instance)
- **ROI** — 7-region maze-segmentation model

Point `pose_model_paths` and `roi_model_paths` in your config at your local copies (or pass `--model` to
`stride pose-inference` / `stride roi-inference`, `--pose-model` / `--roi-model` to `stride run`). There
are no built-in default model paths.

<!-- TODO(before submission): deposit the trained pose + ROI models (with their sleap-nn training
     version) to a citable archive (Zenodo/OSF/HuggingFace) and add the DOI + download steps here. -->
> Trained model weights: *to be deposited (DOI to be added).*

## Data Availability

<!-- TODO(before submission): state where the T-maze videos + derived CSVs are deposited (repository +
     DOI or access procedure), and add a small example clip + expected-output CSV + example config for
     an end-to-end smoke test. -->
> The raw T-maze videos and derived metric tables supporting this work are available at *(repository /
> DOI to be added)*. An example clip and matching config for a smoke test will be provided with the release.

## Citation

If you use STRIDE in your research, please cite:

> Li, Y. et al. (2026). *Dual cognitive and motor phenotyping for early detection of AD progression in APP and Tau models.* (in preparation)

<!-- TODO(before submission): cut a tagged release, archive it to Zenodo for a software DOI, bump the
     pyproject/CITATION.cff version to match, and add a "Software: DOI:xxxxx (vX.Y.Z)" line here. -->
For the software itself, see [`CITATION.cff`](CITATION.cff) *(software DOI to be added on release)*.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Built on [SLEAP](https://sleap.ai) for pose estimation. Gait analysis methodology adapted from [Sheppard et al. (2022)](https://doi.org/10.1016/j.celrep.2022.110231) and [Bellardita & Kiehn (2015)](https://doi.org/10.1016/j.cub.2015.04.005).
