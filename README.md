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
git clone https://github.com/LeoMeow123/STRIDE.git
cd STRIDE

# Install (requires Python >= 3.12)
pip install -e .

# Or with uv
uv pip install -e .
```

> **Inference runtime (stages 1–2).** SLEAP pose/ROI inference runs on a **separate GPU environment**
> with the [sleap-nn](https://github.com/talmolab/sleap-nn) runtime (PyTorch + a CUDA GPU); install it
> per its own instructions. The `pip install` above provides the STRIDE analysis package (stages 3–10).
> `sleap-io` (a dependency here) handles `.slp` file I/O only — it is not an inference engine.

## Quick Start

### Full pipeline (config-driven)

```bash
# Edit config.yaml with your batch paths, then:
python run_all.py --config config.yaml
```

### Individual stages

```bash
# T-maze behavioral analysis
stride analyze-decisions --videos /path/to/videos --yml /path/to/roi_yml --meta meta.csv

# Gait extraction (stride detection)
python -m stride.stages.run_gait_extraction --config config.yaml

# LMM statistical analysis
python -m stride.stages.gait_lmm --config config.yaml

# Classification
python -m stride.stages.classification --config config.yaml
```

### Adding a new batch

1. Add a batch entry to `config.yaml` with video paths, model paths, and metadata
2. Run `python run_all.py --batch your_batch_name`

No code changes required — all batch-specific parameters are config-driven.

## Configuration

All parameters are centralized in `config.yaml`:

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

models:
  M1:
    formula: "{outcome} ~ C(Genotype, Treatment('WT')) + scale(body_length_cm)"
  M2:
    formula: "{outcome} ~ C(Genotype, Treatment('WT')) + scale(stride_speed_cm_s_mean) + scale(body_length_cm)"
```

## Project Structure

```
stride/
  config.yaml              # Pipeline configuration
  run_all.py               # Orchestrator (runs all stages)
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
      safety_checks.py     # Gate checks (calibration / ROI coverage / fps / body-length)
    utils/
      checkpoint.py        # Resume support for batch processing
      parallel.py          # Multi-GPU parallel processing
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
| 9 | `lmm_results_all.csv` | M1/M2 LMM coefficients, p-values, FDR |
| 9 | `heatmap_*.png` | Dual-panel heatmaps (-log10(p) + effect sizes) |
| 10 | `regression_roc_curves.png` | ROC curves for T-maze, Gait, Combined |

## Models

STRIDE requires two trained SLEAP models that are **not bundled in this repository**:
- **Pose** — 15-keypoint top-down model (centroid + centered-instance)
- **ROI** — 7-region maze-segmentation model

Point `pose_model_paths` and `roi_model_paths` in `config.yaml` at your local copies.

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
