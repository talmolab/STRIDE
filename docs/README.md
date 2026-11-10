# STRIDE documentation (in this repository)

The full documentation lives at **<https://talmolab.github.io/STRIDE-docs/>**; the project page is
<https://talmolab.github.io/STRIDE-page/>. The pages below ship with the code and describe the
version you have checked out.

| Page | What it covers |
|---|---|
| [tmaze.md](tmaze.md) | **STRIDE T-maze** (`stride tmaze`): what you need, the one command and its config, the steps, metadata, scale, consensus ROI, the output folder, warnings, limits |
| [gait.md](gait.md) | **STRIDE Gait** (`stride gait`): what you need, keypoints and name mapping, scale, the output folder, and how every column of the gait tables is computed |
| [CALIBRATION.md](CALIBRATION.md) | **Scale** (`stride calibrate`): px/cm from a ChArUco board in view, from the command line, from Python or from a config block; when it refuses |
| [lab-workflow.md](lab-workflow.md) | **STRIDE Lab** (`stride-lab`): the authors' own intake workflow (sort, whiteboard check, rename, per-day scale check). Not needed to use STRIDE |
| [example_run_manifest.yaml](example_run_manifest.yaml) | An annotated example of the run manifest every stage writes |

Also in the repository root: [README](../README.md) (overview, install, quick starts, design),
[config.example.yaml](../config.example.yaml) (the configuration for batch runs and the optional
statistics), [CHANGELOG](../CHANGELOG.md), [PLAN](../PLAN.md) (release plan).
