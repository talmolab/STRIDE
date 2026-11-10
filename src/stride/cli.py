"""
Command-line interface for STRIDE.

Usage:
    stride run --input /path/to/videos --output /path/to/results --meta meta.csv \
               --pose-model <centroid> --pose-model <centered_instance> \
               --roi-model <centroid> --roi-model <centered_instance>
    stride check-distortion --input /path/to/videos
    stride pose-inference --input /path/to/videos --model <centroid> --model <centered_instance>
    stride roi-inference --input /path/to/videos --output /path/to/roi_slp --model ... --model ...
    stride convert-roi --input /path/to/roi_slp --output /path/to/roi_yml
    stride analyze-decisions --videos /path/to/videos --yml /path/to/roi_yml --meta /path/to/meta.csv
    stride analyze-gait --input gait_per_stride.csv
    stride status --output /path/to/results

Videos are searched recursively (per-day subfolders) and case-insensitively (.mp4 / .MP4).
A command that fails exits with a non-zero code and prints the reason.
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from stride import __version__
from stride.config import PipelineConfig
from stride.run_manifest import write_run_manifest

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="stride")
@click.pass_context
def main(ctx):
    """T-maze Pipeline - Automated video analysis for behavioral experiments."""
    ctx.ensure_object(dict)


def _fail(message: str, code: int = 1):
    """Print the reason and exit non-zero."""
    console.print(f"[bold red]ERROR:[/] {message}", soft_wrap=True)
    raise SystemExit(code)


@main.command()
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing input videos (searched recursively)")
@click.option("--output", "-o", "output_dir", type=click.Path(), default="./tmaze_output",
              help="Output directory for results")
@click.option("--meta", "-m", "meta_csv", type=click.Path(exists=True),
              help="Metadata CSV with day,mouse,reward columns")
@click.option("--pose-model", "pose_models", type=click.Path(exists=True), multiple=True,
              help="Pose model path; give twice (centroid, centered-instance). Needed unless --skip-pose")
@click.option("--roi-model", "roi_models", type=click.Path(exists=True), multiple=True,
              help="ROI model path; give twice (centroid, centered-instance). Needed unless --skip-roi")
@click.option("--yml", "-y", "yml_dir", type=click.Path(exists=True), default=None,
              help="Existing ROI .yml folder to use with --skip-roi (default: <output>/roi_yml, "
                   "then ROI files next to the videos)")
@click.option("--workers", "-w", "n_workers", type=int, default=4,
              help="Number of parallel workers")
@click.option("--skip-distortion", is_flag=True, help="Skip distortion check")
@click.option("--skip-pose", is_flag=True, help="Skip pose inference (use existing .slp)")
@click.option("--skip-roi", is_flag=True, help="Skip ROI inference (use existing ROI files)")
@click.option("--batch-size", type=int, default=16, help="Batch size for inference")
@click.option("--device", type=str, default="cuda", help="Device for inference (cuda, cuda:0, cpu)")
@click.pass_context
def run(ctx, input_dir, output_dir, meta_csv, pose_models, roi_models, yml_dir, n_workers,
        skip_distortion, skip_pose, skip_roi, batch_size, device):
    """Run the pipeline on one folder: distortion check, pose + ROI inference, decisions.

    Inference needs the trained SLEAP models (--pose-model / --roi-model, each given twice) and
    the sleap-nn runtime. With --skip-pose --skip-roi it analyses existing pose and ROI files.
    For the gait stages and the statistics use `python run_all.py --config <config>`.
    """
    from stride.stages.distortion_check import check_distortion_batch
    from stride.stages.pose_inference import run_pose_inference_batch
    from stride.stages.roi_inference import run_roi_inference_batch
    from stride.stages.slp_to_yaml import convert_batch
    from stride.stages.decision_analysis import run_decision_analysis
    from stride.utils.parallel import find_videos

    config = PipelineConfig(
        video_dir=Path(input_dir),
        output_dir=Path(output_dir),
        meta_csv=Path(meta_csv) if meta_csv else None,
        n_workers=n_workers,
    )

    # Say what is missing before anything runs
    if not skip_pose and not pose_models:
        _fail("pose inference needs the trained pose model: pass --pose-model <centroid> "
              "--pose-model <centered_instance>, or --skip-pose to use existing <video name>.slp files", 2)
    if not skip_roi and not roi_models:
        _fail("ROI inference needs the trained ROI model: pass --roi-model <centroid> "
              "--roi-model <centered_instance>, or --skip-roi to use existing ROI files", 2)

    config.ensure_output_dirs()

    console.print(f"[bold green]STRIDE v{__version__}[/]")
    console.print(f"Input: {config.video_dir}")
    console.print(f"Output: {config.output_dir}")
    console.print(f"Workers: {config.n_workers}")
    console.print()
    problems: list[str] = []

    # Stage 1: Scan videos
    console.print("[bold cyan]Stage 1: Scanning videos...[/]")
    videos = find_videos(config.video_dir, extensions=(".mp4",))
    console.print(f"  Found {len(videos)} videos")

    if len(videos) == 0:
        _fail(f"no .mp4 videos found under {config.video_dir} (searched recursively, any letter case)")

    # Stage 2: Distortion check (optional)
    if not skip_distortion:
        console.print("\n[bold cyan]Stage 2: Checking distortion...[/]")
        distortion_results = check_distortion_batch(
            config.video_dir,
            num_frames=10,
            output_file=config.output_dir / "distortion_check.csv",
        )
        needs_undistort = sum(1 for r in distortion_results if r.get("needs_undistortion"))
        no_board = sum(1 for r in distortion_results if r.get("needs_undistortion") is None)
        console.print(f"  {needs_undistort}/{len(distortion_results)} videos may need undistortion"
                      + (f" ({no_board} without a ChArUco board in view: not assessed)" if no_board else ""))
    else:
        console.print("\n[bold yellow]Stage 2: Skipped distortion check[/]")

    # Stage 3: Pose inference
    if not skip_pose:
        console.print("\n[bold cyan]Stage 3: Running pose inference...[/]")
        try:
            pose_results = run_pose_inference_batch(
                video_dir=config.video_dir,
                output_dir=None,  # Co-locate with videos
                model_paths=list(pose_models),
                overwrite=False,
                batch_size=batch_size,
                device=device,
            )
        except (ValueError, RuntimeError) as exc:
            _fail(str(exc))
        console.print(f"  Done: {pose_results['done']}, Skipped: {pose_results['skipped']}, Failed: {pose_results['failed']}")
        if pose_results["failed"]:
            problems.append(f"pose inference failed for {pose_results['failed']} video(s)")
    else:
        console.print("\n[bold yellow]Stage 3: Skipped pose inference[/]")

    # Stage 4: ROI inference
    roi_yml_dir = config.output_dir / "roi_yml"
    if not skip_roi:
        console.print("\n[bold cyan]Stage 4: Running ROI inference...[/]")
        try:
            roi_results = run_roi_inference_batch(
                video_dir=config.video_dir,
                output_dir=config.output_dir / "roi_slp",
                model_paths=list(roi_models),
                overwrite=False,
                device=device,
            )
        except (ValueError, RuntimeError) as exc:
            _fail(str(exc))
        console.print(f"  Passed: {roi_results['passed']}, Skipped: {roi_results['skipped']}, Failed: {roi_results['failed']}")
        if roi_results["failed"]:
            problems.append(f"ROI inference failed for {roi_results['failed']} video(s)")

        # Stage 5: SLP to YAML conversion
        console.print("\n[bold cyan]Stage 5: Converting ROI SLP to YAML...[/]")
        convert_results = convert_batch(
            slp_dir=config.output_dir / "roi_slp",
            yml_dir=roi_yml_dir,
            overwrite=False,
        )
        console.print(f"  Converted: {convert_results['ok']}, Skipped: {convert_results['skip']}, Failed: {convert_results['failed']}")
        if convert_results["failed"] or convert_results.get("missing_keypoints"):
            problems.append(f"ROI conversion failed for "
                            f"{convert_results['failed'] + convert_results.get('missing_keypoints', 0)} file(s)")
    else:
        console.print("\n[bold yellow]Stage 4-5: Skipped ROI inference and conversion[/]")
        if yml_dir:
            roi_yml_dir = Path(yml_dir)

    # Stage 6: Decision analysis
    decisions_dir = config.output_dir / "decisions"
    if config.meta_csv and config.meta_csv.exists():
        console.print("\n[bold cyan]Stage 6: Analyzing T-maze decisions...[/]")
        has_yml = roi_yml_dir.exists() and any(roi_yml_dir.glob("*.yml"))
        decision_results = run_decision_analysis(
            video_dir=config.video_dir,
            yml_dir=roi_yml_dir if has_yml else None,   # None: ROI files next to the videos
            meta_csv=config.meta_csv,
            output_dir=decisions_dir,
            n_workers=config.n_workers,
        )
        console.print(f"  Analyzed: {decision_results['ok']}/{decision_results['total']} trials")
        if decision_results["ok"] == 0:
            problems.append("no video could be analysed for decisions (see the summary above)")
    else:
        console.print("\n[bold yellow]Stage 6: Skipped decision analysis (no metadata CSV; pass --meta)[/]")

    # Provenance sidecar: record how this run was produced (code, models, params, env).
    manifest_path = write_run_manifest(
        config.output_dir, "pipeline",
        models=[*pose_models, *roi_models] or None,
        params={"n_workers": n_workers, "batch_size": batch_size, "device": device,
                "skip_distortion": skip_distortion, "skip_pose": skip_pose, "skip_roi": skip_roi},
        inputs={"video_dir": config.video_dir, "meta_csv": config.meta_csv, "n_videos": len(videos)},
        outputs=[decisions_dir / "decisions.csv",
                 config.output_dir / "distortion_check.csv"],
        results={"problems": problems or None},
    )

    if manifest_path:
        console.print(f"Run manifest: {manifest_path}")
    if problems:
        console.print("\n[bold red]Pipeline finished with problems:[/]")
        for pr in problems:
            console.print(f"  - {pr}")
        raise SystemExit(1)
    console.print("\n[bold green]Pipeline complete![/]")
    console.print(f"Results saved to: {config.output_dir}")


@main.command("pose-inference")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos (searched recursively, .mp4 / .MP4)")
@click.option("--output", "-o", "output_dir", type=click.Path(), default=None,
              help="Output directory for .slp files (default: same as video)")
@click.option("--model", "-m", "model_paths", type=click.Path(exists=True), multiple=True,
              help="Model path; give twice (centroid and centered_instance). Required")
@click.option("--batch-size", "-b", type=int, default=16,
              help="Batch size for inference")
@click.option("--device", "-d", type=str, default="cuda",
              help="Device for inference (cuda, cuda:0, cuda:1, cpu)")
@click.option("--gpu", "-g", type=int, default=None,
              help="GPU device ID (shorthand for --device cuda:N)")
@click.option("--worker-id", type=int, default=0,
              help="Worker ID for multi-GPU parallel processing (0-indexed)")
@click.option("--num-workers", type=int, default=1,
              help="Total number of parallel workers (for multi-GPU)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.pass_context
def pose_inference(ctx, input_dir, output_dir, model_paths, batch_size, device, gpu, worker_id, num_workers, overwrite):
    """Run pose estimation inference on videos (mouse body keypoints).

    For multi-GPU parallelism, run multiple instances with different worker IDs:

        # Terminal 1 - GPU 0
        stride pose-inference -i /videos -m <centroid> -m <centered_instance> --gpu 0 --worker-id 0 --num-workers 2

        # Terminal 2 - GPU 1
        stride pose-inference -i /videos -m <centroid> -m <centered_instance> --gpu 1 --worker-id 1 --num-workers 2
    """
    from stride.stages.pose_inference import run_pose_inference_batch

    # Handle --gpu shorthand
    if gpu is not None:
        device = f"cuda:{gpu}"

    console.print(f"[bold]Running pose inference[/]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir or '(co-located with videos)'}")
    console.print(f"Batch size: {batch_size}")
    console.print(f"Device: {device}")
    if num_workers > 1:
        console.print(f"Worker: {worker_id + 1}/{num_workers}")

    model_paths = list(model_paths) if model_paths else None
    try:
        results = run_pose_inference_batch(
            video_dir=Path(input_dir),
            output_dir=Path(output_dir) if output_dir else None,
            model_paths=model_paths,
            overwrite=overwrite,
            batch_size=batch_size,
            device=device,
            worker_id=worker_id,
            num_workers=num_workers,
        )
    except (ValueError, RuntimeError) as exc:
        _fail(str(exc))

    console.print(f"\n[green]Done: {results['done']}/{results['total']} videos[/]")
    console.print(f"Skipped: {results['skipped']}")
    if results['failed'] > 0:
        console.print(f"[red]Failed: {results['failed']} videos[/]")
        for d in [d for d in results["details"] if d.get("status") not in ("ok", "skip")][:5]:
            console.print(f"  {d.get('video')}: {d.get('error') or d.get('reason')}", soft_wrap=True)

    # Provenance sidecar (worker 0 only, to avoid multi-GPU write races).
    if worker_id == 0:
        mpath = write_run_manifest(
            Path(output_dir) if output_dir else Path(input_dir), "pose_inference",
            models=model_paths,
            params={"batch_size": batch_size, "device": device, "num_workers": num_workers, "overwrite": overwrite},
            inputs={"video_dir": input_dir}, results=results,
        )
        if mpath:
            console.print(f"Run manifest: {mpath}")

    if results["total_all"] == 0:
        _fail(f"no .mp4 videos found under {input_dir} (searched recursively, any letter case)")
    if results["failed"] > 0:
        raise SystemExit(1)


@main.command("check-distortion")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos (searched recursively) or single video file")
@click.option("--output", "-o", "output_file", type=click.Path(), default=None,
              help="Output file for results (CSV/JSON)")
@click.option("--num-frames", "-n", type=int, default=10,
              help="Number of frames to sample per video")
@click.option("--squares-x", type=int, default=14, help="Charuco board squares in X")
@click.option("--squares-y", type=int, default=9, help="Charuco board squares in Y")
@click.pass_context
def check_distortion(ctx, input_dir, output_file, num_frames, squares_x, squares_y):
    """Check if videos need undistortion using charuco board analysis."""
    from stride.stages.distortion_check import check_distortion_batch

    input_path = Path(input_dir)
    console.print(f"[bold]Checking distortion for: {input_path}[/]")

    results = check_distortion_batch(
        input_path,
        num_frames=num_frames,
        squares_x=squares_x,
        squares_y=squares_y,
        output_file=Path(output_file) if output_file else None,
    )

    if not results:
        _fail(f"no .mp4 / .avi videos found under {input_path} (searched recursively, any letter case)")

    # Display summary
    table = Table(title="Distortion Check Results")
    table.add_column("Video", style="cyan")
    table.add_column("Line Dev", justify="right")
    table.add_column("Spacing CV", justify="right")
    table.add_column("Reproj Err", justify="right")
    table.add_column("Status", justify="center")

    for r in results:
        status = "[red]UNDISTORT[/]" if r.get("needs_undistortion") else "[green]OK[/]"
        if r.get("needs_undistortion") is None:
            status = "[yellow]NO BOARD[/]"
        if r.get("error"):
            status = "[red]ERROR[/]"
        table.add_row(
            r.get("video", "?"),
            f"{r.get('line_straightness', 0):.2f}",
            f"{r.get('spacing_uniformity', 0):.4f}",
            f"{r.get('reprojection_error', 0):.2f}",
            status,
        )

    console.print(table)
    errors = [r for r in results if r.get("error")]
    for r in errors[:5]:
        console.print(f"[red]{r.get('video')}: {r.get('error')}[/]", soft_wrap=True)
    n_no_board = sum(1 for r in results if r.get("needs_undistortion") is None and not r.get("error"))
    if n_no_board:
        console.print(f"{n_no_board}/{len(results)} video(s) show no ChArUco board in the sampled frames: "
                      f"distortion could not be assessed for them.")
    if errors:
        raise SystemExit(1)


@main.command("roi-inference")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos (searched recursively, .mp4 / .MP4)")
@click.option("--output", "-o", "output_dir", type=click.Path(), required=True,
              help="Output directory for ROI .slp files")
@click.option("--model", "-m", "model_paths", type=click.Path(exists=True), multiple=True,
              help="Model path; give twice (centroid and centered_instance). Required")
@click.option("--device", "-d", type=str, default=None,
              help="Device for inference (cuda, cuda:0, cuda:1, cpu). Default: chosen by sleap-nn")
@click.option("--gpu", "-g", type=int, default=None,
              help="GPU device ID (shorthand for --device cuda:N)")
@click.option("--worker-id", type=int, default=0,
              help="Worker ID for multi-GPU parallel processing (0-indexed)")
@click.option("--num-workers", type=int, default=1,
              help="Total number of parallel workers (for multi-GPU)")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.pass_context
def roi_inference(ctx, input_dir, output_dir, model_paths, device, gpu, worker_id, num_workers, overwrite):
    """Run ROI keypoint inference on videos.

    For multi-GPU parallelism, run multiple instances with different worker IDs:

        # Terminal 1 - GPU 0
        stride roi-inference -i /videos -o /output -m <centroid> -m <centered_instance> --gpu 0 --worker-id 0 --num-workers 2

        # Terminal 2 - GPU 1
        stride roi-inference -i /videos -o /output -m <centroid> -m <centered_instance> --gpu 1 --worker-id 1 --num-workers 2
    """
    from stride.stages.roi_inference import run_roi_inference_batch

    # Handle --gpu shorthand
    if gpu is not None:
        device = f"cuda:{gpu}"

    console.print(f"[bold]Running ROI inference[/]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print(f"Device: {device or 'auto (chosen by sleap-nn)'}")
    if num_workers > 1:
        console.print(f"Worker: {worker_id + 1}/{num_workers}")

    model_paths = list(model_paths) if model_paths else None
    try:
        results = run_roi_inference_batch(
            video_dir=Path(input_dir),
            output_dir=Path(output_dir),
            model_paths=model_paths,
            overwrite=overwrite,
            worker_id=worker_id,
            num_workers=num_workers,
            device=device,
        )
    except (ValueError, RuntimeError) as exc:
        _fail(str(exc))

    console.print(f"[green]Completed: {results['passed']}/{results['total']} videos[/]")
    if results['failed'] > 0:
        console.print(f"[red]Failed: {results['failed']} videos[/]")
        for d in [d for d in results["results"] if d.get("status") not in ("ok", "skip")][:5]:
            console.print(f"  {d.get('video')}: {d.get('error') or d.get('reason')}", soft_wrap=True)

    # Provenance sidecar (worker 0 only, to avoid multi-GPU write races).
    if worker_id == 0:
        mpath = write_run_manifest(
            Path(output_dir), "roi_inference",
            models=model_paths,
            params={"device": device or "auto (chosen by sleap-nn)", "num_workers": num_workers,
                    "overwrite": overwrite},
            inputs={"video_dir": input_dir}, results=results,
        )
        if mpath:
            console.print(f"Run manifest: {mpath}")

    if results["total_all"] == 0:
        _fail(f"no .mp4 videos found under {input_dir} (searched recursively, any letter case)")
    if results["failed"] > 0:
        raise SystemExit(1)


@main.command("convert-roi")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing ROI .slp files")
@click.option("--output", "-o", "output_dir", type=click.Path(), required=True,
              help="Output directory for .yml files")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs")
@click.pass_context
def convert_roi(ctx, input_dir, output_dir, overwrite):
    """Convert ROI .slp files to polygon YAML format."""
    from stride.stages.slp_to_yaml import convert_batch

    console.print(f"[bold]Converting SLP to YAML[/]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")

    results = convert_batch(
        slp_dir=Path(input_dir),
        yml_dir=Path(output_dir),
        overwrite=overwrite,
    )

    console.print(f"[green]Converted: {results['ok']}/{results['total']}[/]")
    if results['missing_keypoints'] > 0:
        console.print(f"[yellow]Missing keypoints: {results['missing_keypoints']}[/]")
    if results['failed'] > 0:
        console.print(f"[red]Failed: {results['failed']}[/]")

    mpath = write_run_manifest(
        Path(output_dir), "roi_to_yaml",
        inputs={"slp_dir": input_dir}, results=results,
    )
    if mpath:
        console.print(f"Run manifest: {mpath}")

    if results["total"] == 0:
        _fail(f"no ROI prediction files (*.preds.v2.best1.slp / *.best1.slp) found in {input_dir}")
    if results["failed"] > 0 or results["missing_keypoints"] > 0:
        console.print(f"[red]{results['failed'] + results['missing_keypoints']} of {results['total']} ROI "
                      f"file(s) were not converted (see above); those videos have no ROI yml.[/]")
        raise SystemExit(1)


@main.command("analyze-decisions")
@click.option("--videos", "-v", "video_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos with pose files next to them (searched recursively)")
@click.option("--yml", "-y", "yml_dir", type=click.Path(exists=True), default=None,
              help="Directory containing ROI .yml files (omit when the ROI files sit next to the videos)")
@click.option("--meta", "-m", "meta_csv", type=click.Path(exists=True), required=True,
              help="Metadata CSV with day,mouse,reward columns")
@click.option("--output", "-o", "output_dir", type=click.Path(), default="decisions",
              help="Output directory for decisions.csv, events.csv, metrics.csv (default: ./decisions)")
@click.option("--pose-suffix", "pose_slp_suffix", type=str, default=".slp",
              help="Suffix of the pose file next to each video (default: .slp)")
@click.option("--workers", "-w", "n_workers", type=int, default=4,
              help="Number of parallel workers")
@click.pass_context
def analyze_decisions(ctx, video_dir, yml_dir, meta_csv, output_dir, pose_slp_suffix, n_workers):
    """Analyze T-maze arm entry decisions.

    Writes decisions.csv, events.csv and metrics.csv (plus decisions_run_manifest.yaml) into the
    output directory.
    """
    from stride.stages.decision_analysis import run_decision_analysis

    out = Path(output_dir)
    if out.suffix.lower() == ".csv":      # `-o something/decisions.csv`: use its folder
        console.print(f"[yellow]--output is a directory; using {out.parent} (files: decisions.csv, "
                      f"events.csv, metrics.csv)[/]")
        out = out.parent

    console.print(f"[bold]Analyzing T-maze decisions[/]")
    console.print(f"Videos: {video_dir}")
    console.print(f"ROI YMLs: {yml_dir or '(next to the videos)'}")
    console.print(f"Metadata: {meta_csv}")

    results = run_decision_analysis(
        video_dir=Path(video_dir),
        yml_dir=Path(yml_dir) if yml_dir else None,
        meta_csv=Path(meta_csv),
        output_dir=out,
        n_workers=n_workers,
        pose_slp_suffix=pose_slp_suffix,
    )

    console.print(f"[green]Analyzed: {results['ok']}/{results['total']} trials[/]")
    if results["ok"] == 0:
        why = "; ".join(f"{n} x {r}" for r, n in (results.get("skip_reasons") or {}).items()) or "no videos found"
        _fail(f"no video could be analysed ({why})")
    console.print(f"Output saved to: {out}/ (decisions.csv, events.csv, metrics.csv)")
    console.print(f"Run manifest: {out / 'decisions_run_manifest.yaml'}")


@main.command("analyze-gait")
@click.option("--input", "-i", "input_csv", type=click.Path(exists=True), required=True,
              help="Input gait per-stride CSV")
@click.option("--confidence", "-c", "conf_csv", type=click.Path(exists=True),
              help="Confidence CSV (optional)")
@click.option("--output", "-o", "output_csv", type=click.Path(), default="gait_filtered.csv",
              help="Output filtered CSV")
@click.option("--threshold", "-t", type=float, default=0.3,
              help="Confidence threshold (default: 0.3)")
@click.pass_context
def analyze_gait(ctx, input_csv, conf_csv, output_csv, threshold):
    """Filter gait strides by keypoint confidence and remove edge strides.

    This is the quick two-step filter. The pipeline's stage 6
    (`python -m stride.stages.run_gait_extraction`) additionally applies the angular-velocity
    and forward-direction filters from the config and therefore keeps fewer strides.
    """
    from stride.stages.gait_filtering import filter_strides

    console.print(f"[bold]Filtering gait strides (confidence + edge strides only)[/]")
    console.print(f"Input: {input_csv}")
    console.print(f"Threshold: {threshold}")

    results = filter_strides(
        stride_csv=Path(input_csv),
        confidence_csv=Path(conf_csv) if conf_csv else None,
        output_csv=Path(output_csv),
        confidence_threshold=threshold,
    )

    console.print(f"[green]Original: {results['original']} strides[/]")
    console.print(f"[green]After filtering: {results['filtered']} strides[/]")
    console.print(f"Output saved to: {output_csv}")
    console.print("Note: no angular-velocity / forward-direction filter here; stage 6 of the pipeline applies them.")

    mpath = write_run_manifest(
        Path(output_csv).parent, "gait_filtering",
        params={"confidence_threshold": threshold, "filters": "confidence + edge strides"},
        inputs={"input_csv": input_csv, "confidence_csv": conf_csv},
        outputs=[output_csv], results=results,
    )
    if mpath:
        console.print(f"Run manifest: {mpath}")


@main.command("count-frames")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos (searched recursively, .mp4 / .MP4 / .avi)")
@click.pass_context
def count_frames(ctx, input_dir):
    """Count total frames across all videos in a directory."""
    from stride.utils.video import count_total_frames

    console.print(f"[bold]Counting frames in: {input_dir}[/]")

    total, n_videos = count_total_frames(Path(input_dir))

    console.print(f"[green]Videos: {n_videos}[/]")
    console.print(f"[green]Total frames: {total:,}[/]")
    if n_videos == 0:
        _fail(f"no .mp4 / .avi videos found under {input_dir} (searched recursively, any letter case)")


@main.command("status")
@click.option("--output", "-o", "output_dir", type=click.Path(), default=".",
              help="Output directory of a run (default: current directory)")
@click.option("--checkpoint", "-c", type=click.Path(), default=None,
              help="(legacy) a progress.log checkpoint file to read")
@click.pass_context
def status(ctx, output_dir, checkpoint):
    """Show what has been run under an output directory.

    Every stage writes a <stage>_run_manifest.yaml next to its outputs; this lists them
    (stage, when, code version, result counts).
    """
    import yaml

    if checkpoint:   # legacy file; nothing in STRIDE writes it
        from stride.utils.checkpoint import read_checkpoints
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            console.print(f"[yellow]No checkpoint file at: {checkpoint}. STRIDE does not write progress.log; "
                          f"use `stride status --output <run output dir>` to list the run manifests.[/]")
            raise SystemExit(1)
        table = Table(title="Pipeline Status (legacy checkpoint file)")
        table.add_column("Stage", style="cyan")
        table.add_column("Status", justify="center")
        for name, state in read_checkpoints(checkpoint_path).items():
            table.add_row(name, state)
        console.print(table)
        return

    root = Path(output_dir)
    manifests = sorted(root.rglob("*_run_manifest.yaml")) if root.exists() else []
    if not manifests:
        console.print(f"[yellow]No run manifests under {root}. Each stage writes "
                      f"<stage>_run_manifest.yaml next to its outputs; run a stage first, or pass "
                      f"--output <the --output folder of your run>.[/]")
        raise SystemExit(1)

    table = Table(title=f"STRIDE runs under {root}")
    table.add_column("Stage", style="cyan", no_wrap=True)
    table.add_column("When (UTC)", overflow="fold")
    table.add_column("Code", overflow="fold")
    table.add_column("Result", overflow="fold")
    table.add_column("Manifest", overflow="fold")
    rows = []
    for m in manifests:
        try:
            d = yaml.safe_load(m.read_text()) or {}
        except Exception as exc:
            rows.append(("?", "", "", f"unreadable: {exc}", str(m.relative_to(root))))
            continue
        git = d.get("git") or {}
        code = f"{d.get('stride_version', '?')} {git.get('commit_short', '')}{'+dirty' if git.get('dirty') else ''}".strip()
        res = d.get("results") or {}
        short = ", ".join(f"{k}={v}" for k, v in res.items()
                          if isinstance(v, (int, float, str)) and not isinstance(v, bool))[:70]
        when = str(d.get("timestamp_utc", ""))[:19].replace("T", " ")
        rows.append((str(d.get("stage", "?")), when, code, short, str(m.relative_to(root))))
    for r in sorted(rows, key=lambda r: r[1]):
        table.add_row(*r)
    console.print(table)



# ---------------------------------------------------------------------------------------------
# STRIDE T-maze: the fully automated pipeline
# ---------------------------------------------------------------------------------------------

def _tmaze_steps():
    from stride.stages.tmaze_pipeline import STEPS
    return STEPS


@main.command("tmaze")
@click.argument("videos", nargs=-1, type=click.Path())
@click.option("--config", "-c", "config_path", type=click.Path(), default=None,
              help="YAML config (`stride init-config tmaze` writes a template). Options given on "
                   "the command line override the config.")
@click.option("--out", "-o", "output", type=click.Path(), default=None, help="Output folder")
@click.option("--pose-model", "pose_model", multiple=True, type=click.Path(),
              help="Your trained pose model: give the option once or twice (centroid, centered instance)")
@click.option("--roi-model", "roi_model", multiple=True, type=click.Path(),
              help="Your trained maze-ROI model: give the option once or twice")
@click.option("--metadata", "-m", type=click.Path(), multiple=True,
              help="Optional CSV (repeatable). Trial rows: `video` (file name) or `day` + `mouse`, plus "
                   "`reward` (L/R); optional `trial`, `sex`, `genotype`. Mouse rows: `mouse` + `sex` / `genotype`")
@click.option("--pose-dir", type=click.Path(), default=None,
              help="Folder with existing pose files, when they are not next to the videos")
@click.option("--roi-dir", type=click.Path(), default=None,
              help="Folder with existing ROI files, when they are not next to the videos")
@click.option("--px-per-cm", "px_per_cm", type=float, default=None,
              help="Scale you measured yourself (wins over the ChArUco board and the ROI width)")
@click.option("--maze-width-cm", "maze_width_cm", type=float, default=None,
              help="Corridor width for the ROI-based scale (default 10)")
@click.option("--board-squares-x", type=int, default=None, help="ChArUco board: squares along x")
@click.option("--board-squares-y", type=int, default=None, help="ChArUco board: squares along y")
@click.option("--board-square-mm", type=float, default=None, help="ChArUco board: square size in mm")
@click.option("--board-marker-mm", type=float, default=None, help="ChArUco board: marker size in mm")
@click.option("--board-dict", type=str, default=None, help="ChArUco board: dictionary, e.g. DICT_5X5_250")
@click.option("--device", type=str, default=None, help="Inference device: auto (default), cuda, cuda:0, cpu")
@click.option("--workers", "-w", type=int, default=None, help="CPU worker processes for the analysis (default 4)")
@click.option("--batch-size", type=int, default=None, help="Pose inference batch size (default 16)")
@click.option("--pose-suffix", type=str, default=None, help="Pose file = <video name><suffix> (default .slp)")
@click.option("--skip-inference", is_flag=True,
              help="Do not run the two inferences: analyse the videos that already have pose and ROI files")
@click.option("--skip", "skip", multiple=True, metavar="STEP",
              help="Switch a step off (repeatable): distortion_check, calibrate, pose_inference, "
                   "roi_inference, roi_consensus, decisions, decision_rules, gait")
@click.option("--enable", "enable", multiple=True, metavar="STEP",
              help="Switch a step on (repeatable), e.g. mixed_models, classifier (off by default)")
@click.option("--stop-after", "stop_after", type=str, default=None, metavar="STEP",
              help="Stop after this step")
@click.option("--force", is_flag=True, help="Redo steps even when their outputs are up to date")
@click.option("--strict", is_flag=True, help="Exit with an error when any video was left out")
@click.pass_context
def tmaze(ctx, videos, config_path, output, pose_model, roi_model, metadata, pose_dir, roi_dir,
          px_per_cm, maze_width_cm,
          board_squares_x, board_squares_y, board_square_mm, board_marker_mm, board_dict, device, workers,
          batch_size, pose_suffix, skip_inference, skip, enable, stop_after, force, strict):
    """STRIDE T-maze: videos + your pose and ROI models in, per-trial and per-stride tables out.

    VIDEOS is a folder (searched recursively), a video file, or a pattern; several may be given.
    Steps: discover, distortion check (report only), scale (px/cm), pose inference, ROI inference
    (+ consensus ROI), decisions, decision rules, gait. Finished work is not repeated on a second run.

    \b
    stride tmaze /data/videos --pose-model P1 --pose-model P2 --roi-model R1 --roi-model R2 -o out
    stride tmaze --config my_tmaze.yaml
    stride tmaze /data/videos -o out --skip-inference        # pose and ROI files already exist
    """
    from stride.stages.tmaze_pipeline import STEPS, TmazeError, TmazeRun, load_settings

    bad = [s for s in list(skip) + list(enable) if s not in STEPS]
    if bad:
        _fail(f"unknown step(s) {bad}. Steps: {', '.join(STEPS)}", 2)
    if stop_after and stop_after not in STEPS:
        _fail(f"--stop-after {stop_after}: unknown step. Steps: {', '.join(STEPS)}", 2)

    ov: dict = {}
    if videos:
        ov["videos"] = list(videos) if len(videos) > 1 else videos[0]
    if metadata:
        ov["metadata"] = list(metadata) if len(metadata) > 1 else metadata[0]
    for key, val in (("output", output), ("device", device), ("workers", workers),
                     ("batch_size", batch_size), ("pose_suffix", pose_suffix), ("stop_after", stop_after),
                     ("pose_dir", pose_dir), ("roi_dir", roi_dir)):
        if val is not None:
            ov[key] = val
    if pose_model:
        ov["pose_model"] = list(pose_model)
    if roi_model:
        ov["roi_model"] = list(roi_model)
    scale: dict = {}
    if px_per_cm is not None:
        scale["px_per_cm"] = px_per_cm
    if maze_width_cm is not None:
        scale["maze_width_cm"] = maze_width_cm
    board = {k: v for k, v in (("squares_x", board_squares_x), ("squares_y", board_squares_y),
                               ("square_mm", board_square_mm), ("marker_mm", board_marker_mm),
                               ("dictionary", board_dict)) if v is not None}
    if board:
        scale["charuco"] = board
    if scale:
        ov["scale"] = scale
    steps = {}
    if skip_inference:
        steps.update(pose_inference=False, roi_inference=False)
    steps.update({s: False for s in skip})
    steps.update({s: True for s in enable})
    if steps:
        ov["steps"] = steps
    if force:
        ov["force"] = True
    if strict:
        ov["strict"] = True

    try:
        settings = load_settings(config_path, ov)
        run = TmazeRun(settings)
    except TmazeError as exc:
        _fail(str(exc), 2)
    summary = run.run()
    if not summary["ok"]:
        raise SystemExit(1)
    if settings.get("strict") and summary["skipped"]:
        console.print(f"[bold red]ERROR:[/] {len(summary['skipped'])} video(s) were left out (--strict); "
                      f"see SUMMARY.md", soft_wrap=True)
        raise SystemExit(1)


@main.command("init-config")
@click.argument("pipeline", type=click.Choice(["tmaze"]), default="tmaze")
@click.option("--output", "-o", "output_file", type=click.Path(), default=None,
              help="File to write (default: ./stride_tmaze.yaml)")
@click.option("--overwrite", is_flag=True, help="Replace the file when it exists")
def init_config(pipeline, output_file, overwrite):
    """Write a commented config template for `stride tmaze --config`."""
    from stride.stages.tmaze_pipeline import TmazeError, write_config_template

    path = Path(output_file) if output_file else Path(f"stride_{pipeline}.yaml")
    try:
        write_config_template(path, overwrite=overwrite)
    except TmazeError as exc:
        _fail(str(exc), 2)
    console.print(f"Wrote {path}. Edit it, then run: stride tmaze --config {path}", soft_wrap=True)


@main.command("apply-decision-rules")
@click.argument("decisions_csv", type=click.Path())
@click.option("--window-s", type=float, default=150.0, show_default=True,
              help="Decision window: an arm entry after this many seconds is no decision")
@click.option("--output", "-o", "out_csv", type=click.Path(), default=None,
              help="File to write (default: decisions_clean.csv next to the input)")
def apply_decision_rules_cmd(decisions_csv, window_s, out_csv):
    """Apply the two protocol rules to a decisions.csv and write decisions_clean.csv.

    A trial stays a decision only if the arm entry cleared the strict 95 % body-in-arm threshold
    and happened within the decision window. The input file is not modified.
    """
    from stride.stages.decision_rules import apply_decision_rules

    try:
        apply_decision_rules(Path(decisions_csv), window_s=window_s, out_csv=Path(out_csv) if out_csv else None)
    except (FileNotFoundError, ValueError) as exc:
        _fail(str(exc))


@main.command("propagate-roi")
@click.argument("video_dir", type=click.Path())
@click.option("--drift-warn", type=float, default=25.0, show_default=True,
              help="Static-maze guard: max per-vertex spread across the detected ROIs, in px")
@click.option("--refuse-on-drift", is_flag=True,
              help="Write nothing when the guard is exceeded (default: write with a warning)")
@click.option("--qc-frame", type=float, default=0.5, show_default=True,
              help="Fraction of the clip at which the QC frame is taken")
@click.option("--qc-out", type=click.Path(), default=None,
              help="Folder for the QC overlays (default: <VIDEO_DIR>/roi_consensus_qc)")
def propagate_roi_cmd(video_dir, drift_warn, refuse_on_drift, qc_frame, qc_out):
    """Give the folder's consensus ROI to the videos without a <video name>.rois.yml.

    The consensus is the per-vertex median of the ROI files detected in the same folder (one
    recording session: maze and camera did not move). Existing ROI files are never overwritten;
    QC overlays and montage sheets are written for a person to check.
    """
    from stride.stages.roi_consensus import ConsensusError, propagate_roi_in_folder

    if not Path(video_dir).is_dir():
        _fail(f"folder not found: {video_dir}", 2)
    try:
        res = propagate_roi_in_folder(Path(video_dir), drift_warn=drift_warn,
                                      on_drift="fail" if refuse_on_drift else "warn",
                                      qc_frame=qc_frame, qc_out=qc_out)
    except ConsensusError as exc:
        _fail(str(exc))
    if not res["recovered"]:
        console.print("nothing to recover.")



# px/cm from a ChArUco board in view (`stride calibrate`); also step `calibrate` of `stride tmaze`
from stride.stages.scale_calibration import calibrate_cmd  # noqa: E402

main.add_command(calibrate_cmd)


if __name__ == "__main__":
    main()
