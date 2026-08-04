"""
Command-line interface for the T-maze pipeline.

Usage:
    stride run --input /path/to/videos --output /path/to/results
    stride check-distortion --input /path/to/videos
    stride pose-inference --input /path/to/videos
    stride roi-inference --input /path/to/videos
    stride convert-roi --input /path/to/slp --output /path/to/yml
    stride analyze-decisions --videos /path/to/videos --yml /path/to/roi_yml --meta /path/to/meta.csv
    stride analyze-gait --input /path/to/data
    stride status
"""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table

from stride import __version__
from stride.config import PipelineConfig

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="stride")
@click.pass_context
def main(ctx):
    """T-maze Pipeline - Automated video analysis for behavioral experiments."""
    ctx.ensure_object(dict)


@main.command()
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing input videos")
@click.option("--output", "-o", "output_dir", type=click.Path(), default="./tmaze_output",
              help="Output directory for results")
@click.option("--meta", "-m", "meta_csv", type=click.Path(exists=True),
              help="Metadata CSV with day,mouse,reward columns")
@click.option("--workers", "-w", "n_workers", type=int, default=4,
              help="Number of parallel workers")
@click.option("--skip-distortion", is_flag=True, help="Skip distortion check")
@click.option("--skip-pose", is_flag=True, help="Skip pose inference (use existing .slp)")
@click.option("--skip-roi", is_flag=True, help="Skip ROI inference (use existing ROI .slp)")
@click.option("--batch-size", type=int, default=16, help="Batch size for inference")
@click.option("--device", type=str, default="cuda", help="Device for inference (cuda, cuda:0, cpu)")
@click.pass_context
def run(ctx, input_dir, output_dir, meta_csv, n_workers, skip_distortion, skip_pose, skip_roi, batch_size, device):
    """Run the full T-maze analysis pipeline."""
    from stride.stages import (
        check_distortion_batch,
        run_pose_inference_batch,
        run_roi_inference_batch,
        convert_batch,
        run_decision_analysis,
    )
    from stride.utils.parallel import find_videos

    config = PipelineConfig(
        video_dir=Path(input_dir),
        output_dir=Path(output_dir),
        meta_csv=Path(meta_csv) if meta_csv else None,
        n_workers=n_workers,
    )
    config.ensure_output_dirs()

    console.print(f"[bold green]T-maze Pipeline v{__version__}[/]")
    console.print(f"Input: {config.video_dir}")
    console.print(f"Output: {config.output_dir}")
    console.print(f"Workers: {config.n_workers}")
    console.print()

    # Stage 1: Scan videos
    console.print("[bold cyan]Stage 1: Scanning videos...[/]")
    videos = find_videos(config.video_dir, pattern="*.mp4")
    console.print(f"  Found {len(videos)} videos")

    if len(videos) == 0:
        console.print("[red]No videos found. Exiting.[/]")
        return

    # Stage 2: Distortion check (optional)
    if not skip_distortion:
        console.print("\n[bold cyan]Stage 2: Checking distortion...[/]")
        distortion_results = check_distortion_batch(
            config.video_dir,
            num_frames=10,
            output_file=config.output_dir / "distortion_check.csv",
        )
        needs_undistort = sum(1 for r in distortion_results if r.get("needs_undistortion"))
        console.print(f"  {needs_undistort}/{len(distortion_results)} videos may need undistortion")
    else:
        console.print("\n[bold yellow]Stage 2: Skipped distortion check[/]")

    # Stage 3: Pose inference
    if not skip_pose:
        console.print("\n[bold cyan]Stage 3: Running pose inference...[/]")
        pose_results = run_pose_inference_batch(
            video_dir=config.video_dir,
            output_dir=None,  # Co-locate with videos
            overwrite=False,
            batch_size=batch_size,
            device=device,
        )
        console.print(f"  Done: {pose_results['done']}, Skipped: {pose_results['skipped']}, Failed: {pose_results['failed']}")
    else:
        console.print("\n[bold yellow]Stage 3: Skipped pose inference[/]")

    # Stage 4: ROI inference
    if not skip_roi:
        console.print("\n[bold cyan]Stage 4: Running ROI inference...[/]")
        roi_results = run_roi_inference_batch(
            video_dir=config.video_dir,
            output_dir=config.output_dir / "roi_slp",
            n_workers=1,  # GPU inference single-threaded
            overwrite=False,
        )
        console.print(f"  Passed: {roi_results['passed']}, Skipped: {roi_results['skipped']}, Failed: {roi_results['failed']}")
    else:
        console.print("\n[bold yellow]Stage 4: Skipped ROI inference[/]")

    # Stage 5: SLP to YAML conversion
    console.print("\n[bold cyan]Stage 5: Converting ROI SLP to YAML...[/]")
    convert_results = convert_batch(
        slp_dir=config.output_dir / "roi_slp",
        yml_dir=config.output_dir / "roi_yml",
        overwrite=False,
    )
    console.print(f"  Converted: {convert_results['ok']}, Skipped: {convert_results['skip']}, Failed: {convert_results['failed']}")

    # Stage 6: Decision analysis
    if config.meta_csv and config.meta_csv.exists():
        console.print("\n[bold cyan]Stage 6: Analyzing T-maze decisions...[/]")
        decision_results = run_decision_analysis(
            video_dir=config.video_dir,
            yml_dir=config.output_dir / "roi_yml",
            meta_csv=config.meta_csv,
            output_csv=config.output_dir / "decisions" / "decisions.csv",
            n_workers=config.n_workers,
        )
        console.print(f"  Analyzed: {decision_results['ok']}/{decision_results['total']} trials")
    else:
        console.print("\n[bold yellow]Stage 6: Skipped decision analysis (no metadata CSV)[/]")

    console.print("\n[bold green]Pipeline complete![/]")
    console.print(f"Results saved to: {config.output_dir}")


@main.command("pose-inference")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos")
@click.option("--output", "-o", "output_dir", type=click.Path(), default=None,
              help="Output directory for .slp files (default: same as video)")
@click.option("--model", "-m", "model_paths", type=click.Path(exists=True), multiple=True,
              help="Model paths (centroid and centered_instance)")
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
        tmaze pose-inference -i /videos --gpu 0 --worker-id 0 --num-workers 2

        # Terminal 2 - GPU 1
        tmaze pose-inference -i /videos --gpu 1 --worker-id 1 --num-workers 2
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

    console.print(f"\n[green]Done: {results['done']}/{results['total']} videos[/]")
    console.print(f"Skipped: {results['skipped']}")
    if results['failed'] > 0:
        console.print(f"[red]Failed: {results['failed']} videos[/]")


@main.command("check-distortion")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos or single video file")
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
        table.add_row(
            r.get("video", "?"),
            f"{r.get('line_straightness', 0):.2f}",
            f"{r.get('spacing_uniformity', 0):.4f}",
            f"{r.get('reprojection_error', 0):.2f}",
            status,
        )

    console.print(table)


@main.command("roi-inference")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos")
@click.option("--output", "-o", "output_dir", type=click.Path(), required=True,
              help="Output directory for ROI .slp files")
@click.option("--model", "-m", "model_paths", type=click.Path(exists=True), multiple=True,
              help="Model paths (centroid and centered_instance)")
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
def roi_inference(ctx, input_dir, output_dir, model_paths, device, gpu, worker_id, num_workers, overwrite):
    """Run ROI keypoint inference on videos.

    For multi-GPU parallelism, run multiple instances with different worker IDs:

        # Terminal 1 - GPU 0
        tmaze roi-inference -i /videos -o /output --gpu 0 --worker-id 0 --num-workers 2

        # Terminal 2 - GPU 1
        tmaze roi-inference -i /videos -o /output --gpu 1 --worker-id 1 --num-workers 2
    """
    from stride.stages.roi_inference import run_roi_inference_batch

    # Handle --gpu shorthand
    if gpu is not None:
        device = f"cuda:{gpu}"

    console.print(f"[bold]Running ROI inference[/]")
    console.print(f"Input: {input_dir}")
    console.print(f"Output: {output_dir}")
    console.print(f"Device: {device}")
    if num_workers > 1:
        console.print(f"Worker: {worker_id + 1}/{num_workers}")

    model_paths = list(model_paths) if model_paths else None
    results = run_roi_inference_batch(
        video_dir=Path(input_dir),
        output_dir=Path(output_dir),
        model_paths=model_paths,
        overwrite=overwrite,
        worker_id=worker_id,
        num_workers=num_workers,
    )

    console.print(f"[green]Completed: {results['passed']}/{results['total']} videos[/]")
    if results['failed'] > 0:
        console.print(f"[red]Failed: {results['failed']} videos[/]")


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


@main.command("analyze-decisions")
@click.option("--videos", "-v", "video_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos with .slp pose files")
@click.option("--yml", "-y", "yml_dir", type=click.Path(exists=True), required=True,
              help="Directory containing ROI .yml files")
@click.option("--meta", "-m", "meta_csv", type=click.Path(exists=True), required=True,
              help="Metadata CSV with day,mouse,reward columns")
@click.option("--output", "-o", "output_csv", type=click.Path(), default="decisions.csv",
              help="Output CSV file")
@click.option("--workers", "-w", "n_workers", type=int, default=4,
              help="Number of parallel workers")
@click.pass_context
def analyze_decisions(ctx, video_dir, yml_dir, meta_csv, output_csv, n_workers):
    """Analyze T-maze arm entry decisions."""
    from stride.stages.decision_analysis import run_decision_analysis

    console.print(f"[bold]Analyzing T-maze decisions[/]")
    console.print(f"Videos: {video_dir}")
    console.print(f"ROI YMLs: {yml_dir}")
    console.print(f"Metadata: {meta_csv}")

    results = run_decision_analysis(
        video_dir=Path(video_dir),
        yml_dir=Path(yml_dir),
        meta_csv=Path(meta_csv),
        output_csv=Path(output_csv),
        n_workers=n_workers,
    )

    console.print(f"[green]Analyzed: {results['ok']}/{results['total']} trials[/]")
    console.print(f"Output saved to: {output_csv}")


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
    """Filter and analyze gait stride data."""
    from stride.stages.gait_filtering import filter_strides

    console.print(f"[bold]Filtering gait strides[/]")
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


@main.command("count-frames")
@click.option("--input", "-i", "input_dir", type=click.Path(exists=True), required=True,
              help="Directory containing videos")
@click.pass_context
def count_frames(ctx, input_dir):
    """Count total frames across all videos in a directory."""
    from stride.utils.video import count_total_frames

    console.print(f"[bold]Counting frames in: {input_dir}[/]")

    total, n_videos = count_total_frames(Path(input_dir))

    console.print(f"[green]Videos: {n_videos}[/]")
    console.print(f"[green]Total frames: {total:,}[/]")


@main.command("status")
@click.option("--checkpoint", "-c", type=click.Path(), default="progress.log",
              help="Checkpoint file to read")
@click.pass_context
def status(ctx, checkpoint):
    """Show pipeline checkpoint status."""
    from stride.utils.checkpoint import read_checkpoints

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        console.print(f"[yellow]No checkpoint file found at: {checkpoint}[/]")
        return

    checkpoints = read_checkpoints(checkpoint_path)

    table = Table(title="Pipeline Status")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", justify="center")

    status_colors = {
        "pending": "[yellow]PENDING[/]",
        "running": "[blue]RUNNING[/]",
        "completed": "[green]COMPLETED[/]",
        "failed": "[red]FAILED[/]",
    }

    for name, status in checkpoints.items():
        table.add_row(name, status_colors.get(status, status))

    console.print(table)


if __name__ == "__main__":
    main()
