#!/usr/bin/env python
"""Inference LAUNCH test: does every inference-side entry point of STRIDE start correctly on one tiny clip?

This is not an accuracy test. SLEAP inference itself is assumed to work; what can go wrong is the launch:
finding the videos, loading the models that were GIVEN (not a built-in default), passing the device on,
writing the output files where the next stage looks for them. Each launch prints one PASS / FAIL line.

Give it a very short clip (16-24 frames of a real recording is enough) and your own trained models:

    python tests/mock/run_inference_launch.py \
        --clip /path/to/tiny_clip.mp4 \
        --pose-models /models/pose/centroid /models/pose/centered_instance \
        --roi-models  /models/roi/centroid  /models/roi/centered_instance \
        --device cpu --out /tmp/stride_launch

Run it with the Python that has sleap-nn installed. The STRIDE code under test is the repository this file
sits in (its src/ folder is put first on PYTHONPATH). With --device cpu every child process gets
CUDA_VISIBLE_DEVICES="" so no GPU is touched; repeat with --device cuda:0 to exercise GPU selection.

Optional:
    --ref-pose-slp FILE --ref-frame-offset N   compare the new pose file with an existing one (clip frame i = reference frame N+i)
    --ref-roi-yml FILE                         compare the new ROI polygons with an existing ROI yml
    --no-sleapnn-python PY                     a Python WITHOUT sleap-nn, to check the message a user gets in that case
    --only a,b  /  --skip a,b                  launch ids (see the printed lines)

Outputs in --out: logs/<id>.log, launch_results.json, and the work folders the launches wrote into.
Exit code 1 if any launch FAILs.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "src"
RESULTS: list[dict] = []
ARGS = None
ENV = None


# ----------------------------------------------------------------------------- helpers
def child_env(device: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""          # make sure nothing lands on a GPU by default
    return env


def run(lid: str, cmd: list[str], cwd: Path | None = None, python: str | None = None) -> tuple[int, str, float]:
    """Run one launch, keep its log, return (exit code, combined output, seconds)."""
    full = [python or sys.executable, *cmd]
    t0 = time.time()
    try:
        p = subprocess.run(full, cwd=str(cwd or REPO), env=ENV, capture_output=True, text=True, timeout=ARGS.timeout)
        rc, out = p.returncode, (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired as e:
        rc, out = 124, f"TIMEOUT after {ARGS.timeout}s\n{e.stdout or ''}{e.stderr or ''}"
    dt = time.time() - t0
    (ARGS.out / "logs").mkdir(parents=True, exist_ok=True)
    (ARGS.out / "logs" / f"{lid}.log").write_text("$ " + " ".join(full) + f"\n[exit {rc}, {dt:.1f}s]\n\n" + out)
    return rc, out, dt


def record(lid: str, entry: str, expect: str, ok: bool, reason: str, seconds: float | None = None, rc: int | None = None):
    RESULTS.append(dict(id=lid, entry=entry, expect=expect, status="PASS" if ok else "FAIL", reason=reason,
                        seconds=None if seconds is None else round(seconds, 1), exit_code=rc))
    t = "" if seconds is None else f" [{seconds:.0f}s]"
    print(f"{'PASS' if ok else 'FAIL'}  {lid:<22} {entry}: {reason}{t}", flush=True)


def last_error(out: str) -> str:
    """The most useful line of a failed launch: last traceback / error line."""
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    for l in reversed(lines):
        if any(k in l for k in ("Error", "ERROR", "error:", "Exception", "No such", "not found", "denied", "Usage:")):
            return l[:220]
    return (lines[-1] if lines else "no output")[:220]


def has_traceback(out: str) -> bool:
    return "Traceback (most recent call last)" in out


def slp_info(path: Path) -> dict:
    import sleap_io as sio
    lab = sio.load_slp(str(path), open_videos=False)
    frames = [lf for lf in lab if lf.instances]
    prov = dict(lab.provenance or {})
    return dict(nodes=[n.name for n in lab.skeleton.nodes], n_frames_with_instance=len(frames), n_labeled_frames=len(lab),
                max_instances=max((len(lf.instances) for lf in lab), default=0),
                model_paths=[str(m) for m in (prov.get("model_paths") or [])], device=str(prov.get("device", "")), labels=lab)


def same_models(loaded: list[str], given: list[str]) -> bool:
    return [os.path.realpath(m) for m in loaded] == [os.path.realpath(m) for m in given]


def check_pose_slp(path: Path, given_models: list[str], device: str) -> tuple[bool, str]:
    if not path.exists():
        return False, f"no pose file written ({path.name} missing)"
    i = slp_info(path)
    problems = []
    if i["n_frames_with_instance"] == 0:
        problems.append("pose file has no predicted instance")
    if i["model_paths"] and not same_models(i["model_paths"], given_models):
        problems.append(f"loaded models {i['model_paths']} are not the ones given")
    if i["device"] and device != "auto" and not i["device"].startswith(device.split(":")[0]):
        problems.append(f"ran on device '{i['device']}', asked for '{device}'")
    msg = (f"{path.name}: {i['n_frames_with_instance']} frames with an instance, {len(i['nodes'])} nodes, "
           f"models = the ones given, device {i['device'] or '?'}")
    if ARGS.ref_pose_slp and path.exists():
        msg += "; " + compare_pose(i["labels"])
    return (not problems), ("; ".join(problems) if problems else msg)


def compare_pose(new) -> str:
    import numpy as np, sleap_io as sio
    ref = sio.load_slp(str(ARGS.ref_pose_slp), open_videos=False)
    if [n.name for n in ref.skeleton.nodes] != [n.name for n in new.skeleton.nodes]:
        return "node names DIFFER from the reference pose file"

    def best(labels, off=0):
        out = {}
        for lf in labels:
            if lf.instances:
                inst = max(lf.instances, key=lambda k: float(getattr(k, "score", 0) or 0))
                out[lf.frame_idx - off] = inst.numpy()[:, :2]
        return out
    a, b = best(new), best(ref, ARGS.ref_frame_offset)
    fr = sorted(set(a) & set(b))
    if not fr:
        return "no common frames with the reference pose file"
    d = np.array([np.linalg.norm(a[f] - b[f], axis=1) for f in fr])
    return (f"vs reference: same node names, {len(fr)} common frames, keypoint distance median "
            f"{np.nanmedian(d):.2f} px / max {np.nanmax(d):.2f} px")


def yml_regions(path: Path) -> dict:
    import yaml
    d = yaml.safe_load(open(path))
    return {r["name"]: r["coordinates"] for r in d.get("rois", [])}


def compare_roi(path: Path) -> str:
    import numpy as np
    new, ref = yml_regions(path), yml_regions(ARGS.ref_roi_yml)
    common = sorted(set(new) & set(ref))
    if not common:
        return "no region names in common with the reference ROI yml"
    worst = 0.0
    for r in common:
        a, b = np.array(new[r], float), np.array(ref[r], float)
        if a.shape == b.shape:
            worst = max(worst, float(np.linalg.norm(a - b, axis=1).max()))
        else:
            return f"region {r}: vertex count differs from the reference"
    return f"vs reference ROI: {len(common)} regions in common, largest vertex distance {worst:.1f} px"


def place(clip: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        shutil.copy2(clip, dst)
    return dst


def wanted(lid: str) -> bool:
    if ARGS.only and lid not in ARGS.only:
        return False
    return lid not in ARGS.skip


def cli(*a: str) -> list[str]:
    return ["-m", "stride.cli", *a]


# ----------------------------------------------------------------------------- launches
def main():
    global ARGS, ENV
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", required=True, type=Path, help="a very short video (16-24 frames is enough)")
    ap.add_argument("--pose-models", required=True, nargs="+", help="pose model folders (centroid, centered instance)")
    ap.add_argument("--roi-models", required=True, nargs="+", help="ROI model folders (centroid, centered instance)")
    ap.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, ... (default cpu)")
    ap.add_argument("--out", required=True, type=Path, help="work + log folder (created)")
    ap.add_argument("--ref-pose-slp", type=Path, default=None)
    ap.add_argument("--ref-frame-offset", type=int, default=0)
    ap.add_argument("--ref-roi-yml", type=Path, default=None)
    ap.add_argument("--no-sleapnn-python", default=None, help="a Python without sleap-nn (checks the message)")
    ap.add_argument("--only", default="", help="comma-separated launch ids to run")
    ap.add_argument("--skip", default="", help="comma-separated launch ids to skip")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds per launch")
    ARGS = ap.parse_args()
    ARGS.only = [s for s in ARGS.only.split(",") if s]
    ARGS.skip = [s for s in ARGS.skip.split(",") if s]
    ARGS.out = ARGS.out.resolve()
    ARGS.out.mkdir(parents=True, exist_ok=True)
    ENV = child_env(ARGS.device)
    dev = ARGS.device
    pose_m, roi_m = [str(Path(m).resolve()) for m in ARGS.pose_models], [str(Path(m).resolve()) for m in ARGS.roi_models]
    for m in pose_m + roi_m:
        if not Path(m).exists():
            sys.exit(f"model path not found: {m}")
    if not ARGS.clip.exists():
        sys.exit(f"clip not found: {ARGS.clip}")

    import cv2
    cap = cv2.VideoCapture(str(ARGS.clip)); n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS); cap.release()
    print(f"clip: {ARGS.clip.name}, {n_frames} frames, {fps:.2f} fps | device: {dev} | python: {sys.executable}")
    print(f"STRIDE under test: {SRC}\n")

    W = ARGS.out / "work"
    lower = place(ARGS.clip, W / "flat_lower" / "Day1_3_Trial1.mp4")          # the case that must work everywhere
    upper = place(ARGS.clip, W / "flat_upper" / "Day1_1_Trial1.MP4")          # camera files usually come as .MP4
    perday = place(ARGS.clip, W / "per_day" / "2026-01-01-DAY1" / "Day1_2_Trial1.mp4")   # one folder per recording day
    mflags_pose = [x for m in pose_m for x in ("-m", m)]
    mflags_roi = [x for m in roi_m for x in ("-m", m)]

    def supports(cmd: list[str], opt: str) -> bool:
        """True when `<cmd> --help` lists the option (the same test serves older and newer STRIDE versions)."""
        try:
            return opt in subprocess.run([sys.executable, *cmd, "--help"], env=ENV, capture_output=True, text=True, timeout=300).stdout
        except Exception:
            return False

    # options added after the first baseline: the config-driven runners take --device, `stride run` takes the models
    dev_inf = ["--device", dev] if supports(["-m", "stride.stages.run_inference"], "--device") else []
    dev_all = ["--device", dev] if supports([str(REPO / "run_all.py")], "--device") else []
    run_models = ([x for m in pose_m for x in ("--pose-model", m)] + [x for m in roi_m for x in ("--roi-model", m)]
                  if supports(cli("run"), "--pose-model") else [])

    # ---- pose-inference ---------------------------------------------------------------------------------
    if wanted("pose_flat_lower"):
        rc, out, dt = run("pose_flat_lower", cli("pose-inference", "-i", str(lower.parent), *mflags_pose, "-d", dev, "-b", "4"))
        ok, why = check_pose_slp(lower.with_suffix(".slp"), pose_m, dev)
        record("pose_flat_lower", "stride pose-inference (flat folder, .mp4, --model, --device)",
               "pose file next to the video, made with the given models on the given device",
               rc == 0 and ok, why if rc == 0 else f"exit {rc}: {last_error(out)}", dt, rc)
    for lid, vid, what in (("pose_flat_upper", upper, "flat folder, .MP4"), ("pose_per_day", perday, "videos in a per-day subfolder")):
        if wanted(lid):
            root = vid.parent if lid == "pose_flat_upper" else W / "per_day"
            rc, out, dt = run(lid, cli("pose-inference", "-i", str(root), *mflags_pose, "-d", dev, "-b", "4"))
            ok, why = check_pose_slp(vid.with_suffix(".slp"), pose_m, dev)
            if not ok and "Done: 0/0" in out:
                why = "the video was not found: 0 videos processed, exit 0, no warning"
            record(lid, f"stride pose-inference ({what})", "the video is found and gets a pose file", rc == 0 and ok,
                   why if rc == 0 else f"exit {rc}: {last_error(out)}", dt, rc)
    if wanted("pose_output_dir"):
        od = W / "pose_out"
        rc, out, dt = run("pose_output_dir", cli("pose-inference", "-i", str(lower.parent), "-o", str(od), *mflags_pose, "-d", dev, "-b", "4"))
        ok, why = check_pose_slp(od / f"{lower.stem}.slp", pose_m, dev)
        record("pose_output_dir", "stride pose-inference --output DIR", "pose file written into DIR", rc == 0 and ok,
               why if rc == 0 else f"exit {rc}: {last_error(out)}", dt, rc)
    if wanted("pose_resume_workers"):
        two = W / "two_clips"
        a = place(ARGS.clip, two / "Day1_4_Trial1.mp4"); b = place(ARGS.clip, two / "Day1_5_Trial1.mp4")
        src_slp = lower.with_suffix(".slp")
        if src_slp.exists():
            for v in (a, b):
                shutil.copy2(src_slp, v.with_suffix(".slp"))
            seen, tot = [], 0.0
            for wid in (0, 1):
                rc, out, dt = run(f"pose_resume_workers_w{wid}", cli("pose-inference", "-i", str(two), *mflags_pose, "-d", dev,
                                                                   "--worker-id", str(wid), "--num-workers", "2"))
                tot += dt
                seen.append(sorted(v.name for v in (a, b) if v.name in out and "SKIP" in out))
            ok = sorted(seen[0] + seen[1]) == sorted([a.name, b.name]) and len(seen[0]) == 1 and len(seen[1]) == 1
            record("pose_resume_workers", "stride pose-inference --worker-id / --num-workers on finished clips",
                   "each worker takes its own share and skips clips that already have a pose file", ok,
                   f"worker 0 -> {seen[0]}, worker 1 -> {seen[1]}, existing files skipped" if ok else f"worker shares {seen}", tot, rc)
        else:
            record("pose_resume_workers", "stride pose-inference --worker-id / --num-workers", "needs the pose file of pose_flat_lower", False,
                   "not run: no pose file from pose_flat_lower")
    if wanted("pose_no_model"):
        nm = place(ARGS.clip, W / "no_model" / "Day1_6_Trial1.mp4")
        rc, out, dt = run("pose_no_model", cli("pose-inference", "-i", str(nm.parent), "-d", dev))
        clear = rc != 0 and "--model" in out and not has_traceback(out)
        record("pose_no_model", "stride pose-inference without --model", "stops with a message that says to pass --model", clear,
               "clear message, non-zero exit" if clear else f"exit {rc}: {last_error(out)}" + (" (Python traceback shown)" if has_traceback(out) else ""), dt, rc)
    if wanted("pose_no_sleapnn") and ARGS.no_sleapnn_python:
        ns = place(ARGS.clip, W / "no_sleapnn" / "Day1_7_Trial1.mp4")
        rc, out, dt = run("pose_no_sleapnn", cli("pose-inference", "-i", str(ns.parent), *mflags_pose, "-d", dev), python=ARGS.no_sleapnn_python)
        clear = rc != 0 and "sleap-nn" in out.lower().replace("_", "-")
        record("pose_no_sleapnn", "stride pose-inference in an environment without sleap-nn", "stops and says sleap-nn is not installed", clear,
               "clear message, non-zero exit" if clear else f"exit {rc}; output says: {last_error(out)}", dt, rc)

    # ---- roi-inference ----------------------------------------------------------------------------------
    roi_slp_dir, roi_yml_dir = W / "roi_slp", W / "roi_yml"
    roi_out = roi_slp_dir / f"{lower.stem}.preds.v2.best1.slp"
    if wanted("roi_flat_lower"):
        rc, out, dt = run("roi_flat_lower", cli("roi-inference", "-i", str(lower.parent), "-o", str(roi_slp_dir), *mflags_roi, "-d", dev))
        if rc == 0 and roi_out.exists():
            i = slp_info(roi_out)
            ok = i["n_frames_with_instance"] >= 1 and (not i["model_paths"] or same_models(i["model_paths"], roi_m))
            conf = (f"models = the ones given, device {i['device']}" if i["model_paths"] else
                    "models and device are not recorded in this file (the best-frame file is rewritten without its provenance)")
            why = (f"{roi_out.name}: {i['n_labeled_frames']} frame kept, {len(i['nodes'])} nodes, {i['max_instances']} instance(s) in it, {conf}"
                   ) if ok else f"loaded models {i['model_paths']} / instances {i['n_frames_with_instance']}"
        else:
            ok, why = False, (f"exit {rc}: {last_error(out)}" if rc else f"no ROI file written: {last_error(out)}")
        record("roi_flat_lower", "stride roi-inference (flat folder, .mp4, --model, --device)",
               "one best-frame ROI file per video, made with the given models", ok, why, dt, rc)
    if wanted("roi_device_forwarded"):
        sys.path.insert(0, str(SRC))
        try:
            from stride.stages import roi_inference as R
            src = inspect.getsource(R.run_single_video_inference)
            # the device must be an argument of both functions and be handed to the sleap-nn call
            # (written either as `device=device` or through a keyword dict `{"device": device}`)
            ok = ("device" in inspect.signature(R.run_roi_inference_batch).parameters
                  and "device" in inspect.signature(R.run_single_video_inference).parameters
                  and ("device=device" in src or '"device": device' in src))
            record("roi_device_forwarded", "stride roi-inference --device / --gpu", "the device reaches the model call", ok,
                   "device is passed to the inference call" if ok else
                   "the option is accepted but never passed on: run_roi_inference_batch has no device argument, so sleap-nn picks the device itself")
        except Exception as e:
            record("roi_device_forwarded", "stride roi-inference --device / --gpu", "the device reaches the model call", False, f"could not inspect: {e}")
    for lid, vid, what in (("roi_flat_upper", upper, "flat folder, .MP4"), ("roi_per_day", perday, "videos in a per-day subfolder")):
        if wanted(lid):
            root = vid.parent if lid == "roi_flat_upper" else W / "per_day"
            od = W / f"{lid}_slp"
            rc, out, dt = run(lid, cli("roi-inference", "-i", str(root), "-o", str(od), *mflags_roi, "-d", dev))
            ok = (od / f"{vid.stem}.preds.v2.best1.slp").exists()
            why = "ROI file written" if ok else ("the video was not found: 0 videos processed, exit 0, no warning" if "0/0" in out else f"exit {rc}: {last_error(out)}")
            record(lid, f"stride roi-inference ({what})", "the video is found and gets an ROI file", rc == 0 and ok, why, dt, rc)
    if wanted("roi_no_model"):
        rc, out, dt = run("roi_no_model", cli("roi-inference", "-i", str(lower.parent), "-o", str(W / "roi_no_model"), "-d", dev))
        clear = rc != 0 and "--model" in out and not has_traceback(out)
        record("roi_no_model", "stride roi-inference without --model", "stops with a message that says to pass --model", clear,
               "clear message, non-zero exit" if clear else f"exit {rc}: {last_error(out)}" + (" (Python traceback shown)" if has_traceback(out) else ""), dt, rc)

    # ---- convert-roi ------------------------------------------------------------------------------------
    yml = roi_yml_dir / f"{lower.stem}.preds.v2.best1.yml"
    if wanted("convert_roi"):
        rc, out, dt = run("convert_roi", cli("convert-roi", "-i", str(roi_slp_dir), "-o", str(roi_yml_dir)))
        if rc == 0 and yml.exists():
            reg = yml_regions(yml)
            ok = len(reg) == 7
            why = f"{yml.name}: {len(reg)} regions ({', '.join(sorted(reg))})" + ("; " + compare_roi(yml) if ARGS.ref_roi_yml else "")
        else:
            ok, why = False, (f"exit {rc}: {last_error(out)}" if rc else f"no yml written: {last_error(out)}")
        record("convert_roi", "stride convert-roi", "one ROI yml with the 7 maze regions", ok, why, dt, rc)

    # ---- check-distortion / count-frames ----------------------------------------------------------------
    for lid, root, name, what in (("distortion_flat_lower", lower.parent, lower.name, "flat folder, .mp4"),
                                  ("distortion_flat_upper", upper.parent, upper.name, "flat folder, .MP4"),
                                  ("distortion_per_day", W / "per_day", perday.name, "per-day subfolder")):
        if wanted(lid):
            csv = W / f"{lid}.csv"
            rc, out, dt = run(lid, cli("check-distortion", "-i", str(root), "-o", str(csv), "-n", "4"))
            listed = Path(name).stem in out or (csv.exists() and Path(name).stem in csv.read_text())
            record(lid, f"stride check-distortion ({what})", "the video is checked and listed", rc == 0 and listed,
                   "video listed in the result" if rc == 0 and listed else (f"exit {rc}: {last_error(out)}" if rc else "the video was not found: empty result, exit 0"), dt, rc)
    for lid, root, what in (("count_flat_lower", lower.parent, "flat folder, .mp4"), ("count_flat_upper", upper.parent, "flat folder, .MP4"),
                            ("count_per_day", W / "per_day", "per-day subfolder")):
        if wanted(lid):
            rc, out, dt = run(lid, cli("count-frames", "-i", str(root)))
            ok = rc == 0 and str(n_frames) in out.replace(",", "")
            record(lid, f"stride count-frames ({what})", f"reports {n_frames} frames", ok,
                   f"{n_frames} frames reported" if ok else (f"exit {rc}: {last_error(out)}" if rc else f"did not report {n_frames} frames: {last_error(out)}"), dt, rc)

    # ---- config-driven: python -m stride.stages.run_inference, python run_all.py --stages 1-2 -----------
    def write_cfg(name: str, video_dir: Path) -> Path:
        import yaml
        cfg = {"batches": {"launch_test": {"name": "Inference launch test", "video_dir": str(video_dir), "fps": float(round(fps, 3)),
                                           "pose_slp_suffix": ".slp", "pose_model_paths": pose_m, "roi_model_paths": roi_m}}}
        p = W / f"{name}.yaml"
        p.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return p

    def check_cfg_run(vid: Path, out_dir: Path, out: str, rc: int) -> tuple[bool, str]:
        parts = []
        pose_ok, pose_why = check_pose_slp(vid.with_suffix(".slp"), pose_m, dev)
        parts.append("pose: " + (pose_why if pose_ok else "FAILED - " + (pose_why if vid.with_suffix('.slp').exists() else last_stage_line(out, "Stage 2a"))))
        r = out_dir / "launch_test" / "roi_slp" / f"{vid.stem}.preds.v2.best1.slp"
        roi_ok = r.exists() and same_models(slp_info(r)["model_paths"], roi_m)
        parts.append("ROI: " + ("made with the configured models" if roi_ok else "FAILED - " + (last_stage_line(out, "Stage 2b") if not r.exists() else "not the configured models")))
        y = out_dir / "launch_test" / "roi_yml" / f"{vid.stem}.preds.v2.best1.yml"
        parts.append("yml: " + ("written" if y.exists() else "FAILED - not written"))
        return (rc == 0 and pose_ok and roi_ok and y.exists()), "; ".join(parts) + (f"; exit code {rc}" if rc else "; exit code 0")

    def last_stage_line(out: str, stage: str) -> str:
        lines = out.splitlines()
        idx = [k for k, l in enumerate(lines) if stage in l]
        if not idx:
            return "stage did not start"
        for l in lines[idx[-1] + 1: idx[-1] + 40]:
            if "ERROR" in l or "Error" in l or "failed" in l or "Done:" in l:
                return l.strip()[:200]
        return lines[idx[-1]].strip()[:200]

    if wanted("run_inference_module"):
        v = place(ARGS.clip, W / "cfg_module" / "Day1_8_Trial1.mp4")
        cfg, od = write_cfg("cfg_module", v.parent), W / "cfg_module_out"
        rc, out, dt = run("run_inference_module", ["-m", "stride.stages.run_inference", "--config", str(cfg), "--batch", "launch_test", "--output", str(od), *dev_inf])
        ok, why = check_cfg_run(v, od, out, rc)
        record("run_inference_module", "python -m stride.stages.run_inference --config --batch --output",
               "pose file, ROI file and ROI yml, all made with the models named in the config", ok, why, dt, rc)
    if wanted("run_all_stage_1_2"):
        v = place(ARGS.clip, W / "cfg_runall" / "Day1_9_Trial1.mp4")
        cfg, od = write_cfg("cfg_runall", v.parent), W / "cfg_runall_out"
        rc, out, dt = run("run_all_stage_1_2", [str(REPO / "run_all.py"), "--stages", "1-2", "--config", str(cfg), "--batch", "launch_test", "--output", str(od), *dev_all])
        ok, why = check_cfg_run(v, od, out, rc)
        record("run_all_stage_1_2", "python run_all.py --stages 1-2 --config --batch --output",
               "pose file, ROI file and ROI yml, all made with the models named in the config", ok, why, dt, rc)
    if wanted("run_inference_per_day"):
        v = place(ARGS.clip, W / "cfg_perday" / "2026-01-01-DAY1" / "Day1_10_Trial1.MP4")
        cfg, od = write_cfg("cfg_perday", W / "cfg_perday"), W / "cfg_perday_out"
        rc, out, dt = run("run_inference_per_day", ["-m", "stride.stages.run_inference", "--config", str(cfg), "--batch", "launch_test", "--output", str(od), "--skip-roi", *dev_inf])
        ok, why = check_pose_slp(v.with_suffix(".slp"), pose_m, dev)
        record("run_inference_per_day", "python -m stride.stages.run_inference (per-day subfolder, .MP4)", "the video is found and gets a pose file", rc == 0 and ok,
               why if ok else f"no pose file; the stage printed: {last_stage_line(out, 'Stage 2a')}; exit code {rc}", dt, rc)

    # ---- stride run ----------------------------------------------------------------------------------------
    if wanted("stride_run"):
        v = place(ARGS.clip, W / "stride_run" / "Day1_11_Trial1.mp4")
        rc, out, dt = run("stride_run", cli("run", "-i", str(v.parent), "-o", str(W / "stride_run_out"), "--device", dev, "--skip-distortion", *run_models))
        ok = rc == 0 and v.with_suffix(".slp").exists()
        record("stride_run", "stride run --input --output --device", "the full pipeline starts and writes a pose file", ok,
               "pipeline ran" if ok else f"exit {rc}: {last_error(out)}", dt, rc)

    # ---- chain: are the fresh files accepted by the next stages? -----------------------------------------
    if wanted("chain_next_stages"):
        chain = f"""
import sys, json, csv
from pathlib import Path
from stride.stages.decision_analysis import process_single_video
from stride.config import PipelineConfig
video, yml_dir, out = Path({str(lower)!r}), Path({str(roi_yml_dir)!r}), Path({str(W / 'chain')!r})
out.mkdir(parents=True, exist_ok=True)
res = {{}}
r = process_single_video(video_path=video, yml_dir=yml_dir, meta_map={{("DAY1", "3"): "L"}}, min_depth_px=50, config=PipelineConfig())
res["decisions_status"] = str(r.get("status"))
meta = out / "metrics_meta.csv"
with open(meta, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["video_path", "day", "mouse", "trial", "Gender", "Genotype"]); w.writerow([str(video), "Day1", "3", 1, "M", "WT"])
from stride.stages.gait_extraction import run_gait_extraction, DEFAULT_PARAMS
g = run_gait_extraction(meta_csv=meta, yml_dir=yml_dir, output_dir=out / "gait", params=dict(DEFAULT_PARAMS), pose_slp_suffix=".slp",
                        column_map={{"mouse_id": ["mouse"], "mouse_id_sep": "", "sex": "Gender"}}, n_workers=1)
res["gait"] = {{k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in (g or {{}}).items()}}
print("CHAIN " + json.dumps(res))
"""
        rc, out, dt = run("chain_next_stages", ["-c", chain])
        line = next((l for l in out.splitlines() if l.startswith("CHAIN ")), None)
        if line:
            r = json.loads(line[6:])
            dec, g = r["decisions_status"], r["gait"]
            accepted = not dec.startswith(("skip", "error")) and not int(g.get("errors", 0) or 0) and not int(g.get("skipped", 0) or 0)
            record("chain_next_stages", "stage 3 (decisions) and stages 4-5 (gait) on the fresh pose file + ROI yml",
                   "the files are read without complaint (a clip this short gives no decision and no strides)", accepted,
                   f"decisions status '{dec}'; gait {g}", dt, rc)
        else:
            record("chain_next_stages", "stage 3 (decisions) and stages 4-5 (gait) on the fresh pose file + ROI yml",
                   "the files are read without complaint", False, f"exit {rc}: {last_error(out)}", dt, rc)

    (ARGS.out / "launch_results.json").write_text(json.dumps(dict(clip=str(ARGS.clip), n_frames=n_frames, fps=fps, device=dev, python=sys.executable,
                                                                  pose_models=pose_m, roi_models=roi_m, results=RESULTS), indent=1))
    n_fail = sum(r["status"] == "FAIL" for r in RESULTS)
    print(f"\n{len(RESULTS) - n_fail} PASS, {n_fail} FAIL  ->  {ARGS.out / 'launch_results.json'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
