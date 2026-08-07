"""Run manifest — a provenance sidecar written after a STRIDE pipeline stage.

Every pipeline stage can drop a small YAML file next to its outputs recording *how*
those outputs were produced: a UTC timestamp, the STRIDE code version and exact git
commit, the SLEAP model(s) used, the parameters, the input/output files (with row
counts), and the runtime environment (Python + key package versions, host, command).

This makes a run reproducible and auditable long after it happened — you can open the
manifest next to a ``decisions.csv`` / ``gait_per_stride.csv`` and know precisely which
code, models, and settings generated it.

Public API::

    from stride.run_manifest import write_run_manifest
    write_run_manifest(output_dir, stage="pose_inference", models=[...], inputs={...},
                       outputs=[...], results={...})

Writing a manifest never raises into the caller: provenance is best-effort and must not
break a completed pipeline run (see :func:`write_run_manifest`).
"""

from __future__ import annotations

import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as _metadata
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml

from stride import __version__

MANIFEST_SCHEMA_VERSION = 1

# The STRIDE source tree, used to resolve the git commit that is running.
_PKG_ROOT = Path(__file__).resolve().parent

# Packages worth pinning in the manifest for reproducibility.
_TRACKED_PACKAGES = (
    "stride",
    "sleap-io",
    "sleap-nn",
    "numpy",
    "pandas",
    "scipy",
    "torch",
    "opencv-python",
    "pyyaml",
    "click",
)


def _git(repo: Path, *args: str) -> Optional[str]:
    """Run a git command in ``repo``; return stripped stdout or None on any failure."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_info(repo: Optional[Path] = None) -> dict:
    """Git provenance for the running STRIDE checkout (commit, branch, dirty, remote)."""
    repo = repo or _PKG_ROOT
    commit = _git(repo, "rev-parse", "HEAD")
    if not commit:
        return {"available": False}
    status = _git(repo, "status", "--porcelain")
    return {
        "available": True,
        "commit": commit,
        "commit_short": commit[:12],
        "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
        "describe": _git(repo, "describe", "--tags", "--always", "--dirty"),
        "remote": _git(repo, "config", "--get", "remote.origin.url"),
        "last_commit_date": _git(repo, "log", "-1", "--format=%cI"),
        "dirty": bool(status),
    }


def _package_versions() -> dict:
    versions = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = _metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def _current_user() -> Optional[str]:
    try:
        import getpass
        return getpass.getuser()
    except Exception:
        return None


def environment_info() -> dict:
    """Runtime environment: Python, platform, host, user, tracked package versions."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": _current_user(),
        "packages": _package_versions(),
    }


def _safe_exists(p: Path) -> bool:
    """``Path.exists()`` that returns False instead of raising on a stat-denied path.

    On Python 3.12+ ``exists()`` re-raises ``PermissionError``/``OSError`` from ``stat``
    (e.g. a model under a mount the process cannot traverse). A bare call would abort
    manifest assembly and silently drop the whole file — the exact failure this feature
    exists to prevent.
    """
    try:
        return p.exists()
    except OSError:
        return False


def _file_summary(path: Any) -> dict:
    """Existence / size / (for CSV) row count for one output path."""
    p = Path(path)
    info: dict = {"path": str(p), "exists": _safe_exists(p)}
    if not info["exists"]:
        return info
    try:
        st = p.stat()
        info["size_bytes"] = st.st_size
        info["modified_utc"] = datetime.fromtimestamp(
            st.st_mtime, timezone.utc
        ).isoformat(timespec="seconds")
        if p.suffix.lower() == ".csv":
            with open(p, "r", newline="") as f:
                n = sum(1 for _ in f)
            info["rows"] = max(0, n - 1)  # minus header
    except OSError:
        pass
    return info


def _model_summary(path: Any) -> dict:
    """Path + fingerprint (size, mtime) for one model file/dir — a cheap version stamp."""
    p = Path(path)
    info: dict = {"path": str(p), "exists": _safe_exists(p)}
    if info["exists"]:
        try:
            st = p.stat()
            info["size_bytes"] = st.st_size
            info["modified_utc"] = datetime.fromtimestamp(
                st.st_mtime, timezone.utc
            ).isoformat(timespec="seconds")
        except OSError:
            pass
    return info


def _normalize(value: Any) -> Any:
    """Make values YAML-safe: Paths/containers handled; anything exotic -> str.

    The final ``str()`` fallback guarantees ``yaml.safe_dump`` never chokes on an
    unexpected value (numpy scalar, set, datetime, custom object) and silently drops the
    manifest. ``bool`` is a subclass of ``int`` so it is preserved as a native scalar.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    return str(value)  # numpy scalars, sets, datetimes, arbitrary objects


def collect_manifest(
    *,
    stage: str,
    models: Optional[Iterable[Any]] = None,
    params: Optional[dict] = None,
    inputs: Optional[dict] = None,
    outputs: Optional[Iterable[Any]] = None,
    results: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble the manifest dict (does not write anything)."""
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stride_version": __version__,
        "git": git_info(),
        "environment": environment_info(),
        "command": " ".join(sys.argv),
        "models": [_model_summary(m) for m in models] if models else None,
        "params": _normalize(params) if params else None,
        "inputs": _normalize(inputs) if inputs else None,
        "outputs": [_file_summary(o) for o in outputs] if outputs else None,
        "results": _normalize(results) if results else None,
        "extra": _normalize(extra) if extra else None,
    }


def write_run_manifest(
    output_dir: Any,
    stage: str,
    *,
    filename: Optional[str] = None,
    **kwargs: Any,
) -> Optional[Path]:
    """Write ``<output_dir>/<stage>_run_manifest.yaml`` and return its path.

    Best-effort: any failure is swallowed and returns None, so a provenance write can
    never break an otherwise-successful pipeline run. Extra keyword arguments
    (``models``, ``params``, ``inputs``, ``outputs``, ``results``, ``extra``) are passed
    through to :func:`collect_manifest`.
    """
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        manifest = collect_manifest(stage=stage, **kwargs)
        path = out / (filename or f"{stage}_run_manifest.yaml")
        with open(path, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False, default_flow_style=False)
        return path
    except Exception as exc:  # best-effort: never break a completed run...
        try:  # ...but surface the drop instead of failing completely silently.
            print(
                f"[stride] WARNING: run manifest not written for stage '{stage}' "
                f"({type(exc).__name__}: {exc})",
                file=sys.stderr,
            )
        except Exception:
            pass
        return None
