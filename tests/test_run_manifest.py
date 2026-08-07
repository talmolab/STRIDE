"""Tests for the run-manifest provenance writer (stride.run_manifest)."""

import yaml

from stride.run_manifest import (
    MANIFEST_SCHEMA_VERSION,
    collect_manifest,
    write_run_manifest,
)


def test_collect_manifest_has_core_fields():
    m = collect_manifest(
        stage="decisions",
        params={"n_workers": 4},
        inputs={"video_dir": "/v"},
        results={"ok": 10, "total": 12},
    )
    assert m["stage"] == "decisions"
    assert m["manifest_schema_version"] == MANIFEST_SCHEMA_VERSION
    assert m["stride_version"]
    # git block is always present (either populated or {"available": False})
    assert "available" in m["git"] or "commit" in m["git"]
    assert m["environment"]["python"]
    assert "packages" in m["environment"]
    assert m["timestamp_utc"].endswith("+00:00")  # UTC


def test_write_run_manifest_writes_valid_yaml(tmp_path):
    out = write_run_manifest(
        tmp_path, "gait_filtering",
        params={"confidence_threshold": 0.3},
        results={"original": 100, "filtered": 60},
    )
    assert out is not None and out.exists()
    assert out.name == "gait_filtering_run_manifest.yaml"
    loaded = yaml.safe_load(out.read_text())
    assert loaded["stage"] == "gait_filtering"
    assert loaded["params"]["confidence_threshold"] == 0.3
    assert loaded["results"]["filtered"] == 60


def test_output_file_summary_counts_csv_rows(tmp_path):
    csv = tmp_path / "decisions.csv"
    csv.write_text("day,mouse,choice\n1,M1-1,L\n1,M1-2,R\n")
    out = write_run_manifest(tmp_path, "decisions", outputs=[csv])
    loaded = yaml.safe_load(out.read_text())
    summ = loaded["outputs"][0]
    assert summ["exists"] is True
    assert summ["rows"] == 2  # data rows, excluding header


def test_paths_are_normalized_to_strings(tmp_path):
    out = write_run_manifest(tmp_path, "decisions", inputs={"video_dir": tmp_path})
    loaded = yaml.safe_load(out.read_text())
    assert loaded["inputs"]["video_dir"] == str(tmp_path)


def test_exotic_values_are_coerced_and_manifest_still_written(tmp_path):
    # An otherwise-unserializable object must be coerced to str, not silently drop the
    # whole manifest (regression guard for the yaml.safe_dump swallow path).
    class Weird:
        def __repr__(self):
            return "WEIRD_OBJ"

    out = write_run_manifest(tmp_path, "decisions", extra={"obj": Weird()})
    assert out is not None and out.exists()
    loaded = yaml.safe_load(out.read_text())
    assert "WEIRD_OBJ" in str(loaded["extra"]["obj"])


def test_missing_model_and_output_paths_do_not_break_manifest(tmp_path):
    # Non-existent model/output paths (incl. under unreadable parents) must yield
    # exists: false and still produce a manifest — not raise or drop it.
    out = write_run_manifest(
        tmp_path, "pose_inference",
        models=[tmp_path / "nope" / "model.ckpt"],
        outputs=[tmp_path / "missing.csv"],
    )
    assert out is not None and out.exists()
    loaded = yaml.safe_load(out.read_text())
    assert loaded["models"][0]["exists"] is False
    assert loaded["outputs"][0]["exists"] is False
