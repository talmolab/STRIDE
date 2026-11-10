"""The lab tests need the lab package (`pip install -e lab/`); without it they are left out, not failed."""
try:
    import stride_lab  # noqa: F401
except ImportError:
    collect_ignore_glob = ["test_*.py"]
