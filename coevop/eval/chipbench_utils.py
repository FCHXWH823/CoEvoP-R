"""Utilities for running ChiPBench in a user-writable workspace."""

from __future__ import annotations

import time
from pathlib import Path


def prepare_chipbench_runtime_dirs(chipbench_root: str | Path) -> list[dict[str, str]]:
    """Ensure ChiPBench flow runtime directories are writable.

    Some local ChiPBench installs contain root-owned flow output directories
    from previous runs. OpenROAD flow writes logs/results/reports under these
    directories, so a non-writable parent causes immediate evaluation failure.
    This helper preserves such directories by moving them aside and recreating
    a writable directory in their place.
    """

    root = Path(chipbench_root)
    repairs = []
    for name in ("def_tmp", "logs", "objects", "results", "reports"):
        path = root / "flow" / name
        if path.exists() and not _is_writable_dir(path):
            backup = _unique_backup_path(path)
            path.rename(backup)
            repairs.append({"path": str(path), "backup": str(backup)})
        path.mkdir(parents=True, exist_ok=True)
    return repairs


def _unique_backup_path(path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.name}.coevop_readonly_backup_{timestamp}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}.coevop_readonly_backup_{timestamp}_{suffix}")
        suffix += 1
    return candidate


def _is_writable_dir(path: Path) -> bool:
    try:
        probe = path / ".coevop_write_probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False
