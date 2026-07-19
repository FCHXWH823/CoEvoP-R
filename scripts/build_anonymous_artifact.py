#!/usr/bin/env python3
"""Build a deterministic, history-free CoEvoP&R source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import zipfile
from pathlib import Path
from typing import Sequence

try:
    from scripts.audit_anonymous_artifact import audit_files, is_release_file
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from audit_anonymous_artifact import audit_files, is_release_file


ARCHIVE_ROOT = "CoEvoPR-Platform-anonymous"
INCLUDE_FILES = {
    ".gitattributes",
    ".gitignore",
    "ARTIFACT_EVALUATION.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
}
INCLUDE_DIRS = {
    "coevop",
    "configs",
    "docs",
    "patches",
    "prompts",
    "scripts",
    "tests",
}


def collect_files(root: Path) -> list[Path]:
    root = root.resolve()
    selected: list[Path] = []
    for name in sorted(INCLUDE_FILES):
        path = root / name
        if path.is_file() and is_release_file(root, path):
            selected.append(path)
    for name in sorted(INCLUDE_DIRS):
        directory = root / name
        if not directory.is_dir():
            continue
        selected.extend(
            path for path in sorted(directory.rglob("*")) if is_release_file(root, path)
        )
    return sorted(set(selected), key=lambda item: item.relative_to(root).as_posix())


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _zip_info(name: str, mode: int = 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | mode) << 16
    return info


def build_archive(
    root: Path,
    output: Path,
    *,
    deny_tokens: Sequence[str] = (),
) -> dict[str, object]:
    root = root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing archive: {output}")
    if root == output.parent or root in output.parents:
        # Output under ignored dist/ is allowed; source files are collected first
        # and the archive itself is never part of the include set.
        pass

    files = collect_files(root)
    if not files:
        raise ValueError(f"no release files found under {root}")
    findings = audit_files(root, files, deny_tokens=deny_tokens)
    if findings:
        details = "; ".join(f"{item.rule}:{item.path}:{item.line}" for item in findings[:10])
        raise ValueError(f"anonymous artifact audit failed: {details}")

    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_digest(path),
            "bytes": path.stat().st_size,
        }
        for path in files
    ]
    manifest = {
        "schema_version": 1,
        "archive_root": ARCHIVE_ROOT,
        "file_count": len(entries),
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            mode = 0o755 if path.suffix == ".sh" or path.name.endswith(".py") else 0o644
            archive.writestr(_zip_info(f"{ARCHIVE_ROOT}/{relative}", mode), path.read_bytes())
        archive.writestr(
            _zip_info(f"{ARCHIVE_ROOT}/ANONYMOUS_ARTIFACT_MANIFEST.json"),
            manifest_bytes,
        )

    return {
        "output": str(output),
        "sha256": _file_digest(output),
        "file_count": len(entries),
        "bytes": output.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deny-token", action="append", default=[])
    args = parser.parse_args()

    result = build_archive(args.root, args.output, deny_tokens=args.deny_token)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
