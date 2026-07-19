#!/usr/bin/env python3
"""Fail closed on common identity, credential, and artifact-release leaks."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


SKIP_DIRS_ANYWHERE = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
}
SKIP_ROOT_DIRS = {
    "anonymous_release",
    "artifacts",
    "build",
    "data",
    "datasets",
    "dist",
    "private_evidence",
    "runs",
    "tmp",
    "venv",
}
SKIP_FILE_PATTERNS = (
    re.compile(r"(^|\.)local\.(json|toml|ya?ml|sh|cmd)$", re.IGNORECASE),
    re.compile(r"^\.env($|\.)", re.IGNORECASE),
)
BLOCKED_SUFFIXES = {
    ".db",
    ".def",
    ".duckdb",
    ".lef",
    ".log",
    ".odb",
    ".pid",
    ".pyc",
    ".sqlite",
}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".cff",
    ".cmd",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".mk",
    ".patch",
    ".py",
    ".sh",
    ".tcl",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_SOURCE_FILE_BYTES = 5 * 1024 * 1024


def _patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    # Split detector literals so this source file does not trigger itself.
    home_prefix = "/" + "home/"
    unc_prefix = "\\" + "\\" + "wsl"
    return (
        (
            "absolute_linux_home",
            re.compile(re.escape(home_prefix) + r"[A-Za-z0-9_.-]+/"),
        ),
        (
            "absolute_windows_path",
            re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:\\[^\s\"'<>]+"),
        ),
        ("wsl_unc_path", re.compile(re.escape(unc_prefix), re.IGNORECASE)),
        (
            "email_address",
            re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
        ),
        (
            "api_secret",
            re.compile(r"\bsk" + r"-[A-Za-z0-9_-]{16,}\b"),
        ),
        (
            "private_key",
            re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        ),
        (
            "credential_assignment",
            re.compile(
                r"\b(?:API_?KEY|TOKEN|PASSWORD|SECRET)\s*[:=]\s*[\"']?"
                r"(?!\.\.\.|<|\$|\{)[A-Za-z0-9_./+-]{12,}",
                re.IGNORECASE,
            ),
        ),
        (
            "network_address",
            re.compile(
                r"(?<![\d.])(?!(?:0\.0\.0\.0|127\.0\.0\.1)(?![\d.]))"
                r"(?:25[0-5]|2[0-4]\d|1?\d?\d)"
                r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])"
            ),
        ),
    )


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str
    line: int | None
    excerpt: str


def is_release_file(root: Path, path: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    directory_parts = relative.parts[:-1]
    if any(part in SKIP_DIRS_ANYWHERE for part in directory_parts):
        return False
    if any(part.endswith(".egg-info") for part in directory_parts):
        return False
    if directory_parts and directory_parts[0] in SKIP_ROOT_DIRS:
        return False
    if any(pattern.search(relative.name) for pattern in SKIP_FILE_PATTERNS):
        return False
    return path.is_file()


def iter_release_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if is_release_file(root, path))


def audit_files(
    root: Path,
    files: Iterable[Path],
    *,
    deny_tokens: Sequence[str] = (),
) -> list[Finding]:
    root = root.resolve()
    findings: list[Finding] = []
    patterns = _patterns()
    normalized_tokens = tuple(token for token in deny_tokens if token.strip())

    for path in sorted({Path(item).resolve() for item in files}):
        if not is_release_file(root, path):
            continue
        relative = path.relative_to(root).as_posix()
        lower_parts = {part.lower() for part in Path(relative).parts}
        if lower_parts.intersection({"api keys", "api_keys", "secrets"}):
            findings.append(Finding("sensitive_path", relative, None, "<redacted>"))
            continue
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            findings.append(Finding("generated_or_eda_file", relative, None, path.suffix))
            continue
        size = path.stat().st_size
        if size > MAX_SOURCE_FILE_BYTES:
            findings.append(Finding("oversized_source_file", relative, None, str(size)))
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            findings.append(
                Finding("unreviewed_binary_file", relative, None, path.suffix or "<none>")
            )
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            findings.append(Finding("unreadable_text_file", relative, None, "<binary>"))
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            for rule, pattern in patterns:
                if pattern.search(line):
                    excerpt = "<redacted>" if rule in {
                        "api_secret",
                        "credential_assignment",
                        "private_key",
                    } else line.strip()[:240]
                    findings.append(Finding(rule, relative, line_number, excerpt))
            for token in normalized_tokens:
                if token.casefold() in line.casefold():
                    findings.append(Finding("deny_token", relative, line_number, "<redacted>"))
    return findings


def audit_tree(root: Path, *, deny_tokens: Sequence[str] = ()) -> list[Finding]:
    root = root.resolve()
    return audit_files(root, iter_release_files(root), deny_tokens=deny_tokens)


def _write_report(path: Path, root: Path, findings: Sequence[Finding]) -> None:
    payload = {
        "root": str(root),
        "passed": not findings,
        "finding_count": len(findings),
        "findings": [asdict(item) for item in findings],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--deny-token", action="append", default=[])
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    findings = audit_tree(root, deny_tokens=args.deny_token)
    if args.json_output:
        _write_report(args.json_output, root, findings)
    if findings:
        for item in findings:
            location = f"{item.path}:{item.line}" if item.line else item.path
            print(f"{item.rule}: {location}: {item.excerpt}")
        print(f"anonymous artifact audit failed with {len(findings)} finding(s)")
        return 1
    print("anonymous artifact audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
