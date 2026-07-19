from __future__ import annotations

import importlib.util
import sys
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_anonymity_audit_detects_identity_paths_and_secrets(tmp_path: Path) -> None:
    audit = _load_script("audit_anonymous_artifact")
    source = tmp_path / "example.txt"
    source.write_text(
        "root=" + "/" + "home/" + "private_user/project\n"
        + "key=" + "sk-" + "x" * 32 + "\n",
        encoding="utf-8",
    )

    findings = audit.audit_files(tmp_path, [source])

    assert {item.rule for item in findings} == {"absolute_linux_home", "api_secret"}


def test_anonymity_audit_skips_local_and_generated_directories(tmp_path: Path) -> None:
    audit = _load_script("audit_anonymous_artifact")
    local = tmp_path / "config.local.toml"
    local.write_text("private", encoding="utf-8")
    run_file = tmp_path / "runs" / "private.log"
    run_file.parent.mkdir()
    run_file.write_text("private", encoding="utf-8")

    assert not audit.is_release_file(tmp_path, local)
    assert not audit.is_release_file(tmp_path, run_file)


def test_anonymous_builder_excludes_git_runs_and_local_configs(tmp_path: Path) -> None:
    builder = _load_script("build_anonymous_artifact")
    (tmp_path / "README.md").write_text("anonymous\n", encoding="utf-8")
    (tmp_path / "LICENSE").write_text("test\n", encoding="utf-8")
    package = tmp_path / "coevop"
    package.mkdir()
    (package / "__init__.py").write_text("\n", encoding="utf-8")
    package_data = package / "datasets"
    package_data.mkdir()
    (package_data / "loader.py").write_text("\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("identity\n", encoding="utf-8")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "result.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "private.local.toml").write_text("secret\n", encoding="utf-8")
    output = tmp_path / "dist" / "anonymous.zip"

    result = builder.build_archive(tmp_path, output)

    assert result["file_count"] == 4
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    assert any(name.endswith("/README.md") for name in names)
    assert any(name.endswith("/coevop/datasets/loader.py") for name in names)
    assert any(name.endswith("/ANONYMOUS_ARTIFACT_MANIFEST.json") for name in names)
    assert not any(".git" in name or "/runs/" in name or ".local." in name for name in names)
