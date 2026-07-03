"""Template loading for OpenEvolve-style CoEvoP&R prompts.

This mirrors OpenEvolve's file-backed prompt-template structure while keeping
the implementation small and dependency-free for CoEvoP&R.
"""

from __future__ import annotations

from pathlib import Path


class TemplateManager:
    """Load prompt templates from ``coevop/prompts/defaults``.

    A custom directory can override any default template by using the same
    ``*.txt`` filename stem.
    """

    def __init__(self, custom_template_dir: str | Path | None = None) -> None:
        self.default_dir = Path(__file__).parent.parent / "prompts" / "defaults"
        self.custom_dir = Path(custom_template_dir) if custom_template_dir else None
        self.templates: dict[str, str] = {}
        self._load_from_directory(self.default_dir)
        if self.custom_dir is not None:
            self._load_from_directory(self.custom_dir)

    def get_template(self, name: str) -> str:
        if name not in self.templates:
            raise ValueError(f"prompt template '{name}' not found")
        return self.templates[name]

    def _load_from_directory(self, directory: Path) -> None:
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.txt")):
            self.templates[path.stem] = path.read_text(encoding="utf-8")
