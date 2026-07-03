"""CircuitNet N14 manifest discovery.

The v0 platform starts from already-downloaded CircuitNet files and records where
feature/label maps live. Heavy data stays outside Git and outside this package.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


MANIFEST_VERSION = 1

ROUTABILITY_FEATURE_DIRS = {
    "rudy": "RUDY/RUDY",
    "rudy_short": "RUDY/RUDY_short",
    "rudy_long": "RUDY/RUDY_long",
    "rudy_pin": "RUDY/RUDY_pin",
    "rudy_pin_long": "RUDY/RUDY_pin_long",
    "cell_density": "cell_density",
    "macro_region": "macro_region",
}

ROUTABILITY_LABEL_DIRS = {
    "drc_all": "DRC/DRC_all",
    "congestion_egr_horizontal_overflow": (
        "congestion/congestion_early_global_routing/overflow_based/"
        "congestion_eGR_horizontal_overflow"
    ),
    "congestion_egr_vertical_overflow": (
        "congestion/congestion_early_global_routing/overflow_based/"
        "congestion_eGR_vertical_overflow"
    ),
    "congestion_gr_horizontal_overflow": (
        "congestion/congestion_global_routing/overflow_based/"
        "congestion_GR_horizontal_overflow"
    ),
    "congestion_gr_vertical_overflow": (
        "congestion/congestion_global_routing/overflow_based/"
        "congestion_GR_vertical_overflow"
    ),
    "congestion_egr_horizontal_util": (
        "congestion/congestion_early_global_routing/utilization_based/"
        "congestion_eGR_horizontal_util"
    ),
    "congestion_egr_vertical_util": (
        "congestion/congestion_early_global_routing/utilization_based/"
        "congestion_eGR_vertical_util"
    ),
    "congestion_gr_horizontal_util": (
        "congestion/congestion_global_routing/utilization_based/"
        "congestion_GR_horizontal_util"
    ),
    "congestion_gr_vertical_util": (
        "congestion/congestion_global_routing/utilization_based/"
        "congestion_GR_vertical_util"
    ),
}


@dataclass(frozen=True)
class CircuitNetSample:
    sample_id: str
    design: str
    features: dict[str, str]
    labels: dict[str, str]


@dataclass(frozen=True)
class CircuitNetDesign:
    name: str
    family: str
    sample_count: int
    samples: list[CircuitNetSample]


@dataclass(frozen=True)
class CircuitNetManifest:
    version: int
    generated_at: str
    circuitnet_root: str
    n14_root: str
    routability_root: str
    designs: list[CircuitNetDesign]


def resolve_n14_root(root: str | Path) -> Path:
    """Accept either the Hugging Face repo root or the CircuitNet-N14 directory."""

    path = Path(root).expanduser().resolve()
    if (path / "routability_features").is_dir():
        return path
    if (path / "CircuitNet-N14" / "routability_features").is_dir():
        return path / "CircuitNet-N14"
    raise FileNotFoundError(
        "Could not find CircuitNet-N14/routability_features under "
        f"{path}. Set CIRCUITNET_ROOT to the dataset root or CircuitNet-N14 directory."
    )


def discover_routability_designs(n14_root: Path) -> list[Path]:
    routability_root = n14_root / "routability_features"
    designs = [
        child
        for child in routability_root.iterdir()
        if child.is_dir() and not child.name.startswith(".")
    ]
    return sorted(designs, key=lambda p: p.name)


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _npz_by_stem(root: Path, relative_dir: str) -> dict[str, Path]:
    directory = root / relative_dir
    if not directory.is_dir():
        return {}
    return {path.stem: path for path in sorted(directory.glob("*.npz"))}


def _family_from_design(design_name: str) -> str:
    for suffix in ("-small", "-large"):
        if design_name.endswith(suffix):
            return design_name[: -len(suffix)]
    return design_name.split("_", maxsplit=1)[0]


def _build_design_manifest(design_root: Path, n14_root: Path) -> CircuitNetDesign:
    features_by_name = {
        name: _npz_by_stem(design_root, relative_dir)
        for name, relative_dir in ROUTABILITY_FEATURE_DIRS.items()
    }
    labels_by_name = {
        name: _npz_by_stem(design_root, relative_dir)
        for name, relative_dir in ROUTABILITY_LABEL_DIRS.items()
    }

    sample_ids = set()
    for files in features_by_name.values():
        sample_ids.update(files)
    for files in labels_by_name.values():
        sample_ids.update(files)

    samples: list[CircuitNetSample] = []
    for sample_id in sorted(sample_ids):
        features = {
            name: _relative(files[sample_id], n14_root)
            for name, files in features_by_name.items()
            if sample_id in files
        }
        labels = {
            name: _relative(files[sample_id], n14_root)
            for name, files in labels_by_name.items()
            if sample_id in files
        }
        if features or labels:
            samples.append(
                CircuitNetSample(
                    sample_id=sample_id,
                    design=design_root.name,
                    features=features,
                    labels=labels,
                )
            )

    return CircuitNetDesign(
        name=design_root.name,
        family=_family_from_design(design_root.name),
        sample_count=len(samples),
        samples=samples,
    )


def build_manifest(root: str | Path, designs: Iterable[str] | None = None) -> CircuitNetManifest:
    n14_root = resolve_n14_root(root)
    requested = set(designs or [])
    design_roots = discover_routability_designs(n14_root)
    if requested:
        design_roots = [path for path in design_roots if path.name in requested]
        missing = requested.difference(path.name for path in design_roots)
        if missing:
            raise FileNotFoundError(f"Requested CircuitNet design(s) not found: {sorted(missing)}")

    design_manifests = [_build_design_manifest(path, n14_root) for path in design_roots]
    circuitnet_root = n14_root.parent if n14_root.name == "CircuitNet-N14" else n14_root
    return CircuitNetManifest(
        version=MANIFEST_VERSION,
        generated_at=datetime.now(timezone.utc).isoformat(),
        circuitnet_root=str(circuitnet_root),
        n14_root=str(n14_root),
        routability_root=str(n14_root / "routability_features"),
        designs=design_manifests,
    )


def manifest_to_dict(manifest: CircuitNetManifest) -> dict:
    return asdict(manifest)


def manifest_from_dict(payload: dict) -> CircuitNetManifest:
    designs = []
    for design_payload in payload.get("designs", []):
        samples = [
            CircuitNetSample(
                sample_id=sample["sample_id"],
                design=sample["design"],
                features=dict(sample.get("features", {})),
                labels=dict(sample.get("labels", {})),
            )
            for sample in design_payload.get("samples", [])
        ]
        designs.append(
            CircuitNetDesign(
                name=design_payload["name"],
                family=design_payload["family"],
                sample_count=int(design_payload["sample_count"]),
                samples=samples,
            )
        )

    return CircuitNetManifest(
        version=int(payload["version"]),
        generated_at=payload["generated_at"],
        circuitnet_root=payload["circuitnet_root"],
        n14_root=payload["n14_root"],
        routability_root=payload["routability_root"],
        designs=designs,
    )


def load_manifest(path: str | Path) -> CircuitNetManifest:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as f:
        return manifest_from_dict(json.load(f))


def write_manifest(manifest: CircuitNetManifest, output: str | Path, pretty: bool = True) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if pretty else None
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(manifest_to_dict(manifest), f, indent=indent, sort_keys=True)
        f.write("\n")
    return output_path
