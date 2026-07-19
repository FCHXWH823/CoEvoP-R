#!/usr/bin/env python3
"""Export timing constraints from the exact Table 1 ChiPBench flow runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_sdc(odb: Path, config: dict[str, object]) -> Path | None:
    for name in ("2_1_floorplan.sdc", "2_floorplan.sdc", "1_synth.sdc"):
        candidate = odb.parent / name
        if candidate.is_file():
            return candidate
    configured = Path(os.path.expandvars(str(config.get("sdc_input") or ""))).expanduser()
    return configured if configured.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--design", action="append")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    designs = args.design or [
        "bp_fe",
        "bp_be",
        "swerv_wrapper",
        "ethernet",
        "dft68",
        "or1200",
        "vga_lcd",
        "mor1kx",
    ]
    rows = []
    for design in designs:
        config_path = args.config_root / f"{design}.json"
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        generation = dict(config.get("coevop_chipbench_generation_audit") or {})
        odb = Path(
            os.path.expandvars(str(generation.get("source_odb") or ""))
        ).expanduser().resolve()
        out_dir = args.output_root / design
        out_dir.mkdir(parents=True, exist_ok=True)
        sdc = _source_sdc(odb, config)
        row = {
            "design": design,
            "config": str(config_path.resolve()),
            "source_odb": str(odb),
            "source_sdc": str(sdc) if sdc else None,
            "status": "failed",
        }
        if not odb.is_file() or sdc is None:
            row["failure_reason"] = "missing floorplan ODB or matching floorplan SDC"
            rows.append(row)
            continue
        output_sdc = out_dir / "floorplan.sdc"
        shutil.copy2(sdc, output_sdc)
        row.update(
            {
                "status": "success",
                "output_sdc": str(output_sdc.resolve()),
                "source_odb_sha256": _sha256(odb),
                "source_sdc_sha256": _sha256(sdc),
                "output_sdc_sha256": _sha256(output_sdc),
            }
        )
        rows.append(row)
    summary = {
        "status": "success" if all(row["status"] == "success" for row in rows) else "partial",
        "source_stage": "ChiPBench floorplan constraints",
        "timing_netlist": "reconstructed from each evaluated placement DEF at runtime",
        "rows": rows,
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
