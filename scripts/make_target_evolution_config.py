#!/usr/bin/env python3
"""Create the CoEvoP&R-E configuration for one target design.

CoEvoP&R-E uses feedback from the same design on which it is evaluated. The
configuration is derived from the primary configuration, so ChiPBench,
Superblue, and ASAP7 targets run the same method with the same budget.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coevop.eval.exposure import FAMILIES, write_target_evolution_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument(
        "--design",
        required=True,
        help="Target design, e.g. bp_fe (chipbench), 16 (superblue), or ibex (asap7).",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    print(
        write_target_evolution_config(
            family=args.family,
            design=args.design,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
