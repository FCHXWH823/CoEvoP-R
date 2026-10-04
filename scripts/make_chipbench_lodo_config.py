#!/usr/bin/env python3
"""Create a leave-one-design-out CoEvoP&R configuration for ChiPBench.

CoEvoP&R-L excludes the target design from objective evolution and prompt
evidence. The configuration is derived from the primary configuration and
differs from it only in the designs that supply feedback.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from coevop.eval.exposure import FAMILIES, write_lodo_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", required=True, choices=sorted(FAMILIES["chipbench"].designs))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    print(
        write_lodo_config(
            heldout=args.heldout,
            output_dir=args.output_dir,
            repo_root=args.repo_root,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
