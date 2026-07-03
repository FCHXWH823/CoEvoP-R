"""Wait for an OpenEvolve process to finish, then compare two run directories."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=600)
    args = parser.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a", encoding="utf-8") as log:
        log.write(f"watching pid={args.pid}\n")
        log.flush()
        while _pid_alive(args.pid):
            log.write(f"pid={args.pid} still running\n")
            log.flush()
            time.sleep(max(1, args.poll_seconds))
        log.write(f"pid={args.pid} exited; generating comparison\n")
        log.flush()
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("compare_openevolve_runs.py")),
            "--previous",
            str(args.previous),
            "--current",
            str(args.current),
            "--json-output",
            str(args.json_output),
            "--markdown-output",
            str(args.markdown_output),
        ]
        result = subprocess.run(cmd, stdout=log, stderr=log)
        log.write(f"comparison command exited rc={result.returncode}\n")
        return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
