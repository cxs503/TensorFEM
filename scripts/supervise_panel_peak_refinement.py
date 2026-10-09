#!/usr/bin/env python3
"""Attach peak-window refinement to an already running generation supervisor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--refined-maximum-step", type=float, required=True)
    parser.add_argument("--wall-seconds", type=float, default=7200.)
    parser.add_argument("--poll-seconds", type=float, default=30.)
    args = parser.parse_args()
    while not (args.cache_dir/"peak-step-sensitivity.json").exists():
        status_path = args.cache_dir/"generation-status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text())
            if status.get("state") == "post_peak":
                command = [
                    sys.executable,
                    str(Path(__file__).with_name(
                        "run_panel_peak_step_refinement.py")),
                    "--cache-dir", str(args.cache_dir),
                    "--refined-maximum-step", str(args.refined_maximum_step),
                    "--wall-seconds", str(args.wall_seconds),
                ]
                raise SystemExit(subprocess.run(command, check=False).returncode)
            if status.get("state") == "stopped":
                raise SystemExit("source generation stopped before post-peak")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
