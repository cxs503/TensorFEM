#!/usr/bin/env python3
"""Advance a recoverable refined-panel peak-window qualification path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorfem.panel_peak_window import execute_panel_peak_window


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--divisions", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--coarse-4x4-manifest", type=Path)
    parser.add_argument("--wall-seconds", type=float)
    args = parser.parse_args()
    coarse = (json.loads(args.coarse_4x4_manifest.read_text())
              if args.coarse_4x4_manifest else None)
    result = execute_panel_peak_window(
        divisions=args.divisions,
        cache_dir=args.cache_dir, target_steps=args.target_steps,
        coarse_4x4_manifest=coarse, maximum_wall_seconds=args.wall_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
