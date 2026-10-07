#!/usr/bin/env python3
"""Build hashed 4/8/12 readiness inputs and strict mesh convergence evidence."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from tensorfem.panel_generation_readiness import (
    aggregate_panel_mesh_readiness, panel_generation_readiness,
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--four-cache", type=Path, required=True)
    parser.add_argument("--eight-cache", type=Path, required=True)
    parser.add_argument("--twelve-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    four = panel_generation_readiness(args.four_cache)
    eight = panel_generation_readiness(args.eight_cache)
    twelve = panel_generation_readiness(args.twelve_cache)
    atomic_json(args.output.parent/"panel-readiness-4x4.json", four)
    atomic_json(args.output.parent/"panel-readiness-8x8.json", eight)
    atomic_json(args.output.parent/"panel-readiness-12x12.json", twelve)
    report = aggregate_panel_mesh_readiness({4: four, 8: eight, 12: twelve})
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
