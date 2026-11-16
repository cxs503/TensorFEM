#!/usr/bin/env python
"""Merge single-mesh shards of the GAP-PP-3 3-grid matrix into one data set.

The matrix is sharded because one mesh takes minutes on the cluster
(``gap_pp3_3grid_study.py --grids 24x12``); the criteria block needs all meshes
together, and it must be computed from the *merged* raw entries so that no
number is hand-copied between files.

Usage
-----
    python scripts/gap_pp3_3grid_merge.py \
        outputs/gap_pp3_3grid_8x4.json outputs/gap_pp3_3grid_16x8.json \
        outputs/gap_pp3_3grid_24x12.json \
        --out outputs/gap_pp3_3grid_results.json

Shards that are missing are reported and simply omitted (the criteria block
then covers fewer meshes and says so through ``criteria.mesh_order``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from gap_pp3_3grid_study import OUT, finalize, headers, write  # noqa: E402


def load_shards(paths: list[str]) -> tuple[dict, list[str], list[str]]:
    merged = headers()
    used, missing = [], []
    for path in paths:
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as handle:
            shard = json.load(handle)
        for name, entry in shard.get("meshes", {}).items():
            merged["meshes"][name] = entry
        for key in ("case", "material", "analytic", "load_ladder_Pa"):
            if key in shard:
                merged[key] = shard[key]
        used.append(os.path.basename(path))
    return merged, used, missing


def main(paths: list[str], out: str = OUT) -> dict:
    merged, used, missing = load_shards(paths)
    if not merged["meshes"]:
        raise SystemExit(f"no shard contained meshes (missing: {missing})")
    finalize(merged)
    merged["shards"] = {"used": used, "missing": missing}
    write(merged, out)
    print(f"merged {used} / missing {missing}")
    print(json.dumps(merged["criteria"], indent=2)[:2000])
    return merged


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("shards", nargs="+", help="per-mesh JSON files")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    main(args.shards, args.out)