#!/usr/bin/env python3
"""Runnability smoke sweep over ``examples/*.py`` (shardable).

Each example is executed as a real subprocess with a wall-clock timeout, from the
repository root (the way a reader of README would run it), with ``PYTHONPATH``
pointing at ``src``. Results are appended as one JSON line per example so the
sweep can be sharded over a cluster without shared state.

Usage
-----
    python3 scripts/smoke_examples.py --shard 1 --shards 8 --timeout 120 \
        --out artifacts/doc_consistency/examples_shard_1.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=1)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    files = sorted(p for p in (ROOT / "examples").rglob("*.py"))
    mine = [p for i, p in enumerate(files) if i % args.shards == args.shard - 1]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), MPLBACKEND="Agg")

    records = []
    for path in mine:
        rel = str(path.relative_to(ROOT))
        t0 = time.time()
        try:
            proc = subprocess.run([sys.executable, str(path)], cwd=str(ROOT), env=env,
                                  capture_output=True, text=True, timeout=args.timeout)
            status = "pass" if proc.returncode == 0 else "fail"
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            status, tail, rc = "timeout", [f"exceeded {args.timeout}s"], None
        rec = {"example": rel, "status": status, "returncode": rc,
               "seconds": round(time.time() - t0, 2), "tail": tail}
        records.append(rec)
        print(f"[shard {args.shard}] {status:7s} {rel} ({rec['seconds']}s)", flush=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    counts: dict[str, int] = {}
    for rec in records:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print(f"[shard {args.shard}] {counts} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())