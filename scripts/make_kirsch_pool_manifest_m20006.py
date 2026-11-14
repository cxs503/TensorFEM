#!/usr/bin/env python3
"""Run manifest for mission 20006 (CASE-FEM-KIRSCH-PLATE, benchmark case pool).

Collects, without hand-copying any number: the environment of the reproduction
run, the two commands that produced the artefacts, the sha256 of every artefact,
and the three pre-checks (knowledge_base capability / negative_knowledge_ledger /
oss_solutions) that this tenant is allowed to read.

Writes runs/m20006_kirsch_plate/manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "m20006_kirsch_plate"
ANCHOR = ROOT / "benchmarks/verified/fem_kirsch_plate"
ASSETS = ROOT / "docs/assets/kirsch_plate"


def sql(q: str) -> str:
    r = subprocess.run(["su", "postgres", "-c", f'psql -d tensorplatform -t -A -F "|" -c "{q}"'],
                       capture_output=True, text=True)
    return r.stdout.strip()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    kb = sql("select id,kb_key,domain,left(title,90) from knowledge_base "
             "where title ilike '%kirsch%' or content ilike '%kirsch%' order by id desc limit 5")
    neg = sql("select count(*) from negative_knowledge_ledger "
              "where title ilike '%kirsch%' or keywords ilike '%kirsch%' "
              "or domain_tags ilike '%TensorFEM%'")
    neg_kirsch = sql("select id,mission_id,left(title,80) from negative_knowledge_ledger "
                     "where title ilike '%kirsch%' or keywords ilike '%kirsch%' limit 5")
    oss = sql("select id,repo,feature_slug,tag from oss_solutions "
              "where repo ilike '%TensorFEM%' or feature_slug ilike '%KIRSCH%' limit 5")
    pool = sql("select id,feature_slug,status,tolerance,reference from tensor_case_pool "
               "where feature_slug ilike '%KIRSCH%' limit 3")
    reg = sql("select count(*) from tensor_benchmark_cases where anchor_slug ilike '%KIRSCH%'")

    files = {}
    for base in (RUNS, ANCHOR, ASSETS):
        for p in sorted(base.glob("*")):
            if p.is_file() and p.suffix in {".json", ".png", ".log"}:
                files[str(p.relative_to(ROOT))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}

    payload = {
        "schema": "tensorfem.run-manifest.v1",
        "mission": 20006,
        "anchor": "CASE-FEM-KIRSCH-PLATE",
        "repo": "TensorFEM",
        "branch": "agent/m20006",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {
            "host": platform.node(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "tensorfem": json.loads((ANCHOR / "result.json").read_text())["environment"],
        },
        "commands": {
            "repro": "OMP_NUM_THREADS=6 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/run.py",
            "probe": "OMP_NUM_THREADS=8 PYTHONPATH=src python "
                     "benchmarks/verified/fem_kirsch_plate/refinement_probe_n96.py  (cluster, cj2429)",
            "tests": "OMP_NUM_THREADS=4 PYTHONPATH=src python -m pytest tests/test_kirsch_plate_anchor.py -q",
        },
        "prechecks": {
            "knowledge_base": {"hits": kb.splitlines() if kb else [],
                               "note": "cap:TensorFEM:m15298_CASE-KIRSCH-PLATE is the L4 capability row for this anchor"},
            "negative_knowledge_ledger_TensorFEM": {"count": neg, "kirsch_hits": neg_kirsch or "none"},
            "oss_solutions_same_domain": {"rows": oss or "no TensorFEM/Kirsch row; nearest rows are other solvers"},
            "case_pool_row": pool or "none",
            "benchmark_case_registry_rows": reg,
        },
        "artifacts": files,
    }
    RUNS.mkdir(parents=True, exist_ok=True)
    out = RUNS / "manifest.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out.relative_to(ROOT)} ({len(files)} artefacts tracked)")
    print(json.dumps(payload["prechecks"], indent=1)[:1500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())