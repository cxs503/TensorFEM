#!/usr/bin/env python3
"""Insert the m20108 board-audit verdict for m20054 (CASE-FEM-HEAT-L2).

Board rule: the verdict must be reproducible from the JSON next to this file, so
the SQL is generated from `verdict_evidence_m20108.json` (no hand-typed numbers).
Connects as postgres over local trust, which is the established board-audit path.

Usage: python3 docs/board_audit/m20108/insert_verdict_m20108.py [--dry-run]
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = json.loads((HERE / "verdict_evidence_m20108.json").read_text(encoding="utf-8"))

REPO = "TensorFEM"
SCOPE = "m20054 CASE-FEM-HEAT-L2"
MISSION_ID = EVIDENCE["audit_mission"]
LEVEL = EVIDENCE["maturity_level"]
VERDICT_BY = EVIDENCE["verdict_by"]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_sql() -> str:
    payload = json.dumps({k: v for k, v in EVIDENCE.items()
                          if k not in ("audit_mission", "maturity_level", "verdict_by")},
                         ensure_ascii=False)
    return (
        "INSERT INTO tensor_maturity_verdicts "
        "(repo, scope, mission_id, level, evidence, verdict_by) VALUES ("
        f"{sql_literal(REPO)}, {sql_literal(SCOPE)}, {MISSION_ID}, {LEVEL}, "
        f"{sql_literal(payload)}::jsonb, {sql_literal(VERDICT_BY)}) "
        "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
        "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, "
        "evidence = EXCLUDED.evidence;"
    )


def psql(sql: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
         "-v", "ON_ERROR_STOP=1", "-c", sql],
        capture_output=True, text=True, timeout=60,
    )


def main() -> int:
    sql = build_sql()
    if "--dry-run" in sys.argv:
        print(sql[:400] + " ...")
        return 0
    r = psql(sql)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        return 1
    verify = psql(
        "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state, "
        "jsonb_array_length(evidence->'impl_files') AS impl_files, "
        "jsonb_array_length(evidence->'run_artifacts') AS run_artifacts, "
        "(evidence ? 'benchmark') AS has_benchmark, "
        "(evidence ? 'repro_cmd') AS has_repro_cmd, "
        "(evidence ? 'external_anchor') AS has_external_anchor "
        f"FROM tensor_maturity_verdicts WHERE mission_id={MISSION_ID} "
        f"AND verdict_by='{VERDICT_BY}';"
    )
    print(verify.stdout or verify.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())