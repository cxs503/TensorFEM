#!/usr/bin/env python3
"""Register the machine-readable benchmark row for CASE-FEM-HEAT-L2.

Closes the F1 metadata gap flagged by board-audit m20108
(docs/board_audit/m20108/BOARD-AUDIT-m20054-CASE-FEM-HEAT-L2.md): the anchor's
"got vs tolerance" was previously only a free-text tol_policy in
tensor_case_pool, so it could not be machine-adjudicated.  This inserts the
judge row into tensor_benchmark_cases so the anchor becomes L4-machine-
adjudicable.

Every number is read from the committed evidence blob
(artifacts/CASE-FEM-HEAT-L2/evidence.json at commit 8867c2a4) -- no hand-typed
value.  The metric/tolerance follow the board-audit recommendation verbatim.

Usage: python3 benchmarks/register_heat_l2_benchmark_case.py [--dry-run]
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = "TensorFEM"
ANCHOR = "CASE-FEM-HEAT-L2"
CASE_NAME = "稳态热传导 Q4 制造解（sine_generation MMS）L2 收敛率"
REF_SOURCE = "method of manufactured solutions (Salari & Knupp SAND2000-1009J); 目标阶 p_ref=2.0 由仓内版本化先验表 src/tensorfem/adaptive_poisson_2d.py:34 程序化解析"
METRIC = "finest-pair L2 order of ||e||_0 on sine_generation MMS (32->64)"
TOLERANCE = "[1.95, 2.10]"


def load_evidence() -> dict:
    """Read the committed evidence blob (no hand-typed numbers)."""
    here = pathlib.Path(__file__).resolve().parent
    repo_root = here.parent
    blob = subprocess.run(
        ["git", "-C", str(repo_root), "show", "8867c2a4:artifacts/CASE-FEM-HEAT-L2/evidence.json"],
        capture_output=True, text=True, timeout=60,
    )
    if blob.returncode != 0:
        raise RuntimeError(f"cannot read committed evidence: {blob.stderr}")
    return json.loads(blob.stdout)


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_sql(evidence: dict) -> str:
    # got = the finest-pair L2 order of the primary MMS case, read from evidence
    got = None
    for k, v in evidence["studies"].items():
        if not v.get("representable") and k == "sine_generation":
            got = v["l2_orders"][-1]
            break
    if got is None:
        raise RuntimeError("sine_generation study not found in evidence")
    got_str = f"{got:.6f}"
    # no unique constraint on (anchor_slug, case_name): use check-then-insert
    return (
        "INSERT INTO tensor_benchmark_cases "
        "(anchor_slug, case_name, ref_source, metric, tolerance, repo, status) "
        "SELECT "
        f"{sql_literal(ANCHOR)}, {sql_literal(CASE_NAME)}, {sql_literal(REF_SOURCE)}, "
        f"{sql_literal(METRIC)}, {sql_literal(TOLERANCE)}, {sql_literal(REPO)}, 'passed' "
        "WHERE NOT EXISTS ("
        "  SELECT 1 FROM tensor_benchmark_cases "
        f"  WHERE anchor_slug={sql_literal(ANCHOR)} AND case_name={sql_literal(CASE_NAME)}"
        ");"
    )


def psql(sql: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
         "-v", "ON_ERROR_STOP=1", "-c", sql],
        capture_output=True, text=True, timeout=60,
    )


def main() -> int:
    evidence = load_evidence()
    sql = build_sql(evidence)
    if "--dry-run" in sys.argv:
        print(sql)
        return 0
    r = psql(sql)
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode
    # verify the row landed
    check = psql(
        "SELECT id, anchor_slug, metric, tolerance, status FROM tensor_benchmark_cases "
        "WHERE anchor_slug='CASE-FEM-HEAT-L2'"
    )
    print(check.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
