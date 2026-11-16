#!/usr/bin/env python3
"""Register the judge rows for JOUR-DIFFLBM-L3-INVERSION (FEM) in the platform.

Why
---
``benchmarks/body_force_inversion_benchmark.py`` produces ``ref``/``got``/``tolerance``
rows as JSON.  A JSON file is evidence for a reader, but the maturity gate can
only adjudicate what is registered, so each row is inserted into
``tensor_benchmark_cases`` with the reference, the measured value, the
tolerance and the verdict written into the row text as well.

No number is hand-typed: everything is read from the committed evidence blob
``artifacts/body_force_inversion/benchmark_rows.json``.  Rows whose *claim* is a
self-flagging control (status ``flagged``) are registered as ``passed`` because
the benchmark of record is "does it flag itself"; the one honest FAIL (1 %
sensor noise) is registered as ``failed`` on purpose -- deleting it would make
the suite advertise more than it measured.

Usage
-----
``python3 benchmarks/register_body_force_inversion_case.py [--dry-run]``
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

REPO = "TensorFEM"
ANCHOR = "JOUR-DIFFLBM-L3-INVERSION"
EVIDENCE = pathlib.Path(__file__).resolve().parents[1] / "artifacts" / "body_force_inversion" / "benchmark_rows.json"


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_rows() -> list[dict]:
    payload = json.loads(EVIDENCE.read_text())
    if payload["anchor_slug"] != ANCHOR or payload["repo"] != REPO:
        raise RuntimeError("evidence blob is for a different anchor/repo")
    return payload["rows"]


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _got_text(row: dict) -> str:
    got = row["got"]
    parts = [f"{key}={_fmt(value)}" for key, value in got.items() if key != "states"]
    return "; ".join(parts)


def _ref_text(row: dict) -> str:
    ref = row["ref"]
    return "; ".join(f"{key}={_fmt(value)}" for key, value in ref.items())


def build_sql(rows: list[dict]) -> str:
    statements = []
    for row in rows:
        status = "failed" if row["status"] == "failed" else "passed"
        case_name = row["case_name"]
        ref_source = (
            f"{row['ref_source']} | reference value(s): {_ref_text(row)} | "
            f"repro: {row['repro_cmd']}"
        )
        metric = f"{row['metric']} | measured: {_got_text(row)}"
        if row.get("note"):
            metric += f" | note: {row['note']}"
        statements.append(
            "INSERT INTO tensor_benchmark_cases "
            "(anchor_slug, case_name, ref_source, metric, tolerance, repo, status) "
            "SELECT "
            f"{sql_literal(ANCHOR)}, {sql_literal(case_name)}, {sql_literal(ref_source)}, "
            f"{sql_literal(metric)}, {sql_literal(row['tolerance'])}, {sql_literal(REPO)}, "
            f"{sql_literal(status)} "
            "WHERE NOT EXISTS ("
            "  SELECT 1 FROM tensor_benchmark_cases "
            f"  WHERE anchor_slug={sql_literal(ANCHOR)} AND case_name={sql_literal(case_name)}"
            ");"
        )
    return "\n".join(statements)


def psql(sql: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
         "-v", "ON_ERROR_STOP=1", "-c", sql],
        capture_output=True, text=True, timeout=120,
    )


def main() -> int:
    rows = load_rows()
    sql = build_sql(rows)
    if "--dry-run" in sys.argv:
        print(sql)
        return 0
    result = psql(sql)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode
    check = psql(
        "SELECT id, status, left(case_name, 60) AS case_name FROM tensor_benchmark_cases "
        f"WHERE anchor_slug='{ANCHOR}' ORDER BY id"
    )
    print(check.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())