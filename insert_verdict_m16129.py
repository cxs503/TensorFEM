#!/usr/bin/env python3
"""Insert m16067 board-audit verdict into tensor_maturity_verdicts.

Usage: PGPASSWORD=<pass> python3 insert_verdict_m16129.py

This script attempts to INSERT the board-audit verdict for mission m16067
(CASE-FEM-SCORDELIS-LO-ROOF) into the tensor_maturity_verdicts table.

If direct DB access is unavailable, the verdict is recorded in:
- /root/tp-agent/work/m16129/audit_report.md (full audit report)
- /root/tp-agent/work/m16129/verdict_evidence.json (evidence JSON)
"""
import json
import os
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = json.loads((HERE / "verdict_evidence.json").read_text())

SQL = """
INSERT INTO tensor_maturity_verdicts
    (repo, scope, mission_id, level, evidence, verdict_by)
VALUES
    ('TensorFEM', 'm16067 CASE-FEM-SCORDELIS-LO-ROOF', 16129, 3, %s, 'board-audit');
"""

def main():
    evidence_json = json.dumps(EVIDENCE, ensure_ascii=False)

    # Try psql
    pgpass = os.environ.get("PGPASSWORD", "")
    pguser = os.environ.get("PGUSER", "postgres")
    pgdb = os.environ.get("TP_PG_DB", "tensorplatform")

    if pgpass:
        env = os.environ.copy()
        env["PGPASSWORD"] = pgpass
        try:
            result = subprocess.run(
                ["psql", "-h", "127.0.0.1", "-U", pguser, "-d", pgdb,
                 "-c", SQL, "-v", f"evidence={evidence_json}"],
                capture_output=True, text=True, env=env, timeout=10
            )
            if result.returncode == 0:
                print("Verdict inserted successfully via psql.")
                print(result.stdout)
                return
            else:
                print(f"psql failed (rc={result.returncode}):")
                print(result.stderr)
        except Exception as e:
            print(f"psql error: {e}")

    # Fallback: print SQL for manual execution
    print("=" * 60)
    print("SQL for manual execution:")
    print("=" * 60)
    print(SQL.replace("%s", f"'{evidence_json}'"))
    print("=" * 60)
    print("Verdict summary:")
    print(f"  repo: TensorFEM")
    print(f"  scope: m16067 CASE-FEM-SCORDELIS-LO-ROOF")
    print(f"  mission_id: 16129 (this audit mission)")
    print(f"  level: 3 (L3 = system integration)")
    print(f"  verdict_by: board-audit")
    print(f"  evidence: see {HERE / 'verdict_evidence.json'}")
    print("=" * 60)

if __name__ == "__main__":
    main()
