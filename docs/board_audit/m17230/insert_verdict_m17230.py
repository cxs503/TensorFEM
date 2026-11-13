#!/usr/bin/env python3
"""Insert the board-audit verdict for subject m17194 (audit mission 17230).

Run:  python3 insert_verdict_m17230.py
Peer auth via the local unix socket (psql -U postgres -d tensorplatform) is used,
so no password is needed; --print only renders the SQL.
"""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = json.loads((HERE / "verdict_evidence.json").read_text())
LEVEL = 4
SCOPE = "m17194 CASE-FEM-JINTEGRAL-EDGE-CRACK"

SQL = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorFEM', '{scope}', 17230, {level}, '{ev}'::jsonb, 'board-audit');"
)


def main() -> int:
    ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
    sql = SQL.format(scope=SCOPE, level=LEVEL, ev=ev)
    if "--print" in sys.argv:
        print(sql)
        return 0
    existing = subprocess.run(
        ["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-c",
         "SELECT count(*) FROM tensor_maturity_verdicts WHERE scope='" + SCOPE + "'"],
        capture_output=True, text=True).stdout.strip()
    if existing not in ("", "0"):
        print(f"verdict already present ({existing} rows) — skipped")
        return 0
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-c", sql],
                       capture_output=True, text=True)
    print(r.stdout or r.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())