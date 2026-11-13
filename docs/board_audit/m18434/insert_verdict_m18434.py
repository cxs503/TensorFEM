#!/usr/bin/env python3
"""Insert m18434 board-audit verdict (auditing m18409) into tensor_maturity_verdicts.

Usage: python3 insert_verdict_m18434.py
Connects as postgres via runuser (local trust); evidence JSON from verdict_evidence.json.
"""
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
EVIDENCE = json.loads((HERE / "verdict_evidence.json").read_text())

SQL = f"""
INSERT INTO tensor_maturity_verdicts
    (repo, scope, mission_id, level, evidence, verdict_by)
VALUES
    ('TensorFEM', 'm18409 XVAL-LBM-SHAN-CHEN-MCMP-BARYCENTRIC-GUO', 18434, 2,
     '{json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")}', 'board-audit');
"""


def main() -> None:
    r = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
         "-c", SQL, "-v", "ON_ERROR_STOP=1"],
        capture_output=True, text=True, timeout=30,
    )
    print(r.stdout)
    if r.returncode != 0:
        print(r.stderr)
        raise SystemExit(f"INSERT failed rc={r.returncode}")
    chk = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-c",
         "SELECT id, repo, scope, mission_id, level, verdict_by FROM tensor_maturity_verdicts "
         "WHERE mission_id=18434 AND verdict_by='board-audit'"],
        capture_output=True, text=True, timeout=30,
    )
    print(chk.stdout)


if __name__ == "__main__":
    main()
