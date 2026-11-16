#!/usr/bin/env python3
"""Insert the m22811 board-audit verdict for m22758 (JOUR-DIFFLBM-REF-VERIFY).

Writes one row into ``tensor_maturity_verdicts`` with the evidence contract of
the maturity gate (impl_files / entry / run_artifacts / tests /
integration_callers / benchmark / repro_cmd / external_anchor) and then prints
the row back plus the gate's own dry-run judgement, so the declared level can
be compared with what the machine will accept.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DATA = json.loads((HERE / "verdict_evidence_m22758.json").read_text())

SQL_TMPL = (
    "INSERT INTO tensor_maturity_verdicts "
    "(repo, scope, mission_id, level, evidence, verdict_by) VALUES "
    "('{repo}', '{scope}', {mission_id}, {level}, '{evidence}'::jsonb, '{verdict_by}') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
    "mission_id = EXCLUDED.mission_id, level = EXCLUDED.level, "
    "evidence = EXCLUDED.evidence, gate_state = 'pending', gate_level = NULL, "
    "gate_note = NULL "
    "RETURNING id, gate_state, level;"
)


def psql(sql: str) -> str:
    r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d",
                        "tensorplatform", "-v", "ON_ERROR_STOP=1", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr)
    return r.stdout


def main() -> int:
    mission_id = DATA["mission_id"]
    scope = f"m{DATA['audited_mission']} {DATA['anchor_slug']}"
    evidence = json.dumps(DATA["evidence"], ensure_ascii=False).replace("'", "''")
    sql = SQL_TMPL.format(repo=DATA["repo"], scope=scope, mission_id=mission_id,
                          level=DATA["maturity_level"], evidence=evidence,
                          verdict_by="board-audit")
    print(psql(sql))

    # --- gate dry-run: what will maturity_gate judge this verdict? ---
    sys.path.insert(0, "/root/tp-agent")
    try:
        import maturity_gate as mg
        gl, why = mg.judge(DATA["maturity_level"], DATA["evidence"],
                           DATA["repo"], mission_id)
        print(f"gate dry-run: gate_level={gl} reasons={why}")
        print("predicted action:", "confirmed" if gl >= 2 else "reopened")
    except Exception as e:  # gate module is outside the worktree; import is optional
        print(f"gate dry-run skipped: {type(e).__name__} {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())