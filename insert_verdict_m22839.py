#!/usr/bin/env python3
"""Insert m22839 JOUR-DIFFLBM-REF-VERIFY maturity verdict into tensor_maturity_verdicts."""

import json
import subprocess
import sys
from pathlib import Path

VERDICT_PATH = Path(__file__).parent / "verdict_evidence_m22839.json"

def main():
    with open(VERDICT_PATH) as f:
        data = json.load(f)

    evidence_json = json.dumps(data["evidence"])
    level = data["maturity_level"]
    mission_id = data["mission_id"]
    repo = data["repo"]
    scope = f"m{mission_id} {data['anchor_slug']}"
    verdict_by = "board-audit"

    # Escape single quotes for SQL
    evidence_escaped = evidence_json.replace("'", "''")

    sql = (
        f"INSERT INTO tensor_maturity_verdicts "
        f"(repo, scope, mission_id, level, evidence, verdict_by) VALUES "
        f"('{repo}', '{scope}', {mission_id}, {level}, '{evidence_escaped}', '{verdict_by}');"
    )

    result = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
         "-v", "ON_ERROR_STOP=1", "-c", sql],
        capture_output=True, text=True, timeout=30,
    )

    if result.returncode != 0:
        print(f"ERROR: {result.stderr}", file=sys.stderr)
        return result.returncode

    print(result.stdout)

    # Verify insertion
    check = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform",
         "-c", f"SELECT id, repo, scope, mission_id, level, verdict_by FROM tensor_maturity_verdicts WHERE mission_id={mission_id}"],
        capture_output=True, text=True, timeout=30,
    )
    print(check.stdout)
    return 0

if __name__ == "__main__":
    sys.exit(main())
