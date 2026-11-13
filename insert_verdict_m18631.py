#!/usr/bin/env python3
"""Insert m18631 GAP-PP-3-ICE-STRUCT-PLASTIC-SET maturity verdict into tensor_maturity_verdicts."""

import json
import sqlite3
from pathlib import Path

VERDICT_PATH = Path(__file__).parent / "verdict_evidence_m18631.json"

def main():
    with open(VERDICT_PATH) as f:
        evidence = json.load(f)

    evidence_json = json.dumps(evidence["evidence"])
    level = evidence["maturity_level"]
    mission_id = evidence["mission_id"]

    db_candidates = [
        Path("/root/tp-agent/data/tensor_platform.db"),
        Path("/root/tp-agent/tensor_platform.db"),
        Path("/root/tp-agent/work/tensor_platform.db"),
    ]

    db_path = None
    for p in db_candidates:
        if p.exists():
            db_path = p
            break

    if db_path is None:
        print(f"WARNING: No database found among candidates: {db_candidates}")
        print("Verdict data prepared but not inserted. Evidence JSON:")
        print(json.dumps(evidence, indent=2))
        return

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tensor_maturity_verdicts'")
    if not cursor.fetchone():
        print("WARNING: tensor_maturity_verdicts table does not exist. Creating...")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tensor_maturity_verdicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                repo TEXT NOT NULL,
                scope TEXT NOT NULL,
                mission_id INTEGER NOT NULL,
                level INTEGER NOT NULL,
                evidence TEXT NOT NULL,
                verdict_by TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    cursor.execute(
        "SELECT id FROM tensor_maturity_verdicts WHERE mission_id = ? AND verdict_by = 'board-audit'",
        (mission_id,)
    )
    existing = cursor.fetchone()

    if existing:
        print(f"Verdict for mission {mission_id} already exists (id={existing[0]}), updating...")
        cursor.execute(
            "UPDATE tensor_maturity_verdicts SET level = ?, evidence = ? WHERE mission_id = ? AND verdict_by = 'board-audit'",
            (level, evidence_json, mission_id)
        )
    else:
        cursor.execute(
            "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES (?, ?, ?, ?, ?, ?)",
            ("TensorFEM", "m18631 GAP-PP-3-ICE-STRUCT-PLASTIC-SET", mission_id, level, evidence_json, "board-audit")
        )

    conn.commit()
    conn.close()
    print(f"Verdict inserted: repo=TensorFEM, mission_id={mission_id}, level={level}, verdict_by=board-audit")

if __name__ == "__main__":
    main()
