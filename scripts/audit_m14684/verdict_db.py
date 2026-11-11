"""Read/insert tensor_maturity_verdicts for the m14959 board-audit mission.

Credentials come from the platform .env (never inlined on the command line).
Usage:
  python3 scratch/verdict_db.py read
  python3 scratch/verdict_db.py insert <verdict.json>
"""
from __future__ import annotations

import json
import re
import sys

ENV = "/root/tensor-ecosystem/TensorPlatform/.env"


def dsn():
    url = None
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            url = line.split("=", 1)[1].strip()
    assert url, "DATABASE_URL not found"
    m = re.match(r"postgresql\+\w+://([^:]+):([^@]+)@([^/]+)/(\S+)", url)
    user, pw, host, db = m.groups()
    db = db.split("?")[0]
    host, _, port = host.partition(":")
    return dict(host=host or "127.0.0.1", port=int(port or 5432),
                user=user, password=pw, dbname=db)


def connect():
    import psycopg
    return psycopg.connect(**dsn(), autocommit=True)


def read(limit=8):
    with connect() as conn:
        cur = conn.execute(
            "SELECT id, repo, scope, mission_id, level, verdict_by, gate_state,"
            " gate_level, gate_note, evidence FROM tensor_maturity_verdicts"
            " ORDER BY id DESC LIMIT %s", (limit,))
        for row in cur.fetchall():
            print(json.dumps({
                "id": row[0], "repo": row[1], "scope": row[2], "mission_id": row[3],
                "level": row[4], "verdict_by": row[5], "gate_state": row[6],
                "gate_level": row[7], "gate_note": row[8],
                "evidence": row[9]}, ensure_ascii=False)[:1500])


def insert(path):
    payload = json.load(open(path))
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO tensor_maturity_verdicts"
            " (repo, scope, mission_id, level, evidence, verdict_by)"
            " VALUES (%s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (repo, scope, verdict_by) DO UPDATE"
            " SET level = EXCLUDED.level, evidence = EXCLUDED.evidence,"
            " mission_id = EXCLUDED.mission_id"
            " RETURNING id, repo, scope, mission_id, level, verdict_by, gate_state",
            (payload["repo"], payload["scope"], payload["mission_id"],
             payload["level"], json.dumps(payload["evidence"], ensure_ascii=False),
             payload["verdict_by"]))
        print(json.dumps(cur.fetchone(), ensure_ascii=False))


if __name__ == "__main__":
    {"read": read, "insert": lambda: insert(sys.argv[2])}[sys.argv[1]]()