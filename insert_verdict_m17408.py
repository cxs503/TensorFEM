#!/usr/bin/env python3
"""Insert the m17408 board-audit verdict for m17361 into tensor_maturity_verdicts."""
import json
import psycopg

EV = "artifacts/audit_m17361/verdict_evidence_m17408.json"
DSN = "dbname=tensorplatform user=postgres"
d = json.load(open(EV))
row = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       "VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (repo, scope, verdict_by) DO UPDATE "
       "SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id "
       "RETURNING id, level, gate_state")
with psycopg.connect(DSN) as c, c.cursor() as cur:
    cur.execute(row, ("TensorFEM", "m17361 XVAL-FEM-IMPLICIT-IBM-AITKEN-RELAXATION",
                      d["audit_mission"], d["maturity_level"],
                      json.dumps(d["evidence"]), "board-audit"))
    print("inserted/updated:", cur.fetchone())
    c.commit()
