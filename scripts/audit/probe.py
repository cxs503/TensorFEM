#!/usr/bin/env python3
"""Read-only probe of the platform registry for the m16138 CASE-FEM-MORLEY-SKEW-PLATE audit."""
import json
import sys

DSN = None
for line in open("/root/tensor-ecosystem/TensorPlatform/.env"):
    if line.startswith("DATABASE_URL="):
        DSN = line.split("=", 1)[1].strip().replace("postgresql+psycopg://", "postgresql://")
        break

ANCHOR = "CASE-FEM-MORLEY-SKEW-PLATE"
SUBJECT = 16138


def q(sql, params=None):
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(DSN, row_factory=dict_row) as c, c.cursor() as cur:
        cur.execute(sql, params or ())
        return cur.fetchall()


def dump(rows):
    print(json.dumps(rows, indent=1, default=str, ensure_ascii=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "cases"
    if what == "cases":
        dump(q("SELECT id,anchor_slug,case_name,ref_source,metric,tolerance,repo,status,passed_at "
               "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", (ANCHOR,)))
        print("--- total rows for anchor:", len(q(
            "SELECT id FROM tensor_benchmark_cases WHERE anchor_slug=%s", (ANCHOR,))))
        dump(q("SELECT anchor_slug,count(*) n FROM tensor_benchmark_cases GROUP BY 1 ORDER BY 1"))
    elif what == "mission":
        dump(q("SELECT id,repo,title,status,goal,result,created_at,finished_at,commit_sha,branch "
               "FROM tensor_dev_missions WHERE id=%s", (SUBJECT,)))
    elif what == "mission_cols":
        dump(q("SELECT column_name,data_type FROM information_schema.columns "
               "WHERE table_name='tensor_dev_missions' ORDER BY ordinal_position"))
    elif what == "reports":
        dump(q("SELECT id,title,created_at,mission_id,scenario_key,length(markdown) ml, "
               "array_length(figures,1) nfig FROM tensor_reports WHERE mission_id=%s ORDER BY id",
               (SUBJECT,)))
    elif what == "report_md":
        rows = q("SELECT id,title,markdown FROM tensor_reports WHERE mission_id=%s ORDER BY id", (SUBJECT,))
        for r in rows:
            print("=" * 70)
            print("report id", r["id"], r["title"])
            print(r["markdown"])
    elif what == "verdicts":
        dump(q("SELECT id,repo,scope,mission_id,level,verdict_by,gate_state,gate_level,created_at, "
               "evidence::text ev FROM tensor_maturity_verdicts WHERE scope ILIKE %s ORDER BY id",
               ("%MORLEY%",)))
    elif what == "verdict_cols":
        dump(q("SELECT column_name,data_type FROM information_schema.columns "
               "WHERE table_name='tensor_maturity_verdicts' ORDER BY ordinal_position"))
    elif what == "casepool":
        dump(q("SELECT * FROM tensor_case_pool WHERE anchor_slug=%s OR mission_id=%s ORDER BY id",
               (ANCHOR, SUBJECT)))