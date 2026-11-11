#!/usr/bin/env python3
"""m15313 交差审计：CASE-NAFEMS-LE1（椭圆膜应力集中, TensorFEM）取证与判级。

只读取证据（+ 独立复算），不改被审工作树。产出 verify_output.json，
其中每个数字都有实测来源：文件 sha256 / SQL 行 / 本单重算 / 重跑日志。
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
W_TARGET = pathlib.Path("/root/tp-agent/work/m15238")   # LE1 真实载体（见 scope_binding）
W_SCOPE = pathlib.Path("/root/tp-agent/work/m15261")   # 任务单点名的 mission（实为 LE11）


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "MISSING"


def sql(q: str) -> str:
    r = subprocess.run(["su", "postgres", "-c", f'psql -d tensorplatform -t -A -F "|" -c "{q}"'],
                       capture_output=True, text=True)
    return r.stdout.strip()


def git(repo: pathlib.Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True).stdout.strip()


out: dict = {"auditor_mission": 15313, "anchor": "CASE-NAFEMS-LE1"}

# ---------- A. scope binding ----------
out["scope_binding"] = {
    "named_mission_in_task": 15261,
    "named_mission_task_key": sql("SELECT task_key FROM tensor_dev_missions WHERE id=15261"),
    "named_mission_goal_anchor": "CASE-NAFEMS-LE11（3D 椭球壳体应力）",
    "named_mission_tree_evidence": f"{W_SCOPE}/runs/nafems_le11/nafems_le11_results.json exists="
                                   f"{(W_SCOPE / 'runs/nafems_le11/nafems_le11_results.json').is_file()}",
    "anchor_true_carrier_mission": 15238,
    "anchor_true_carrier_task_key": sql("SELECT task_key FROM tensor_dev_missions WHERE id=15238"),
    "anchor_true_carrier_head": git(W_TARGET, "rev-parse", "--short", "HEAD"),
    "anchor_true_carrier_merge_commit_in_main": git(W_TARGET, "log", "--oneline", "-n", "50", "origin/main")
                                                .split("\n"),
    "note": "任务单把 mission_id=15261 与锚点 CASE-NAFEMS-LE1 绑在一起；实测 15261=LE11（另有 m15318 专项审计）。"
            "本单按锚点（CASE-NAFEMS-LE1=唯一裁判对象）核验其真实载体 m15238，并保留模板 scope 串以便机器对账。",
}
out["scope_binding"]["merge_commit"] = [l for l in out["scope_binding"]["anchor_true_carrier_merge_commit_in_main"]
                                        if "m15238" in l]
del out["scope_binding"]["anchor_true_carrier_merge_commit_in_main"]

# ---------- B. 三件套 ----------
BENCH = W_TARGET / "benchmarks/nafems_le1.py"
DOC = W_TARGET / "docs/nafems_le1_report.md"
J = W_TARGET / "artifacts/nafems_le1/nafems_le1_report.json"
REPRO = W_TARGET / "runs/nafems_le1/repro.md"
FIGS = sorted((W_TARGET / "docs/assets/nafems_le1").glob("*.png"))
committed_raw = subprocess.run(["git", "show", "HEAD:artifacts/nafems_le1/nafems_le1_report.json"],
                               cwd=W_TARGET, capture_output=True, text=True).stdout
data = json.loads(committed_raw)   # 提交版（repro 重跑会覆盖磁盘文件，故从 HEAD 取）

out["B1_run_artifacts"] = {
    "json": {"path": str(J.relative_to(W_TARGET)), "sha256": sha(J), "bytes": J.stat().st_size},
    "md": {"path": "artifacts/nafems_le1/nafems_le1_summary.md",
           "sha256": sha(W_TARGET / "artifacts/nafems_le1/nafems_le1_summary.md")},
    "figures": [{"path": str(f.relative_to(W_TARGET)), "sha256": sha(f), "bytes": f.stat().st_size} for f in FIGS],
    "in_main": {p: git(W_TARGET, "cat-file", "-e", f"origin/main:{p}") == "" and True or False
                for p in ["benchmarks/nafems_le1.py", "docs/nafems_le1_report.md",
                          "artifacts/nafems_le1/nafems_le1_report.json"]},
    "on_disk_vs_main_note": "bench/docs 已随 PR #686 合入 main；artifacts/*.json 只在工作树（门禁滤除），复现须重跑",
}
out["B1_run_artifacts"]["in_main"] = {
    p: subprocess.run(["git", "cat-file", "-e", f"origin/main:{p}"], cwd=W_TARGET,
                      capture_output=True).returncode == 0
    for p in ["benchmarks/nafems_le1.py", "docs/nafems_le1_report.md",
              "artifacts/nafems_le1/nafems_le1_report.json"]
}

# ref vs got（基准裁判=注册案例行；实测无该行 → 退回闭式解析锚点）
rows = [{"n": r["n"], "dofs": r["dofs"], "fe": r["transverse_tip"]["hoop_fe"],
         "exact": r["transverse_tip"]["exact"], "err_pct": r["transverse_tip"]["relative_error_pct"]}
        for r in data["convergence_results"]]
out["B2_benchmark"] = {
    "registered_judge_sql": "SELECT anchor_slug FROM tensor_benchmark_cases WHERE anchor_slug='CASE-NAFEMS-LE1'",
    "registered_rows": sql("SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug='CASE-NAFEMS-LE1'"),
    "registered_total_rows": sql("SELECT count(*) FROM tensor_benchmark_cases"),
    "ref_declared_in_report": data["reference"],
    "tolerance_pct_declared": data["tolerance_pct"],
    "rows_ref_vs_got": rows,
    "finest_error_pct": rows[-1]["err_pct"],
    "within_tolerance": rows[-1]["err_pct"] < data["tolerance_pct"],
    "honest_coarse_fail": [r["n"] for r in rows if r["err_pct"] >= data["tolerance_pct"]],
}

# 独立复算解析锚点（不用被审模块）：Inglis 1913，a 沿载半轴、b 横向半轴
a, b, T = 1.0, 0.5, 1.0
kt_indep = 1.0 + 2.0 * b / a
out["B2_benchmark"]["independent_analytic_check"] = {
    "formula": "K_t = 1 + 2b/a（Inglis 1913，横向尖点 x 向正应力）",
    "kt_independent": kt_indep, "module_exact": rows[-1]["exact"],
    "agrees": abs(kt_indep - rows[-1]["exact"]) < 1e-12,
    "load_axis_tip_independent": {"formula": "K_t' = -1（沿载方向尖点，σ_xx=-T）",
                                  "json_value": data["convergence_results"][-1]["load_axis_tip"]["exact"]},
}

# 报告表 vs JSON 一致性（零手抄核验）
import re
doc = DOC.read_text()
table = re.findall(r"^\| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+)% \|", doc, re.M)
parsed = [{"n": int(n), "dofs": int(d), "fe": float(f), "exact": float(e), "err_pct": float(er)}
          for n, d, f, e, er in table]
mismatch = []
for p, r in zip(parsed, rows):
    if p["n"] != r["n"] or abs(p["fe"] - r["fe"]) > 5e-6 or abs(p["err_pct"] - r["err_pct"]) > 5e-3:
        mismatch.append({"doc": p, "json": r})
out["B2_benchmark"]["doc_table_vs_json"] = {"parsed_rows": len(parsed), "mismatches": mismatch,
                                            "consistent": not mismatch and len(parsed) == len(rows),
                                            "tolerance": "四舍五入容差 fe<=5e-6 / err_pct<=5e-3（doc 显示 6 位与 2 位小数）"}

# 报告口径复算：收敛阶 / Richardson
def order(e1, e2, n1, n2):
    return math.log(e1 / e2) / math.log(n2 / n1)

pairs = [(rows[i], rows[i + 1]) for i in range(len(rows) - 1)]
orders = [{"pair": f"n{r1['n']}->n{r2['n']}", "p": round(order(r1["err_pct"], r2["err_pct"], r1["n"], r2["n"]), 4)}
          for r1, r2 in pairs]
r1, r2 = rows[-2], rows[-1]
rich = (r2["n"] * r2["fe"] - r1["n"] * r1["fe"]) / (r2["n"] - r1["n"])
out["B2_benchmark"]["report_claim_probes"] = {
    "doc_claims_observed_order": re.findall(r"观测收敛阶约 ([\d.]+)", doc),
    "recomputed_orders": orders,
    "doc_claims_richardson": re.findall(r"Richardson 外推给出极限值 ([\d.]+)", doc),
    "richardson_p1_from_n40_n48": round(rich, 6),
    "richardson_rel_err_vs_exact": round(abs(rich - 2.0) / 2.0 * 100, 5),
    "doc_claims_grading_penalty": re.findall(r"误差将增大 ~?(\d+)%", doc),
    "reference_evidence_json_observed_rates_mild": json.loads(
        (W_TARGET / "benchmarks/elliptic_hole_reference.json").read_text())["fe_cases"]["mild $K_t$"]["observed_rates"],
    "reference_evidence_json_sharp_observed_order": json.loads(
        (W_TARGET / "benchmarks/elliptic_hole_reference.json").read_text())["conclusion"]["sharp_Kt_observed_order"],
}

# ---------- C. repro ----------
repro_log = pathlib.Path("/tmp/m15313/le1_repro.log")
rerun_json = W_TARGET / "artifacts/nafems_le1/nafems_le1_report.json"
rerun = json.loads(rerun_json.read_text())
rerun_rows = [{"n": r["n"], "fe": r["transverse_tip"]["hoop_fe"], "err_pct": r["transverse_tip"]["relative_error_pct"]}
              for r in rerun["convergence_results"]]
repro_diff = [{"n": a2["n"], "fe_committed": a2["fe"], "fe_repro": b2["fe"],
               "d_fe": abs(a2["fe"] - b2["fe"]), "err_pct_committed": a2["err_pct"], "err_pct_repro": b2["err_pct"]}
              for a2, b2 in zip(rows, rerun_rows)]
out["B3_repro"] = {
    "cmd": "cd /root/tp-agent/work/m15238 && PYTHONPATH=src python3 benchmarks/nafems_le1.py",
    "repro_md": {"path": str(REPRO.relative_to(W_TARGET)), "sha256": sha(REPRO)},
    "rerun_log_bytes": repro_log.stat().st_size if repro_log.is_file() else -1,
    "rerun_rows": rerun_rows,
    "max_abs_fe_diff": max(d["d_fe"] for d in repro_diff),
    "max_abs_err_pct_diff": max(abs(d["err_pct_repro"] - d["err_pct_committed"]) for d in repro_diff),
    "rows": repro_diff,
    "reproduced": max(abs(d["err_pct_repro"] - d["err_pct_committed"]) for d in repro_diff) < 1e-6,
}

# 真实性守卫：重跑进程若未写出工件（stdout 0B 且 json mtime 早于启动），不得记为复现成功
_mtime = rerun_json.stat().st_mtime if rerun_json.is_file() else 0
_repro_started = 1770000000 if False else 0
out["B3_repro"]["rerun_artifact_mtime"] = _mtime
if out["B3_repro"]["rerun_log_bytes"] == 0:
    out["B3_repro"]["reproduced"] = False
    out["B3_repro"]["repro_note"] = ("重跑进程 22:05:20 启动，审计窗口（~12 min）内未写出工件：stdout 缓冲 0B、"
        "artifacts json mtime 仍为 21:02:58（提交版）→ 上栏 max diff 是『提交版 vs 提交版』的自比对，不构成复现证据；"
        "复现性判定=NOT_COMPLETED（未证实亦未否证）。")
else:
    out["B3_repro"]["repro_note"] = "重跑写出工件，与提交版逐网格比对见 rows"

# ---------- D. level gates ----------
out["D_level_gates"] = {
    "impl_files": [
        {"path": "benchmarks/nafems_le1.py", "lines": len(BENCH.read_text().splitlines()), "sha256": sha(BENCH),
         "in_main": out["B1_run_artifacts"]["in_main"]["benchmarks/nafems_le1.py"]},
        {"path": "src/tensorfem/elliptic_hole.py", "lines": len((W_TARGET / "src/tensorfem/elliptic_hole.py").read_text().splitlines()),
         "sha256": sha(W_TARGET / "src/tensorfem/elliptic_hole.py"),
         "preexisting": subprocess.run(["git", "merge-base", "--is-ancestor", "origin/main",
                                        "HEAD~1"], cwd=W_TARGET, capture_output=True).returncode == 0,
         "note": "Q4 求解面（ContinuumModel+solve_continuum）+ Kolosov-Muskhelishvili 解析面，本单未改"},
    ],
    "entry": "benchmarks/nafems_le1.py::main（PYTHONPATH=src python3 benchmarks/nafems_le1.py）",
    "tests": {
        "dedicated_le1_test": (W_TARGET / "tests/test_nafems_le1.py").is_file(),
        "module_tests": ["tests/test_elliptic_hole_reference.py", "tests/test_elliptic_hole_sharp_tip.py"],
        "module_tests_in_main": [subprocess.run(["git", "cat-file", "-e", f"origin/main:{p}"], cwd=W_TARGET,
                                                capture_output=True).returncode == 0
                                 for p in ["tests/test_elliptic_hole_reference.py",
                                           "tests/test_elliptic_hole_sharp_tip.py"]],
        "note": "底层模块有门（Kirsch 极限/牵引自由/本构/强式平衡/Q4 尖点 3% 门），"
                "但 LE1 这个算例本身没有专门回归测试（改 benchmarks 不会被测试挡住）",
    },
    "integration_callers": ["benchmarks/nafems_le1.py:41 from tensorfem.elliptic_hole import solve_case",
                            "src/tensorfem/elliptic_hole.py:457 ContinuumModel(...) / :718 solve_continuum(model)",
                            "scripts/audit_elliptic_hole.py:22（同模块既有审计入口）"],
    "platform_report": {"tensor_reports_rows_for_mission_15238":
                        sql("SELECT count(*) FROM tensor_reports WHERE mission_id=15238"),
                        "road_docs_only": "docs/nafems_le1_report.md（六段齐全，见 E）"},
    "report_owed_flag": sql("SELECT report_owed FROM tensor_dev_missions WHERE id=15238"),
}
sections = re.findall(r"^## [①②③④⑤⑥]", doc, re.M)
out["D_level_gates"]["report_sections"] = {"count": len(sections), "found": sections}

# ---------- E. probes (adversarial) ----------
gp = pathlib.Path("/tmp/m15313/grad_probe.json")
out["E_probes"] = {
    "P1_grading_sensitivity": json.loads(gp.read_text()) if gp.is_file() else "not_run",
    "P2_bc_carries_answer": {
        "statement": "外边界 ρ₀=5.0 施加的是**精确 Inglis 面力**（elliptic_hole.py:406 consistent_arc_loads），"
                     "故域截断误差被消除、模型退化为 patch test：所测 1.76% 是纯插值/恢复误差，"
                     "不是对 NAFEMS 有限板原算例的复现。合法但减弱『与手册目标值对比』的说服力。",
        "file": "src/tensorfem/elliptic_hole.py:397-431, benchmarks/nafems_le1.py:96-99",
    },
    "P3_ref_source": {"declared": "NAFEMS A Finite Element Benchmark Set (1990), Test LE1",
                      "cited_page_or_table": bool(re.search(r"(page|Table|p\.\s*\d|原表|原页)", doc)),
                      "actual_value_source": "Inglis 1913 闭式 K_t=1+2b/a（自推/自算，非手册查得）",
                      "knowledge_base_hit": sql("SELECT count(*) FROM tensor_reports WHERE markdown ILIKE '%NAFEMS%LE1%'")},
}
(HERE / "verify_output.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps({k: out[k] for k in ("scope_binding", "B2_benchmark", "B3_repro")},
                 ensure_ascii=False, indent=1)[:4000])
print("WROTE", HERE / "verify_output.json")