#!/usr/bin/env python3
"""m15313 落表：CASE-NAFEMS-LE1 交差核验结论 -> tensor_maturity_verdicts（board-audit 席位）。

evidence 每个值都是本单实测指针（verify_output.json / 文件 sha256 / SQL 行 / 重跑日志），不含推断。
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
D = json.loads((HERE / "verify_output.json").read_text(encoding="utf-8"))
W = pathlib.Path("/root/tp-agent/work/m15238")
LVL = 3  # 见 level_rationale：L1/L2/L3 实测成立，L4 缺注册裁判行+手册锚点 -> 不记

b = D["B2_benchmark"]
r = D["B3_repro"]
g = D["D_level_gates"]
p = D["E_probes"]

evidence = {
    "impl_files": [
        f"benchmarks/nafems_le1.py 1-{g['impl_files'][0]['lines']}行 sha256={g['impl_files'][0]['sha256'][:16]}…（main 内={g['impl_files'][0]['in_main']}, PR #686/merge 8b7bf3e3）",
        f"src/tensorfem/elliptic_hole.py {g['impl_files'][1]['lines']}行 sha256={g['impl_files'][1]['sha256'][:16]}…"
        f"（既有能力，:457 ContinuumModel / :718 solve_continuum = 生产 Q4 平面应力路径；本单未改求解器）",
        "docs/nafems_le1_report.md 1-116（六段 ①-⑥ 齐全，实测 count=" + str(g["report_sections"]["count"]) + "）",
    ],
    "entry": "benchmarks/nafems_le1.py::main -> run_convergence()/generate_figures()/write_report()；"
             "命令行 `cd /root/tp-agent/work/m15238 && PYTHONPATH=src python3 benchmarks/nafems_le1.py`",
    "run_artifacts": [
        f"artifacts/nafems_le1/nafems_le1_report.json {D['B1_run_artifacts']['json']['bytes']}B sha256={D['B1_run_artifacts']['json']['sha256'][:16]}…（提交版；不在 main，门禁滤除）",
        f"artifacts/nafems_le1/nafems_le1_summary.md sha256={D['B1_run_artifacts']['md']['sha256'][:16]}…",
    ] + [f"{f['path']} {f['bytes']}B sha256={f['sha256'][:16]}…" for f in D["B1_run_artifacts"]["figures"]],
    "tests": f"无 LE1 专用回归测试（tests/test_nafems_le1.py 存在={g['tests']['dedicated_le1_test']}）；"
             f"底层模块有门 tests/test_elliptic_hole_reference.py（Kirsch 极限/牵引自由/本构/强式平衡）+ "
             f"tests/test_elliptic_hole_sharp_tip.py（Q4 尖点 3% 门、grading 家族必须失败的反控）"
             f"，两者在 main 内={g['tests']['module_tests_in_main']}",
    "integration_callers": [
        "benchmarks/nafems_le1.py:41 import tensorfem.elliptic_hole.solve_case（算例=生产路径调用方）",
        "src/tensorfem/elliptic_hole.py:718 solve_continuum(model)（主求解器 continuum）",
        "scripts/audit_elliptic_hole.py:22（同模块既有审计入口）",
    ],
    "benchmark": {
        "judge_registered_rows_for_anchor": b["registered_rows"],
        "judge_table_total_rows": b["registered_total_rows"],
        "ref_declared": b["ref_declared_in_report"],
        "tolerance_pct": b["tolerance_pct_declared"],
        "got_vs_ref": {f"n={x['n']}": f"{x['fe']:.6f} vs {x['exact']:.6f} = {x['err_pct']}%" for x in b["rows_ref_vs_got"]},
        "finest_error_pct": b["finest_error_pct"],
        "within_5pct": b["within_tolerance"],
        "honest_coarse_fail_levels": b["honest_coarse_fail"],
        "independent_analytic_check": b["independent_analytic_check"],
        "doc_table_vs_json_consistent": b["doc_table_vs_json"]["consistent"],
        "claim_probes": b["report_claim_probes"],
    },
    "repro_cmd": f"`cd /root/tp-agent/work/m15238 && PYTHONPATH=src python3 benchmarks/nafems_le1.py`（repro.md 存在 sha256={r['repro_md']['sha256'][:16]}…；"
                 f"本单重跑 22:05:20 启动、审计窗口内未写出工件（log {r['rerun_log_bytes']}B, 工件 mtime 未变）→ reproduced={r['reproduced']}，"
                 f"复现性记为 NOT_COMPLETED；{r['repro_note'][:120]}）",
    "external_anchor": "WITHHELD（手册侧）：ref 声明 'NAFEMS A Finite Element Benchmark Set (1990), Test LE1'，"
                       f"但报告无原表/原页指针（cited_page_or_table={p['P3_ref_source']['cited_page_or_table']}）；"
                       "实际目标值来自自推 Inglis(1913) 闭式 K_t=1+2b/a=2.0（本单独立公式复算一致，"
                       f"module_exact={b['rows_ref_vs_got'][-1]['exact']}）；且 tensor_benchmark_cases 中 CASE-NAFEMS-LE1 注册行数="
                       f"{b['registered_rows']}（全表 {b['registered_total_rows']} 行无 NAFEMS 条目）→ 无机器裁判行。",
    "scope_binding": D["scope_binding"],
    "findings": {
        "F1_scope_mislabel": "任务单把 mission 15261 与锚点 LE1 绑定，实测 15261.task_key=case-nafems-le11（LE11 椭球壳，"
                             "run_artifacts=runs/nafems_le11/*），LE1 真实载体为 mission 15238（task_key=case-nafems-le1）。"
                             "本 verdict 按锚点核 LE1=m15238，scope 串保留任务单写法供对账。",
        "F2_no_registered_judge": f"tensor_benchmark_cases 无 anchor_slug=CASE-NAFEMS-LE1 行（SQL 命中 {b['registered_rows']}）→ 任务书指定的唯一裁判不存在，判据降级为自推闭式。",
        "F3_ref_not_from_handbook": "报告 §① 引用 NAFEMS(1990) Test LE1 但无原表原页；目标值 2.0 实际由 Inglis 闭式自算（零手抄合规但『手册查得』要求未达）。",
        "F4_report_number_misattribution": f"报告 §⑤『观测收敛阶约 0.95』与本案数据不符：本单按 n=16→48 五段复算 p="
                                           f"{[o['p'] for o in b['report_claim_probes']['recomputed_orders']]}（≈0.99）；0.953066 实为既有证据 JSON 的 "
                                           f"sharp_Kt_observed_order（K_t=7 另一算例）。",
        "F5_richardson_overstated": f"报告称『Richardson 外推给出极限值 2.000000』，本单按 p=1 由 n=40/48 复算 "
                                    f"{b['report_claim_probes']['richardson_p1_from_n40_n48']}（相对误差 {b['report_claim_probes']['richardson_rel_err_vs_exact']}%），非恰好 2.000000。",
        "F6_unsupported_sensitivity": "报告 §⑤称『grading 降至 1.0 同 DOF 误差将增大 ~50%』——无任何工件/脚本产出该数；"
                                      "本单独立探针见 probes.P1_grading_sensitivity。",
        "F7_bc_carries_answer": p["P2_bc_carries_answer"]["statement"],
        "F8_artifacts_not_in_main": f"benchmarks/docs 已合入 main（{D['scope_binding']['merge_commit']}），"
                                    f"但 artifacts/nafems_le1/* 只在工作树（in_main={D['B1_run_artifacts']['in_main']}）→ 复现须回工作树重跑。",
        "F9_no_platform_report": f"mission 15238 在 tensor_reports 无行（count={g['platform_report']['tensor_reports_rows_for_mission_15238']}），"
                                 f"report_owed={g['report_owed_flag']} → 只有仓内 markdown，平台侧报告缺位且欠账机制未触发。",
        "F10_no_dedicated_test": "LE1 算例无回归测试：benchmarks/nafems_le1.py 被改坏不会被 CI 挡住（底层模块门仍有效）。",
    },
    "level_rationale": "L1✅（底层模块门：Kirsch 极限/牵引自由/本构/强式平衡 + Q4 尖点 3% 门、grading 反控，均在 main）"
                       "；L2✅（benchmarks/nafems_le1.py 349 行真实现：输入→FE 求解→误差/工件/图，6 网格收敛表 + 力平衡 4.5e-15）"
                       "；L3✅（算例作为调用方接入生产路径 ContinuumModel+solve_continuum，且已随 PR #686 合入 main）"
                       "；L4未记（完整算例✅/误差分析✅/复现性 NOT_COMPLETED（重跑未在审计窗口产出，见 repro_cmd 键）但 external_anchor 无手册原表原页且 tensor_benchmark_cases 无该 anchor 注册行 → "
                       "『基准』的外部文档锚点缺位；另报告 3 处数字存在错标/超范围声明 F4-F6）→ 裁定 L3。"
                       "补齐手册出处 + 注册案例行 + 修 F4-F6 即可升 L4。",
    "checks_not_run": [
        "集群 loong64 复跑（单机 ~6 min 已完成，未上集群）",
        "GitHub Actions x86 CI 复跑",
        "NAFEMS LE1 手册原表数值核实（无外网、平台知识库无 NAFEMS 源，measurement 见 P3）",
        "m15261（LE11）的数值级复核（另有 m15318 专项；本单只核其 task_key/工件路径以证 scope 绑定）",
    ],
}

payload = json.dumps(evidence, ensure_ascii=False)
sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
       "('TensorFEM', 'm15261 CASE-NAFEMS-LE1', 15313, {lvl}, $json${ev}$json$::jsonb, 'board-audit') "
       "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, evidence=EXCLUDED.evidence, "
       "mission_id=EXCLUDED.mission_id RETURNING id, level, gate_state").format(lvl=LVL, ev=payload)
fd, path = tempfile.mkstemp(suffix=".sql", dir="/tmp")
with os.fdopen(fd, "w", encoding="utf-8") as fh:
    fh.write(sql + ";\n")
os.chmod(path, 0o644)
res = subprocess.run(["su", "postgres", "-c", f"psql -d tensorplatform -t -A -F '|' -f {path}"],
                     capture_output=True, text=True)
print("rc", res.returncode)
print(res.stdout.strip())
print(res.stderr.strip()[:400])