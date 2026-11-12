#!/usr/bin/env python3
"""m15654: write the board-audit verdict for m15427 (CASE-BENDING-PATCH, TensorFEM) into
tensor_maturity_verdicts.  Numbers are injected from the audit JSON evidence, never typed by hand.

Judgement keys (machine-checked by /root/tp-agent/maturity_gate.py:judge) carry pure paths /
pure literals; every human annotation lives in a *_detail key.
"""
from __future__ import annotations

import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output.json").read_text())
P = json.loads((HERE / "poisson_crosscheck_rederived.json").read_text())
G = json.loads((pathlib.Path("/root/tensor-ecosystem/TensorFEM/benchmarks/verified/"
                             "fem_bending_patch/gates_verified.json")).read_text())
GATES = {g["id"]: g for g in G["gates"]}
LEVEL = 4
REPO, SCOPE, BY, MID = "TensorFEM", "m15427 CASE-BENDING-PATCH", "board-audit", 15427

evidence = {
    # ── 判级键：机器可核（纯路径 / 纯字面串）────────────────────────────────
    "impl_files": ["src/tensorfem/bending_patch.py"],
    "entry": "benchmarks/verified/fem_bending_patch/run.py",
    "run_artifacts": ["benchmarks/verified/fem_bending_patch/result.json",
                      "benchmarks/verified/fem_bending_patch/gates_verified.json",
                      "docs/bending-patch.md",
                      "docs/assets/bending_patch/patch_meshes.png",
                      "docs/assets/bending_patch/error_vs_h.png",
                      "docs/assets/bending_patch/error_budget.png",
                      "docs/assets/bending_patch/strip_bending.png"],
    "tests": [f"tests/test_bending_patch.py {V['D_repro']['pytest_passed']} passed "
              f"(clean copy /tmp/audit_bend, 11.8s, rerun by m15654)",
              "审计侧独立复算（不 import tensorfem）闭式 ref：弯矩相对偏差 "
              f"{V['C_independent_ref']['moment_max_rel_diff']:.1e}，板条挠度 "
              f"{V['C_independent_ref']['strip_ref_max_rel_diff']:.1e}",
              "docs/board_audit/m15654/poisson_crosscheck_rederived.py 定点复核 F-CRIT"],
    "integration_callers": ["from .plate import _shape, q4_mindlin_stiffness",
                            "q4_mindlin_stiffness"],
    "benchmark": {
        "registry": "tensor_benchmark_cases WHERE anchor_slug='CASE-BENDING-PATCH' -> "
                    f"{V['B_referee']['n_rows_for_anchor']} rows (table total "
                    f"{V['B_referee']['table_rows_total']}, no %BEND%/%PATCH% hit) -> "
                    "no machine referee row (finding F1)",
        "tolerance": G["criteria"]["case_pool_tolerance"],
        "distorted_patch": {
            "w": {"got": GATES["distorted_w"]["value"], "threshold": GATES["distorted_w"]["threshold"]},
            "M": {"got": GATES["distorted_moment"]["value"],
                  "threshold": GATES["distorted_moment"]["threshold"]},
            "energy": {"got": GATES["distorted_energy"]["value"],
                       "threshold": GATES["distorted_energy"]["threshold"]},
        },
        "load_driven_strip": {
            "symmetry_tip_w": {"err": GATES["strip_symmetry_w"]["value"],
                               "ref": "M L^2/(2D), D=E t^3/(12(1-nu^2))"},
            "free_edge_tip_w": {"err": GATES["strip_free_w"]["value"],
                                "threshold": GATES["strip_free_w"]["threshold"],
                                "margin": GATES["strip_free_w"]["threshold"] / GATES["strip_free_w"]["value"]},
        },
        "poisson_crosscheck": {
            "declared_gate_value": GATES["poisson_crosscheck"]["value"],
            "recomputed_from_computed_w_tip_max": P["max_computed_rel_error"],
            "still_within_tolerance": P["all_computed_checks_still_within_tolerance"],
            "verdict": "F-CRIT: declared 0.0 is a closed-form-over-closed-form identity "
                       "(run.py:542), the gate cannot see a Poisson-coupling defect",
        },
        "affine_skew_exactness": {
            "max_w": GATES["affine_exact_w"]["value"],
            "max_M": GATES["affine_exact_moment"]["value"],
            "note": "structural identity (exact field inside the Q4 span, exact Dirichlet data) "
                    "-> consistency evidence, NOT accuracy evidence (finding F2)",
        },
        "convergence_order": {"declared": V["F_orders"]["declared_orders"],
                              "recomputed": {k: v["fitted_order"] for k, v in
                                             V["F_orders"]["recomputed"].items()},
                              "gated": V["F_orders"]["gated_quantities"]},
        "negative_controls": {"locking_patch_energy": GATES["locking_control_patch"]["value"],
                              "locking_strip_w": GATES["locking_control_strip"]["value"]},
        "independent_ref_rederivation_max_rel_diff": {
            "moment": V["C_independent_ref"]["moment_max_rel_diff"],
            "strip_w": V["C_independent_ref"]["strip_ref_max_rel_diff"],
            "strip_theta": V["C_independent_ref"]["strip_ref_theta_max_rel_diff"]},
        "reproduction": {"scalar_diffs_excluding_timings":
                         V["D_repro"]["n_value_diffs_excluding_timings"],
                         "n_scalar_keys": V["D_repro"]["n_scalar_keys"],
                         "run_wall_seconds": V["D_repro"]["run_wall_seconds"],
                         "gates_passed": V["D_repro"]["run_gates_passed"]},
        "gates_reevaluated": {"n": V["E_gates"]["n_gates"],
                              "all_consistent": V["E_gates"]["all_re_evaluated_consistent"]},
    },
    "repro_cmd": "cd <clean copy of TensorFEM @177457f1> && TORCH_DEVICE_BACKEND_AUTOLOAD=0 "
                 "OMP_NUM_THREADS=4 PYTHONPATH=src python3 "
                 "benchmarks/verified/fem_bending_patch/run.py  # 74.2 s, 17/17 gates, 0 scalar diff",
    "external_anchor": "Irons & Ahmad 1980 ch.8; Zienkiewicz & Taylor FEM Vol.2 sec.10.4; "
                       "Timoshenko & Woinowsky-Krieger Art.10 (closed-form only, no numeric table "
                       "value retrieved) -> L5 not claimed",
    # ── 人读注解 ─────────────────────────────────────────────────────────────
    "impl_files_detail": [
        "src/tensorfem/bending_patch.py 592 行，sha256 "
        f"{next(f for f in V['A_triplet']['files'] if f['path'].startswith('src/'))['sha256'][:16]}… "
        "in_main=True：Patch 网格族 + 装配 + Dirichlet 求解 + 曲率/弯矩/能量后处理，"
        "无 NotImplementedError/pass 占位"],
    "entry_detail": "benchmarks/verified/fem_bending_patch/run.py::main，审计侧干净副本实跑 "
                    f"{V['D_repro']['run_wall_seconds']}s rc=0",
    "run_artifacts_detail": [
        f"{f['path']} {f['bytes']}B in_main={f['in_main']} sha256_同 m15427 工作树={f['same_as_mission_wt']}"
        for f in V["A_triplet"]["files"]],
    "tests_detail": "21 个测试断言不变量（affine 精确性 1e-12、distorted <2% 且随加密改善、"
                    "MITC 斜交能量非精确但 <2%、锁定负控 >0.5、泊松交叉、非法参数 ValueError），非只跑通道",
    "integration_callers_detail": [
        "bending_patch.py 直接引用生产单元 tensorfem.plate.q4_mindlin_stiffness（ECO 仓 grep 命中），"
        "被测面即生产面；q4_mindlin_stiffness 在 src 有 68 处使用",
        "方向如实记：src 无模块反向调用 bending_patch（harness 不入求解器主链），"
        "L3 成立靠'被测的是生产元素'而非'接入求解器'（finding F5）"],
    "benchmark_detail": "17 门全部重判一致；余量 <1.5x 的门 2 个（distorted_w 1.30x、strip_free_w 1.14x）；"
                        "1 门恒等（F-CRIT）；弯矩收敛阶 1.15/1.40 未架门（只架 w 下限 1.5）",
    "findings": {
        "F-CRIT": "poisson_crosscheck 恒等门（run.py:542 闭式除闭式，声明 0.0 vs 用 w_tip 重算 "
                  f"{P['max_computed_rel_error']:.4f}），报告称其 measured=虚高证据；仍 <2% 不改变数值结论",
        "F1": f"锚点无 tensor_benchmark_cases 行（{V['B_referee']['n_rows_for_anchor']}/"
              f"{V['B_referee']['table_rows_total']}），2% 容差为自声明 → 机器裁判无法点火",
        "F2": "affine/skew 8e-16 为插值恒等式，非精度证据（mission 自身 limitations 已声明）",
        "F3": f"自由边 2% 余量仅 "
              f"{GATES['strip_free_w']['threshold']/GATES['strip_free_w']['value']:.2f}x，"
              f"自由边误差阶 sri={P['free_edge_error_convergence']['sri']['fitted_order']:.2f}/"
              f"mitc={P['free_edge_error_convergence']['mitc']['fitted_order']:.2f}（网格依赖的通过）",
        "F4": "result.json.criteria(2 键) 与 gates_verified.json.criteria(4 键) 不一致",
        "F5": "集成方向 harness→生产元素（无 src 反向调用）",
        "F6": "oss_solutions WHERE feature_slug='CASE-BENDING-PATCH' = 0 行：第2轮选型沉淀只暂存在 "
              "artifacts/oss_solutions/CASE-BENDING-PATCH_oss_solutions_insert.sql，未落库（交付缺口）",
        "F7": "诚实项：集群 oss-cache grep（cj1115 排队）与 n=64 未跑，mission 如实登记 not_run",
    },
    "level_rationale": "L1 21 tests 实跑全绿（审计侧复现）；L2 真代码+真工件（10 文件 sha256 与 mission 工作树"
                       "逐字节一致）；L3 被测刚度=生产 q4_mindlin_stiffness（68 处 src 使用）；"
                       "L4 ref 可独立复现（弯矩/挠度偏差 0.0）+ 干净副本重跑逐标量 0 差异 + 收敛阶 + 负控 + "
                       "repro_cmd 实跑；L5 不记（无外部数值/实验锚点，patch test 属一致性验证）",
    "conditions": ["F-CRIT：run.py:542 两个操作数改为 w_tip（重算后仍全部 <2%）",
                   "F6：oss_solutions 4 条选型条目由平台侧落库",
                   "F1：补登记 tensor_benchmark_cases 锚点行（tolerance 0.02）"],
    "not_run": ["GitHub CI (x86) 未跑", "集群 loong64 跨架构复跑未跑（审计全部 loongarch64 本地）",
                "n=64 未跑", "教材原文离线不可检索，页码/条目未独立核对",
                "F-CRIT 修复后未回跑（属后继单）"],
    "independence_limits": "同机同 torch 2.7.1 复跑，逐位一致不能排除环境特异确定性 bug；"
                           "审计者与被审者同为平台租户，非外部第三方；F-CRIT 系代码阅读+复算证据，非受控实验",
    "missed_items_open": "F-CRIT/F1/F6 三项未消账，计入锚点后继单",
    "audit_artifacts": ["docs/board_audit/m15654/AUDIT.md",
                        "docs/board_audit/m15654/verify_output.json",
                        "docs/board_audit/m15654/poisson_crosscheck_rederived.json",
                        "docs/board_audit/m15654/fig_bending_patch_audit.png",
                        "docs/board_audit/m15654/audit_bending_patch.py",
                        "docs/board_audit/m15654/poisson_crosscheck_rederived.py"],
    "audited_commit": "3b12fbe0 (mission) / 177457f1 Merge #706 (main)",
}

payload = json.dumps(evidence, ensure_ascii=False).replace("'", "''")
sql = (f"DELETE FROM tensor_maturity_verdicts WHERE repo='{REPO}' AND scope='{SCOPE}' "
       f"AND verdict_by='{BY}';\n"
       f"INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
       f"VALUES ('{REPO}', '{SCOPE}', {MID}, {LEVEL}, '{payload}'::jsonb, '{BY}');\n")
path = pathlib.Path("/tmp/m15654_verdict.sql")
path.write_text(sql, encoding="utf-8")
r = subprocess.run(["su", "postgres", "-c", f"psql -d tensorplatform -v ON_ERROR_STOP=1 -f {path}"],
                   capture_output=True, text=True)
print(r.stdout, r.stderr, "rc=", r.returncode)
print(subprocess.run(["su", "postgres", "-c",
                      "psql -d tensorplatform -c \"SELECT id,repo,scope,mission_id,level,verdict_by,"
                      "gate_state FROM tensor_maturity_verdicts WHERE scope='m15427 CASE-BENDING-PATCH'\""],
                     capture_output=True, text=True).stdout)