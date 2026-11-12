#!/usr/bin/env python3
"""m15654: publish the paper-grade audit report (six sections, figures inlined as base64)."""
from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tools"))
import tp_api  # noqa: E402

V = json.loads((HERE / "verify_output.json").read_text())
P = json.loads((HERE / "poisson_crosscheck_rederived.json").read_text())
G = json.loads((REPO / "benchmarks/verified/fem_bending_patch/gates_verified.json").read_text())
R = json.loads((REPO / "benchmarks/verified/fem_bending_patch/result.json").read_text())
GA = {g["id"]: g for g in G["gates"]}
AUD = (HERE / "AUDIT.md").read_text()


def b64(p: pathlib.Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


figs = [
    {"name": "fig_bending_patch_audit.png",
     "caption": "审计三面板：(1) distorted patch h 收敛（唯一非恒等的 patch 证据）；"
                "(2) 17 门余量与恒等门（红）；(3) Poisson 交叉：声明 0.0 vs 由计算挠度重算",
     "png_base64": b64(HERE / "fig_bending_patch_audit.png")},
    {"name": "mission_error_vs_h.png",
     "caption": "被审工件自带图（m15427 docs/assets/bending_patch/error_vs_h.png）：distorted patch "
                "位移/弯矩误差 vs h 与 slope-2 参考线（审计侧已复核该图由 result.json 程序生成）",
     "png_base64": b64(REPO / "docs/assets/bending_patch/error_vs_h.png")},
]

prow = "\n".join(
    f"| {r['shear']} | {r['n_x']} | {r['w_free']:.4f} | {r['w_symmetry']:.4f} | "
    f"{r['ratio_computed']:.6f} | {100*r['rel_error_computed']:.3f}% |"
    for r in P["recomputed_from_computed_tip_deflections"])

md = f"""# 交差核验 m15427 — CASE-BENDING-PATCH（TensorFEM）：L4，附 1 项恒等门缺陷

**审计单 m15654**｜对象 m15427（succeeded，PR #706，commit `3b12fbe0`，merged `177457f1`）
｜锚点 `CASE-BENDING-PATCH`｜裁定 **L4**｜处置 **approve with notes**（3 项返修条件见 §6）

## ① 案例介绍（物理背景/几何/参数）

Q4 Mindlin-Reissner 板单元的**常量曲率（弯曲）patch test**：在「无横向剪切的常曲率状态」下，
单元装配体（内节点自由、边界只给该状态本身的节点值）必须精确复现该状态。这是**一致性必要条件**，
不是工程精度算例；因此本审计的核心工作是把"哪些数字是恒等、哪些是真精度信息"分开。

- 材料：E = {R['material']['young']:g} MPa，ν = {R['material']['poisson']:g}，t = {R['material']['thickness']:g} mm
- 网格族：affine / skew / distorted；{len(R['matrix'])} 个 patch 用例 + {len(R['ladder'])} 行细化阶梯 + {len(R['strip'])} 个载荷驱动板条用例
- 基准出处：常量曲率闭式场（Irons & Ahmad 1980 ch.8；Zienkiewicz & Taylor Vol.2 §10.4）；
  悬臂板条 w = M L²/(2D)（Timoshenko & Woinowsky-Krieger Art.10）

## ② 怎么做的（核验方法/工具链）

1. 三件套取证：main 与 m15427 工作树逐文件 sha256 比对（{V['A_triplet']['n_files']} 个文件，
   全部存在且逐字节一致）。
2. **ref 独立性**：不 import tensorfem，纯数学独立重算 D、D_b、M = D_b κ、w = M L²/(2D)、θ = −M L/D
   → 弯矩 ref 偏差 **{V['C_independent_ref']['moment_max_rel_diff']:.1e}**、板条挠度 ref 偏差
   **{V['C_independent_ref']['strip_ref_max_rel_diff']:.1e}**（ref 不是代码自身输出）。
3. **复现**：干净副本（去 .git）重跑 `run.py` + `pytest`，与提交的 `result.json` 逐标量 diff。
4. **门自检**：17 门逐条按各自 operator/threshold 重判；两份工件 criteria 一致性；
   ladder 独立 log-log 最小二乘重算收敛阶。
5. **恒等性检查**：定位结构性恒等（插值恒等式）与真实精度信息；
   并对 `poisson_crosscheck` 做定点复核（读 `run.py:542` 的分子分母来源）。

## ③ 计算结果（数字表，程序注入）

| 项 | 结果 |
|---|---|
| pytest（干净副本重跑） | {V['D_repro']['pytest_passed']} passed |
| run.py 墙钟 | {V['D_repro']['run_wall_seconds']} s（m15427 自报 74.3 s） |
| 17 门 | passed={V['D_repro']['run_gates_passed']}，failed={V['D_repro']['run_failed_gates']} |
| result.json 逐标量 diff（排除 timing） | **{V['D_repro']['n_value_diffs_excluding_timings']} 处**（共 {V['D_repro']['n_scalar_keys']} 键） |
| 弯矩 ref（独立复算） | {V['C_independent_ref']['moment_got']} vs {V['C_independent_ref']['moment_ref_independent']} |

非恒等关键量（真实精度信息）：

| 量 | 误差 | 容差 |
|---|---|---|
| distorted patch w | {100*GA['distorted_w']['value']:.2f}% | 2% |
| distorted patch M | {100*GA['distorted_moment']['value']:.2f}% | 2% |
| 板条对称胞 tip w（载荷驱动） | {GA['strip_symmetry_w']['value']:.2e} | 2% |
| 板条自由边 tip w（载荷驱动） | {100*GA['strip_free_w']['value']:.2f}% | 2%（余量 {GA['strip_free_w']['threshold']/GA['strip_free_w']['value']:.2f}×） |

恒等量（**不计为精度证据**）：affine/skew w {GA['affine_exact_w']['value']:.2e}、M {GA['affine_exact_moment']['value']:.2e}。

## ④ 结果图

图 1 = 审计三面板（收敛/门余量与恒等门/Poisson 交叉重算）；图 2 = 被审工件自带的 error_vs_h。

## ⑤ 注意点（发现，按严重度）

- **F-CRIT｜`poisson_crosscheck` 是恒等门**：`run.py:542` 取
  `w_exact_kirchhoff(free)/w_exact_kirchhoff(symmetry)`，两个操作数都是闭式解，比值代数上恒为
  1/(1−ν²)，故 `rel_error = {P['declared_value']['rel_error']}` 是构造出来的，该门**看不到**元素的
  Poisson 耦合缺陷；而报告 §4.3 写 "Measured ratio … (error 0.0%)"。用**计算挠度** `w_tip` 重算：

| 剪切 | n_x | w_free(计算) | w_symmetry(计算) | 比值 | 相对误差 |
|---|---|---|---|---|---|
{prow}

  → 真实误差 {100*P['max_computed_rel_error']:.2f}%，**仍全部 <2%**：这是**证据虚高**（可一行修复，物理结论不变），
  不是数值造假。
- **F1｜锚点无机器裁判行**：`tensor_benchmark_cases` 共 {V['B_referee']['table_rows_total']} 行，
  `anchor_slug='CASE-BENDING-PATCH'` → {V['B_referee']['n_rows_for_anchor']} 行；2% 容差只以
  `result.json.criteria` 自声明形式存在 → 「基准判据是唯一裁判」的机制对本案无法自动点火。
- **F2｜affine/skew 8e-16 是插值恒等式**（精确场在 Q4 空间内 + 精确 Dirichlet 数据），
  证明一致性非精度；m15427 的 `limitations` 已如实声明。
- **F3｜自由边 2% 余量 1.14×，阶 sri {P['free_edge_error_convergence']['sri']['fitted_order']:.2f} /
  mitc {P['free_edge_error_convergence']['mitc']['fitted_order']:.2f}**：Bygh 是边界层建模差，
  但"<2%"是网格依赖的通过。
- **F6｜交付缺口**：`oss_solutions WHERE feature_slug='CASE-BENDING-PATCH'` = 0 行，
  第 2 轮选型沉淀只暂存在 SQL 文件未落库。
- **F7｜诚实项**：集群 oss-cache grep（cj1115 排队）与 n=64 未跑，mission 如实登记 not_run。

## ⑥ 误差判定（是否 <3%）

| 关键量 | 误差 | <3%? |
|---|---|---|
| 弯矩 ref（独立复算） | {V['C_independent_ref']['moment_max_rel_diff']:.1e} | 是 |
| distorted patch w | {100*GA['distorted_w']['value']:.2f}% | 是 |
| 板条对称胞 tip w | {GA['strip_symmetry_w']['value']:.2e} | 是 |
| 板条自由边 tip w | {100*GA['strip_free_w']['value']:.2f}% | 是（余量薄） |
| Poisson 交叉（计算值重算） | {100*P['max_computed_rel_error']:.2f}% | 是 |
| Poisson 交叉（声明门值 0.0） | 恒等，**不计** | 不适用 |

**回答：非恒等关键量全部 <3%（最大 {100*max(GA['distorted_w']['value'], GA['strip_free_w']['value']):.2f}%）**，
但须扣除 affine/skew 的恒等式与 Poisson 门的 0.0 两处不应记分项。

## 判级与处置

**L4**（L1 21 tests 全绿；L2 真代码+真工件；L3 被测刚度=生产 `q4_mindlin_stiffness`；
L4 ref 可独立复现+逐标量 0 差异复跑+收敛阶+负控；**L5 不记**：无外部数值/实验锚点，
patch test 属一致性验证）。**approve with notes**：F-CRIT（改用 w_tip）、F6（oss_solutions 落库）、
F1（补 registry 行）作为锚点后继单的消账条件。

未跑（not_run，不记通过）：GitHub CI(x86)、集群 loong64 跨架构复跑、n=64、教材原文页码独立核对。
独立性限制：同机同 torch 2.7.1 复跑；审计者与被审者同为平台租户；F-CRIT 为代码阅读+复算证据。

完整审计正文（含全部表格与工件清单）：`docs/board_audit/m15654/AUDIT.md`。
"""

tok, st, _ = tp_api.login()
payload = {
    "title": "交差核验 m15427 CASE-BENDING-PATCH（TensorFEM）：L4，附 1 项恒等门缺陷 F-CRIT",
    "summary": "独立复核 m15427：干净副本逐标量 0 差异复现（21 tests、17/17 门、74.2s），"
               "ref 不 import tensorfem 独立复算偏差 0.0；判定 L4。发现 poisson_crosscheck 门为恒等门"
               "（声明 0.0 vs 计算挠度重算 1.76%，仍 <2%）+ 锚点无 registry 行 + oss_solutions 0 行。",
    "scenario_key": "CASE-BENDING-PATCH",
    "mission_id": 15654,
    "tags": ["board-audit", "m15427", "TensorFEM", "patch-test", "maturity-L4", "tautological-gate"],
    "markdown": md,
    "figures": figs,
}
st, d = tp_api.req("tensor/reports", "POST", payload, token=tok)
print("status", st, "report id", (d or {}).get("id") if isinstance(d, dict) else d)
if st >= 300:
    print(json.dumps(d, ensure_ascii=False)[:800])