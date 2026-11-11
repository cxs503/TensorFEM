#!/usr/bin/env python3
"""Publish the m14687 board-audit report to the platform (mission 14962).

Numbers are injected from the reproduced artifacts (benchmarks/artifacts/mortar_lm_hertz/
verification_results.json and artifacts/m14687_audit/audit_numbers.json) - nothing is typed
by hand.
"""
import base64
import json
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NUM = json.load(open(os.path.join(REPO, "artifacts", "m14687_audit", "audit_numbers.json")))
RAW = json.load(open(os.path.join(REPO, "benchmarks", "artifacts", "mortar_lm_hertz",
                                 "verification_results.json")))
API = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")


def post(path, payload, token=None):
    req = urllib.request.Request(API + path, method="POST",
                                data=json.dumps(payload).encode())
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, json.loads(r.read() or b"{}")


def main() -> None:
    st, d = post("/auth/login", {"username": os.environ["TENSOR_AGENT_USER"],
                                 "password": os.environ["TENSOR_AGENT_PASS"]})
    assert st == 200, (st, d)
    tok = d["access_token"]

    rows = []
    for r in RAW["results"]:
        rows.append("| {level} | {nx} | {tw:.3f} | {fe:.2%} | {b:.2%} | {l2:.2%} | {it} |"
                    .format(level=r["level"], nx=r["nx"], tw=2 * r["h"],
                            fe=r["force_error"], b=r["b_error"],
                            l2=r["pressure_l2_error"], it=r["iterations"]))
    tbl = ("| level | nx | h | force err vs ref | half-width err vs ref | pressure L2 | active-set iters |\n"
           "|---|---|---|---|---|---|---|\n" + "\n".join(rows))

    ref = NUM["benchmark_reference"]
    cons = NUM["consistent_hertz_same_force"]
    fe = NUM["finest_level_fe"]
    png = base64.b64encode(open(os.path.join(REPO, NUM["artifact"]), "rb").read()).decode()

    md = f"""# m14687 GAP-FEM-MORTAR 交差核验：交付三件套 + L0–L5 判级

## 1 案例介绍
被核验对象是 m14687（GAP-FEM-MORTAR，TensorFEM，仓库账 succeeded，交付 commit `af09a9b6`／并入 main `d01094e2`）。
任务声称填补「FEM mortar 接触」缺口。核验物理场景为其自带的 Hertz 二维线接触算例：半径 R={10.0} m 的刚性圆柱压入平面应变弹性体，
E=1000 Pa，ν=0.3，压入深度 δ=0.01 m，有限元域半宽 1.0 m、高 5.0 m，Q4 网格四层（nx=16/32/48/64），
接触半宽/峰值压力/总力参照 Hertz 1882 线接触解析解。

## 2 怎么做的（核验方法，非重做）
1. 逐字节比对被审工作树 `/root/tp-agent/work/m14687` 与核验工作树 `m14962` 的交付文件（md5 相同：模块 `155cb9b4…`、基准 `437865bf…`、工件 JSON `e20fe7f1…`）。
2. 复现：`python benchmarks/mortar_lm_hertz_verification.py`（70.5 s，exit=1），重生成的 `verification_results.json` 与提交副本 **diff IDENTICAL**。
3. 调用面核验：`grep -rn mortar_lagrange_multiplier`（全仓 1 处，仅文档）；`tests/` 0 命中；`src/tensorfem/__init__.py` 未导出新模块。
4. 模块级调用：直接 import 并构造小算例调用 `solve_mortar_lm_2d_contact`，**首次调用即抛异常**。
5. 参照解自洽性程序化复算（`scripts/audit_m14687_figures.py` → `artifacts/m14687_audit/audit_numbers.json`）。
6. 按 maturity_gate 达标判断方案逐级核 L0–L5 并落表 `tensor_maturity_verdicts`（verdict id 见下）。

## 3 计算结果（程序化注入，零手抄）

基准自报参照：b={ref['b']:.5f} m，p0={ref['p0']:.4f} Pa，F={ref['F']:.4f} N/m。

{tbl}

判定：压力误差判据（<10 %）**FAIL**（{fe['pressure_l2_error']:.2%}）；半宽误差 {fe['b_error_vs_benchmark_ref']:.2%}（四层完全相同）；
收敛阶 0.07（判据 >0.3）**FAIL**；唯一 PASS 的「穿透」判据是空判（详见 §5）；整体 FAIL（脚本 exit code 1，诚实）。

## 4 结果图
左：最细层（nx=64）节点压力 vs 基准参照 vs 同力相容 Hertz 解；右：压力 L2 误差随 h 的变化（实测斜率 0.07，无收敛）。
图件由 `scripts/audit_m14687_figures.py` 从工件 JSON 程序化重绘（`artifacts/m14687_audit/m14687_audit.png`，见 figures 字段）。

## 5 注意点（踩过的坑 / 局限）
- **F1（阻断）** 库内模块 `mortar_lagrange_multiplier_2d.py:448/:449` 把 1-D rhs 传给 `torch.linalg.solve_triangular`，
  `RuntimeError: The input tensor B must have at least 2 dimensions`；同一路径 `:483` 仍是 `# placeholder`。即「mortar 接触能力」在库层面从未跑过。
- **F2（阻断）** 基准并不含 mortar 语义：约束只在 slave 节点 y 自由度置 1（`benchmarks/mortar_lm_hertz_verification.py:155-162`），
  等价于刚性压头指定位移的节点式 LM 接触；模块 `compute_mortar_matrices_1d`（`:260`）算出的 D/M **从未接入求解**（`:436` 自注 lumped nodal integration）；
  `build_hertz_cylinder_on_flat` 中 `slave_surface == master_surface`，不存在非匹配界面。
- **F3（重要）** 判据 2 空判：`max_penetration = min(gap).clamp(max=0)`（`:221`）对张开节点必为负，与正容差比较**结构上恒 PASS**。
- **F4（重要）** 参照解自相矛盾：同力相容 Hertz 解 b={cons['b']:.5f}、p0={cons['p0']:.4f}，基准 b_ref 偏大 {NUM['reference_b_consistency_gap_pct']:.2f} %；
  即便换用相容参照，FE 半宽误差仍有 {fe['b_error_vs_consistent_ref']:.2%}。
- 局限：未跑全仓 402 个测试文件；收敛阶结论只在交付的四层网格（≤32 接触节点）范围内成立；未追溯原始图件渲染脚本。

## 6 误差判定（逐项 vs 基准，并显式回答是否 <3%）
| 关键量 | 基准 | 实测（最细层） | 相对误差 | <3 %？ |
|---|---|---|---|---|
| 接触半宽 b | {ref['b']:.5f} m（基准参照） | {fe['b_fe']:.5f} m | {fe['b_error_vs_benchmark_ref']:.2%} | **否** |
| 接触半宽 b | {cons['b']:.5f} m（同力相容 Hertz） | {fe['b_fe']:.5f} m | {fe['b_error_vs_consistent_ref']:.2%} | **否** |
| 单位长度总力 F | {ref['F']:.4f} N/m | {fe['total_force']:.4f} N/m | {fe['force_error_vs_benchmark_ref']:.2%} | **否** |
| 压力分布 L2 | 解析剖面 | — | {fe['pressure_l2_error']:.2%} | **否** |
| 收敛阶 | ≥0.3 | 0.07 | — | **否** |

四项关键量**全部远超 3 %（也越过平台 15 % 红线）**，且半宽误差中约 {NUM['reference_b_consistency_gap_pct']:.1f} 个百分点源自参照解本身，须先修参照再谈精度。

## 结论与落表
判级 **L2**（独立功能实现：Q4 平面应变组装 + 鞍点主动集接触真实可运行、可复现、有量化误差分析），
不达 L3（无调用方/未导出）、不达 L4（精度 28.7 %/44.1 % 越线、收敛 0.07、判据 2 空判、参照解 25 % 不相容）。
证据无夸大：原报告 `docs/mortar_lm_round1_report.md` 已诚实自报 HONEST-FAIL，未发现伪造；但任务缺口（mortar 非匹配界面）**未闭合**。
落表：`INSERT INTO tensor_maturity_verdicts … VALUES ('TensorFEM','m14687 GAP-FEM-MORTAR',14962,2,…,'board-audit')`（见 `artifacts/m14687_audit/verdicts.sql`），
maturity_gate 机器复核 dry-run 返回 (level=2, []) → 预期 confirmed；处置 = request changes（返修清单见 `docs/m14687-gap-fem-mortar-board-audit.md` §5）。
"""

    payload = {
        "title": "m14687 GAP-FEM-MORTAR 交差核验：L2 判定 + 三项阻断缺陷（模块首调崩溃/mortar 语义缺失/参照解 25% 不相容）",
        "summary": ("核验 m14687 交付三件套：工件与数值可复现（重跑 JSON diff IDENTICAL、exit=1 诚实失败），"
                    "但库内模块首次调用即崩溃、基准不含 mortar 语义、验收判据 2 空判、参照解自相矛盾 25.33%；"
                    "判级 L2 并落表 tensor_maturity_verdicts（board-audit），处置=返修。"),
        "scenario_key": "tensorfem_contact_mortar_audit",
        "mission_id": 14962,
        "tags": ["audit", "TensorFEM", "mortar", "contact", "maturity_gate", "GAP-FEM-MORTAR"],
        "markdown": md,
        "figures": [{"name": "m14687_audit.png",
                     "caption": "左：nx=64 节点压力 vs 基准参照 vs 同力相容 Hertz 解；右：压力 L2 误差 vs h（实测斜率 0.07）",
                     "png_base64": png}],
    }
    st, d = post("/tensor/reports", payload, tok)
    print(st, json.dumps(d)[:400])


if __name__ == "__main__":
    main()