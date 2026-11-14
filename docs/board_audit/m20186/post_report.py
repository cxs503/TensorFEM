#!/usr/bin/env python3
"""Publish the m20186 audit report (numbers injected from the audit JSON, not hand-copied)."""
from __future__ import annotations

import base64
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "verify_heat_l2.json").read_text())
lv = {r["mesh"]: r for r in V["reproduced_levels"]}
orders_all = V["orders"]["observed_order_all_nodes"]
orders_free = V["orders"]["observed_order_free_dofs"]
mean_order = V["orders"]["declared_mean_order"]
declared = V["declared_vs_reproduced"]


def row(m: str) -> str:
    r = lv[m]
    d = next((x for x in declared if x["mesh"] == m), None)
    rep = f"{d['rel_diff']:.1e}" if d else "n/a (audit extension)"
    return (f"| {m} | {r['n_dof']} | {r['h']:.5f} | {r['l2_relative_all_nodes']:.6e} | "
            f"{r['l2_relative_free_dofs']:.6e} | {rep} | {r['solve_residual']:.2e} |")


rows = "\n".join(row(m) for m in ["4x4", "8x8", "16x16", "32x32", "64x64"])
pc = " / ".join(f"{o:.3f}" for o in orders_all)
pf = " / ".join(f"{o:.3f}" for o in orders_free)

md = f"""# m20186 交差核验：m20143 / CASE-FEM-HEAT-L2（TensorFEM）—— 未进入仓库的平行实现 + 度量口径缺陷

## ① 案例介绍（物理背景/几何/参数）

稳态热传导制造解验证（MMS, method of manufactured solutions）。单位正方形域 Ω=[0,1]²，无内热源，
导热系数 k=1 W/(m·K)，四边 Dirichlet 温度由精确场给定：

T(x,y) = sin(πx)·sinh(πy)/sinh(π)，∇²T = 0（调和场，Zienkiewicz & Taylor, FEM vol.1, ch.3-4）

离散：Q4 双线性四边形单元，2×2 Gauss 积分，均匀 Cartesian 网格 4×4 → 32×32（本审计再补 64×64）。
判定对象是"稳态热传导 L2 收敛率 ≥2 阶"这一锚点主张。

参数表：k=1.0；ρ=1.0；c_p=1.0；thickness=1.0；dtype=torch.float64；Dirichlet 节点=全部边界节点
（4×4 网格 16 个边界节点，自由 dof 9；32×32 网格 124 边界节点，自由 dof 961）。

## ② 怎么做的（方法/流程/关键设置/工具链）

审计流程（不重做算例，只核验）：

1. **裁判行取证**：`tensor_benchmark_cases WHERE anchor_slug='CASE-FEM-HEAT-L2'` → **0 行**（全表 125 行）；
   按任务榜令"基准判据是唯一裁判"，该锚点**没有机器裁判行**，其容差（finest L2<3%、mean order≥1.8）只存在于
   交付物自带的 `gates_verified.json`。同表可比锚点如 `CASE-FEM-CLAMPED-PLATE-UNIFORM`（<3%）才是有裁判的形态。
2. **ref 通道核验**：交付报告声明 ref = 解析制造解（程序化求值，非手抄），无 DB ref → ref 可用。
3. **got 通道独立复算**：审计自写驱动 `verify_heat_l2.py`，**不使用**交付物 `run.py`，仅 import 被测模块
   `tensorfem.heat_conduction_l2._solve_level`，自行重算误差度量；结果与交付 `result.json` 逐级对比（见 ③ 列"偏差"）。
4. **度量反事实**：对同一批 FEM 解，另算"仅自由 dof 的节点 RMS"（剔除边界精确节点造成的零误差稀释），
   并把梯子延长到 64×64（本地），检验观测阶是否真趋于理论值 2。
5. **交付面核验**：`git status` / `git log main..agent/m20143` / `git for-each-ref` / `git ls-tree -r HEAD`；
   `tensor_reports WHERE mission_id=20143`；`tensor_maturity_verdicts WHERE mission_id=20143`（审计前 0 行）。
6. **测试核验**：`PYTHONPATH=src pytest tests/test_heat_conduction_l2.py -q`（8 passed / 6.39 s），并逐条读断言，
   区分"断言不变量"与"仅执行代码"。
7. **交叉锚点**：同锚点另一交付 m20054（commit `8867c2a4`，branch `agent/m20054`，已判 L4，verdict id=828）
   使用独立 Gauss 单元积分 L2 口径（`src/tensorfem/heat_l2_convergence.py:112-113,233`），finest 阶 1.998–2.000。

关键设置：本地 `OMP_NUM_THREADS=4`，torch.float64 直接解（确定性方程），因此复算应 bit 级一致；128×128 级别本地超时，
已递交计算门（作业 cj2563），未在窗口内回收则记 not_run。

## ③ 计算结果（数字表，程序化注入）

| 网格 | 自由 dof | h | L2_rel（m20143 口径：全节点 RMS） | L2_rel（审计口径：仅自由 dof） | 与 result.json 相对偏差 | 解算残差 max&#124;K_ff T_f−f_f&#124; |
|---|---|---|---|---|---|---|
{rows}

观测阶（全节点口径，m20143 原口径）：{pc}，均值 **{mean_order:.4f}**
观测阶（仅自由 dof 口径，审计）：{pf}

对照：m20143 交付 `result.json` 声明 {V["orders"]["declared_orders_all_nodes"][0]:.6f} / {V["orders"]["declared_orders_all_nodes"][1]:.6f} / {V["orders"]["declared_orders_all_nodes"][2]:.6f}，
均值 {mean_order:.6f} —— 审计复算逐级相对偏差 ≤1.2e-16，**数值没有编造/硬编码**。

## ④ 结果图

图 1：复现的收敛曲线（log-log，全节点口径 vs 仅自由 dof 口径，附 h² 参考斜率）与逐对观测阶（含 1.8 自定门限与 2 阶锚点要求）。
图 2：主张 vs 可复现证据（交付均值 1.876、交付最细对 1.935、审计 64×64 自由 dof 2.038、m20054 Gauss 积分 2.000 四个数并排）。

## ⑤ 注意点（踩过的坑/局限/适用范围）

- **度量口径是本案的根因**：把所有节点的 RMS 当 L2 范数时，精确强加 Dirichlet 的边界节点误差恒为 0，
  而边界节点占比随 h 变化（4×4：16/25，32×32：124/1089），稀释比例随网格改变 → 观测阶被系统性压低到 2 以下；
  剔除边界节点后同一批解给出 2.392/2.173/2.080/2.038，从上方收敛到 2。故"1.876 < 2"是**度量缺陷**而非 Q4 核缺陷。
- **门 3 是同义反复**：`heat_balance_residual` 就是线性解算残差 max|K_ff T_f − f_f|，任何直接解都 ~1e-15，
  不构成独立守恒校验（不是净热流积分闭合）。测试也只断言了这个恒真量。
- **门限事后放宽**：锚点要求 2 阶，交付代码默认 `order_threshold=1.8`（`heat_conduction_l2.py`），gates 文件同值。
- **报告方法学描述失真**：§② 称"Richardson 外推计算观测阶"，实际是 log-log 斜率（`heat_conduction_l2.py:157-163`）。
- **交付面缺失**：实现/测试/工件/报告全部为工作树未提交文件；`agent/m20143` tip == `main@f634c7c5`，
  `git log --all -- src/tensorfem/heat_conduction_l2.py` 为空，`git ls-tree -r HEAD | grep -ci heat` = 0；
  `tensor_reports` 无 mission_id=20143 行（报告未发布）；审计前 `tensor_maturity_verdicts` 亦无 20143 行。
- **重复劳动**：同锚点已由 m20054（`8867c2a4`）交付并经 m20108 判 L4（verdict id=828），口径正确（Gauss 积分）、
  数值 1.998–2.000。m20143 是同一锚点的平行实现且更弱，应去重而非双重记账。
- 适用范围局限：仅 Cartesian 均匀网格、无源项、常物性；剪切/曲线网格与源项牵引未覆盖（交付报告 §⑤ 亦自承）。

## ⑥ 误差判定（关键量 vs 基准逐项，显式回答是否 <3%）

| 关键量 | 基准（解析） | got | 相对误差 | 是否 <3% |
|---|---|---|---|---|
| finest（32×32）L2 相对误差 | 0（与解析解之差） | {lv["32x32"]["l2_relative_all_nodes"]:.6e} | 4.62e-4（相对误差本身） | **是**（远小于 3%） |
| 观测阶（均值，与理论 2.0 比） | 2.0 | {mean_order:.4f} | **{abs(2.0-mean_order)/2.0*100:.2f}%** | **否（>3%）** |
| 观测阶（最细对 16→32，与 2.0 比） | 2.0 | {orders_all[2]:.4f} | {abs(2.0-orders_all[2])/2.0*100:.2f}% | **否（>3%）** |
| 观测阶（全节点口径，含 64×64 对） | 2.0 | {orders_all[3]:.4f} | {abs(2.0-orders_all[3])/2.0*100:.2f}% | **否（>3%）** |
| 观测阶（自由 dof 口径，最细对） | 2.0 | {orders_free[3]:.4f} | {abs(2.0-orders_free[3])/2.0*100:.2f}% | **是（<3%）** |
| 数值可复现性（got vs 交付 result.json） | 交付值 | 审计复算 | ≤1.2e-16 | **是** |

**显式回答**：交付的"L2 收敛率 ≥2 阶"主张，按其自身口径与数据**不满足 <3% 判据**（均值偏低 {abs(2.0-mean_order)/2.0*100:.1f}%，
最细对偏低 {abs(2.0-orders_all[2])/2.0*100:.1f}%）；改用正确度量（仅自由 dof）后偏差 {abs(2.0-orders_free[3])/2.0*100:.2f}% <3%，
结论是**度量缺陷**而非求解器缺陷。交付数值本身真实可复现（偏差 1e-16），但**其结论表述与门限设置不成立**。

## 判定与处置

- **maturity verdict（已落表）**：`tensor_maturity_verdicts` id=**851**，repo=TensorFEM，scope='m20143 CASE-FEM-HEAT-L2'，
  mission_id=20186，**level=0**，verdict_by='board-audit'。
  level 依据：m20143 对仓库的实质贡献为 0（无 ref、无报告行、无裁判行、分支 tip 与 main 同点）；
  工作树级实现虽为真（可复现，L2 级代码），但从未入库；即便入库上限也只是 L2（无裁判行→无 L4，无 caller→无 L3），
  且"≥2 阶"头条与其自身数据冲突。
- **处置 = reopened（返修）**：① 补发 `tensor_reports`（mission 20143）；② 若仍主张该锚点，改用单元积分/仅自由 dof 的
  L2 口径并提交 ref；③ 否则明确作废并与 m20054 去重；④ 平台侧欠账：`tensor_benchmark_cases` 需补 `CASE-FEM-HEAT-L2` 裁判行，
  否则任何该锚点的 L4 主张都没有机器裁判。
"""

report = {
    "title": "m20186 交差核验：m20143 CASE-FEM-HEAT-L2（未入库平行实现 + L2 度量口径缺陷，level=0/reopened）",
    "summary": ("m20143 的 CASE-FEM-HEAT-L2 实现/工件/报告仅存在于未提交工作树（无 ref、报告未发布、无裁判行），"
                "且其 \"≥2 阶\" 头条与其自身数据冲突（均值 1.876<2，门限自降到 1.8）：根因是 L2 范数把精确边界节点算进 RMS，"
                "仅自由 dof 口径下同一批解给出 2.038（→2）。判 level=0、reopened，锚点能力归 m20054（8867c2a4，L4/verdict 828）。"),
    "scenario_key": "CASE-FEM-HEAT-L2",
    "mission_id": 20186,
    "tags": ["TensorFEM", "board-audit", "CASE-FEM-HEAT-L2", "m20143", "convergence", "MMS", "L2-norm", "maturity"],
    "markdown": md,
    "figures": [
        {
            "name": "fig_m20186_convergence.png",
            "caption": "CASE-FEM-HEAT-L2 复现收敛曲线：红线=交付口径（全节点 RMS，含精确 Dirichlet 边界），"
                       "蓝线=审计口径（仅自由 dof），虚线=h² 参考斜率；右图为逐对观测阶与 1.8 自定门限/2 阶锚点要求。",
            "png_base64": base64.b64encode((HERE / "fig_m20186_convergence.png").read_bytes()).decode(),
        },
        {
            "name": "fig_m20186_claim.png",
            "caption": "主张 vs 可复现证据：交付均值 1.876、交付最细对 1.935、审计 64×64 自由 dof 2.038、"
                       "同锚点 m20054 Gauss 积分 L2 2.000；黑虚线=锚点 2 阶要求，橙点线=交付自定门限 1.8。",
            "png_base64": base64.b64encode((HERE / "fig_m20186_claim.png").read_bytes()).decode(),
        },
    ],
}

out = Path("/tmp/report_m20186.json")
out.write_text(json.dumps(report, ensure_ascii=False))
print("report bytes:", out.stat().st_size)

base = os.environ["TENSOR_API_BASE"]
tok = subprocess.run(
    ["curl", "-s", "-X", "POST", f"{base}/auth/login",
     "-H", "Content-Type: application/json",
     "-d", json.dumps({"username": os.environ["TENSOR_AGENT_USER"],
                       "password": os.environ["TENSOR_AGENT_PASS"]})],
    capture_output=True, text=True).stdout
token = json.loads(tok).get("access_token") or json.loads(tok).get("token")
print("token ok:", bool(token))
r = subprocess.run(
    ["curl", "-s", "-X", "POST", f"{base}/tensor/reports",
     "-H", f"Authorization: Bearer {token}", "-H", "Content-Type: application/json",
     "-d", f"@{out}"], capture_output=True, text=True)
print(r.stdout[:600])