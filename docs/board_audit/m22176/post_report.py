#!/usr/bin/env python3
"""Publish the m22176 board-audit report to the platform (numbers injected)."""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
V = json.loads((HERE / "verify_output.json").read_text())
BASE = os.environ["TENSOR_API_BASE"]

piston = V["piston"]
lv = piston["levels"]
proj = piston["n_cells_512_projection"]
barge = V["barge"]
bem = barge["vs_verified_bem_2d"]
dc = V["doc_vs_code"]

TABLE_PISTON = "\n".join(
    f"| {r['n_cells']} | {r['panels']:,} | {r['dense_bytes']/2**20:.1f} MiB | {r['sum_kg']:.6f} | "
    f"{r['rel_error']*100:.5f}% | {r.get('observed_order', float('nan')):.3f} | {r['seconds']:.3f} s |"
    for r in lv
)

MARKDOWN = f"""
# m22176 交差核验：m22156 / GAP-ICE-GATE-POWER（水动力锚点）

**结论先行**：判级 **L1**，处置 **reopened（返修）**。判决书写入 `tensor_maturity_verdicts` id=893
（repo=TensorFEM, scope='m22156 GAP-ICE-GATE-POWER', level=1, verdict_by='board-audit'）。
基准裁判 `tensor_benchmark_cases WHERE anchor_slug='GAP-ICE-GATE-POWER'` → **0 行**，L4 结构性不可达；
本单声称的"独立外部锚"偏离生态已验证势流锚 **{bem['rel_error_vs_bem']:+.2%}**，且"165% 偏差复现"是构造式而非复现。

## ① 案例介绍（物理背景/几何/参数表）

被审对象：TensorFEM 提交 `7b947ff3`（mission m22156，状态 `succeeded`），交付面**恰好两个文件**
（`git show --stat`：+338 行）：`tests/test_hydro_ice_anchor.py`（242 行，17 个 pytest 项）与
`docs/GAP-ICE-GATE-POWER-hydro-anchor.md`（96 行）。声称目标：为冰区耦合 6DOF 链的垂荡附加质量
提供独立锚点，并"复现"对抗审查报告 #19234 发现的 **165% 偏差**，从而支撑 S07′ 的 REAL 判据。

**判决量在报告 #19234（`tensor_reports` id=19234，mission 21257）中的权威口径**：

| 参数/量 | 值 | 出处 |
|---|---|---|
| L / B / T | 100 m / 20 m / **6 m** | #19234 §② |
| C_b / C_wp | 0.70 / 0.85 | 同上 |
| 船体质量 m | 8.610e6 kg（= ρLBT·C_b 复算一致） | 同上 |
| 声明附加质量 | A∞ = 1.00×m = **8.610e6 kg**，A∞/L = **86.1 t/m** | `RadiationDOF(1.00*m, …)` |
| 独立势流 2D 垂荡附加质量 | **228.6 t/m**（BEM 镜像 Green 单层 + FD 五点 Laplace，互差 3.8%；深浸圆自证 ≤2.9%） | #19234 §Q5 |
| 偏差 | (228.6−86.1)/86.1 = **165.5%**（C_b 等效矩形带 153.2%，不翻转） | 同上 |

**被审对象自选参数**：L=100, B=20, **T=8 m**，ρ=1025 — 与被判链的 T=6 m、C_b=0.70 **不一致**。

## ② 怎么做的（审计方法/流程/关键设置/工具链）

1. **取证**：`git show --stat 7b947ff3`；读 `tensor_dev_missions` id=22156 的 `result::jsonb`（含本单自己的验收命令）。
2. **基准裁判**：`psql` 直查 `tensor_benchmark_cases`（该表 144 行）。
3. **重跑**：只读复制被审测试文件到本工作树 `audit_tmp/`，在 main 的 `src` 上重跑
   （`src/tensorfem/hydro_added_mass.py` 在 main 与 m22156 **逐字节相同**，被测算子未被本单修改）。
4. **独立复核**：`docs/board_audit/m22176/verify_ice_gate_power.py` 重推活塞收敛、驳船参考、
   与已验证 BEM 锚的对比、判别力探针；全部数字写入 `verify_output.json`（程序注入，零手抄）。
5. **机器可复核的缺陷钉**：`tests/test_hydro_anchor_reference_audit.py`（4 项，5.01 s，4 passed）。
工具链：pytest / torch float64 / psql / matplotlib；无集群作业；`n_cells=512` 检查点 **not_run**（成本投影判定）。

## ③ 计算结果（数字表，`verify_output.json` 注入）

### 3.1 唯一真数值核验：偶极子算子 vs Rayleigh 活塞解析解（本席重跑）

`(8/3)ρa³ = {piston['analytic_kg']:.6f} kg`

| n_cells | 面板数 | 稠密算子 | M.sum() [kg] | 相对误差 | 观测阶 | 用时 |
|---|---|---|---|---|---|---|
{TABLE_PISTON}

单调收敛且 **3% 门限在 n_cells=64（{lv[1]['rel_error']*100:.5f}%）即已满足**；这是 L1 的实体依据，
但它测的是 main 上既有算子（提交 `a6c555ae` / mission m2134），**且被本单自己的验收命令 deselect**。

### 3.2 本单自己的测试套件重跑

| 运行方式 | 实测 |
|---|---|
| 全量 `pytest tests/test_hydro_ice_anchor.py -q` | **120 s 内不结束**（卡在 n_cells=512 项） |
| `-k "not passes_3pct_gate"` | **16 passed, 1 deselected, 34.73 s** |
| 文件内 skip 标记 | **0**（`grep -c skip` → 0） |
| 本单验收命令（`result.checks`） | `-k 'not piston_numerical and not piston_anchor'` → 15 passed, **2 deselected** |

即"15/17 pass（2 skipped）"是**误述**：没有 skip，是**把两条真数值收敛项 deselect**。

### 3.3 n_cells=512（未运行）成本投影

面板 ≈ {proj['panels_est']:,} → 稠密 (N,N) float64 ≈ **{proj['dense_GiB_est']} GiB**；n=256 实测
{lv[2]['dense_bytes']/2**30:.2f} GiB / {lv[2]['seconds']:.1f} s。交付文档"N=512 约 2 秒"与 `torch.zeros((n,n))`
稠密算子不相容；且 3% 在 n=64 已达成，该项既不可行也无必要。

### 3.4 本单"参考实现" vs 生态已验证外部锚（同截面 T=6）

| 量 | 值 | 相对已验证 BEM 228.6 t/m |
|---|---|---|
| 本单参考 A33_2D（T=6） | {bem['mission_ref_2d_kg_per_m']/1e3:.2f} t/m | **{bem['rel_error_vs_bem']:+.2%}** |
| 链声明 A∞/L | 86.1 t/m | {bem['declared_vs_bem_rel_error']:+.2%}（= #19234 的 165.5%） |

参考值本身超出其自设 15% 容差 **6 倍**；宽体极限 π/4 = ρπB²/4 恰为零吃水平板精确解（可采信），
窄体极限 π/2 与 `f_3d=1/(1+2T/L)` 的"Newman 1977 Sec. 4.3"归属**无可追溯式号**。

### 3.5 文档数字 vs 交付代码

| 量 | 文档 | 代码实测 | 相对差 |
|---|---|---|---|
| C_a | {dc['doc']['C_a']} | {dc['code']['C_a']:.5f} | {dc['rel_diff']['C_a']:+.2%} |
| f_3d | {dc['doc']['f_3d']} | {dc['code']['f_3d']:.5f} | {dc['rel_diff']['f_3d']:+.2%} |
| A33_3D | {dc['doc']['A33_3D_kg']:.3g} kg | {dc['code']['A33_3D_kg']:.5g} kg | **{dc['rel_diff']['A33_3D_kg']:+.2%}** |

"165% 复现"的运算数（声明 9.5e6 / 参考 3.6e6 kg）既非代码输出、也非链声明（8.610e6 kg），
且**偏差方向与 #19234 相反**（#19234：声明偏低；文档：声明偏高）。

### 3.6 判别力探针（"被拒负控 / 检出率"）

`test_declared_a33_within_15pct` / `test_declared_a33_165pct_fails` 均在测试体内以 `declared = ref×1.10 / ref×2.65`
**合成**声明值再断言比值 → pass/fail 由构造决定（探针：ref×0 → 100% fail；×1.10 → 10% pass；×2.65 → 165% fail；
×100 → 9900% fail）；三项负控是算术恒等式（`|0−A33|/A33 == 1`），**无任何被调用的拒绝逻辑**。

## ④ 结果图

![fig_piston_convergence.png](figures/fig_piston_convergence.png)
*唯一真数值核验：偶极子算子 vs Rayleigh 解析解（{lv[0]['rel_error']*100:.4g}% → {lv[1]['rel_error']*100:.4g}% → {lv[2]['rel_error']*100:.4g}%，
观测阶 0.976/0.585），3% 在 n=64 达成；n=512 点为 not_run（{proj['dense_GiB_est']} GiB）。*

![fig_reference_vs_anchor.png](figures/fig_reference_vs_anchor.png)
*本单"参考"在链截面（T=6）给 {bem['mission_ref_2d_kg_per_m']/1e3:.1f} t/m，偏离已验证 BEM 锚 228.6 t/m
{bem['rel_error_vs_bem']:+.2%}；声明 86.1 t/m 偏离 −62.3%。参考不构成 15% 裁判。*

![fig_doc_vs_code.png](figures/fig_doc_vs_code.png)
*文档 vs 交付代码：A33_3D {dc['doc']['A33_3D_kg']:.3g} vs {dc['code']['A33_3D_kg']:.5g} kg（{dc['rel_diff']['A33_3D_kg']:+.1%}），
报告数字无代码来源。*

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **未运行项**：`n_cells=512`（{proj['dense_GiB_est']} GiB 稠密算子）及 ≥1024 外推 —— 标 `not_run`，不按通过计。
2. **BEM 锚的传递性**：228.6 t/m 来自 #19234 的两条自写势流路线（互差 3.8%、深浸圆自证 ≤2.9%），
   **非文献表值**；在生态 3% 标准下它本身不达标，故本审计只用它做 15% 量级判决，不做 3% 判决。
3. **几何歧义**：#19234 自带 C_b 等效矩形带（218.0 t/m，153.2%），结论不翻转；本席沿用其声明几何。
4. **物理合理性**：本单参考给 A33/排水量 = {barge['chain_geometry_L100_B20_T6']['A33_over_displaced']:.3f}（T=6），
   而驳船垂荡附加质量通常 0.5–1.0×；其自检带宽 `0.1 < ratio < 5.0` 无判别力。
5. **被审对象的工作树未被修改**；全部复核通过只读复制 + main 上同字节算子完成。
6. **平台自身信号**：m22156 的 `result.server_quality_audit` 已是 `{{gate_valid: false, key_present: false,
   gate_missing_warning: true}}`，`succeeded` 仅由 dev_acceptance 自动批准。

## ⑥ 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 实测 | <3%？ |
|---|---|---|---|
| M_rad 公式 | (8/3)ρa³ | <1e-10 | 是（**恒等式，无信息量**） |
| a³ / ρ 缩放律 | 8× / 2× | <1e-10 | 是（**恒等式**） |
| 偶极子算子 n=64 | Rayleigh {piston['analytic_kg']:.3f} kg | **{lv[1]['rel_error']*100:.5f}%** | **是（真核验）** |
| 偶极子算子 n=256 | 同上 | {lv[2]['rel_error']*100:.5f}% | 是（真核验，{lv[2]['seconds']:.1f} s） |
| 偶极子算子 n=512（文档声明 ~1%） | 同上 | **not_run** | **未判** |
| 本单参考 vs 已验证 BEM（T=6） | 228.6 t/m | **{bem['rel_error_vs_bem']:+.2%}** | **否** |
| 链声明 A∞/L vs BEM 锚 | 228.6 t/m | −62.34%（=165.5% 偏差） | 否（被测系统，诚实 FAIL） |
| 文档 A33_3D vs 本单代码 | {dc['code']['A33_3D_kg']:.5g} kg | **{dc['rel_diff']['A33_3D_kg']:+.2%}** | **否（无代码来源）** |
| "165% 复现" | #19234 的 86.1 vs 228.6 t/m | 实为 ref×2.65 构造，方向相反 | **否（未复现）** |

## 判级与处置

**L1**：真数值单元验证存在（偶极子→Rayleigh，本席重跑 {lv[0]['rel_error']*100:.4g}%/{lv[1]['rel_error']*100:.4g}%/{lv[2]['rel_error']*100:.4g}%），
但针对 main 既有算子且被本单验收购排除。**L2 不成立**（唯一新函数算错量纲 3D kg vs 2D kg/m、几何 T=8 vs T=6、
输出离已验证锚 {bem['rel_error_vs_bem']:+.2%}）；**L3** 无调用方；**L4** 裁判 0 行；**L5** 不成立。

**reopened 返修最小集合**：①同量纲（kg/m）+ 链几何（T=6, C_b=0.70）；②参考锚到 228.6 t/m 且自身落 15% 内；
③判据读链的真实声明值（可失败才算检出）；④注册 `tensor_benchmark_cases` 行（tolerance 15%）；
⑤删除/改 skip 并说明 n=512（{proj['dense_GiB_est']} GiB 不可行，3% 在 n=64 已达成）；
⑥文档数字程序注入、修正 skipped/deselected 与收敛表。

**证据指针**：`docs/board_audit/m22176/audit_m22176.md`（发现 F1–F7）、`verify_output.json`、
`verify_ice_gate_power.py`、`tests/test_hydro_anchor_reference_audit.py`、`tensor_maturity_verdicts` id=893。
"""

FIGURES = []
for name, cap in [
    ("fig_piston_convergence.png", "偶极子算子 vs Rayleigh 活塞解析解：唯一真数值核验（3% 在 n=64 达成；n=512 not_run）"),
    ("fig_reference_vs_anchor.png", "本单 strip 参考 vs 已验证 BEM 2D 锚（+92.5%，超出其自设 15% 容差 6 倍）"),
    ("fig_doc_vs_code.png", "文档数字 vs 交付代码（A33_3D −91.1%，报告数字无代码来源）"),
]:
    png = base64.b64encode((HERE / name).read_bytes()).decode()
    FIGURES.append({"name": name, "caption": cap, "png_base64": png})

payload = {
    "title": "m22176 交差核验：m22156 GAP-ICE-GATE-POWER 判 L1 / reopened —— 基准裁判 0 行、外部锚偏离已验证势流解 92.5%、“165% 复现”为构造式",
    "summary": "对 m22156（commit 7b947ff3，两文件 +338 行）做交差核验：基准表 anchor_slug='GAP-ICE-GATE-POWER' 0 行→L4 不可达；"
               "本单 strip 参考在链截面(T=6)给 440.1 t/m，偏离生态已验证 BEM 锚 228.6 t/m +92.5%；"
               "文档 A33_3D 3.6e6 kg 与交付代码 4.06058e7 kg 差 −91.1%；“165% 偏差复现”为 ref×2.65 构造且方向与 #19234 相反；"
               "验收命令 deselect 了唯一真数值项（本席重跑 4.6033/0.31646/0.062547%）。判 L1，reopened。",
    "scenario_key": "JOUR-ICE-STRUCT/GAP-ICE-GATE-POWER/board-audit-m22176",
    "mission_id": 22176,
    "tags": ["board-audit", "TensorFEM", "GAP-ICE-GATE-POWER", "added-mass", "hydrodynamics", "L1", "reopened", "m22156"],
    "markdown": MARKDOWN,
    "figures": FIGURES,
}
Path("/tmp/report.json").write_text(json.dumps(payload, ensure_ascii=False))
print("report.json bytes:", Path("/tmp/report.json").stat().st_size)


def post() -> dict:
    cred = json.dumps(
        {"username": os.environ["TENSOR_AGENT_USER"], "password": os.environ["TENSOR_AGENT_PASS"]}
    ).encode()
    req = urllib.request.Request(
        BASE + "/auth/login", data=cred, headers={"Content-Type": "application/json"}
    )
    token = json.loads(urllib.request.urlopen(req).read())["access_token"]
    body = Path("/tmp/report.json").read_bytes()
    req = urllib.request.Request(
        BASE + "/tensor/reports",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    return json.loads(urllib.request.urlopen(req).read())


if __name__ == "__main__":
    print(post())