#!/usr/bin/env python3
"""Render ``docs/CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md`` from the three
study artifacts.  Every number in the report is injected from JSON here, so the
document cannot drift from the data by hand-copying."""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOC = os.path.join(HERE, "..", "..", "..", "docs",
                   "CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md")
FIG = "docs/assets/case-fem-hertz-contact"


def load(name):
    with open(os.path.join(HERE, name)) as fh:
        return json.load(fh)


def pct(v, nd=4):
    return "n/a" if v is None else f"{v:+.{nd}f}%"


def mark(ok):
    return "✅ 是" if ok else "❌ 否"


def main():
    far = load("farfield_discretisation.json")
    zres = load("nearfield_resolution.json")
    conv = load("converged_model.json")
    cert = load("result.json")

    limit = 3.0
    arms = zres["arms"]
    a48 = conv.get("vertical_ladder_a48")
    zrows = conv["depth_ladder_dz_a24"]
    ext = conv["extrapolation"]
    prior = conv["prior_extrapolated_bias_pct"]
    share = conv["share_of_prior_bias_explained_pct"]
    e_inf = ext["e_inf_pct"]
    dv = zres["arm_verdict"]

    # --- consolidated arm table (vertical ladder incl. the a/48 point) --------
    vert_rows = [a for a in arms]
    if a48:
        vert_rows = vert_rows + [a48]
    vt = ["| 臂 | $n_{z,\\rm fine}$ | $\\Delta z/a$ | $\\Delta r/a$ | 单元数 | $e_a$ | $e_\\delta$ | $e_{p_0}$ | 墙钟 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for a in vert_rows:
        vt.append(
            f"| {a['label']} | {a['n_zfine']} | {a['near_field_dz_over_a']:.4f} | "
            f"{a['near_field_dr_over_a']:.4f} | {a['n_elements']} | "
            f"{pct(a['error_pct']['contact_radius_linear'])} | "
            f"{pct(a['error_pct']['indentation'])} | "
            f"{pct(a['error_pct']['peak_pressure_consistent'])} | {a['wall_s']:.0f} s |")

    ft = ["| 臂 | 远场层数（径/竖） | 最大竖向远场单元 $/a$ | $e_a$ | $e_\\delta$ | $e_{p_0}$ |",
          "|---|---:|---:|---:|---:|---:|"]
    for r in far["farfield_ladder"]:
        ft.append(f"| far({r['n_outer']},{r['n_grow']}) | {r['n_outer']}/{r['n_grow']} | "
                  f"{r['max_far_vertical_element']:.1f} | "
                  f"{pct(r['error_pct']['contact_radius_linear'])} | "
                  f"{pct(r['error_pct']['indentation'])} | "
                  f"{pct(r['error_pct']['peak_pressure_consistent'])} |")

    dt = ["| 域深 $/a$ | $e_\\delta$（$\\Delta z=a/24$） | $e_a$ | $e_{p_0}$ |",
          "|---:|---:|---:|---:|"]
    for r in zrows:
        dt.append(f"| {r['depth_over_a']:.0f} | {pct(r['error_pct']['indentation'])} | "
                  f"{pct(r['error_pct']['contact_radius_linear'])} | "
                  f"{pct(r['error_pct']['peak_pressure_consistent'])} |")

    crow = cert["runs"]["ratio0.05_h0.0075_d96"]
    rrow = cert["runs"]["ratio0.10_h0.0075_d96"]
    a_ok = abs(crow["error_pct"]["contact_radius_linear"]) <= limit
    d_ok = abs(crow["error_pct"]["indentation"]) <= limit
    r_a_ok = abs(rrow["error_pct"]["contact_radius_linear"]) <= limit
    r_d_ok = abs(rrow["error_pct"]["indentation"]) <= limit
    p_ok = abs(crow["error_pct"]["peak_pressure_consistent"]) <= 5.0

    e_a24 = arms[2]["error_pct"]
    verdict_txt = {
        "H_zres": "**H_zres 被接受**：残余偏移来自竖向近场分辨率",
        "H_bias": "**H_bias 被接受**：竖向加密不改变偏移，偏移属接触模型自身",
        "inconclusive": "**两臂都不满足预声明阈值（inconclusive）**：竖向加密只解释了一部分偏移，"
                        "剩余部分在本单未归因——如实记为开放项",
    }[dv["accepted"]]

    text = f"""# CASE-FEM-HERTZ-CONTACT — 压入深平台（−1.01%）的机制归因：**只加密了 r，从未加密 z**

【锚点=CASE-FEM-HERTZ-CONTACT】｜仓=TensorFEM｜证据分级：REAL（解析基准 + 可离线复算闸门 + 预声明判据）
| 前序：`docs/CASE-FEM-HERTZ-CONTACT-verification.md`（证书，14 闸门）、
`docs/CASE-FEM-HERTZ-CONTACT-truncation-extrapolation.md`（截断外推，残余 −0.4278% 记为
"irreducible model bias"，其 `limitations` 明写 **机制未定位**）

## ① 问题 + ref 出处

算例池 `FEM-HERTZ-CONTACT`（id 104）：刚性抛物型（球）压头压入弹性半空间，**载荷 P 给定**，
判据 `<3%`，指标接触半径 `a` 与压入深 `δ`。

解析基准（ref）：Hertz 1882, *J. reine angew. Math.* **92**:156；Johnson 1985 *Contact Mechanics*
§4.2, DOI:10.1017/cbo9781139171731：
`a = (3PR/(4E*))^{{1/3}}`、`δ = a²/R`、`p₀ = 3P/(2πa²)`，`E* = E/(1−ν²)`（刚性压头/弹性半空间）。

**本单要回答的问题**（前序两单留下的唯一未解释项）：载荷控制求解在有限域每加深一级、误差按幂律衰减，
外推到"无限域"后仍留下 **−0.4278%** 的压入深误差。前序把它记为"网格无关的模型偏差"，但**没有指出机制**。
本次给出三个互斥假说并逐一定量检验：

| 假说 | 内容 | 预言 |
|---|---|---|
| H_far | 远场是**拉伸**（径向 26 层 / 竖向 16 层几何级数铺到 768a）而不是加密，双线性 Q4 层厚达数百 a 无法表达 1/z 衰减场 | 只加密远场，$|e_\\delta|$ 显著下降 |
| **H_zres** | 判据阶梯 `h/a = 0.03/0.015/0.0075` **只加密径向**；竖向细带固定 24 层即 `Δz = a/6`，故"网格无关"从未被真正检验 | 只加密竖向近场（径向逐位不变），$|e_\\delta|$ 下降 |
| H_bias | 偏移来自接触离散本身（节点乘子面片、抛物面弦、ALM 容差），对任何加密不变 | 两臂都动不了 $|e_\\delta|$ |

## ② 怎么做（方法/流程/关键设置）

三臂独立、每次只动一个变量，近场/远场**逐位复用**同一网格构造函数（`run.py::run_case` 的 `two_zone_axis`
+ `graded_depth`，本单把 `n_outer`、`n_grow`、`n_zfine` 三个层数参数暴露出来）：

1. **基准臂（farfield_discretisation.py）**——固定深度 768a、`h/a=0.015`、`a/R=0.05`，只把远场层数从
   (26,16) 提到 (52,32)、(104,64)；再把深度阶梯 96a→192a→384a→768a 用**加密后的远场**重跑并做
   Richardson 外推（仓内 `_harness.richardson`，与截断单同一个算术）。
2. **竖向臂（nearfield_resolution.py）**——固定深度 768a、径向近场逐位不变（`Δr = a/100`，即 `h/a=0.015`），
   只把竖向细带层数 24 → 48 → 96（`Δz = a/6 → a/12 → a/24`），外加一档"竖向加密 + 远场加密"。
3. **收敛臂（converged_model.py）**——把竖向阶梯推到 `Δz = a/48`，并在 `Δz = a/24` 下重跑深度阶梯、
   外推到无限域，回答"偏移是否随加密消失"。

阈值**先于运行写死在脚本 `CRITERIA` 里**（未在看过数后调整）：
H_far 成立需 `|e_δ| ≤ 0.15%`（且外推极限 `≤ 0.10%`）；H_bias 成立需远场/竖向加密把 `e_δ` 移动
`≤ 0.05 pp`；算例池判据 3%。每次求解都把完整行追加进 `*_rows.jsonl`，工件可用 `--from-jsonl` 零算力重建。

## ③ 参数表（无量纲，完整）

| 项 | 值 |
|---|---|
| 载荷点 / 目标接触半径 | `a/R = 0.05`（主），`P = 4E*a³/(3R) = {crow['load']:.4f}` |
| 弹性常数 | `ν = 0.3`，`E* = E/(1−ν²)`；长度 /R、力 /(E R²)（SI 等价 R=10 mm、E=100 GPa） |
| 径向近场间距 | `Δr/a = {arms[0]['near_field_dr_over_a']:.4f}`（`h/a = 0.015`，接触带 `r ≤ 3a` 均匀） |
| 竖向近场间距（被检验变量） | `Δz/a = {arms[0]['near_field_dz_over_a']:.4f} → {arms[1]['near_field_dz_over_a']:.4f} → {arms[2]['near_field_dz_over_a']:.4f}`{f" → {a48['near_field_dz_over_a']:.4f}" if a48 else ""} |
| 细带厚度（不变） | `4a`（其下几何增长铺到域底） |
| 远场层数（径/竖） | 基准 (26,16)；加密 (52,32)、(104,64) |
| 域深 / 外径 | 96a / 192a / 384a / 768a（外径同时等于域深） |
| 接触算法 | Alart–Curnier / Simo–Laursen 增广 Lagrange，`ρ/(E/a)=1`，`alm_tol=1e-10`，载荷控制 regula falsi `load_tol=1e-10` |
| 收敛质量（各臂内） | 载荷残差 `≤ {max(r['load_residual'] for r in zrows):.1e}`、穿透/δ `≤ {max(r['penetration_over_delta'] for r in zrows):.1e}`、平衡残差 `≤ {max(r['equilibrium_residual_rel'] for r in zrows):.1e}` |

## ④ 对照表（ref vs got，误差打勾叉）

**竖向臂（深度 768a，径向逐位不变）**——ref 为 Hertz 解析：

{chr(10).join(vt)}

**远场臂（深度 768a）**：

{chr(10).join(ft)}

**深度臂（$\\Delta z=a/24$，远场 (52,32)）**：

{chr(10).join(dt)}

**算例池判据（前序证书 `result.json`，模型即 `Δz=a/6`、深度 96a，未被本单改动）**：

| 关键量 | ref | got | 误差 | <3%? |
|---|---|---:|---:|---|
| `a`（a/R=0.05, h/a=0.0075） | `(3PR/4E*)^{{1/3}}` | {crow['numeric']['contact_radius_linear']:.6f} | {pct(crow['error_pct']['contact_radius_linear'])} | {mark(a_ok)} |
| `δ`（同上） | `a²/R` | {crow['numeric']['indentation']:.6f} | {pct(crow['error_pct']['indentation'])} | {mark(d_ok)} |
| `a`（a/R=0.10, h/a=0.0075） | 同上 | {rrow['numeric']['contact_radius_linear']:.6f} | {pct(rrow['error_pct']['contact_radius_linear'])} | {mark(r_a_ok)} |
| `δ`（同上） | 同上 | {rrow['numeric']['indentation']:.6f} | {pct(rrow['error_pct']['indentation'])} | {mark(r_d_ok)} |
| `p₀`（a/R=0.05, 一致压强场） | `3P/(2πa²)` | {crow['numeric']['peak_pressure_consistent']:.3f} | {pct(crow['error_pct']['peak_pressure_consistent'])} | {mark(p_ok)}（判据 5%） |

## ⑤ 收敛性 / 参数敏感性（至少两点）

**(1) 远场敏感性——H_far 被否定。** 远场层数 16 → 64（最粗的竖向远场单元
{far['farfield_ladder'][0]['max_far_vertical_element']:.0f}a → {far['farfield_ladder'][2]['max_far_vertical_element']:.0f}a），
`e_δ` 仅从 {pct(far['farfield_ladder'][0]['error_pct']['indentation'])} 变到
{pct(far['farfield_ladder'][2]['error_pct']['indentation'])}
（移动 {far['arm_verdict']['shift_pp']:.4f} pp ≤ 0.05 pp 判据）；用加密远场重做深度阶梯，外推极限
{("n/a" if far["extrapolation_refined"].get("e_inf_pct") is None else pct(far["extrapolation_refined"]["e_inf_pct"]))}
对比前序 {pct(-prior)}。**远场拉伸不是原因。**

**(2) 竖向分辨率敏感性——两个数量级的差别。** `Δz = a/6 → a/24` 使
`e_δ: {pct(arms[0]['error_pct']['indentation'])} → {pct(arms[2]['error_pct']['indentation'])}`，
而**径向**阶梯 `h/a=0.03→0.0075` 在 `Δz=a/6` 下几乎不动
（{pct(min(r['error_pct']['indentation'] for r in cert['runs'].values() if abs(r['a_over_R']-0.05)<1e-9 and abs(r['domain_depth_over_a']-96.0)<1e-9))} …
{pct(max(r['error_pct']['indentation'] for r in cert['runs'].values() if abs(r['a_over_R']-0.05)<1e-9 and abs(r['domain_depth_over_a']-96.0)<1e-9))}）。
这解释了"平台"为何看起来网格无关：**判据阶梯只动了一个方向**。
副产物：一致峰值压强误差 `{pct(arms[0]['error_pct']['peak_pressure_consistent'])} → {pct(arms[2]['error_pct']['peak_pressure_consistent'])}`
（改善 {abs(arms[0]['error_pct']['peak_pressure_consistent']/arms[2]['error_pct']['peak_pressure_consistent']):.0f}×）；
接触半径估计量随 `Δz` 变号（{pct(arms[0]['error_pct']['contact_radius_linear'])} → {pct(arms[2]['error_pct']['contact_radius_linear'])}），
说明边缘零点估计量的噪声约 ±1 pp，`a` 的精度不是被求解精度而是被估计量限制。

**(3) 深度外推（$\\Delta z=a/24$）。** 阶梯 {", ".join(pct(r['error_pct']['indentation']) for r in zrows)}
→ Richardson 极限 **{pct(e_inf)}**（观测阶 {ext['richardson'].get('observed_order'):.3f}）。前序
`Δz=a/6` 的极限为 {pct(-prior)}；本单竖向加密解释了其中
**{share:.1f}%**，剩余 {pct(e_inf)} 仍是**开放项**。

**判决：** {verdict_txt}。

## ⑥ 失败模式与工件

* **F1（本单结论，量化）** "网格无关的模型偏差"这一表述的前序证据链不成立：判据阶梯只加密径向，
  竖向细带被钉死在 `Δz=a/6`。竖向单轴加密即可移动 `e_δ` {dv['shift_pp']:.3f} pp。
* **F2（诚实 FAIL）** 预声明阈值 `|e_δ| ≤ 0.15%` **未达成**：竖向推到 `Δz=a/48` 后
  `e_δ = {pct(a48['error_pct']['indentation']) if a48 else 'n/a'}`，深度外推后仍有 {pct(e_inf)}。
  故剩余 {pct(e_inf)} 的**机制未定位**（候选：Q4 双线性在近表面陡梯度层的过刚、节点乘子面片、
  ALM 容差）。report 中不把它写成"收敛"。
* **F3（估计量，非求解器）** 接触半径的边缘零点估计量对 `Δz` 敏感（符号翻转），`a` 的误差
  {pct(arms[0]['error_pct']['contact_radius_linear'])} → {pct(e_a24['contact_radius_linear'])}；
  算例池 `a` 的判定仍由前序证书的多估计量交叉覆盖，本单不改判据。
* **生态摩擦**：`run_shell` 单次调用有 120 s 硬超时并按进程组收割，`nohup` 子进程会被一起杀掉，
  必须 `setsid` + 行级 JSONL 增量落盘（本单两次因写入失败/进程组收割丢掉已完成解，靠 JSONL 与日志
  复原；脚本已内置 `--from-jsonl` 零算力重建，避免重复算力）。

**工件**（全部在 `benchmarks/verified/fem_hertz_contact_loadcontrol/`）：
`farfield_discretisation.py|json`、`nearfield_resolution.py|json`、`converged_model.py|json`、
`*_rows.jsonl`（每次求解一行，可零算力重建）、`plateau_figures.py`、
`docs/assets/case-fem-hertz-contact/hertz_plateau_mechanism.png|hertz_plateau_arms.png|hertz_plateau_depth.png`。

## 误差判定（逐项回答"是否 <3%"）

| 关键量 | 基准 | 误差 | <3%? |
|---|---|---:|---|
| `a`，算例池接受模型 | Hertz `a=(3PR/4E*)^{{1/3}}` | {pct(crow['error_pct']['contact_radius_linear'])} | {mark(a_ok)} |
| `δ`，算例池接受模型 | Hertz `δ=a²/R` | {pct(crow['error_pct']['indentation'])} | {mark(d_ok)} |
| `a`，a/R=0.10 接受模型 | 同上 | {pct(rrow['error_pct']['contact_radius_linear'])} | {mark(r_a_ok)} |
| `δ`，a/R=0.10 接受模型 | 同上 | {pct(rrow['error_pct']['indentation'])} | {mark(r_d_ok)} |
| `δ`，竖向加密 + 深度外推（本单最优） | 同上 | {pct(e_inf)} | {mark(abs(e_inf) <= limit if e_inf is not None else False)} |
| `p₀`，竖向加密（`Δz=a/24`） | `3P/(2πa²)` | {pct(arms[2]['error_pct']['peak_pressure_consistent'])} | {mark(abs(arms[2]['error_pct']['peak_pressure_consistent']) <= 5.0)}（判据 5%） |
"""
    with open(DOC, "w") as fh:
        fh.write(text)
    print(f"wrote {DOC} ({len(text)} chars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())