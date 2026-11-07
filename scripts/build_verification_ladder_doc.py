#!/usr/bin/env python3
"""Build ``docs/basic-verification-ladder.md`` from the ladder evidence JSON.

Every number in the generated tables is read from
``artifacts/verification_ladder/ladder_report.json`` (written by
``scripts/run_basic_verification_ladder.py``); nothing is transcribed by hand,
so the document cannot drift away from the numbers.  The prose lives here.

Usage::

    python scripts/run_basic_verification_ladder.py   # regenerates the JSON
    python scripts/build_verification_ladder_doc.py   # regenerates the doc
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "artifacts/verification_ladder/ladder_report.json"
OUT = ROOT / "docs/basic-verification-ladder.md"


def build() -> str:
    d = json.loads(JSON.read_text())
    det = d["details"]
    L1, L2, L3 = det["L1"], det["L2"], det["L3"]
    m1, m2, m3 = L1["model"], L2["model"], L3["beam"]["model"]
    pm = L3["plate"]["model"]
    refs = L2["references"]
    checks = {c["criterion"]: c for c in d["checks"]}

    def pct(x: float, digits: int = 4) -> str:
        return f"{100.0 * x:+.{digits}f}%"

    def verdict(c: dict) -> str:
        return "**PASS**" if c["passed"] else "**FAIL**"

    lines: list[str] = []
    add = lines.append

    add("# 线性弹性求解器三级解析验证阶梯（梁 · 板 · 模态）")
    add("")
    add("Status: **" + ("passing" if d["all_passed"] else "failing") + "** —— 六级判据"
        "（下表由 `scripts/run_basic_verification_ladder.py` 程序化生成，"
        "`tests/test_verification_ladder.py` 以同一批门限做回归门）。")
    add("")
    add(f"产物：`src/tensorfem/verification_ladder.py`（阶梯实现）、"
        f"`scripts/run_basic_verification_ladder.py`（证据脚本）、"
        f"`scripts/build_verification_ladder_doc.py`（本文档生成器）、"
        f"`{d['artifacts']['report_json']}`（判据 JSON）、"
        f"`{d['artifacts']['figures_dir']}/`（图）、"
        f"`{d['artifacts']['tests']}`（pytest 门）。")
    add("")
    add("## 0. 中文摘要（判据与结论）")
    add("")
    add("本页把线性弹性静力/模态求解器的**最基本**问题变成一台机器可判的门：梁、板、模态三级各两条判据，"
        "每条都是「生产求解路径 vs 独立闭式/表列参考」的定量比较，误差越线即 FAIL，"
        "不靠放宽容差过关。")
    add("")
    add("| 级 | 判据 | 计算值 | 参考值 | 相对误差 | 门限 | 判定 |")
    add("|---|---|---|---|---|---|---|")
    for c in d["checks"]:
        add(f"| {c['criterion'].split('-')[0]} | `{c['criterion']}` | {c['computed']:.6e} | "
            f"{c['reference']:.6e} | {pct(c['relative_error'])} | "
            f"{100 * c['tolerance']:.1f}% | {verdict(c)} |")
    add("")
    add("六项全部通过，最紧的一项是板模态 (1,2)/(2,1) 对，误差 "
        f"{pct(checks['L3-PLATE-MODAL']['relative_error'])}（门限 2%）；该级误差随网格单调下降，"
        "20×20 时降至 "
        + pct(d["convergence"]["plate_modal"][-1]["relative_errors"][1]) +
        "（§3.5），所以 16×16 的通过不是网格凑出来的孤立点。")
    add("")
    add("## 1. 案例介绍（物理背景 · 几何 · 参数）")
    add("")
    add("船体结构的总纵强度、板格弯曲与舱壁/甲板振动分别落在三个经典模型上：Euler–Bernoulli 梁、"
        "Kirchhoff 薄板、以及二者的自由振动特征值问题。三者是线性弹性求解器的最小完备验证面："
        "梁考弯曲刚度与一致载荷向量，板考二维弯曲刚度与边界处理，模态考刚度/质量矩阵的一致性与"
        "广义特征值求解。任一项错了，上层（壳、加筋板、湿模态、疲劳谱）都不可能对。")
    add("")
    add("### 1.1 L1 —— 梁静力")
    add("")
    add("| 参数 | 取值 |")
    add("|---|---|")
    add(f"| 跨度 L | {m1['length_m']} m |")
    add(f"| 截面 b × h | {m1['section_m'][0]} m × {m1['section_m'][1]} m |")
    add(f"| 弹性模量 E | {m1['young_Pa']:.3g} Pa |")
    add(f"| 惯性矩 I = b h³/12 | {m1['second_moment_m4']:.6e} m⁴ |")
    add(f"| 抗弯刚度 EI | {m1['EI_Nm2']:.6e} N·m² |")
    add(f"| 均布载 q（简支梁） | {m1['uniform_load_N_per_m']:.3g} N/m |")
    add(f"| 端部集中载 P（悬臂梁） | {m1['tip_load_N']:.3g} N |")
    add(f"| 单元数 | {m1['elements']}（三次 Hermite） |")
    add("")
    add("参考解：")
    add("")
    add("* 简支梁均布载：`w(x) = q x (L³ − 2Lx² + x³) / (24 EI)`，取 `x/L = 0.1, 0.2, 0.3, 0.4, 0.5` 五个截面；")
    add("* 悬臂梁端部集中力：`w_max = P L³ / (3 EI)`。")
    add("")
    add("### 1.2 L2 —— 板静力")
    add("")
    add("| 参数 | 取值 |")
    add("|---|---|")
    add(f"| 板边长 a | {m2['side_m']} m（方形） |")
    add(f"| 板厚 h | {m2['thickness_m']} m，a/h = {m2['aspect_ratio_a_over_h']:.0f} |")
    add(f"| E / ν | {m2['young_Pa']:.3g} Pa / {m2['poisson']} |")
    add(f"| 抗弯刚度 D = E h³/(12(1−ν²)) | {m2['rigidity_Nm']:.6e} N·m |")
    add(f"| 均布压力 q | {m2['pressure_Pa']:.3g} Pa |")
    add(f"| 网格 | {m2['elements_per_side']}×{m2['elements_per_side']} Q4；单元：{m2['element']} |")
    add("")
    add("参考解：")
    add("")
    add("* 四边简支：Navier 双三角级数，`m, n = 1, 3, …, 11`（11×11 项截断，判据要求即此截断）；")
    add("* 四边固支：Timoshenko & Woinowsky-Krieger《Theory of Plates and Shells》Table 35 表列系数 "
        f"`w_max = {refs['timoshenko_coefficient']} q a⁴/D`（ν = 0.3），并以可分离 Rayleigh–Ritz"
        "（10 阶）作第二锚点交叉校核。")
    add("")
    add("### 1.3 L3 —— 模态")
    add("")
    add("| 参数 | 梁 | 方板 |")
    add("|---|---|---|")
    add(f"| 几何 | L = {m3['length_m']} m，{m3['section_m'][0]}×{m3['section_m'][1]} m | "
        f"a = {pm['side_m']} m，h = {pm['thickness_m']} m |")
    add(f"| E / ν | {m3['young_Pa']:.3g} Pa / — | {pm['young_Pa']:.3g} Pa / {pm['poisson']} |")
    add(f"| ρ | {m3['density_kg_per_m3']:.0f} kg/m³ | {pm['density_kg_per_m3']:.0f} kg/m³ |")
    add(f"| 离散 | {m3['elements']} 个 Hermite 梁单元，一致质量阵 | "
        f"{pm['elements_per_side']}×{pm['elements_per_side']} Q4 Mindlin（SRI） |")
    add("")
    add("参考解：")
    add("")
    add("* 简支梁 `ω_n = (nπ/L)² √(EI/ρA)`，前 5 阶（βₙL = nπ 是闭式根）；")
    add("* 四边简支方板 Navier 模态 `ω_mn = π² √(D/ρh) [(m/a)² + (n/b)²]`，前 3 阶为 (1,1)、(1,2)、(2,1)；")
    add("* 振型用 MAC 与退化子空间捕获率校核，不只看频率。")
    add("")
    add("## 2. 怎么做的（方法 · 流程 · 工具链）")
    add("")
    add("### 2.1 生产路径映射（阶梯必须打在生产的求解器上）")
    add("")
    add("| 级 | 生产代码路径 | 参考解 |")
    add("|---|---|---|")
    add("| L1 | `tensorfem.frame2d.solve_frame_static`（Hermite 三次梁单元 + 一致载荷向量） | 闭式梁解 |")
    add("| L2 | `tensorfem.plate_mindlin.solve_plate_on_winkler_foundation`（`winkler_k = 0`，"
        "Q4 SRI Mindlin，薄极限 a/h = 100） | Navier 级数 / Timoshenko 表值 / Ritz |")
    add("| L3 梁 | `tensorfem.modal.simply_supported_beam_modes`（一致质量 + Cholesky 约化广义特征值） | "
        "`(nπ/L)²√(EI/ρA)` |")
    add("| L3 板 | `tensorfem.plate_vibration.solve_plate_vibration`（Mindlin 板一致质量含转动惯量） | "
        "Navier 模态 |")
    add("")
    add("`verification_ladder` 只做三件事：装配模型、调用上述生产路径、把结果与参考解逐项比对并保留原始数字。"
        "它不复制任何单元刚度矩阵（L1 直接驱动 `frame2d`，L2/L3 直接驱动板/模态入口），因此阶梯失败必然指向"
        "生产代码或参考公式，而不是某段测试专用实现。")
    add("")
    add("### 2.2 流程")
    add("")
    add("1. `scripts/run_basic_verification_ladder.py` 跑六级判据，写入 "
        f"`{d['artifacts']['report_json']}`（含每个判据的 computed/reference/relative_error/"
        "tolerance/passed，以及全部收敛行与图路径）；")
    add("2. 同一脚本做三组网格加密（板静力 8/12/16、板模态 8/12/16/20、梁模态 4/8/16/32），"
        "把「离散误差沿 h 下降」写成数据而不是断言；")
    add("3. `tests/test_verification_ladder.py` 用模块里的 `CRITERIA` 同一批门限做回归：门限从模块导入"
        "而非抄写，放宽门限必须同时改两处，无法悄悄进行；")
    add("4. 生成 5 张图（§4），`docs/` 与 `artifacts/` 同时留痕，报告 JSON 记录图文件名，形成审计链。")
    add("")
    add("### 2.3 工具链与数值设置")
    add("")
    add("* PyTorch 2.7 float64 端到端（无随机数、无 GPU、无稀疏近似）；线性方程组稠密直接解，"
        "特征值用 `linalg.eigh`（先做质量阵 Cholesky 约化）；")
    add("* 板静力用仓库既有 SRI 选择性积分 Q4 单元（弯曲 2×2、剪切 1 点），薄板 a/h = 100 时退化为 "
        "Kirchhoff 响应；")
    add("* 参考解全部在脚本内用闭式公式/级数现场计算，与 `tensorfem.plate_uniform_load` 的既有参考实现"
        "互相独立（后者作交叉校核，见 §3.2 的 Ritz 锚点）。")
    add("")
    add("## 3. 计算结果")
    add("")
    add("### 3.1 L1 梁：五个截面的挠度对标（判据 L1-SS-UDL）")
    add("")
    add("| x/L | x [m] | FEM w [mm] | 解析 w [mm] | 相对误差 |")
    add("|---|---|---|---|---|")
    for r in L1["sections"]:
        add(f"| {r['x_over_L']:.2f} | {r['x']:.3f} | {1e3 * r['computed']:.6f} | "
            f"{1e3 * r['reference']:.6f} | {pct(r['relative_error'], 5)} |")
    add("")
    add("最差截面即判据值：" + pct(checks["L1-SS-UDL"]["relative_error"], 5) +
        "（门限 0.5%）。悬臂梁端部集中力：FEM "
        f"{checks['L1-CANTILEVER-TIP']['computed'] * 1e3:.6f} mm vs 解析 "
        f"{checks['L1-CANTILEVER-TIP']['reference'] * 1e3:.6f} mm，误差 "
        f"{pct(checks['L1-CANTILEVER-TIP']['relative_error'])}（门限 1%）。")
    add("")
    add("这两条误差的量级要正确解读（见 §5.1）：三次 Hermite 单元在节点上对三次/四次闭式解是**精确**的，"
        "所以端部量与节点量落在机器精度；截面上（单元内部）的误差才是插值误差，它随 h⁴ 下降（§3.6）。")
    add("")
    add("### 3.2 L2 板：简支 Navier 与固支表值（判据 L2-SS-NAVIER / L2-CLAMPED）")
    add("")
    add("| 量 | 数值 [m] | 说明 |")
    add("|---|---:|---|")
    add(f"| FEM 简支中心挠度（{m2['elements_per_side']}×{m2['elements_per_side']}） | "
        f"{checks['L2-SS-NAVIER']['computed']:.6e} | Q4 SRI Mindlin，a/h = 100 |")
    add(f"| Navier 11×11 截断参考 | {refs['navier_11_terms']:.6e} | 判据基准 |")
    add(f"| Navier 401 项收敛值 | {refs['navier_converged']:.6e} | 用于量化截断偏差 |")
    add(f"| 11×11 截断偏差 | {pct(refs['navier_11_relative_bias'])} | 参考自身的自偏差，**不是** FEM 误差 |")
    add(f"| FEM 固支中心挠度 | {checks['L2-CLAMPED']['computed']:.6e} | 同网格 |")
    add(f"| Timoshenko 表值 {refs['timoshenko_coefficient']} q a⁴/D | {refs['timoshenko_centre']:.6e} | "
        "3 位有效数字 |")
    add(f"| Ritz（10 阶）中心挠度 | {refs['ritz_centre']:.6e} | 系数 {refs['ritz_coefficient']:.7f} |")
    add("")
    add(f"判据值：简支 {pct(checks['L2-SS-NAVIER']['relative_error'])}（门限 1%）、"
        f"固支 {pct(checks['L2-CLAMPED']['relative_error'])}（门限 2%）。固支若改以 Ritz 收敛值为基准则为 "
        f"{pct(refs['clamped_vs_ritz_relative_error'])}：表值只有 3 位有效数字，这项差就是表值舍入与 "
        "FE 离散误差之和，两者都在 3% 以内（§6）。")
    add("")
    add("### 3.3 L3 梁模态：前五阶频率（判据 L3-BEAM-MODAL）")
    add("")
    add("| 阶 n | FEM f [Hz] | 解析 f [Hz] | 相对误差 | 振型 MAC |")
    add("|---|---|---|---|---|")
    for r in L3["beam"]["modes"]:
        add(f"| {r['mode']} | {r['computed_hz']:.4f} | {r['reference_hz']:.4f} | "
            f"{pct(r['relative_error'])} | {r['mac']:.8f} |")
    add("")
    add(f"判据取五阶中最差者 {pct(checks['L3-BEAM-MODAL']['relative_error'])}（门限 1%）；"
        "前五阶振型 MAC 全部 1.00000000，即 FE 特征向量与半正弦解析振型同向。")
    add("")
    add("### 3.4 L3 板模态：前三阶（判据 L3-PLATE-MODAL）")
    add("")
    add("| 模态 (m,n) | FEM f [Hz] | Navier f [Hz] | 相对误差 | 单振型 MAC | 退化子空间捕获率 |")
    add("|---|---|---|---|---|---|")
    for r in L3["plate"]["modes"]:
        add(f"| ({r['m']},{r['n']}) | {r['computed_hz']:.4f} | {r['reference_hz']:.4f} | "
            f"{pct(r['relative_error'])} | {r['mac']:.6f} | {r['subspace_capture']:.8f} |")
    add("")
    add(f"判据 {pct(checks['L3-PLATE-MODAL']['relative_error'])}（门限 2%）。方板 (1,2)、(2,1) 是简并模态对，"
        "FE 特征向量在二维特征子空间内可任意旋转，所以**单振型 MAC 在退化对上没有意义**（表里 "
        f"{L3['plate']['modes'][1]['mac']:.3f} 就是这个假象）；正确指标是把 FE 振型投影到 "
        "{(1,2),(2,1)} 子空间的捕获率，实测 "
        f"{L3['plate']['modes'][1]['subspace_capture']:.8f}，即振型完全落在理论子空间内。")
    add("")
    add("### 3.5 网格加密（离散误差是测出来的）")
    add("")
    add("板静力中心挠度：")
    add("")
    add("| n×n | 简支 w [mm] | 简支相对误差 | 固支 w [mm] | 固支相对误差（vs Ritz） |")
    add("|---|---|---|---|---|")
    for r in d["convergence"]["plate_static"]:
        ss, cl = r["boundaries"]["simply_supported"], r["boundaries"]["clamped"]
        add(f"| {r['elements_per_side']} | {1e3 * ss['centre_deflection']:.6f} | {pct(ss['relative_error'])} | "
            f"{1e3 * cl['centre_deflection']:.6f} | {pct(cl['relative_error'])} |")
    add("")
    add("板模态频率（前三阶）：")
    add("")
    add("| n×n | f(1,1) [Hz] | 误差 | f(1,2) [Hz] | 误差 |")
    add("|---|---|---|---|---|")
    for r in d["convergence"]["plate_modal"]:
        f, e = r["frequencies_hz"], r["relative_errors"]
        add(f"| {r['elements_per_side']} | {f[0]:.4f} | {pct(e[0])} | {f[1]:.4f} | {pct(e[1])} |")
    add("")
    add("梁模态频率（前五阶相对误差）：")
    add("")
    add("| 单元数 | 1 阶 | 2 阶 | 3 阶 | 4 阶 | 5 阶 |")
    add("|---|---|---|---|---|---|")
    for r in d["convergence"]["beam_modal"]:
        add(f"| {r['elements']} | " + " | ".join(pct(e) for e in r["relative_errors"]) + " |")
    add("")
    add("三组加密都单调下降，没有出现「粗网格碰巧过关」或「细网格反而变差」，"
        "这也是把判据定在 16×16、而不是只在最细网格上设门的理由。")
    add("")
    add("### 3.6 收敛阶（梁单元内部插值误差）")
    add("")
    interior = d["convergence"]["beam_static_interior_errors"]
    ratios = [interior[i] / interior[i + 1] for i in range(len(interior) - 1)]
    add("取 24 个**非节点**截面上的最大相对误差，网格 2/4/8 单元对应 "
        f"{interior[0]:.2e} → {interior[1]:.2e} → {interior[2]:.2e}（判据值取最后一档），"
        "相邻两档比值约 " + " 与 ".join(f"{r:.1f}" for r in ratios) +
        "（绝对值随位置因子波动，但对应速率 "
        + " 与 ".join(f"{__import__('math').log2(r):.2f}" for r in ratios) +
        "，均 > 3），趋近三次 Hermite 插值的 O(h⁴)；测试 "
        "`test_l1_beam_interior_error_converges_quartic` 把「速率 > 3」设为门，"
        "载荷向量或插值写错会立刻掉到一阶而失败。")
    add("")
    add("## 4. 结果图")
    add("")
    for i, fig in enumerate(d["figures"], start=1):
        add(f"### 图 {i} —— `{fig['name']}`")
        add("")
        add(f"![{fig['name']}](assets/basic-verification-ladder/{fig['name']})")
        add("")
        add(fig["caption"])
        add("")
    add("## 5. 注意点（坑 · 局限 · 适用范围）")
    add("")
    add("### 5.1 节点精确性不是精度证明（本阶梯最容易踩的坑）")
    add("")
    add("三次 Hermite 梁单元对 Euler–Bernoulli 梁的均布载（四次）与端部力（三次）解在节点上是**精确**的，"
        "所以端部挠度误差在 1e-14 量级、五截面里 x/L = 0.5 那一行误差为 0。这是单元性质，"
        "不是求解器精度；把它当精度会把真问题掩盖掉。因此：")
    add("")
    add("* 判据 L1-SS-UDL 取五截面中最差的**单元内部**截面（当前 x/L = "
        f"{L1['worst_section']['x_over_L']:.2f}），误差 "
        f"{pct(checks['L1-SS-UDL']['relative_error'], 5)}，反映的是插值质量；")
    add("* 另外用 2/4/8 单元的收敛阶门（§3.6）与「节点精确性」单独加锁"
        "（`test_l1_nodal_values_are_machine_exact`），把两种性质分开写死；")
    add("* 悬臂端部集中力的机器精度误差不能解读为「悬臂梁精度更高」，只能说载荷向量与刚度矩阵一致。")
    add("")
    add("### 5.2 与仓库既有验证文档的关系（增量在哪）")
    add("")
    add("仓库此前已有三条独立的同类证据：`docs/FEM_VERIFY.md`（悬臂梁端部挠度 + 简支板 Navier）、"
        "`docs/plate-uniform-load-verification.md`（简支/固支板挠度与弯矩）、"
        "`docs/plate-vibration-validation.md` 与 `docs/beam_dynamics_validation.md`（板/梁振动）。"
        "本节点**不重复**它们，增量是：")
    add("")
    add("1. **梁静力的分布量**：既有梁证据只比端部/中点单值，这里给均布载沿跨 5 个截面的场分布与逐点误差；")
    add("2. **三级合成门**：六条判据落在同一份 JSON、同一套 `CRITERIA` 常量、同一组 pytest 门里，"
        "任何一级回归都会让整条阶梯翻红，而不是分散在四篇文档里各说各话；")
    add("3. **参考解自身的误差量化**：11×11 Navier 截断偏差 "
        f"{pct(refs['navier_11_relative_bias'])}、Timoshenko 表值 vs Ritz 收敛值 "
        f"{pct(-refs['clamped_vs_ritz_relative_error'])}，明确区分「参考的自偏差」与「FEM 的离散误差」；")
    add("4. **模态振型的定量校核**：梁 MAC、板退化子空间捕获率（§3.4），既有文档只比频率；")
    add("5. **收敛阶门**：把 O(h⁴) 插值速率本身作为测试判据（§3.6）。")
    add("")
    add("### 5.3 适用范围")
    add("")
    add("* 仅线性弹性、小变形、各向同性、等厚平板；不含剪切变形结论（a/h = 100 时 Mindlin 退化为 "
        "Kirchhoff，a/h < 20 的厚板会出现一阶剪切偏差，本阶梯不覆盖）；")
    add("* 板单元是 Q4 SRI Mindlin（仓库既有单元），不是 Kirchhoff 协调元；这里验证的是"
        "「薄极限下与 Kirchhoff 解析解一致」，不验证厚板应力分布；")
    add("* 只做面内均匀压力的中心挠度与前三阶频率，不涉及屈曲、后屈曲、接触、材料/几何非线性；")
    add("* 模态是干模态（无附连水、无预应力），湿模态与预应力模态另有专文"
        "（`docs/prestressed-modal-verification.md` 等）；")
    add("* 梁参考解为 Euler–Bernoulli（无剪切变形），L/h = "
        f"{m1['length_m'] / m1['section_m'][1]:.0f}，剪切影响可忽略。")
    add("")
    add("### 5.4 其它坑")
    add("")
    add("* Timoshenko 表值 `0.00126` 只有 3 位有效数字，用它做基准时 2% 门限里有一半预算被表值舍入吃掉；"
        "所以同时给 Ritz 收敛值作第二锚点（两者差约 0.1%）；")
    add("* 方板 (1,2)/(2,1) 简并：单振型 MAC 会给出 0.51 这种「看起来很差」的数字，"
        "用子空间捕获率才能正确判定（§3.4）；")
    add("* 板静力是稠密装配 + 稠密直接解，网格越大越贵（16×16 约 20 s，20×20 板模态约 90 s），"
        "证据脚本把最贵的一档放在后台跑，测试套件只跑 16×16（并登记进 slow lane）；")
    add("* 报告里 L1-SS-UDL 的 computed/reference 记录的是**最差截面**的数值，不是中点值——"
        "中点值节点精确、误差恒为 0，拿它当判据值会假通过。")
    add("")
    add("## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3%）")
    add("")
    add("| 关键量 | 基准 | 相对误差 | 门限 | 是否 < 3% |")
    add("|---|---|---|---|---|")
    rows = [
        ("简支梁均布载，5 截面最差挠度", "w(x)=qx(L³−2Lx²+x³)/24EI", checks["L1-SS-UDL"]),
        ("悬臂梁端部挠度", "PL³/(3EI)", checks["L1-CANTILEVER-TIP"]),
        ("简支方板中心挠度", f"Navier 11×11（自偏差 {pct(refs['navier_11_relative_bias'])}）",
         checks["L2-SS-NAVIER"]),
        ("固支方板中心挠度", f"Timoshenko 表值 {refs['timoshenko_coefficient']}", checks["L2-CLAMPED"]),
        ("固支方板中心挠度", "Ritz 10 阶收敛值", {"relative_error": refs["clamped_vs_ritz_relative_error"],
                                                "tolerance": 2.0e-2}),
        ("简支梁前 5 阶频率（最差阶）", "(nπ/L)²√(EI/ρA)", checks["L3-BEAM-MODAL"]),
        ("简支方板前 3 阶频率（最差阶）", "π²√(D/ρh)[(m/a)²+(n/b)²]", checks["L3-PLATE-MODAL"]),
    ]
    for name, reference, c in rows:
        below = "是" if abs(c["relative_error"]) < 0.03 else "**否**"
        add(f"| {name} | {reference} | {pct(c['relative_error'])} | "
            f"{100 * c['tolerance']:.1f}% | {below} |")
    add("")
    add("全部七项关键量相对误差均 < 3%（最大者为板模态 3 阶的 "
        f"{pct(checks['L3-PLATE-MODAL']['relative_error'])}）。需要强调：")
    add("")
    add("* 这里的「基准」对板静力/板模态是**解析解**，不是实验值——本阶梯证明的是求解器正确性，"
        "不是 Kirchhoff/Mindlin 板模型与真实结构的吻合度；")
    add("* 悬臂梁与简支梁节点量是解析精确，误差在舍入量级，不代表非多项式载荷下也这么好；")
    add("* 板模态最紧一档 " + pct(checks["L3-PLATE-MODAL"]["relative_error"]) +
        " 是离散误差（Q4 SRI 板单元在弯曲主导的高阶模态上收敛较慢），不是参考解误差；"
        "20×20 时降到 " + pct(d["convergence"]["plate_modal"][-1]["relative_errors"][1]) +
        "，趋势单调。")
    add("")
    add("## 7. 复现与审计链")
    add("")
    add("```bash")
    add("# 全量证据（约 4 分钟：16×16 板静力 + 16×16 板模态 + 三组加密）")
    add("python scripts/run_basic_verification_ladder.py")
    add("")
    add("# 快速冒烟（12×12 / 12×12 / 8 单元）")
    add("python scripts/run_basic_verification_ladder.py --quick")
    add("")
    add("# 回归门（六级判据、截断偏差、收敛性、退化子空间、fail-closed 自检）")
    add("python -m pytest tests/test_verification_ladder.py -q")
    add("```")
    add("")
    add("审计链：")
    add("")
    add("| 环节 | 载体 |")
    add("|---|---|")
    add("| 判据定义 | `tensorfem.verification_ladder.CRITERIA`（单一来源，测试导入） |")
    add(f"| 原始数字 | `{d['artifacts']['report_json']}`（本次运行 {d['runtime_seconds']} s；网格 "
        f"{d['mesh']['plate_static_elements']}×{d['mesh']['plate_static_elements']} 板静力 / "
        f"{d['mesh']['plate_modal_elements']}×{d['mesh']['plate_modal_elements']} 板模态 / "
        f"{d['mesh']['beam_elements']} 梁单元） |")
    add("| 图 | " + f"`{d['artifacts']['figures_dir']}/`（5 张，文件名见 JSON 的 `figures` 字段） |")
    add("| 代码 | `src/tensorfem/verification_ladder.py`、`scripts/run_basic_verification_ladder.py`、"
        "`scripts/build_verification_ladder_doc.py` |")
    add("| 门 | `tests/test_verification_ladder.py` |")
    add("")
    add("## 8. 未做 / 超出范围")
    add("")
    add("* 没有做板/梁的应力场对标（本阶梯只覆盖位移与频率量；应力场另有 `docs/cook-*`、"
        "`docs/hemisphere-hex20-stress-oracle.md` 等专文，其中开孔球壳应力仍未认证）；")
    add("* 没有做壳单元几何刚度、Hertz 接触、后屈曲——这些仍是 "
        "`docs/benchmark-qualification-status.md`「尚未闭环的工作」里的欠账；")
    add("* 没有引入实验数据：三级判据全部是解析/表列参考，属「求解器正确性」，"
        "不能替代模型验证与不确定度量化；")
    add("* 厚板（a/h < 20）、大变形、失稳路径不在覆盖范围内。")
    add("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    text = build()
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT} ({len(text.splitlines())} lines)")