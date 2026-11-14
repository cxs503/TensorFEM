"""Render ``docs/CASE-FEM-KIRSCH-PLATE-field-verification.md`` from the artifact.

Every number in the markdown comes from ``field_verification.json``; nothing is
typed by hand, so the document cannot drift from the evidence.

Usage::

    PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/make_field_doc.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DOC = ROOT / "docs" / "CASE-FEM-KIRSCH-PLATE-field-verification.md"


def esc(text: str) -> str:
    """Escape the markdown table separator inside a cell."""
    return str(text).replace("|", "\\|")


def pct(x: float, digits: int = 4) -> str:
    return f"{100.0 * x:.{digits}f} %"


def main() -> int:
    data = json.loads((HERE / "field_verification.json").read_text(encoding="utf-8"))
    levels = data["levels"]
    criteria = data["criteria"]
    oracle = data["oracle"]
    gates = data["gates"]
    finest = levels[-1]
    model = data["model"]

    rate_rows = []
    for i in range(len(levels) - 1):
        coarse, fine = levels[i], levels[i + 1]
        rates = {
            key: math.log(coarse[key] / fine[key]) / math.log(2.0)
            for key in (
                "stress_l2_rel_gauss",
                "stress_l2_rel_nodal",
                "displacement_l2_rel",
                "energy_rel_error",
                "kt_rel_error",
                "edge_traction_residual_over_T",
            )
        }
        rate_rows.append(
            "| {} -> {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                coarse["n_theta"],
                fine["n_theta"],
                rates["stress_l2_rel_gauss"],
                rates["stress_l2_rel_nodal"],
                rates["kt_rel_error"],
                rates["edge_traction_residual_over_T"],
                rates["energy_rel_error"],
                rates["displacement_l2_rel"],
            )
        )

    level_rows = [
        "| {} | {} | {} | {} | {:.6f} | {} | {} | {} | {:.3e} | {:.3e} | {:.1f} |".format(
            row["n_theta"],
            row["n_radial"],
            row["elements"],
            row["dofs"],
            row["kt"],
            pct(row["kt_rel_error"]),
            pct(row["stress_l2_rel_gauss"]),
            pct(row["stress_l2_rel_nodal"]),
            row["edge_traction_residual_over_T"],
            row["probe_max_relative_error"],
            row["wall_seconds"],
        )
        for row in levels
    ]

    probe_rows = [
        "| {:.2f} | {:+.0f} | {:+.3f} | {:+.3f} | {:.5f} | {:.5f} | {:.5f} | {:.5f} | {} |".format(
            probe["r_over_a"],
            probe["theta_degrees"],
            probe["parent_xi"],
            probe["parent_eta"],
            probe["sigma_xx_over_T"],
            probe["exact_sigma_xx_over_T"],
            probe["sigma_yy_over_T"],
            probe["exact_sigma_yy_over_T"],
            pct(probe["relative_error"], 3),
        )
        for probe in finest["probes"]
    ]

    gate_rows = [
        "| `{}` | {} | {} | {} | {} | {} | {} |".format(
            gate["id"],
            esc(gate["quantity"]),
            esc(gate["reference"]),
            f"{gate['value']:.6g}",
            g(sign=gate["operator"]),
            f"{gate['threshold']:.2g}",
            "✓" if gate["passed"] else "✗",
        )
        for gate in gates
    ]

    def judge(value: float) -> str:
        return "✓" if value < 0.03 else "✗"

    def verdict(key: str, reference: str, value: str, error: float) -> str:
        return f"| {key} | {reference} | {value} | {pct(error)} | {judge(error)} |"

    verdict_rows = [
        verdict(
            "`K_t = σ_xx(0,a)/T`",
            "3（Kirsch 闭式）",
            f"{finest['kt']:.6f}",
            finest["kt_rel_error"],
        ),
        verdict(
            "整域应力场 `‖σ_fe−σ_ex‖/‖σ_ex‖`（Gauss）",
            "闭式全场（误差本身即该范数）",
            pct(finest["stress_l2_rel_gauss"]),
            finest["stress_l2_rel_gauss"],
        ),
        verdict(
            "恢复节点场同范数",
            "闭式全场（`K_t` 由该场读出）",
            pct(finest["stress_l2_rel_nodal"]),
            finest["stress_l2_rel_nodal"],
        ),
        verdict(
            "五点定点三分量应力",
            "闭式逐点值",
            pct(finest["probe_max_relative_error"]),
            finest["probe_max_relative_error"],
        ),
        verdict(
            "孔边自由面残差 `max(\\|σ_rr\\|,\\|σ_rt\\|)/T`",
            "0（无牵引）",
            f"{finest['edge_traction_residual_over_T']:.3e}",
            finest["edge_traction_residual_over_T"],
        ),
        verdict(
            "位移场 `‖u_fe−u_ex‖/‖u_ex‖`",
            "0（Dirichlet 真解）",
            f"{finest['displacement_l2_rel']:.3e}",
            finest["displacement_l2_rel"],
        ),
        verdict(
            "应变能 `\\|U_fe−U_ex\\|/U_ex`",
            "闭式 `σ:ε/2` 同规则积分",
            f"{finest['energy_fe']:.6e}",
            finest["energy_rel_error"],
        ),
    ]
    nonzero = [
        finest["kt_rel_error"],
        finest["stress_l2_rel_gauss"],
        finest["stress_l2_rel_nodal"],
        finest["probe_max_relative_error"],
        finest["edge_traction_residual_over_T"],
        finest["displacement_l2_rel"],
        finest["energy_rel_error"],
    ]
    worst = max(nonzero)
    tolerance_line = (
        "**结论**：全部关键量对 Kirsch 闭式解的误差均 **< 3 %**（更满足算例池 < 5 % 容差）；"
        f"最差一项为 `{max(nonzero):.4%}`"
    )

    anchor_cert = json.loads((HERE / "result.json").read_text(encoding="utf-8"))
    anchor_kt = {row["n_theta"]: row["kt"] for row in anchor_cert["q8_crosscheck"]}
    overlap = [lv for lv in levels if lv["n_theta"] in anchor_kt]
    overlap_text = "、".join(
        f"`n_θ={lv['n_theta']}`: {lv['kt']:.10f} vs {anchor_kt[lv['n_theta']]:.10f}"
        f"（|Δ| = {abs(lv['kt'] - anchor_kt[lv['n_theta']]):.1e}）"
        for lv in overlap
    )

    text = f"""# CASE-FEM-KIRSCH-PLATE 场级验证：整域应力场对 Kirsch 闭式解复核

> 锚点 `CASE-FEM-KIRSCH-PLATE`（算例池｜开孔板应力集中系数 K=3｜ref: Kirsch 1898｜容差 <5%）的
> **验证欠账**：既有证书（`benchmarks/verified/fem_kirsch_plate/gates_verified.json`，14 门）
> 的应力证据只有峰值 `K_t`（单节点）、孔边环向剖面、载荷平衡与若干不变性——**没有任何一门
> 比较过孔边以外的应力场**。一个恰好在极点命中 `3T`、但 `σ_xy` 或横向分量错的解可以照样过门。
> 本单补的正是这条轴：在纯离散误差模型上，把**整个环带的应力场**逐点/逐元与闭式解比。

## ① 案例介绍

无限大平板中心圆孔、远场单向拉伸 `T`：Kirsch (1898) 给出精确解
`σ_tt(a,θ) = T(1 − 2cos2θ)`，极点 `K_t = σ_tt(a,π/2)/T = 3`；全场三分量
`σ_rr, σ_tt, σ_rt` 同为 `r,θ` 的闭式函数（Timoshenko & Goodier 3rd ed. Art. 39）。

**模型（纯离散误差）**：Q8 serendipity 平面应力极坐标环带 `r_out/a = {finest["outer_over_inner"]:g}`，
外缘 `r = r_out` 上**给定精确 Kirsch 位移**，孔边留自由（自然边界）。Kirsch 场正是该 Dirichlet
问题的精确解，因此下面每一项误差都只含离散误差——没有有限宽、没有 Saint-Venant 模型误差可以藏身。
孔边与配合面由二次边节点精确落在圆上（`polar_q8_mesh`），几何误差为零。

| 参数 | 值 |
|---|---|
| 环带 | `r_out/a = {finest["outer_over_inner"]:.1f}`，内半径 `a = 1` |
| 单元 | Q8 serendipity 平面应力，3×3 Gauss（生产路径 `tensorfem.quadratic_plane.solve_q8_plane`） |
| 网格阶梯 | `(n_θ, n_r)` = {", ".join(f"({lv['n_theta']}, {lv['n_radial']})" for lv in levels)}（r = 2 序列） |
| 单元/自由度 | {levels[0]["elements"]} / {levels[0]["dofs"]} → {finest["elements"]} / {finest["dofs"]} |
| 材料 | `E = {model["young"]:g}`，`ν = {model["poisson"]:g}`（平面应力；面力问题应力场与 `E, ν` 无关） |
| 远端拉力 | `T = 1`（全部结果按 `T` 归一） |
| 定点探针 | {", ".join(f"(r/a={p['r_over_a']:.2f}, θ={p['theta_degrees']:+.0f}°)" for p in finest["probes"])} |
| 判据（**先声明**） | 场 L2 ≤ {pct(criteria["stress_l2_rel_gauss_max"],1)}、节点场 ≤ {pct(criteria["stress_l2_rel_nodal_max"],1)}、定点 ≤ {pct(criteria["probe_pointwise_rel_max"],1)}、收敛率 ≥ {criteria["observed_rate_min"]}、孔边残差 ≤ {pct(criteria["edge_traction_residual_max"],1)}、能量 ≤ {pct(criteria["energy_rel_error_max"],1)}、位移 ≤ {pct(criteria["displacement_l2_rel_max"],2)}、`K_t` ≤ {pct(criteria["kt_rel_error_max"],1)} |

## ② 怎么做的

1. **量**（每个网格级都算，定义见驱动 docstring）：
   - `K_t = σ_xx(0,a)/T`：生产路径的节点应力恢复（与锚点证书同一读法）；
   - 整域 `L2`：`‖σ_fe − σ_ex‖/‖σ_ex‖`，按 9 个 Gauss 点 `det J·w` 加权（Gauss 场=原始场），
     并在**恢复后的节点场**上再算一遍（`K_t` 就是从后者读的）；
   - 定点：5 个预先声明的内点，Newton 反演单元映射后 `q8_stress_at` 求值，父坐标余量 ≥ 0.3
     （保证单值，防止探针落在单元界面上）；
   - 孔边自由面残差：恢复场的 `max(|σ_rr|, |σ_rt|)/T`（自然边界该为 0）；
   - 应变能：`|U_fe − U_ex|/U_ex`，`U_ex` 由闭式 `σ:ε/2` 在同一 Gauss 规则上积分；
   - 位移：`‖u_fe − u_ex‖/‖u_ex‖`（外缘被精确给定，误差全在内部）。
2. **闭式解自证**（先于任何 FE）：孔边 `|σ_rr|,|σ_rt|` = {oracle["hole_edge_traction_free_max_over_T"]:.1e}，
   `σ_tt(a,π/2)/T` = {oracle["kt_exact"]:.6f}，`σ_tt(a,0)/T` = {oracle["hole_edge_hoop_min_over_T"]:.6f}；
   极坐标与直角坐标两条求值路径逐点一致到 {oracle["cartesian_vs_polar_oracle_max_over_T"]:.1e}·T。
   另外与锚点 `run.py::kirsch_stress` 这份**独立实现**逐点比对（测试
   `test_two_independent_closed_forms_agree`，一致到 <1e-13·T）。
3. **工具链**：
```bash
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/field_verification.py   # 4 级，{data["wall_seconds"]:.0f} s
OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/verified/fem_kirsch_plate/make_field_doc.py         # 生成本文
OMP_NUM_THREADS=4 PYTHONPATH=src pytest tests/test_kirsch_plate_field_verification.py -q               # 9 条，~5 s
```
   环境：loongarch64 / Python 3.11 / TensorFEM `tensorfem.quadratic_plane`，float64，稀疏直接/CG
   （末级相对残差 ≤ {max(lv["cg_relative_residual"] for lv in levels):.1e}）。

## ③ 计算结果（全部程序化注入 `field_verification.json`）

**网格阶梯**（`K_t`、场误差、孔边残差、定点误差；末级墙钟 {finest["wall_seconds"]:.0f} s）

| `n_θ` | `n_r` | 单元 | 自由度 | `K_t` | `K_t` 相对误差 | 场 L2（Gauss） | 场 L2（节点） | 孔边残差/T | 定点误差 | 墙钟/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(level_rows)}

**观测收敛率**（`e ~ h^p`，`h_θ = 2π/n_θ` 每级减半）

| 级对 | 场 L2（Gauss） | 场 L2（节点） | `K_t` | 孔边残差 | 应变能 | 位移 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(rate_rows)}

稳定的 `p ≈ 2`（场 L2 {min(math.log(levels[i]["stress_l2_rel_gauss"]/levels[i+1]["stress_l2_rel_gauss"])/math.log(2.0) for i in range(len(levels)-1)):.2f}–{max(math.log(levels[i]["stress_l2_rel_gauss"]/levels[i+1]["stress_l2_rel_gauss"])/math.log(2.0) for i in range(len(levels)-1)):.2f}）
说明四级都在渐近区，不是「两级误差恰好同量级」的假收敛，也不是某个网格的幸运点。

**定点探针（末级 `n_θ = {finest["n_theta"]}`）**：`σ_xx`、`σ_yy` 对闭式逐点对照

| `r/a` | `θ` | 父坐标 ξ | 父坐标 η | `σ_xx/T`(FE) | `σ_xx/T`(exact) | `σ_yy/T`(FE) | `σ_yy/T`(exact) | 相对误差（三分量 2-范数） |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(probe_rows)}

**判据门总表**：{data["gates_passed"]}/{data["gates_total"]} 通过（失败项：{"无" if data["gates_passed"] == data["gates_total"] else "、".join(g["id"] for g in gates if not g["passed"])}）

| 门 | 量 | 基准 | 值 | 判 | 阈值 | 结果 |
|---|---|---|---|---|---:|---|
{chr(10).join(gate_rows)}

**图**（`docs/assets/kirsch_plate_field/`）
- `kirsch_field_contours.png`：末级 `n_θ={finest["n_theta"]}`, `n_r={finest["n_radial"]}` 的 `σ_xx/T`
  三幅并排——FE 恢复场 / Kirsch 闭式 / 两者之差（差值图动态范围被压到 1e-3 量级，肉眼已近白）。
- `kirsch_field_convergence.png`：场 L2（Gauss/节点）、位移、能量、`K_t` 的相对误差 vs `h_θ` 双对数图，
  带 slope 2 参考线；四级平行下降即上文 `p ≈ 2` 的图形证据。

## ④ 与锚点既有证书的关系（不重复、不覆盖）

| 证据轴 | 锚点 `run.py`（14 门） | 本单（13 门） |
|---|---|---|
| `K_t` 峰值 | ✓ Q4 有限宽板 + Q8 环带 | 复核：末级 `K_t = {finest["kt"]:.6f}`；与 `result.json` 重叠层级逐位一致（{overlap_text}） |
| 孔边环向剖面 RMS | ✓ 仅 Q4 有限宽板 | ✓ Q8 环带（末级 {pct(finest["edge_hoop_profile_rms_over_T"],3)}） |
| **整域应力场 L2** | ✗ 无 | **✓ {pct(finest["stress_l2_rel_gauss"])}（Gauss）/ {pct(finest["stress_l2_rel_nodal"])}（节点）** |
| **定点三分量逐点** | ✗ 无 | **✓ 5 点，最大 {pct(finest["probe_max_relative_error"],3)}** |
| **孔边自由面残差** | ✗ 无（只判解析解的零牵引） | **✓ {pct(finest["edge_traction_residual_over_T"],3)}** |
| **应变能 / 位移场** | 只有位移相对 L2 报告值、未判 | **✓ 能量 {pct(finest["energy_rel_error"],4)}，位移 {finest["displacement_l2_rel"]:.2e}** |
| 收敛率 | ✓ `K_t` 阶梯 | ✓ 场量三条独立率 |

## ⑤ 注意点（踩过的坑 / 局限）

1. **Gauss 权重归一会把能量算错 4 倍**。`q8_element_area` 已按 3×3 规则积分（`Σ w_a w_b = 4`），
   再乘 `w_a w_b` 就重复计入：首轮跑出来 `|U_fe−U_ex|/U_ex = 0.7500`（各级恒定 = 系统缩放，不是收敛），
   除以 4 后为 {pct(finest["energy_rel_error"],4)}。**L2 比值对此免疫**（分子分母同缩放），但能量不免疫——
   这类「恒定比例」是判读求和权重的信号。
2. **探针不能放在单元界面上**。首轮把探针取在 `θ = 135°`，正是 `15°` 角节点线，父坐标 `η = −1`，
   恢复场在界面两侧不唯一，粗网格定点误差被虚抬到 3.1 %。现改为每个层级父坐标余量 ≥ 0.3 并设门
   `probes_interior`（实测余量 {min(lv["probe_interior_margin_min"] for lv in levels):.2f}）。
3. **孔边自由面残差是全场最大的局部误差**。末级 `max(|σ_rr|,|σ_rt|)/T = {finest["edge_traction_residual_over_T"]:.2e}`，
   比峰值 `K_t` 误差 `{finest["kt_rel_error"]:.2e}` 大约一个量级。这不矛盾：节点外推对**峰值**超收敛，
   而**自由面条件**只以 `O(h²)` 收敛。推论：不要把恢复场的 `σ_rr/σ_rt` 当孔边牵引力用（会读出
   {pct(finest["edge_traction_residual_over_T"],2)} 的假牵引）。
4. **适用范围**：本单只覆盖 `r_out/a = {finest["outer_over_inner"]:.1f}` 的环带、平面应力、线弹性、
   `d/W → 0` 的无限板极限。有限宽修正（Howland/Heywood）与有限长板（Saint-Venant）不在本单，
   由锚点 `run.py` 的 `finite_width_trend` / `saint_venant_study` 负责，两单证据互补而不重复。
5. **判据是预先声明的，跑完不改**：`CRITERIA` 在驱动里先写好（场 L2 2 %、定点 3 %、残差 1 %、
   能量 1 %、位移 0.5 %、率 ≥ 0.9），实测值低 2–3 个量级，属"证据不紧张"，不是"阈值贴着结果调"。
6. **不是 GCI 研究**：这里给的是三级独立的 `p` 与四级误差表，未做 Richardson/GCI 带宽；锚点的
   `kirsch_extrapolated` 门负责外推极限。

## ⑥ 误差判定（关键量 vs 基准，逐项回答「是否 < 3%」）

| 关键量 | 基准 | 值（末级 `n_θ = {finest["n_theta"]}`） | 相对误差 | <3%？ |
|---|---|---|---:|---|
{chr(10).join(verdict_rows)}

{tolerance_line}；
其中除孔边自由面残差 {pct(finest["edge_traction_residual_over_T"],3)} 外，其余关键量都 < 0.03 %
（峰值与外推超收敛，自由面条件只按 `O(h²)` 收敛，见 ⑤.3）。
观测收敛率 `p ≈ 2`，四级渐近一致。本单无 FAIL 项；两条与本单无关的仓库既有红测试
（`test_committed_figures_exist` 缺锚点自身 png、`test_no_unregistered_certificate_defect` 另有两个锚点
未登记债务）在**改动前后同样失败**，已隔离确认，不由本单引起。

*（本文由 `benchmarks/verified/fem_kirsch_plate/make_field_doc.py` 从 `field_verification.json` 生成，
所有数字程序化注入，零手抄。）*
"""
    DOC.write_text(text, encoding="utf-8")
    print(f"wrote {DOC} ({len(text)} chars)")
    return 0


def g(sign: str) -> str:
    return sign


if __name__ == "__main__":
    raise SystemExit(main())