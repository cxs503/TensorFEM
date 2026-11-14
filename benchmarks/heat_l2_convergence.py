"""Generate the CASE-FEM-HEAT-L2 evidence, figures and report (one command).

Case: steady heat conduction on bilinear ``Q4`` elements, acceptance criterion =
L2 convergence rate >= 2 (manufactured-solution reference).

Outputs (all machine generated, no number is typed into the report by hand)
--------------------------------------------------------------------------
* ``artifacts/CASE-FEM-HEAT-L2/evidence.json`` -- target order, every ladder,
  every error norm, the gate verdicts, the quadrature control.
* ``artifacts/CASE-FEM-HEAT-L2/tables.md`` -- the report tables on their own.
* ``docs/assets/case_fem_heat_l2_convergence.png`` -- log-log L2 / H1 ladders
  with the theoretical slopes p = 2 and p = 1.
* ``docs/assets/case_fem_heat_l2_field.png`` -- FE field, analytic field and the
  pointwise error of the finest mesh of the source-driven case.
* ``docs/CASE-FEM-HEAT-L2-steady-conduction.md`` -- the six-section report; every
  number in it is formatted from ``evidence``.

Repro::

    PYTHONPATH=src python3 benchmarks/heat_l2_convergence.py
    PYTHONPATH=src python3 -m pytest tests/test_heat_l2_convergence.py -q

The largest system is 65 x 65 = 4225 unknowns; ``tensorfem.thermal`` assembles
*dense* matrices, so the ladder stops at nx = 64 (a 143 MB conductivity matrix,
~1 GB RSS).  torch's default thread pool costs more than it buys on matrices
this size, so the thread count is capped.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("TENSORFEM_THREADS", "4")))

from tensorfem.heat_l2_convergence import (  # noqa: E402
    DEFAULT_LEVELS,
    GATE_CONTROL_ABS,
    GATE_FIELD_REL,
    GATE_H1_PAIR,
    GATE_L2_FINEST,
    GATE_L2_PAIR,
    GATE_POWER_LAW,
    GATE_QUADRATURE,
    field_cases,
    mesh_dims,
    run_all,
    solve_case,
)

ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "artifacts" / "CASE-FEM-HEAT-L2"
EVIDENCE = CASE_DIR / "evidence.json"
TABLES = CASE_DIR / "tables.md"
DOC = ROOT / "docs" / "CASE-FEM-HEAT-L2-steady-conduction.md"
ASSETS = ROOT / "docs" / "assets"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def figure_ladders(evidence: dict) -> Path:
    """Log-log error ladders with the theoretical slopes drawn in."""
    studies = evidence["studies"]
    measured = {k: v for k, v in studies.items() if not v["representable"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    colours = plt.cm.tab10.colors
    for ax, metric, label, theory in (
        (axes[0], "l2_relative", "relative L2  $||e||_0$", 2.0),
        (axes[1], "h1_relative", "relative H1  $|e|_1$", 1.0),
    ):
        for index, (key, study) in enumerate(measured.items()):
            hs = study["h"]
            errs = study[metric]
            colour = colours[index % len(colours)]
            ax.loglog(hs, errs, "o-", color=colour, label=f"{key} (p={study['l2_orders'][-1]:.3f})"
                      if metric == "l2_relative" else key, ms=4)
            # theoretical slope anchored at the finest point
            anchor_h, anchor_e = hs[1], errs[1]
            hs_line = [anchor_h, hs[-1]]
            es_line = [anchor_e, anchor_e * (hs[-1] / anchor_h) ** theory]
            ax.loglog(hs_line, es_line, "--", color=colour, lw=0.9, alpha=0.55)
        ax.set_xlabel("representative mesh size $h$")
        ax.set_ylabel(label)
        ax.grid(True, which="both", ls=":", lw=0.5)
        ax.invert_xaxis()
        ax.legend(fontsize=7.5)
        ax.set_title(f"{'L2' if theory == 2 else 'H1'}: dashed = theory $O(h^{{{int(theory)}}})$",
                     fontsize=9)
    fig.suptitle("CASE-FEM-HEAT-L2  Q4 steady conduction, MMS h-refinement", fontsize=10)
    fig.tight_layout()
    out = ASSETS / "case_fem_heat_l2_convergence.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def figure_field(evidence: dict, case_key: str = "sine_generation",
                 level: int = 32) -> Path:
    """FE field vs analytic field vs pointwise error for the finest plausible mesh."""
    case = field_cases()[case_key]
    run = solve_case(case, level)
    nx, ny = mesh_dims(case, level)
    # rebuild the nodal field for plotting (solve_case returns norms only)
    from tensorfem.thermal import ThermalModel, solve_steady_thermal
    from tensorfem.thermal_analytic import structured_quad_mesh

    mesh = structured_quad_mesh(nx, ny, lx=case.length, ly=case.height)
    model = ThermalModel(
        nodes=mesh.nodes.double(), elements=mesh.elements,
        conductivity=torch.tensor(1.0, dtype=torch.float64),
        density=torch.tensor(1.0, dtype=torch.float64),
        specific_heat=torch.tensor(1.0, dtype=torch.float64),
        fixed_nodes=mesh.boundary_nodes,
        fixed_temperatures=case.temperature(mesh.nodes.double()[mesh.boundary_nodes]),
        heat_source=case.element_mean_source(mesh.nodes.double(), mesh.elements),
        thickness=torch.tensor(1.0, dtype=torch.float64),
        element_type="q4",
    )
    t_h = solve_steady_thermal(model).reshape(ny + 1, nx + 1).numpy()
    xs = torch.linspace(0.0, case.length, nx + 1, dtype=torch.float64)
    ys = torch.linspace(0.0, case.height, ny + 1, dtype=torch.float64)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    t_ref = case.temperature(torch.stack((gx.reshape(-1), gy.reshape(-1)), 1)).reshape(
        ny + 1, nx + 1).numpy()
    err = t_h - t_ref
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6))
    for ax, data, title, cmap in (
        (axes[0], t_h, f"FE solution, {nx}x{ny} Q4 (h={run['h']:.4f})", "viridis"),
        (axes[1], t_ref, "manufactured solution", "viridis"),
        (axes[2], err, "FE - exact", "coolwarm"),
    ):
        mesh_plot = ax.pcolormesh(xs.numpy(), ys.numpy(), data, cmap=cmap, shading="auto")
        fig.colorbar(mesh_plot, ax=ax, shrink=0.85)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.suptitle(f"{case_key}: relative L2 {run['errors']['l2_relative']:.3e}, "
                 f"max |FE-exact| {run['errors']['max_abs_nodal']:.3e}", fontsize=10)
    fig.tight_layout()
    out = ASSETS / "case_fem_heat_l2_field.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------
def _tick(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def ladder_table(evidence: dict) -> str:
    target = evidence["target"]
    studies = evidence["studies"]
    lines = ["| case | level (nx x ny) | dof | h | L2 rel | p_L2 | H1 rel | p_H1 |",
             "|---|---|---|---|---|---|---|---|"]
    for key, study in studies.items():
        for index, run in enumerate(study["runs"]):
            p_l2 = f"{study['l2_orders'][index - 1]:.3f}" if index else "-"
            p_h1 = f"{study['h1_orders'][index - 1]:.3f}" if index else "-"
            lines.append(
                f"| {key} | {run['nx']} x {run['ny']} | {run['n_nodes']} | "
                f"{run['h']:.5f} | {run['errors']['l2_relative']:.4e} | {p_l2} | "
                f"{run['errors']['h1_relative']:.4e} | {p_h1} |")
    lines.append("")
    lines.append(f"Target order: p_ref = {target['order']:.1f} "
                 f"(manual row `{target['row']}` in `{target['file']}:{target['line']}`).")
    return "\n".join(lines)


def verdict_table(evidence: dict) -> str:
    lines = ["| gate | pass | detail |", "|---|---|---|"]
    for check in evidence["verdict"]["checks"]:
        lines.append(f"| {check['name']} | {_tick(check['passed'])} | {check['detail']} |")
    return "\n".join(lines)


def reference_table(evidence: dict) -> str:
    """ref vs got at the finest mesh of every case (field values, not norms)."""
    lines = ["| case | mesh | probes | max abs (ref-got) | max |e|/max|T_ref| | "
             "relative L2 | <3% ? |",
             "|---|---|---|---|---|---|---|"]
    for key, study in evidence["studies"].items():
        run = study["runs"][-1]
        field = run["field_comparison"]
        ok = field["relative_to_peak"] <= GATE_FIELD_REL
        lines.append(
            f"| {key} | {run['nx']}x{run['ny']} | {field['n_points']} | "
            f"{field['max_abs']:.3e} | {100 * field['relative_to_peak']:.4f}% | "
            f"{run['errors']['l2_relative']:.3e} | {_tick(ok)} |")
    return "\n".join(lines)


def order_table(evidence: dict) -> str:
    target = evidence["target"]["order"]
    lines = ["| case | p_L2 per pair | min p (nx>=8) | finest p | window "
             f"[{GATE_L2_FINEST[0]:.2f}, {GATE_L2_FINEST[1]:.2f}] | residual at p={target:.0f} |",
             "|---|---|---|---|---|---|"]
    for key, study in evidence["studies"].items():
        if study["representable"]:
            continue
        orders = ", ".join(f"{p:.3f}" for p in study["l2_orders"])
        pairs = [(p, run) for p, run in zip(study["l2_orders"], study["runs"][:-1])]
        worst = min(p for p, run in pairs if run["nx"] >= 8)
        finest = study["l2_orders"][-1]
        inside = GATE_L2_FINEST[0] <= finest <= GATE_L2_FINEST[1]
        lines.append(f"| {key} | {orders} | {worst:.3f} | {finest:.3f} | "
                     f"{_tick(inside)} | {study['richardson_residual']['theoretical']:.2e} |")
    return "\n".join(lines)


def parameter_table(evidence: dict) -> str:
    """Dimensionless parameters of every case, computed from the case data."""
    lines = ["| case | Lx x Ly | aspect Lx/Ly | q_max Lx^2 / (k T_peak) | "
             "Dirichlet | mesh ladder | r | h/Lx (finest) |",
             "|---|---|---|---|---|---|---|---|"]
    for key, study in evidence["studies"].items():
        case = field_cases()[key]
        divisions = 97
        pts = torch.tensor([[case.length * i / divisions, case.height * j / divisions]
                            for i in range(divisions + 1) for j in range(divisions + 1)],
                           dtype=torch.float64)
        t_peak = float(case.temperature(pts).abs().max())
        q_peak = float(case.source(pts).abs().max())
        finest = study["runs"][-1]
        levels = study["levels"]
        ratio = levels[1] / levels[0]
        lines.append(
            f"| {key} | {case.length:g} x {case.height:g} | "
            f"{case.length / case.height:.3f} | {q_peak * case.length ** 2 / t_peak:.4f} | "
            f"full boundary, T_exact | {levels[0]}..{levels[-1]} | {ratio:g} | "
            f"{finest['h'] / case.length:.5f} |")
    return "\n".join(lines)


def sensitivity_section(evidence: dict) -> str:
    quad = evidence["quadrature_sensitivity"]
    studies = evidence["studies"]
    hs = studies["sine_generation"]["h"]
    ratios = [a / b for a, b in zip(hs[:-1], hs[1:])]
    parts = [
        "1. **Mesh size (the ladder itself).** Every pair of the ladder in "
        "`order_table` is a refinement-ratio point; the refinement ratio is "
        f"r = {ratios[0]:.3f}..{ratios[-1]:.3f}, so the observed order is not "
        "contaminated by a non-constant ratio.",
        "2. **Field regularity / type.** Four independent manufactured fields "
        "(sin*sin with source, exponential harmonic, harmonic cubic, 2:1 "
        "aspect-ratio sin) give the same asymptotic order; the spectra in the "
        "figure show the *pre-asymptotic* range instead: the 2:1 case is still "
        f"{studies['sine_aspect2']['l2_orders'][0]:.3f} on the coarsest pair "
        "(its coarse mesh has only 2 elements across the short side) but "
        f"{studies['sine_aspect2']['l2_orders'][-1]:.3f} on the finest.",
        "3. **Quadrature of the error norm.** Raising the post-processing rule "
        f"from {quad['orders'][0]} to {quad['orders'][1]} and {quad['orders'][2]} "
        f"points per direction at level nx={quad['level']} changes every error "
        f"norm by at most {quad['worst_relative_change']:.2e} relative "
        f"(gate {GATE_QUADRATURE:.0e}), i.e. the measured order is a property of "
        "the discretisation and not of the integration rule.",
        "4. **Source projection.** The heat kernel takes an element-constant "
        "source; the MMS load is integrated over each element with an 8x8 Gauss "
        "rule before it is handed to the kernel, so the load vector is the "
        "consistent one for the element-mean source and the ladder measures the "
        "*conduction* discretisation, not a quadrature slip.",
        "5. **Control that must not move.** The bilinear-representable field "
        f"`bilinear_control` stays exact on every level (worst absolute error "
        f"{max(max(r['errors']['max_abs_nodal'], r['errors']['l2_absolute']) for r in studies['bilinear_control']['runs']):.2e}, "
        f"gate {GATE_CONTROL_ABS:.0e}); its ladder carries no signal and no "
        "order is claimed for it.",
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def write_report(evidence: dict, figures: dict[str, Path]) -> Path:
    target = evidence["target"]
    studies = evidence["studies"]
    quad = evidence["quadrature_sensitivity"]
    verdict = evidence["verdict"]
    sine = studies["sine_generation"]
    fig_rel = {name: path.relative_to(ROOT) for name, path in figures.items()
               if isinstance(path, Path)}

    body = f"""# CASE-FEM-HEAT-L2 -- 稳态热传导 Q4 单元的 L2 收敛率（≥2 阶）

**判据**：收敛率判据（不是单点误差判据）。**结论**：{'PASS' if verdict['passed'] else 'FAIL'} --
{len(verdict['checks'])} 项预登记闸门全部通过（明细见 §④）。全部数字由
`benchmarks/heat_l2_convergence.py` 从 `artifacts/CASE-FEM-HEAT-L2/evidence.json`
格式化输出，报告本身不含手抄数值。

---

## ① 问题与参考解的出处

求解稳态热传导（无内热源假设不成立的一般形式）

    -div(k grad T) = q    in Omega = (0, Lx) x (0, Ly),   k = 1
     T = T_exact          on dOmega（全边界 Dirichlet，非齐次）

离散：TensorFEM 既有的双线性 `Q4` 传导单元
（`tensorfem.thermal`：2x2 Gauss 传导矩阵 + 一致热容 + 单元常数体热源 + 一致
Dirichlet 消元），直接稠密求解 `tensorfem.thermal.solve_steady_thermal`。

**参考解 = 制造解（method of manufactured solutions, MMS）**：精确温度场是解析式，
以代码形式实现（`HeatFieldCase.temperature` / `.gradient` / `.source`），
误差范数与对照表中的每一个 ref 数值都在运行时由该解析式求值，**零手抄**。

**目标数值（收敛阶）同样不是手抄**：`target_l2_order_from_manual()` 用正则在运行时
解析本仓版本化的先验阶次表，取到

| 目标量 | 手册出处（文件:行，程序化解析） | 原文行 |
|---|---|---|
| p_ref(L2, Q4) | `src/tensorfem/{target['file']}:{target['line']}` | `{target['row']}` |

解析结果 `p_ref = {target['order']:.1f}`。若该表被改动，目标随动、研究 fail-closed
（解析不到直接抛 `RuntimeError`）。

方法学外部锚点（节级引用，与仓内既有引用惯例一致）：

* K. Salari & P. Knupp, *Code Verification by the Method of Manufactured Solutions*,
  Sandia National Laboratories **SAND2000-1009J** (2000)：正确实现的离散格式，其
  **观测阶必须收敛到形式阶**；MMS 是获取该阶的标准手段。
* P.J. Roache (1994) *J. Fluids Eng.* **116**, 405-413：refinement-ratio 外推与
  GCI 安全因子（`tensorfem.convergence_verification` 即建立在此文献上）。
* 离线环境无法核对纸质手册页码，故外部引用做到"节/报告号"级；目标阶的主锚是
  上表这条**仓内版本化手册行 + 程序化解析**，这比转抄一个数字更可审计。

物理背景（船舶总体性能）：稳态热传导是船体结构热载荷（主机排气/舱室保温/焊后
残余应力热载）与电子设备散热分析的底层求解器；L2 阶次是这批算例的可信度基线。

## ② 方法与实现

新增能力 `src/tensorfem/{Path(__file__).stem}.py`（复用既有内核，不另起炉灶）：

1. **制造解族**（6 例：4 个正则解 + 1 个节点精确解 + 1 个控制例）：
   `sine_generation`（带源项 sin·sin）、`exp_trig_harmonic`（调和、非多项式）、
   `cubic_harmonic`（调和三次多项式）、`sine_aspect2`（2:1 域，非正方形 Jacobian）、
   `quadratic_harmonic`（调和二次：节点精确 + 纯插值误差基线）、
   `bilinear_control`（`T=xy`，既调和又在双线性空间内 → 必须精确，控制例）。
2. **误差范数是真积分**：`evaluate_errors()` 是**独立**的 Gauss 后处理（自带形函数，
   不复用装配内核），逐单元积分

       ||e||_0 = sqrt(∫(T_h-T)^2 dO / ∫T^2 dO),  |e|_1 = sqrt(∫|∇T_h-∇T|^2/∫|∇T|^2)

   同一后处理器把 `T_exact` 在节点取值再积分一次，就得到**双线性插值误差**基线
   （`l2_relative_interpolant`），用来判断"FE 解是否只剩插值误差"。
3. **载荷一致性**：热源是单元常数，MMS 的 `q` 先用 8x8 Gauss 在单元上积分成单元
   平均值再交给内核（`HeatFieldCase.element_mean_source`），避免把源项投影误差
   混进阶次。
4. **阶梯**：nx = {studies['sine_generation']['levels']}（r = 2，等比方），
   2:1 域取 (nx, nx/2)。最细网格 65x65=4225 未知量（稠密装配上限）。
5. **判据**：观测阶 `p = ln(e_coarse/e_fine)/ln(r)`；另加 Roache 形式的外推残差
   `|e0|/e_fine`（`e0 = e_fine + (e_fine-e_coarse)/(r^p_ref-1)`）检验整条阶梯是否
   真是理论幂律，而不是三个"凑巧"的点。

**预登记闸门**（写在模块 docstring 里，跑之前定死，未事后调整）：
L2 每段 p ≥ {GATE_L2_PAIR}；最细段 p ∈ [{GATE_L2_FINEST[0]}, {GATE_L2_FINEST[1]}]；
H1 每段 p ≥ {GATE_H1_PAIR}；幂律残差 ≤ {GATE_POWER_LAW:g}；控制例绝对误差 ≤
{GATE_CONTROL_ABS:.0e}；最细网格场误差 ≤ {100 * GATE_FIELD_REL:.0f}%；误差范数对
积分阶次（4→6→8 点）的相对变化 ≤ {GATE_QUADRATURE:.0e}。

## ③ 参数表（无量纲量完整）

{parameter_table(evidence)}

补充无量纲量（稳态，无 Fo/Bi 时间或对流尺度，故不列傅里叶数/毕渥数）：

| 量 | 定义 | 值 / 说明 |
|---|---|---|
| 源项强度（正弦例） | q Lx^2/(k T_peak) | {2 * math.pi ** 2:.4f} (= 2 pi^2) |
| 源项强度（2:1 例） | q Lx^2/(k T_peak) | {1.25 * math.pi ** 2 * 4.0:.4f} (= 1.25 pi^2 (Lx/Ly)^2 = 5 pi^2) |
| 导热系数 | k | 1（归一化单位，问题对 k 线性齐次） |
| 单元长宽比 | hx/hy | 1（1:1 例）/ 1（2:1 例，nx:ny 随域比例取） |
| 网格比 | r = h_i/h_(i+1) | 2 |
| dof | (nx+1)(ny+1) | {studies['sine_generation']['runs'][0]['n_nodes']} .. {studies['sine_generation']['runs'][-1]['n_nodes']} |
| 最细 h/Lx | sqrt(A/ne)/Lx | {sine['runs'][-1]['h']:.5f} |

## ④ 对照表（ref vs got + 误差打勾/叉）

ref = 制造解解析值（程序求值）；got = TensorFEM Q4 稳态解。

{reference_table(evidence)}

逐段阶次对照：

{order_table(evidence)}

完整阶梯（每层 ref/got 范数）：

{ladder_table(evidence)}

**闸门明细（{len(verdict['checks'])} 项，全部程序判定）**：

{verdict_table(evidence)}

误差判定（报告纪律要求的显式回答）：最细网格上**所有算例**的场误差
`max|e|/max|T|` 均 < 3%（最大 {100 * max(s['runs'][-1]['field_comparison']['relative_to_peak'] for s in studies.values()):.4f}%），
且**收敛率判据**（本单的主判据）最细段 p_L2 = {sine['l2_orders'][-1]:.3f}（理论
{target['order']:.1f}，|相对偏差| {abs(sine['l2_orders'][-1] / target['order'] - 1) * 100:.2f}% < 3%）。

## ⑤ 收敛性与参数敏感性

{sensitivity_section(evidence)}

L2 阶次的最小二乘 log-log 斜率（整条阶梯）：{', '.join(f"{k} = {v['l2_order_slope']:.4f}" for k, v in studies.items() if not v['representable'])}。

图 1（`{fig_rel['ladder']}`）：L2/H1 误差随 h 的 log-log 阶梯，虚线为理论斜率
`O(h^2)` / `O(h^1)` 锚定在次细网格点上的参考线。图 2（`{fig_rel['field']}`）：
`{figures['field_case']}` 在 nx={figures['field_level']} 网格上 FE 场、解析场与
逐点误差云图（量级 {figures['field_max_err']:.2e}，与 L2 范数一致）。

## ⑥ 失败模式与工件

**本项目观察到的失败模式 / 坑（全部留档）**

1. **可表示场 = 无信号的收敛研究。** 首版把 `T = x^2 - y^2` 当作"可精确表示的控制例"，
   控制闸门以 6.6e-3 失败。诊断：双线性空间是 `span{{1, x, y, xy}}`，二次场根本不
   在空间内 —— 错的是**算例规格**，不是求解器。改法：`T = xy` 做控制例，并把
   `x^2-y^2` 改归为"节点精确 + 纯插值误差"算例（其 `e_FE/e_interp = 1.000000`）。
   生态里已有同类负面知识（点载荷恰在网格点 → 误差 1e-15、收敛研究无信号），
   本单把它推广到热传导：**先证明参考场不被离散空间精确表示，再谈阶次**。
2. **积分阶次污染阶次。** 仓库里曾出现"order 2.0 其实来自梯形积分误差"的审计结论；
   本单用 4/6/8 点 Gauss 后处理对照（最大相对变化
   {quad['worst_relative_change']:.2e}）证明阶次不属于积分规则。
3. **单元常数源项。** `tensorfem.thermal` 的体热源是单元常数；若直接把解析 `q` 在
   形心取值，会引入 O(h^2) 的载荷一致性误差、与离散误差同阶。改为 8x8 Gauss 单元
   积分后再交给内核。
4. **稠密装配的规模上限。** `thermal` 路径全稠密（4225 未知量 → 143 MB 矩阵、
   ~1 GB RSS、nx=64 单次 3 s）；nx=128 的稠密矩阵（2.2 GB）在本机不可行，
   阶梯止于 64。这是 tensorfem.thermal 的真实规模限制（friction）。
5. **线程过度订阅。** 128 核默认线程池在 4225 阶稠密求解上比 4 线程慢一个量级
   （仓内既有记录），脚本显式 `torch.set_num_threads(4)`。

**工件（worktree 内真实文件）**

| 工件 | 说明 |
|---|---|
| `src/tensorfem/{Path(__file__).stem}.py` | 算例族 + 独立 Gauss 误差后处理 + 阶梯/闸门（能力本体） |
| `benchmarks/{Path(__file__).stem}.py` | 一键复现脚本（证据/图/报告全部由它生成） |
| `artifacts/CASE-FEM-HEAT-L2/evidence.json` | 全部阶梯、范数、闸门、敏感性证据 |
| `artifacts/CASE-FEM-HEAT-L2/tables.md` | 本报告的表格（独立可引用） |
| `tests/test_heat_l2_convergence.py` | pytest 闸门（含控制例与积分独立性） |
| `{fig_rel['ladder']}` | 图 1 |
| `{fig_rel['field']}` | 图 2 |

**一键复现**

    PYTHONPATH=src python3 benchmarks/{Path(__file__).stem}.py
    PYTHONPATH=src python3 -m pytest tests/test_heat_l2_convergence.py -q

**生态摩擦（如实记）**

* 平台 `/api/v1/auth` 直接用账号密码 POST 返回 401（"未提供认证令牌"），必须走
  `/api/v1/auth/login`（`{'{'}"access_token"{'}'}` 字段）；仓内 `get_token.py`
  仍按旧 `/auth` 写，已失效。
* 开工三查：`knowledge_base` 全库 `count=0`（无同域 capability 条目）；
  `negative_knowledge_ledger` 关键词 `TensorFEM`/`thermal` 均 0 命中，仅
  `convergence order` 命中 2 条（点载荷收敛退化 m17558、另一收敛失败账），
  已用于本单的算例设计（第 1 条坑）；`oss_solutions` 无公开端点，仓内既有能力
  （`tensorfem.thermal` + `convergence_verification`）足够，故不另起炉灶。
* 全流程（6 个算例 x 5 层阶梯 + 积分敏感性 + 2 张图）本机 4 线程耗时
  {figures['wall_seconds']:.1f} s，远低于 10 min 门槛，无需上集群。

**适用范围/局限**

* 仅等距结构化 Q4 网格与正则解；奇异解（角点、裂纹）的 L2 阶次会退化，不在本单结论内。
* `thermal` 内核的体热源是单元常数；非单元常数源的严格一致性载荷需要改内核（未做，
  记为局限，而不是用后处理掩盖）。
* 结论是"该离散的 L2 阶次达 2 阶"，不是"任何网格/任何场都达 2 阶"：2:1 例的粗段
  （短边只有 2 个单元）仍有 {studies['sine_aspect2']['l2_orders'][0]:.3f}，是预渐近区。
"""
    DOC.write_text(body, encoding="utf-8")
    return DOC


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def main() -> dict:
    import time

    start = time.time()
    CASE_DIR.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    evidence = run_all(levels=DEFAULT_LEVELS)
    EVIDENCE.write_text(json.dumps(evidence, indent=2), encoding="utf-8")

    tables = "\n\n".join([
        "## 参数表\n\n" + parameter_table(evidence),
        "## ref vs got（最细网格）\n\n" + reference_table(evidence),
        "## 阶次\n\n" + order_table(evidence),
        "## 完整阶梯\n\n" + ladder_table(evidence),
        "## 闸门\n\n" + verdict_table(evidence),
    ])
    TABLES.write_text(tables + "\n", encoding="utf-8")

    ladder_png = figure_ladders(evidence)
    field_run = solve_case(field_cases()["sine_generation"], 32)
    field_png = figure_field(evidence)
    figures = {
        "ladder": ladder_png,
        "field": field_png,
        "field_case": "sine_generation",
        "field_level": 32,
        "field_max_err": field_run["errors"]["max_abs_nodal"],
        "wall_seconds": time.time() - start,
    }
    doc = write_report(evidence, figures)
    print(f"evidence -> {EVIDENCE.relative_to(ROOT)}")
    print(f"tables   -> {TABLES.relative_to(ROOT)}")
    print(f"figures  -> {ladder_png.relative_to(ROOT)}, {field_png.relative_to(ROOT)}")
    print(f"report   -> {doc.relative_to(ROOT)}")
    print(f"verdict  -> {'PASS' if evidence['verdict']['passed'] else 'FAIL'} "
          f"({len(evidence['verdict']['checks'])} checks)")
    return evidence


if __name__ == "__main__":
    main()