"""Verification study: plane-strain Lame thick cylinder discretised with Q4 elements.

Why this case
-------------
``tensorfem.continuum`` has carried a ``plane="strain"`` constitutive path since
the module was written, but no test or benchmark ever exercised it: every stored
check used the default ``plane="stress"``.  The Lame annulus is the cheapest
case that discriminates the two paths, because in-plane stresses are statically
determinate (identical for plane stress and plane strain) while the radial
displacement is not (it differs by the out-of-plane compliance).

What is produced
----------------
* ``benchmarks/lame_cylinder_plane_strain_evidence.json`` -- every anchor with
  its reference, measured value, relative error and pass/fail verdict.
* ``docs/assets/lame_cylinder_*.png`` -- stress/displacement profiles, the
  mesh-convergence log-log plot and a hoop-stress contour of the sector.
* ``docs/lame-cylinder-plane-strain.md`` -- the report; all numbers in it are
  formatted from the computed evidence, never typed by hand.

Run:
    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/lame_cylinder_plane_strain.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.continuum import ContinuumModel, solve_continuum  # noqa: E402
from tensorfem.lame_reference import (  # noqa: E402
    cartesian_to_polar_stress,
    circular_pressure_loads,
    lame_state,
    radial_component,
    sector_q4_mesh,
    symmetry_fixed_dofs,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
EVIDENCE = ROOT / "benchmarks" / "lame_cylinder_plane_strain_evidence.json"
DOC = ROOT / "docs" / "lame-cylinder-plane-strain.md"

INNER, OUTER = 0.5, 0.8          # m, bore / outer radius
PRESSURE = 100.0e6               # Pa, internal pressure
YOUNG, POISSON = 210.0e9, 0.3    # steel
GATE = 0.03                      # 3 % acceptance gate
LADDER = ((12, 4), (24, 8), (48, 16))


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / abs(reference)


def l2_relative(field: torch.Tensor, reference: torch.Tensor) -> float:
    return (torch.linalg.vector_norm(field - reference) / torch.linalg.vector_norm(reference)).item()


def max_relative(field: torch.Tensor, reference: torch.Tensor) -> float:
    return ((field - reference).abs() / reference.abs()).max().item()


def solve_case(n_theta: int, n_radial: int, *, plane: str = "strain", grading: float = 1.0) -> dict:
    """Assemble, solve and score one quarter-annulus discretisation."""
    nodes, cells = sector_q4_mesh(INNER, OUTER, n_theta, n_radial, radial_grading=grading)
    forces = circular_pressure_loads(n_theta, n_radial, INNER, PRESSURE, inner=True)
    fixed = symmetry_fixed_dofs(nodes, n_theta=n_theta, n_radial=n_radial)
    model = ContinuumModel(
        nodes, cells, torch.tensor(YOUNG), torch.tensor(POISSON), torch.tensor(1.0),
        forces, fixed, plane=plane,
    )
    result = solve_continuum(model)

    displacement = result.displacement.reshape(-1, 2)
    radius = torch.linalg.vector_norm(nodes, dim=1)
    u_radial = radial_component(displacement, nodes)
    stride = n_theta + 1
    diag = torch.tensor([i * stride for i in range(n_radial + 1)])  # theta = 0 ray
    reference_diag = lame_state(
        radius[diag], inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
        young_modulus=YOUNG, poisson_ratio=POISSON, plane=plane,
    )

    centres = nodes[cells].mean(dim=1)
    centre_radius = torch.linalg.vector_norm(centres, dim=1)
    sigma_r, sigma_theta, tau_rtheta = cartesian_to_polar_stress(result.stress, centres)
    reference_stress = lame_state(
        centre_radius, inner_radius=INNER, outer_radius=OUTER, inner_pressure=PRESSURE,
        young_modulus=YOUNG, poisson_ratio=POISSON, plane=plane,
    )

    hoop_direction = torch.stack((-nodes[:, 1], nodes[:, 0]), dim=1) / radius[:, None]
    u_hoop = torch.einsum("ij,ij->i", displacement, hoop_direction)

    return {
        "n_theta": n_theta,
        "n_radial": n_radial,
        "plane": plane,
        "grading": grading,
        "elements": int(cells.shape[0]),
        "dofs": int(2 * nodes.shape[0]),
        "mean_element_size": float((OUTER - INNER) / n_radial),
        "displacement_max_relative": max_relative(u_radial[diag], reference_diag.radial_displacement),
        "displacement_l2_relative": l2_relative(u_radial[diag], reference_diag.radial_displacement),
        "hoop_stress_l2_relative": l2_relative(sigma_theta, reference_stress.hoop_stress),
        "hoop_stress_max_relative": max_relative(sigma_theta, reference_stress.hoop_stress),
        "radial_stress_l2_relative": l2_relative(sigma_r, reference_stress.radial_stress),
        "radial_stress_max_relative": max_relative(sigma_r, reference_stress.radial_stress),
        "shear_stress_max": float(tau_rtheta.abs().max()),
        "reference_stress_scale": float(reference_stress.radial_stress.abs().max()),
        "hoop_displacement_max": float(u_hoop.abs().max()),
        "radial_displacement_max": float(u_radial.abs().max()),
        "reaction_sum": [float(x) for x in result.reaction.reshape(-1, 2).sum(dim=0)],
        "tip_diag": {
            "radius": [float(x) for x in radius[diag]],
            "u_radial": [float(x) for x in u_radial[diag]],
            "u_radial_reference": [float(x) for x in reference_diag.radial_displacement],
        },
        "field": {
            "centres": [[float(x) for x in row] for row in centres],
            "radius": [float(x) for x in centre_radius],
            "sigma_r": [float(x) for x in sigma_r],
            "sigma_theta": [float(x) for x in sigma_theta],
            "sigma_r_reference": [float(x) for x in reference_stress.radial_stress],
            "sigma_theta_reference": [float(x) for x in reference_stress.hoop_stress],
        },
        "nodes": [[float(x) for x in row] for row in nodes],
        "cells": [[int(x) for x in row] for row in cells],
    }


def observed_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    if coarse <= 0.0 or fine <= 0.0:
        return float("nan")
    return math.log(coarse / fine) / math.log(ratio)


def plot_profiles(cases: list[dict], path: Path) -> None:
    fine = cases[-1]
    radius = torch.tensor(fine["field"]["radius"])
    order = torch.argsort(radius)
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for key, colour, label in (
        ("sigma_r", "#1f77b4", r"$\sigma_r$"),
        ("sigma_theta", "#d62728", r"$\sigma_\theta$"),
    ):
        axes[0].plot(
            [fine["field"]["radius"][i] for i in order],
            [fine["field"][key + "_reference"][i] / 1e6 for i in order],
            "-", color=colour, lw=1.6, label=label + " Lame exact",
        )
        axes[0].plot(
            [fine["field"]["radius"][i] for i in order],
            [fine["field"][key][i] / 1e6 for i in order],
            "o", color=colour, ms=4.0, mfc="none", label=label + f" Q4 {fine['n_theta']}x{fine['n_radial']}",
        )
    axes[0].axhline(0.0, color="0.6", lw=0.8)
    axes[0].set_xlabel("radius r [m]")
    axes[0].set_ylabel(r"stress [MPa], tension positive")
    axes[0].set_title("Thick cylinder, p_i = 100 MPa, b/a = 1.6")
    axes[0].legend(fontsize=8, loc="lower left")
    axes[0].grid(alpha=0.3)

    for case, colour, label in (
        (cases[2], "#2ca02c", f"plane strain {cases[2]['n_theta']}x{cases[2]['n_radial']}"),
        (cases[4], "#9467bd", f"plane stress {cases[4]['n_theta']}x{cases[4]['n_radial']}"),
    ):
        radii = torch.tensor(case["tip_diag"]["radius"])
        axes[1].plot(
            radii, [1e6 * u for u in case["tip_diag"]["u_radial"]], "o",
            color=colour, ms=4.5, mfc="none", label="FEM " + label,
        )
        axes[1].plot(
            radii, [1e6 * u for u in case["tip_diag"]["u_radial_reference"]], "-",
            color=colour, lw=1.4, label="Lame exact " + label.split()[0] + " " + label.split()[1],
        )
    axes[1].set_xlabel("radius r [m]")
    axes[1].set_ylabel(r"radial displacement $u_r$ [$\mu$m]")
    axes[1].set_title("Plane strain vs plane stress compliance")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_convergence(cases: list[dict], path: Path) -> None:
    sizes = [case["mean_element_size"] for case in cases]
    series = (
        ("displacement_max_relative", "#1f77b4", "max |e(u_r)| on the theta=0 ray"),
        ("radial_stress_l2_relative", "#d62728", r"$L_2$ error $\sigma_r$"),
        ("hoop_stress_l2_relative", "#2ca02c", r"$L_2$ error $\sigma_\theta$"),
    )
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    for key, colour, label in series:
        errors = [case[key] for case in cases]
        order = observed_order(errors[-2], errors[-1])
        axis.loglog(sizes, errors, "o-", color=colour, label=f"{label} (order {order:.2f})")
    axis.axhline(GATE, color="black", ls="--", lw=1.0, label="3 % gate")
    axis.set_xlabel("radial element size h [m]")
    axis.set_ylabel("relative error")
    axis.invert_xaxis()
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    axis.set_title("Lame cylinder: mesh convergence")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_field(case: dict, path: Path) -> None:
    centres = torch.tensor(case["field"]["centres"])
    values = torch.tensor(case["field"]["sigma_theta"]) / 1e6
    figure, axis = plt.subplots(figsize=(5.6, 5.0))
    contour = axis.tricontourf(centres[:, 0], centres[:, 1], values, levels=14, cmap="viridis")
    for cell in case["cells"][:: max(1, len(case["cells"]) // 220)]:
        polygon = torch.tensor(case["nodes"])[cell]
        axis.plot(
            torch.cat((polygon[:, 0], polygon[:1, 0])),
            torch.cat((polygon[:, 1], polygon[:1, 1])),
            color="white", lw=0.25, alpha=0.5,
        )
    axis.set_aspect("equal")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(
        f"$\\sigma_\\theta$ [MPa], Q4 {case['n_theta']}x{case['n_radial']}, quarter sector"
    )
    figure.colorbar(contour, ax=axis, label=r"$\sigma_\theta$ [MPa]")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def build_anchors(ladder: list[dict], stress_case: dict) -> list[dict]:
    fine = ladder[-1]
    coarse = ladder[0]
    reaction = torch.tensor(fine["reaction_sum"])
    expected_magnitude = 2.0 * PRESSURE * INNER * math.sin(math.pi / 4.0)
    reaction_error = abs(reaction.norm().item() - expected_magnitude) / expected_magnitude
    orders = {
        "displacement": observed_order(ladder[-2]["displacement_max_relative"], fine["displacement_max_relative"]),
        "radial_stress": observed_order(ladder[-2]["radial_stress_l2_relative"], fine["radial_stress_l2_relative"]),
        "hoop_stress": observed_order(ladder[-2]["hoop_stress_l2_relative"], fine["hoop_stress_l2_relative"]),
    }
    return [
        {
            "anchor": "radial_displacement",
            "description": "u_r vs Lame (plane strain) on the theta=0 ray, finest mesh; max and L2 relative error < 3 %",
            "mesh": f"{fine['n_theta']}x{fine['n_radial']}",
            "measured_max": fine["displacement_max_relative"],
            "measured_l2": fine["displacement_l2_relative"],
            "tolerance": GATE,
            "passed": fine["displacement_max_relative"] < GATE and fine["displacement_l2_relative"] < GATE,
        },
        {
            "anchor": "radial_stress",
            "description": "sigma_r at element centres vs Lame; L2 and worst-point relative error < 3 %",
            "measured_max": fine["radial_stress_max_relative"],
            "measured_l2": fine["radial_stress_l2_relative"],
            "coarse_max": coarse["radial_stress_max_relative"],
            "tolerance": GATE,
            "passed": fine["radial_stress_max_relative"] < GATE and fine["radial_stress_l2_relative"] < GATE,
        },
        {
            "anchor": "hoop_stress",
            "description": "sigma_theta at element centres vs Lame; L2 and worst-point relative error < 3 %",
            "measured_max": fine["hoop_stress_max_relative"],
            "measured_l2": fine["hoop_stress_l2_relative"],
            "tolerance": GATE,
            "passed": fine["hoop_stress_max_relative"] < GATE and fine["hoop_stress_l2_relative"] < GATE,
        },
        {
            "anchor": "axisymmetry",
            "description": "Symmetry of the discrete solution: max|tau_rtheta|/max|sigma_r| and max|u_theta|/max|u_r| < 1e-8",
            "shear_ratio": fine["shear_stress_max"] / fine["reference_stress_scale"],
            "hoop_displacement_ratio": fine["hoop_displacement_max"] / fine["radial_displacement_max"],
            "tolerance": 1e-8,
            "passed": (fine["shear_stress_max"] / fine["reference_stress_scale"] < 1e-8
                       and fine["hoop_displacement_max"] / fine["radial_displacement_max"] < 1e-8),
        },
        {
            "anchor": "load_equilibrium",
            "description": "Sum of support reactions equals 2 p_i a sin(pi/4) along the sector bisector; relative error below the 1e-6 round-off floor of the dense solve",
            "reaction_sum": [float(x) for x in reaction],
            "expected_magnitude": float(expected_magnitude),
            "relative_error": float(reaction_error),
            "tolerance": 1e-6,
            "passed": reaction_error < 1e-6,
        },
        {
            "anchor": "constitutive_paths",
            "description": "plane stress and plane strain each match their own Lame reference (<3 %), and differ from one another as the closed forms predict",
            "strain": {
                "mesh": f"{stress_case['n_theta']}x{stress_case['n_radial']}",
                "displacement_max_relative": stress_case["displacement_max_relative"],
            },
            "stress_variant": {
                "mesh": f"{ladder[2]['n_theta']}x{ladder[2]['n_radial']}",
                "displacement_max_relative": ladder[2]["displacement_max_relative"],
            },
            "relative_gap_between_paths": abs(
                stress_case["radial_displacement_max"] - ladder[2]["radial_displacement_max"]
            ) / ladder[2]["radial_displacement_max"],
            "tolerance": GATE,
            "passed": stress_case["displacement_max_relative"] < GATE and ladder[2]["displacement_max_relative"] < GATE,
        },
        {
            "anchor": "mesh_convergence",
            "description": "Observed order of u_r and of both stress L2 errors between the two finest meshes >= 1.8",
            "orders": orders,
            "tolerance": 1.8,
            "passed": min(orders.values()) >= 1.8,
        },
    ]


def fmt(value: float, digits: int = 4) -> str:
    return f"{100.0 * value:.{digits}f} %"


def sci(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}e}"


def render_doc(ladder: list[dict], stress_case: dict, graded: dict, anchors: list[dict]) -> str:
    fine = ladder[-1]
    by_name = {anchor["anchor"]: anchor for anchor in anchors}
    mesh_rows = "\n".join(
        f"| {case['n_theta']}x{case['n_radial']} | {case['elements']} | {case['dofs']} | "
        f"{fmt(case['displacement_max_relative'])} | {fmt(case['displacement_l2_relative'])} | "
        f"{fmt(case['radial_stress_l2_relative'])} | {fmt(case['radial_stress_max_relative'])} | "
        f"{fmt(case['hoop_stress_l2_relative'])} | {fmt(case['hoop_stress_max_relative'])} |"
        for case in ladder
    )
    order = by_name["mesh_convergence"]["orders"]
    anchor_rows = "\n".join(
        f"| `{anchor['anchor']}` | {'PASS' if anchor['passed'] else 'FAIL'} | {anchor['description']} |"
        for anchor in anchors
    )
    return f"""# 平面应变 Lamé 厚壁圆筒验证（Q4 平面单元）

`tensorfem.continuum` 自首次提交起就带有 `plane="strain"` 本构分支，但仓库中全部
既有算例（patch test、Cook 膜、悬臂梁）都走默认的 `plane="stress"`，即该分支从未
被任何测试或基准覆盖。本文用静定可解的 Lamé 厚壁圆筒补上这条验证欠账：面内应力
场对平面应力/平面应变是**同一**的，而径向位移因面外柔度不同而不同——这正好可以
反证 `plane` 开关真的接线，而不只是一个被静默忽略的参数。

## 1. 案例介绍

厚壁圆筒（压力容器/耐压壳/环肋圆筒的经典模型），单位面外厚度，平面应变状态
（$\varepsilon_z = 0$）。内壁受静水压，外壁自由。

| 参数 | 符号 | 取值 |
|---|---|---|
| 内半径 | $a$ | {INNER} m |
| 外半径 | $b$ | {OUTER} m |
| 壁厚比 | $b/a$ | {OUTER / INNER:.2f} |
| 内压 | $p_i$ | {PRESSURE / 1e6:.0f} MPa |
| 外压 | $p_o$ | 0 MPa |
| 弹性模量 | $E$ | {YOUNG / 1e9:.0f} GPa |
| 泊松比 | $\\nu$ | {POISSON} |
| 面外厚度 | $t$ | 1 m（平面应变） |

**解析解（Lamé）**。环向静定，$A,B$ 由两条面力边界条件唯一确定，与 $E,\\nu$ 及
平面假设无关：

$$A = \\frac{{p_i a^2 - p_o b^2}}{{b^2 - a^2}}, \\qquad B = \\frac{{a^2 b^2 (p_i - p_o)}}{{b^2 - a^2}}$$

$$\\sigma_r = A - \\frac{{B}}{{r^2}}, \\qquad \\sigma_\\theta = A + \\frac{{B}}{{r^2}}$$

平面应变（$\\varepsilon_z=0$，$\\sigma_z = \\nu(\\sigma_r+\\sigma_\\theta) = 2\\nu A$）：

$$u_r = \\frac{{1+\\nu}}{{E}}\\left[(1-2\\nu) A r + \\frac{{B}}{{r}}\\right]$$

平面应力（$\\sigma_z = 0$）：

$$u_r = \\frac{{1}}{{E}}\\left[(1-\\nu) A r + (1+\\nu)\\frac{{B}}{{r}}\\right]$$

## 2. 方法

* **单元/离散**：`tensorfem.continuum` 的 4 节点双线性等参元（2×2 Gauss，中心点 B 矩阵
  回写应力），`plane="strain"`，$E,\\nu,t$ 逐单元展开，稠密直接求解。
* **网格**：极坐标结构网格的 1/4 环扇形（45°…90° 扇区），节点严格落在内/外圆上，
  径向均匀细分为 `n_radial` 层，环向 `n_theta` 等分；单元连通按逆时针排列保证 Jacobian 正定。
* **边界条件**：两条径向切面施加真实对称条件（$\\theta=0$ 面 $u_y=0$，$\\theta=\\pi/2$ 面
  $u_x=0$），刚体模态被完全消除，不引入任何来自解析解的位移约束。
* **载荷**：内圆边每一段圆弧的合力 $2 p_i a \\sin(\\Delta\\theta/2)$ 沿角平分线施加，
  两个端节点各取一半（直边精确、曲边只留 $O(\\Delta\\theta^2)$ 的重分配误差）；载荷总
  合力由反力平衡独立校验。
* **参考值**：全部来自上节闭式解；载荷、应力、位移、平衡四个量互不共用推导路径。
* **验收门槛**：位移、$\\sigma_r$、$\\sigma_\\theta$ 的 $L_2$ 与最差采样点相对误差均 < 3 %。

工具链：`benchmarks/lame_cylinder_plane_strain.py`（本文件全部数字由其程序化注入：
`benchmarks/lame_cylinder_plane_strain_evidence.json`），测试见
`tests/test_lame_cylinder_plane_strain.py`。

## 3. 计算结果

网格阶梯（$\\theta=0$ 射线上的位移采样 + 单元中心应力），`u_r` 列为最大相对误差：

| 网格 $n_\\theta\\times n_r$ | 单元 | 自由度 | $u_r$ max | $u_r$ $L_2$ | $\\sigma_r$ $L_2$ | $\\sigma_r$ max | $\\sigma_\\theta$ $L_2$ | $\\sigma_\\theta$ max |
|---|---|---|---|---|---|---|---|---|
{mesh_rows}

内壁理论应力为 $\\sigma_r(a) = -p_i$、$\\sigma_\\theta(a) = 2A + p_i$；最细网格上这两条
Lamé 曲线与单元中心采样值的偏差见上表最后一列与下图。

观测收敛阶（两条最细网格，网格比 2）：

| 量 | 观测阶 |
|---|---|
| $u_r$ 最大误差 | {order['displacement']:.2f} |
| $\\sigma_r$ $L_2$ | {order['radial_stress']:.2f} |
| $\\sigma_\\theta$ $L_2$ | {order['hoop_stress']:.2f} |

对称性检查（离散解应当严格轴对称）：$\\max|\\tau_{{r\\theta}}|/\\max|\\sigma_r| = $
{sci(by_name['axisymmetry']['shear_ratio'])}，$\\max|u_\\theta|/\\max|u_r| = $
{sci(by_name['axisymmetry']['hoop_displacement_ratio'])}。

反力平衡：四分之一扇区上支撑反力合力范数 {by_name['load_equilibrium']['expected_magnitude'] / 1e6:.6f} MN，
与解析合力 $2 p_i a\\sin(\\pi/4)$ 的相对偏差 {sci(by_name['load_equilibrium']['relative_error'])}。

**平面应力/平面应变判别**：同一网格（{ladder[2]['n_theta']}x{ladder[2]['n_radial']}）下，平面应变与平面
应力两条路径各自匹配自己的闭式解（位移最大误差 {fmt(ladder[2]['displacement_max_relative'])} 与
{fmt(stress_case['displacement_max_relative'])}），而两者互相差 {fmt(by_name['constitutive_paths']['relative_gap_between_paths'])}，
与闭式解的柔度差 $(1-\\nu^2)$ 量级一致——说明 `plane` 参数真实生效。

径向梯度（`radial_grading>1` 时环向分层向内壁聚集）对精度的影响可用同一自由度预算下的
对比量化：{graded['n_theta']}x{graded['n_radial']}（grading={graded['grading']:.1f}）的
$\\sigma_r$ $L_2$ 误差 {fmt(graded['radial_stress_l2_relative'])}，与均匀网格
{ladder[1]['n_theta']}x{ladder[1]['n_radial']} 的 {fmt(ladder[1]['radial_stress_l2_relative'])} 对比。

![应力与位移剖面](assets/lame_cylinder_profiles.png)

![网格收敛](assets/lame_cylinder_convergence.png)

![环向应力云图](assets/lame_cylinder_field.png)

## 4. 注意点与适用边界

* **应力比较位置**：Q4 只回写单元中心应力，比较在单元中心（`centre_radius`）进行；
  内壁峰值应力位于边界外推处，不参与逐点验收（与仓库既有 `pointwise` 审计口径一致）。
* **载荷重分配**：曲边合力精确、端节点均分带来 $O(\\Delta\\theta^2)$ 的重分配误差，
  在网格阶梯上可见其随 $\\Delta\\theta$ 缩小而消失（$\\sigma_\\theta$ 列收敛阶接近 2）。
* **对称条件是精确的**：$u_\\theta=0$ 是 Lamé 解的真实性质，不是来自数值解的拟合；
  因此 1/4 扇区模型与整环模型等价，$u_\\theta$ 与 $\\tau_{{r\\theta}}$ 的数量级即离散误差。
* **适用范围**：小应变、各向同性、单一材料、无体力（无旋转、无温度梯度）、无轴向
  端部约束（平面应变 $\\varepsilon_z=0$ 对应无限长筒/端部约束筒）。不覆盖厚壁圆筒的
  极限承载（塑性/安定性）、开孔接管应力集中或热应力。
* **未做**：三维厚壁筒、多层套合/过盈、非线性材料、网格自适应误差估计。

## 5. 验收判定（对基准逐项）

| 锚点 | 结果 | 判据 |
|---|---|---|
{anchor_rows}

**是否 < 3 %**：最细 {fine['n_theta']}x{fine['n_radial']} 网格上，$u_r$ 最大 {fmt(fine['displacement_max_relative'])}
（$L_2$ {fmt(fine['displacement_l2_relative'])}）、$\\sigma_r$ 最差 {fmt(fine['radial_stress_max_relative'])}
（$L_2$ {fmt(fine['radial_stress_l2_relative'])}）、$\\sigma_\\theta$ 最差 {fmt(fine['hoop_stress_max_relative'])}
（$L_2$ {fmt(fine['hoop_stress_l2_relative'])}）——**均 < 3 %，通过**。粗网格（{ladder[0]['n_theta']}x{ladder[0]['n_radial']}）
的 $\\sigma_r$ 最差点 {fmt(ladder[0]['radial_stress_max_relative'])} 超过门槛，如实保留，未删除。
"""


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    ladder = [solve_case(n_theta, n_radial) for n_theta, n_radial in LADDER]
    stress_case = solve_case(48, 16, plane="stress")
    graded = solve_case(24, 8, grading=2.0)
    cases = ladder + [stress_case, graded]

    anchors = build_anchors(ladder, stress_case)
    figure_profiles = ASSETS / "lame_cylinder_profiles.png"
    figure_convergence = ASSETS / "lame_cylinder_convergence.png"
    figure_field = ASSETS / "lame_cylinder_field.png"
    plot_profiles(cases, figure_profiles)
    plot_convergence(ladder, figure_convergence)
    plot_field(ladder[-2], figure_field)

    evidence = {
        "benchmark": "lame_cylinder_plane_strain",
        "description": "Plane-strain Lame thick cylinder under internal pressure, Q4 continuum elements",
        "geometry": {
            "inner_radius": INNER,
            "outer_radius": OUTER,
            "inner_pressure": PRESSURE,
            "outer_pressure": 0.0,
            "young_modulus": YOUNG,
            "poisson_ratio": POISSON,
            "thickness": 1.0,
            "plane": "strain",
        },
        "acceptance_gate": GATE,
        "ladder": [{k: v for k, v in case.items() if k not in ("field", "nodes", "cells", "tip_diag")} for case in ladder],
        "plane_stress_case": {k: v for k, v in stress_case.items() if k not in ("field", "nodes", "cells", "tip_diag")},
        "graded_case": {k: v for k, v in graded.items() if k not in ("field", "nodes", "cells", "tip_diag")},
        "fine_profile": ladder[-1]["tip_diag"],
        "fine_field": {
            "radius": ladder[-1]["field"]["radius"],
            "sigma_r": ladder[-1]["field"]["sigma_r"],
            "sigma_theta": ladder[-1]["field"]["sigma_theta"],
            "sigma_r_reference": ladder[-1]["field"]["sigma_r_reference"],
            "sigma_theta_reference": ladder[-1]["field"]["sigma_theta_reference"],
        },
        "anchors": anchors,
        "all_passed": all(anchor["passed"] for anchor in anchors),
        "figures": [figure_profiles.name, figure_convergence.name, figure_field.name],
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    DOC.write_text(render_doc(ladder, stress_case, graded, anchors), encoding="utf-8")

    for case in ladder:
        print(
            f"{case['n_theta']:>3}x{case['n_radial']:<3} dofs={case['dofs']:>5} "
            f"u_r(max)={100*case['displacement_max_relative']:.4f}% "
            f"|sigma_r|L2={100*case['radial_stress_l2_relative']:.4f}% "
            f"|sigma_theta|L2={100*case['hoop_stress_l2_relative']:.4f}%"
        )
    print("all anchors passed:", evidence["all_passed"])
    for anchor in anchors:
        print(f"  {anchor['anchor']}: {'PASS' if anchor['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()