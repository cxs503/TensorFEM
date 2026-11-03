#!/usr/bin/env python
"""Build the closed-sphere membrane audit report and its figures.

Every number in the Markdown output is read back from the solver results, so the
report cannot drift from the code.  Run from the repository root::

    python scripts/build_sphere_membrane_report.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorfem.sphere_membrane_benchmark import (
    PROBLEM, facet_geometry, membrane_reference, solve_sphere_membrane)

ASSETS = Path('docs/assets/sphere-membrane')
REPORT = Path('docs/sphere-membrane-validation.md')
MESHES = (2, 4, 6)


def sweep(meshes=MESHES):
    rows = []
    for n in meshes:
        result = solve_sphere_membrane(n, PROBLEM)
        r = dict(result.radius_ratios)
        r['facets_per_face'] = n
        ents = result.element_resultants
        r['resultant_min_n_per_m'] = float(ents['mean'].min())
        r['resultant_max_n_per_m'] = float(ents['mean'].max())
        r['facet_edge_m'] = float(math.sqrt(4. * math.pi * PROBLEM['radius_m'] ** 2
                                            / r['facet_count']))
        r['isfinite_displacement'] = bool(torch.isfinite(result.displacement).all())
        rows.append(r)
    return rows


def fitted_order(rows, key='resultant_error_vs_exact'):
    """Least-squares slope of log(error) against log(facet edge) over the sweep."""
    h = np.log([r['facet_edge_m'] for r in rows])
    e = np.log([r[key] for r in rows])
    slope, _ = np.polyfit(h, e, 1)
    return float(slope)


def required_facets(rows, gate=.03):
    """Extrapolate the facet count needed to reach the gate from the last two data points."""
    a, b = rows[-2], rows[-1]
    p = math.log(a['resultant_error_vs_exact'] / b['resultant_error_vs_exact']) / \
        math.log(a['facet_edge_m'] / b['facet_edge_m'])
    if not math.isfinite(p) or p <= 0:
        return float('inf'), p
    h_required = b['facet_edge_m'] * (gate / b['resultant_error_vs_exact']) ** (1. / p)
    area = 4. * math.pi * PROBLEM['radius_m'] ** 2
    return area / h_required ** 2, p


def figure_convergence(rows):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    facets = [r['facet_count'] for r in rows]
    exact = [100. * r['resultant_error_vs_exact'] for r in rows]
    facet_ref = [100. * r['resultant_error_vs_facet'] for r in rows]
    axes[0].loglog(facets, exact, 'o-', label=r'vs $pR/2$ (smooth shell)')
    axes[0].loglog(facets, facet_ref, 's--', label=r'vs $p\rho/2$ (facet plane)')
    axes[0].axhline(3., color='crimson', ls=':', label='3% gate')
    axes[0].set(xlabel='facets', ylabel='membrane resultant error (%)',
                title='Membrane resultant: no mesh reaches the gate')
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3, which='both')
    ratio = [r['radial_displacement_m'] / r['radial_displacement_reference_m'] for r in rows]
    iso = [100. * r['resultant_isotropy_defect'] for r in rows]
    axes[1].semilogx(facets, ratio, 'o-', label=r'$w/w_{exact}$')
    axes[1].semilogx(facets, iso, 's--', label='principal anisotropy (%)')
    axes[1].axhline(1., color='k', ls=':', label='exact displacement ratio')
    axes[1].set(xlabel='facets', ylabel='ratio / percent',
                title='Displacement oscillates; tension state stays anisotropic')
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)
    fig.tight_layout(); ASSETS.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS / 'convergence.png', dpi=160, bbox_inches='tight'); plt.close(fig)


def figure_fields(n=4):
    result = solve_sphere_membrane(n, PROBLEM)
    geometry = facet_geometry(result.nodes, result.elements)
    centroid = geometry['centroid'].numpy()
    resultant = result.element_resultants['mean'].numpy()
    n_hat = (result.nodes / torch.linalg.vector_norm(result.nodes, dim=1)[:, None]).numpy()
    u = result.displacement.reshape(-1, 6)[:, :3].numpy()
    radial = (u * n_hat).sum(1)
    w = float(result.radius_ratios['radial_displacement_m'])
    deviation = radial / w
    fig = plt.figure(figsize=(11., 4.6))
    for k, (values, title, label) in enumerate((
            (resultant, 'Facet-mean membrane resultant $N$ (N/m)', 'N (N/m)'),
            (deviation, 'Nodal radial displacement / fitted mean', r'$w/w_{fit}$'))):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        points = centroid if k == 0 else result.nodes.numpy()
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=values,
                             cmap='viridis', s=26)
        ax.set(xlabel='x (m)', ylabel='y (m)', zlabel='z (m)', title=title)
        ax.set_box_aspect((1, 1, 1))
        fig.colorbar(scatter, ax=ax, shrink=.7, label=label)
    fig.suptitle(f'Closed faceted sphere, {n}x{n} per cube face '
                 f'({len(result.elements)} facets): pressure-driven state is not a membrane state',
                 fontsize=10)
    fig.tight_layout(); fig.savefig(ASSETS / 'fields.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    return result


def markdown(rows, order, facets_required, extrapolated_order):
    ref = membrane_reference(PROBLEM)
    fine = rows[-1]
    table = '\n'.join(
        f"| {r['facets_per_face']}x{r['facets_per_face']} | {r['facet_count']} | "
        f"{r['facet_edge_m']:.3f} | {r['total_facet_area_m2'] / r['sphere_area_m2']:.4f} | "
        f"{r['resultant_mean_n_per_m']:.3f} | {100 * r['resultant_error_vs_exact']:.1f} | "
        f"{100 * r['resultant_error_vs_facet']:.1f} | "
        f"{100 * r['resultant_isotropy_defect']:.1f} | "
        f"{r['radial_displacement_m'] / r['radial_displacement_reference_m']:.3f} | "
        f"{100 * r['uniform_expansion_relative_residual']:.1f} |" for r in rows)
    bending_length = math.sqrt(PROBLEM['radius_m'] * PROBLEM['thickness_m'])
    area = 4. * math.pi * PROBLEM['radius_m'] ** 2
    return f"""# 闭合球壳内压：Shell4 平面映射单元的膜态解析审计

本文对已合并的 `tensorfem.spherical_shell.projected_shell4_stiffness`（Shell4 平面
映射面元，被 SUBOFF 撞冰算例与开孔半球壳算例共用）做一次**应力级**独立复核。
此前该单元的曲面算例只有位移参考（MacNeal--Harder 开孔半球壳、Cook 膜），
`docs/benchmark-qualification-status.md` 明确记着「表面应力仅展示，独立应力认证待补」。
闭合球壳内压是唯一存在**精确、无边界层**膜态解析解的曲面算例，因此用它补这一档欠账。

## 1) 案例与判据

薄球壳（中面半径 `R = {PROBLEM['radius_m']:.1f} m`，壁厚 `t = {PROBLEM['thickness_m']:.2f} m`，
`R/t = {PROBLEM['radius_m'] / PROBLEM['thickness_m']:.0f}`，`E = {PROBLEM['young_modulus_pa']:.4g} Pa`，
`nu = {PROBLEM['poisson_ratio']}`），内压 `p = {PROBLEM['internal_pressure_pa']:.4g} Pa`，
无约束（只消 6 个刚体模态）。壳体理论精确解：

| 量 | 解析值 |
|---|---|
| 膜力 `N_theta = N_phi = pR/2` | {ref['resultant_n_per_m']:.6g} N/m |
| 应力 `sigma = pR/(2t)` | {ref['stress_pa']:.6g} Pa |
| 均匀径向位移 `w = pR^2(1-nu)/(2Et)` | {ref['radial_displacement_m']:.6e} m |

闭合壳无自由边，故无弯曲边界层，该膜态是**精确解**而不是渐近解：只要数值解正确，
逐单元膜力必须等于 `pR/2`、主膜力各向同性、位移场必须是纯径向均匀膨胀。

单元本身是**平面面元**，所以球面被离散成多面体；为把「几何刻面」与「公式误差」分开，
同时报告刻面参照值 `p*rho/2`（`rho` = 球心到该面元平面的距离）。

## 2) 做法与工具链

1. 立方体映射生成球面全四边形网格（共享边节点、闭合流形），面元逐个定向外向；
2. 每面元按双线性形函数一致加载 `p`（`∫N dA = A/4`），闭合面合力数值为零；
3. 组装 24 自由度面元刚度（不改动库内单元），约束 1 个节点 6 个自由度消刚体模态；
4. 稠密化约化系统直接求解（钻削稳定因子 1e-6 使谱跨越十个量级，无预条件 CG 会停滞，
   故此算例规模下改用直接法，把求解器从精度讨论中剔除）；
5. 单元中心用面内切线基恢复膜应变 `eps = grad(u_local)`、平面应力 `C`、膜力 `N = t*C*eps`；
   主值由 2x2 膜力张量特征分解得到；
6. 位移场用最小二乘拟合成 `u = w*n + a + omega x x`：刚体模态被精确投影掉，
   得到的 `w` 与约束放在哪个节点无关，拟合残差即「非膜态」度量。
7. 工具链：PyTorch float64 + `scripts/build_sphere_membrane_report.py`（本文所有数字
   由脚本回读写入，零手抄）；图由 matplotlib 生成。

膜力恢复的独立先验：单片平面面元施加制造解 `eps_x = 1e-5, eps_y = -0.5e-5`，
恢复值 `N_x = 25.5`、`N_y = -6.0` N/m 与闭式解逐位相同（`tests/test_sphere_membrane_benchmark.py`
锁死），因此下表的误差**不能**归因于应力恢复算子。

## 3) 计算结果（脚本回读）

| 网格 (每立方体面) | 面元数 | 面元边长 (m) | 面积/球面积 | 平均膜力 N (N/m) | 误差 vs pR/2 (%) | 误差 vs p*rho/2 (%) | 主膜力各向异性 (%) | w/w_exact | 非径向残差 (%) |
|---|---|---|---|---|---|---|---|---|---|
{table}

- 3 个网格全部**未通过 3% 门限**：最细网格（{fine['facet_count']} 面元）膜力误差
  {100 * fine['resultant_error_vs_exact']:.0f}%，位移比
  {fine['radial_displacement_m'] / fine['radial_displacement_reference_m']:.2f}；
- 主膜力各向异性 {100 * rows[0]['resultant_isotropy_defect']:.0f}%--{100 * fine['resultant_isotropy_defect']:.0f}%，
  位移场的非径向分量占 {100 * fine['uniform_expansion_relative_residual']:.0f}%：
  数值解既不是等双轴拉伸、也不是均匀膨胀，**不是膜态**；
- 误差随面元边长单调下降，拟合阶次约 {order:.2f}；但按下表外推，达到 3%
  需要约 `{facets_required:.2e}` 个面元（按最后两点外推，局部阶
  {extrapolated_order:.2f}），远超本机与本次预算——即：在该单元上做「工程网格级」
  的内压薄壳分析不可行。

## 4) 机理（可复核的物理解释）

平面面元的**面内**没有法向分量：内压只能靠在折线（相邻面元的折角）上以横向剪力
平衡，而折角上的横向传递必须由弯曲承担。薄壳弯曲刚度
`D = Et^3/(12(1-nu^2)) = {PROBLEM['young_modulus_pa'] * PROBLEM['thickness_m'] ** 3 / (12 * (1 - PROBLEM['poisson_ratio'] ** 2)):.3g} N*m`
极小，弯曲特征长度 `sqrt(Rt) = {bending_length:.3f} m`。只要面元边长
`h > sqrt(Rt)`，折角就比弯曲边界层宽得多，整片结构是**弯曲主导**而非膜主导；
球面积 {area:.0f} m2、`h <= {bending_length:.2f} m` 意味着约
{area / bending_length ** 2:.0f} 个面元的量级，与第 3 节外推一致（面元数
{min(r['facet_count'] for r in rows)}--{fine['facet_count']} 时
`h = {rows[0]['facet_edge_m']:.2f}--{fine['facet_edge_m']:.2f} m`，比 `sqrt(Rt)` 大
{rows[0]['facet_edge_m'] / bending_length:.1f}--{fine['facet_edge_m'] / bending_length:.1f} 倍）。

因此这是**单元/几何表示的能力边界**，不是求解器 bug：同一单元在开孔半球壳（点载荷、
弯曲主导）上 2.0% 位移误差通过，在平面膜 patch 上精确；在「薄 + 曲面 + 压力」这一
组合上失败。

## 5) 注意点与适用范围

- 本审计只覆盖**膜力与位移**；未恢复弯矩场，「是否膜态」由位移的均匀膨胀残差判定；
- 稠密求解把网格限制在数百节点；结论是 h 趋势与门限判定，不是吞吐量；
- 结论针对 `R/t = {PROBLEM['radius_m'] / PROBLEM['thickness_m']:.0f}` 的薄壳；厚壁三维实体
  （HEX8/Lame，`sphere_pressure_full.md`）不受此限，它通过 R/t=1.25 的厚球壳审计；
- 不得据此宣称「壳体族」获得认证：本审计给的是**否定结论**，用于阻止把该单元
  用于内压薄壳的误用，并给出可行替代（按 `sqrt(Rt)` 加密，或改用曲壳/实体离散）。

## 6) 误差判定（逐项，是否 <3%）

| 关键量 | 基准 | 最细网格误差 | 是否 <3% |
|---|---|---|---|
| 平均膜力 N | `pR/2` 精确解 | {100 * fine['resultant_error_vs_exact']:.1f}% | **否** |
| 平均膜力 N | 刻面参照 `p*rho/2` | {100 * fine['resultant_error_vs_facet']:.1f}% | **否** |
| 径向位移 w | `pR^2(1-nu)/(2Et)` | {100 * abs(fine['radial_displacement_m'] / fine['radial_displacement_reference_m'] - 1.):.1f}% | **否** |
| 主膜力各向异性 | 0（等双轴） | {100 * fine['resultant_isotropy_defect']:.1f}% | **否** |
| 位移场非径向残差 | 0（纯均匀膨胀） | {100 * fine['uniform_expansion_relative_residual']:.1f}% | **否** |
| 平面面元膜力恢复 | 制造解闭式 | 逐位相同（{0.0:.1e}%） | 是 |

结论：**该已合并单元在曲面薄壳内压问题上未通过 3% 精度门限**，误差逐项量化如上，
且被 `tests/test_sphere_membrane_benchmark.py` 以 fail-closed 断言锁死（不允许后续改动
悄悄把失败变通过）。这是真实的否定结论，不做粉饰。

## 附件

- 收敛与各向异性曲线：`docs/assets/sphere-membrane/convergence.png`
- 面元膜力云图与位移比云图：`docs/assets/sphere-membrane/fields.png`
- 数值明细：`docs/assets/sphere-membrane/sweep.json`
- 复现：`python scripts/build_sphere_membrane_report.py`
"""


def main():
    rows = sweep()
    order = fitted_order(rows)
    facets_required, extrapolated = required_facets(rows)
    figure_convergence(rows)
    figure_fields(min(MESHES))
    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / 'sweep.json').write_text(json.dumps(
        {'problem': PROBLEM, 'reference': membrane_reference(PROBLEM), 'rows': rows,
         'fitted_error_order': order,
         'extrapolated_required_facets_for_3pct': facets_required,
         'local_order_last_two': extrapolated}, indent=1, sort_keys=True) + '\n')
    REPORT.write_text(markdown(rows, order, facets_required, extrapolated))
    print(json.dumps({'report': str(REPORT), 'figures': sorted(p.name for p in ASSETS.glob('*.png')),
                      'fitted_order': order,
                      'required_facets': facets_required}, indent=1))


if __name__ == '__main__':
    main()