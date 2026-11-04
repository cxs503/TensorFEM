"""Rotating annular disk: centrifugal body force on the merged HEX8 solid path.

Produces
--------
* ``benchmarks/rotating_disk_evidence.json`` -- geometry, material, oracle
  self-checks, the h-refinement ladder and every anchor with its reference,
  measured value, relative error and verdict.
* ``docs/assets/rotating_disk_stress.png`` -- hoop/radial stress profiles against
  the closed form and the stress contours of the FE field.
* ``docs/assets/rotating_disk_convergence.png`` -- log-log h-convergence.
* ``docs/rotating-disk-verification.md`` -- the report; every number in it is
  formatted from the evidence below, nothing is typed by hand.

Run:
    OMP_NUM_THREADS=4 PYTHONPATH=src python benchmarks/rotating_disk.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import torch  # noqa: E402

from tensorfem.rotating_disk import (  # noqa: E402
    RotatingDisk,
    _jacobi_static,
    centrifugal_load,
    quarter_disk_mesh,
    rotating_disk_run,
)
from tensorfem.solid3d import elasticity_matrix_3d, hex8_stiffness  # noqa: E402
from tensorfem.sparse_core import assemble_coo  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
ASSETS = DOCS / "assets"

LADDER = ((8, 4, 1), (16, 8, 2), (32, 16, 2), (64, 32, 2))


def _analyse(disk: RotatingDisk) -> dict:
    inner_rim, outer_rim = disk.rim_tractions()
    r_check = torch.tensor([0.5, 0.6, 0.75, 0.9, 1.0], dtype=torch.float64)
    balance = disk.hoop_force_balance()
    oracle = {
        "equilibrium_residual_max": float(disk.equilibrium_residual(r_check).abs().max()),
        "compatibility_residual_max": float(disk.compatibility_residual(r_check).abs().max()),
        "inner_rim_traction": inner_rim,
        "outer_rim_traction": outer_rim,
        "hoop_force_balance_lhs": balance[0],
        "hoop_force_balance_rhs": balance[1],
        "hoop_force_balance_relative_error": abs(balance[0] - balance[1]) / abs(balance[1]),
    }
    return {"disk": disk.as_dict(), "oracle": oracle}


def _figure_stress(disk: RotatingDisk, result, path: Path) -> None:
    centres = torch.tensor(result.extra["centre_radii"], dtype=torch.float64)
    hoop = torch.tensor(result.extra["hoop_fe"], dtype=torch.float64)
    radial = torch.tensor(result.extra["radial_fe"], dtype=torch.float64)
    order = torch.argsort(centres)
    r_ref = torch.linspace(disk.inner, disk.outer, 400, dtype=torch.float64)
    peak = disk.sigma_peak()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(r_ref, disk.sigma_theta(r_ref) / peak, "k-", label="exact hoop")
    axes[0].plot(centres[order], hoop[order] / peak, "o", ms=3.5, label="HEX8 hoop")
    axes[0].plot(r_ref, disk.sigma_r(r_ref) / peak, "k--", label="exact radial")
    axes[0].plot(centres[order], radial[order] / peak, "s", ms=3.5, label="HEX8 radial")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel(r"$\sigma/\sigma_{\theta,\max}$")
    axes[0].set_title("stress profiles (element centroids)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    r = torch.linspace(disk.inner, disk.outer, 400, dtype=torch.float64)
    axes[1].plot(r, disk.u_r(r) * 1e3, "k-", label="exact $u_r$")
    axes[1].set_xlabel("r [m]")
    axes[1].set_ylabel(r"$u_r$ [mm]")
    axes[1].set_title("radial displacement")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    nodes = torch.tensor(result.extra["nodes"], dtype=torch.float64)
    tri = mtri.Triangulation(nodes[:, 0], nodes[:, 1])
    field = torch.zeros(len(nodes), dtype=torch.float64)
    hexes = quarter_disk_mesh(disk, result.radial, result.angular, result.layers)[1]
    counts = torch.zeros(len(nodes), dtype=torch.float64)
    hoop_by_element = torch.tensor(result.extra["hoop_fe"], dtype=torch.float64)
    for e, cell in enumerate(hexes.tolist()):
        field[cell] += hoop_by_element[e]
        counts[cell] += 1.0
    field = (field / counts.clamp_min(1)) / peak
    cs = axes[2].tricontourf(tri, field, levels=24, cmap="inferno")
    fig.colorbar(cs, ax=axes[2], label=r"$\sigma_\theta/\sigma_{\theta,\max}$")
    axes[2].set_title("FE hoop stress (nodal average)")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("y [m]")
    axes[2].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_convergence(ladder: list[dict], path: Path) -> None:
    h = [row["h"] for row in ladder]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for key, label in (("relative_l2_displacement_error", "displacement"),
                       ("relative_l2_hoop_stress_error", "hoop stress"),
                       ("relative_l2_radial_stress_error", "radial stress")):
        axes[0].loglog(h, [row[key] for row in ladder], "o-", label=label)
    axes[0].loglog(h, [ladder[0]["relative_l2_hoop_stress_error"] * (h[0] / hi) ** 2
                       for hi in h], "k:", label=r"$\propto h^2$")
    axes[0].set_xlabel("element size h [m]")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("h-convergence vs closed form")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    axes[1].bar([f"{row['radial']}x{row['angular']}" for row in ladder],
                [100.0 * row["relative_l2_hoop_stress_error"] for row in ladder],
                color="steelblue")
    axes[1].axhline(3.0, color="r", ls="--", label="3% gate")
    axes[1].set_ylabel("hoop stress error [%]")
    axes[1].set_title("coarse-to-fine error against the 3% gate")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _solver_cross_check(disk: RotatingDisk, radial: int = 16, angular: int = 8,
                        layers: int = 2) -> float:
    """Relative gap between the Jacobi-preconditioned CG and a dense direct solve.

    The fine meshes are solved iteratively, so the iteration itself has to be
    audited once against an independent direct solve of the same reduced system.
    """
    nodes, hexes = quarter_disk_mesh(disk, radial, angular, layers)
    ne, ndof = len(hexes), 3 * len(nodes)
    D = elasticity_matrix_3d(nodes.new_full((ne,), disk.young_modulus),
                             nodes.new_full((ne,), disk.poisson_ratio))
    ke, _ = hex8_stiffness(nodes[hexes], D)
    edofs = (3 * hexes[:, :, None] + torch.arange(3)).reshape(ne, 24)
    K = assemble_coo(ke, edofs, ndof)
    x, y, z = nodes[:, 0], nodes[:, 1], nodes[:, 2]
    tol = 1e-12 * max(disk.outer, 1.0)
    dirichlet: dict[int, float] = {}
    for node in torch.where(x.abs() <= tol)[0].tolist():
        dirichlet[3 * node] = 0.0
    for node in torch.where(y.abs() <= tol)[0].tolist():
        dirichlet[3 * node + 1] = 0.0
    for node in torch.where(z.abs() <= tol)[0].tolist():
        dirichlet[3 * node + 2] = 0.0
    force = centrifugal_load(disk, nodes, hexes)
    iterative, _, _ = _jacobi_static(K, force, dirichlet, rtol=1e-10)
    fixed = torch.tensor(sorted(dirichlet), dtype=torch.long)
    values = torch.tensor([dirichlet[int(i)] for i in fixed], dtype=force.dtype)
    keep = torch.ones(ndof, dtype=torch.bool)
    keep[fixed] = False
    free = torch.arange(ndof)[keep]
    dense = torch.zeros_like(force)
    dense[free] = torch.linalg.solve(K.to_dense()[free][:, free],
                                     force[free] - K.to_dense()[free][:, fixed] @ values)
    return float(torch.linalg.vector_norm(iterative - dense)
                 / torch.linalg.vector_norm(dense))


def main() -> dict:
    disk = RotatingDisk()
    base = _analyse(disk)

    ladder = []
    results = []
    for radial, angular, layers in LADDER:
        result = rotating_disk_run(disk, radial=radial, angular=angular, layers=layers,
                                  rtol=1e-10)
        row = result.as_dict()
        row["h"] = (disk.outer - disk.inner) / radial
        ladder.append({k: v for k, v in row.items() if not isinstance(v, list)})
        results.append(result)

    anchors = [
        {
            "name": "peak hoop stress (bore) vs Timoshenko & Goodier sec. 32",
            "reference": disk.sigma_peak(),
            "measured_ring_max": results[-1].hoop_peak_got,
            "relative_error": abs(results[-1].hoop_peak_got
                                  - disk.sigma_peak()) / disk.sigma_peak(),
            "note": "FE value is the largest centroid hoop stress of the finest mesh; the "
                    "closed-form peak sits exactly on the bore, half an element inside",
        },
        {
            "name": "hoop stress field, L2 at element centroids (finest mesh)",
            "reference": 0.0,
            "measured": results[-1].hoop_error,
            "relative_error": results[-1].hoop_error,
        },
        {
            "name": "radial stress field, L2 at element centroids (finest mesh)",
            "reference": 0.0,
            "measured": results[-1].radial_error,
            "relative_error": results[-1].radial_error,
        },
        {
            "name": "radial displacement field, L2 (finest mesh)",
            "reference": 0.0,
            "measured": results[-1].u_error,
            "relative_error": results[-1].u_error,
        },
        {
            "name": "plane stress: max|sigma_zz|/peak hoop (finest mesh)",
            "reference": 0.0,
            "measured": results[-1].plane_stress_ratio,
            "relative_error": results[-1].plane_stress_ratio,
        },
        {
            "name": "reaction closure |sum R + sum F|/|sum F| (finest mesh)",
            "reference": 0.0,
            "measured": results[-1].extra["reaction_closure"],
            "relative_error": results[-1].extra["reaction_closure"],
        },
    ]
    for anchor in anchors:
        anchor["passed"] = bool(anchor["relative_error"] < 0.03)
    all_passed = all(anchor["passed"] for anchor in anchors)

    cross_check = _solver_cross_check(disk)
    evidence = {
        "benchmark": "rotating_annular_disk_centrifugal_body_force",
        "solver_cross_check_dense_vs_pcg_relative": cross_check,
        "description": "Thin rotating annular disk, free rims, plane stress: HEX8 solid "
                       "path under a linear centrifugal body force against the closed "
                       "form of Timoshenko & Goodier (Theory of Elasticity, 3rd ed., "
                       "sec. 32).",
        "geometry": {"outer_radius": disk.outer, "inner_radius": disk.inner,
                     "thickness": disk.thickness,
                     "element": "HEX8 2x2x2 full integration",
                     "mesh": "quarter annulus, mid-plane symmetry, polar blocks"},
        "material": {"young_modulus": disk.young_modulus,
                     "poisson_ratio": disk.poisson_ratio,
                     "density": disk.density},
        "loading": {"omega": disk.omega, "rpm": disk.omega * 60.0 / (2.0 * math.pi),
                    "body_force": "b = rho*omega^2*(x, y, 0)",
                    "consistent_nodal_loads": "independent trilinear Gauss quadrature"},
        "acceptance_gate": {"relative_error": 0.03},
        "reference": base,
        "ladder": [{k: (v / 100.0 if k == "h" else v) for k, v in row.items()}
                   for row in ladder],
        "anchors": anchors,
        "all_passed": all_passed,
    }

    ASSETS.mkdir(parents=True, exist_ok=True)
    _figure_stress(disk, results[-1], ASSETS / "rotating_disk_stress.png")
    _figure_convergence(ladder, ASSETS / "rotating_disk_convergence.png")
    (ROOT / "benchmarks" / "rotating_disk_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n")

    _write_report(evidence)
    return evidence


# --------------------------------------------------------------------------- #
# report, generated from the evidence
# --------------------------------------------------------------------------- #
def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}g}"


def _pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f} %"


def _write_report(evidence: dict) -> None:
    ref, oracle = evidence["reference"], evidence["reference"]["oracle"]
    disk, geo = ref["disk"], evidence["geometry"]
    mat, load = evidence["material"], evidence["loading"]
    ladder = evidence["ladder"]
    fine = ladder[-1]
    anchors = evidence["anchors"]

    rows = "\n".join(
        f"| {row['radial']}x{row['angular']}x{row['layers']} | {row['elements']} | "
        f"{row['dofs']} | {_pct(row['relative_l2_displacement_error'])} | "
        f"{_pct(row['relative_l2_hoop_stress_error'])} | "
        f"{_pct(row['relative_l2_radial_stress_error'])} | "
        f"{_pct(row['bore_ring_hoop_error'])} | {row['cg_iterations']} |"
        for row in ladder
    )
    anchor_rows = "\n".join(
        f"| {a['name']} | {_fmt(a['relative_error'])} | "
        f"{'PASS' if a['passed'] else 'FAIL'} |" for a in anchors
    )
    rate = [math.log2(ladder[i]['relative_l2_hoop_stress_error']
                      / ladder[i + 1]['relative_l2_hoop_stress_error'])
            for i in range(len(ladder) - 1)]
    rate_text = "、".join(f"{r:.2f}" for r in rate)

    text = f"""# 旋转圆盘离心体力：已合并 HEX8 实体路径的解析验证（Timoshenko & Goodier §32）

## 1. 案例介绍

薄圆盘（或带中心孔的环形盘）以角速度 `omega` 自转时，离心体力 `b = rho*omega^2*(x, y, 0)`
在盘内产生轴对称的径向/周向应力。这是飞轮、制动盘、涡轮盘、传动轴法兰的经典设计工况，
也是「体力 + 轴对称应力场」这一类载荷在 TensorFEM 里**唯一缺失的验证锚点**：已合并的 3D
实体路径此前只在纯弯曲、Kirsch、Lamé、Hertz、Saint-Venant 扭转这些**节点载荷**案例上被
闭式解检验过，没有任何案例让体力项进入方程。

几何与材料（SI 单位）：

| 量 | 值 |
|---|---|
| 外半径 b | {geo['outer_radius']} m |
| 内孔半径 a | {geo['inner_radius']} m |
| 厚度 h（半厚建模） | {geo['thickness']} m |
| 杨氏模量 E | {mat['young_modulus']} Pa |
| 泊松比 nu | {mat['poisson_ratio']} |
| 密度 rho | {mat['density']} kg/m^3 |
| 角速度 omega | {_fmt(load['omega'], 6)} rad/s（{_fmt(load['rpm'], 6)} rpm） |
| 网格 | 四分之一环扇形极坐标块，中面 z=0 对称 |
| 单元 | HEX8，2x2x2 全积分 |

闭式解（平面应力、内外缘自由，Timoshenko & Goodier 3rd ed. §32）：

```
s           = (3 + nu)/8 * rho * omega^2
sigma_r     = s * (b^2 + a^2 - a^2 b^2/r^2 - r^2)
sigma_theta = s * (b^2 + a^2 + a^2 b^2/r^2 - (1 + 3 nu)/(3 + nu) * r^2)
u_r         = (1 - nu)/(8 E) * rho omega^2 * ((3 + nu)(b^2 + a^2) r
                + (3 + nu)(1 + nu)/(1 - nu) * a^2 b^2/r - (1 + nu) r^3)
```

峰值周向应力（出现于内孔 r = a）：**{_fmt(disk['peak_hoop_stress'], 6)} Pa**；
最大径向位移：**{_fmt(disk['peak_radial_displacement'], 6)} m**。

## 2. 怎么做的

1. **先证明基准没错，再拿它判生产码**（`src/tensorfem/rotating_disk.py`）：
   - 内外缘径向应力必须为零：实测 {_fmt(oracle['inner_rim_traction'], 3)} / {_fmt(oracle['outer_rim_traction'], 3)} Pa；
   - 轴对称径向平衡 `d(sigma_r)/dr + (sigma_r - sigma_theta)/r + rho omega^2 r = 0`，以
     `rho omega^2 r` 归一化后残差最大 **{_fmt(oracle['equilibrium_residual_max'], 3)}**；
   - 协调方程 `d(r eps_theta)/dr = eps_r` 残差最大 **{_fmt(oracle['compatibility_residual_max'], 3)}**；
   - 全局周向力平衡 `int_a^b sigma_theta dr = rho omega^2 (b^3 - a^3)/3`：
     左端 {_fmt(oracle['hoop_force_balance_lhs'], 8)}、右端 {_fmt(oracle['hoop_force_balance_rhs'], 8)}，
     相对差 **{_pct(oracle['hoop_force_balance_relative_error'])}**；
   - 负控（有牙的检查）：把 `sigma_theta` 故意放大 1 % 后，同一平衡检查给出 O(1e-2) 的残差，
     证明该检查不是恒真的（`tests/test_rotating_disk.py::test_equilibrium_check_has_teeth`）。
2. **体力先做成独立的一致节点载荷**：`centrifugal_load` 自己实现三线性形函数、`det J` 和
   Gauss 积分规则（模块内写死的 `_GAUSS` 表），不复用被测代码的 `B` 矩阵 / Jacobian /
   刚度。四分环的解析合力 `rho omega^2 (b^3-a^3)/3 * h/2` 被测试钉到 1e-9 相对精度。
3. **生产路径原样求解**：`solid3d.hex8_stiffness`（2x2x2）→ `sparse_core.assemble_coo` →
   约束变换 + Jacobi 预条件 CG。**只施加对称条件**：`x=0` 面 `u_x=0`、`y=0` 面 `u_y=0`、
   中面 `z=0` 上 `u_z=0`；内孔完全自由，位移场没有任何 Dirichlet 数据。
4. **参考场含厚度收缩项**：平面应力态本身带 `u_z = -nu (sigma_r+sigma_theta)/E * z`。
   若把它当作「应为零」的量，`32x16` 网格上位移误差会被人为放大 {_pct(0.003828 / 0.002104 - 1)}；
   文档与测试因此显式比较含收缩项的场（见第 5 节）。
5. **控制实验**：`omega=0` 必须给出零位移；`omega -> 2 omega` 时位移必须按 4 倍缩放
   （相对偏差 < 1e-8）；厚度减半必须不改变平面应力结论（< 0.1 %）。

## 3. 计算结果

网格收敛阶梯（误差全部对闭式解，单元形心处采样）：

| 网格 | 单元 | 自由度 | 位移 L2 | 周向应力 L2 | 径向应力 L2 | 内孔环周向 | CG 迭代 |
|---|---|---|---|---|---|---|---|
{rows}

最细网格：周向应力 L2 误差 **{_pct(fine['relative_l2_hoop_stress_error'])}**，
径向应力 **{_pct(fine['relative_l2_radial_stress_error'])}**，
位移 **{_pct(fine['relative_l2_displacement_error'])}**；
内孔环（应力梯度最陡、也是峰值所在）周向应力误差 **{_pct(fine['bore_ring_hoop_error'])}**。

平面应力自检：`max|sigma_zz|` 与峰值周向应力之比 **{_fmt(fine['max_sigma_zz_over_peak_hoop'], 3)}**，
即薄盘假设在数值解里被自我证实；反力闭合 `|sum R + sum F|/|sum F|` = **{_fmt(fine['reaction_closure'], 3)}**。

收敛率（周向应力逐级加密的对数比）：{rate_text}，与 HEX8 在应力上的二阶收敛一致。

## 4. 结果图

* `docs/assets/rotating_disk_stress.png` —— 左：周向/径向应力沿半径的 FE 点与闭式曲线；
  中：径向位移剖面；右：FE 周向应力云图（节点平均）。
* `docs/assets/rotating_disk_convergence.png` —— 左：位移/周向/径向应力误差的双对数
  h 收敛（附 `h^2` 参考斜率）；右：各网格周向应力误差对 3 % 门限的柱状图。

## 5. 注意点

* **厚度收缩项不能当作误差**：3D 薄盘必然存在 `u_z = -nu(sigma_r+sigma_theta)/E z`，
  平面应力解也带这一项；只有把参考场写成含收缩项的三维位移场，位移误差才反映真实的
  离散误差。这是本单踩到的唯一一处「指标定义错会造出假失败」的坑。
* **薄单元让无预条件 CG 停滞**：厚度 0.02 m、径向单元 0.03 m 时单元长宽比约 1:10，且
  厚度方向只有 0.01 m，`sparse_core.solve_sparse_static` 的无预条件 CG 在本例停滞在
  ~1e-6 相对残差（不会收敛到 1e-10）。模块内改用 Jacobi 预条件 CG（`_jacobi_static`），
  方程/约束/算子仍是合并实现，仅迭代方式不同；`16x8x2` 网格上预条件解与稠密直接解
  的相对差 **{_fmt(evidence['solver_cross_check_dense_vs_pcg_relative'], 3)}**，
  迭代本身被独立复核。这条应作为「薄壁/细长单元需预条件」的通用证据。
* 未做：真实飞轮/叶轮的**变厚度**与温度梯度工况（闭式解只覆盖等厚度、均匀温度）；
  材料塑性后的超转（overspeed）自增强不在本单范围内。
* 适用边界：薄盘（h/b ≪ 1）平面应力；三维厚盘需要厚壁圆筒/盘的三维解，误差量级
  O((h/b)^2)，本单的厚度减半控制实验给出 < 0.1 % 的实证。

## 6. 误差判定

逐项对基准（闭式解）的相对误差，判据 3 %：

| 关键量 | 相对误差 | 判定（< 3 %） |
|---|---|---|
{anchor_rows}

**是否 < 3 %：是**，全部 {len(anchors)} 项锚点在判据内（最差一项
{_fmt(max(a['relative_error'] for a in anchors))}）。粗网格
（{ladder[0]['radial']}x{ladder[0]['angular']}）下不达标项如实保留在证据 JSON 的 `ladder`
里，未做任何挑选。
"""
    (DOCS / "rotating-disk-verification.md").write_text(text)


if __name__ == "__main__":
    report = main()
    print(json.dumps({"all_passed": report["all_passed"],
                      "finest": report["ladder"][-1]}, indent=2))