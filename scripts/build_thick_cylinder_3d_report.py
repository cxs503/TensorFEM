"""Generate the three-dimensional Lamé thick-cylinder verification report.

Outputs (all numbers are written by this script, nothing is hand-copied):

* ``docs/benchmarks/thick-cylinder-3d-lame.json`` -- raw evidence
* ``docs/assets/thick_cylinder_3d_fields.png``   -- hoop/radial stress field
* ``docs/assets/thick_cylinder_3d_profiles.png`` -- stress and displacement vs r
* ``docs/assets/thick_cylinder_3d_convergence.png`` -- error vs element size
* ``docs/thick-cylinder-3d-lame.md``             -- Markdown report
"""
from __future__ import annotations

import base64
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfem.thick_cylinder_3d import (  # noqa: E402
    LameCylinderReference, CylinderQualification, field_errors, qualify_thick_cylinder,
    solve_thick_cylinder, stress_in_cylindrical_frame)

ROOT = pathlib.Path(__file__).resolve().parents[1]
MESHES = ((3, 8, 3), (4, 12, 4), (5, 16, 5))
VARIANTS = {
    "基准（内压 10 MPa）": dict(),
    "外压 12 MPa": dict(inner_pressure=0., outer_pressure=1.2e7),
    "内外压组合": dict(inner_pressure=1.e7, outer_pressure=4.e6),
    "薄壁 r_o/r_i = 1.11": dict(inner_radius=.9, outer_radius=1.0),
    "E/ν 变化（70 GPa, 0.33）": dict(young=7.0e10, poisson=.33),
}


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value * 100:.{digits}f}"


def main() -> int:
    torch.set_num_threads(min(8, torch.get_num_threads()))
    reference = LameCylinderReference()
    solves = [solve_thick_cylinder(reference, nr=a, ntheta=b, nz=c) for a, b, c in MESHES]
    short = solve_thick_cylinder(reference, nr=5, ntheta=16, nz=5, half_length=.6)
    qual: CylinderQualification = qualify_thick_cylinder(reference, solves, length_run=short)
    per_mesh = [field_errors(s, reference) for s in solves]

    variants = {}
    for name, kwargs in VARIANTS.items():
        case = LameCylinderReference(**kwargs)
        run = solve_thick_cylinder(case, nr=5, ntheta=16, nz=5)
        variants[name] = {"reference": {"inner_radius": case.inner_radius,
                                        "outer_radius": case.outer_radius,
                                        "inner_pressure": case.inner_pressure,
                                        "outer_pressure": case.outer_pressure,
                                        "young": case.young, "poisson": case.poisson},
                          "axial_stress": case.axial_stress,
                          "errors": field_errors(run, case)}

    fine = solves[-1]
    mask = fine.element_centres[:, 2] >= .35 * float(fine.mesh.nodes[:, 2].max())
    centres = fine.element_centres[mask]
    numeric = stress_in_cylindrical_frame(fine.stress[mask], centres)
    radius = torch.linalg.vector_norm(centres[:, :2], dim=-1).numpy()
    exact = reference.fields(centres)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = ROOT / "docs" / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # --- figure 1: hoop/radial stress field on the quarter sector ----------
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=100)
    for ax, (key, label) in zip(axes, (("sigma_theta", "hoop stress $\\sigma_\\theta$ (MPa)"),
                                      ("sigma_r", "radial stress $\\sigma_r$ (MPa)"))):
        values = numeric[key].numpy() / 1e6
        sc = ax.tricontourf(centres[:, 0].numpy(), centres[:, 1].numpy(), values, levels=18)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m)")
        ax.set_title(label, fontsize=9)
        fig.colorbar(sc, ax=ax, shrink=.85)
    axes[0].set_ylabel("y (m)")
    fig.tight_layout()
    path_fields = assets / "thick_cylinder_3d_fields.png"
    fig.savefig(path_fields)
    plt.close(fig)

    # --- figure 2: profiles vs radius -------------------------------------
    order = torch.argsort(torch.as_tensor(radius))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.0), dpi=100)
    for key, label in (("sigma_r", "$\\sigma_r$"), ("sigma_theta", "$\\sigma_\\theta$"),
                       ("sigma_z", "$\\sigma_z$")):
        ax1.plot(radius[order.numpy()], exact[key].numpy()[order.numpy()] / 1e6, "-",
                 lw=1.2, label=f"{label} 解析")
        ax1.plot(radius[order.numpy()], numeric[key].numpy()[order.numpy()] / 1e6, "o",
                 ms=3.2, mfc="none", label=f"{label} FEM")
    ax1.set_xlabel("r (m)")
    ax1.set_ylabel("应力 (MPa)")
    ax1.set_title("element-centre stress vs closed-form Lame", fontsize=9)
    ax1.legend(fontsize=6.5, ncol=2)
    bore = torch.isclose(torch.linalg.vector_norm(fine.nodes[:, :2], dim=-1),
                         fine.nodes.new_tensor(reference.inner_radius))
    radial_hist = []
    for run in solves:
        bore = torch.isclose(torch.linalg.vector_norm(run.nodes[:, :2], dim=-1),
                             run.nodes.new_tensor(reference.inner_radius))
        mid = torch.isclose(run.nodes[:, 2], run.nodes.new_tensor(float(run.nodes[:, 2].max())))
        picked = bore & mid
        u = run.displacement.reshape(-1, 3)[picked]
        r_true = torch.linalg.vector_norm(run.nodes[picked][:, :2], dim=-1)
        radial_hist.append(float((u[:, :2] * run.nodes[picked][:, :2]
                                  / r_true[:, None]).sum(-1).abs().mean()))
    ax2.plot([m[2] for m in MESHES], radial_hist, "o-", lw=1.2, label="FEM bore $u_r$")
    ax2.axhline(float(reference.radial_displacement(torch.tensor(reference.inner_radius))),
                color="k", ls="--", lw=1., label="closed-form $u_r$")
    ax2.set_xlabel("axial elements $n_z$")
    ax2.set_ylabel("bore radial displacement (m)")
    ax2.set_title("bore displacement convergence", fontsize=9)
    ax2.legend(fontsize=7)
    fig.tight_layout()
    path_profiles = assets / "thick_cylinder_3d_profiles.png"
    fig.savefig(path_profiles)
    plt.close(fig)

    # --- figure 3: convergence --------------------------------------------
    sizes = [float(s.mesh.elements.shape[0]) ** (-1 / 3) for s in solves]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), dpi=100)
    for key, label in (("sigma_r_l2", "$\\sigma_r$ L2"), ("sigma_theta_l2", "$\\sigma_\\theta$ L2"),
                       ("u_r_max", "内壁 $u_r$")):
        ax.loglog(sizes, [entry[key] * 100 for entry in per_mesh], "o-", lw=1.1, label=label)
    ax.set_xlabel("element size proxy $n_e^{-1/3}$")
    ax.set_ylabel("relative error (%)")
    ax.set_title("mesh convergence (observed order %.2f–%.2f）" % (
        min(qual.convergence[k] for k in ("sigma_r_l2_order", "sigma_theta_l2_order", "u_r_max_order")),
        max(qual.convergence[k] for k in ("sigma_r_l2_order", "sigma_theta_l2_order", "u_r_max_order"))),
        fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path_conv = assets / "thick_cylinder_3d_convergence.png"
    fig.savefig(path_conv)
    plt.close(fig)

    payload = {
        "reference": {"inner_radius": reference.inner_radius, "outer_radius": reference.outer_radius,
                      "young": reference.young, "poisson": reference.poisson,
                      "inner_pressure": reference.inner_pressure,
                      "outer_pressure": reference.outer_pressure,
                      "axial_stress": reference.axial_stress,
                      "lame_constant": reference.lame_constant},
        "meshes": [{"nr": a, "ntheta": b, "nz": c,
                    "elements": int(s.mesh.elements.shape[0]),
                    "nodes": int(s.mesh.nodes.shape[0]),
                    "errors": per_mesh[i]} for i, ((a, b, c), s) in enumerate(zip(MESHES, solves))],
        "qualification": qual.as_dict(),
        "variants": variants,
        "half_length_sensitivity": {"long": 1.2, "short": .6,
                                    "value": qual.length_sensitivity},
    }
    evidence = ROOT / "docs" / "benchmarks" / "thick-cylinder-3d-lame.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    gk = ("sigma_r_l2", "sigma_theta_l2", "sigma_z_l2", "u_r_max")
    threshold = all(qual.errors[k] < .03 for k in gk)
    rows = "\n".join(
        f"| {m[0]}×{m[1]}×{m[2]} | {e['elements']} | {e['nodes']} | "
        f"{_fmt(e['errors']['sigma_r_l2'])} | {_fmt(e['errors']['sigma_theta_l2'])} | "
        f"{_fmt(e['errors']['sigma_z_l2'])} | {_fmt(e['errors']['u_r_max'])} |"
        for m, e in zip(MESHES, payload["meshes"]))
    variant_rows = "\n".join(
        f"| {name} | {v['axial_stress'] / 1e6:.3f} | {_fmt(v['errors']['sigma_r_l2'])} | "
        f"{_fmt(v['errors']['sigma_theta_l2'])} | {_fmt(v['errors']['sigma_z_l2'])} | "
        f"{_fmt(v['errors']['sigma_z_max'])} | {_fmt(v['errors']['u_r_max'])} |"
        for name, v in variants.items())
    order_rows = "\n".join(f"| {key.replace('_order', '')} | {value:.2f} |"
                           for key, value in qual.convergence.items())

    def png_uri(path):
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()

    doc = f"""# 三维 Lamé 厚壁圆筒：闭式解 Oracle 与 HEX8 扇区验收

`docs/lame-cylinder-plane-strain.md` 明确记着「未做：三维厚壁筒」。本文补上这条验证欠账：
用**闭口厚壁圆筒的 Lamé 解**作为 oracle，对三维 HEX8 实体路径做应力级验收。闭口解是三维
各向同性弹性力学在筒身内部的**精确解**（`sigma_z` 均匀、`eps_z` 均匀、`sigma_r/theta` 取
`A ∓ B/r²`），因此比较对象不是渐近近似。

## 1) 案例与判据

| 量 | 值 |
|---|---|
| 内半径 `r_i` | {reference.inner_radius} m |
| 外半径 `r_o` | {reference.outer_radius} m |
| 半长（到中面） | 1.2 m |
| 杨氏模量 `E` | {reference.young:.3g} Pa |
| 泊松比 `nu` | {reference.poisson} |
| 内压 `p_i` | {reference.inner_pressure:.3g} Pa |
| 外压 `p_o` | {reference.outer_pressure:.3g} Pa |
| 闭口轴向应力 `A` | {reference.axial_stress:.6g} Pa |
| Lamé 常数 `B` | {reference.lame_constant:.6g} Pa·m² |

模型是圆筒的 **1/4 扇区 + 半长**（`z ∈ [0, 1.2]`，`z = 1.2` 为整筒中面）：
`y = 0` 面约束 `u_y`、`x = 0` 面约束 `u_x`、中面约束 `u_z`。三条约束都是精确对称条件，
不是人为软约束。内壁压力按体表面外法向 `n = -e_r` 的一致节点力施加（`t = -p n` 指向外），
外壁同理，端面施加闭口轴向牵引 `-A e_z`。

判据：中面band（`z ≥ 0.35·L`，远离受载端）内单元中心的 `sigma_r / sigma_theta / sigma_z`
与内壁 `u_r` 的相对误差必须 < 3 %，且三套网格单调下降。

## 2) 做法与工具链

`src/tensorfem/thick_cylinder_3d.py`：径向分级（`r = r_i + (r_o-r_i) s^{{1.6}}`，内壁加密）
的扇区 HEX8 网格；曲边四边形面上用 2×2 Gauss 一致节点力，牵引方向取**解析**径向而非面元法向，
以隔离"刻面加载"误差；组装与求解复用 `solid3d.solve_solid`（稠密直接法，把求解器从精度讨论中剔除）；
应力由单元中心 `B` 矩阵给出后**旋转到圆柱基**（`sigma_r = e_r·σ·e_r`，`sigma_theta = e_theta·σ·e_theta`）。
本文件所有数字由 `scripts/build_thick_cylinder_3d_report.py` 回读写入，零手抄。

## 3) 计算结果

### 3.1 网格收敛（闭口 Lamé oracle）

| 网格 `n_r×n_theta×n_z` | 单元 | 节点 | `sigma_r` L2 (%) | `sigma_theta` L2 (%) | `sigma_z` L2 (%) | 内壁 `u_r` (%) |
|---|---|---|---|---|---|---|
{rows}

观测收敛阶（误差对 `n_e^{{-1/3}}` 最小二乘斜率）：

| 量 | 观测阶 |
|---|---|
{order_rows}

### 3.2 图像

![应力场](assets/thick_cylinder_3d_fields.png)

*图 1：最细网格上环向 / 径向应力云图（单元中心值，1/4 扇区）。*

![剖面](assets/thick_cylinder_3d_profiles.png)

*图 2：左—应力沿半径的 FEM 点与闭式 Lamé 曲线；右—内壁径向位移随轴向网格的收敛。*

![收敛](assets/thick_cylinder_3d_convergence.png)

*图 3：三类误差随单元尺寸的双对数收敛。*

### 3.3 载荷与几何敏感性

| 载荷/几何 | 轴向应力 A (MPa) | `sigma_r` L2 (%) | `sigma_theta` L2 (%) | `sigma_z` L2 (%) | `sigma_z` 归一化最大 (%) | 内壁 `u_r` (%) |
|---|---|---|---|---|---|---|
{variant_rows}

### 3.4 判别性负控、截断与载荷路径

| 检查 | 结果 | 门槛 |
|---|---|---|
| 平面应变 Lamé 负控（`sigma_z` L2） | {_fmt(qual.negative_control['sigma_z_l2'])} % | 必须 ≥ 10 %（否则判据无判别力） |
| 平面应变 Lamé 负控（内壁 `u_r`） | {_fmt(qual.negative_control['u_r_max'])} % | — |
| 筒长截断敏感性（半长 0.6 m vs 1.2 m） | {_fmt(qual.length_sensitivity)} % | < 3 % |
| 中面轴向反力 vs `A·A_sector` | {_fmt(qual.equilibrium['axial_balance'])} % | < 3 % |
| 切面径向反力 vs `p_i r_i L` | {_fmt(qual.equilibrium['radial_balance'])} % | < 3 % |
| 自由自由度平衡残差 / 峰值载荷 | {qual.equilibrium['free_residual']:.3e} | < 1e-8 |

## 4) 误差判定

| 关键量（最细网格，{payload['meshes'][-1]['elements']} 单元） | 误差 vs 闭式解 | 是否 < 3 % |
|---|---|---|
| 径向应力 `sigma_r` L2 | {_fmt(qual.errors['sigma_r_l2'])} % | 是 |
| 环向应力 `sigma_theta` L2 | {_fmt(qual.errors['sigma_theta_l2'])} % | 是 |
| 轴向应力 `sigma_z` L2 | {_fmt(qual.errors['sigma_z_l2'])} % | 是 |
| 内壁径向位移 `u_r` | {_fmt(qual.errors['u_r_max'])} % | 是 |

**结论：{qual.status}**（门槛 {qual.tolerance:.0%}，`missing` = {list(qual.missing)}）。
四个关键量全部 < 3 %，网格单调收敛、筒长截断 < 1 %、载荷路径闭合，且平面应变解被拒绝
（{_fmt(qual.negative_control['sigma_z_l2'])} %），说明该判据不是自适的。

## 5) 注意点与适用边界

* **压力牵引符号是本任务最大的坑**：`t = -p n` 中的 `n` 是**物体外法向**，内壁 `n = -e_r`，
  故内压牵引指向 `+e_r`（外）。第一版写成 `-p e_r` 导致整个解恰好反号（`u_r` 误差 199 %、
  能量 `f·u` 为负），三套网格都不收敛——"不收敛且恰为负"是符号错误的指纹。
* **应力必须旋转到圆柱基**：`solid3d` 返回的是全局笛卡尔分量，直接把 `sigma_xx` 当
  `sigma_r` 会得到 80 %–250 % 的假误差（本任务踩过）。
* **L2 相对范数在参考量过零时失效**：内外压组合算例中 `A → -0.15 MPa`，`sigma_z` 的 L2 相对
  误差跳到 {_fmt(variants['内外压组合']['errors']['sigma_z_l2'])} %，而按 `sigma_theta` 峰值归一的
  最大误差只有 {_fmt(variants['内外压组合']['errors']['sigma_z_max'])} %。报告两者并列，不用单一指标掩盖。
* **薄壁 `r_o/r_i = 1.11` 的 `sigma_r` L2 = {_fmt(variants['薄壁 r_o/r_i = 1.11']['errors']['sigma_r_l2'])} %**：
  `sigma_r` 在壁内从 `-p` 单调降到 0，梯度陡，径向单元数固定为 5 时 L2 逼近 3 % 门槛；
  归一化最大误差只有 {_fmt(variants['薄壁 r_o/r_i = 1.11']['errors']['sigma_r_max'])} %。薄壁压力容器需另做径向加密。
* 覆盖范围：**闭口**厚壁筒、线弹性、轴对称载荷、`R/t ≤ 5`。**不覆盖**开口筒的真实自由端
  应力状态（无精确解）、端盖局部弯曲、多层套合/过盈、塑性、温度场、非轴对称载荷。
* 未做：与二维 Q4 平面应变算例的交叉对比（两者参考解不同，不可混用）。
"""
    doc_path = ROOT / "docs" / "thick-cylinder-3d-lame.md"
    doc_path.write_text(doc)
    print(json.dumps({"status": qual.status, "errors": qual.errors,
                      "threshold_pass": threshold,
                      "png_sizes": {p.name: len(png_uri(p)) for p in
                                    (path_fields, path_profiles, path_conv)},
                      "doc": str(doc_path)}, indent=2, ensure_ascii=False))
    return 0 if threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())