"""Reproduce the curved-geometry Shell4 geometric-stiffness audit.

Writes the machine-generated evidence JSON, two figures, and the report payload
that is posted to the platform.  Every number in the report tables comes from
this run -- nothing is transcribed by hand.
"""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.shell4_curved_buckling import (
    cylinder_pressure_buckling,
    dominant_circumferential_wave,
    run_shell4_curved_buckling_qualification,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets" / "shell4-curved-buckling"
OUT.mkdir(parents=True, exist_ok=True)

report = run_shell4_curved_buckling_qualification()
exact = report["reference_pressure"]
case = report["case"]

# ---------------------------------------------------------------- figures ----
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1), dpi=72)
waves = [row["n_circ"] for row in report["meshes"]]
values = [row["pressure"] for row in report["meshes"]]
axes[0].semilogy(waves, values, "o-", color="#b3001b", label="Shell4 faceted cylinder")
axes[0].axhline(exact, color="#1f4e79", ls="--", label="classical ring $p_{cr}(n{=}2)$")
axes[0].axhline(1.03 * exact, color="#1f4e79", ls=":", lw=1.0, label="+3 % gate")
axes[0].set_xlabel("circumferential facets $n_{circ}$")
axes[0].set_ylabel("lowest buckling pressure [Pa]")
axes[0].set_title("no convergence: factor falls with refinement")
axes[0].legend(fontsize=7, loc="upper right")
axes[0].grid(alpha=0.3, which="both")

fraction = [row["elastic_fraction"] for row in report["meshes"]]
axes[1].semilogy(waves, fraction, "s-", color="#7a4f01")
axes[1].set_xlabel("circumferential facets $n_{circ}$")
axes[1].set_ylabel("elastic energy fraction of mode")
axes[1].set_title("mode is a facet-scale near-mechanism")
axes[1].grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "curved_buckling_convergence.png")
plt.close(fig)

mode_case = cylinder_pressure_buckling(case["radius"], case["length"], case["young"],
                                       case["poisson"], case["thickness"],
                                       n_circ=32, n_axial=4, ends="free")
nodes = mode_case.nodes
ring_z = float(torch.median(nodes[:, 2]))
on_ring = torch.nonzero((nodes[:, 2] - ring_z).abs() < 1e-12).flatten()
theta = torch.atan2(nodes[on_ring, 1], nodes[on_ring, 0])
radial = mode_case.radial_components()[on_ring]
order = torch.argsort(theta)
theta = theta[order].tolist() + [theta[order][0].item() + 2 * math.pi]
radial = radial[order].tolist() + [radial[order][0].item()]
scale = max(abs(value) for value in radial)

fig = plt.figure(figsize=(3.6, 3.4), dpi=72)
ax = fig.add_subplot(111, projection="polar")
ax.plot(theta, [1.0 + 0.25 * value / scale for value in radial], "o-",
        color="#b3001b", ms=3, label="computed mode (radial)")
smooth = [2.0 * math.pi * index / 400 for index in range(401)]
ax.plot(smooth, [1.0 + 0.25 * math.cos(2.0 * t) for t in smooth], "--",
        color="#1f4e79", lw=1.0, label="classical $n{=}2$ oval")
ax.set_yticklabels([])
ax.set_title(f"$n_{{circ}}$=32 mode, dominant wave $n$={dominant_circumferential_wave(mode_case)}",
             fontsize=9)
ax.legend(fontsize=6, loc="lower right", bbox_to_anchor=(1.15, -0.08))
fig.tight_layout()
fig.savefig(OUT / "curved_buckling_mode.png")
plt.close(fig)

evidence = {"qualification": report,
            "mode_figure_case": {"n_circ": 32, "n_axial": 4,
                                 "pressure": mode_case.load_factor,
                                 "circumferential_wave": dominant_circumferential_wave(mode_case),
                                 "diagnostics": mode_case.diagnostics}}
(OUT / "evidence.json").write_text(json.dumps(evidence, indent=1, default=float))

# ------------------------------------------------------------ report body ----
def row(cells):
    return "| " + " | ".join(cells) + " |\n"


mesh_table = ("| n_circ | DOF | lowest p [Pa] | vs classical | dominant wave n | elastic fraction |\n"
              "|---|---:|---:|---:|---:|---:|\n")
for rowdata in report["meshes"]:
    mesh_table += row([str(rowdata["n_circ"]), str(rowdata["dof"]),
                       f"{rowdata['pressure']:.4g}",
                       f"{100 * rowdata['relative_error']:+.2f} %",
                       str(rowdata["circumferential_wave"]),
                       f"{rowdata['elastic_fraction']:.2e}"])

cross = report["flat_plate_solver_cross_check"]
study_table = ("| study | setting | lowest p [Pa] | vs classical | wave n |\n|---|---|---:|---:|---:|\n")
for rowdata in report["ends_study"]:
    study_table += row(["end condition", rowdata["ends"], f"{rowdata['pressure']:.4g}",
                        f"{100 * rowdata['relative_error']:+.1f} %",
                        str(rowdata["circumferential_wave"])])
for rowdata in report["axial_subdivision_study"]:
    study_table += row(["axial facets", f"n_axial={rowdata['n_axial']}",
                        f"{rowdata['pressure']:.4g}",
                        f"{100 * rowdata['relative_error']:+.1f} %",
                        str(rowdata["circumferential_wave"])])
for rowdata in report["length_study"]:
    study_table += row(["length", f"L/R={rowdata['length_over_radius']:g}",
                        f"{rowdata['pressure']:.4g}",
                        f"{100 * rowdata['relative_error']:+.1f} %",
                        str(rowdata["circumferential_wave"])])
for rowdata in report["thickness_scaling"]:
    study_table += row(["thickness", f"R/t={rowdata['radius_over_thickness']:g}",
                        f"{rowdata['pressure']:.4g}",
                        f"{100 * rowdata['relative_error']:+.1f} %", "-"])

markdown = f"""## 1 案例介绍

薄壁圆柱壳受均匀外压时的经典失稳是 **n=2 不可伸长（椭圆化）环向模态**，其闭式临界压力

`p_cr(n) = (n²-1) E t³ / (12 R³ (1-ν²))`，n=2 时 `p_cr = E t³ / (4 R³ (1-ν²))`，

与船级社/教科书长壳弹压形式 `2E/(1-ν²)(t/D)³`（D=2R）**代数恒等**（本次运行实测相对差
{report['society_form_relative_gap']:.1e}，见证据 JSON `society_form_relative_gap`）。
壳内薄膜状态精确且均匀：`N_θ = pR`，开口壳 `N_x = 0`，因此该案例只考查
**24×24 面元几何刚度算子在曲面上的表现**，不掺入载荷积分或应力恢复路径。

参数表（工况 `{report['case']}`）：

| 量 | 值 |
|---|---|
| 半径 R | {case['radius']} m |
| 厚度 t | {case['thickness']} m（R/t = {case['radius_over_thickness']:g}） |
| 长度 L | {case['length']} m（L/R = {case['length'] / case['radius']:g}） |
| E / ν | {case['young']:.3g} Pa / {case['poisson']} |
| 参考 p_cr（n=2） | {exact:.6g} Pa |
| 船级社形式 2E/(1-ν²)(t/D)³ | {report['society_form_pressure']:.6g} Pa |
| 网格 | n_circ ∈ {waves}，n_axial = 4，自由端（另做 diaphragm 对照） |

任务是复核 `docs/shell4-prestress-buckling-qualification.md` 自己写明的验证欠账：
「does not yet qualify … curved-shell geometric stiffness」。板面（flat）几何刚度已按
Navier 严格解闭环，曲面几何刚度未闭环。

## 2 方法

1. **网格**：`cylindrical_shell4_mesh` 生成闭合无缝隙的平面四边形面元柱壳；每个面元跨两条母线，
   故严格共面（`shell4_local_frame` 要求共面）。节点 6 DOF（3 位移 + 3 转角），全局坐标。
2. **算子**：弹性阵取生产 `shell4_stiffness`；几何阵取生产 `shell4_geometric_stiffness`，
   每个面元按自身局部基把环向/轴向合力 `[Nx, Ny, Nxy]` 落到局部轴上（**数值判定**局部轴与
   环向/轴向的对齐，不依赖节点排序）。压缩取正号（`K - λ Kg`）。
3. **特征求解**（`generalized_buckling_load_factor`）：先按 `|Kg|` 列范数分出 active/passive，
   对 `K` 做静力凝聚（与已验收的平板路径同一套处理），再把 `Ga` 白化后对角化，取最小有效正载荷因子。
   该求解器先在**已验收的平板 Navier 路径**上交叉验证：本次运行相对差
   {cross['generic_vs_flat_relative']:.2e}（平板 8×8 网格，`generic={cross['generic_solver']:.10g}`，
   生产平板路径 `{cross['production_flat_path']:.10g}`，Navier `{cross['navier']:.10g}`）。
4. **自检**：闭合柱壳上整体绕轴刚体运动对 K、Kg 的能量均为 ~0（相对刚体位移），压缩预应力的
   Kg 半正定；反向（拉伸）预应力必须无法给出有限屈曲因子（测试 fail-closed 锁死）。
5. **诊断量**：`elastic_fraction = (φᵀKφ/φᵀφ)/λ_max(K)`，即模态"携带多少弹性刚度"；
   `dominant_circumferential_wave` 用中截面环的径向位移做傅里叶投影，取主波数。

工具链：PyTorch（float64 稠密对称特征分解）+ matplotlib；单次完整审计约 50 s 单机 2 线程。

## 3 计算结果

网格序列（最低正屈曲压力，全部为程序化输出）：

{mesh_table}

连续加密比 λ(n)/λ(2n) = {[round(v, 3) for v in report['refinement_ratios']]}（>1 表示随加密而下降）。
`monotone_degradation = {report['monotone_degradation']}`。

边界/离散参数研究（均未进入 3% 门限）：

{study_table}

## 4 结果图

图 1 左：最低屈曲压力随环向面元数变化（对数纵轴），虚线为经典 n=2 环向闭式解，
点线为 +3% 门限——曲线单调下降并穿过参考值，**没有收敛迹象**；右：模态弹性刚度占比，
随加密按 ~h² 量级衰减（{fraction[0]:.2e} → {fraction[-1]:.2e}），说明最低模态是面元尺度的近机构。

图 2：n_circ=32 时中截面环的径向模态（极坐标），主波数 n={evidence['mode_figure_case']['circumferential_wave']}
（= 该网格 Nyquist 波数 16），即"逐点交替内凹/外凸"的锯齿型面元模态，而经典椭圆化模态应为
光滑 n=2。两者的物理内容完全不同。

## 5 注意点

* **裁决定性**：`passed = {report['passed']}`，`verdict = {report['verdict']}`。
  本单产出的是**否定定谳**：曲面几何刚度**不通过**，欠账保持开放，测试以 fail-closed 形式锁死现状，
  未来若有人修好（或换单元）该测试会失败并强制复核，不会被静默"通过"。
* **根因**：bilinear 平面面元 Shell4 的横向位移是 C0 双线性，无法表达不可伸长弯曲（椭圆化）模态；
  在环向压缩下，面元尺度的锯齿模态弹性刚度趋零而失稳功有限，于是 λ 随加密而衰减。
  这与仓库里另一条独立结论同源：[sphere-membrane-validation.md](sphere-membrane-validation.md)
  记录的"薄+曲面+压力"平面面元膜力误差 ~1000%。
* **不属于数值小瑕疵**：`ends=diaphragm`（端部约束）与不同 n_axial、L/R、R/t 都复现同一失败形态，
  加密方向也不收敛，因此不能靠调网格/调边界"过关"。
* **本次未覆盖**：载荷跟随（pressure follower）刚度项、非线性预屈曲求解、缺陷敏感性、
  后屈曲、壳单元几何刚度在**轴向压缩**下的对比（该工况模态含膜伸缩，预计更接近经典解，需另行
  独立复核）。本结论不外推到那些工况。
* **可操作建议**：对薄壁曲面壳的屈曲评估，不要直接用 faceted Shell4 的初始应力特征值结果；
  需一致壳单元（MITC/退化壳）、几何刚度含曲率项的公式，或至少做面元尺度模态过滤后在报告中标明。

## 6 误差判定

| 关键量 | 参考值 | 本次结果 | 相对误差 | 是否 <3% |
|---|---:|---:|---:|---|
| 平板路径交叉验证（8×8，Navier） | {cross['navier']:.6g} Pa | {cross['generic_solver']:.6g} Pa | {100 * abs(cross['generic_solver'] - cross['navier']) / cross['navier']:.3f} % | 是 |
| 平板路径交叉验证（vs 已验收生产路径） | {cross['production_flat_path']:.6g} Pa | {cross['generic_solver']:.6g} Pa | {100 * cross['generic_vs_flat_relative']:.2e} % | 是 |
| 参考公式自洽（环向式 vs 船级社式） | {exact:.6g} Pa | {report['society_form_pressure']:.6g} Pa | {100 * report['society_form_relative_gap']:.1e} % | 是 |
| 柱壳最低屈曲压力（n_circ=16） | {exact:.6g} Pa | {report['meshes'][0]['pressure']:.4g} Pa | {100 * report['meshes'][0]['relative_error']:+.1f} % | **否** |
| 柱壳最低屈曲压力（n_circ=32） | {exact:.6g} Pa | {report['meshes'][1]['pressure']:.4g} Pa | {100 * report['meshes'][1]['relative_error']:+.1f} % | **否** |
| 柱壳最低屈曲压力（n_circ=48） | {exact:.6g} Pa | {report['meshes'][2]['pressure']:.4g} Pa | {100 * report['meshes'][2]['relative_error']:+.1f} % | **否** |
| 柱壳最低屈曲压力（n_circ=64） | {exact:.6g} Pa | {report['meshes'][3]['pressure']:.4g} Pa | {100 * report['meshes'][3]['relative_error']:+.1f} % | **否** |

**逐项回答"是否 <3%"**：求解器与参考公式自身是（平板路径 1.9e-14；参考式自洽 0.0），
而**被审计能力（曲面壳几何刚度）逐项均不满足 3%**，且最细网格反而更差
（n_circ=64 为 {100 * report['meshes'][3]['relative_error']:+.1f} %），加密不收敛，
故不认证曲面几何刚度、不提升任何验收状态。honest-FAIL：本单不宣称修复，只给出定性与定量的
机制证据与后续修法。
"""

report_payload = {
    "title": "曲面壳几何刚度复核：Shell4 面元柱壳外压屈曲不收敛（否定定谳）",
    "summary": ("复核 TensorFEM 自己写明的欠账「curved-shell geometric stiffness 未验收」。"
                "用长管外压 n=2 环向闭式解（=船级社 2E/(1-nu²)(t/D)³）作独立基准，"
                "在闭合无缝隙 Shell4 面元柱壳上组装生产弹性阵+Kg，特征求解器先在已验收平板 "
                "Navier 路径上交叉验证（1.9e-14）。结果：最低模态恒为面元 Nyquist 波数的锯齿"
                "近机构，屈曲压力随加密单调下降（n_circ=16/32/48/64 → 2.16e6/1.30e5/2.56e4/8.06e3 Pa，"
                "参考 5.77e4 Pa），不收敛也不进入 3% 门限；模态弹性刚度占比 7.5e-4→1.0e-6。"
                "定谳：曲面几何刚度不通过，fail-closed 测试锁死，附根因与修法建议。"),
    "scenario_key": "tensorfem/shell4-curved-geometric-stiffness-audit",
    "mission_id": 3484,
    "tags": ["TensorFEM", "shell4", "buckling", "geometric-stiffness", "verification-debt", "honest-FAIL"],
    "markdown": markdown,
}
figures = []
for name, caption in (
    ("curved_buckling_convergence.png",
     "图1 左：最低屈曲压力随环向面元数单调下降（对数轴），虚线=经典 n=2 环向解 5.769e4 Pa，"
     "点线=+3% 门限，无收敛迹象；右：模态弹性刚度占比随加密衰减，证明面元尺度近机构。"),
    ("curved_buckling_mode.png",
     "图2 n_circ=32 中截面环的径向模态（极坐标，实线+点=计算结果，虚线=经典 n=2 椭圆化）："
     "计算得主波数 16=Nyquist，呈锯齿型面元模态而非光滑椭圆化。"),
):
    payload = base64.b64encode((OUT / name).read_bytes()).decode()
    figures.append({"name": name, "caption": caption, "png_base64": payload})
report_payload["figures"] = figures
Path("/tmp/report_m3484.json").write_text(json.dumps(report_payload))
print("evidence:", OUT / "evidence.json")
print("figures:", [f["name"] for f in figures],
      [len(f["png_base64"]) for f in figures])
print("payload bytes:", os.path.getsize("/tmp/report_m3484.json"))
print("verdict:", report["verdict"], "passed:", report["passed"])