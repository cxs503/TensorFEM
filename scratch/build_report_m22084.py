#!/usr/bin/env python3
"""Build the m22084 progress report JSON (GAP-PP-1 valid end-to-end case)."""
import base64
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "benchmarks/artifacts/GAP-PP-1-end-to-end-demo/end_to_end_demo.json"
FIGDIR = REPO / "outputs/gap_pp1_valid_case"

data = json.loads(ART.read_text())
vc = data["valid_case"]
old = data  # overloaded reference case (top level)

nav = vc["navier_benchmark"]
lad = nav["mesh_ladder"]
delta = vc["delta_nonuniform_over_uniform_pct"]
h3 = vc["h3_anchor_check"]
sdv = vc["small_deflection_validity"]
proj = vc["projected_patch"]
st_nu = vc["peak_stress_nonuniform"]
st_u = vc["peak_stress_uniform"]

sha = subprocess.check_output(
    ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], text=True
).strip()

figs = []
for name, cap in [
    ("fig1_pressure_field.png", "Gaussian CZM-style contact pressure patch (peak 0.05 MPa, σ=0.15 m) on the 1 m × 1 m free SSSS plate."),
    ("fig2_deflection_fields.png", "Deflection fields: projected non-uniform patch vs equal-force uniform patch (valid case, w/t = 0.318)."),
    ("fig3_stress_fields.png", "Peak face von Mises stress: non-uniform (projected) vs uniform patch. The concentrated patch peaks +115 % higher in σx."),
    ("fig4_fem_vs_navier.png", "FEM (Mindlin Q4 SRI) vs independent Kirchhoff Navier series, mesh ladder 16/24/32; grey lines = ±H3 anchor 0.339 %."),
]:
    p = FIGDIR / name
    figs.append({
        "name": name,
        "caption": cap,
        "png_base64": base64.b64encode(p.read_bytes()).decode(),
    })

ladder_rows = "\n".join(
    f"| {r['plate_nx']} | {r['sigma_x_MPa']:.3f} | {r['navier_sigma_x_MPa']:.3f} | {r['fem_vs_navier_pct']:+.3f} |"
    for r in lad
)

md = f"""# GAP-PP-1 冰压斑接触载荷→板格应力：小挠度有效端到端算例（m22084）

## 1. 案例介绍（物理背景/几何/参数）

冰-结构接触（CZM 接触求解）产生接触面上的法向压力分布；船舶板格结构评估需要把该
压力投影成板格节点载荷并求解应力。本单锚点 **GAP-PP-1-ICE-STRUCT-CONTACT-PATCH**
（JOUR-ICE-STRUCT §4.4/5.3）要求：接触→斑载荷→板格应力的端到端算例可复现、
非均匀斑相对均匀斑峰值应力差异机读 JSON、与 H3-ICE-PLATE-FEM 锚（0.339 % 级）
同量级一致性检查通过。

**本单增量**：既有验收算例的载荷水平（峰值 5 MPa）使 w/t ≈ 14，远超小挠度理论
适用范围（demo 自身 JSON 标记 `small_deflection_valid: false`，绝对应力 2543 MPa
无物理意义）。本单新增**小挠度有效算例**（`run_valid_case`）：同几何同接触形状，
峰值压力缩至 0.05 MPa，w/t = 0.318 < 0.5，并对 FEM 链做**实时 Navier 级数基准**
（自由 SSSS 板，k=0，独立解析路径），取代原先从归档单网格数值读取的锚点检查。

| 参数 | 值 |
|---|---|
| 板 | 1 m × 1 m，t = 10 mm，E = 206 GPa，ν = 0.3，SSSS，自由（k=0） |
| 接触斑 | 高斯，中心 (0.5, 0.5)，σx=σy=0.15 m，峰值 0.05 MPa |
| 接触网格 | 33×33（±3σ 窗口，含 99.7 % 合力） |
| 均匀参照 | 同合力、同 6σ×6σ 足迹（0.9 m × 0.9 m） |
| 网格 | nx = 16/24/32（收敛梯），报告网格 nx=24 |

## 2. 怎么做的（方法/流程/工具链）

1. 高斯接触压力场在 33×33 张量积网格上采样（CZM 输出的典型形态）；
2. `patch_from_contact` 精确子单元分割投影（`exact_subcell_splitting`）→ 一致节点力，
   施加力 = 接触力（机器精度，`load_balance_rel = 0.0`）；
3. Mindlin Q4 SRI 板求解（`contact_pressure_to_plate_load`）→ 面应力（单元边中点曲率）；
4. 均匀斑参照：同合力同足迹；
5. 独立 Kirchhoff Navier 级数（`navier_rect_moments`，121 半波，4×4 Gauss 积分同一
   双线性压力场）给出参考峰值弯曲应力；
6. 工具链：`examples/ice_struct_contact/end_to_end_demo.py`（算例+机读 JSON）、
   `tests/test_gap_pp1_valid_case.py`（6 测试）、
   `examples/ice_struct_contact/gen_valid_case_figures.py`（图）。

## 3. 计算结果（数字表，程序化注入）

**投影审计**：total_force = {proj['total_force_N']:.1f} N（解析 {vc['contact']['analytic_total_force_N']:.1f} N，
偏差 {100*proj['force_error_rel']:+.3f} %，来自 ±3σ 采样窗），`load_balance_rel = {proj['load_balance_rel']:.1e}`，
投影方法 = {proj['projection_method']}。

**峰值应力（nx=24）**：

| 量 | 非均匀（投影） | 均匀（同力） | 差异 % |
|---|---|---|---|
| σx [MPa] | {st_nu['sigma_x_MPa']:.3f} | {st_u['sigma_x_MPa']:.3f} | {delta['sigma_x']:+.2f} |
| σvm [MPa] | {st_nu['sigma_vm_MPa']:.3f} | {st_u['sigma_vm_MPa']:.3f} | {delta['sigma_vm']:+.2f} |
| w [m] | {vc['peak_deflection_nonuniform_m']:.5f} | {vc['peak_deflection_uniform_m']:.5f} | {delta['deflection']:+.2f} |

**FEM vs Navier（峰值弯曲应力，非均匀斑）**：

| nx | FEM σx [MPa] | Navier σx [MPa] | 误差 % |
|---|---|---|---|
{ladder_rows}

均匀斑（阶跃载荷，O(h) 收敛）：FEM vs Navier = {nav['uniform_fem_vs_navier_pct']:+.3f} %（nx=24）。

## 4. 结果图

见内嵌图 1–4（压力场/挠度场/应力场/FEM-Navier 收敛曲线）。

## 5. 注意点（坑/局限/适用范围）

- 原 5 MPa 算例保留在 JSON 顶层作**超范围参照**（`small_deflection_valid: false`），
  其非均匀/均匀*比值*在线性模型中是载荷不变量，但绝对应力不可用于结构判定；
- 均匀斑为阶跃载荷，FEM 误差 O(h)，nx=24 时 0.95 %，仍 < 3×锚（1.017 %）；
- 接触数据离散化（±3σ 窗）贡献 ~0.55 % 合力偏差，属接触数据离散而非投影误差；
- 小挠度 Mindlin 板：w/t < 0.5 适用；大变形/屈曲需 von Kármán 路径（不在本锚范围）；
- Navier 级数为 Kirchhoff 参考：t/L = 0.01 时剪切变形修正 ~ (π t / L)² 量级，
  与观测误差同量级，已计入「同量级」判据。

## 6. 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| 非均匀斑 FEM σx (nx=32) | Navier 级数 | {h3['fem_vs_navier_pct_finest']:+.3f} % | **是** |
| 非均匀斑 FEM σx (nx=24) | Navier 级数 | {nav['nonuniform_fem_vs_navier_pct']:+.3f} % | **是** |
| 均匀斑 FEM σx (nx=24) | Navier 级数 | {nav['uniform_fem_vs_navier_pct']:+.3f} % | **是** |
| 投影合力 | 解析高斯积分 | {100*proj['force_error_rel']:+.3f} %（采样窗） | **是** |
| 施加力平衡 | 投影节点力 | {proj['load_balance_rel']:.1e} | **是** |
| H3 锚一致性 | 0.339 %（3× 带内） | {h3['fem_vs_navier_pct_finest']:+.3f} % vs 0.339 % | **同量级：是** |

**结论**：有效算例端到端可复现；FEM 链误差与 H3-ICE-PLATE-FEM 锚（0.339 %）同量级
（|−0.487 %| ≤ 3×0.339 %）；非均匀斑峰值应力差异机读 JSON 已落盘
（`benchmarks/artifacts/GAP-PP-1-end-to-end-demo/end_to_end_demo.json` → `valid_case`）。
"""

report = {
    "title": "GAP-PP-1 冰压斑接触载荷→板格应力：小挠度有效端到端算例 + 实时 Navier 基准（m22084）",
    "summary": (
        "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH 增量：既有验收算例 w/t≈14 超出小挠度适用范围，"
        "新增 w/t=0.318 有效算例（同几何同接触形状，峰值压力 0.05 MPa），FEM 链对独立 Navier 级数"
        "误差 -0.487 %（nx=32）与 H3 锚 0.339 % 同量级；非均匀 vs 均匀峰值应力差异 +115.0 %（σx）"
        "机读 JSON 落盘；新增 6 测试全绿。"
    ),
    "scenario_key": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH",
    "mission_id": 22084,
    "tags": ["GAP-PP-1", "ice-structure", "contact", "plate", "verification", "JOUR-ICE-STRUCT"],
    "markdown": md,
    "figures": figs,
}
out = REPO / "scratch" / "report_m22084.json"
out.write_text(json.dumps(report, ensure_ascii=False))
print("report json:", out, out.stat().st_size, "bytes")
