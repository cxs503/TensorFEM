#!/usr/bin/env python
"""Lightweight verification driver for GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

Runs only the analytical reference solutions (no FE) and produces:
- Numerical results table
- Figures (reference solutions, flow stress)
- Report JSON for the paper seat

FE-heavy computations (transition, limit load, convergence) are documented
with the audit findings and pending HPC job status.
"""
from __future__ import annotations

import json
import math
import os
import sys
import base64
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from tensorfem.ice_plate_plasticity import (
    flow_stress_at_temperature,
    elastic_first_yield_pressure,
    navier_centre_moments,
    yield_line_limit_pressure,
)
from tensorfem.plate_patch_plastic_limit import (
    elastic_patch_plate,
    strip_lower_bound,
    ilyushin_sine_upper_bound,
    yield_line_patch_upper_bound,
)


# ---------------------------------------------------------------------------
# Case definition (representative ice-structure panel)
# ---------------------------------------------------------------------------

CASE = dict(
    a=2.0,          # plate length [m]
    b=1.0,          # plate width [m]
    t=0.020,        # thickness [m]
    grade="DH36",
    temperature=-40.0,
    young=206e9,
    poisson=0.3,
    hardening=1000e6,
    patch=(0.5, 0.5),  # patch size [m]
)

SY = flow_stress_at_temperature(CASE["grade"], CASE["temperature"])
print(f"Case: {CASE['a']}x{CASE['b']}x{CASE['t']} m, {CASE['grade']} at {CASE['temperature']}°C")
print(f"  Yield stress: {SY/1e6:.1f} MPa")


# ---------------------------------------------------------------------------
# 1. Reference solutions (analytical, no FE)
# ---------------------------------------------------------------------------

print("\n=== Reference Solutions ===")

# First-yield pressure (returns float)
fy_pressure = elastic_first_yield_pressure(
    CASE["a"], CASE["b"], CASE["t"], SY, CASE["poisson"])
print(f"First-yield pressure (Navier): {fy_pressure/1e6:.4f} MPa")

# Yield-line upper bound (returns float)
yl_pressure = yield_line_limit_pressure(
    CASE["a"], CASE["b"], SY, CASE["t"])
print(f"Yield-line upper bound: {yl_pressure/1e6:.4f} MPa")

# Sine upper bound (returns dict)
sine_result = ilyushin_sine_upper_bound(
    CASE["a"], CASE["b"], CASE["t"], SY, CASE["patch"])
print(f"Sine upper bound: {sine_result['pressure']/1e6:.4f} MPa")

# Strip lower bound (returns dict)
strip_result = strip_lower_bound(
    CASE["a"], CASE["b"], CASE["t"], SY, CASE["patch"])
print(f"Strip lower bound: {strip_result['pressure']/1e6:.4f} MPa")

# Yield-line patch upper bound (returns dict)
yl_patch_result = yield_line_patch_upper_bound(
    CASE["a"], CASE["b"], CASE["t"], SY, CASE["patch"])
print(f"Yield-line patch UB: {yl_patch_result['pressure']/1e6:.4f} MPa")

# Navier centre moments (for Timoshenko table comparison)
mx, my = navier_centre_moments(CASE["a"], CASE["b"], CASE["poisson"])
print(f"\nNavier centre moments (p=1 Pa): Mx={mx:.6f}, My={my:.6f}")

# Bracket
print(f"\nBracket: {strip_result['pressure']/1e6:.4f} < {sine_result['pressure']/1e6:.4f} < {yl_patch_result['pressure']/1e6:.4f} MPa")
print(f"  Ratio sine/lower: {sine_result['pressure']/strip_result['pressure']:.2f}")
print(f"  Ratio yield-line/sine: {yl_patch_result['pressure']/sine_result['pressure']:.2f}")


# ---------------------------------------------------------------------------
# 2. Generate figures
# ---------------------------------------------------------------------------

print("\n=== Generating Figures ===")

figures = []

# Figure 1: Load-deflection curve (elastic)
fig, ax = plt.subplots(figsize=(8, 5))

p_vals = np.linspace(0, fy_pressure * 1.5, 30)
w_elastic = []
for p in p_vals:
    ep = elastic_patch_plate(
        CASE["a"], CASE["b"], CASE["t"], CASE["young"], CASE["poisson"],
        CASE["patch"], p, terms=41)
    w_elastic.append(ep["deflection"][0])

ax.plot(p_vals / 1e6, [w * 1000 for w in w_elastic], 'b-', linewidth=2,
        label="Elastic (Navier)")

ax.axvline(x=fy_pressure/1e6, color='r', linestyle='--', alpha=0.7,
           label=f"First yield ({fy_pressure/1e6:.3f} MPa)")
ax.axvline(x=yl_pressure/1e6, color='g', linestyle='--', alpha=0.7,
           label=f"Yield-line UB ({yl_pressure/1e6:.3f} MPa)")

ax.set_xlabel("Patch pressure [MPa]", fontsize=12)
ax.set_ylabel("Centre deflection [mm]", fontsize=12)
ax.set_title("Elastic load-deflection with yield markers\n"
             f"{CASE['a']}×{CASE['b']}×{CASE['t']} m, {CASE['grade']} at {CASE['temperature']}°C",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

buf = BytesIO()
plt.savefig(buf, format="png", dpi=150)
buf.seek(0)
figures.append({
    "name": "elastic_load_deflection",
    "caption": "Elastic load-deflection curve (Navier double-series) with first-yield "
               f"({fy_pressure/1e6:.3f} MPa) and yield-line upper bound "
               f"({yl_pressure/1e6:.3f} MPa) markers.",
    "png_base64": base64.b64encode(buf.read()).decode(),
})
plt.close()

# Figure 2: Reference solution bracket
fig, ax = plt.subplots(figsize=(8, 5))

bounds = ["Strip LB", "Sine UB", "Yield-line UB"]
values = [strip_result['pressure']/1e6, sine_result['pressure']/1e6, yl_patch_result['pressure']/1e6]
colors = ['green', 'orange', 'red']

bars = ax.barh(bounds, values, color=colors, alpha=0.7, edgecolor='black')
for bar, val in zip(bars, values):
    ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
            f"{val:.3f} MPa", va='center', fontsize=11)

ax.axvline(x=fy_pressure/1e6, color='blue', linestyle='--', alpha=0.7,
           label=f"First yield ({fy_pressure/1e6:.3f} MPa)")
ax.set_xlabel("Pressure [MPa]", fontsize=12)
ax.set_title("Plastic limit load bounds\n"
             f"{CASE['a']}×{CASE['b']}×{CASE['t']} m, σ_y = {SY/1e6:.0f} MPa",
             fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()

buf = BytesIO()
plt.savefig(buf, format="png", dpi=150)
buf.seek(0)
figures.append({
    "name": "limit_load_bounds",
    "caption": "Plastic limit load bounds: strip lower bound, sine upper bound, "
               "and yield-line upper bound. The true collapse pressure lies between "
               f"the strip LB ({strip_result['pressure']/1e6:.3f} MPa) and the "
               f"yield-line UB ({yl_patch_result['pressure']/1e6:.3f} MPa).",
    "png_base64": base64.b64encode(buf.read()).decode(),
})
plt.close()

# Figure 3: Temperature-dependent flow stress
fig, ax = plt.subplots(figsize=(8, 5))

temps = np.linspace(-60, 20, 50)
for grade in ["DH36", "EH36", "FH36"]:
    stresses = []
    for T in temps:
        try:
            s = flow_stress_at_temperature(grade, T)
            stresses.append(s / 1e6)
        except ValueError:
            stresses.append(np.nan)
    ax.plot(temps, stresses, '-', linewidth=2, label=grade)

ax.set_xlabel("Temperature [°C]", fontsize=12)
ax.set_ylabel("Yield stress [MPa]", fontsize=12)
ax.set_title("Low-temperature flow stress for Arctic-grade steels", fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()

buf = BytesIO()
plt.savefig(buf, format="png", dpi=150)
buf.seek(0)
figures.append({
    "name": "flow_stress_temperature",
    "caption": "Temperature-dependent yield stress for Arctic-grade hull steels "
               "(DH36, EH36, FH36). Linear interpolation between published data points.",
    "png_base64": base64.b64encode(buf.read()).decode(),
})
plt.close()

# Figure 4: Navier moment coefficients vs aspect ratio
fig, ax = plt.subplots(figsize=(8, 5))

aspects = np.linspace(0.5, 10, 50)
mx_vals, my_vals = [], []
for asp in aspects:
    a_val = asp * 1.0
    b_val = 1.0
    mx_c, my_c = navier_centre_moments(a_val, b_val, 0.3)
    # Normalize by p*b^2
    mx_vals.append(mx_c / (b_val ** 2))
    my_vals.append(my_c / (b_val ** 2))

ax.plot(aspects, mx_vals, 'b-', linewidth=2, label="M_x/(p b²)")
ax.plot(aspects, my_vals, 'r-', linewidth=2, label="M_y/(p b²)")
ax.axhline(y=0.0479, color='g', linestyle='--', alpha=0.5, label="Timoshenko 1×1: 0.0479")
ax.axhline(y=0.1017, color='orange', linestyle='--', alpha=0.5, label="Timoshenko 2×1: 0.1017")
ax.axhline(y=0.125, color='purple', linestyle='--', alpha=0.5, label="Strip limit: 0.125")

ax.set_xlabel("Aspect ratio a/b", fontsize=12)
ax.set_ylabel("M/(p b²)", fontsize=12)
ax.set_title("Navier centre moment coefficients vs aspect ratio", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

buf = BytesIO()
plt.savefig(buf, format="png", dpi=150)
buf.seek(0)
figures.append({
    "name": "navier_moment_coefficients",
    "caption": "Navier centre moment coefficients M/(p b²) vs aspect ratio. "
               "Dashed lines show Timoshenko & Woinowsky-Krieger table values.",
    "png_base64": base64.b64encode(buf.read()).decode(),
})
plt.close()

print(f"Generated {len(figures)} figures")


# ---------------------------------------------------------------------------
# 3. Summary
# ---------------------------------------------------------------------------

print("\n=== Acceptance Criteria Summary ===")

summary = {
    "case": CASE,
    "yield_stress_MPa": SY / 1e6,
    "reference_solutions": {
        "first_yield_pressure_MPa": fy_pressure / 1e6,
        "yield_line_upper_bound_MPa": yl_pressure / 1e6,
        "sine_upper_bound_MPa": sine_result['pressure'] / 1e6,
        "strip_lower_bound_MPa": strip_result['pressure'] / 1e6,
        "yield_line_patch_ub_MPa": yl_patch_result['pressure'] / 1e6,
        "navier_mx": mx,
        "navier_my": my,
    },
    "fe_status": {
        "note": "FE computations (transition, limit load, convergence) require HPC due to high system load.",
        "audit_fixes_applied": [
            "F1: Bisection for elastic-plastic transition (was step-quantised)",
            "F2: Removed false-pass fallback on FE failure",
            "F3: Fixed yield-line formula (was 50% low for square plate)",
            "F4: Fixed first-yield pressure (was using wrong alpha guess)",
            "F5: Fixed mesh convergence (was using hardcoded target_load)",
            "F6: Corrected docstring (no arc-length continuation)",
        ],
        "remaining_bug_fixed": "Removed 'value = target_load' line that overwrote computed limit load in verify_mesh_convergence",
        "pending_hpc_job": "cj3469 (3-grid lateral matrix, submitted by audit)",
    },
}

print(f"\nReference solutions computed successfully:")
print(f"  First-yield pressure: {fy_pressure/1e6:.4f} MPa")
print(f"  Yield-line UB: {yl_pressure/1e6:.4f} MPa")
print(f"  Sine UB: {sine_result['pressure']/1e6:.4f} MPa")
print(f"  Strip LB: {strip_result['pressure']/1e6:.4f} MPa")
print(f"  Yield-line patch UB: {yl_patch_result['pressure']/1e6:.4f} MPa")


# ---------------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------------

output_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Save summary JSON
with open(os.path.join(output_dir, "gap_pp3_plastic_set_verification.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# 5. Generate markdown report
# ---------------------------------------------------------------------------

markdown_content = f"""# GAP-PP-3-ICE-STRUCT-PLASTIC-SET 验证报告

## 1. 案例介绍

**物理背景**: 冰-结构相互作用中，板格在大斑载荷下的失效模式由塑性储备与永久挠度控制，
而非弹性应力上限。本验证针对北极级船体结构钢（DH36/EH36/FH36）在低温下的弹塑性板单元能力。

**几何参数**:

| 参数 | 值 | 单位 |
|------|-----|------|
| 板长 a | {CASE['a']} | m |
| 板宽 b | {CASE['b']} | m |
| 板厚 t | {CASE['t']} | m |
| 斑载荷尺寸 | {CASE['patch'][0]}×{CASE['patch'][1]} | m |
| 钢种 | {CASE['grade']} | - |
| 温度 | {CASE['temperature']} | °C |
| 屈服应力 σ_y | {SY/1e6:.0f} | MPa |
| 弹性模量 E | {CASE['young']/1e9:.0f} | GPa |
| 泊松比 ν | {CASE['poisson']} | - |
| 硬化模量 H | {CASE['hardening']/1e6:.0f} | MPa |

## 2. 方法

**板单元**: Layered Shell4 (Q4, 2×2 Gauss, 5层厚度积分), J2 弹塑性 + 等向硬化。

**参考解**:
- 弹性首屈压力: Navier 双正弦级数 (独立于 FE 网格)
- 塑性极限载荷上界: 屈服线法 (virtual work minimisation)
- 塑性极限载荷下界: 双向条带法 (statically admissible)

**加载**: 单调递增斑压力，Newton-Raphson 迭代，二分法精化屈服点/极限点。

## 3. 计算结果

### 3.1 参考解

| 解 | 值 [MPa] |
|----|----------|
| 弹性首屈压力 (Navier) | {fy_pressure/1e6:.4f} |
| 屈服线上界 | {yl_pressure/1e6:.4f} |
| 正弦上界 | {sine_result['pressure']/1e6:.4f} |
| 条带下界 | {strip_result['pressure']/1e6:.4f} |
| 屈服线斑载荷上界 | {yl_patch_result['pressure']/1e6:.4f} |

### 3.2 弹塑性过渡

**状态**: 算法已修复（二分法替代步长量化），FE 数值待 HPC 作业 cj3469 返回。

| 量 | 值 |
|----|-----|
| 解析屈服载荷 | {fy_pressure/1e6:.4f} MPa |
| FE 屈服载荷 | 待 HPC |
| 载荷分辨率 | ~0.0002 MPa (14 次二分精化) |
| 判据 (<5%) | 待验证 |

### 3.3 塑性极限载荷

**状态**: 算法已修复（移除 false-pass fallback），FE 数值待 HPC 作业 cj3469 返回。

| 量 | 值 |
|----|-----|
| Ilyushin 上界 (屈服线) | {yl_pressure/1e6:.4f} MPa |
| FE 极限载荷 | 待 HPC |
| 判据 (<10%) | 待验证 |

### 3.4 网格收敛

**状态**: 算法已修复（移除 `value = target_load` MOCK 行），FE 数值待 HPC。

## 4. 结果图

见 figures 字段（4 张图）：
1. 弹性载荷-挠度曲线（含屈服标记）
2. 塑性极限载荷界限图
3. 低温钢流变应力-温度曲线
4. Navier 弯矩系数 vs 长宽比

## 5. 审计修复记录

本单完成以下审计发现 (AUDIT-GAP-PP-3-ICE-STRUCT-PLASTIC-SET) 的修复：

| 编号 | 严重度 | 问题 | 修复 |
|------|--------|------|------|
| F1 | blocking | 步长量化的接受准则 | 二分法精化 |
| F2 | blocking | FE 失败时 fallback 到参考值 | 移除 fallback |
| F3 | blocking | 屈服线公式错误（方板低 50%） | 虚功最小化 |
| F4 | blocking | 首屈压力使用错误 alpha 猜测 | Navier 级数 |
| F5 | blocking | 网格收敛使用硬编码 target_load | 逐网格计算 |
| F6 | overstated | 文档声称 arc-length 续延 | 修正文档 |
| **本单** | **blocking** | **`value = target_load` 覆盖计算值** | **删除该行** |

## 6. 注意点

1. **小转动假设**: 板单元为小转动 Mindlin 壳，挠度/厚度比在几以内有效；
   大挠度薄膜 stiffening 未捕捉。
2. **流变应力模型**: 线性插值拟合，不含应变率效应和韧脆转变。
3. **斑载荷**: 通过 tributary area 施加节点力，未实现切割单元压力积分。
4. **极限载荷**: 载荷控制可能在真实极限点前停止（Newton 发散），
   因此 FE 极限载荷是下界估计。
5. **网格**: 本验证使用解析参考解；完整 FE 验证需提交 HPC 作业。

## 7. 误差判定

| 关键量 | 基准 | FE 值 | 误差 | 是否<3% |
|--------|------|-------|------|---------|
| 弹塑性过渡点 | {fy_pressure/1e6:.4f} MPa | 待 HPC | - | 待验证 |
| 塑性极限载荷 | {yl_pressure/1e6:.4f} MPa | 待 HPC | - | 待验证 |

**诚实声明**: 本单完成了参考解验证和审计修复（含一个残留 MOCK bug 修复），
FE 数值验证因系统负载过高 (load15 > 150) 无法本地执行，需 HPC 作业完成。
"""

# Save markdown report
with open(os.path.join(output_dir, "gap_pp3_plastic_set_verification.md"), "w") as f:
    f.write(markdown_content)

# Save report JSON for API
report = {
    "title": "GAP-PP-3-ICE-STRUCT-PLASTIC-SET: 板格低温钢弹塑性 + 永久挠度验证报告",
    "summary": (
        f"Verification of elastoplastic plate capability for ice-structure interaction. "
        f"Case: {CASE['a']}×{CASE['b']}×{CASE['t']} m, {CASE['grade']} at {CASE['temperature']}°C "
        f"(σ_y = {SY/1e6:.0f} MPa). "
        f"Reference solutions computed: first-yield {fy_pressure/1e6:.4f} MPa, "
        f"yield-line UB {yl_pressure/1e6:.4f} MPa, strip LB {strip_result['pressure']/1e6:.4f} MPa. "
        f"Audit fix: removed 'value = target_load' MOCK line in verify_mesh_convergence. "
        f"FE verification pending HPC (system load > 150)."
    ),
    "scenario_key": "GAP-PP-3-ICE-STRUCT-PLASTIC-SET",
    "mission_id": 21784,
    "tags": ["ice-structure", "plasticity", "plate", "verification", "low-temperature-steel", "audit-fix"],
    "markdown": markdown_content,
    "figures": figures,
}

with open("/tmp/report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

print(f"\nReport saved to /tmp/report.json")
print(f"Summary saved to {output_dir}/gap_pp3_plastic_set_verification.json")
print(f"Markdown saved to {output_dir}/gap_pp3_plastic_set_verification.md")
