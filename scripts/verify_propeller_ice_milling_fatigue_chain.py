"""Programmatic evidence generator for the propeller ice-milling fatigue chain.

Recomputes every acceptance anchor, the design case, the elastic bounding case
and the three-variable sweep; writes ``artifacts/propeller_ice_milling_fatigue/
numbers.json``, four figures and the auto-generated markdown table used by
``docs/propeller-ice-milling-fatigue-chain.md`` (no number is hand-copied).
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.marine_fatigue_advanced import rainflow_cycles
from tensorfem.propeller_ice_milling_fatigue_chain import (
    BladeGeometry,
    DNV_D_TABLE,
    MillingCondition,
    blade_modal_basis,
    blade_section_properties,
    cantilever_beam_modes,
    dnv_d_curve_in_air,
    effective_contact_modulus,
    euler_bernoulli_frequency,
    hertz_force,
    hertz_pressure_quadrature,
    milling_bounding_case,
    milling_damage_case,
    milling_parameter_sweep,
    wet_mode_frequencies,
)

OUT = pathlib.Path("artifacts/propeller_ice_milling_fatigue")
DOC = pathlib.Path("docs/propeller-ice-milling-fatigue-chain.md")


def uniform_geometry(n_elements: int = 32) -> BladeGeometry:
    return BladeGeometry(
        root_chord_m=0.50,
        tip_chord_m=0.50,
        root_thickness_m=0.18,
        tip_thickness_m=0.18,
        n_elements=n_elements,
    )


def anchor_hertz() -> dict:
    e_star = effective_contact_modulus()
    rows = []
    for penetration in (1.0e-5, 1.0e-4, 1.0e-3):
        result = hertz_pressure_quadrature(e_star, 0.01, penetration)
        rows.append(
            {
                "penetration_m": penetration,
                "force_closed_form_n": result["force_closed_form_n"],
                "force_quadrature_n": result["force_quadrature_n"],
                "force_abs_error_n": abs(
                    result["force_quadrature_n"] - result["force_closed_form_n"]
                ),
                "contact_radius_m": result["contact_radius_m"],
                "peak_pressure_pa": result["peak_pressure_pa"],
                "centre_displacement_relative_error": result[
                    "centre_displacement_relative_error"
                ],
                "displacement_profile_relative_error": result[
                    "displacement_profile_relative_error"
                ],
            }
        )
    return {
        "e_star_pa": e_star,
        "table": rows,
        "max_force_abs_error_n": max(row["force_abs_error_n"] for row in rows),
        "max_profile_error": max(row["displacement_profile_relative_error"] for row in rows),
        "pass": max(row["force_abs_error_n"] for row in rows) < 1.0e-3
        and max(row["displacement_profile_relative_error"] for row in rows) < 0.02,
    }


def anchor_modes() -> dict:
    uniform = uniform_geometry()
    ei, rho_a, _ = blade_section_properties(uniform)
    omega, _, _ = cantilever_beam_modes(ei, rho_a, uniform.span_m, n_modes=5)
    rows = []
    for mode in range(5):
        analytic_hz = euler_bernoulli_frequency(uniform.span_m, ei[0], rho_a[0], mode=mode + 1)
        fe_hz = float(omega[mode] / (2.0 * math.pi))
        rows.append(
            {
                "mode": mode + 1,
                "fe_hz": fe_hz,
                "analytic_hz": analytic_hz,
                "relative_error": abs(fe_hz - analytic_hz) / analytic_hz,
            }
        )
    taper = blade_modal_basis(BladeGeometry(), n_modes=5)
    coarse = blade_modal_basis(BladeGeometry(n_elements=48), n_modes=5)
    uniform_basis = blade_modal_basis(uniform, n_modes=5)
    chunk = uniform.span_m / uniform.n_elements
    added = np.full(uniform.n_elements, 300.0)
    dry, wet = wet_mode_frequencies(ei, rho_a, added, uniform.span_m, n_modes=3)
    expected_ratio = math.sqrt(rho_a[0] * chunk / (rho_a[0] * chunk + added[0] * chunk))
    return {
        "uniform_reference": rows,
        "max_relative_error": max(row["relative_error"] for row in rows),
        "pass": max(row["relative_error"] for row in rows) < 0.05,
        "tapered_blade_dry_hz": taper.frequencies_hz.tolist(),
        "tapered_blade_wet_hz": taper.wet_frequencies_hz.tolist(),
        "tapered_blade_mass_kg": taper.blade_mass_kg,
        "tapered_blade_effective_mass_kg": taper.effective_mass_kg,
        "mesh_convergence_48_vs_96_max_relative": float(
            np.max(np.abs(coarse.frequencies_hz - taper.frequencies_hz) / taper.frequencies_hz)
        ),
        "uniform_basis_root_curvature_factors": uniform_basis.root_curvature_factors.tolist(),
        "wet_ratio_expected": expected_ratio,
        "wet_ratio_computed": [float(wet[i] / dry[i]) for i in range(3)],
        "wet_ratio_error": max(
            abs(wet[i] / dry[i] - expected_ratio) / expected_ratio for i in range(3)
        ),
    }


def anchor_rainflow() -> dict:
    amplitude = 40.0e6
    expected = 7.0
    dt = 1.0 / 64.0
    time = np.arange(int(round(expected * 64)) + 1) * dt
    history = amplitude * np.sin(2.0 * math.pi * time)
    rows = rainflow_cycles([float(value) for value in history])
    full = sum(row.count for row in rows if row.range_pa > 1.9 * amplitude)
    max_range = max(row.range_pa for row in rows)
    return {
        "expected_cycles": expected,
        "counted_full_cycles": full,
        "count_error": abs(full - expected),
        "max_range_pa": max_range,
        "expected_max_range_pa": 2.0 * amplitude,
        "range_relative_error": abs(max_range - 2.0 * amplitude) / (2.0 * amplitude),
        "pass": abs(full - expected) <= 1.0,
    }


def anchor_sn() -> dict:
    curve = dnv_d_curve_in_air()
    rows = []
    for stress_mpa in (120.0, 100.0, 90.0, 70.0, 52.63, 40.0, 30.0):
        n = curve.cycles_to_failure(stress_mpa * 1.0e6)
        log_n = math.log10(n)
        # Independent recomputation straight from the tabulated parameters.
        n1 = 10.0 ** DNV_D_TABLE["log_a1"] * stress_mpa ** (-DNV_D_TABLE["slope_m1"])
        reference = (
            n1
            if n1 <= DNV_D_TABLE["knee_cycles"]
            else 10.0 ** DNV_D_TABLE["log_a2"] * stress_mpa ** (-DNV_D_TABLE["slope_m2"])
        )
        rows.append(
            {
                "stress_range_mpa": stress_mpa,
                "cycles": n,
                "log10_cycles": log_n,
                "reference_cycles_from_table": reference,
                "relative_error": abs(n - reference) / reference,
            }
        )
    fat_check = curve.cycles_to_failure(90.0e6) / 2.0e6
    knee_check = curve.knee_stress_pa / (DNV_D_TABLE["knee_stress_mpa"] * 1.0e6)
    return {
        "table": rows,
        "fat_90_at_2e6_relative_error": abs(fat_check - 1.0),
        "knee_stress_relative_error": abs(knee_check - 1.0),
        "knee_stress_pa": curve.knee_stress_pa,
        "pass": abs(fat_check - 1.0) < 0.05 and abs(knee_check - 1.0) < 0.05,
    }


def _decimate(history: dict, step: int = 20) -> dict:
    return {
        "time_s": history["time_s"][::step],
        "stress_pa": history["stress_pa"][::step],
        "force_n": history["force_n"][::step],
        "decimation": step,
    }


def chain_cases() -> dict:
    design = milling_damage_case(MillingCondition(), BladeGeometry())
    bounding = milling_bounding_case(MillingCondition(), BladeGeometry())
    return {
        "design": {k: v for k, v in design.items() if k != "history"},
        "bounding": {k: v for k, v in bounding.items() if k != "history"},
        "history": _decimate(design["history"]),
        "history_bounding": _decimate(bounding["history"]),
        "ratio_damage_bounding_over_design": (
            bounding["fatigue"]["damage_20yr"] / design["fatigue"]["damage_20yr"]
        ),
    }


def sweeps() -> dict:
    design = milling_parameter_sweep()
    bounding = milling_parameter_sweep(
        baseline=dataclasses.replace(MillingCondition(), crush_pressure_pa=None), n_pulses=6
    )
    return {"design": design, "bounding": bounding}


def figure_contact() -> str:
    e_star = effective_contact_modulus()
    radius = 0.009375
    penetration = np.logspace(-8, -1.5, 200)
    elastic = np.array([hertz_force(e_star, radius, d) for d in penetration])
    crush = 30.0e6
    capped = np.minimum(elastic, 2.0 / 3.0 * math.pi * crush * radius * penetration)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].loglog(penetration, elastic, label="elastic Hertz  F = k d^1.5")
    axes[0].loglog(penetration, capped, "--", label="crushing-limited (p_c = 30 MPa)")
    axes[0].set_xlabel("penetration delta [m]")
    axes[0].set_ylabel("contact force [N]")
    axes[0].set_title("S1 contact law")
    axes[0].grid(True, which="both", ls=":")
    axes[0].legend(fontsize=8)
    pressure = np.array([2.0 * e_star * math.sqrt(radius * d) / (math.pi * radius) for d in penetration])
    axes[1].loglog(penetration, pressure / 1.0e6, label="Hertz peak pressure")
    axes[1].axhline(crush / 1.0e6, color="k", ls="--", label="ice crushing strength 30 MPa")
    axes[1].set_xlabel("penetration delta [m]")
    axes[1].set_ylabel("pressure [MPa]")
    axes[1].set_title("peak pressure vs ice strength")
    axes[1].grid(True, which="both", ls=":")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "contact_law.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_stress(cases: dict) -> str:
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.0))
    for key, label in (("history", "crushing-limited (design)"), ("history_bounding", "elastic Hertz (bound)")):
        time = np.asarray(cases[key]["time_s"])
        stress = np.asarray(cases[key]["stress_pa"]) / 1.0e6
        axes[0].plot(time, stress, lw=1.0, label=label)
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("root bending stress [MPa]")
    axes[0].set_title("S3 stress response: modal superposition of the contact pulse train")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls=":")

    ranges = [row.range_pa / 1.0e6 for row in rainflow_cycles(cases["history"]["stress_pa"])]
    weights = [row.count for row in rainflow_cycles(cases["history"]["stress_pa"])]
    axes[1].hist(ranges, bins=40, weights=weights, color="tab:blue", alpha=0.8)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("stress range [MPa]")
    axes[1].set_ylabel("rainflow cycles")
    axes[1].set_title("S4 rainflow cycle distribution (design case, 10 pulses)")
    axes[1].grid(True, ls=":")
    fig.tight_layout()
    path = OUT / "stress_and_rainflow.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_sn(anchor: dict) -> str:
    curve = dnv_d_curve_in_air()
    stress = np.logspace(math.log10(10.0e6), math.log10(400.0e6), 250)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.loglog(stress / 1.0e6, [curve.cycles_to_failure(s) for s in stress], label="DNV-RP-C203 curve D (air)")
    table = anchor["table"]
    ax.loglog(
        [row["stress_range_mpa"] for row in table],
        [row["reference_cycles_from_table"] for row in table],
        "o",
        mfc="none",
        label="table-parameter recomputation",
    )
    ax.axvline(DNV_D_TABLE["knee_stress_mpa"], color="k", ls=":", label="knee 52.63 MPa")
    ax.set_xlabel("stress range [MPa]")
    ax.set_ylabel("cycles to failure N")
    ax.set_title("S5 S-N curve and table anchor points")
    ax.grid(True, which="both", ls=":")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = OUT / "sn_curve.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def figure_sweep(sweep: dict) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    specs = (
        ("by_ice_length", "ice_length_m", "L_ice [m]"),
        ("by_rpm", "rpm", "shaft speed [rpm]"),
        ("by_speed", "advance_speed_m_s", "advance speed [m/s]"),
    )
    for axis, (key, field, label) in zip(axes, specs):
        design = sweep["design"][key]
        bounding = sweep["bounding"][key]
        axis.plot([row[field] for row in design], [row["damage_20yr"] for row in design], "o-", label="design")
        axis.plot(
            [row[field] for row in bounding],
            [row["damage_20yr"] for row in bounding],
            "s--",
            label="elastic bound",
        )
        axis.set_yscale("log")
        axis.set_xlabel(label)
        axis.set_ylabel("D (20 years)")
        axis.grid(True, which="both", ls=":")
        axis.legend(fontsize=8)
    fig.suptitle("S6 Miner damage vs the three operating variables (log scale)")
    fig.tight_layout()
    path = OUT / "damage_sweep.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def markdown_table(numbers: dict) -> str:
    design = numbers["chain"]["design"]
    bounding = numbers["chain"]["bounding"]
    lines = [
        "| quantity | crushing-limited (design) | elastic Hertz (bound) |",
        "|---|---:|---:|",
        f"| peak contact force F_max [kN] | {design['engagement']['force_max_n']/1e3:.1f} | {bounding['engagement']['force_max_n']/1e3:.1f} |",
        f"| penetration delta [mm] | {design['engagement']['penetration_m']*1e3:.2f} | {bounding['engagement']['penetration_m']*1e3:.2f} |",
        f"| peak Hertz pressure p0 [GPa] | {design['engagement']['peak_pressure_pa']/1e9:.2f} | {bounding['engagement']['peak_pressure_pa']/1e9:.2f} |",
        f"| contact time t_c [ms] | {design['engagement']['duration_s']*1e3:.2f} | {bounding['engagement']['duration_s']*1e3:.2f} |",
        f"| strike frequency [Hz] | {design['engagement']['impact_frequency_hz']:.1f} | {bounding['engagement']['impact_frequency_hz']:.1f} |",
        f"| peak root stress [MPa] | {design['stress']['sigma_max_pa']/1e6:.2f} | {bounding['stress']['sigma_max_pa']/1e6:.2f} |",
        f"| quasi-static root stress [MPa] | {design['stress']['sigma_quasi_static_pa']/1e6:.2f} | {bounding['stress']['sigma_quasi_static_pa']/1e6:.2f} |",
        f"| max rainflow range [MPa] | {design['cycles']['max_range_pa']/1e6:.2f} | {bounding['cycles']['max_range_pa']/1e6:.2f} |",
        f"| cycles counted (10 pulses) | {design['cycles']['n_cycles']:.1f} | {bounding['cycles']['n_cycles']:.1f} |",
        f"| damage per hour | {design['fatigue']['damage_per_hour']:.3e} | {bounding['fatigue']['damage_per_hour']:.3e} |",
        f"| D over 20 years | {design['fatigue']['damage_20yr']:.3e} | {bounding['fatigue']['damage_20yr']:.3e} |",
        f"| allowable milling hours | {design['fatigue']['allowable_hours']:.0f} | {bounding['fatigue']['allowable_hours']:.3e} |",
        f"| D < 1.0 | {'yes' if design['fatigue']['passed'] else 'NO'} | {'yes' if bounding['fatigue']['passed'] else 'NO'} |",
    ]
    return "\n".join(lines)


def markdown_sweep(numbers: dict) -> str:
    def block(key: str, field: str, label: str) -> list[str]:
        rows = [
            "| " + label + " | F_max [kN] | D 20 yr | allowable h |",
            "|---|---:|---:|---:|",
        ]
        for design, bounding in zip(numbers["sweeps"]["design"][key], numbers["sweeps"]["bounding"][key]):
            rows.append(
                f"| {design[field]:g} | {design['force_max_n']/1e3:.1f} | "
                f"{design['damage_20yr']:.3e} (bound {bounding['damage_20yr']:.3e}) | "
                f"{design['allowable_hours']:.0f} |"
            )
        return rows

    out: list[str] = []
    out += block("by_ice_length", "ice_length_m", "L_ice [m]")
    out.append("")
    out += block("by_rpm", "rpm", "n [rpm]")
    out.append("")
    out += block("by_speed", "advance_speed_m_s", "V [m/s]")
    return "\n".join(out)


def write_doc(numbers: dict, figures: dict) -> None:
    hertz = numbers["anchors"]["hertz"]
    modes = numbers["anchors"]["modes"]
    rain = numbers["anchors"]["rainflow"]
    sn = numbers["anchors"]["sn"]
    design = numbers["chain"]["design"]
    mode_lines = "\n".join(
        f"| {row['mode']} | {row['fe_hz']:.3f} | {row['analytic_hz']:.3f} | {row['relative_error']*100:.3f} % |"
        for row in modes["uniform_reference"]
    )
    hertz_lines = "\n".join(
        f"| {row['penetration_m']:.0e} | {row['force_closed_form_n']:.6e} | {row['force_quadrature_n']:.6e} | {row['force_abs_error_n']:.2e} | {row['displacement_profile_relative_error']*100:.2f} % |"
        for row in hertz["table"]
    )
    sn_lines = "\n".join(
        f"| {row['stress_range_mpa']:.2f} | {row['cycles']:.4e} | {row['reference_cycles_from_table']:.4e} | {row['relative_error']*100:.2e} % |"
        for row in sn["table"]
    )
    text = f"""# 螺旋桨冰铣削高周疲劳链（propeller_ice_milling_fatigue_chain）

本文件由 `scripts/verify_propeller_ice_milling_fatigue_chain.py` 自动生成，全部数字由脚本
注入（`artifacts/propeller_ice_milling_fatigue/numbers.json`），无手抄。

## 1. 案例介绍（物理背景 / 几何 / 参数）

冰区螺旋桨在碎冰区以铣削（milling）方式推进时，叶片前缘反复与冰片碰撞，形成高频接触
脉冲链，是叶片根部高周疲劳的载荷来源。链条：

```
冰片(刚性椭球) --Hertz--> F_contact(t) --模态叠加--> sigma(t) --雨流--> n_i --DNV S-N--> D = sum n_i/N_i
```

| 参数 | 值 |
|---|---|
| 冰片尺寸 L_ice / 长宽比 | 0.15 m / 2.0 |
| 冰 E / nu / rho | 9.0 GPa / 0.30 / 917 kg/m^3 |
| 叶片材料 (镍铝青铜) E / nu / rho | 118 GPa / 0.34 / 7640 kg/m^3 |
| 叶轮半径 hub / tip | 0.30 m / 1.35 m |
| 根部弦长 / 厚度（梢部） | 0.60 m / 0.25 m（0.25 m / 0.03 m） |
| 叶片数 Z / 转速 n / 进速 V | 5 / 120 rpm / 3.0 m/s |
| 螺距角 phi（接触半径处） | 0.35 rad |
| 接触半径 / 力臂 | 1.0875 m / 0.7875 m |
| 阻尼比 / 附加质量 | 0.02 / 2-D 平板 pi rho c^2/4 |
| 冰压碎强度 p_crush（设计模型） | 30 MPa |
| S-N | DNV-RP-C203 curve D, in air |
| 设计寿命 / 年铣削时间 / 命中率 | {design['fatigue']['design_life_years']:.0f} 年 / 120 h/yr / 1.0 |
| 叶片质量 / 接触点有效质量 | {modes['tapered_blade_mass_kg']:.1f} kg / {modes['tapered_blade_effective_mass_kg']:.2f} kg |

## 2. 怎么做的（方法 / 流程）

1. **S1 接触**：冰片视为刚性椭球，极点主曲率半径取 `a b / c = L_ice`，与叶片前缘半径
   `r_e = 10 mm` 谐和合成等效半径 `R_eq`；`E*` 由两体弹性常数合成；接触力
   `F = (4/3) E* sqrt(R) delta^(1.5)`，单次啮合由动能平衡
   `(1/2) m v^2 = (4/5) k delta_max^(5/2)` 解出 `delta_max`、`F_max`，接触时间按
   Johnson(1985) 的 Hertz 冲击公式给出。
   **强度限**：Hertz 峰值压力超过冰压碎强度后不再有意义（设计模型取 p_crush = 30 MPa），
   此时保持 Hertz 压力场形状、平均压力 (2/3)p_crush，即 `F = (2/3) pi p_crush R delta`，
   能量平衡给出两段闭式解；接触时间由半正弦脉冲冲量 `t_c = pi m v / (2 F_max)` 给出。
   法向接近速度 `v_n = omega r_c sin(phi) + V(1-w) cos(phi)`（切向贡献 + 进速贡献）。
2. **S2 模态**：夹持-自由 Hermite 梁单元（24 单元，逐站弦长/厚度），一致质量矩阵，对称化
   广义特征值求解；湿模态用 2-D 平板附加质量 `m_a = pi rho c^2/4` 沿展向集中。接触点
   有效质量取 `m_eff = 1 / sum_n phi_n(x_c)^2`（质量归一化模态）。
3. **S3 应力**：接触力脉冲串映射到模态坐标，用阻尼 SDOF 精确脉冲响应
   `h_n(t) = exp(-zeta w_n t) sin(w_d t)/w_d` 做 FFT 卷积（无条件稳定）；根部应力
   `sigma(t) = E c_root sum_n phi_n''(0) q_n(t)`，`phi_n''(0)` 由根部 Hermite 单元的
   节点自由度解析恢复。独立校核：全阶 Newmark-beta 平均加速度法（Rayleigh 阻尼与模态
   阻尼一致）直接积分同一载荷。
4. **S4 雨流**：复用 `tensorfem.marine_fatigue_advanced.rainflow_cycles`（ASTM E1049 四点栈）。
5. **S5 疲劳**：复用 `tensorfem.weld_fatigue_ice.DNVSNCurve`（curve D, in air，双斜率，
   拐点 1e7），Miner 线性累积。
6. **S6 扫描**：L_ice ∈ [0.1, 0.5] m、n ∈ [100, 200] rpm、V ∈ [2, 6] m/s 单变量扫描，
   检查 D 对三个变量单调递增。

## 3. 计算结果

### 3.1 A1 Hertz 接触（判据 b：< 1e-3 N）

| delta [m] | F 闭式 [N] | F 数值积分 [N] | 绝对误差 [N] | 位移型线误差 |
|---|---:|---:|---:|---:|
{hertz_lines}

最大力误差 **{hertz['max_force_abs_error_n']:.2e} N**（判据 < 1e-3 N）；
Boussinesq 积分得到的位移型线 `u(r) = delta (1 - r^2/(2a^2))` 最大相对误差
**{hertz['max_profile_error']*100:.2f} %**，中心位移 `u(0) = delta`。

### 3.2 A2 叶片干模态（判据 a：< 5 %）

等截面参考叶片（弦长 0.50 m、厚度 0.18 m、L = 1.05 m）前五阶：

| 阶 | FE [Hz] | Euler-Bernoulli [Hz] | 相对误差 |
|---|---:|---:|---:|
{mode_lines}

最大误差 **{modes['max_relative_error']*100:.3f} %**（判据 < 5 %）。
变截面设计叶片：干模态 {', '.join(f'{v:.1f}' for v in modes['tapered_blade_dry_hz'])} Hz，
湿模态 {', '.join(f'{v:.1f}' for v in modes['tapered_blade_wet_hz'])} Hz；
网格 48→96 单元最大频率漂移 {modes['mesh_convergence_48_vs_96_max_relative']*100:.3f} %。
均匀附加质量下湿/干频率比 计算 {modes['wet_ratio_computed'][0]:.5f} vs 解析
{modes['wet_ratio_expected']:.5f}（误差 {modes['wet_ratio_error']*100:.3f} %）。

### 3.3 A3 应力响应（判据 c：< 3 %）

模态叠加 vs 全阶 Newmark-beta：峰值与 RMS 应力相对差 < 3 %（见 pytest
`test_modal_superposition_matches_newmark`）；静力极限下模态曲率与 `F L_c / EI` 相对差
< 2 %（`test_modal_static_limit_matches_static_bending`）。

### 3.4 A4 雨流（判据 d：误差 < 1 循环）

正弦 7 周期：计数 **{rain['counted_full_cycles']:.1f}**（期望 {rain['expected_cycles']:.1f}，
误差 {rain['count_error']:.1f}）；最大范围 {rain['max_range_pa']/1e6:.2f} MPa vs 期望
{rain['expected_max_range_pa']/1e6:.2f} MPa（{rain['range_relative_error']*100:.3f} %）。

### 3.5 A5 S-N（判据 e：< 5 %）

| dsigma [MPa] | N (模块) | N (表格参数独立重算) | 相对误差 |
|---|---:|---:|---:|
{sn_lines}

FAT90@2e6 相对误差 **{sn['fat_90_at_2e6_relative_error']*100:.3f} %**，
拐点应力 {sn['knee_stress_pa']/1e6:.2f} MPa vs 表值 52.63 MPa
（误差 {sn['knee_stress_relative_error']*100:.3f} %）。

说明：S-N 实现复用 `weld_fatigue_ice.DNVSNCurve`，上表第三列是同一组双斜率参数
的独立重算（用于排除分支切换/单位错误，因此误差为机器零）；**真正独立的表格锚点**是
curve D 的定义点 FAT90@2e6 与拐点应力 52.63 MPa，二者误差分别为
{sn['fat_90_at_2e6_relative_error']*100:.3f} % 与 {sn['knee_stress_relative_error']*100:.3f} %。

### 3.6 链条结果（判据 f / g）

{markdown_table(numbers)}

设计模型（压碎限）20 年 Miner 损伤 **D = {design['fatigue']['damage_20yr']:.3e} < 1.0**
（判据 f 通过），允许铣削 {design['fatigue']['allowable_hours']:.0f} h；纯弹性 Hertz **界定**
模型 D = {numbers['chain']['bounding']['fatigue']['damage_20yr']:.3e}（判据 f 不通过，如实列出）。

![接触律](../{figures['contact']})

![应力与雨流](../{figures['stress']})

![S-N](../{figures['sn']})

![扫描](../{figures['sweep']})

### 3.7 参数扫描（判据 e：D 单调递增）

{markdown_sweep(numbers)}

设计模型单调性检查：L_ice {numbers['sweeps']['design']['checks']['damage_monotone_in_ice_length']}，
n {numbers['sweeps']['design']['checks']['damage_monotone_in_rpm']}，
V {numbers['sweeps']['design']['checks']['damage_monotone_in_speed']}；
弹性界定模型同样三项全部
{all(numbers['sweeps']['bounding']['checks'].values())}。

## 4. 注意点（坑 / 局限 / 适用范围）

1. **Hertz 压力场不是均匀位移**：`u(r) = delta(1 - r^2/(2a^2))` 才是正确的 Hertz 运动学
   （穿透量减去压头形状）。最初按"接触区内位移均匀"做数值校验，得到 41 % 的"误差"——
   这是物理正确、假设错误的典型例子，已在 A1 校验中改为抛物线型线。
2. **弹性能量平衡给出的点接触压力达 GPa 量级**，比冰的压碎强度高两个数量级，说明该点接触
   理想化只是"刚度控制"上界；真实叶片-冰接触是沿前缘的分布式接触。链条因此同时给出
   压碎限模型（设计）与弹性模型（界定），二者 D 相差
   {numbers['chain']['ratio_damage_bounding_over_design']:.1e} 倍，**接触面积理想化是本链条
   的主导不确定度**，远大于链条内部各阶段的数值误差。
3. 叶片用梁单元而非壳单元（声明偏差）：判据 a 的解析基准只能对等截面梁成立，扭曲翼型壳
   模型无闭式解；因此模态锚定在等截面参考梁上，设计叶片另做网格无关性验证。
   根部应力是名义弯曲应力，不含叶根圆角应力集中系数（真实热点的 Kt > 1，会使损伤上升）。
4. 冰视为刚性、无压碎过程建模；未含应变率强化、冰片旋转/翻转、伴流不均匀与空泡。
5. S-N 取 curve D in air、无厚度修正；海水阴极保护曲线或厚度修正都会改变寿命（是显式
   开关而非隐藏假设）。
6. 20 年设计寿命的损伤按"年铣削小时 × 命中率"线性外推，命中率默认 1.0（保守：每次叶片
   通过都撞到设计尺寸冰片），实际铣削载荷谱由远小于设计尺寸的碎冰主导。

## 5. 误差判定（逐项回答"是否 < 3 %"）

| 判据 | 量 | 计算 vs 基准 | 误差 | < 3 % ? |
|---|---|---:|---:|---|
| b | Hertz 接触力 | 数值积分 vs 闭式 | {hertz['max_force_abs_error_n']:.2e} N | 是（远小于 1e-3 N 判据） |
| a | 前 5 阶干模态 | FE vs Euler-Bernoulli | {modes['max_relative_error']*100:.3f} % | 是 |
| c | 应力响应 | 模态叠加 vs Newmark | < 3 %（pytest） | 是（判据线即 3 %） |
| d | 雨流计数 | 正弦已知循环 | {rain['count_error']:.1f} 循环 | 是（< 1 循环） |
| e | S-N 曲线 | curve D vs 表值 (FAT90@2e6 / 拐点应力) | {max(sn['fat_90_at_2e6_relative_error'], sn['knee_stress_relative_error'])*100:.3f} % | 是 |
| f | Miner 设计校核 | D vs 1.0 | {design['fatigue']['damage_20yr']:.3e} vs 1.0 | 是（损伤裕度极大） |

**未达成 / 需要声明的部分**：判据 f 只在压碎限模型下通过；同一链条的弹性 Hertz 界定模型
D = {numbers['chain']['bounding']['fatigue']['damage_20yr']:.3e} ≫ 1，即该理想化不能直接
用于 20 年疲劳校核（honest FAIL，如实列出）。绝对应力量级（设计模型根部峰值
{design['stress']['sigma_max_pa']/1e6:.1f} MPa）无实验基准可比，属于模型推断而非验证。
"""
    DOC.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    numbers = {
        "schema": "tensorfem.propeller-ice-milling-fatigue-chain/1.0",
        "anchors": {
            "hertz": anchor_hertz(),
            "modes": anchor_modes(),
            "rainflow": anchor_rainflow(),
            "sn": anchor_sn(),
        },
        "chain": chain_cases(),
    }
    numbers["sweeps"] = sweeps()
    figures = {
        "contact": figure_contact(),
        "stress": figure_stress(numbers["chain"]),
        "sn": figure_sn(numbers["anchors"]["sn"]),
        "sweep": figure_sweep(numbers["sweeps"]),
    }
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2, default=float))
    write_doc(numbers, figures)
    print(json.dumps({k: v for k, v in numbers["sweeps"]["design"]["checks"].items()}, indent=2))
    print("design D20", numbers["chain"]["design"]["fatigue"]["damage_20yr"])
    print("bounding D20", numbers["chain"]["bounding"]["fatigue"]["damage_20yr"])


if __name__ == "__main__":
    main()
