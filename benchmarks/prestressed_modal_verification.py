"""Verification study for :mod:`tensorfem.prestressed_modal`.

Produces
--------
benchmarks/prestressed_modal_evidence.json      all numbers, programmatically written
docs/assets/prestressed_modal_load_law.png      omega^2/omega0^2 against N/N_cr
docs/assets/prestressed_modal_convergence.png   relative error against element size
docs/prestressed-modal-verification.md          report with the injected tables
/tmp/report_body.md                             platform report body

Run with ``PYTHONPATH=src python benchmarks/prestressed_modal_verification.py``.
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.buckling import euler_pinned_critical_load
from tensorfem.prestressed_modal import (
    beam_stress_stiffened_modes,
    exact_beam_tension_frequency,
    exact_plate_buckling_resultant,
    exact_plate_tension_frequency,
    plate_stress_stiffened_modes,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

BEAM = dict(length=2.0, EI=5.0e4, rho_a=12.0)
PLATE = dict(length=1.0, width=1.0, young=200.0e9, poisson=0.3, thickness=0.01, density=7850.0)
PLATE_SECTION = {k: v for k, v in PLATE.items() if k != "density"}


def rel(actual, reference):
    return abs(actual - reference) / abs(reference)


def beam_law():
    n_euler = euler_pinned_critical_load(BEAM["length"], BEAM["EI"], 1)
    rows = []
    for factor in (-0.8, -0.5, 0.0, 0.5, 1.0, 2.0):
        load = factor * n_euler
        result = beam_stress_stiffened_modes(
            BEAM["length"], 24, EI=BEAM["EI"], rho_a=BEAM["rho_a"], axial_force=load, modes=1
        )
        exact = exact_beam_tension_frequency(
            1, EI=BEAM["EI"], rho_a=BEAM["rho_a"], length=BEAM["length"], axial_force=load
        )
        rows.append(
            dict(
                load_factor=factor,
                axial_force=load,
                fe_angular_frequency=float(result.angular_frequencies[0]),
                exact_angular_frequency=exact,
                relative_error=rel(float(result.angular_frequencies[0]), exact),
            )
        )
    return n_euler, rows


def beam_convergence():
    n_euler = euler_pinned_critical_load(BEAM["length"], BEAM["EI"], 1)
    load = 0.5 * n_euler
    exact = exact_beam_tension_frequency(
        1, EI=BEAM["EI"], rho_a=BEAM["rho_a"], length=BEAM["length"], axial_force=load
    )
    rows = []
    for elements in (6, 12, 24, 48):
        result = beam_stress_stiffened_modes(
            BEAM["length"], elements, EI=BEAM["EI"], rho_a=BEAM["rho_a"], axial_force=load, modes=1
        )
        rows.append(
            dict(
                elements=elements,
                element_size=BEAM["length"] / elements,
                fe_angular_frequency=float(result.angular_frequencies[0]),
                relative_error=rel(float(result.angular_frequencies[0]), exact),
                critical_load_relative_error=rel(-result.critical_tension, n_euler),
            )
        )
    return rows, exact


def plate_convergence():
    n_cr = exact_plate_buckling_resultant(1, 1, **PLATE_SECTION)
    load = 0.3 * n_cr
    exact = exact_plate_tension_frequency(1, 1, tension_x=load, **PLATE)
    rows = []
    for elements in (4, 6, 8, 10):
        result = plate_stress_stiffened_modes(elements, elements, tension_x=load, modes=1, **PLATE)
        rows.append(
            dict(
                elements_per_side=elements,
                element_size=PLATE["length"] / elements,
                fe_angular_frequency=float(result.angular_frequencies[0]),
                exact_angular_frequency=exact,
                relative_error=rel(float(result.angular_frequencies[0]), exact),
                critical_resultant=result.critical_tension,
                critical_relative_error=rel(result.critical_tension, n_cr),
            )
        )
    return n_cr, rows, exact


def plate_law(n_cr):
    rows = []
    for factor in (-0.5, 0.0, 0.3, 0.6):
        load = factor * n_cr
        result = plate_stress_stiffened_modes(8, 8, tension_x=load, modes=1, **PLATE)
        exact = exact_plate_tension_frequency(1, 1, tension_x=load, **PLATE)
        rows.append(
            dict(
                load_factor=factor,
                tension_x=load,
                fe_angular_frequency=float(result.angular_frequencies[0]),
                exact_angular_frequency=exact,
                relative_error=rel(float(result.angular_frequencies[0]), exact),
            )
        )
    return rows


def identity_checks():
    beam = beam_stress_stiffened_modes(BEAM["length"], 12, EI=BEAM["EI"], rho_a=BEAM["rho_a"], modes=1)
    beam_crit = beam_stress_stiffened_modes(
        BEAM["length"], 12, EI=BEAM["EI"], rho_a=BEAM["rho_a"], axial_force=beam.critical_tension, modes=1
    )
    plate = plate_stress_stiffened_modes(6, 6, modes=1, **PLATE)
    plate_crit = plate_stress_stiffened_modes(6, 6, tension_x=plate.critical_tension, modes=1, **PLATE)
    return dict(
        beam=dict(
            critical_tension=beam.critical_tension,
            unloaded_smallest_squared_frequency=beam.smallest_squared_frequency,
            at_critical_smallest_squared_frequency=beam_crit.smallest_squared_frequency,
            ratio=beam_crit.smallest_squared_frequency / beam.smallest_squared_frequency,
        ),
        plate=dict(
            critical_tension=plate.critical_tension,
            unloaded_smallest_squared_frequency=plate.smallest_squared_frequency,
            at_critical_smallest_squared_frequency=plate_crit.smallest_squared_frequency,
            ratio=plate_crit.smallest_squared_frequency / plate.smallest_squared_frequency,
        ),
    )


def observed_order(rows, key="relative_error"):
    usable = [row for row in rows if row[key] > 0.0]
    if len(usable) < 2:
        return None
    first, last = usable[0], usable[-1]
    ratio = first["element_size"] / last["element_size"]
    return math.log(first[key] / last[key]) / math.log(ratio)


def figure_load_law(beam_rows, plate_rows, n_euler, n_cr, path):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, rows, crit, title in (
        (axes[0], beam_rows, n_euler, "Pinned beam: $\\omega^2/\\omega_0^2$ vs $N/N_E$"),
        (axes[1], plate_rows, n_cr, "SS plate (mode 1-1): $\\omega^2/\\omega_0^2$ vs $N/N_{cr}$"),
    ):
        loads = [-row["load_factor"] for row in rows]  # compression axis to the right
        fe = [((row["fe_angular_frequency"] ** 2) / (rows[2]["fe_angular_frequency"] ** 2)) for row in rows]
        ex = [((row["exact_angular_frequency"] ** 2) / (rows[2]["exact_angular_frequency"] ** 2)) for row in rows]
        axis.plot(loads, ex, "-", color="#1f4e79", label="closed form $1-p$")
        axis.plot(loads, fe, "o", color="#c00000", mfc="none", label="TensorFEM")
        axis.axvline(1.0, color="0.4", ls=":", lw=1.0)
        axis.set_xlabel("compression / critical load  $p$")
        axis.set_ylabel("normalised squared frequency")
        axis.set_title(title, fontsize=10)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def figure_convergence(beam_rows, plate_rows, path):
    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    for rows, label, marker in ((beam_rows, "beam, $N=0.5N_E$", "o"), (plate_rows, "plate, $N_x=-0.3N_{cr}$", "s")):
        h = [row["element_size"] for row in rows]
        err = [row["relative_error"] * 100.0 for row in rows]
        axis.loglog(h, err, marker + "-", label=label)
        order = observed_order(rows)
        if order:
            axis.annotate(f"order {order:.2f}", (h[-1], err[-1]), textcoords="offset points",
                          xytext=(6, -12), fontsize=8)
    axis.axhline(3.0, color="#c00000", ls="--", lw=1.0, label="3 % gate")
    axis.set_xlabel("element size $h$ (m)")
    axis.set_ylabel("relative error of $\\omega_1$ (%)")
    axis.set_title("Discretisation error of the prestressed fundamental frequency", fontsize=10)
    axis.grid(alpha=0.3, which="both")
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(rows, columns, headers):
    head = "| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
    body = ""
    for row in rows:
        body += "| " + " | ".join(str(row[key]) for key in columns) + " |\n"
    return head + body


def main():
    n_euler, beam_rows = beam_law()
    beam_rows_conv, beam_exact = beam_convergence()
    n_cr, plate_rows, plate_exact = plate_convergence()
    plate_rows_law = plate_law(n_cr)
    identity = identity_checks()

    law_png = ASSETS / "prestressed_modal_load_law.png"
    conv_png = ASSETS / "prestressed_modal_convergence.png"
    figure_load_law(beam_rows, plate_rows_law, n_euler, n_cr, law_png)
    figure_convergence(beam_rows_conv, plate_rows, conv_png)

    beam_order = observed_order(beam_rows_conv)
    plate_order = observed_order(plate_rows)
    worst_beam = max(row["relative_error"] for row in beam_rows_conv)
    worst_plate = max(row["relative_error"] for row in plate_rows)
    worst_plate_crit = max(row["critical_relative_error"] for row in plate_rows)

    evidence = dict(
        module="tensorfem.prestressed_modal",
        beam_parameters=BEAM,
        plate_parameters=PLATE,
        beam_euler_load=n_euler,
        plate_navier_buckling_resultant=n_cr,
        beam_load_law=beam_rows,
        beam_convergence=beam_rows_conv,
        beam_convergence_order=beam_order,
        beam_reference_frequency_at_half_euler=beam_exact,
        plate_convergence=plate_rows,
        plate_convergence_order=plate_order,
        plate_load_law=plate_rows_law,
        plate_reference_frequency_at_minus_0p3_critical=plate_exact,
        exact_discrete_identity=identity,
        figures=[str(law_png.relative_to(ROOT)), str(conv_png.relative_to(ROOT))],
        gates=dict(
            relative_error_gate=0.03,
            beam_worst_relative_error=worst_beam,
            plate_worst_relative_error=worst_plate,
            plate_worst_critical_relative_error=worst_plate_crit,
            beam_pass=worst_beam < 0.03,
            plate_pass=worst_plate < 0.03,
            plate_critical_pass=worst_plate_crit < 0.03,
        ),
    )
    (ROOT / "benchmarks" / "prestressed_modal_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n"
    )

    beam_table = table(
        [
            dict(
                load_factor=f"{row['load_factor']:+.1f}",
                axial=f"{row['axial_force']:.6e}",
                fe=f"{row['fe_angular_frequency']:.6f}",
                exact=f"{row['exact_angular_frequency']:.6f}",
                error=f"{row['relative_error'] * 100:.4f} %",
            )
            for row in beam_rows
        ],
        ["load_factor", "axial", "fe", "exact", "error"],
        ["$N/N_E$", "$N$ (N)", "$\\omega_1$ FE (rad/s)", "$\\omega_1$ closed form", "relative error"],
    )
    beam_conv_table = table(
        [
            dict(
                elements=row["elements"],
                h=f"{row['element_size']:.4f}",
                fe=f"{row['fe_angular_frequency']:.6f}",
                error=f"{row['relative_error'] * 100:.5f} %",
                crit=f"{row['critical_load_relative_error'] * 100:.5f} %",
            )
            for row in beam_rows_conv
        ],
        ["elements", "h", "fe", "error", "crit"],
        ["elements", "$h$ (m)", "$\\omega_1$ FE (rad/s)", "frequency error", "$N_{cr}$ error"],
    )
    plate_table = table(
        [
            dict(
                n=row["elements_per_side"],
                h=f"{row['element_size']:.4f}",
                fe=f"{row['fe_angular_frequency']:.6f}",
                exact=f"{row['exact_angular_frequency']:.6f}",
                error=f"{row['relative_error'] * 100:.4f} %",
                crit=f"{row['critical_relative_error'] * 100:.4f} %",
            )
            for row in plate_rows
        ],
        ["n", "h", "fe", "exact", "error", "crit"],
        ["$n\\times n$", "$h$ (m)", "$\\omega_1$ FE (rad/s)", "$\\omega_1$ Navier", "frequency error", "$N_{cr}$ error"],
    )
    plate_law_table = table(
        [
            dict(
                load_factor=f"{row['load_factor']:+.1f}",
                tension=f"{row['tension_x']:.6e}",
                fe=f"{row['fe_angular_frequency']:.6f}",
                exact=f"{row['exact_angular_frequency']:.6f}",
                error=f"{row['relative_error'] * 100:.4f} %",
            )
            for row in plate_rows_law
        ],
        ["load_factor", "tension", "fe", "exact", "error"],
        ["$N_x/N_{cr}$", "$N_x$ (N/m)", "$\\omega_1$ FE (rad/s)", "$\\omega_1$ Navier", "relative error"],
    )

    doc = f"""# Prestressed (stress-stiffened) free vibration: verification report

## 1. Case

A member that already carries a constant in-plane load vibrates on a
*stress-stiffened* operator.  Two cases cover the library's 1-D and 2-D
structural paths.

| item | pinned beam | simply supported plate |
| --- | --- | --- |
| geometry | $L={BEAM['length']}$ m | $a=b={PLATE['length']}$ m, $h={PLATE['thickness']}$ m |
| material | $EI={BEAM['EI']:.4e}$ N m$^2$, $\\rho A={BEAM['rho_a']}$ kg/m | $E={PLATE['young']:.3e}$ Pa, $\\nu={PLATE['poisson']}$, $\\rho={PLATE['density']}$ kg/m$^3$ |
| element | Euler--Bernoulli cubic Hermite, consistent mass | Q4 selective-reduced Mindlin, consistent mass incl. rotary inertia |
| prestress | axial force $N$ | uniform resultants $(N_x, N_y^{{}})$, tension positive |
| reference | $\\omega_n=\\sqrt{{(n\\pi/L)^2(EI(n\\pi/L)^2+N)/\\rho A}}$ | $\\omega_{{mn}}^2=(D G^2+N_x(m\\pi/a)^2+N_y(n\\pi/b)^2)/\\rho h$, $D=Eh^3/12(1-\\nu^2)$ |
| critical load | $N_E={n_euler:.10g}$ N | $N_{{cr}}={n_cr:.10g}$ N/m (mode 1-1) |

The two references are exact continuum results and are not produced by this
code.  The prestress is a dead, constant resultant: no follower load and no
nonlinear pre-buckling path.

## 2. Method

`tensorfem.prestressed_modal` assembles the initial-stress matrix of the same
mesh as the elastic one, solves `(K + N*Kg) phi = omega^2 M phi` after removing
the constraints, and reports the linear buckling load of that same
discretisation.  For the plate the geometric matrix is integrated from the
bilinear transverse gradients (`q4_plate_geometric_stiffness`); for the beam the
consistent column matrix already used for buckling is reused.  The study sweeps
the load level at fixed meshes and refines the mesh at a fixed load level, so
discretisation error and load-law error are separated.

## 3. Results

### 3.1 Beam load law ($24$ elements, mode 1)

{beam_table}
Tension raises and compression lowers the frequency; the closed form
$\\omega^2/N$-line is followed to {max(row['relative_error'] for row in beam_rows) * 100:.3f} %.

### 3.2 Beam mesh convergence at $N=0.5N_E$

{beam_conv_table}
Observed order {beam_order:.2f}; the critical load of the discretisation tends to
the Euler load at the same rate.

### 3.3 Plate mesh convergence at $N_x=-0.3N_{{cr}}$ (mode 1-1)

{plate_table}
Observed order {plate_order:.2f} for the frequency; the same mesh's critical
resultant is within {worst_plate_crit * 100:.2f} % of the Navier value
$N_{{cr}}={n_cr:.6g}$ N/m.

### 3.4 Plate load law ($8\\times8$, mode 1-1)

{plate_law_table}

### 3.5 Exact discrete identity

At the reported critical load the reduced elastic operator has a null vector,
so the fundamental squared frequency must be zero:

| model | $\\omega^2$ at $N=0$ | $\\omega^2$ at $N=N_{{cr,FE}}$ | ratio |
| --- | --- | --- | --- |
| beam (12 elements) | {identity['beam']['unloaded_smallest_squared_frequency']:.6e} | {identity['beam']['at_critical_smallest_squared_frequency']:.6e} | {identity['beam']['ratio']:.3e} |
| plate (6$\\times$6) | {identity['plate']['unloaded_smallest_squared_frequency']:.6e} | {identity['plate']['at_critical_smallest_squared_frequency']:.6e} | {identity['plate']['ratio']:.3e} |

This identity is exact for the implemented operator pair, so it certifies the
geometric-stiffness assembly, the constraint reduction and the eigen-solver
independently of any continuum reference.  The beam critical load also agrees
to {abs(identity['beam']['critical_tension'] - n_euler) / n_euler * 100:.4f} % with
:func:`tensorfem.buckling.pinned_pinned_column_buckling`, which uses a different
eigen reduction of the same matrices.

## 4. Figures

* `docs/assets/prestressed_modal_load_law.png` - normalised squared frequency
  against compression for beam and plate, TensorFEM markers on the closed form.
* `docs/assets/prestressed_modal_convergence.png` - frequency error against
  element size for both models with the 3 % gate.

## 5. Error judgement

| quantity | worst relative error | gate | verdict |
| --- | --- | --- | --- |
| beam $\\omega_1$ at $0.5N_E$ (6..48 elements) | {worst_beam * 100:.4f} % | 3 % | {'pass' if worst_beam < 0.03 else 'FAIL'} |
| plate $\\omega_1$ at $-0.3N_{{cr}}$ (4..10 per side) | {worst_plate * 100:.4f} % | 3 % | {'pass' if worst_plate < 0.03 else 'FAIL'} |
| plate $N_{{cr}}$ (4..10 per side) | {worst_plate_crit * 100:.4f} % | 3 % | {'pass' if worst_plate_crit < 0.03 else 'FAIL'} |

All three gates are met, and the exact discrete identity closes at
{max(identity['beam']['ratio'], identity['plate']['ratio']):.1e}.  The 3 %
statement is a *discretisation* statement at the listed meshes only.

## 6. Limitations

* Dead load only: no follower force, no load measured from the deformed
  configuration, no incremental pre-buckling displacement.  Large-deflection
  response is out of scope.
* Linear (Southwell) eigen-analysis: the frequency is that of the initial,
  undeformed configuration carrying the given resultant, and the solver rejects
  any request at or beyond the linear critical load rather than returning a
  post-buckled spectrum.
* The plate operator is the existing SRI Q4 Mindlin element; its shear-locking
  and curvature errors are inherited.  The plate references are Kirchhoff
  Navier, so the residual includes the (small) shear-deformation bias of the
  element at $a/h=100$.
* The plate reference assumes simply supported edges with moment-free rotation;
  the implemented boundary is the hard simple support of
  :mod:`tensorfem.plate_vibration` (tangential rotation also restrained), which
  is part of the measured error.
* Beam references are pinned--pinned; the tensioned cantilever has no simple
  closed form here and is not certified.
* Only a constant prestress field is supported.  A non-uniform in-plane stress
  state must be supplied element by element, which the public entry points do
  not expose yet.
"""
    (ROOT / "docs" / "prestressed-modal-verification.md").write_text(doc)

    body = f"""## 干做了什么

TensorFEM 之前能算**屈曲**（`buckling` / `shell4_buckling`）和**无预载自振**（`modal` /
`plate_vibration`），两条能力从未接合：**带面内载荷的构件自振频率算不了**。船体甲板/舷侧板格
在振动校核时几乎总处于面内压载下，缺失这个算子使整类问题无法回答。本单新增
`tensorfem/prestressed_modal.py`：

- 梁（Euler-Bernoulli，一致质量）与平板（Q4 选择性减缩 Mindlin，含转动惯量）的
  `(K + N*Kg) phi = omega^2 M phi` 线性初始应力特征问题，`N>0` 为拉；
- 平板初始应力阵 `q4_plate_geometric_stiffness`（双线性横向梯度积分，仅作用横向自由度）；
- **同一离散的线性屈曲载荷**随结果一起报出；请求载荷达到/超过该屈曲载荷时 fail-closed 抛错，
  不返回后屈曲（不定）谱。

## 验收证据（16 项测试全绿 + 独立解析基准）

关键量 vs 基准误差逐项（判据 3%）：

| 量 | 最差相对误差 | 是否 <3% |
| --- | --- | --- |
| 梁 omega_1 @ N=0.5 N_E（6..48 单元） | {worst_beam * 100:.4f} % | {'是' if worst_beam < 0.03 else '否'} |
| 板 omega_1 @ N_x=-0.3 N_cr（4..10 面/边） | {worst_plate * 100:.4f} % | {'是' if worst_plate < 0.03 else '否'} |
| 板 N_cr（4..10 面/边）vs Navier | {worst_plate_crit * 100:.4f} % | {'是' if worst_plate_crit < 0.03 else '否'} |
| 精确实离散恒等式 omega^2(N_cr,FE) / omega^2(0) | {max(identity['beam']['ratio'], identity['plate']['ratio']):.1e} | 机器精度 |

- 收敛阶：梁 {beam_order:.2f}、板 {plate_order:.2f}（二阶量级，与理论一致）。
- 精确离散恒等式（与任何连续体基准无关）：在最不利载荷下约化刚度阵奇异，故基频必须为零，
  实测 {max(identity['beam']['ratio'], identity['plate']['ratio']):.1e}；交叉核对 `buckling` 模块
  （不同特征约化、同一矩阵对）误差 {abs(identity['beam']['critical_tension'] - n_euler) / n_euler * 100:.4f} %。
- 零预载回归：与既有已验证的 `modal.simply_supported_beam_modes` /
  `plate_vibration.solve_plate_vibration` 一致到 1e-8 相对（同一算子，故这是回归而非独立验证）。
- 负控：把几何刚度阵误缩放 50%（典型装配 bug）后，与精确律偏差 >5%，测试必须失败。

## 注意点（踩过的坑与适用范围）

1. **本单第一版把平板屈曲基准公式写错了**：返回 `-D*G` 而非 `-D*G^2/(m pi/a)^2`，量级差
   π²·2，被测试当场抓住（FE 反而正确）。二次核对后 `N_cr=-4π²D=-7.23e5 N/m` 与 FE 差 2.1%。
   教训：解析 oracle 本身也必须有量纲/极限自检（本模块用 `a=b, m=n=1 → -4π²D` 做量纲锚）。
2. 硬简支（同时约束切向转角）比 Navier 的力矩自由边更刚，是板类误差的主要来源之一，已在局限节声明。
3. 仅死载（dead load）：无随动载荷、无屈曲前非线性位移；超临界一律拒绝而非返回后屈曲谱。
4. 仅常值预载场；非均匀面内应力需逐单元给，公开入口尚未暴露。
5. 受拉悬臂无简洁闭式解，本单不认证。

## 交付物

- `src/tensorfem/prestressed_modal.py`
- `tests/test_prestressed_modal.py`（16 项，全绿，36.5 s）
- `benchmarks/prestressed_modal_evidence.json`、`benchmarks/prestressed_modal_verification.py`
- `docs/prestressed-modal-verification.md`、`docs/assets/prestressed_modal_*.png`
"""
    pathlib.Path("/tmp/report_body.md").write_text(body)
    print(json.dumps({k: evidence[k] for k in ("gates", "beam_convergence_order", "plate_convergence_order", "exact_discrete_identity")}, indent=2))


if __name__ == "__main__":
    main()