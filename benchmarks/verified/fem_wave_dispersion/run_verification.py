"""Verification anchor: Rayleigh--Lamb dispersion exact oracle vs Bloch--Floquet FEM.

Run:  PYTHONPATH=src python benchmarks/verified/fem_wave_dispersion/run_verification.py

Writes (all numbers in the markdown come from this script -- zero hand-copying):
  result.json, gates_verified.json, dispersion.png, convergence.png,
  docs/lamb-wave-dispersion-verification.md
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tensorfem.continuum import rectangular_q4_mesh
from tensorfem.wave_dispersion import (
    bloch_cell_frequencies,
    bloch_cell_matrices,
    bloch_cell_modes,
    midplane_family_fraction,
    reduced_dof_count,
    flexural_phase_velocity,
    lamb_residual,
    longitudinal_speed,
    plate_speed,
    rayleigh_lamb_frequencies,
    rayleigh_speed,
    shear_speed,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DOC = REPO / "docs" / "lamb-wave-dispersion-verification.md"

E, NU, RHO = 210.0e9, 0.3, 7850.0          # steel
H = 0.01                                    # half thickness -> plate 20 mm
C_L = longitudinal_speed(E, NU, RHO)
C_T = shear_speed(E, NU, RHO)
C_P = plate_speed(E, NU, RHO)
C_R = rayleigh_speed(E, NU, RHO)

KH_ANALYTIC = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
KH_FEM = (0.2, 0.5, 1.0)
MESHES = ((8, 4), (16, 8), (32, 16))


def analytic_table() -> list[dict]:
    rows = []
    for kh in KH_ANALYTIC:
        k = kh / H
        roots = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=3)
        a0 = roots["antisymmetric"][0]
        s0 = roots["symmetric"][0]
        rows.append(dict(
            kh=kh,
            a0_phase=a0 / k / C_T,
            s0_phase=s0 / k / C_T,
            a0_omega=a0,
            s0_omega=s0,
            flexural_phase=flexural_phase_velocity(k, E, NU, RHO, H) / C_T,
            a0_residual=abs(lamb_residual(a0, k, C_L, C_T, H, "antisymmetric")),
            s0_residual=abs(lamb_residual(s0, k, C_L, C_T, H, "symmetric")),
        ))
    return rows


def extrapolate_plate_speed(rows: list[dict]) -> float:
    """Fit c(kh) = cp (1 + a kh^2) through the three smallest kh points."""
    xs = np.array([r["kh"] ** 2 for r in rows[:3]])
    ys = np.array([r["s0_phase"] for r in rows[:3]])
    a = np.vstack([np.ones_like(xs), xs]).T
    coef, *_ = np.linalg.lstsq(a, ys, rcond=None)
    return float(coef[0])


def fem_table() -> list[dict]:
    """Bloch--Floquet cell vs the exact branches, with oracle-free branch picking.

    The cell's eigenmodes are ordered by frequency, and the order of the branches
    changes with ``kh`` (A0 is mode 0 everywhere, but S0 is mode 1 at kh>=0.5 and
    mode 2 at kh=0.2 because an A1 branch slips in between).  Selecting ``omega[0]``
    / ``omega[1]`` positionally therefore compares the wrong branch, so each mode is
    classified by its mid-plane parity instead: ``S`` modes are symmetric in
    ``u_x`` and antisymmetric in ``u_z``, ``A`` modes the other way round.  That
    classifier reads the mode shape only -- the exact roots are used for scoring,
    never for choosing which mode to score.
    """
    rows = []
    for kh in KH_FEM:
        k = kh / H
        roots = rayleigh_lamb_frequencies(k, H, E, NU, RHO, n_roots=1)
        exact_a0 = roots["antisymmetric"][0] / k / C_T
        exact_s0 = roots["symmetric"][0] / k / C_T
        for nx, ny in MESHES:
            cell_length = 2.0 * math.pi / (k * 4.0)
            nodes, _ = rectangular_q4_mesh(cell_length, 2.0 * H, nx, ny)
            omega, modes = bloch_cell_modes(k, half_thickness=H, young=E, poisson=NU,
                                            density=RHO, nx=nx, ny=ny, n_modes=8)
            fractions = [midplane_family_fraction(nodes, modes[:, i])
                         for i in range(len(omega))]
            a_index = next(i for i, fr in enumerate(fractions) if fr < 0.5)
            s_index = next(i for i, fr in enumerate(fractions) if fr >= 0.5)
            fem_a0 = omega[a_index] / k / C_T
            fem_s0 = omega[s_index] / k / C_T
            rows.append(dict(kh=kh, nx=nx, ny=ny,
                             a0_index=int(a_index), s0_index=int(s_index),
                             a0_symmetry=float(fractions[a_index]),
                             s0_symmetry=float(fractions[s_index]),
                             a0_fem=fem_a0, s0_fem=fem_s0,
                             a0_exact=exact_a0, s0_exact=exact_s0,
                             a0_error=abs(fem_a0 / exact_a0 - 1.0),
                             s0_error=abs(fem_s0 / exact_s0 - 1.0)))
    return rows


def assembly_checks() -> dict:
    """Gamma-point (k=0) structural self-checks of the periodic cell.

    With the Floquet condition the cell owns exactly **two** zero-energy modes, the
    two rigid translations; the rigid rotation is not compatible with the periodic
    identification and the A0/S0 acoustic branches degenerate onto those same two
    translations at ``k=0``.  The reduced dof count is structural: two unknowns per
    retained node, i.e. ``2 (n_nodes - n_right_column)``.
    """
    k = 0.0
    cell = bloch_cell_matrices(k, half_thickness=H, young=E, poisson=NU, density=RHO,
                               nx=8, ny=4)
    omega = bloch_cell_frequencies(k, half_thickness=H, young=E, poisson=NU, density=RHO,
                                   nx=8, ny=4, n_modes=8)
    area = cell.cell_length * 2.0 * H
    # The consistent mass matrix carries one rho*int(Ni Nj) block per translation
    # direction, so its full sum is 2*rho*area for a two-dimensional element.
    expected = 2.0 * RHO * area
    coords = np.asarray(cell.nodes)
    n_right = int(np.sum(np.isclose(coords[:, 0], cell.cell_length)))
    zero_tol = 1e-10 * float(omega[-1])
    return dict(
        total_mass=cell.total_mass,
        expected_mass=expected,
        mass_relative_error=abs(cell.total_mass / expected - 1.0),
        zero_mode_count=int(np.sum(omega <= zero_tol)),
        zero_mode_ratio=float(omega[1] / omega[2]),
        gamma_omega=[float(v) for v in omega],
        n_nodes=len(coords),
        n_right_column=n_right,
        reduced_dofs=reduced_dof_count(cell),
        expected_reduced_dofs=2 * (len(coords) - n_right),
    )


def convergence_order(rows: list[dict], kh: float, quantity: str) -> float:
    """Observed order from the three coarsest-propagation meshes (finest n_y each)."""
    sub = {}
    for row in rows:
        if row["kh"] != kh:
            continue
        key = row["nx"]
        if key not in sub or row["ny"] < sub[key]["ny"]:
            sub[key] = row
    ordered = [sub[key] for key in sorted(sub)][:3]
    if len(ordered) < 3:
        return float("nan")
    e1, e2, e3 = (r[f"{quantity}_error"] for r in ordered)
    if min(e1, e2, e3) <= 0.0 or e2 == e3:
        return float("nan")
    return float(math.log(abs(e1 - e2) / abs(e2 - e3)) / math.log(ordered[1]["nx"] / ordered[0]["nx"]))


def draw_figures(analytic: list[dict], fem: list[dict]) -> dict[str, Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    khs = [r["kh"] for r in analytic]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.plot(khs, [r["a0_phase"] for r in analytic], "o-", label="A0 exact (Rayleigh--Lamb)")
    ax.plot(khs, [r["s0_phase"] for r in analytic], "s-", label="S0 exact (Rayleigh--Lamb)")
    ax.plot(khs, [r["flexural_phase"] for r in analytic], "--", lw=1.0,
            label="A0 thin-plate limit $kh/\\sqrt{3}\\,c_p/c_T$")
    finest = sorted([r for r in fem if r["nx"] == MESHES[-1][0]], key=lambda r: r["kh"])
    ax.plot([r["kh"] for r in finest], [r["a0_fem"] for r in finest], "^",
            fillstyle="none", ms=9, label=f"FEM A0 ({MESHES[-1][0]}x{MESHES[-1][1]} Floquet cell)")
    ax.plot([r["kh"] for r in finest], [r["s0_fem"] for r in finest], "v",
            fillstyle="none", ms=9, label=f"FEM S0 ({MESHES[-1][0]}x{MESHES[-1][1]} Floquet cell)")
    ax.axhline(C_R / C_T, color="0.4", ls=":", lw=1.0, label="$c_R/c_T$ (Rayleigh limit)")
    ax.axhline(C_L / C_T, color="0.7", ls=":", lw=1.0, label="$c_L/c_T$")
    ax.set_xscale("log")
    ax.set_xlabel("dimensionless wavenumber $kh$")
    ax.set_ylabel("phase velocity $c_{ph}/c_T$")
    ax.set_title("Rayleigh--Lamb dispersion: exact oracle vs Bloch--Floquet Q4 cell")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "dispersion.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for kh in KH_FEM:
        sub = sorted([r for r in fem if r["kh"] == kh], key=lambda r: r["nx"])
        ax.loglog([r["nx"] for r in sub], [max(r["a0_error"], 1e-12) for r in sub], "o-",
                  label=f"A0, kh={kh}")
        ax.loglog([r["nx"] for r in sub], [max(r["s0_error"], 1e-12) for r in sub], "s--",
                  label=f"S0, kh={kh}")
    ref_x = np.array([8.0, 32.0])
    ax.loglog(ref_x, 3e-3 * (ref_x / 8.0) ** (-2.0), "k:", lw=1.0, label="slope $-2$")
    ax.set_xlabel("cells along propagation direction $n_x$ (thickness $n_y$ per MESHES)")
    ax.set_ylabel("relative phase-velocity error vs exact")
    ax.set_title("Mesh convergence of the Floquet cell (consistent mass, Q4 plane strain)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(HERE / "convergence.png", dpi=150)
    plt.close(fig)
    return {"dispersion": HERE / "dispersion.png", "convergence": HERE / "convergence.png"}


def _pass(op: str, value: float, threshold: float) -> bool:
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    if op == "==":
        return value == threshold
    raise ValueError(op)


def build_gates(analytic, fem, assembly, cp_fit, flex_ratio, orders) -> list[dict]:
    finest = [r for r in fem if (r["nx"], r["ny"]) == MESHES[-1]]
    a0_thick = [r for r in analytic if r["kh"] == 20.0][0]
    s0_thick = [r for r in analytic if r["kh"] == 20.0][0]
    min_order = min(v for v in orders.values() if v == v)
    gates = [
        dict(id="s0_plate_velocity_extrapolated",
             quantity="|c_fit/(c_p/c_T) - 1|, S0 phase velocity extrapolated to kh->0 (both sides normalised by c_T)",
             reference="c_p = sqrt(E/(rho (1-nu^2)))",
             value=abs(cp_fit / (C_P / C_T) - 1.0), threshold=2e-3, operator="<="),
        dict(id="a0_flexural_limit",
             quantity="|c_A0(kh=0.05)/c_Kirchhoff - 1|",
             reference="c = kh c_p / sqrt(3) (thin-plate flexural limit)",
             value=abs(flex_ratio[0] - 1.0), threshold=1e-2, operator="<="),
        dict(id="a0_rayleigh_limit",
             quantity="|c_A0(kh=20)/(c_R/c_T) - 1|, both sides normalised by c_T",
             reference="c_R from the classical cubic z^3-8z^2+8z(3-2eta)-16(1-eta)=0",
             value=abs(a0_thick["a0_phase"] / (C_R / C_T) - 1.0), threshold=1e-2, operator="<="),
        dict(id="s0_rayleigh_limit",
             quantity="|c_S0(kh=20)/(c_R/c_T) - 1|, both sides normalised by c_T",
             reference="c_R (both lowest Lamb branches approach the Rayleigh speed as kh grows)",
             value=abs(s0_thick["s0_phase"] / (C_R / C_T) - 1.0), threshold=1e-2, operator="<="),
        dict(id="root_residual",
             quantity="max normalised Rayleigh--Lamb residual over every reported analytic root",
             reference="0 (the root is a root)",
             value=max(max(r["a0_residual"], r["s0_residual"]) for r in analytic),
             threshold=1e-9, operator="<="),
        dict(id="fem_phase_error_fine",
             quantity="max relative phase-velocity error of the %dx%d Floquet cell, A0 and S0, kh=0.2..1 (branches picked by mid-plane parity, not by index)" % MESHES[-1],
             reference="exact Rayleigh--Lamb roots",
             value=max(max(r["a0_error"], r["s0_error"]) for r in finest),
             threshold=3e-2, operator="<="),
        dict(id="fem_convergence_order",
             quantity="observed convergence order of the A0/S0 phase-velocity error, kh=0.2, three meshes (%s)" % "/".join("%dx%d" % m for m in MESHES[:3]),
             reference="theoretical order for Q4 with a consistent mass matrix >= 1.5",
             value=min_order, threshold=1.5, operator=">="),
        dict(id="mass_conservation",
             quantity="|sum(M)/(2 rho L 2h) - 1|, consistent mass matrix of the cell",
             reference="2 rho L 2h (one block per translation direction)",
             value=assembly["mass_relative_error"], threshold=1e-12, operator="<="),
        dict(id="gamma_zero_mode_count",
             quantity="number of zero-energy modes of the periodic cell at k=0, tol = 1e-10 max(omega)",
             reference="2 (the two rigid translations; the rigid rotation is incompatible with the Floquet identification)",
             value=float(assembly["zero_mode_count"]), threshold=2.0, operator="=="),
        dict(id="floquet_reduction_dofs",
             quantity="dofs of the Floquet-reduced cell",
             reference="2 (n_nodes - n_right_column): two unknowns per retained node",
             value=float(assembly["reduced_dofs"]), threshold=float(assembly["expected_reduced_dofs"]),
             operator="=="),
    ]
    for gate in gates:
        gate["value"] = float(gate["value"])
        gate["threshold"] = float(gate["threshold"])
        gate["passed"] = bool(_pass(gate["operator"], gate["value"], gate["threshold"]))
    return gates


def main() -> dict:
    analytic = analytic_table()
    fem = fem_table()
    assembly = assembly_checks()
    cp_fit = extrapolate_plate_speed(analytic)
    kh_small = [r for r in analytic if r["kh"] in (0.05, 0.1, 0.2)]
    flex_ratio = [r["a0_phase"] / r["flexural_phase"] for r in kh_small]
    orders = {q: convergence_order(fem, 0.2, q) for q in ("a0", "s0")}
    gates = build_gates(analytic, fem, assembly, cp_fit, flex_ratio, orders)

    result = dict(
        anchor="GAP-FEM-WAVE-DISPERSION",
        scenario="fem-lamb-dispersion",
        material=dict(young=E, poisson=NU, density=RHO, half_thickness=H),
        wave_speeds=dict(c_long=C_L, c_shear=C_T, c_plate=C_P, c_rayleigh=C_R,
                         c_long_over_c_shear=C_L / C_T,
                         c_plate_over_c_shear=C_P / C_T,
                         c_rayleigh_over_c_shear=C_R / C_T),
        analytic=analytic,
        fem=fem,
        assembly=assembly,
        plate_velocity_fit=cp_fit,
        a0_flexural_ratio=flex_ratio,
        observed_orders=orders,
        gates={g["id"]: dict(value=g["value"], threshold=g["threshold"],
                             operator=g["operator"], passed=g["passed"]) for g in gates},
    )
    (HERE / "result.json").write_text(json.dumps(result, indent=2))
    artifacts = {
        "result": "benchmarks/verified/fem_wave_dispersion/result.json",
        "gates": "benchmarks/verified/fem_wave_dispersion/gates_verified.json",
        "dispersion_figure": "benchmarks/verified/fem_wave_dispersion/dispersion.png",
        "convergence_figure": "benchmarks/verified/fem_wave_dispersion/convergence.png",
        "document": "docs/lamb-wave-dispersion-verification.md",
        "solver": "src/tensorfem/wave_dispersion.py",
        "tests": "tests/test_wave_dispersion.py",
    }
    payload = dict(
        schema="tensorfem.verified-gates.v1",
        anchor="fem_wave_dispersion",
        title="Rayleigh--Lamb dispersion: exact oracle vs a Bloch--Floquet Q4 plane-strain cell",
        mission="GAP-FEM-WAVE-DISPERSION",
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        criteria={g["id"]: g["threshold"] for g in gates},
        gates=gates,
        passed=all(g["passed"] for g in gates),
        artifacts=artifacts,
    )
    (HERE / "gates_verified.json").write_text(json.dumps(payload, indent=2))
    figures = draw_figures(analytic, fem)
    DOC.write_text(render_markdown(result, figures))
    print(json.dumps({g["id"]: (round(g["value"], 12), g["passed"]) for g in gates}, indent=1))
    print("overall passed:", payload["passed"])
    print("wrote", HERE / "result.json", "and", DOC)
    return result


def _fmt(value: float, digits: int = 6) -> str:
    if value != value:
        return "n/a"
    if value != 0.0 and (abs(value) < 1e-4 or abs(value) >= 1e6):
        return f"{value:.{digits - 1}e}"
    return f"{value:.{digits}f}"


def render_markdown(result: dict, figures: dict[str, Path]) -> str:
    a, f, g = result["analytic"], result["fem"], result["gates"]
    vs, asm, orders = result["wave_speeds"], result["assembly"], result["observed_orders"]
    passed = sum(1 for v in g.values() if v["passed"])
    total = len(g)
    worst = g["fem_phase_error_fine"]["value"]
    lines: list[str] = []
    add = lines.append
    add("# Rayleigh--Lamb 波色散：解析 oracle × Bloch--Floquet Q4 单元复核\n")
    add("**锚点** `GAP-FEM-WAVE-DISPERSION`（板内弹性波色散解析对标面）\n")
    add("**结论先行**（全部数字由 `run_verification.py` 写盘，零手抄）："
        f"门禁 **{passed}/{total} PASS**（总判定 `passed={str(all(v['passed'] for v in g.values())).lower()}`）；"
        f"细网格（{MESHES[-1][0]}×{MESHES[-1][1]}）上 A0/S0 相位速度对解析 Rayleigh--Lamb 的最大相对误差 "
        f"**{_fmt(worst*100, 3)}%**，**{'是' if worst < 3e-2 else '否'}** < 3%；"
        f"观测收敛阶 A0 **{_fmt(orders['a0'], 2)}** / S0 **{_fmt(orders['s0'], 2)}**。\n")

    add("## 1 案例介绍（物理背景 / 几何 / 参数）\n")
    add("均匀各向同性弹性板中的导波（Lamb 波）是船体板架振动—噪声链、冰激振动与水弹性耦合的"
        "最底层传播通道。板厚 `2h`、平面应变、上下表面无面力；对每个实数波数 `k` 两族模态满足精确超越方程：\n")
    add("$$\\begin{aligned}\n"
        "S_n:\\quad &4k^2pq\\,\\sin(ph)\\cos(qh)+(q^2-k^2)^2\\cos(ph)\\sin(qh)=0\\\\\n"
        "A_n:\\quad &4k^2pq\\,\\sin(qh)\\cos(ph)+(q^2-k^2)^2\\sin(ph)\\cos(qh)=0\n"
        "\\end{aligned}$$\n")
    add("其中 `p^2=ω^2/cL^2-k^2`、`q^2=ω^2/cT^2-k^2`，`p`、`q` 在各自截止频率以下为纯虚数，"
        "实现时按 `tan(ix)=i·tanh(x)` 分三段写成实数算式（见 §5 的坑）。\n")
    add("| 参数 | 值 |\n|---|---|")
    add(f"| 杨氏模量 E | {E:.1e} Pa |")
    add(f"| 泊松比 ν | {NU} |")
    add(f"| 密度 ρ | {RHO:.0f} kg/m³ |")
    add(f"| 半板厚 h | {H} m（板厚 20 mm） |")
    add(f"| c_L | {vs['c_long']:.2f} m/s（{vs['c_long_over_c_shear']:.6f} c_T） |")
    add(f"| c_T | {vs['c_shear']:.2f} m/s |")
    add(f"| 板纵波速 c_p=√(E/(ρ(1-ν²))) | {vs['c_plate']:.2f} m/s（{vs['c_plate_over_c_shear']:.6f} c_T） |")
    add(f"| Rayleigh 波速 c_R | {vs['c_rayleigh']:.2f} m/s（{vs['c_rayleigh_over_c_shear']:.6f} c_T） |\n")

    add("## 2 怎么做的\n")
    add("1. **解析侧**：`tensorfem.wave_dispersion.lamb_residual` 在三段频率区间用纯实数算术求值，"
        "对数网格扫描 + 二分求根，残差按自身两项幅度归一化。`c_L`/`c_T`/`c_p`/`c_R` 全部闭式。")
    add("2. **FEM 侧**：单个周期单胞（长 `L=2π/(4k)`、厚 `2h`），刚度用生产单元 "
        "`continuum.q4_stiffness`（平面应变、2×2 Gauss），质量用一致质量阵 `q4_consistent_mass`；"
        "施加 Floquet 条件 `u(x+L)=e^{ikL}u(x)`，以复归约矩阵 `T^H K T`、`T^H M T` 解 Hermitian 广义特征值问题。"
        "**归约在自由度空间进行**：每个保留节点保留 `u_x`、`u_z` 两个未知量，归约后自由度 "
        f"{asm['reduced_dofs']} = 2×({asm['n_nodes']}−{asm['n_right_column']})（见 §8 修复的缺陷）。")
    add("3. **分支识别不看 oracle**：单胞本征模态按频率排序，而 S0 的序号随 `kh` 变化"
        "（`kh=0.2` 时 A1 支插在 A0 与 S0 之间，S0 落在第 3 个模态）。因此按**中面奇偶性**分类："
        "S 族 `u_x` 偶 / `u_z` 奇，A 族反之。分类器只读模态形状，解析根只用于打分，不用于挑选。")
    add("4. **网格梯**：`(n_x, n_y) = " + " / ".join("%d,%d" % m for m in MESHES) +
        f"`，`kh ∈ {list(KH_FEM)}`。")
    add("5. 全部数字由本脚本写盘（`result.json` / `gates_verified.json` / 本文件），零手抄；图由 matplotlib 现场生成。\n")

    add("## 3 计算结果\n")
    add("### 3.1 解析 Rayleigh--Lamb 两族最低支（无量纲相位速度 c_ph/c_T）\n")
    add("| kh | A0 c/c_T | S0 c/c_T | A0 薄板极限 kh·c_p/(√3 c_T) | A0 残差 | S0 残差 |")
    add("|---|---|---|---|---|---|")
    for r in a:
        add(f"| {r['kh']:g} | {_fmt(r['a0_phase'])} | {_fmt(r['s0_phase'])} | "
            f"{_fmt(r['flexural_phase'])} | {_fmt(r['a0_residual'], 3)} | {_fmt(r['s0_residual'], 3)} |")
    add("")
    add("### 3.2 Bloch--Floquet 单元 vs 解析（相位速度相对误差）\n")
    add("| kh | 网格 nx×ny | A0 FEM（序号/奇偶分数） | A0 解析 | A0 误差 | S0 FEM（序号/奇偶分数） | S0 解析 | S0 误差 |")
    add("|---|---|---|---|---|---|---|---|")
    for r in f:
        add(f"| {r['kh']:g} | {r['nx']}×{r['ny']} | {_fmt(r['a0_fem'])} ({r['a0_index']}/{r['a0_symmetry']:.2f}) | "
            f"{_fmt(r['a0_exact'])} | {r['a0_error']*100:.4f}% | {_fmt(r['s0_fem'])} "
            f"({r['s0_index']}/{r['s0_symmetry']:.2f}) | {_fmt(r['s0_exact'])} | {r['s0_error']*100:.4f}% |")
    add("")
    add("### 3.3 装配与 Γ 点自检\n")
    add("| 项 | 值 |\n|---|---|")
    add(f"| ΣM（一致质量阵全和） | {_fmt(asm['total_mass'])} |")
    add(f"| ρ·L·2h | {_fmt(asm['expected_mass'])} |")
    add(f"| 相对误差 | {_fmt(asm['mass_relative_error'], 3)} |")
    add(f"| 归约后自由度 / 期望 | {asm['reduced_dofs']} / {asm['expected_reduced_dofs']} |")
    add(f"| Γ 点零模态数（期望 2：两个刚体平移） | {asm['zero_mode_count']} |")
    add(f"| Γ 点 ω₂（首个非零频率） | {_fmt(asm['gamma_omega'][2], 6)} rad/s |\n")

    add("## 4 结果图\n")
    add(f"![dispersion]({figures['dispersion'].relative_to(REPO)})")
    add("")
    add("*图 1 精确 Rayleigh--Lamb A0/S0 支、薄板极限、c_R/c_L 渐近线与 Floquet 单元结果对照。*\n")
    add(f"![convergence]({figures['convergence'].relative_to(REPO)})")
    add("")
    add("*图 2 网格收敛：A0/S0 相位速度误差随单胞沿传播方向单元数下降，参考斜率 −2。*\n")

    add("## 5 注意点 / 踩过的坑\n")
    add("1. **复数相位的陷阱**：最初用 `cmath` 求 `sin(ph)`/`cos(ph)` 再取实部，"
        "在 `ω<k c_T` 区间残差恒为 0，A0 支被整体漏掉。改法：按三段区间用实数算术显式写 `tan`/`tanh`。")
    add("2. **按序号取模态是错的**：S0 在 `kh=0.2` 落在第 3 个模态（A1 支插入其间），"
        "取 `omega[1]` 会把 A1 当成 S0——修复前的锚点正是这样报出「S0 误差 122%」。"
        "现改为按中面奇偶性分类（§2 第 3 条），与 `kh` 无关。")
    add("3. **归约必须在自由度空间做**（§8）：按节点编号建变换矩阵会把不同节点的 `u_x`/`u_z` 混叠，"
        "结果看起来「有收敛阶」却是收敛到错误的算子。")
    add("4. **适用域**：均匀各向同性板、平面应变、线性弹性、无衰减；不含流固耦合加载、层合板、"
        "粘弹阻尼、有限变形。")
    add("5. **解析扫描的退化根**：`ω = k c_T`（`q=0`）使 A 族残差的一项恒为零，"
        "`rayleigh_lamb_frequencies(n_roots>=2)` 的 A 族列表里会出现 `ω/k/c_T = 1.0` 的伪根"
        "（见 `result.json` 的 `analytic` 表只用最低根，不受影响）。这是已知局限，未在本单修复。\n")

    add("## 6 误差判定（是否 <3%）\n")
    add("| 关键量 | 基准（独立来源） | 本实现误差/指标 | 门限 | 是否 PASS | 是否 <3% |")
    add("|---|---|---|---|---|---|")
    rows = [
        ("s0_plate_velocity_extrapolated", "S0 相位速度 kh→0 外推", f"√(E/(ρ(1-ν²))) = {C_P:.2f} m/s", True),
        ("a0_flexural_limit", "A0 相位速度 kh=0.05", "Kirchhoff 极限 kh·c_p/√3", True),
        ("a0_rayleigh_limit", "A0 相位速度 kh=20", f"Rayleigh 波速 {C_R:.2f} m/s", True),
        ("s0_rayleigh_limit", "S0 相位速度 kh=20", f"Rayleigh 波速 {C_R:.2f} m/s", True),
        ("fem_phase_error_fine", f"FEM A0/S0 相位速度（kh=0.2–1，{MESHES[-1][0]}×{MESHES[-1][1]}）", "解析 Rayleigh--Lamb", True),
        ("root_residual", "解析根归一化残差", "0（根应满足方程）", True),
        ("mass_conservation", "一致质量阵质量守恒", "ρ·L·2h", False),
        ("fem_convergence_order", "网格收敛阶（min of A0/S0）", "理论 ≥1.5（Q4 一致质量）", False),
        ("gamma_zero_mode_count", "Γ 点零模态数", "2（两个刚体平移）", False),
        ("floquet_reduction_dofs", "归约后自由度", "2(n_nodes − n_right)", False),
    ]
    for key, label, reference, percent in rows:
        gate = g[key]
        shown = f"{gate['value']*100:.4f}%" if percent else f"{gate['value']:.6g}"
        below3 = ("是" if gate["value"] < 3e-2 else "否") if percent else "—（非百分量）"
        add(f"| {label} | {reference} | {shown} | {gate['operator']} {gate['threshold']:g} | "
            f"{'PASS' if gate['passed'] else 'FAIL'} | {below3} |")
    add("")
    add(f"**总判定**：门禁 {passed}/{total} PASS（`passed={str(all(v['passed'] for v in g.values())).lower()}`）；"
        f"FEM 相位速度对解析基准的最大误差 **{_fmt(worst*100, 4)}%**，"
        f"**{'< 3%，达标' if worst < 3e-2 else '≥ 3%，未达标'}**。\n")

    add("## 7 复现\n")
    add("```bash")
    add("PYTHONPATH=src python benchmarks/verified/fem_wave_dispersion/run_verification.py")
    add("PYTHONPATH=src pytest tests/test_wave_dispersion.py -q")
    add("python tools/verified_anchor_audit.py --check")
    add("```\n")

    add("## 8 已修复的缺陷：Floquet 归约建在节点空间（m16896）\n")
    add("修复前的锚点是**诚实的负面结果**：Γ 点零模态数不足、FEM 相位速度误差对网格不敏感，"
        "`tests/test_wave_dispersion.py::test_bloch_floquet_reduction_defect_is_open` 以 fail-closed 方式锁死。"
        "根因是本单定位的：`bloch_cell_matrices` 把约化变换矩阵的列索引取成**节点号**（`nred` = 保留节点数），"
        f"而自由度是每节点 2 个，于是 `u_x` 与 `u_z` 被混叠进同一列——`kh=0.2`、`8×4` 时归约矩阵只有 40×40 而正确维数是 "
        f"{asm['expected_reduced_dofs']}×{asm['expected_reduced_dofs']}。修复：按自由度建变换矩阵"
        "（列号 `2·node+d`），并在 `bloch_cell_modes` 中把模态还原到未归约自由度空间以便做中面奇偶分类。"
        "修复前后数字见 git 历史（`benchmarks/verified/fem_wave_dispersion/gates_verified.json` 的上一个版本），"
        "本文件与门禁文件由同一脚本重生成。\n")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
