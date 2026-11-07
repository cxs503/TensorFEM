"""Structural verification ladder driver: beam / plate / modal analytic anchors.

Run with ``python3 benchmarks/structural_ladder.py``.  It writes, for each of the
three rungs,

* ``benchmarks/verified/<case>/result.json``   -- every number + the verdict
* ``benchmarks/verified/<case>/convergence.png``-- error vs mesh size
* ``benchmarks/verified/<case>/README.md``     -- table rendered from the JSON

plus ``benchmarks/verified/structural_ladder_summary.json`` and the paper-level
note ``docs/verification/STRUCTURAL_LADDER.md``.  Every table in those documents
is rendered from the computed numbers, nothing is typed by hand.

The physics: see :mod:`tensorfem.structural_verification`.  Gates (from the
campaign node definition):

* G1 Euler--Bernoulli cantilever, uniform load: tip deflection vs ``qL^4/8EI``
  within 1 %, field ``L2`` convergence order >= 1.5;
* G2 Kirchhoff SSSS square plate, uniform pressure: centre coefficient vs
  0.00406 within 2 %, observed order >= 1.5;
* G3 cantilever modes 1--5: frequency error < 1 % each, observed order >= 1.5.
"""

from __future__ import annotations

import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem import structural_verification as sv  # noqa: E402
from tensorfem.modal import cantilever_beam_modes  # noqa: E402

OUT = ROOT / "benchmarks" / "verified"
DOCS = ROOT / "docs" / "verification"

# ---------------------------------------------------------------------------
# Case data
# ---------------------------------------------------------------------------
BEAM = dict(length=1.0, width=0.010, height=0.010, young=200.0e9, load=5.0)
BEAM["inertia"] = BEAM["width"] * BEAM["height"] ** 3 / 12.0
BEAM["EI"] = BEAM["young"] * BEAM["inertia"]
BEAM_LEVELS = (2, 4, 8, 16, 32, 64)

PLATE = dict(side=1.0, young=210.0e9, poisson=0.3, thickness=0.010, pressure=1.0e4)
PLATE_LEVELS = (2, 4, 6, 8, 10, 12, 16, 20, 24)
PLATE_GATE_MESH = 16

MODAL = dict(
    length=1.0,
    width=0.010,
    height=0.010,
    young=200.0e9,
    density=7850.0,
)
MODAL["inertia"] = MODAL["width"] * MODAL["height"] ** 3 / 12.0
MODAL["EI"] = MODAL["young"] * MODAL["inertia"]
MODAL["rho_a"] = MODAL["density"] * MODAL["width"] * MODAL["height"]
MODAL_LEVELS = (4, 8, 16, 32, 64)
MODAL_GATE_MESH = 16
MODES = 5


def _env() -> dict:
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "machine": platform.machine(),
        "dtype": "torch.float64 / numpy.float64",
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Rung 1: Euler--Bernoulli cantilever, uniform load
# ---------------------------------------------------------------------------
def run_beam() -> dict:
    exact = sv.euler_bernoulli_cantilever_tip(BEAM["length"], BEAM["EI"], BEAM["load"])
    levels = []
    for n in BEAM_LEVELS:
        sol = sv.solve_cantilever_udl(BEAM["length"], BEAM["EI"], BEAM["load"], n)
        levels.append(
            {
                "elements": n,
                "dofs": 2 * (n + 1),
                "h": BEAM["length"] / n,
                "tip_deflection_m": sol.tip_deflection,
                "tip_rotation_rad": sol.tip_rotation,
                "clamped_shear_recovered_N_per_m": sol.clamped_shear(),
                "tip_relative_error": abs(sol.tip_deflection - exact["tip_deflection"])
                / exact["tip_deflection"],
                "rotation_relative_error": abs(sol.tip_rotation - exact["tip_rotation"])
                / exact["tip_rotation"],
                "field_l2_relative_error": sv.beam_l2_relative_error(sol),
            }
        )
    order_fe = sv.observed_order(
        np.array([row["h"] for row in levels[:-1]]),
        np.array([row["field_l2_relative_error"] for row in levels[:-1]]),
    )
    gate_mesh = 8
    gate = next(row for row in levels if row["elements"] == gate_mesh)
    # The clamp reaction is the negative of the applied load in the assembled
    # sign convention, so equilibrium is checked on magnitudes.
    equilibrium_error = abs(
        abs(gate["clamped_shear_recovered_N_per_m"]) - BEAM["load"] * BEAM["length"]
    ) / (BEAM["load"] * BEAM["length"])
    gates = [
        {
            "id": "G1.tip_deflection",
            "description": "|w_tip(FE) - qL^4/(8EI)| / (qL^4/(8EI)) < 1e-2 at 8 elements",
            "value": gate["tip_relative_error"],
            "limit": 1e-2,
            "relation": "le",
            "passed": bool(gate["tip_relative_error"] < 1e-2),
        },
        {
            "id": "G1.field_order",
            "description": "field L2 convergence order >= 1.5 (levels 2..32)",
            "value": order_fe,
            "limit": 1.5,
            "relation": "ge",
            "passed": bool(order_fe >= 1.5),
        },
        {
            "id": "G1.equilibrium",
            "description": "recovered clamp shear vs qL, relative error < 1e-8",
            "value": equilibrium_error,
            "limit": 1e-8,
            "relation": "le",
            "passed": bool(equilibrium_error < 1e-8),
        },
    ]
    return {
        "case": "Euler-Bernoulli cantilever, uniform load q",
        "geometry": {
            "length_m": BEAM["length"],
            "width_m": BEAM["width"],
            "height_m": BEAM["height"],
            "second_moment_m4": BEAM["inertia"],
            "young_pa": BEAM["young"],
            "EI_n_m2": BEAM["EI"],
            "load_n_per_m": BEAM["load"],
            "slenderness_L_over_h": BEAM["length"] / BEAM["height"],
        },
        "reference": {
            "method": "closed form, Euler-Bernoulli",
            "deflection_field": "w(x)=q x^2 (6L^2-4Lx+x^2)/(24 EI)",
            "tip_deflection_m": exact["tip_deflection"],
            "tip_rotation_rad": exact["tip_rotation"],
            "clamped_moment_n": exact["clamped_moment"],
        },
        "element": "2-node cubic Hermite (4 dof), tensorfem.modal.beam_bending_stiffness",
        "consistent_load": "q*[L/2, L^2/12, L/2, -L^2/12]",
        "levels": levels,
        "observed_order_field_l2": order_fe,
        "gates": gates,
        "verdict": "PASS" if all(g["passed"] for g in gates) else "FAIL",
    }


# ---------------------------------------------------------------------------
# Rung 2: Kirchhoff plate, simply supported, uniform pressure
# ---------------------------------------------------------------------------
def run_plate() -> dict:
    series = sv.plate_centre_deflection_series(
        side=PLATE["side"],
        young=PLATE["young"],
        poisson=PLATE["poisson"],
        thickness=PLATE["thickness"],
        pressure=PLATE["pressure"],
    )
    rigidity = PLATE["young"] * PLATE["thickness"] ** 3 / (12.0 * (1.0 - PLATE["poisson"] ** 2))
    alpha_series = series * rigidity / (PLATE["pressure"] * PLATE["side"] ** 4)
    levels = []
    for n in PLATE_LEVELS:
        sol = sv.kirchhoff_ss_plate_uniform_load(
            side=PLATE["side"],
            young=PLATE["young"],
            poisson=PLATE["poisson"],
            thickness=PLATE["thickness"],
            pressure=PLATE["pressure"],
            n_x=n,
        )
        levels.append(
            {
                "n_x": n,
                "n_y": n,
                "n_dof": sol.n_dof,
                "n_free_dof": sol.n_free,
                "h": PLATE["side"] / n,
                "centre_deflection_m": sol.centre_deflection,
                "alpha": sol.alpha,
                "error_vs_timoshenko_00406": abs(sol.alpha - sv.PLATE_ALPHA_TIMOSHENKO)
                / sv.PLATE_ALPHA_TIMOSHENKO,
                "error_vs_navier_series": abs(sol.alpha - alpha_series) / alpha_series,
            }
        )
    gate = next(row for row in levels if row["n_x"] == PLATE_GATE_MESH)
    order = sv.observed_order(
        np.array([row["h"] for row in levels]),
        np.array([row["error_vs_navier_series"] for row in levels]),
    )
    gates = [
        {
            "id": "G2.centre_deflection",
            "description": "|alpha_FE - 0.00406| / 0.00406 < 2e-2 at 16x16 elements",
            "value": gate["error_vs_timoshenko_00406"],
            "limit": 2e-2,
            "relation": "le",
            "passed": bool(gate["error_vs_timoshenko_00406"] < 2e-2),
        },
        {
            "id": "G2.centre_deflection_vs_series",
            "description": "|alpha_FE - alpha_Navier| / alpha_Navier < 2e-2 at 16x16 elements",
            "value": gate["error_vs_navier_series"],
            "limit": 2e-2,
            "relation": "le",
            "passed": bool(gate["error_vs_navier_series"] < 2e-2),
        },
        {
            "id": "G2.order",
            "description": "observed convergence order >= 1.5 (levels 2..24)",
            "value": order,
            "limit": 1.5,
            "relation": "ge",
            "passed": bool(order >= 1.5),
        },
    ]
    return {
        "case": "Kirchhoff thin plate, SSSS square, uniform pressure",
        "geometry": {
            "side_m": PLATE["side"],
            "thickness_m": PLATE["thickness"],
            "young_pa": PLATE["young"],
            "poisson": PLATE["poisson"],
            "pressure_pa": PLATE["pressure"],
            "bending_stiffness_n_m": rigidity,
            "a_over_h": PLATE["side"] / PLATE["thickness"],
        },
        "reference": {
            "method": "Timoshenko & Woinowsky-Krieger table 8 (0.00406) and Navier double-sine series",
            "alpha_timoshenko": sv.PLATE_ALPHA_TIMOSHENKO,
            "alpha_navier_series": alpha_series,
            "centre_deflection_series_m": series,
        },
        "element": "Bogner-Fox-Schmit C1 bicubic Hermite rectangle (16 dof; w, w,x, w,y, w,xy)",
        "boundary_condition": "hard simple support (w = 0 and tangential slope = 0 on edges)",
        "levels": levels,
        "observed_order": order,
        "gates": gates,
        "verdict": "PASS" if all(g["passed"] for g in gates) else "FAIL",
    }


# ---------------------------------------------------------------------------
# Rung 3: cantilever modes 1..5
# ---------------------------------------------------------------------------
def run_modal() -> dict:
    ladder = sv.cantilever_modal_ladder(
        length=MODAL["length"],
        EI=MODAL["EI"],
        rho_a=MODAL["rho_a"],
        elements=MODAL_LEVELS,
        modes=MODES,
    )
    roots = sv.cantilever_roots(MODES)
    table_dev = max(
        abs(roots[i] - sv.CANTILEVER_BETA_L_TABLE[i]) / sv.CANTILEVER_BETA_L_TABLE[i]
        for i in range(MODES)
    )
    # Residual of the characteristic equation, normalised by cosh(z) so the
    # hyperbolic growth does not mask the root accuracy.
    residual = max(
        abs(math.cos(z) * math.cosh(z) + 1.0) / math.cosh(z) for z in roots
    )
    errors = ladder.relative_errors()
    levels = []
    for i, n in enumerate(ladder.elements):
        levels.append(
            {
                "elements": n,
                "dofs": 2 * (n + 1),
                "h": MODAL["length"] / n,
                "frequencies_hz": ladder.fe_hertz[i].tolist(),
                "relative_errors": errors[i].tolist(),
            }
        )
    orders = {
        f"mode_{m + 1}": sv.observed_order(
            np.array([row["h"] for row in levels]),
            np.array([row["relative_errors"][m] for row in levels]),
        )
        for m in range(MODES)
    }
    gate_row = next(row for row in levels if row["elements"] == MODAL_GATE_MESH)
    worst_gate = max(gate_row["relative_errors"])
    gates = [
        {
            "id": "G3.frequencies",
            "description": "max relative frequency error over modes 1-5 < 1e-2 at 16 elements",
            "value": float(worst_gate),
            "limit": 1e-2,
            "relation": "le",
            "passed": bool(worst_gate < 1e-2),
        },
        {
            "id": "G3.roots_vs_table",
            "description": "bisection roots reproduce the tabulated beta_n L to < 1e-5 (source precision)",
            "value": float(table_dev),
            "limit": 1e-5,
            "relation": "le",
            "passed": bool(table_dev < 1e-5),
        },
        {
            "id": "G3.characteristic_residual",
            "description": "max |cos(z) cosh(z) + 1| / cosh(z) at the computed roots < 1e-12",
            "value": float(residual),
            "limit": 1e-12,
            "relation": "le",
            "passed": bool(residual < 1e-12),
        },
        {
            "id": "G3.order",
            "description": "observed frequency convergence order >= 1.5 for every mode",
            "value": float(min(orders.values())),
            "limit": 1.5,
            "relation": "ge",
            "passed": bool(min(orders.values()) >= 1.5),
        },
    ]
    return {
        "case": "Euler-Bernoulli cantilever free vibration, modes 1-5",
        "geometry": {
            "length_m": MODAL["length"],
            "width_m": MODAL["width"],
            "height_m": MODAL["height"],
            "young_pa": MODAL["young"],
            "density_kg_m3": MODAL["density"],
            "EI_n_m2": MODAL["EI"],
            "rho_a_kg_per_m": MODAL["rho_a"],
            "slenderness_L_over_h": MODAL["length"] / MODAL["height"],
        },
        "reference": {
            "method": "cos(beta L) cosh(beta L) = -1 (Blevins 1979 table 8-1)",
            "beta_l": list(roots),
            "beta_l_table": list(sv.CANTILEVER_BETA_L_TABLE),
            "angular_frequencies_rad_s": list(ladder.exact_angular),
            "frequencies_hz": [w / (2.0 * math.pi) for w in ladder.exact_angular],
        },
        "element": "2-node cubic Hermite beam, consistent mass (tensorfem.modal)",
        "levels": levels,
        "observed_orders": orders,
        "gates": gates,
        "verdict": "PASS" if all(g["passed"] for g in gates) else "FAIL",
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def fig_beam(beam: dict, outdir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    h = np.array([row["h"] for row in beam["levels"][:-1]])
    err = np.array([row["field_l2_relative_error"] for row in beam["levels"][:-1]])
    ax = axes[0]
    ax.loglog(h, err, "o-", label="FE field $L_2$ error")
    slope = beam["observed_order_field_l2"]
    ax.loglog(h, err[0] * (h / h[0]) ** 4, "k--", alpha=0.6, label=r"$O(h^4)$ reference")
    _style(ax, "element length $h$ [m]", r"$\|w_h-w_{ex}\|_{L_2}/\|w_{ex}\|_{L_2}$",
           f"cantilever UDL: field convergence (fitted order {slope:.2f})")
    ax.legend(fontsize=8)
    ax.annotate(f"slope = {slope:.2f}", xy=(h[len(h) // 2], err[len(err) // 2]),
                xytext=(0.05, 0.85), textcoords="axes fraction", fontsize=9)

    sol = sv.solve_cantilever_udl(BEAM["length"], BEAM["EI"], BEAM["load"], 8)
    x, w_fe, w_ex = sv.beam_field(sol, 8)
    order = np.argsort(x)
    ax = axes[1]
    ax.plot(x[order] * 1e3, w_fe[order] * 1e3, "o", ms=4, label="FE (8 elements)")
    xx = np.linspace(0, BEAM["length"], 400)
    exact_curve = np.array(
        [
            sv.euler_bernoulli_cantilever_deflection(t, BEAM["length"], BEAM["EI"], BEAM["load"])
            for t in xx
        ]
    )
    ax.plot(xx * 1e3, exact_curve * 1e3, "-", lw=1.2, label="exact quartic")
    ax.invert_yaxis()
    _style(ax, "x [mm]", "deflection w [mm]", "deflection profile, q = 5 N/m")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = outdir / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_plate(plate: dict, outdir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    h = np.array([row["h"] for row in plate["levels"]])
    err = np.array([row["error_vs_navier_series"] for row in plate["levels"]])
    ax = axes[0]
    ax.loglog(h, err, "o-", label="|alpha$_{FE}$-alpha$_{Navier}$|/alpha")
    ax.loglog(h, err[0] * (h / h[0]) ** 2, "k--", alpha=0.6, label=r"$O(h^2)$ reference")
    ax.axhline(0.02, color="r", ls=":", label="2 % gate")
    _style(ax, "element size $h=a/n$ [m]", "relative error of $w_{max}$",
           f"SSSS plate: centre deflection convergence (order {plate['observed_order']:.2f})")
    ax.legend(fontsize=8)

    sol = sv.kirchhoff_ss_plate_uniform_load(
        side=PLATE["side"], young=PLATE["young"], poisson=PLATE["poisson"],
        thickness=PLATE["thickness"], pressure=PLATE["pressure"], n_x=PLATE_GATE_MESH,
    )
    X, Y, W = sol.grid()
    ax = axes[1]
    cs = ax.contourf(X * 1e3, Y * 1e3, W * 1e3, levels=20, cmap="viridis")
    ax.set_aspect("equal")
    fig.colorbar(cs, ax=ax, label="w [mm]")
    _style(ax, "x [mm]", "y [mm]", f"deflection field, {PLATE_GATE_MESH}x{PLATE_GATE_MESH} BFS elements")
    fig.tight_layout()
    path = outdir / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


def fig_modal(modal: dict, outdir: Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    levels = modal["levels"]
    ax = axes[0]
    for m in range(MODES):
        err = [row["relative_errors"][m] for row in levels]
        ax.loglog([row["h"] for row in levels], err, "o-", label=f"mode {m + 1}")
    ax.axhline(0.01, color="r", ls=":", label="1 % gate")
    _style(ax, "element length $h$ [m]", "relative frequency error",
           "cantilever: frequency convergence, modes 1-5")
    ax.legend(fontsize=7)

    result = cantilever_beam_modes(MODAL["length"], 32, MODAL["EI"], MODAL["rho_a"], modes=MODES)
    xs = np.linspace(0.0, MODAL["length"], 201)
    le = MODAL["length"] / 32
    ax = axes[1]
    for m in range(MODES):
        dofs = result.modes[:, m].numpy()
        w = np.zeros_like(xs)
        for i, t in enumerate(xs):
            e = min(int(t / le), 31)
            xi = (t - e * le) / le
            h, _, _ = sv.hermite_cubic_1d(xi, le)
            local = np.array([dofs[2 * e], dofs[2 * e + 1], dofs[2 * e + 2], dofs[2 * e + 3]])
            w[i] = local @ h
        w = w / np.max(np.abs(w))
        ax.plot(xs, w + (MODES - 1 - m) * 2.0,
                label=f"$f_{m + 1}$ = {result.frequencies_hz[m]:.1f} Hz")
    _style(ax, "x [m]", "normalised mode shape (offset)", "mode shapes, 32 elements")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = outdir / "convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path.name


# ---------------------------------------------------------------------------
# Markdown rendering (tables injected from computed data)
# ---------------------------------------------------------------------------
def readme_beam(b: dict) -> str:
    lines = [
        "# fem_beam - Euler-Bernoulli cantilever under uniform load",
        "",
        f"Verdict: **{b['verdict']}**  ",
        f"Element: {b['element']}; consistent load {b['consistent_load']}.",
        "",
        "| n_elem | h [mm] | w_tip FE [mm] | rel. err tip | L2 field rel. err | shear recovered / qL |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in b["levels"]:
        lines.append(
            "| {elements} | {h:.3f} | {tip:.6f} | {e:.3e} | {l2:.3e} | {s:.6f} |".format(
                elements=row["elements"], h=row["h"] * 1e3,
                tip=row["tip_deflection_m"] * 1e3,
                e=row["tip_relative_error"], l2=row["field_l2_relative_error"],
                s=row["clamped_shear_recovered_N_per_m"] / (BEAM["load"] * BEAM["length"]),
            )
        )
    exact = b["reference"]["tip_deflection_m"]
    lines += [
        "",
        f"Analytic tip deflection `qL^4/(8EI)` = {exact * 1e3:.6f} mm, tip rotation "
        f"`qL^3/(6EI)` = {b['reference']['tip_rotation_rad']:.6f} rad.",
        f"Field L2 convergence order (2..32 elements): **{b['observed_order_field_l2']:.3f}**.",
        "",
        "Gates: " + "; ".join(
            f"{g['id']} = {g['value']:.3e} ({'<=' if g['relation'] == 'le' else '>='} "
            f"{g['limit']:.1e}) -> "
            f"{'PASS' if g['passed'] else 'FAIL'}" for g in b["gates"]
        ),
        "",
        "Note: nodal displacements of this element are superconvergent (exact for the "
        "quartic exact solution), so the tip error sits at round-off while the *field* "
        "error measures the real O(h^4) interpolation error.",
        "",
    ]
    return "\n".join(lines)


def readme_plate(p: dict) -> str:
    lines = [
        "# fem_plate - Kirchhoff SSSS square plate, uniform pressure",
        "",
        f"Verdict: **{p['verdict']}**  ",
        f"Element: {p['element']}; BC: {p['boundary_condition']}.",
        "",
        "| n x n | n_dof | h [mm] | w_max [mm] | alpha = w D/(q a^4) | err vs 0.00406 | err vs Navier |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in p["levels"]:
        lines.append(
            "| {n} | {dof} | {h:.2f} | {w:.6f} | {a:.6f} | {e1:.3%} | {e2:.3%} |".format(
                n=row["n_x"], dof=row["n_dof"], h=row["h"] * 1e3,
                w=row["centre_deflection_m"] * 1e3, a=row["alpha"],
                e1=row["error_vs_timoshenko_00406"], e2=row["error_vs_navier_series"],
            )
        )
    ref = p["reference"]
    lines += [
        "",
        f"Reference: Timoshenko table value alpha = {ref['alpha_timoshenko']}, Navier series "
        f"alpha = {ref['alpha_navier_series']:.9f}.",
        f"Observed convergence order: **{p['observed_order']:.3f}** (deflection, from below).",
        "",
        "Gates: " + "; ".join(
            f"{g['id']} = {g['value']:.3e} ({'<=' if g['relation'] == 'le' else '>='} "
            f"{g['limit']:.1e}) -> "
            f"{'PASS' if g['passed'] else 'FAIL'}" for g in p["gates"]
        ),
        "",
    ]
    return "\n".join(lines)


def readme_modal(m: dict) -> str:
    lines = [
        "# fem_modal - Euler-Bernoulli cantilever, modes 1-5",
        "",
        f"Verdict: **{m['verdict']}**  ",
        f"Element: {m['element']}.",
        "",
        "| mode | beta_n L | f_exact [Hz] | " + " | ".join(f"err @ {row['elements']} el." for row in m["levels"]) + " |",
        "|---:|---:|---:|" + "---:|" * len(m["levels"]),
    ]
    ref = m["reference"]
    for k in range(MODES):
        cells = " | ".join(
            f"{row['relative_errors'][k]:.2e}" for row in m["levels"]
        )
        lines.append(
            f"| {k + 1} | {ref['beta_l'][k]:.6f} | {ref['frequencies_hz'][k]:.3f} | {cells} |"
        )
    lines += [
        "",
        "Observed orders: " + ", ".join(
            f"{k} = {v:.3f}" for k, v in m["observed_orders"].items()
        ) + ".",
        "",
        "Gates: " + "; ".join(
            f"{g['id']} = {g['value']:.3e} ({'<=' if g['relation'] == 'le' else '>='} "
            f"{g['limit']:.1e}) -> "
            f"{'PASS' if g['passed'] else 'FAIL'}" for g in m["gates"]
        ),
        "",
    ]
    return "\n".join(lines)


def docs_note(beam: dict, plate: dict, modal: dict, summary: dict) -> str:
    b = beam
    p = plate
    m = modal
    rows = "\n".join(
        f"| {item['rung']} | {item['case']} | {item['quantity']} | {item['fem']:.6g} | "
        f"{item['reference']:.6g} | {item['relative_error']:.2%} | {'PASS' if item['passed'] else 'FAIL'} |"
        for item in summary["anchors"]
    )
    return f"""# TensorFEM structural verification ladder (beam / plate / modal)

Generated by `benchmarks/structural_ladder.py` on {summary['generated_at']}.
Environment: Python {summary['environment']['python']}, torch {summary['environment']['torch']},
numpy {summary['environment']['numpy']}, {summary['environment']['machine']}.

## 1. Why this ladder exists

TensorFEM's verification surface was concentrated on 2-D/3-D continuum,
contact, shell and buckling problems.  The three one-dimensional textbook chains
that a structural code is normally trusted on first -- beam bending, plate
bending, beam vibration -- had no single place where a reviewer could see
"finite element vs closed form vs mesh size".  This note is that place.  It adds
`tensorfem.structural_verification` (solver side) and this benchmark rung
(evidence side); nothing in the production solver paths was changed apart from
exposing two already-existing quadrature helpers of the BFS plate element.

## 2. Cases, methods, tool chain

| rung | case | discretisation | reference |
|---|---|---|---|
| fem_beam | Euler-Bernoulli cantilever, uniform load `q` | 2-node cubic Hermite, consistent load `q[L/2, L^2/12, L/2, -L^2/12]` | `w(x)=qx^2(6L^2-4Lx+x^2)/(24EI)`, tip `qL^4/(8EI)` |
| fem_plate | Kirchhoff SSSS square plate, uniform pressure | Bogner-Fox-Schmit C1 bicubic Hermite, 16 dof, hard simple support | Timoshenko table 8 (`alpha=0.00406`) and Navier series (`0.00406235`) |
| fem_modal | cantilever modes 1-5 | same Hermite beam + consistent mass, generalised eigenproblem | `cos(beta L)cosh(beta L)=-1`, `omega_n=(beta_n L)^2 sqrt(EI/(rho A L^4))` |

Materials and geometry are in each `result.json`; the beam is 1 m x 10 mm x 10 mm
steel (`E` = 200 GPa), the plate is 1 m x 1 m x 10 mm (`E` = 210 GPa,
`nu` = 0.3, `a/h` = 100), the modal beam is the same 1 m bar with
`rho` = 7850 kg/m^3.

Reproduce: `python3 benchmarks/structural_ladder.py` (writes `benchmarks/verified/`),
tests: `pytest tests/test_structural_verification.py -q`.

## 3. Results

| rung | case | quantity | FE | reference | rel. error | verdict |
|---|---|---|---|---|---|---|
{rows}

### fem_beam - field convergence

{b['levels'][0]['elements']} -> {b['levels'][-1]['elements']} elements: field `L2` relative error
{b['levels'][0]['field_l2_relative_error']:.3e} -> {b['levels'][-1]['field_l2_relative_error']:.3e},
fitted order **{b['observed_order_field_l2']:.3f}**.  Nodal deflections are superconvergent
(tip error {b['levels'][3]['tip_relative_error']:.2e} at 16 elements), so the field norm, not the
tip value, is what proves the rate.  Equilibrium: the reaction recovered from element
end forces equals `qL` to {b['gates'][2]['value']:.1e} relative.

### fem_plate - centre coefficient

alpha converges from below to the Navier value
{p['reference']['alpha_navier_series']:.9f} with fitted order **{p['observed_order']:.3f}**;
at {PLATE_GATE_MESH}x{PLATE_GATE_MESH} elements the error is
{p['levels'][6]['error_vs_navier_series']:.3%} against the series and
{p['levels'][6]['error_vs_timoshenko_00406']:.3%} against the tabulated 0.00406.
Convergence from below is the expected signature of a conforming displacement
element (Kirchhoff energy overestimates stiffness).

### fem_modal - frequencies

Max error over modes 1-5 at {MODAL_GATE_MESH} elements:
{max(m['levels'][2]['relative_errors']):.3e}; at 64 elements
{max(m['levels'][-1]['relative_errors']):.3e}.  Rates:
{', '.join(f"mode {k.split('_')[1]} {v:.2f}" for k, v in m['observed_orders'].items())}.
The bisection roots agree with the tabulated values (Blevins 1979) to
{m['gates'][1]['value']:.1e} relative, i.e. within the ~5 significant digits of the
source, and satisfy the characteristic equation to
{m['gates'][2]['value']:.1e}; the fifth root is {m['reference']['beta_l'][4]:.9f}
(14.137 to the four digits of the mission table).

## 4. Figures

* `benchmarks/verified/fem_beam/convergence.png` - L2 error vs `h` with `O(h^4)`
  guide, and the deflection profile FE vs exact.
* `benchmarks/verified/fem_plate/convergence.png` - centre-coefficient error vs
  `h` with the 2 % gate line, and the deflection contour at
  {PLATE_GATE_MESH}x{PLATE_GATE_MESH}.
* `benchmarks/verified/fem_modal/convergence.png` - per-mode frequency error vs
  `h` with the 1 % gate line, and the five mode shapes.

## 5. Caveats and limits

* The plate rung is **thin-plate only** (`a/h = 100`); it says nothing about
  transverse-shear effects, which the Mindlin Q4 path covers separately
  (`benchmarks/plate_uniform_load_evidence.json`).  The 2 % gate is applied at
  16x16 elements; coarser meshes (<= 8) are 2.3-9 % stiff and are reported, not
  hidden.
* The hard simple support fixes the tangential slope along each edge, which is
  the exact SSSS condition for a straight edge, but is *not* the "soft" support
  used by some codes; results should not be transferred to soft supports.
* The beam and modal rungs assume Euler-Bernoulli kinematics (no shear
  deformation, no rotary inertia); at `L/h = 100` the shear correction is
  ~1e-4 relative, i.e. below the reported errors, so the thin-beam comparison is
  self-consistent.
* Convergence of a discretisation is necessary, not sufficient, evidence of
  model validity; the anchors here are closed-form solutions of the *same*
  mathematical model, so they verify the implementation, not the model.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    beam = run_beam()
    plate = run_plate()
    modal = run_modal()

    meta = {
        "generated_by": "benchmarks/structural_ladder.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": _env(),
        "module": "tensorfem.structural_verification",
    }
    for name, payload in (("fem_beam", beam), ("fem_plate", plate), ("fem_modal", modal)):
        payload = dict(payload, **meta)
        outdir = OUT / name
        outdir.mkdir(parents=True, exist_ok=True)
        _write_json(outdir / "result.json", payload)
        if name == "fem_beam":
            fig_beam(payload, outdir)
            (outdir / "README.md").write_text(readme_beam(payload))
        elif name == "fem_plate":
            fig_plate(payload, outdir)
            (outdir / "README.md").write_text(readme_plate(payload))
        else:
            fig_modal(payload, outdir)
            (outdir / "README.md").write_text(readme_modal(payload))

    gate = next(row for row in beam["levels"] if row["elements"] == 8)
    anchors = [
        {
            "rung": "fem_beam",
            "case": "cantilever UDL",
            "quantity": "tip deflection w(L) [m]",
            "fem": gate["tip_deflection_m"],
            "reference": beam["reference"]["tip_deflection_m"],
            "relative_error": gate["tip_relative_error"],
            "passed": gate["tip_relative_error"] < 1e-2,
        },
        {
            "rung": "fem_plate",
            "case": "SSSS plate UDL",
            "quantity": "centre coefficient alpha",
            "fem": plate["levels"][6]["alpha"],
            "reference": sv.PLATE_ALPHA_TIMOSHENKO,
            "relative_error": plate["levels"][6]["error_vs_timoshenko_00406"],
            "passed": plate["levels"][6]["error_vs_timoshenko_00406"] < 2e-2,
        },
        {
            "rung": "fem_modal",
            "case": "cantilever modes",
            "quantity": f"frequency f_{MODES} [Hz] (worst of modes 1-5 at this mesh)",
            "fem": modal["levels"][2]["frequencies_hz"][MODES - 1],
            "reference": modal["reference"]["frequencies_hz"][MODES - 1],
            "relative_error": modal["levels"][2]["relative_errors"][MODES - 1],
            "passed": modal["levels"][2]["relative_errors"][MODES - 1] < 1e-2,
        },
    ]
    summary = {
        **meta,
        "rungs": {
            "fem_beam": beam["verdict"],
            "fem_plate": plate["verdict"],
            "fem_modal": modal["verdict"],
        },
        "gate_meshes": {"fem_beam": 8, "fem_plate": PLATE_GATE_MESH, "fem_modal": MODAL_GATE_MESH},
        "observations": {
            "beam_field_order": beam["observed_order_field_l2"],
            "plate_order": plate["observed_order"],
            "modal_orders": modal["observed_orders"],
        },
        "anchors": anchors,
        "verdict": "PASS" if all(v == "PASS" for v in
                                 (beam["verdict"], plate["verdict"], modal["verdict"])) else "FAIL",
    }
    _write_json(OUT / "structural_ladder_summary.json", summary)
    (DOCS / "STRUCTURAL_LADDER.md").write_text(docs_note(beam, plate, modal, summary))

    print(f"fem_beam  : {beam['verdict']}  order {beam['observed_order_field_l2']:.3f}")
    print(f"fem_plate : {plate['verdict']}  order {plate['observed_order']:.3f}  "
          f"alpha {plate['levels'][6]['alpha']:.6f}")
    print(f"fem_modal : {modal['verdict']}  worst {max(modal['levels'][2]['relative_errors']):.3e}")
    print(f"ladder    : {summary['verdict']}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())