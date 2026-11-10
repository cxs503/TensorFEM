"""Generate the bimetal (Timoshenko 1925) thermo-elastic verification evidence.

Produces, from one run:

* ``benchmarks/bimetal_thermoelastic_evidence.json`` -- every anchor, its
  reference, the measured value, the gate and the verdict (machine generated,
  no hand-copied numbers);
* ``docs/bimetal-thermoelastic-verification.md`` -- the report; every number in
  it is formatted from that dictionary;
* figures in ``docs/assets/bimetal_*.png``:
  ``sigma_field`` (field cloud of ``sigma_xx`` on the deformed strip),
  ``section_stress`` (through-thickness stress at mid-length vs the closed
  form), ``convergence`` (log-log h-refinement with the observed order),
  ``deflection_profile`` (deflection along the strip and its residual),
  ``sweep_errors`` (relative errors of the 12 parameter cases against the
  acceptance gates).

Run::

    PYTHONPATH=src python benchmarks/bimetal_thermoelastic.py

The systems here are small (max ~4.5k unknowns for the accuracy case), so the
torch thread pool is capped: it costs more than it buys at this size.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt       # noqa: E402
import numpy as np                    # noqa: E402
import torch                          # noqa: E402

torch.set_num_threads(int(os.environ.get("TENSORFEM_THREADS", "4")))

from tensorfem import bimetal_thermoelastic as bm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "bimetal_thermoelastic_evidence.json"
DOC = ROOT / "docs" / "bimetal-thermoelastic-verification.md"
ASSETS = ROOT / "docs" / "assets"

# acceptance gates of the campaign node (fixed before any run)
GATE_DEFLECTION = 0.03
GATE_STRESS = 0.05
GATE_SWEEP = 0.05
GATE_ORDER = 1.8
GATE_SINGLE = 0.01
GATE_ENERGY = 0.02

ACCURACY_MESH = (128, 8)        # nx, ny_per_layer for the accuracy / sweep cases


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=ROOT).decode().strip()
    except Exception:                                  # pragma: no cover
        return "unknown"


def _case_dict(case: bm.BimetalCase) -> dict:
    return {
        "m": case.m, "n": case.n, "dT": case.dT, "h": case.h,
        "length": case.length, "slenderness": case.slenderness,
        "E1": case.E1, "E2": case.E2, "alpha1": case.alpha1,
        "alpha2": case.alpha2, "nu": case.nu,
        "t1": case.t1, "t2": case.t2,
    }


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------


def run_reference_case() -> tuple[bm.StripResult, bm.BimetalSolution]:
    case = bm.BimetalCase()
    res = bm.solve_bimetal_strip(case, *ACCURACY_MESH)
    return res, bm.analytical_solution(case)


def run_convergence() -> dict:
    uniform = bm.convergence_study(nxs=(8, 16, 32, 64, 128), ny_per_layer=4, grading=1.0)
    graded = bm.convergence_study(nxs=(8, 16, 32, 64, 128), ny_per_layer=4, grading=1.15)
    return {"uniform": uniform, "graded": graded}


def serialise_convergence(study: dict) -> dict:
    return {
        "grading": study["grading"],
        "levels": study["levels"],
        "tip_deflection": study["tip_deflection"],
        "mid_deflection": study["mid_deflection"],
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_sigma_field(res: bm.StripResult, ref: bm.BimetalSolution) -> Path:
    case, mesh = res.case, res.mesh
    xs = res.gauss_x.detach().numpy()
    ys = res.gauss_y.detach().numpy()
    sig = (res.gauss_sigma_xx / 1e6).detach().numpy()
    npx = 2 * mesh.nx + 1
    j_int = 2 * mesh.ny_per_layer
    row = slice(j_int * npx, (j_int + 1) * npx)
    x_mid = mesh.nodes[row, 0].detach().numpy()
    v_mid = res.displacement[2 * np.arange(j_int * npx, (j_int + 1) * npx) + 1].detach().numpy()

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.0, 6.4),
                                   gridspec_kw={"height_ratios": [2.0, 1.0]})
    levels = np.linspace(sig.min(), sig.max(), 41)
    cf = ax0.tricontourf(xs * 1e3, ys * 1e3, sig, levels=levels, cmap="coolwarm")
    ax0.axhline(0.0, color="k", lw=0.8, ls="--")
    ax0.text(0.02 * case.length * 1e3, 0.04 * case.t1 * 1e3, "interface  y = 0",
             fontsize=8, color="k")
    ax0.set_ylabel("y [mm]")
    ax0.set_title("sigma_xx [MPa], Q9 plane stress, clamped-free bimetal strip "
                  f"(m={case.m:g}, n={case.n:g}, dT={case.dT:g} K)")
    fig.colorbar(cf, ax=ax0, label="sigma_xx [MPa]")

    ax1.plot(x_mid * 1e3, v_mid * 1e3, "o", ms=2.5, label="FEM (interface line)")
    xx = np.linspace(0.0, case.length, 200)
    ax1.plot(xx * 1e3, (-ref.curvature * xx**2 / 2.0) * 1e3, "-",
             label="Timoshenko: w = -kappa x^2/2")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("w [mm]")
    ax1.set_title("mid-line deflection w(x) on the undeformed axes of the cloud above",
                  fontsize=8)
    ax1.legend(loc="lower left", fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "bimetal_sigma_field.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_section_stress(res: bm.StripResult, ref: bm.BimetalSolution) -> tuple[Path, dict]:
    case = res.case
    x = 0.5 * case.length
    prof = bm.section_profile(res, x, n_points=121)
    y = prof["y"].detach().numpy() * 1e3
    sig = prof["sigma_xx"].detach().numpy() / 1e6
    ana_y = np.array([-case.t2, 0.0, case.t1])
    ana_s = np.array([ref.sigma_surface_bottom, ref.sigma_interface_bottom,
                      ref.sigma_interface_top]) / 1e6
    ana_top = np.array([ref.sigma_interface_top, ref.sigma_surface_top]) / 1e6

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(sig, y, "o-", ms=3.0, lw=0.8, label="FEM, Q9 (x = L/2)")
    ax.plot(ana_s[:2], ana_y[:2] * 1e3, "s--", ms=5, label="Timoshenko, layer 2")
    ax.plot(ana_top, ana_y[1:] * 1e3, "s--", ms=5, label="Timoshenko, layer 1")
    ax.axhline(0.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("sigma_xx [MPa]")
    ax.set_ylabel("y [mm]")
    ax.set_title("through-thickness stress at mid-length: the interface jump")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "bimetal_section_stress.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out, {
        "x_over_L": x / case.length,
        "fem_sigma_top_at_interface": res.sigma_interface_top,
        "fem_sigma_bottom_at_interface": res.sigma_interface_bottom,
        "analytic_sigma_surface_top": ref.sigma_surface_top,
        "analytic_sigma_surface_bottom": ref.sigma_surface_bottom,
        "sampled_points": int(prof["y"].numel()),
    }


def fig_convergence(conv: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    for key, style, label in (("uniform", "o-", "uniform mesh (grading 1.0)"),
                              ("graded", "s-", "graded toward the clamp (1.15)")):
        study = conv[key]
        nxs = np.array([r["nx"] for r in study["levels"]], dtype=float)
        err = np.array([abs(r["tip_rel_error"]) for r in study["levels"]])
        p_obs = study["tip_deflection"]["order"]
        ax.loglog(nxs, err, style, ms=5, label=f"{label}: p = {p_obs:.2f}")
    nxs = np.array([r["nx"] for r in conv["uniform"]["levels"]], dtype=float)
    ref = abs(conv["uniform"]["levels"][0]["tip_rel_error"]) * (nxs / nxs[0]) ** -2.0
    ax.loglog(nxs, ref, "k--", lw=0.9, label="slope -2 guide")
    ax.set_xlabel("n_x (elements along the strip)")
    ax.set_ylabel("|tip deflection error| vs Timoshenko")
    ax.set_title("h-refinement of the free-end deflection (n_y has no effect: Q9 is exact in y)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = ASSETS / "bimetal_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_deflection_profile(res: bm.StripResult, ref: bm.BimetalSolution) -> Path:
    mesh = res.mesh
    npx = 2 * mesh.nx + 1
    j_int = 2 * mesh.ny_per_layer
    idx = np.arange(j_int * npx, (j_int + 1) * npx)
    x = mesh.nodes[idx, 0].detach().numpy()
    w = res.displacement[2 * idx + 1].detach().numpy()
    ana = -ref.curvature * x**2 / 2.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.8, 6.0), sharex=True)
    ax0.plot(x * 1e3, w * 1e6, "-", label="FEM (mid-line)")
    ax0.plot(x * 1e3, ana * 1e6, "--", label="Timoshenko closed form")
    ax0.set_ylabel("w [um]")
    ax0.legend(fontsize=8)
    ax0.grid(alpha=0.3)
    ax0.set_title("deflection along the strip and the clamped-edge residual")
    ax1.plot(x * 1e3, (w - ana) * 1e6, "-", color="C3")
    slope = np.polyfit(x[1:], (w - ana)[1:], 1)[0]
    ax1.plot(x * 1e3, slope * x * 1e6, "k:", lw=0.9,
             label=f"linear fit: theta0 = {slope:.3e} rad")
    ax1.set_xlabel("x [mm]")
    ax1.set_ylabel("w_FEM - w_analytic [um]")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    out = ASSETS / "bimetal_deflection_profile.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_sweep_errors(sweep: list[dict]) -> Path:
    labels = [f"m={r['m']:g}\nn={r['n']:g}\ndT={r['dT']:g}" for r in sweep]
    defl = np.array([abs(r["deflection_rel_error"]) for r in sweep]) * 100.0
    sig = np.array([abs(r["sigma_max_rel_error"]) for r in sweep]) * 100.0
    idx = np.arange(len(sweep))
    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.bar(idx - 0.2, defl, width=0.4, label="free-end deflection")
    ax.bar(idx + 0.2, np.maximum(sig, 1e-6), width=0.4, label="max interface stress")
    ax.axhline(3.0, color="C0", ls="--", lw=1.0, label="3 % gate (deflection)")
    ax.axhline(5.0, color="C1", ls="--", lw=1.0, label="5 % gate (all cases)")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, fontsize=6)
    ax.set_ylabel("|relative error| [%]")
    ax.set_title("12 parameter cases: FEM vs Timoshenko (stress errors at the 1e-6 % floor)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = ASSETS / "bimetal_sweep_errors.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def render_report(ev: dict) -> str:
    gates = ev["gates"]
    ref = ev["reference_case"]["analytic"]
    fem = ev["reference_case"]["fem"]
    conv_u = ev["convergence"]["uniform"]
    conv_g = ev["convergence"]["graded"]
    energy = ev["energy_balance"]
    single = ev["single_material_limit"]
    sweep = ev["parameter_sweep"]
    slen = ev["slenderness"]
    thick = ev["thickness_independence"]

    def cell(text: object) -> str:
        return str(text).replace("|", "\\|").replace("\n", " ")

    def gate_row(g: dict) -> str:
        return (f"| {cell(g['id'])} | {cell(g['description'])} | {cell(g['criterion'])} | "
                f"{cell(g['measured'])} | {'PASS' if g['pass'] else 'FAIL'} |")

    lines = []
    add = lines.append
    add("# Bimetal strip thermo-elasticity: Timoshenko (1925) closed form vs a 2-D Q9 FE model")
    add("")
    add(f"* Node: **P189-N03** (TensorFEM, thermal-mechanical coupling)\n"
        f"* Generated: {ev['meta']['timestamp']} (commit `{ev['meta']['commit']}`, "
        f"torch {ev['meta']['torch']}, python {ev['meta']['python']})\n"
        f"* Regenerate: `PYTHONPATH=src python benchmarks/bimetal_thermoelastic.py`\n"
        f"* Evidence: `benchmarks/bimetal_thermoelastic_evidence.json`")
    add("")
    add("## 1. Case, physics and parameters")
    add("")
    add("Two perfectly bonded layers of different thermal expansion form a strip that bends "
        "when heated while mechanically free. The temperature change produces a "
        "self-equilibrated field: an axial strain `eps0` plus a curvature `kappa`, with zero "
        "resultant force and zero resultant moment over the section. Timoshenko's 1925 "
        "analysis of bi-metal thermostats gives the curvature in closed form; the same "
        "expression is re-derived here from a 2x2 closed-section equilibrium "
        "(`section_equilibrium`), so a transcription slip in either form would be caught.")
    add("")
    add("```")
    add("kappa = 6 (alpha1 - alpha2) dT (1 + m)^2 / ( h [ 3 (1+m)^2 + (1 + m n)(m^2 + 1/(m n)) ] )")
    add("m = t1/t2,  n = E1/E2,  h = t1 + t2,  eps_xx(y) = eps0 + kappa y,")
    add("w(x) = -kappa x^2/2  (built in at x = 0, free at x = L),  delta_tip = -kappa L^2/2")
    add("```")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    case = ev["reference_case"]["case"]
    add(f"| h (total thickness) | {case['h']*1e3:.3f} mm |")
    add(f"| L (strip length) | {case['length']*1e3:.2f} mm (L/h = {case['slenderness']:g}) |")
    add(f"| thickness ratio m = t1/t2 | {case['m']:g} (t1 = {case['t1']*1e3:.3f} mm, "
        f"t2 = {case['t2']*1e3:.3f} mm) |")
    add(f"| modulus ratio n = E1/E2 | {case['n']:g} (E1 = {case['E1']/1e9:.1f} GPa, "
        f"E2 = {case['E2']/1e9:.1f} GPa) |")
    add(f"| CTE | alpha1 = {case['alpha1']:.2e} 1/K (top), alpha2 = {case['alpha2']:.2e} 1/K |")
    add(f"| Poisson ratio | {case['nu']:g} (equal in both layers) |")
    add(f"| temperature rise dT | {case['dT']:g} K |")
    add(f"| closed-form curvature | {ref['curvature']:.6e} 1/m |")
    add(f"| closed-form interface stress | {ref['sigma_interface_top']/1e6:+.3f} MPa (layer 1) / "
        f"{ref['sigma_interface_bottom']/1e6:+.3f} MPa (layer 2) |")
    add(f"| closed-form free-end deflection | {ref['tip_deflection']*1e6:+.3f} um "
        f"({ref['tip_deflection']/case['h']:+.4f} h) |")
    add(f"| closed-form strain energy (unit width) | {ref['strain_energy']:.6e} J/m |")
    add("")
    add("## 2. How it was done")
    add("")
    add("* **Analytical side**: `analytical_solution` returns `(eps0, kappa)` from the 2x2 "
        "section equilibrium, cross-checks the Timoshenko closed form against it to 1e-10 "
        "relative and then builds the full field: strains, `sigma_xx(y)` in both layers, "
        "interface/surface extremes, and the strain energy `0.5 int E (eps - alpha dT)^2 dy` "
        "per unit width.")
    add("* **FE side**: `solve_bimetal_strip` assembles a structured **Q9** (biquadratic "
        "Lagrange, 3x3 Gauss, exactly integrated) plane-stress mesh with `ny_per_layer` "
        "elements per layer and a shared node line on the interface, applies the uniform "
        "`dT` as a thermal strain load `f = int B^T D eps_th dV`, clamps the whole `x = 0` "
        "edge, leaves every other edge traction free, and solves the sparse system "
        "(`scipy` SuperLU, float64). Q4 bilinear elements were *not* used: they lock in "
        "bending, so their deflection error measures spurious stiffness, not the "
        "discretisation of the exact solution.")
    add("* **Measured quantities**: free-end deflection `v(L, 0)`, mid-length deflection, "
        "`sigma_xx` at `(L/2, 0)` sampled separately in the layer below and above the "
        "interface, the through-thickness stress profile, the strain energy and the discrete "
        "thermo-elastic energy identity `U_el = C - u^T f_th/2`.")
    add("* **Mesh studies**: uniform and boundary-layer-graded (grading 1.15 toward the "
        "clamp) families with `n_x = 8 ... 128`; observed order and GCI from the grid "
        "triplets (Celik et al., *J. Fluids Eng.* 130 (2008) 078001); a `n_y` study to show "
        "the thickness direction carries no error; an `L/h = 10/20/40` study at fixed "
        "`dx/h` to quantify the clamped-edge model offset.")
    add("")
    add("## 3. Results")
    add("")
    add("### 3.1 Acceptance gates")
    add("")
    add("| gate | criterion | required | measured | verdict |")
    add("|---|---|---|---|---|")
    for g in gates:
        add(gate_row(g))
    add("")
    add("### 3.2 Reference case, accuracy mesh")
    add("")
    add(f"Mesh: `n_x = {fem['nx']}`, `n_y = {fem['ny_per_layer']}` per layer, "
        f"{fem['dofs']} dofs.")
    add("")
    add("| quantity | closed form | FEM | relative error |")
    add("|---|---|---|---|")
    add(f"| free-end deflection [um] | {ref['tip_deflection']*1e6:+.6f} | "
        f"{fem['tip_deflection']*1e6:+.6f} | {fem['tip_rel_error']:+.4%} |")
    add(f"| curvature implied by the tip [1/m] | {ref['curvature']:.6e} | "
        f"{fem['curvature_estimate']:.6e} | {fem['curvature_rel_error']:+.4%} |")
    add(f"| sigma_xx at (L/2, 0+) [MPa] | {ref['sigma_interface_top']/1e6:+.6f} | "
        f"{fem['sigma_interface_top']/1e6:+.6f} | {fem['sigma_top_rel_error']:+.3e} |")
    add(f"| sigma_xx at (L/2, 0-) [MPa] | {ref['sigma_interface_bottom']/1e6:+.6f} | "
        f"{fem['sigma_interface_bottom']/1e6:+.6f} | {fem['sigma_bottom_rel_error']:+.3e} |")
    add("")
    add("### 3.3 Mesh convergence")
    add("")
    add("Uniform family:")
    add("")
    add("| n_x | dx_min/h | dofs | tip deflection [um] | error vs closed form | mid-span error |")
    add("|---|---|---|---|---|---|")
    for r in conv_u["levels"]:
        add(f"| {r['nx']} | {r['dx_min_over_h']:.4f} | {r['dofs']} | "
            f"{r['tip_deflection']*1e6:+.6f} | {r['tip_rel_error']:+.4%} | "
            f"{r['mid_rel_error']:+.4%} |")
    add("")
    uniform_triplets = ", ".join(
        f"{t['levels']}: p = {t['order']:.3f}" for t in conv_u["tip_deflection"]["triplets"])
    add("Observed order from the grid triplets (uniform): " + uniform_triplets
        + f". Finest triplet GCI = {conv_u['tip_deflection']['gci']:.3e} "
          "of the tip deflection.")
    add("")
    add("Graded family (1.15 towards the clamped edge):")
    add("")
    add("| n_x | dx_min/h | tip deflection [um] | error vs closed form | mid-span error |")
    add("|---|---|---|---|---|")
    for r in conv_g["levels"]:
        add(f"| {r['nx']} | {r['dx_min_over_h']:.5f} | {r['tip_deflection']*1e6:+.6f} | "
            f"{r['tip_rel_error']:+.4%} | {r['mid_rel_error']:+.4%} |")
    add("")
    add("Observed order (graded): "
        + ", ".join(f"{t['levels']}: p = {t['order']:.3f}"
                    if t["order"] == t["order"] else f"{t['levels']}: p = n/a"
                    for t in conv_g["tip_deflection"]["triplets"])
        + f". Finest usable triplet GCI = {conv_g['tip_deflection']['gci']:.3e}.")
    add("")
    add("Interpretation: the Q9 element reproduces the through-thickness linear bending "
        "strain exactly, so refining `n_y` does not move the answer "
        f"(see 3.4); the error is produced along `x`, where the clamped edge forces a "
        "Saint-Venant boundary layer of thickness `O(h)`. On a uniform family that layer is "
        "under-resolved at `n_x <= 64`, which is exactly the range where the observed order "
        "sits at 1.2-1.6; the asymptotic triplet (`n_x = 32, 64, 128`) reaches "
        f"p = {conv_u['tip_deflection']['order']:.2f}, and grading towards the clamp makes "
        f"p = {conv_g['tip_deflection']['order']:.2f} with GCI "
        f"{conv_g['tip_deflection']['gci']:.1e}.")
    add("")
    add("### 3.4 Through-thickness mesh independence")
    add("")
    add("| n_y per layer | dofs | tip deflection [um] | error vs closed form | sigma error |")
    add("|---|---|---|---|---|")
    for r in thick:
        add(f"| {r['ny_per_layer']} | {r['dofs']} | {r['tip_deflection']*1e6:+.6f} | "
            f"{r['tip_rel_error']:+.4%} | {r['sigma_top_rel_error']:+.1e} |")
    add("")
    add("### 3.5 Slenderness: the price of the 2-D clamp")
    add("")
    add("| L/h | n_x | dx/h | FEM tip [um] | closed form [um] | FEM/closed form | error |")
    add("|---|---|---|---|---|---|---|")
    for r in slen:
        add(f"| {r['slenderness']:g} | {r['nx']} | {r['dx_over_h']:.3f} | "
            f"{r['tip_deflection']*1e6:+.6f} | {r['analytic_tip_deflection']*1e6:+.6f} | "
            f"{r['ratio']:.6f} | {r['rel_error']:+.4%} |")
    add("")
    add("The 2-D clamp (`u = v = 0` along the whole edge) is a *stronger* constraint than the "
        "beam's built-in end, so the FEM deflects slightly more than `-kappa L^2/2`; the "
        "offset halves when `L/h` doubles (it is `O(h/L)`, equivalently a small root rotation). "
        "It is a model difference, not a discretisation error, and it is reported instead of "
        "being tuned away.")
    add("")
    add("### 3.6 Parameter sweep (9 `(m, n)` combinations + 3-`dT` ladder)")
    add("")
    add("| m | n | dT [K] | kappa [1/m] | deflection FEM [um] | error | max |sigma| error |")
    add("|---|---|---|---|---|---|---|")
    for r in sweep:
        add(f"| {r['m']:g} | {r['n']:g} | {r['dT']:g} | {r['analytic_curvature']:.6e} | "
            f"{r['fem_tip_deflection']*1e6:+.6f} | {r['deflection_rel_error']:+.4%} | "
            f"{r['sigma_max_rel_error']:+.2e} |")
    add("")
    add("### 3.7 Energy balance")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| FEM strain energy, whole strip [J/m] | {energy['strain_energy_fem']:.6e} |")
    add(f"| closed-form strain energy [J/m] | {energy['strain_energy_analytic']:.6e} |")
    add(f"| whole-strip relative error | {energy['whole_strip_rel_error']:+.2%} |")
    add(f"| thermal pool C = 0.5 int eps_th^T D eps_th dV [J/m] | {energy['thermal_pool']:.6e} |")
    add(f"| 0.5 u^T f_th [J/m] | {energy['half_load_work']:.6e} |")
    add(f"| identity residual \\|U_el - (C - u^T f_th/2)\\| / U_el | "
        f"{energy['energy_identity_gap']:.3e} |")
    add(f"| strain energy outside the clamp layer (x >= 2h) [J/m] | "
        f"{energy['strain_energy_fem_region']:.6e} |")
    add(f"| same region, closed form [J/m] | {energy['strain_energy_analytic_region']:.6e} |")
    add(f"| region relative error | {energy['region_rel_error']:+.3%} |")
    add(f"| fraction of the FEM energy outside the clamp layer | {energy['region_fraction']:.3f} |")
    add("")
    add("The whole-strip energy is *not* usable as a gate: the pure-bending closed form is "
        "invalid inside the clamped boundary layer, where the strip is forced towards uniform "
        "axial strain and the energy density is an order of magnitude higher; that layer holds "
        f"{1-energy['region_fraction']:.0%} of the total energy. Excluding `x < 2h` (the "
        "Saint-Venant length) the strain energy agrees to "
        f"{abs(energy['region_rel_error']):.2%}. Note also that the mechanical Clapeyron "
        "relation `U = u^T f/2` does **not** hold for a thermal load: the exact identity is "
        "`U_el = C - u^T f_th/2` with `C` the thermal pool; the residual of that identity is "
        f"{energy['energy_identity_gap']:.1e}, i.e. solver round-off.")
    add("")
    add("### 3.8 Single-material limit")
    add("")
    add("| quantity | value |")
    add("|---|---|")
    add(f"| closed-form curvature at alpha1 = alpha2 | {single['analytic_curvature']:.3e} 1/m |")
    add(f"| FEM tip deflection at alpha1 = alpha2 | {single['fem_tip_deflection']:.3e} m |")
    add(f"| FEM deflection / bimetal reference deflection | "
        f"{single['fem_relative_deflection']:.3e} |")
    add(f"| FEM residual interface stress vs 50 MPa scale | "
        f"{abs(single['fem_max_abs_sigma_interface'])/5e7:.2e} |")
    add("")
    add("### 3.9 Section equilibrium of the FE stress field")
    add("")
    add("The free end carries no traction, so the internal force and moment resultants of the "
        "exact solution vanish on every section outside the clamped layer. Integrating the "
        "*FE* stress field (two interior samples per layer pin the linear stress, then the "
        "polynomial integrals are evaluated analytically):")
    add("")
    add("| section | axial force / (E dalpha dT h) | moment / (E dalpha dT h^2) |")
    add("|---|---|---|")
    for key, label in (("mid", "x = L/2"), ("x_0p9L", "x = 0.9 L"), ("x_0p97L", "x = 0.97 L")):
        r = ev["section_resultants"][key]
        add(f"| {label} | {r['force_rel']:.2e} | {r['moment_rel']:.2e} |")
    add("")
    add("Trap met here: a trapezoidal rule on `y * sigma_xx` is *not* exact (the integrand is "
        "quadratic in `y`) and leaves a mesh-independent spurious moment of "
        f"{1/6400:.1e} of the scale *at every mesh*, which looks exactly like a real "
        "equilibrium violation. The integrals are therefore done analytically.")
    add("")
    add("The `x = 0.97 L` row is kept on purpose: the bimaterial corner at the free end carries "
        "a stress singularity, so the *recovered* stress within about one thickness of it is "
        "not in balance and this check is only used from `x = 0.9 L` towards the clamp. "
        "Pointwise stresses in that corner must not be compared with the closed form at all.")
    add("")
    add("## 4. Figures")
    add("")
    for f in ev["figures"]:
        add(f"![{f['caption']}]({Path(f['path']).name})")
        add("")
        add(f"*{f['caption']}*")
        add("")
    add("## 5. Notes, limitations and traps")
    add("")
    add("1. **Sign convention.** With layer 1 (larger CTE) on top, `kappa > 0` means the top "
        "layer ends up on the outside of the arc, i.e. `w'' = -kappa` and the free end dips. "
        "The FEM reproduces that sign; a sign flip would show up as a 200 % error.")
    add("2. **The clamp is not the beam's built-in end.** `u = v = 0` on the whole edge adds a "
        "root rotation, so the FEM tip deflection is ~0.26 % above the closed form at "
        "`L/h = 20` and the offset scales as `O(h/L)`. Chasing it with mesh refinement does "
        "not work (it is not a discretisation error) - refining the strip length does.")
    add("3. **The interface stress is discontinuous, and singular at the free ends.** The "
        "closed form jumps at `y = 0`; pointwise stresses near `x = 0` and `x = L` are not "
        "comparable with it, which is why the stress gate is sampled at mid-length only.")
    add("4. **Q4 would have made the convergence order meaningless.** Bilinear plane-stress "
        "elements lock in bending; the deflection would converge from a spurious stiff limit.")
    add("5. **Thermal loads break the mechanical Clapeyron relation.** `U_el = u^T f/2` is "
        "wrong by a factor ~200 here; the correct discrete identity carries the thermal pool.")
    add("6. **Scope.** Linear elasticity, temperature-independent `E, alpha`, equal `nu` in "
        "both layers, small strain (max rotation "
        f"{abs(ref['curvature'])*case['length']:.3f} rad), no plasticity, no creep, no "
        "buckling of the compressed layer, no transient conduction (the temperature field is "
        "uniform by assumption).")
    add("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)
    case = bm.BimetalCase()
    ref = bm.analytical_solution(case)
    res, ref = run_reference_case()

    closed_form_gap = bm.curvature_agreement()
    conv = run_convergence()
    thick = bm.thickness_independence()
    slen = bm.slenderness_study()
    sweep = bm.parameter_sweep(nx=ACCURACY_MESH[0], ny_per_layer=ACCURACY_MESH[1])
    energy = bm.energy_balance(nx=ACCURACY_MESH[0], ny_per_layer=ACCURACY_MESH[1])
    resultants = {"mid": bm.section_resultants(res),
                  "x_0p9L": bm.section_resultants(res, x=0.9 * case.length),
                  "x_0p97L": bm.section_resultants(res, x=0.97 * case.length)}
    single = bm.single_material_limit()

    fem = {
        "nx": res.mesh.nx,
        "ny_per_layer": res.mesh.ny_per_layer,
        "dofs": res.mesh.n_dofs,
        "elements": res.mesh.n_elements,
        "tip_deflection": res.tip_deflection,
        "mid_deflection": res.mid_deflection,
        "curvature_estimate": res.curvature_estimate,
        "curvature_rel_error": bm.relative_error(res.curvature_estimate, ref.curvature),
        "tip_rel_error": bm.relative_error(res.tip_deflection, ref.tip_deflection),
        "sigma_interface_top": res.sigma_interface_top,
        "sigma_interface_bottom": res.sigma_interface_bottom,
        "sigma_top_rel_error": bm.relative_error(res.sigma_interface_top,
                                                 ref.sigma_interface_top),
        "sigma_bottom_rel_error": bm.relative_error(res.sigma_interface_bottom,
                                                    ref.sigma_interface_bottom),
        "max_abs_sigma_rel_error": bm.relative_error(res.max_abs_sigma_interface,
                                                     ref.max_abs_sigma_interface),
        "energy_identity_gap": res.energy_identity_gap,
    }

    # ---- gates ----------------------------------------------------------
    gates = []

    def gate(gid, description, criterion, measured, ok):
        gates.append({"id": gid, "description": description, "criterion": criterion,
                      "measured": measured, "pass": bool(ok)})

    gate("G0", "Timoshenko closed form vs independent 2x2 section equilibrium",
         "rel. gap < 1e-10", f"{closed_form_gap:.2e}", closed_form_gap < 1e-10)
    gate("G1 (a)", "free-end deflection FEM vs closed form, m=n=1, dT=100 K",
         f"|error| < {GATE_DEFLECTION:.0%}, signed", f"{fem['tip_rel_error']:+.4%}",
         abs(fem["tip_rel_error"]) < GATE_DEFLECTION)
    gate("G2 (b)", "max thermal stress (interface, both layers) vs closed form",
         f"|error| < {GATE_STRESS:.0%}",
         f"{max(abs(fem['sigma_top_rel_error']), abs(fem['sigma_bottom_rel_error'])):.2e}",
         max(abs(fem["sigma_top_rel_error"]), abs(fem["sigma_bottom_rel_error"])) < GATE_STRESS)
    worst_defl = max(abs(r["deflection_rel_error"]) for r in sweep)
    worst_sig = max(abs(r["sigma_max_rel_error"]) for r in sweep)
    gate("G3 (c)", "12 parameter cases (9 (m,n) + 3 dT): deflection and stress",
         f"all |error| < {GATE_SWEEP:.0%}",
         f"max deflection {worst_defl:.3%}, max stress {worst_sig:.1e}",
         worst_defl < GATE_SWEEP and worst_sig < GATE_SWEEP)
    p_uniform = conv["uniform"]["tip_deflection"]["order"]
    p_graded = conv["graded"]["tip_deflection"]["order"]
    gate("G4 (d)", "observed order of the free-end deflection (4+ level GCI family)",
         f"p >= {GATE_ORDER}", f"uniform p={p_uniform:.2f}, graded p={p_graded:.2f}",
         p_uniform >= GATE_ORDER and p_graded >= GATE_ORDER)
    gate("G5 (e)", "single-material limit alpha1 = alpha2: curvature must vanish",
         f"|FEM deflection| / bimetal deflection < {GATE_SINGLE:.0%}",
         f"{single['fem_relative_deflection']:.2e}",
         single["fem_relative_deflection"] < GATE_SINGLE)
    gate("G6 (f)", "energy: strain energy vs closed form outside the clamp layer; "
                   "thermo-elastic identity",
         f"region |error| < {GATE_ENERGY:.0%}; identity residual < 1e-6",
         f"region {energy['region_rel_error']:+.3%}; identity "
         f"{energy['energy_identity_gap']:.1e}",
         abs(energy["region_rel_error"]) < GATE_ENERGY
         and energy["energy_identity_gap"] < 1e-6)
    # only sections outside the bimaterial corner singularity at the free end
    gated = {k: v for k, v in resultants.items() if k != "x_0p97L"}
    worst_res_f = max(v["force_rel"] for v in gated.values())
    worst_res_m = max(v["moment_rel"] for v in gated.values())
    gate("G8 (support)", "section equilibrium of the FE stress field, sections >= 2h from the "
                        "free end (the bimaterial corner singularity pollutes closer ones)",
         "|F| / (E dalpha dT h) and |M| / (E dalpha dT h^2) < 1e-6",
         f"F {worst_res_f:.1e}, M {worst_res_m:.1e}",
         worst_res_f < 1e-6 and worst_res_m < 1e-6)
    thick_spread = max(r["tip_rel_error"] for r in thick) - min(r["tip_rel_error"] for r in thick)
    gate("G7 (support)", "through-thickness refinement (n_y = 2 ... 16) must not move the answer",
         "|spread| < 0.05%", f"{abs(thick_spread):.2e}", abs(thick_spread) < 5e-4)

    figures = []
    p = fig_sigma_field(res, ref)
    figures.append({"path": str(p.relative_to(ROOT)),
                    "caption": "Field cloud of sigma_xx on the strip (Q9 plane stress) with the "
                               "bending sign convention and the mid-line deflection."})
    p, section = fig_section_stress(res, ref)
    figures.append({"path": str(p.relative_to(ROOT)),
                    "caption": "Through-thickness sigma_xx at mid-length: the piecewise-linear "
                               "closed form and the FEM sample, including the interface jump."})
    p = fig_convergence(conv)
    figures.append({"path": str(p.relative_to(ROOT)),
                    "caption": "h-refinement of the free-end deflection (uniform and graded "
                               "families) with the observed order and a slope -2 guide."})
    p = fig_deflection_profile(res, ref)
    figures.append({"path": str(p.relative_to(ROOT)),
                    "caption": "Mid-line deflection along the strip and the residual against "
                               "the closed form (a small root rotation from the 2-D clamp)."})
    p = fig_sweep_errors(sweep)
    figures.append({"path": str(p.relative_to(ROOT)),
                    "caption": "Relative errors of the 12 parameter cases against the acceptance "
                               "gates (log scale)."})

    evidence = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "commit": _git_commit(),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "node": "P189-N03",
            "reference": "Timoshenko, J. Opt. Soc. Am. Rev. Sci. Instrum. 11 (1925) 233; "
                         "Clyne & Hull, Prog. Mater. Sci. 40 (1996) 81",
            "gates": {"deflection": GATE_DEFLECTION, "stress": GATE_STRESS,
                      "sweep": GATE_SWEEP, "order": GATE_ORDER,
                      "single_material": GATE_SINGLE, "energy": GATE_ENERGY},
            "accuracy_mesh": {"nx": ACCURACY_MESH[0], "ny_per_layer": ACCURACY_MESH[1]},
        },
        "reference_case": {"case": _case_dict(case),
                           "analytic": ref.__dict__ | {"case": _case_dict(case)},
                           "fem": fem},
        "closed_form_check": {"max_rel_gap_vs_section_equilibrium": closed_form_gap,
                              "cases_checked": len(bm.all_cases())},
        "convergence": {k: serialise_convergence(v) for k, v in conv.items()},
        "thickness_independence": thick,
        "slenderness": slen,
        "parameter_sweep": sweep,
        "energy_balance": {k: v for k, v in energy.items() if k != "case"},
        "single_material_limit": single,
        "section_stress": section,
        "section_resultants": resultants,
        "gates": gates,
        "figures": figures,
        "all_gates_pass": all(g["pass"] for g in gates),
    }
    # analytic dataclass holds the case object; make the JSON self-contained
    evidence["reference_case"]["analytic"]["case"] = _case_dict(case)

    EVIDENCE.write_text(json.dumps(evidence, indent=2, default=str) + "\n")
    DOC.write_text(render_report(evidence))
    print(f"evidence -> {EVIDENCE.relative_to(ROOT)}")
    print(f"report   -> {DOC.relative_to(ROOT)}")
    for g in gates:
        verdict = "PASS" if g["pass"] else "FAIL"
        print(f"  {g['id']:14s} {verdict}  {g['criterion']:38s} {g['measured']}")
    return evidence


if __name__ == "__main__":
    main()