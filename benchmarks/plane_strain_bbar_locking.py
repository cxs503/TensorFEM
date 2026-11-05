"""Evidence + figures for the 2-D plane-strain near-incompressible locking closure.

Runs the oracle self-check, the element-level prerequisites (patch consistency,
rigid-body null modes), the Poisson sweep of the standard Q4 against the new
mean-dilatation Q4, and the refinement study at ``nu = 0.4999``.  Writes
``benchmarks/plane_strain_bbar_locking_evidence.json`` and PNGs under
``docs/assets/plane_strain_bbar_locking/``.

    PYTHONPATH=src python3 benchmarks/plane_strain_bbar_locking.py
"""
from __future__ import annotations

import json
import os

import torch

# The dense Q4 assembly + rank SVDs are tiny problems: on a 128-core host the
# default OpenMP fan-out costs more than the arithmetic, so cap the threads.
torch.set_num_threads(int(os.environ.get("TENSORFEM_THREADS", "8")))

from tensorfem.continuum import solve_continuum
from tensorfem.lame_reference import cartesian_to_polar_stress, lame_state, radial_component
from tensorfem.plane_strain_locking import (DEFAULT_POISSONS, GEOMETRY, observed_order,
                                           patch_test, plane_locking_sweep,
                                           plane_refinement_study, reference_self_check,
                                           rigid_body_check, solve_case)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets", "plane_strain_bbar_locking")
EVIDENCE = os.path.join(ROOT, "benchmarks", "plane_strain_bbar_locking_evidence.json")

COARSE = dict(n_theta=16, n_radial=8)
REFINEMENT = ((8, 4), (16, 8), (24, 12), (32, 16), (48, 24))
GATE = 0.03  # 3 % acceptance gate used throughout the project


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    return float(value)


def _table(records, keys):
    return [{"poisson_ratio": r["poisson_ratio"], "formulation": r["formulation"],
             "n_theta": r["n_theta"], "n_radial": r["n_radial"], "dofs": r["dofs"],
             "elements": r["elements"]} | {k: r["errors"][k] for k in keys}
            for r in records]


def _constraint_table(records):
    seen = {}
    for record in records:
        if record["n_theta"] in seen:
            continue
        counts = record["volumetric_constraints"]
        seen[record["n_theta"]] = {"n_theta": record["n_theta"], "n_radial": record["n_radial"],
                                   "dofs": record["dofs"], "elements": record["elements"],
                                   "standard": counts["standard"], "bbar": counts["bbar"]}
    return list(seen.values())


def _plot_sweep(sweep, gate, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), sharex=True)
    styles = {"standard": ("o-", "standard Q4 (2x2 Gauss)"),
              "bbar": ("s-", "mean-dilatation Q4 (B-bar)")}
    for ax, key, label in ((axes[0], "displacement_l2", "radial displacement $L_2$ error [%]"),
                           (axes[1], "radial_stress_l2", "radial stress $L_2$ error [%]")):
        for formulation, (style, legend) in styles.items():
            pts = [r for r in sweep if r["formulation"] == formulation]
            ax.plot([r["poisson_ratio"] for r in pts], [100.0 * r["errors"][key] for r in pts],
                    style, label=legend)
        ax.axhline(100.0 * gate, color="k", ls=":", lw=1.0, label="3 % gate")
        ax.set_xlabel(r"Poisson ratio $\nu$")
        ax.set_ylabel(label)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_title(f"same mesh: {sweep[0]['n_theta']}x{sweep[0]['n_radial']} Q4, "
                      f"{sweep[0]['dofs']} DOF")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _plot_profiles(nu, geometry, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = {f: solve_case(nu, formulation=f, **COARSE) for f in ("standard", "bbar")}
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    radius = torch.linspace(geometry["inner_radius"], geometry["outer_radius"], 200, dtype=torch.float64)
    exact = lame_state(radius, inner_radius=geometry["inner_radius"], outer_radius=geometry["outer_radius"],
                       inner_pressure=geometry["inner_pressure"], young_modulus=geometry["young_modulus"],
                       poisson_ratio=nu, plane="strain")
    for ax, exact_curve, label, index in (
            (axes[0], exact.radial_stress, r"$\sigma_r$ [MPa]", 0),
            (axes[1], exact.hoop_stress, r"$\sigma_\theta$ [MPa]", 1)):
        ax.plot(radius, exact_curve / 1e6, "k-", lw=2.0, label="exact Lamé")
        for formulation, marker in (("standard", "o"), ("bbar", "s")):
            record = records[formulation]
            centres = record["node_coordinates"][record["element_connectivity"]].mean(1)
            sigma_r, sigma_theta, _ = cartesian_to_polar_stress(record["stress"], centres)
            values = (sigma_r, sigma_theta)[index]
            order = torch.argsort(torch.linalg.vector_norm(centres, dim=1))
            ax.plot(torch.linalg.vector_norm(centres, dim=1)[order], values[order] / 1e6,
                    marker, ms=4, mfc="none", label=f"{formulation} (element centres)")
        ax.set_xlabel("radius $r$ [m]")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_title(rf"$\nu = {nu}$ field comparison (16x8 Q4)")
    axes[0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _plot_convergence(refinement, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    for formulation, marker in (("standard", "o-"), ("bbar", "s-")):
        pts = [r for r in refinement if r["formulation"] == formulation]
        dofs = [r["dofs"] for r in pts]
        axes[0].plot(dofs, [100.0 * r["errors"]["displacement_l2"] for r in pts], marker,
                     label=f"{formulation} displacement")
        axes[0].plot(dofs, [100.0 * r["errors"]["radial_stress_l2"] for r in pts], marker,
                     mfc="none", label=f"{formulation} radial stress")
        axes[1].plot(dofs, [(r["volumetric_constraints"]["bbar"] or float("nan")) / r["dofs"]
                            for r in pts], marker, label=f"{formulation}: mean dilatation")
        axes[1].plot(dofs, [(r["volumetric_constraints"]["standard"] or float("nan")) / r["dofs"]
                            for r in pts], marker, mfc="none",
                     label=f"{formulation}: 2x2 Gauss dilatation")
    axes[0].axhline(100.0 * GATE, color="k", ls=":", lw=1.0, label="3 % gate")
    axes[0].set_xlabel("degrees of freedom")
    axes[0].set_ylabel(r"$L_2$ error [%]")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title(r"refinement at $\nu = 0.4999$")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].set_xlabel("degrees of freedom")
    axes[1].set_ylabel("independent volumetric constraints / DOF")
    axes[1].set_xscale("log")
    axes[1].set_title("why: constraint count")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    references = {str(nu): reference_self_check(nu) for nu in (0.3, 0.4999)}
    prerequisites = {"patch": patch_test(), "rigid": rigid_body_check()}
    sweep = plane_locking_sweep(n_theta=COARSE["n_theta"], n_radial=COARSE["n_radial"])
    refinement = plane_refinement_study(0.4999, meshes=REFINEMENT)

    orders = {}
    for formulation in ("standard", "bbar"):
        pts = [r for r in refinement if r["formulation"] == formulation]
        last, previous = pts[-1], pts[-2]
        ratio = last["n_theta"] / previous["n_theta"]
        orders[formulation] = {key: observed_order(previous["errors"][key], last["errors"][key], ratio)
                               for key in ("displacement_l2", "radial_stress_l2")}

    flatness = {}
    for formulation in ("standard", "bbar"):
        pts = {r["poisson_ratio"]: r["errors"] for r in sweep if r["formulation"] == formulation}
        flatness[formulation] = {
            key: pts[0.4999][key] / pts[0.3][key] for key in ("displacement_l2", "radial_stress_l2")}

    evidence = {
        "case": {
            "problem": "thick cylinder under internal pressure, plane strain",
            "geometry": {k: v for k, v in GEOMETRY.items()},
            "mesh": {"n_theta": COARSE["n_theta"], "n_radial": COARSE["n_radial"],
                     "elements": sweep[0]["elements"], "dofs": sweep[0]["dofs"]},
            "formulations": ["standard Q4 (2x2 Gauss)", "mean-dilatation Q4 (B-bar)"],
            "reference": "closed-form Lame (plane strain), stress state Poisson independent",
            "gate": GATE,
        },
        "reference_self_check": references,
        "element_prerequisites": prerequisites,
        "poisson_sweep": _table(sweep, ("displacement_l2", "radial_stress_l2", "hoop_stress_l2",
                                        "bore_displacement_relative")),
        "refinement_nu_4999": _table(refinement, ("displacement_l2", "radial_stress_l2",
                                                  "hoop_stress_l2", "bore_displacement_relative")),
        "observed_order_last_pair": orders,
        "error_ratio_0p4999_over_0p3": flatness,
        "volumetric_constraints": _constraint_table(refinement),
        "verdict": {
            "bbar_locking_free": flatness["bbar"]["radial_stress_l2"] < 1.01,
            "bbar_meets_gate_at_16x8_displacement": sweep[-1]["errors"]["displacement_l2"] < GATE,
            "bbar_meets_gate_at_16x8_radial_stress": sweep[-1]["errors"]["radial_stress_l2"] < GATE,
            "standard_q4_still_locks": sweep[-2]["errors"]["radial_stress_l2"] > 0.5,
        },
    }
    with open(EVIDENCE, "w") as handle:
        json.dump(_plain(evidence), handle, indent=2, sort_keys=True)

    _plot_sweep(sweep, GATE, os.path.join(ASSETS, "error_vs_poisson.png"))
    _plot_profiles(0.4999, GEOMETRY, os.path.join(ASSETS, "stress_profile_nu_4999.png"))
    _plot_convergence(refinement, os.path.join(ASSETS, "convergence_nu_4999.png"))

    print(json.dumps(_plain({"sweep": evidence["poisson_sweep"],
                             "refinement": evidence["refinement_nu_4999"],
                             "orders": orders, "flatness": flatness,
                             "verdict": evidence["verdict"]}), indent=2))


if __name__ == "__main__":
    main()