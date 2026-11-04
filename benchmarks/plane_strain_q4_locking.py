"""Regenerate the plane-strain Q4 locking qualification (tables, figures, JSON).

Run with ``PYTHONPATH=src python3 benchmarks/plane_strain_q4_locking.py``.
All numbers in ``docs/plane-strain-q4-locking.md`` come from this script; the
figures are written to ``docs/assets/plane_strain_q4_locking/`` and the scalars to
``benchmarks/plane_strain_q4_locking_evidence.json``.
"""
from __future__ import annotations

import json
import math
import os
import pathlib

import torch

# keep BLAS civilised on a shared, heavily loaded host
torch.set_num_threads(int(os.environ.get("TENSORFEM_THREADS", "8")))

from tensorfem.plane_strain_locking import (DEFAULT_GEOMETRY, DEFAULT_POISSONS,  # noqa: E402
                                            FORMULATIONS, flatness, locking_sweep,
                                            observed_order, reference_self_check,
                                            refinement_study, solve_case)

ROOT = pathlib.Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets" / "plane_strain_q4_locking"
EVIDENCE = ROOT / "benchmarks" / "plane_strain_q4_locking_evidence.json"
SECTOR = (16, 8)
MESHES = ((8, 4), (16, 8), (32, 16), (64, 32))
FINE_POISSON = 0.4999


def _table(records, metric, formulation):
    return {r["poisson_ratio"]: r["errors"][metric] for r in records if r["formulation"] == formulation}


def main() -> dict:
    ASSETS.mkdir(parents=True, exist_ok=True)

    self_check = {str(nu): reference_self_check(nu) for nu in (0.3, 0.45, 0.499, 0.4999)}
    sweep = locking_sweep(DEFAULT_POISSONS, n_theta=SECTOR[0], n_radial=SECTOR[1])
    refinement = refinement_study(FINE_POISSON, meshes=MESHES)

    orders = {}
    for metric in ("displacement_l2", "radial_stress_l2", "hoop_stress_l2", "mean_stress_l2"):
        for formulation in FORMULATIONS:
            try:
                orders[f"{metric}/{formulation}"] = observed_order(refinement, metric, formulation)
            except ValueError:
                orders[f"{metric}/{formulation}"] = float("nan")

    # field profiles on the reference mesh, for the field figure
    profiles = {}
    for formulation in FORMULATIONS:
        record = solve_case(FINE_POISSON, n_theta=SECTOR[0], n_radial=SECTOR[1],
                            formulation=formulation)
        centres = record["nodes"][record["cells"]].mean(1)
        from tensorfem.lame_reference import cartesian_to_polar_stress

        sigma_r, sigma_theta, _ = cartesian_to_polar_stress(record["stress"], centres)
        profiles[formulation] = {
            "radius": torch.linalg.vector_norm(centres, dim=1).tolist(),
            "radial_stress": sigma_r.tolist(),
            "hoop_stress": sigma_theta.tolist(),
        }
    radii = torch.linspace(DEFAULT_GEOMETRY["inner_radius"], DEFAULT_GEOMETRY["outer_radius"], 200,
                           dtype=torch.float64)
    from tensorfem.lame_reference import lame_state

    exact = lame_state(radii, inner_radius=DEFAULT_GEOMETRY["inner_radius"],
                       outer_radius=DEFAULT_GEOMETRY["outer_radius"],
                       inner_pressure=DEFAULT_GEOMETRY["pressure"],
                       young_modulus=DEFAULT_GEOMETRY["young_modulus"],
                       poisson_ratio=FINE_POISSON, plane="strain")

    evidence = {
        "case": {k: v for k, v in DEFAULT_GEOMETRY.items()},
        "sector_mesh": {"n_theta": SECTOR[0], "n_radial": SECTOR[1]},
        "poissons": list(DEFAULT_POISSONS),
        "fine_poisson": FINE_POISSON,
        "meshes": [list(m) for m in MESHES],
        "reference_self_check": self_check,
        "locking_sweep": sweep,
        "refinement": refinement,
        "observed_orders": orders,
        "flatness_displacement_l2": {f: flatness(sweep, "displacement_l2", f) for f in FORMULATIONS},
        "flatness_radial_stress_l2": {f: flatness(sweep, "radial_stress_l2", f) for f in FORMULATIONS},
    }
    EVIDENCE.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    _figures(sweep, refinement, profiles, exact, radii, orders)

    print(f"wrote {EVIDENCE.relative_to(ROOT)} and figures in {ASSETS.relative_to(ROOT)}")
    _print_tables(sweep, refinement, self_check, orders, evidence)
    return evidence


def _print_tables(sweep, refinement, self_check, orders, evidence):
    print("\nreference self-check")
    for nu, values in self_check.items():
        print(f"  nu={nu:>7} eq={values['equilibrium']:.2e} trac={values['traction']:.2e} "
              f"cons={values['constitutive']:.2e} vol={values['volumetric_strain_over_strain_scale']:.3e}")
    print("\nlocking sweep (%d x %d sector)" % SECTOR)
    print(f"{'nu':>8} {'form':>9} {'disp':>10} {'sig_r':>10} {'sig_t':>10} {'mean':>10} {'bore':>10}")
    for r in sweep:
        e = r["errors"]
        print(f"{r['poisson_ratio']:>8} {r['formulation']:>9} "
              f"{100*e['displacement_l2']:>10.4f} {100*e['radial_stress_l2']:>10.4f} "
              f"{100*e['hoop_stress_l2']:>10.4f} {100*e['mean_stress_l2']:>10.4f} "
              f"{100*e['bore_displacement_relative']:>10.4f}")
    print(f"\nrefinement at nu={FINE_POISSON}")
    print(f"{'mesh':>8} {'dofs':>6} {'form':>9} {'disp':>10} {'sig_r':>10} {'mean':>10}")
    for r in refinement:
        e = r["errors"]
        print(f"{r['n_theta']:>3}x{r['n_radial']:<4} {r['dofs']:>6} {r['formulation']:>9} "
              f"{100*e['displacement_l2']:>10.4f} {100*e['radial_stress_l2']:>10.4f} "
              f"{100*e['mean_stress_l2']:>10.4f}")
    print("\nobserved orders")
    for key, value in orders.items():
        print(f"  {key:38s} {value:6.3f}")
    print("\nflatness over nu sweep (max/min)")
    for key in ("flatness_displacement_l2", "flatness_radial_stress_l2"):
        print(f"  {key}: " + ", ".join(f"{f}={v:.2f}" for f, v in evidence[key].items()))


def _figures(sweep, refinement, profiles, exact, radii, orders):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"standard": "#d62728", "bbar": "#1f77b4", "sri": "#2ca02c"}
    markers = {"standard": "o", "bbar": "s", "sri": "^"}
    labels = {"standard": "standard Q4 (full integration)",
              "bbar": "B-bar Q4 (volumetric projection)",
              "sri": "Q4 SRI (1-pt volumetric)"}

    # (1) error vs Poisson ratio
    for metric, filename, ylabel in (
            ("displacement_l2", "error_vs_poisson.png", "relative $L_2$ error (%), displacement"),
            ("radial_stress_l2", "stress_error_vs_poisson.png", "relative $L_2$ error (%), $\\sigma_r$")):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for formulation in FORMULATIONS:
            values = _table(sweep, metric, formulation)
            ax.semilogy(list(values), [100 * v for v in values.values()], marker=markers[formulation],
                        color=colours[formulation], label=labels[formulation], lw=1.8)
        ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % gate")
        ax.set_xlabel(r"Poisson ratio $\nu$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Plane-strain Q4, $\\nu$-sweep ({SECTOR[0]}x{SECTOR[1]} sector, 306 DOF)")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(ASSETS / filename, dpi=160)
        plt.close(fig)

    # (2) mesh refinement at the incompressible limit
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for ax, metric, title in ((axes[0], "displacement_l2", "displacement $L_2$"),
                              (axes[1], "radial_stress_l2", "radial stress $L_2$")):
        for formulation in FORMULATIONS:
            subset = sorted((r for r in refinement if r["formulation"] == formulation),
                            key=lambda r: r["dofs"])
            ax.loglog([r["dofs"] for r in subset], [100 * r["errors"][metric] for r in subset],
                      marker=markers[formulation], color=colours[formulation],
                      label=labels[formulation], lw=1.8)
        ax.axhline(3.0, color="k", ls="--", lw=1.0, label="3 % gate")
        ax.set_xlabel("degrees of freedom")
        ax.set_ylabel(f"relative {title} error (%)")
        order = orders.get(f"{metric}/bbar", float("nan"))
        ax.set_title(f"{title}: B-bar observed order $p$={order:.2f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Mesh refinement at $\\nu$={FINE_POISSON}")
    fig.tight_layout()
    fig.savefig(ASSETS / "refinement_convergence.png", dpi=160)
    plt.close(fig)

    # (3) field profiles against the exact Lame field
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    for ax, key, exact_values, title in (
            (axes[0], "radial_stress", exact.radial_stress, "$\\sigma_r$ (MPa)"),
            (axes[1], "hoop_stress", exact.hoop_stress, "$\\sigma_\\theta$ (MPa)")):
        ax.plot(radii.tolist(), (exact_values / 1e6).tolist(), "k-", lw=2.0, label="exact Lame")
        for formulation in FORMULATIONS:
            profile = profiles[formulation]
            order = sorted(range(len(profile["radius"])), key=lambda i: profile["radius"][i])
            ax.plot([profile["radius"][i] for i in order],
                    [profile[key][i] / 1e6 for i in order], marker=markers[formulation], ms=3.5,
                    ls="", color=colours[formulation], label=labels[formulation])
        ax.set_xlabel("radius r (m)")
        ax.set_ylabel(title)
        ax.set_title(title.split(" (")[0] + f", $\\nu$={FINE_POISSON}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Recovered field (element centres) vs closed form, {SECTOR[0]}x{SECTOR[1]} sector")
    fig.tight_layout()
    fig.savefig(ASSETS / "stress_profiles_nu_4999.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()