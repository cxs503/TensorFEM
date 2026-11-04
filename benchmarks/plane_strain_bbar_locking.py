"""Evidence + figures for the plane-strain B-bar Q4 locking study.

Writes ``benchmarks/plane_strain_bbar_locking_evidence.json`` and PNGs under
``docs/assets/plane_strain_bbar/``.  Every number in the report is taken from
the JSON this script produces; nothing is transcribed by hand.

Run with ``PYTHONPATH=src python3 benchmarks/plane_strain_bbar_locking.py``.
"""
from __future__ import annotations

import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.plane_strain_locking import (DEFAULT_POISSONS, FORMULATIONS, conditioning_study,
                                            lame_plane_strain_self_check, locking_sweep,
                                            observed_order, refinement_study, solve_lame_case)

# The dense annular solves are tiny; capping threads keeps the run reproducible on
# a loaded 128-core host instead of thrashing through 128 OpenMP teams.
torch.set_num_threads(4)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "docs", "assets", "plane_strain_bbar")
EVIDENCE = os.path.join(ROOT, "benchmarks", "plane_strain_bbar_locking_evidence.json")
REFINEMENT = ((16, 8), (24, 12), (32, 16), (40, 20))
THRESHOLD = 3.0


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
             "n_theta": r["n_theta"], "n_radial": r["n_radial"], "elements": r["elements"],
             "dofs": r["dofs"], "h": r["h"]}
            | {k: r["errors"][k] for k in keys}
            for r in records]


def figure_error_vs_poisson(sweep, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    styles = {"standard": dict(marker="o", linestyle="-", color="#c0392b", label="Q4 fully integrated (production)"),
              "bbar": dict(marker="s", linestyle="-", color="#1f6f3f", label="Q4 B-bar (mean dilatation)")}
    for key, axis, title in (("displacement_l2", axes[0], r"radial displacement $L_2$"),
                             ("radial_stress_l2", axes[1], r"radial stress $L_2$")):
        for formulation in FORMULATIONS:
            rows = [r for r in sweep if r["formulation"] == formulation]
            axis.loglog([r["poisson_ratio"] for r in rows],
                        [100.0 * r["errors"][key] for r in rows], **styles[formulation])
        axis.axhline(THRESHOLD, color="0.4", linestyle="--", linewidth=1.0)
        axis.text(0.31, THRESHOLD * 1.15, "3 % gate", color="0.35", fontsize=8)
        axis.set_xlabel(r"Poisson ratio $\nu$")
        axis.set_ylabel("relative $L_2$ error [%]")
        axis.set_title(title)
        axis.grid(True, which="both", alpha=0.3)
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Plane-strain Lame annulus, 16x8 Q4 (306 DOF): locking is a formulation error")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_refinement(refinement, path):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for key, axis, title in (("radial_stress_l2", axes[0], r"radial stress $L_2$"),
                             ("displacement_l2", axes[1], r"radial displacement $L_2$")):
        for formulation, style in (("standard", dict(marker="o", color="#c0392b", linestyle="-")),
                                   ("bbar", dict(marker="s", color="#1f6f3f", linestyle="-"))):
            rows = [r for r in refinement if r["formulation"] == formulation]
            axis.loglog([r["h"] for r in rows], [100.0 * r["errors"][key] for r in rows],
                        label=formulation, **style)
            if formulation == "bbar" and len(rows) >= 2:
                order = observed_order(rows[-2], rows[-1])
                last = rows[-1]
                line_h = [row["h"] for row in rows]
                reference = 100.0 * last["errors"][key]
                axis.plot(line_h, [reference * (h / last["h"]) ** order for h in line_h],
                          color="0.35", linestyle="--", linewidth=1.0,
                          label=f"slope {order:.2f}")
        axis.axhline(THRESHOLD, color="0.4", linestyle=":", linewidth=1.0)
        axis.set_xlabel("element size $h$ [m]")
        axis.set_ylabel("relative $L_2$ error [%]")
        axis.set_title(title)
        axis.invert_xaxis()
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle(r"Refinement at $\nu=0.4999$: fully integrated Q4 stalls, B-bar converges")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_stress_profile(case, path):
    fields = case["fields"]
    radius = fields["centre_radius"].numpy()
    order = radius.argsort()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for axis, stress, exact, label in ((axes[0], "sigma_r", "exact_sigma_r", r"$\sigma_r$"),
                                       (axes[1], "sigma_theta", "exact_sigma_theta", r"$\sigma_\theta$")):
        axis.plot(radius[order] / 0.5, fields[exact][order].numpy() / 1e6, "k-", linewidth=1.6,
                  label="Lame exact")
        axis.plot(radius[order] / 0.5, fields[stress][order].numpy() / 1e6, "s", markersize=3.2,
                  markerfacecolor="none", markeredgecolor="#1f6f3f", label="Q4 B-bar")
        axis.set_xlabel("$r/a$")
        axis.set_ylabel(label + " [MPa]")
        axis.set_title(label + r" at $\nu = 0.4999$")
        axis.grid(True, alpha=0.3)
        axis.legend(frameon=False, fontsize=9)
    figure.suptitle("Element-centre stress recovery against the closed form (16x8 Q4, B-bar)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def figure_error_map(cases, path):
    figure, axes = plt.subplots(1, len(cases), figsize=(5.0 * len(cases), 4.3))
    for axis, case in zip(axes, cases):
        fields = case["fields"]
        stress = fields["sigma_r"].numpy()
        exact = fields["exact_sigma_r"].numpy()
        error = 100.0 * (stress - exact) / abs(exact).max()
        scatter = axis.scatter(fields["centre_x"].numpy() / 0.5, fields["centre_y"].numpy() / 0.5,
                               c=error, cmap="coolwarm", vmin=-60.0, vmax=60.0, s=26, marker="s")
        axis.set_aspect("equal")
        axis.set_xlabel("$x/a$")
        axis.set_ylabel("$y/a$")
        axis.set_title(f"{case['formulation']}: $(\\sigma_r-\\sigma_r^{{exact}})/\\max|\\sigma_r^{{exact}}|$ [%]")
        figure.colorbar(scatter, ax=axis, fraction=0.06)
    figure.suptitle(r"Radial-stress error across the wall at $\nu = 0.4999$ (same mesh, same load)")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    self_checks = {str(nu): lame_plane_strain_self_check(nu) for nu in (0.30, 0.4999)}
    for check in self_checks.values():
        assert check["equilibrium"] < 1e-9, check
        assert check["traction"] < 1e-12, check
        assert check["constitutive"] < 1e-9, check
        assert check["poisson_invariance"] < 1e-12, check

    sweep = locking_sweep(DEFAULT_POISSONS, n_theta=16, n_radial=8)
    refinement = refinement_study(0.4999, levels=REFINEMENT)
    conditioning = conditioning_study(DEFAULT_POISSONS, n_theta=16, n_radial=8)
    headline_cases = [solve_lame_case(0.4999, formulation=formulation, n_theta=16, n_radial=8)
                      for formulation in FORMULATIONS]

    keys = ("displacement_l2", "radial_stress_l2", "hoop_stress_l2", "trace_stress_l2",
            "support_equilibrium")
    bbar_refinement = [r for r in refinement if r["formulation"] == "bbar"]
    standard_refinement = [r for r in refinement if r["formulation"] == "standard"]
    payload = {
        "study": "plane-strain Q4 mean-dilatation (B-bar) locking qualification on the Lame annulus",
        "reference": "Lame (1852) closed form, plane strain: sigma_r = A - B/r^2, sigma_theta = A + B/r^2, "
                     "u_r = (1+nu)/E [(1-2nu) A r + B/r]; A, B from the tractions only",
        "setup": {"inner_radius": 0.5, "outer_radius": 0.8, "inner_pressure": 100.0e6,
                  "young_modulus": 210.0e9, "thickness": 1.0, "sector": "quarter annulus (pi/2)",
                  "element": "Q4 plane strain, production continuum path",
                  "loads": "consistent nodal pressure on the bore (tensorfem.lame_reference)",
                  "constraints": "symmetry cuts only (tensorfem.lame_reference)",
                  "dense_solver": True, "dtype": "float64",
                  "threads": 4},
        "reference_self_check": self_checks,
        "sweep": _table(sweep, keys),
        "refinement_nu_0_4999": _table(refinement, keys),
        "observed_orders_nu_0_4999": {
            "bbar": {k: observed_order(bbar_refinement[-2], bbar_refinement[-1], k)
                     for k in ("displacement_l2", "radial_stress_l2", "trace_stress_l2")},
            "standard": {k: observed_order(standard_refinement[-2], standard_refinement[-1], k)
                         for k in ("displacement_l2", "radial_stress_l2", "trace_stress_l2")},
        },
        "conditioning": [{k: r[k] for k in ("poisson_ratio", "formulation", "n_dofs_free",
                                            "condition_number")} for r in conditioning],
        "headline_nu_0_4999_16x8": _table(headline_cases, keys),
        "thresholds": {"gate_percent": THRESHOLD,
                       "standard_failure_floor_percent": 50.0,
                       "locking_flatness_allowed_ratio": 1.10},
        "figures": ["error_vs_poisson.png", "refinement_nu_4999.png",
                    "stress_profile_nu_4999.png", "error_map_nu_4999.png"],
    }
    with open(EVIDENCE, "w") as handle:
        json.dump(_plain(payload), handle, indent=2, sort_keys=True)

    figure_error_vs_poisson(sweep, os.path.join(ASSETS, "error_vs_poisson.png"))
    figure_refinement(refinement, os.path.join(ASSETS, "refinement_nu_4999.png"))
    figure_stress_profile(headline_cases[1], os.path.join(ASSETS, "stress_profile_nu_4999.png"))
    figure_error_map(headline_cases, os.path.join(ASSETS, "error_map_nu_4999.png"))

    summary = {
        "nu_0_4999_16x8": {r["formulation"]: {k: round(100.0 * r["errors"][k], 4) for k in keys}
                           for r in headline_cases},
        "observed_orders": payload["observed_orders_nu_0_4999"],
        "worst_bbar_error_percent": round(100.0 * max(r["errors"]["radial_stress_l2"]
                                                     for r in bbar_refinement), 4),
    }
    print(json.dumps(_plain(summary), indent=2))


if __name__ == "__main__":
    main()