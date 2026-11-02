"""h-convergence / GCI evidence for the TET4-HEX8 solid elements and its figure.

Runs two analytic-oracle studies and writes machine-readable evidence plus the
figure used by ``docs/h-convergence-gci-verification.md``:

* ``bending``      -- pure-bending Dirichlet problem, ``(2n,n,n)`` HEX8 meshes,
  closed-form displacement and ``U = E k^2 L I / 2``;
* ``manufactured`` -- smooth trigonometric manufactured solution with a body
  force differentiated from the analytic field, ``(2n,n,3)`` HEX8 meshes.

For each study the script reports the observed order of accuracy per mesh triple
(Richardson), the Richardson extrapolated value and the Roache GCI band of the
finest pair, and compares both against the analytic value.  Nothing is compared
with a value produced by the solver itself.

Usage::

    python benchmarks/h_convergence_gci.py [--out docs/assets/h-convergence-gci]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfem.convergence_verification import (  # noqa: E402
    ConvergenceLevel,
    evaluate_convergence_study,
    observed_order_from_errors,
)
from tensorfem.elasticity_convergence import (  # noqa: E402
    ManufacturedElasticity,
    PureBendingOracle,
    hex8_dirichlet_study,
    hex8_manufactured_study,
)

DEFAULT_OUT = os.path.join("docs", "assets", "h-convergence-gci")


def bending_case() -> PureBendingOracle:
    return PureBendingOracle(
        length=2.0, width=0.4, height=0.6,
        young_modulus=210e9, poisson_ratio=0.3, curvature=1e-3,
    )


def manufactured_case() -> ManufacturedElasticity:
    return ManufacturedElasticity(
        length=2.0, width=0.4, height=0.6,
        young_modulus=210e9, poisson_ratio=0.3, amplitude=1e-4, planar=True,
    )


def bending_evidence() -> dict[str, object]:
    case = bending_case()
    records = hex8_dirichlet_study(case, [1, 2, 4])
    errors = [record["relative_l2_displacement_error"] for record in records]
    energy_errors = [record["relative_strain_energy_error"] for record in records]
    return {
        "oracle": "pure bending (Navier), closed form U = E k^2 L I / 2",
        "case": case.as_dict(),
        "levels": records,
        "nodal_exactness_max_relative_error": max(errors),
        "strain_energy_order": observed_order_from_errors(energy_errors, [2.0, 2.0]),
        "note": (
            "The bending field is reproduced nodally to machine precision on a "
            "uniform mesh (superconvergence), so it is used as a consistency "
            "oracle; the strain-energy error shown is the interpolation error of "
            "the element, which still decays at order 2."
        ),
    }


def manufactured_evidence() -> dict[str, object]:
    case = manufactured_case()
    records = hex8_manufactured_study(case, [2, 4, 8])
    exact_energy = case.strain_energy_exact()
    levels = [ConvergenceLevel(r["mesh_size"], r["strain_energy"], f"n={r['divisions']}")
              for r in records]
    report = evaluate_convergence_study(
        levels,
        qoi="strain_energy",
        expected_order=2.0,
        order_band=0.15,
        reference_value=exact_energy,
        reference_label="analytic int sigma:eps/2 dV",
        reference_tolerance=0.03,
    )
    orders = observed_order_from_errors(
        [r["relative_l2_displacement_error"] for r in records], [2.0, 2.0]
    )
    return {
        "oracle": "manufactured trigonometric field, body force from autograd of div(sigma)",
        "case": case.as_dict(),
        "levels": records,
        "l2_observed_orders": orders,
        "energy_study": report,
        "pass": bool(report["passed"]) and orders[-1] > 1.85,
    }


def figure(bending: dict, manufactured: dict, path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"matplotlib unavailable ({exc}); skipping figure")
        return None

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # (a) field comparison for the pure-bending case: sigma_xx analytic vs FE
    ax = axes[0]
    case = bending_case()
    n = 4
    nodes, hexes = _bending_mesh(case, n)
    from tensorfem.solid3d import SolidModel, solve_solid

    exact = case.displacement(nodes)
    boundary = ((nodes[:, 0] == 0) | (nodes[:, 0] == case.length)
                | (nodes[:, 1] == 0) | (nodes[:, 1] == case.width)
                | (nodes[:, 2] == 0) | (nodes[:, 2] == case.height))
    bn = torch.where(boundary)[0]
    fixed = torch.stack((3 * bn, 3 * bn + 1, 3 * bn + 2), 1).reshape(-1)
    result = solve_solid(SolidModel(
        nodes, hexes, torch.tensor(case.young_modulus, dtype=torch.float64),
        torch.tensor(case.poisson_ratio, dtype=torch.float64),
        torch.zeros(3 * len(nodes), dtype=torch.float64), fixed, "hex8",
        exact.reshape(-1)[fixed],
    ))
    centroids = nodes[hexes].mean(1).detach()
    sigma_fe = result.stress[:, 0].detach() / 1e6
    sigma_exact = case.axial_stress(centroids) / 1e6
    ax.scatter(sigma_exact, sigma_fe, s=8, color="tab:blue")
    lim = [float(sigma_exact.min()), float(sigma_exact.max())]
    ax.plot(lim, lim, "k--", lw=1, label="1:1")
    ax.set_xlabel(r"analytic $\sigma_{xx}$ [MPa] at element centroid")
    ax.set_ylabel(r"HEX8 $\sigma_{xx}$ [MPa]")
    ax.set_title("(a) pure bending field, $n$=4 ($8\\times4\\times4$ HEX8)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=.3)

    # (b) log-log convergence against the theoretical slope
    ax = axes[1]
    levels = manufactured["levels"]
    h = [r["mesh_size"] for r in levels]
    l2 = [r["relative_l2_displacement_error"] for r in levels]
    en = [r["relative_strain_energy_error"] for r in levels]
    ax.loglog(h, l2, "o-", label="L2 displacement error")
    ax.loglog(h, en, "s-", label="relative strain-energy error")
    p = manufactured["l2_observed_orders"][-1]
    ref = [l2[-1] * (hh / h[-1]) ** 2 for hh in h]
    ax.loglog(h, ref, "k:", label="slope 2 reference")
    ax.set_xlabel("mesh size h [m]")
    ax.set_ylabel("relative error")
    ax.set_title(f"(b) manufactured solution: observed $p$ = {p:.2f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")

    # (c) GCI band vs the true analytic error
    ax = axes[2]
    study = manufactured["energy_study"]
    gci = study["gci"]
    idx = range(len(levels))
    ax.bar([i - 0.2 for i in idx], [r["relative_strain_energy_error"] for r in levels],
           width=0.4, label="true error vs analytic $U$")
    ax.bar([i + 0.2 for i in idx], [gci["relative_gci"]] * len(levels), width=0.4,
           label="GCI band (finest order)")
    ax.set_xticks(list(idx))
    ax.set_xticklabels([f"n={r['divisions']}" for r in levels])
    ax.set_yscale("log")
    ax.set_ylabel("relative uncertainty")
    ax.set_title(
        f"(c) Richardson error {study['reference']['extrapolated_relative_error']:.1e}"
        f" vs analytic, order {study['finest_order']:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _bending_mesh(case: PureBendingOracle, n: int):
    from tensorfem.solid3d import structured_hex_mesh

    return structured_hex_mesh(case.length, case.width, case.height, 2 * n, n, n)


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    bending = bending_evidence()
    manufactured = manufactured_evidence()
    figure_path = figure(bending, manufactured, os.path.join(out_dir, "convergence.png"))

    payload = {
        "schema": "tensorfem.h-convergence-gci/1.0",
        "bending": bending,
        "manufactured": manufactured,
        "figure": os.path.basename(figure_path) if figure_path else None,
        "pass": bool(manufactured["pass"] and bending["nodal_exactness_max_relative_error"] < 1e-11),
    }
    json_path = os.path.join(out_dir, "evidence.json")
    with open(json_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps({
        "evidence": json_path,
        "figure": figure_path,
        "bending_nodal_max_error": bending["nodal_exactness_max_relative_error"],
        "bending_energy_orders": bending["strain_energy_order"],
        "manufactured_l2_orders": manufactured["l2_observed_orders"],
        "manufactured_energy_order": manufactured["energy_study"]["finest_order"],
        "richardson_vs_analytic_rel_error": manufactured["energy_study"]["reference"][
            "extrapolated_relative_error"],
        "gci_relative": manufactured["energy_study"]["gci"]["relative_gci"],
        "pass": payload["pass"],
    }, indent=2))
    return payload


if __name__ == "__main__":
    main()