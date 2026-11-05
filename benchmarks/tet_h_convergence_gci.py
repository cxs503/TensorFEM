"""TET4 / TET10 h-convergence and GCI evidence (closes the "hex8 only" gap).

The HEX8 evidence in ``docs/h-convergence-gci-verification.md`` states that
``tet4``, TET10, shells, plates and contact are not covered.  This script runs
the same methodology for the two tetrahedral families and writes machine
readable evidence plus a figure:

* ``manufactured`` -- smooth trigonometric manufactured field with a body force
  obtained by automatic differentiation of ``div(sigma)``; the consistent nodal
  loads are integrated with an independent collapsed Gauss rule
  (:func:`tensorfem.tet_convergence.tet_consistent_body_force`), never with the
  ``B`` matrix or stiffness under test;
* ``dirichlet`` -- closed-form pure bending prescribed on the mesh boundary with
  no body force.  The field belongs to the TET10 space, so TET10 must reproduce
  the nodal values to roundoff; TET4 cannot and its error decays with ``h``.

Usage::

    python benchmarks/tet_h_convergence_gci.py --out docs/assets/tet-h-convergence-gci
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
from tensorfem.elasticity_convergence import ManufacturedElasticity, PureBendingOracle  # noqa: E402
from tensorfem.tet_convergence import (  # noqa: E402
    tet4_manufactured_study,
    tet10_manufactured_study,
    tet_dirichlet_study,
)

DEFAULT_OUT = os.path.join("docs", "assets", "tet-h-convergence-gci")
DIVISIONS = (2, 4, 8)
LAYERS = 3


def manufactured_case() -> ManufacturedElasticity:
    return ManufacturedElasticity(
        length=2.0, width=0.4, height=0.6,
        young_modulus=210e9, poisson_ratio=0.3, amplitude=1e-4, planar=True,
    )


def bending_case() -> PureBendingOracle:
    return PureBendingOracle(
        length=2.0, width=0.4, height=0.6,
        young_modulus=210e9, poisson_ratio=0.3, curvature=1e-3,
    )


def _study(records, *, qoi: str, expected_order: float, reference: float, label: str) -> dict:
    levels = [
        ConvergenceLevel(
            mesh_size=record["mesh_size"],
            value=record["strain_energy"] if qoi == "strain_energy"
            else record["relative_l2_displacement_error"],
            label=f"{record['mesh'][0]}x{record['mesh'][1]}x{record['mesh'][2]}",
        )
        for record in records
    ]
    return evaluate_convergence_study(
        levels,
        qoi=qoi,
        expected_order=expected_order,
        order_band=0.25,
        reference_value=reference,
        reference_label=label,
        reference_tolerance=0.03,
    )


def manufactured_evidence(family: str) -> dict:
    case = manufactured_case()
    runner = tet4_manufactured_study if family == "tet4" else tet10_manufactured_study
    records = runner(case, DIVISIONS, layers=LAYERS)
    ratios = [
        a["mesh_size"] / b["mesh_size"] for a, b in zip(records, records[1:])
    ]
    l2_orders = observed_order_from_errors(
        [r["relative_l2_displacement_error"] for r in records], ratios
    )
    energy_orders = observed_order_from_errors(
        [r["relative_strain_energy_error"] for r in records], ratios
    )
    energy_study = _study(
        records,
        qoi="strain_energy",
        expected_order=2.0 if family == "tet4" else 3.0,
        reference=float(records[0]["exact_strain_energy"]),
        label="analytic int sigma:eps/2 dV",
    )
    return {
        "family": family,
        "oracle": "manufactured trigonometric field, body force from autograd(dlb.sigma)",
        "case": case.as_dict(),
        "levels": records,
        "l2_observed_orders": l2_orders,
        "energy_observed_orders": energy_orders,
        "energy_study": energy_study,
    }


def dirichlet_evidence(family: str) -> dict:
    oracle = bending_case()
    records = tet_dirichlet_study(oracle, DIVISIONS, family=family, layers=LAYERS)
    ratios = [a["mesh_size"] / b["mesh_size"] for a, b in zip(records, records[1:])]
    return {
        "family": family,
        "oracle": "closed-form pure bending, Dirichlet on the whole boundary, no body force",
        "case": oracle.as_dict(),
        "levels": records,
        "l2_observed_orders": observed_order_from_errors(
            [r["relative_l2_displacement_error"] for r in records], ratios
        ),
        "roundoff_floor": family == "tet10"
        and max(r["relative_max_displacement_error"] for r in records) < 1e-9,
    }


def figure(manufactured: dict, dirichlet: dict, path: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for family, style in (("tet4", "o-"), ("tet10", "s--")):
        record = manufactured[family]["levels"]
        h = [level["mesh_size"] for level in record]
        e = [level["relative_l2_displacement_error"] for level in record]
        axes[0].loglog(h, e, style, label=f"{family} manufactured")
        du = [level["relative_strain_energy_error"] for level in record]
        axes[1].loglog(h, du, style, label=f"{family} energy")
        d = dirichlet[family]["levels"]
        axes[1].loglog(
            [level["mesh_size"] for level in d],
            [level["relative_strain_energy_error"] for level in d],
            style, alpha=0.6, label=f"{family} bending (Dirichlet)",
        )
    # Theory slopes for reference.
    h0 = manufactured["tet4"]["levels"][-1]["mesh_size"]
    e0 = manufactured["tet4"]["levels"][-1]["relative_l2_displacement_error"]
    for order, text in ((2.0, "slope 2"), (3.0, "slope 3")):
        hh = [h0 * 2, h0]
        axes[0].loglog(hh, [e0 * 4**order, e0], ":", color="0.6", lw=1)
        axes[0].annotate(text, (hh[0], e0 * 4**order), fontsize=8, color="0.4")
    axes[0].set_xlabel("h = L/nx")
    axes[0].set_ylabel("relative L2 displacement error")
    axes[1].set_xlabel("h = L/nx")
    axes[1].set_ylabel("relative strain-energy error")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.3)
        axis.legend(fontsize=8)
    axes[0].set_title("Manufactured solution: displacement error")
    axes[1].set_title("Strain energy (manufactured) and Dirichlet consistency")
    fig.tight_layout()
    out = os.path.join(path, "tet-h-convergence.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    torch.set_grad_enabled(True)
    manufactured = {family: manufactured_evidence(family) for family in ("tet4", "tet10")}
    dirichlet = {family: dirichlet_evidence(family) for family in ("tet4", "tet10")}
    fig = figure(manufactured, dirichlet, args.out) if manufactured else None
    evidence = {
        "schema": "tensorfem.tet-h-convergence-gci/1",
        "divisions": list(DIVISIONS),
        "layers": LAYERS,
        "mesh_family": "(2n, n, 3) box, six tets per hex, conforming TET10 upgrade",
        "manufactured": manufactured,
        "dirichlet": dirichlet,
        "figure": os.path.basename(fig) if fig else None,
        "pass": all(manufactured[f]["energy_study"]["passed"] for f in manufactured)
        and dirichlet["tet10"]["roundoff_floor"],
    }
    path = os.path.join(args.out, "evidence.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, sort_keys=True)
    print(json.dumps({
        "pass": evidence["pass"],
        "tet4_l2_orders": manufactured["tet4"]["l2_observed_orders"],
        "tet10_l2_orders": manufactured["tet10"]["l2_observed_orders"],
        "tet4_energy_orders": manufactured["tet4"]["energy_observed_orders"],
        "tet10_energy_orders": manufactured["tet10"]["energy_observed_orders"],
        "evidence": path,
        "figure": fig,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())