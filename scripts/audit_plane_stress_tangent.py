"""Verify the plane-stress J2 tangential operator against a branch-preserving reference.

The reference is a Richardson-extrapolated forward difference of the actual
condensed response that stays inside the branch taken by the return mapping.
Two candidate tangents are compared:

* ``legacy``: the pre-patch implementation, which differentiated the response with
  centred difference of step ``eps**(1/3)`` and solved the local ``sigma_zz=0``
  condensation with a ``sqrt(eps)`` difference;
* ``analytic``: the Schur condensation of the exact radial-return algorithmic
  tangent (``plasticity.j2_consistent_tangent``).

Run: ``PYTHONPATH=src python scripts/audit_plane_stress_tangent.py``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from tensorfem.layered_shell4_plasticity import (
    LayeredShell4Model,
    LayeredShell4State,
    _tensor_from_inplane,
    plane_stress_j2_update,
)
from tensorfem.plasticity import j2_consistent_tangent, update_j2

D = torch.float64


def model_and_state():
    nodes = torch.tensor([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]], dtype=D)
    model = LayeredShell4Model(nodes, torch.tensor([[0, 1, 2, 3]]), 200000., .3, .1, 250., 1200.)
    return model, LayeredShell4State.virgin(model).points[0][0][0]


def uniaxial(model, target: float) -> torch.Tensor:
    alpha = max((target - model.yield_stress) / model.hardening, 0.)
    return torch.tensor([target / model.young + alpha,
                         -model.poisson * target / model.young - alpha / 2., 0.], dtype=D)


def legacy_response(model, committed, value: torch.Tensor) -> torch.Tensor:
    """Pre-patch local condensation: FD derivative of sigma_zz in eps_zz."""
    zz = -model.poisson / (1 - model.poisson) * (value[0] + value[1])
    for _ in range(30):
        full = _tensor_from_inplane(value, zz)
        sigma = update_j2(full, model.young, model.poisson, model.yield_stress,
                          model.hardening, committed)[0]
        residual = sigma[2, 2]
        if abs(float(residual)) <= 2e-12 * max(model.yield_stress, 1.):
            break
        h = math.sqrt(torch.finfo(D).eps) * max(1., abs(float(zz)))
        sp = update_j2(_tensor_from_inplane(value, zz + h), model.young, model.poisson,
                       model.yield_stress, model.hardening, committed)[0][2, 2]
        sm = update_j2(_tensor_from_inplane(value, zz - h), model.young, model.poisson,
                       model.yield_stress, model.hardening, committed)[0][2, 2]
        zz = zz - residual / ((sp - sm) / (2 * h))
    return torch.stack((sigma[0, 0], sigma[1, 1], sigma[0, 1]))


def legacy_tangent(model, committed, strain: torch.Tensor) -> torch.Tensor:
    base = torch.finfo(D).eps ** (1. / 3.)
    columns = []
    for j in range(3):
        h = base * max(1., abs(float(strain[j])))
        delta = torch.zeros_like(strain); delta[j] = h
        columns.append((legacy_response(model, committed, strain + delta)
                        - legacy_response(model, committed, strain - delta)) / (2 * h))
    return torch.stack(columns, 1)


def reference_column(model, committed, strain: torch.Tensor, j: int) -> torch.Tensor:
    """Branch-preserving one-sided derivative (forward, Richardson extrapolated)."""
    unit = torch.zeros(3, dtype=D); unit[j] = 1.
    base = float(strain[j])
    def derivative(h):
        return (plane_stress_j2_update(strain + h * unit, model, committed, tangent=False)[0]
                - plane_stress_j2_update(strain, model, committed, tangent=False)[0]) / h
    scale = max(1., abs(base))
    d1 = derivative(8e-7 * scale); d2 = derivative(4e-7 * scale); d3 = derivative(2e-7 * scale)
    return _extrap(d1, d2, d3)


def _extrap(d1, d2, d3):
    """Richardson extrapolation of three halving forward differences."""
    return (4 * (4 * d3 - d2) / 3 - (4 * d2 - d1) / 3) / 3


def reference_tangent(model, committed, strain):
    return torch.stack([reference_column(model, committed, strain, j) for j in range(3)], 1)


def relative_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(candidate - reference)
                 / torch.linalg.matrix_norm(reference))


def main():
    model, state = model_and_state()
    targets = (100., 200., 240., 245., 249., 249.9, 250.5, 251., 260., 310., 400.)
    rows = []
    for target in targets:
        strain = uniaxial(model, target)
        legacy = legacy_tangent(model, state, strain)
        analytic = plane_stress_j2_update(strain, model, state)[1]
        reference = reference_tangent(model, state, strain)
        rows.append({
            "sigma_xx_MPa": target,
            "branch": "plastic" if target > model.yield_stress else "elastic",
            "legacy_relative_error": relative_error(legacy, reference),
            "analytic_relative_error": relative_error(analytic, reference),
            "reference_C_xxxx": float(reference[0, 0]),
            "legacy_C_xxxx": float(legacy[0, 0]),
            "analytic_C_xxxx": float(analytic[0, 0]),
            "elastic_plane_stress_C_xxxx": 200000. / (1 - .3 ** 2),
        })
    worst_legacy = max(rows, key=lambda r: r["legacy_relative_error"])
    worst_analytic = max(rows, key=lambda r: r["analytic_relative_error"])
    payload = {
        "model": {"young": 200000., "poisson": .3, "yield_stress": 250., "hardening": 1200.},
        "reference": "branch-preserving forward difference, Richardson extrapolated",
        "rows": rows,
        "worst_legacy": worst_legacy,
        "worst_analytic": worst_analytic,
        "gate": {"analytic_max_relative_error": 1e-6, "legacy_max_relative_error": 1e-2},
    }
    out = Path("artifacts/plane_stress_tangent")
    out.mkdir(parents=True, exist_ok=True)
    (out / "verification.json").write_text(json.dumps(payload, indent=2))

    print("| sigma_xx [MPa] | branch | legacy rel. error | analytic rel. error | ref C_xxxx | legacy C_xxxx | analytic C_xxxx |")
    print("|---:|---|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['sigma_xx_MPa']:.1f} | {r['branch']} | {r['legacy_relative_error']:.3e} | "
              f"{r['analytic_relative_error']:.3e} | {r['reference_C_xxxx']:.6f} | "
              f"{r['legacy_C_xxxx']:.6f} | {r['analytic_C_xxxx']:.6f} |")
    print(f"worst legacy {worst_legacy['sigma_xx_MPa']} MPa -> {worst_legacy['legacy_relative_error']:.3e}")
    print(f"worst analytic {worst_analytic['sigma_xx_MPa']} MPa -> {worst_analytic['analytic_relative_error']:.3e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        x = [r["sigma_xx_MPa"] for r in rows]
        axes[0].semilogy(x, [r["legacy_relative_error"] for r in rows], "o-", label="legacy centred FD")
        axes[0].semilogy(x, [max(r["analytic_relative_error"], 1e-17) for r in rows], "s-",
                         label="analytic (Schur, exact)")
        axes[0].axvline(250., color="k", ls="--", lw=1, label="yield (250 MPa)")
        axes[0].axhline(1e-6, color="r", ls=":", lw=1, label="1e-6 gate")
        axes[0].set_xlabel("applied sigma_xx [MPa]")
        axes[0].set_ylabel("relative Frobenius error of tangent")
        axes[0].set_title("tangent error vs branch-preserving reference")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=.3, which="both")
        axes[1].plot(x, [r["reference_C_xxxx"] / 1e3 for r in rows], "k--", label="reference")
        axes[1].plot(x, [r["legacy_C_xxxx"] / 1e3 for r in rows], "o-", label="legacy")
        axes[1].plot(x, [r["analytic_C_xxxx"] / 1e3 for r in rows], "s-", label="analytic")
        axes[1].axvline(250., color="k", ls="--", lw=1)
        axes[1].set_xlabel("applied sigma_xx [MPa]")
        axes[1].set_ylabel("condensed C_xxxx [GPa]")
        axes[1].set_title("condensed plane-stress stiffness")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out / "tangent_error.png", dpi=130)
        print("wrote", out / "tangent_error.png")
    except Exception as error:  # pragma: no cover - figure is optional evidence
        print("figure skipped:", error)


if __name__ == "__main__":
    main()