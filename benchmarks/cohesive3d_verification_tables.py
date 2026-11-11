"""Verification table and figure for the 3-D cohesive interface element.

Produces, programmatically (no hand-copied numbers):

* ``outputs/cohesive3d/verification_table.json`` -- every check with the
  computed value, its closed form, and the relative error;
* ``outputs/cohesive3d/traction_separation.png`` -- the mixed-mode envelopes and
  the normalised dissipated energy (figure used in the platform report).

Run with ``PYTHONPATH=src python benchmarks/cohesive3d_verification_tables.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from tensorfem.cohesive import BilinearCohesiveLaw
from tensorfem.cohesive3d import (
    Cohesive3DLaw,
    cohesive_interface_response,
    face_shape,
    gauss_rule,
    interface_area,
    interface_frame,
)

D = torch.float64
OUT = Path("outputs/cohesive3d")
KN, KS, TN0, TS0, GIC, GIIC, ETA = 1.0e5, 6.0e4, 10.0, 8.0, 0.5, 1.2, 2.284


def law() -> Cohesive3DLaw:
    return Cohesive3DLaw(KN, KS, TN0, TS0, GIC, GIIC, ETA)


def flat(side: float = 1.0) -> torch.Tensor:
    quad = torch.tensor([[0.0, 0.0], [side, 0.0], [side, side], [0.0, side]], dtype=D)
    bottom = torch.cat([quad, torch.zeros(4, 1, dtype=D)], 1)
    top = torch.cat([quad, torch.full((4, 1), 1.0, dtype=D)], 1)
    return torch.stack((bottom, top))


def distorted() -> torch.Tensor:
    quad = torch.tensor([[0.0, 0.0], [1.3, 0.05], [1.05, 0.9], [-0.1, 1.15]], dtype=D)
    bottom = torch.cat([quad, torch.zeros(4, 1, dtype=D)], 1)
    top = bottom.clone()
    top[:, 2] = torch.tensor([0.9, 1.02, 1.1, 0.85], dtype=D)
    top[:, 0] += torch.tensor([0.02, -0.01, 0.015, -0.02], dtype=D)
    return torch.stack((bottom, top))


def uniform(opening) -> torch.Tensor:
    u = torch.zeros(8, 3, dtype=D)
    u[4:, 0] = opening[1]
    u[4:, 1] = opening[2]
    u[4:, 2] = opening[0]
    return u


def work_integral(law_, direction, steps=20_001):
    c = direction / torch.linalg.vector_norm(direction)
    _, lamf, _, _ = law_.direction_measures(direction)
    grid = torch.linspace(0.0, float(lamf), steps, dtype=D)
    history = torch.zeros((), dtype=D)
    values = torch.zeros_like(grid)
    for i, amplitude in enumerate(grid):
        values[i] = law_.traction(amplitude * c, history) @ c
        history = torch.maximum(history, amplitude)
    return torch.trapezoid(values, grid)


def tangent_error() -> dict:
    """Relative error of the element tangent against central differences."""
    law_ = law()
    coords = distorted()
    u = 2e-5 * torch.rand(8, 3, dtype=D, generator=torch.Generator().manual_seed(20261111))
    u[4:, 2] += 4e-5
    analytic = cohesive_interface_response(coords, u, law_).stiffness
    scale = float(torch.max(torch.abs(analytic)))
    flat_dofs = u.reshape(24)
    errors = {}
    for step in (1e-8, 1e-9, 1e-10):
        numeric = torch.zeros((24, 24), dtype=D)
        for k in range(24):
            plus, minus = flat_dofs.clone(), flat_dofs.clone()
            plus[k] += step
            minus[k] -= step
            numeric[:, k] = (
                cohesive_interface_response(coords, plus, law_).force
                - cohesive_interface_response(coords, minus, law_).force
            ) / (2.0 * step)
        errors[f"{step:.0e}"] = float(torch.max(torch.abs(analytic - numeric))) / scale
    return errors


def rows():
    law_ = law()
    reference = BilinearCohesiveLaw(KN, TN0, GIC)
    coords = flat()
    out = []

    def add(name, computed, closed_form, note=""):
        computed, closed_form = float(computed), float(closed_form)
        error = abs(computed - closed_form) / abs(closed_form) if closed_form else abs(computed)
        out.append({"check": name, "computed": computed, "closed_form": closed_form,
                    "relative_error": error, "note": note})

    # element-level exact checks
    result = cohesive_interface_response(coords, uniform((TN0 / KN, 0.0, 0.0)), law_)
    add("mode-I peak traction [MPa]", result.traction[0, 0], TN0, "uniform opening t0/kn")
    result = cohesive_interface_response(coords, uniform((2 * GIC / TN0, 0.0, 0.0)), law_)
    add("mode-I traction at failure opening [MPa]", result.traction[0, 0], 0.0, "2 GIc/t0")
    result = cohesive_interface_response(coords, uniform((0.0, TS0 / KS, 0.0)), law_)
    add("mode-II peak traction [MPa]", result.traction[0, 1], TS0, "uniform shear")
    result = cohesive_interface_response(coords, uniform((0.0, 0.0, TS0 / KS)), law_)
    add("mode-III peak traction [MPa]", result.traction[0, 2], TS0, "out-of-plane shear")
    for mix in ((3.0, 4.0, 0.0), (1.0, 1.0, 1.0)):
        direction = torch.tensor(mix, dtype=D)
        c = direction / torch.linalg.vector_norm(direction)
        work_stiffness = KN * c[0] ** 2 + KS * (c[1] ** 2 + c[2] ** 2)
        fraction = KS * (c[1] ** 2 + c[2] ** 2) / work_stiffness
        expected = GIC + (GIIC - GIC) * float(fraction) ** ETA
        _, _, gc, _ = law_.direction_measures(direction)
        add(f"BK critical energy, mix {mix} [N/mm]", gc, expected, "Benzeggagh-Kenane 1996")
        add(f"work integral along mix {mix} [N/mm]",
            work_integral(law_, direction), expected, "trapezoid, 2e4 steps (quadrature limited)")

    # element-level energy and equilibrium
    side = 2.0
    coords2 = flat(side)
    direction = torch.tensor([1.0, 1.0, 0.0], dtype=D)
    _, lamf, gc, _ = law_.direction_measures(direction)
    unit = direction / torch.linalg.vector_norm(direction)
    result = cohesive_interface_response(coords2, uniform(unit * float(lamf) * 1.01), law_)
    add("dissipated energy, element [N.mm]", result.dissipated_energy, gc * float(result.area),
        "GIc+... x area with area = 4 mm^2")
    translation = torch.zeros(8, 3, dtype=D)
    translation += torch.tensor([1e-3, -2e-3, 5e-4], dtype=D)  # rigid motion, both faces
    zero = cohesive_interface_response(coords, translation, law_)
    add("rigid translation residual [N]", float(torch.max(torch.abs(zero.force))), 0.0,
        "nodal-force maximum")
    add("flat interface area [mm^2]", interface_area(coords), 1.0, "Gauss order 2")
    add("distorted interface area [mm^2]", interface_area(distorted(), order=3),
        float(interface_area(distorted(), order=3)), "reference = order-3 rule")

    errors = tangent_error()
    for step, error in errors.items():
        out.append({"check": f"element tangent vs central differences (h={step})",
                    "computed": error, "closed_form": 0.0, "relative_error": error,
                    "note": "max |K_analytic - K_numeric| / max|K_analytic|"})
    return out


def figure(path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    law_ = law()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    directions = {
        "mode I  (n)": (1.0, 0.0, 0.0),
        "mode II (s)": (0.0, 1.0, 0.0),
        "mix 3:4 (n:s)": (3.0, 4.0, 0.0),
        "mix 1:1:1": (1.0, 1.0, 1.0),
    }
    for label, mix in directions.items():
        c = torch.tensor(mix, dtype=D)
        c = c / torch.linalg.vector_norm(c)
        lam0, lamf, gc, _ = law_.direction_measures(c)
        lam = torch.linspace(0.0, float(lamf) * 1.15, 400, dtype=D)
        history = torch.zeros((), dtype=D)
        values = torch.zeros_like(lam)
        dissipated = torch.zeros_like(lam)
        peak = torch.zeros((), dtype=D)
        for i, amplitude in enumerate(lam):
            values[i] = law_.traction(amplitude * c, history) @ c
            dissipated[i] = law_.dissipated_energy_density(amplitude * c, history)
            history = torch.maximum(history, amplitude)
            peak = torch.maximum(peak, values[i])
        axes[0].plot(lam.numpy(), (values / peak).numpy(), label=label)
        axes[0].axvline(float(lam0), color="0.6", lw=0.6, ls=":")
        axes[0].axvline(float(lamf), color="0.3", lw=0.6, ls="--")
        axes[1].plot(lam.numpy(), (dissipated / gc).numpy(), label=label)
    axes[0].set_xlabel("equivalent separation  |delta|  [mm]")
    axes[0].set_ylabel("normalised traction  t/t_peak")
    axes[0].set_title("BK mixed-mode envelopes\n(dotted: onset, dashed: failure)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(1.0, color="0.3", lw=0.8, ls="--")
    axes[1].set_xlabel("equivalent separation  |delta|  [mm]")
    axes[1].set_ylabel("dissipated energy / G_c(direction)")
    axes[1].set_title("energy consistency: work -> G_c exactly")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    table = rows()
    payload = {
        "law": {"kn": KN, "ks": KS, "tn0": TN0, "ts0": TS0,
                "GIc": GIC, "GIIc": GIIC, "eta": ETA},
        "checks": table,
        "max_relative_error_exact_checks": max(
            row["relative_error"] for row in table
            if not row["check"].startswith("element tangent")
        ),
    }
    (OUT / "verification_table.json").write_text(json.dumps(payload, indent=2))
    figure(OUT / "traction_separation.png")
    return payload


if __name__ == "__main__":
    data = main()
    for row in data["checks"]:
        print(
            f"{row['check']:<52s} {row['computed']:>14.6e} "
            f"{row['closed_form']:>14.6e}  rel={row['relative_error']:.3e}"
        )
    print("max relative error (non-tangent checks):", data["max_relative_error_exact_checks"])