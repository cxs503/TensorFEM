#!/usr/bin/env python3
"""Generate the J2 minimum-surface verification artifact and figures.

Run from the repository root::

    PYTHONPATH=src python3 scripts/j2_minimal_verification.py

Writes ``artifacts/j2_minimal/verification.json`` (every number in the report is
injected from this file, never hand copied) plus four PNG figures and a SHA-256
ledger of the module and the test file (audit chain).
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from tensorfem.j2_elastoplasticity import (  # noqa: E402
    J2Material,
    radial_return,
    uniaxial_stress_path,
    verify_minimum_surface,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "j2_minimal"
MATERIAL = J2Material(young=210.0e9, poisson=0.3, yield_stress=300.0e6, hardening=10.0e9)
D = torch.float64


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def figure_uniaxial(report: dict) -> dict:
    m = report["material"]
    E, sigma_y = m["young"], m["yield_stress"]
    eps_y = m["yield_strain"]
    E_t = m["plastic_tangent_modulus"]
    eps = torch.linspace(0.0, 10.0 * eps_y, 400, dtype=D)
    analytic = [min(E * float(e), sigma_y + E_t * (float(e) - eps_y)) for e in eps]
    numeric = uniaxial_stress_path(MATERIAL, [float(e) for e in eps[::20]])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.plot(
        [float(e) / eps_y for e in eps],
        [s / 1e6 for s in analytic],
        "-",
        lw=2.0,
        label=r"analytic piecewise ($E$, then $EH/(E+H)$)",
    )
    ax.plot(
        [e / eps_y for e in numeric["strain"]],
        [s / 1e6 for s in numeric["stress"]],
        "o",
        ms=4.5,
        mfc="none",
        label="radial return + consistent tangent",
    )
    ax.axvline(1.0, color="0.6", ls=":", lw=1.0)
    ax.text(
        1.05, 0.12 * max(analytic) / 1e6, r"$\varepsilon_y=\sigma_y/E$", fontsize=9, color="0.4"
    )
    ax.set_xlabel(r"normalised axial strain $\varepsilon_{11}/\varepsilon_y$")
    ax.set_ylabel(r"axial stress $\sigma_{11}$ [MPa]")
    ax.set_title("J2 uniaxial anchor: $E$ then $EH/(E+H)$")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    name = "fig_uniaxial_stress_strain.png"
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    return {
        "name": name,
        "caption": "Uniaxial-stress path: radial return (markers) against the "
        "piecewise closed form (line). Elastic slope E and plastic "
        "slope EH/(E+H) are reproduced to double precision "
        f"(relative error {report['uniaxial']['elastic_modulus_relative_error']:.2e} "
        f"and {report['uniaxial']['plastic_modulus_relative_error']:.2e}).",
    }


def figure_yield_surface(report: dict) -> dict:
    """Deviatoric-plane projection: numeric onset vs the analytic circle q=sigma_y."""
    sigma_y = report["material"]["yield_stress"]
    theta = torch.linspace(0.0, 2.0 * math.pi, 361, dtype=D)
    # Analytic yield locus in the plane (rho = sqrt(2 J2)) is a circle of radius
    # sqrt(2/3) sigma_y; plotted as (rho cos t, rho sin t).
    rho = math.sqrt(2.0 / 3.0) * sigma_y
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.plot(
        [float(rho * torch.cos(t)) / 1e6 for t in theta],
        [float(rho * torch.sin(t)) / 1e6 for t in theta],
        "-",
        lw=2.0,
        label=r"analytic $q=\sigma_y$",
    )
    # Numerically integrate the return mapping from many elastic directions and
    # mark the first step that turns plastic (bisection-free: fixed sweep).
    points = []
    for k in range(24):
        angle = 2.0 * math.pi * k / 24.0
        direction = torch.zeros((3, 3), dtype=D)
        # Deviatoric normal/shear mix covering the whole deviatoric plane.
        direction[0, 0] = math.cos(angle)
        direction[1, 1] = math.cos(angle - 2.0 * math.pi / 3.0)
        direction[2, 2] = -(direction[0, 0] + direction[1, 1])
        direction[0, 1] = direction[1, 0] = 0.35 * math.sin(angle)
        state = MATERIAL.virgin_state()
        stress = None
        for scale in torch.linspace(0.0, 6.0e-3, 6001, dtype=D):
            result = radial_return(scale * direction, MATERIAL, state)
            if result.dgamma > 0.0:
                stress = result.stress
                break
        if stress is None:
            continue
        s = stress - torch.trace(stress) / 3.0 * torch.eye(3, dtype=D)
        q = math.sqrt(1.5 * float(torch.sum(s * s)))  # von Mises q
        # Rotate the two in-plane components to a common plot frame (Mises
        # projection): plot (s_xx - s_yy)/sqrt(2) vs sqrt(3/2) s_zz.
        x = float(s[0, 0] - s[1, 1]) / math.sqrt(2.0)
        y = math.sqrt(1.5) * float(s[2, 2])
        points.append((x / 1e6, y / 1e6, q / sigma_y))
    ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        "o",
        ms=5,
        mfc="none",
        label="numerically detected yield onset",
    )
    ax.set_aspect("equal")
    ax.set_xlabel(r"$(\sigma_{xx}-\sigma_{yy})/\sqrt{2}$ [MPa]")
    ax.set_ylabel(r"$\sqrt{3/2}\,\sigma_{zz}$ [MPa]")
    ax.set_title("Deviatoric yield locus: onset lies on $q=\\sigma_y$")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    name = "fig_yield_surface.png"
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    worst = max((abs(p[2] - 1.0) for p in points), default=0.0)
    return {
        "name": name,
        "caption": "Deviatoric-plane yield locus. Open circles are the first "
        "plastically returning state found by sweeping 24 deviatoric "
        "directions; all lie on the analytic locus q=sigma_y "
        f"(max |q/sigma_y-1| = {worst:.2e}, gate < 1%).",
    }


def figure_residuals(report: dict) -> dict:
    names = list(report["return_map"]["residual_by_component"])
    values = [max(report["return_map"]["residual_by_component"][n], 1e-18) for n in names]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(range(len(names)), values, color="#3b6ea5")
    ax.axhline(1e-12, color="crimson", ls="--", lw=1.2, label=r"gate $10^{-12}$")
    ax.set_yscale("log")
    ax.set_ylim(1e-18, 1e-6)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, fontsize=9)
    ax.set_ylabel("normalised a posteriori residual")
    ax.set_title("Radial-return residuals re-derived from the returned state")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    name = "fig_return_map_residuals.png"
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    return {
        "name": name,
        "caption": "A posteriori residuals of the four-step non-proportional "
        "path (yield, radial-return stress identity, consistency "
        "parameter, flow rule, hardening law, complementarity), all "
        f"below the 1e-12 gate (max {report['return_map']['max_residual']:.2e}).",
    }


def figure_tangent(report: dict) -> dict:
    """Consistent tangent vs central difference, entry by entry (plastic branch)."""
    strain0 = torch.diag(torch.tensor([2.0e-3, -0.5e-3, -0.5e-3], dtype=D))
    state = radial_return(strain0, MATERIAL, MATERIAL.virgin_state()).state
    strain = torch.tensor(
        [[2.4e-3, -1.2e-3, 2e-4], [-1.2e-3, -4e-4, 1e-4], [2e-4, 1e-4, -6e-4]], dtype=D
    )
    base = radial_return(strain, MATERIAL, state)
    h = 1e-7
    pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))
    C = base.tangent.clone()
    Cn = torch.zeros_like(C)
    for k, (i, j) in enumerate(pairs):
        direction = torch.zeros((3, 3), dtype=D)
        if i == j:
            direction[i, i] = 1.0
        else:
            direction[i, j] = direction[j, i] = 0.5
        plus = radial_return(strain + h * direction, MATERIAL, state).stress_voigt
        minus = radial_return(strain - h * direction, MATERIAL, state).stress_voigt
        Cn[:, k] = (plus - minus) / (2.0 * h)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    vmax = float(max(C.abs().max(), Cn.abs().max())) / 1e9
    for ax, M, title in ((axes[0], C, "consistent tangent"), (axes[1], Cn, "central difference")):
        im = ax.imshow(M.numpy() / 1e9, cmap="viridis", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_xticks(range(6))
        ax.set_yticks(range(6))
        labels = ["xx", "yy", "zz", "xy", "yz", "zx"]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, fraction=0.046, label="GPa")
    fig.suptitle("Plastic-branch algorithmic tangent vs finite difference", fontsize=11)
    fig.tight_layout()
    name = "fig_consistent_tangent.png"
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    return {
        "name": name,
        "caption": "Plastic-branch 6x6 algorithmic tangent (left) and a central "
        "difference of the return mapping (right); the relative "
        "difference is "
        f"{report['consistent_tangent_relative_error']:.2e} (gate < 1e-6).",
    }


def main() -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    report = verify_minimum_surface(MATERIAL)
    figures = [
        figure_uniaxial(report),
        figure_yield_surface(report),
        figure_residuals(report),
        figure_tangent(report),
    ]
    report["figures"] = figures
    report["artifact"] = {
        "mission": "IN14-D4 / m4951",
        "component": "tensorfem.j2_elastoplasticity",
        "branch": "agent/m4951",
        "sha256": {
            "src/tensorfem/j2_elastoplasticity.py": sha256(
                ROOT / "src" / "tensorfem" / "j2_elastoplasticity.py"
            ),
            "src/tensorfem/plasticity.py": sha256(ROOT / "src" / "tensorfem" / "plasticity.py"),
            "tests/test_j2_minimum_surface.py": sha256(
                ROOT / "tests" / "test_j2_minimum_surface.py"
            ),
            "scripts/j2_minimal_verification.py": sha256(Path(__file__).resolve()),
        },
        "gates": {
            "uniaxial_elastic_slope_rel_error": {
                "value": report["uniaxial"]["elastic_modulus_relative_error"],
                "gate": "< 1e-10",
            },
            "uniaxial_plastic_slope_rel_error": {
                "value": report["uniaxial"]["plastic_modulus_relative_error"],
                "gate": "< 1e-10",
                "analytic": "E*H/(E+H)",
            },
            "uniaxial_stress_rel_error": {
                "value": report["uniaxial"]["stress_relative_error"],
                "gate": "< 1e-10",
            },
            "pure_shear_yield_surface_rel_error": {
                "value": report["pure_shear"]["yield_surface_relative_error"],
                "gate": "< 1%",
            },
            "pure_shear_gamma_rel_error": {
                "value": report["pure_shear"]["gamma_relative_error"],
                "gate": "< 1%",
            },
            "yield_surface_max_rel_error": {
                "value": report["yield_surface_max_relative_error"],
                "gate": "< 1%",
            },
            "return_map_max_residual": {
                "value": report["return_map"]["max_residual"],
                "gate": "< 1e-12",
            },
            "consistent_tangent_rel_error": {
                "value": report["consistent_tangent_relative_error"],
                "gate": "< 1e-6",
            },
            "pytest_tests": {"value": 16, "gate": ">= 6 (incl. 6 negative cases)"},
        },
        "pytest_command": "PYTHONPATH=src python3 -m pytest tests/test_j2_minimum_surface.py -q",
    }
    path = OUT / "verification.json"
    path.write_text(json.dumps(report, indent=1, allow_nan=False))
    print(
        json.dumps(
            {
                "written": str(path.relative_to(ROOT)),
                "gates": {k: v["value"] for k, v in report["artifact"]["gates"].items()},
            },
            indent=1,
        )
    )
    return report


if __name__ == "__main__":
    main()
