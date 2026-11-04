"""Generate the rotating-disk (centrifugal body load) verification evidence.

Run from the repository root::

    PYTHONPATH=src python3 scripts/verify_rotating_disk.py

Outputs
-------
``docs/benchmarks/tutorials/rotating-disk-verification.json``
    Every number quoted in ``docs/centrifugal-rotating-disk-verification.md``.
``docs/assets/rotating_disk_stress_profile.png``
    Hoop and radial stress along the meridian, FE element centres vs the
    Timoshenko closed form.
``docs/assets/rotating_disk_convergence.png``
    Relative error against the closed form versus element size, with the
    observed order fitted from the two finest meshes.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.rotating_disk import RotatingDiskReference, solve_rotating_disk

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / "docs" / "benchmarks" / "tutorials" / "rotating-disk-verification.json"
OUT_PROFILE = ROOT / "docs" / "assets" / "rotating_disk_stress_profile.png"
OUT_CONVERGENCE = ROOT / "docs" / "assets" / "rotating_disk_convergence.png"

PARAMS = dict(
    inner_radius=0.25,
    outer_radius=0.5,
    density=7850.0,
    omega=100.0,
    young_modulus=200e9,
    poisson_ratio=0.3,
    thickness=0.02,
)
MESHES = ((4, 24), (8, 48), (12, 72), (16, 96))


def _summary(record: dict) -> dict:
    keys = (
        "mesh", "elements", "nodes", "stress_rel_error", "peak_hoop_error",
        "peak_radial_error", "displacement_rel_error", "peak_displacement_error",
        "equilibrium_residual", "cut_balance_error", "energy_balance_error",
        "max_hoop_stress_fem", "max_hoop_stress_exact", "x_reaction",
        "analytic_resultant",
    )
    out = {k: record[k] for k in keys}
    out["load_resultant"] = record["load_resultant"]
    return out


def main() -> None:
    stress_runs = [solve_rotating_disk(n, m) for n, m in MESHES]
    strain_runs = [solve_rotating_disk(n, m, plane="strain") for n, m in MESHES[1:]]
    graded_runs = [
        solve_rotating_disk(n, 6 * n, inner_radius=0.025, grading=1.15) for n in (8, 12, 16)
    ]
    controls = {
        "mass_lumped_load": solve_rotating_disk(8, 48, lumped=True),
        "omega_mismatch_5pct": solve_rotating_disk(8, 48, omega_scale=1.05, reference_omega=100.0),
        "zero_speed": solve_rotating_disk(8, 48, omega=0.0),
    }

    # Observed convergence order from the two finest plane-stress meshes.
    h = [math.sqrt(PARAMS["outer_radius"] ** 2 - PARAMS["inner_radius"] ** 2) / n for n, _ in MESHES]
    e = [run["stress_rel_error"] for run in stress_runs]
    order = math.log(e[-2] / e[-1]) / math.log(h[-2] / h[-1])

    reference = RotatingDiskReference(
        PARAMS["inner_radius"], PARAMS["outer_radius"], PARAMS["density"],
        PARAMS["omega"], PARAMS["young_modulus"], PARAMS["poisson_ratio"], "stress",
    )
    r = torch.linspace(PARAMS["inner_radius"], PARAMS["outer_radius"], 200, dtype=torch.float64)
    exact_r, exact_theta = reference.stress(r)
    exact_u = reference.displacement(r)

    payload = {
        "case": "rotating annular disk, quarter model with symmetry planes",
        "parameters": PARAMS,
        "reference": "Timoshenko & Goodier, Theory of Elasticity, 3rd ed., Rotating disks",
        "plane_stress": [_summary(run) for run in stress_runs],
        "plane_strain_long_cylinder": [_summary(run) for run in strain_runs],
        "near_solid_disk_ab_0.05": [_summary(run) for run in graded_runs],
        "negative_controls": {k: _summary(v) for k, v in controls.items()},
        "observed_order_plane_stress": order,
        "peak_values": {
            "hoop_stress_exact_max": float(exact_theta.max()),
            "radial_stress_exact_max": float(exact_r.max()),
            "displacement_exact_max": float(exact_u.abs().max()),
            "hoop_stress_fem_max": stress_runs[-1]["max_hoop_stress_fem"],
        },
        "exact_profile": {
            "radius": [float(v) for v in r],
            "hoop": [float(v) for v in exact_theta],
            "radial": [float(v) for v in exact_r],
            "displacement": [float(v) for v in exact_u],
        },
        "all_within_three_percent": all(
            run["stress_rel_error"] < 0.03 and run["displacement_rel_error"] < 0.03
            for run in stress_runs + strain_runs + graded_runs
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    fine = stress_runs[-1]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.plot([v for v in payload["exact_profile"]["radius"]],
            [v / 1e6 for v in payload["exact_profile"]["hoop"]],
            "k-", lw=1.4, label="exact hoop $\\sigma_\\theta$")
    ax.plot([v for v in payload["exact_profile"]["radius"]],
            [v / 1e6 for v in payload["exact_profile"]["radial"]],
            "k--", lw=1.4, label="exact radial $\\sigma_r$")
    ax.plot(fine["cell_radius"], [v / 1e6 for v in fine["cell_hoop_stress"]],
            "o", ms=3.5, label="FE hoop (16x96 centre)")
    ax.plot(fine["cell_radius"], [v / 1e6 for v in fine["cell_radial_stress"]],
            "s", ms=3.0, mfc="none", label="FE radial (16x96 centre)")
    ax.set_xlabel("radius r [m]")
    ax.set_ylabel("stress [MPa]")
    ax.set_title("Rotating annulus a/b=0.5, $\\omega$=100 rad/s")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PROFILE, dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    ax.loglog(h, e, "o-", label="plane stress annulus")
    ax.loglog(h[1:], [run["stress_rel_error"] for run in strain_runs], "s-", label="plane strain")
    ax.loglog(h, [e[-1] * (h[-1] / hi) ** 2 for hi in h], "k:", lw=1.0, label="slope 2")
    ax.set_xlabel("element size h [m]")
    ax.set_ylabel("relative stress error")
    ax.set_title(f"Convergence, observed order {order:.2f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT_CONVERGENCE, dpi=110)
    plt.close(fig)
    print(json.dumps({
        "json": str(OUT_JSON),
        "figures": [str(OUT_PROFILE), str(OUT_CONVERGENCE)],
        "plane_stress_errors": e,
        "displacement_errors": [run["displacement_rel_error"] for run in stress_runs],
        "plane_strain_errors": [run["stress_rel_error"] for run in strain_runs],
        "graded_errors": [run["stress_rel_error"] for run in graded_runs],
        "controls": {k: v["stress_rel_error"] for k, v in controls.items()},
        "order": order,
        "peak": payload["peak_values"],
        "all_pass": payload["all_within_three_percent"],
    }, indent=2))


if __name__ == "__main__":
    main()
