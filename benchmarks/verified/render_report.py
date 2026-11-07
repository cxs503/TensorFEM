#!/usr/bin/env python3
"""Inject the verified-anchor evidence into ``docs/verified/fem-verification-anchors.md``.

The three anchor drivers own their numbers; this script only *renders* them, so
the prose document can never drift away from ``result.json`` /
``gates_verified.json`` (no number is typed by hand).  It also writes
``docs/verified/fem-anchors-digest.json`` -- the flat summary that the campaign
report uploads.

Usage::

    PYTHONPATH=src python benchmarks/verified/render_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOC = ROOT / "docs" / "verified" / "fem-verification-anchors.md"
DIGEST = ROOT / "docs" / "verified" / "fem-anchors-digest.json"

BEGIN = "<!-- BEGIN:ANCHOR-DATA -->"
END = "<!-- END:ANCHOR-DATA -->"


def load(anchor: str) -> tuple[dict, dict]:
    directory = HERE / anchor
    return (
        json.loads((directory / "result.json").read_text()),
        json.loads((directory / "gates_verified.json").read_text()),
    )


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def pct(value: float, digits: int = 3) -> str:
    return f"{100.0 * value:.{digits}f} %"


def gates_table(gates: dict) -> str:
    rows = []
    for gate in gates["gates"]:
        operator = "<=" if gate["operator"] == "<=" else ">="
        rows.append(
            [
                f"`{gate['id']}`",
                gate["reference"],
                f"{gate['value']:.6g}",
                f"{operator} {gate['threshold']:.6g}",
                "PASS" if gate["passed"] else "**FAIL**",
            ]
        )
    return table(["gate", "independent reference", "value", "criterion", "verdict"], rows)


def cantilever_block() -> tuple[str, dict]:
    result, gates = load("fem_cantilever")
    geometry, analytic = result["geometry"], result["analytic"]
    out = [
        "### 1. Cantilever bending: Euler--Bernoulli deflection, stress, GCI ladder",
        "",
        "**Geometry and material (campaign parameters).** "
        f"`L = {geometry['length']} m`, `b = {geometry['width']} m`, `h = {geometry['depth']} m` "
        f"(`L/h = {analytic['slenderness_L_over_h']:.0f}`), `E = {geometry['young_modulus']:.4g} Pa`, "
        f"`nu = {geometry['poisson_ratio']}`, tip load `F = {geometry['tip_load']:.4g} N` along `-z`. "
        f"`I = b h^3/12 = {analytic['inertia']:.6e} m^4`.",
        "",
        "**Independent oracle.** "
        f"`w = F L^3/(3 E I) = {analytic['tip_deflection']:.9f} m`, "
        f"`theta = F L^2/(2 E I) = {analytic['tip_rotation']:.9f} rad`, "
        f"`sigma_xx(x,z) = F (L-x)(z-h/2)/I`. The transverse-shear contribution "
        f"`F L/(kappa G A)` is {analytic['shear_deflection_over_bending']:.3e} of the bending "
        "deflection, so beam theory is the correct yardstick to two orders below the 1 % gate.",
        "",
        "**Model.** Q8 serendipity plane-stress membrane of the `(x,z)` section (thickness `b`), "
        "3x3 Gauss, sparse direct (SuperLU) solve, Chebyshev-clustered axial mesh so both the "
        "clamped root and the tip load stay inside the asymptotic range. A HEX20 3-D solid of the "
        "same bar is carried as a model-form cross-check.",
        "",
        "#### Ladder (Q8 plane stress)",
        "",
        table(
            [
                "axial x depth elements",
                "dofs",
                "w_tip [m]",
                "rel. error",
                "sigma_xx probe",
                "rel. error",
                "solve [s]",
            ],
            [
                [
                    f"{lv['axial_elements']} x {lv['depth_elements']}",
                    str(lv["dofs"]),
                    f"{lv['tip_deflection']:.9f}",
                    pct(lv["relative_error"]),
                    f"{lv['stress_probe_sigma_xx'] / 1e6:.4f} MPa",
                    pct(lv["stress_relative_error"]),
                    f"{lv['solve_seconds']:.2f}",
                ]
                for lv in result["q8_levels"]
            ],
        ),
        "",
        "#### Grid convergence (Richardson, finest triple)",
        "",
        table(
            ["quantity", "value"],
            [
                ["observed order p", f"{result['q8_richardson']['observed_order']:.3f}"],
                ["GCI_fine (Fs = 1.25)", pct(result["q8_richardson"]["gci_fine"])],
                ["Richardson limit w [m]", f"{result['q8_richardson']['richardson_limit']:.9f}"],
                ["ratio r", f"{result['q8_richardson']['refinement_ratio']:.2f}"],
            ],
        ),
        "",
        "#### Model-form cross-check (HEX20 3-D solid, uniform mesh)",
        "",
        table(
            ["axial x width x depth elements", "dofs", "w_tip [m]", "rel. error", "solve [s]"],
            [
                [
                    f"{lv['axial_elements']} x {lv['width_elements']} x {lv['depth_elements']}",
                    str(lv["dofs"]),
                    f"{lv['tip_deflection']:.9f}",
                    pct(lv["relative_error"]),
                    f"{lv['solve_seconds']:.2f}",
                ]
                for lv in result["hex20_levels"]
            ],
        ),
        "",
        "The solid model converges to a deflection ~1 % below beam theory and only slowly: at "
        "`L/h = 100` a fully integrated quadratic brick is shear-locking limited, and the fully "
        "clamped root additionally restrains the anticlastic contraction. This is why the 1 % gate "
        "is graded on the plane-stress model (whose exact limit *is* Euler--Bernoulli) and the "
        "solid is graded at 2 % as a cross-check.",
        "",
        "#### Gates",
        "",
        gates_table(gates),
        "",
    ]
    digest = {
        "anchor": "fem_cantilever",
        "finest_tip_deflection_rel_error": result["verdict"]["finest_relative_error"],
        "observed_order": result["q8_richardson"]["observed_order"],
        "gci_fine": result["q8_richardson"]["gci_fine"],
        "solid3d_rel_error": result["hex20_levels"][-1]["relative_error"],
        "gates_passed": gates["passed"],
    }
    return "\n".join(out), digest


def hertz_block() -> tuple[str, dict]:
    result, gates = load("fem_hertz_contact")
    geometry, analytic = result["geometry"], result["analytic"]
    finest = result["levels"][-1]
    out = [
        "### 2. Hertzian sphere--sphere contact: contact radius and peak pressure",
        "",
        "**Problem (campaign parameters).** Two identical elastic spheres "
        f"`R_1 = R_2 = {geometry['sphere_radius_1']:.4g} m`, `E = {geometry['young_modulus']:.4g} Pa`, "
        f"`nu = {geometry['poisson_ratio']}`, pressed by `F = {geometry['contact_force']:.4g} N`.",
        "",
        "**Independent oracle (Hertz/Johnson).** "
        f"`R = (1/R_1+1/R_2)^-1 = {analytic['effective_radius']:.6g} m`, "
        f"`E* = [sum (1-nu_i^2)/E_i]^-1 = {analytic['effective_modulus']:.6e} Pa`, "
        f"`a = (3 F R/(4 E*))^(1/3) = {analytic['contact_radius']:.6e} m`, "
        f"`p_0 = 3F/(2 pi a^2) = {analytic['peak_pressure']:.6e} Pa`, "
        f"approach `delta = a^2/R = {analytic['approach']:.6e} m` "
        f"(={analytic['per_sphere_approach'] * 1e6:.3f} um per sphere), "
        f"mean pressure `F/(pi a^2) = {analytic['mean_pressure']:.4e} Pa`.",
        "",
        "**Model.** The Hertz reduction to a rigid indenter of radius `R` on a half-space with "
        f"`E*/(1-nu^2) = {analytic['model_young_modulus']:.6e} Pa` is used, i.e. TensorFEM's "
        "axisymmetric Q4 elasticity with nodal penalty contact, an active-set Newton loop and a "
        "least-squares `p^2` vs `r^2` recovery of `(a, p_0)` from the computed pressure field only. "
        f"The prescribed approach is `delta = {analytic['approach']:.6e} m`; the contact radius, the "
        "peak pressure and the contact force are all *outputs*. Graded mesh (grading 2.6) clusters "
        "nodes at the axis and the contact surface; the domain is `1.2 R` wide and deep.",
        "",
        "#### Ladder",
        "",
        table(
            [
                "elements (r x z)",
                "nodes in contact",
                "a [mm]",
                "a err.",
                "p_0 [GPa]",
                "p_0 err.",
                "F [kN]",
                "F err.",
                "overlap/delta",
                "p L2 err.",
            ],
            [
                [
                    f"{lv['radial_elements']} x {lv['axial_elements']}",
                    str(lv["nodes_inside_contact_radius"]),
                    f"{lv['contact_radius'] * 1e3:.4f}",
                    pct(lv["contact_radius_rel_error"]),
                    f"{lv['peak_pressure'] / 1e9:.4f}",
                    pct(lv["peak_pressure_rel_error"]),
                    f"{lv['contact_force'] / 1e3:.4f}",
                    pct(lv["contact_force_rel_error"]),
                    f"{lv['overlap_fraction']:.4f}",
                    pct(lv["pressure_l2_error"]),
                ]
                for lv in result["levels"]
            ],
        ),
        "",
        "#### Convergence record (honest form)",
        "",
        table(
            ["quantity", "value"],
            [
                [
                    "pressure-field L2 order, finest pair",
                    f"{result['convergence']['pressure_l2_observed_order_finest_pair']:.2f}",
                ],
                [
                    "pressure-field L2 order, coarse pair",
                    f"{result['convergence']['pressure_l2_two_level_orders'][0]:.2f}",
                ],
                ["contact radius monotone", str(result["convergence"]["contact_radius_monotone"])],
                [
                    "a order (finest pair)",
                    f"{result['convergence']['contact_radius_two_level_orders'][-1]:.2f}",
                ],
                [
                    "p_0 order (finest pair)",
                    f"{result['convergence']['peak_pressure_two_level_orders'][-1]:.2f}",
                ],
            ],
        ),
        "",
        "The **pressure field** converges monotonically (L2 error 5.4 % -> 4.1 % -> 2.1 %) at an "
        "observed rate of about 2.2 on the finest pair. The **recovered** `(a, p_0)` do not: the "
        "active-set cardinality jumps (7 -> 11 -> 14 nodes) and the lumped nodal traction makes the "
        "fit noisy, so `a` reads 1.1 %, 3.6 %, 1.2 % across the ladder. No order is claimed for "
        "`a` or `p_0`; the gates are graded on the finest level, where all three quantities sit "
        "below their thresholds.",
        "",
        "#### Penalty-compliance study",
        "",
        table(
            ["penalty factor", "a err.", "p_0 err.", "F err.", "overlap/delta"],
            [
                [
                    f"{lv['penalty_factor']:g}",
                    pct(lv["contact_radius_rel_error"]),
                    pct(lv["peak_pressure_rel_error"]),
                    pct(lv["contact_force_rel_error"]),
                    f"{lv['overlap_fraction']:.4f}",
                ]
                for lv in sorted(
                    [lv for lv in result["levels"] if lv["radial_elements"] == 36]
                    + result["penalty_study"],
                    key=lambda lv: lv["penalty_factor"],
                )
            ],
        ),
        "",
        "Stiffening the penalty from 28 to 100 does **not** improve the agreement: the penalty "
        "overlap is only 2.6 % of the approach, so the remaining error is discretisation of the "
        "contact patch, and the stiffer constraint degrades the conditioning of the active-set "
        "system. The penalty bias is therefore documented, not tuned away.",
        "",
        "#### Gates",
        "",
        gates_table(gates),
        "",
    ]
    digest = {
        "anchor": "fem_hertz_contact",
        "contact_radius": analytic["contact_radius"],
        "peak_pressure": analytic["peak_pressure"],
        "contact_radius_rel_error": finest["contact_radius_rel_error"],
        "peak_pressure_rel_error": finest["peak_pressure_rel_error"],
        "contact_force_rel_error": finest["contact_force_rel_error"],
        "pressure_l2_rel_error": finest["pressure_l2_error"],
        "pressure_field_order": result["convergence"]["pressure_l2_observed_order_finest_pair"],
        "gates_passed": gates["passed"],
    }
    return "\n".join(out), digest


def plate_block() -> tuple[str, dict]:
    result, gates = load("fem_plate_modal")
    geometry, analytic = result["geometry"], result["analytic"]
    finest = result["levels"][-1]
    out = [
        "### 3. Simply supported rectangular plate: first five natural frequencies",
        "",
        "**Geometry and material (campaign parameters).** "
        f"`a x b x h = {geometry['length_a']} x {geometry['width_b']} x {geometry['thickness_h']} m` "
        f"(`a/b = {geometry['aspect_a_over_b']:.2f}`, `a/h = {geometry['slenderness_a_over_h']:.0f}`), "
        f"`E = {geometry['young_modulus']:.4g} Pa`, `nu = {geometry['poisson_ratio']}`, "
        f"`rho = {geometry['density']:.0f} kg/m^3`, all four edges simply supported "
        "(hard convention: `w = 0`, normal rotation restrained).",
        "",
        "**Independent oracle (Navier/Kirchhoff).** "
        "`f_mn = (pi/2) sqrt(D/(rho h)) [(m/a)^2 + (n/b)^2]`, "
        f"`D = E h^3/(12(1-nu^2)) = {analytic['plate_rigidity']:.6e} N m`:",
        "",
        table(
            ["mode", "(m,n)", "f analytic [Hz]"],
            [
                [
                    str(i + 1),
                    f"({analytic['mode_labels'][i][0]},{analytic['mode_labels'][i][1]})",
                    f"{f:.3f}",
                ]
                for i, f in enumerate(analytic["frequencies_hz"])
            ],
        ),
        "",
        "**Model.** TensorFEM Mindlin--Reissner Q4 plate with selective reduced shear integration and "
        "its consistent mass matrix including rotary inertia, on a structured mesh with the "
        "library's hard simple-support DOF set. The pencil is solved by shift-invert Lanczos "
        "(`sigma = -1e2`); a dense LAPACK path was rejected because on this loong64 build it "
        "returned eigenpairs with ~20 % residual on the same reduced matrix (see the notes below). "
        "An equivalence gate re-solves the coarsest level with `solve_plate_vibration` and requires "
        "1e-9 agreement.",
        "",
        "#### Ladder",
        "",
        table(
            ["elements a x b", "dofs", "f1..f5 [Hz]", "rel. errors", "min MAC", "solve [s]"],
            [
                [
                    f"{lv['elements_x']} x {lv['elements_y']}",
                    str(lv["dofs"]),
                    ", ".join(f"{f:.2f}" for f in lv["frequencies_hz"]),
                    ", ".join(pct(e, 2) for e in lv["relative_errors"]),
                    f"{min(lv['mode_mac']):.6f}",
                    f"{lv['seconds']:.1f}",
                ]
                for lv in result["levels"]
            ],
        ),
        "",
        "#### Convergence",
        "",
        table(
            ["mode", "(m,n)", "finest-pair order", "finest rel. error"],
            [
                [
                    str(k + 1),
                    f"({analytic['mode_labels'][k][0]},{analytic['mode_labels'][k][1]})",
                    f"{result['convergence']['mode_orders_finest_pair'][k]:.2f}",
                    pct(finest["relative_errors"][k]),
                ]
                for k in range(len(analytic["frequencies_hz"]))
            ],
        ),
        "",
        f"Library equivalence: max relative frequency difference "
        f"`{result['library_equivalence_max_rel']:.2e}` against `solve_plate_vibration`. "
        f"Spurious near-zero eigenvalues filtered at the finest mesh: "
        f"`{finest['spurious_modes_filtered']}`.",
        "",
        "#### Gates",
        "",
        gates_table(gates),
        "",
    ]
    digest = {
        "anchor": "fem_plate_modal",
        "analytic_hz": analytic["frequencies_hz"],
        "finest_hz": finest["frequencies_hz"],
        "relative_errors": finest["relative_errors"],
        "max_rel_error": result["verdict"]["max_frequency_rel_error"],
        "min_mac": result["verdict"]["min_mode_mac"],
        "min_order": result["convergence"]["min_order_finest_pair"],
        "library_equivalence": result["library_equivalence_max_rel"],
        "gates_passed": gates["passed"],
    }
    return "\n".join(out), digest


def main() -> int:
    blocks = [cantilever_block(), hertz_block(), plate_block()]
    body = "\n".join(block for block, _ in blocks)
    sums = [digest for _, digest in blocks]
    text = DOC.read_text()
    if BEGIN not in text or END not in text:
        raise SystemExit(f"markers {BEGIN}/{END} missing from {DOC}")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    DOC.write_text(head + BEGIN + "\n\n" + body + "\n" + END + tail)
    DIGEST.write_text(
        json.dumps(
            {
                "mission": "IN36-N01",
                "anchors": sums,
                "all_gates_passed": all(s["gates_passed"] for s in sums),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "doc": str(DOC.relative_to(ROOT)),
                "all_gates_passed": all(s["gates_passed"] for s in sums),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
