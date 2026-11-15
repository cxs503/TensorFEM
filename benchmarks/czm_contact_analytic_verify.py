#!/usr/bin/env python3
"""Analytical verification of the CZM contact solve (spherical ice edge).

Closes the verification debt flagged by the GAP-PP-1 adversarial audit
(``docs/AUDIT-GAP-PP-1-adversarial-verification.md``, section 4): the physical
correctness of the CZM contact solver itself (``czm_contact_patch``) had never
been checked against an independent reference -- only force balance, the
ceiling, monotonicity and the trivial flat-edge limit were tested.

Reference problem (closed form, derived here and in the test docstrings)
--------------------------------------------------------------------------
A spherical ice edge of radius ``R`` and cap amplitude ``amp`` has relief
``s(r) = amp * (r/R)^2`` (``r`` from the edge centre).  The local opening is
``g(r) = delta0 - s(r)`` and the undamaged compressive branch of the bilinear
cohesive law is the linear penalty ``p = k * g`` (``g > 0``), capped by the
crushing ceiling.  For ``delta0 < amp`` and no active ceiling the contact is a
circle of radius ``a = R * sqrt(delta0/amp)`` with the *parabolic* Winkler
pressure

    p(r) = p0 * (1 - r^2/a^2),   p0 = k * delta0,

    F = pi * k * R^2 * delta0^2 / (2 * amp),   mean = F / (pi a^2) = p0 / 2.

This is the standard paraboloid-on-Winkler-foundation contact (the Hertz
analogue with a linear foundation instead of an elastic half-space).  The
solver's bisection + quadrature is therefore checked against a hand-derived
integral, not against itself.

Usage
-----
    python benchmarks/czm_contact_analytic_verify.py \
        --out benchmarks/artifacts/GAP-PP-1-czm-contact-analytic
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfem.ice_panel_patch_contact import (  # noqa: E402
    czm_contact_patch,
    default_law,
)

PANEL = (0.8, 0.7)
CENTRE = (0.4, 0.35)


def analytic_uncapped(k: float, R: float, amp: float, delta0: float) -> dict:
    """Closed-form paraboloid-on-Winkler solution (ceiling inactive)."""
    p0 = k * delta0
    a = R * math.sqrt(delta0 / amp)
    F = math.pi * k * R**2 * delta0**2 / (2.0 * amp)
    return {
        "delta0_m": delta0,
        "peak_pressure_Pa": p0,
        "contact_radius_m": a,
        "contact_area_m2": math.pi * a**2,
        "total_force_N": F,
        "mean_pressure_Pa": p0 / 2.0,
        "peak_over_mean": 2.0,
    }


def analytic_capped(k: float, R: float, amp: float, delta0: float, ceiling: float) -> dict:
    """Closed-form solution with the crushing ceiling active (p0 > ceiling)."""
    p0 = k * delta0
    a = R * math.sqrt(delta0 / amp)
    rc = R * math.sqrt((delta0 - ceiling / k) / amp)
    F_cap = math.pi * ceiling * rc**2
    F_tail = 2.0 * math.pi * k * (
        delta0 * (a**2 - rc**2) / 2.0 - amp * (a**4 - rc**4) / (4.0 * R**2)
    )
    return {
        "delta0_m": delta0,
        "peak_pressure_Pa": ceiling,
        "contact_radius_m": a,
        "capped_radius_m": rc,
        "capped_area_m2": math.pi * rc**2,
        "total_force_N": F_cap + F_tail,
    }


def _profile_error(patch, p0: float, a: float, centre: tuple[float, float]) -> dict:
    """Max/L2 relative error of the solved pressure vs the analytic parabola."""
    xg = patch.x.numpy()
    yg = patch.y.numpy()
    dX, dY = np.meshgrid(xg - centre[0], yg - centre[1], indexing="ij")
    r2 = dX**2 + dY**2
    p_ref = np.where(r2 < a**2, p0 * (1.0 - r2 / a**2), 0.0)
    p_num = patch.pressure.numpy()
    inside = p_ref > 0.05 * p0  # skip the near-zero rim (relative error ill-conditioned)
    rel = np.abs(p_num - p_ref) / np.maximum(p_ref, 1e-30)
    l2 = math.sqrt(float(((p_num - p_ref) ** 2).sum())) / math.sqrt(float((p_ref**2).sum()))
    return {
        "max_rel_error_interior": float(rel[inside].max()),
        "l2_rel_error": l2,
        "n_interior_points": int(inside.sum()),
    }


def run_case(grid: int) -> dict:
    """One-way CZM contact solve for the uncapped spherical edge + analytic check."""
    law = default_law()
    k = float(law.stiffness)
    R, amp, half = 1.0, 0.02, 0.10
    delta0 = 1.5e-4  # p0 = 0.6 MPa < 2 MPa ceiling; a = 0.0866 m < 0.10 box half
    ref = analytic_uncapped(k, R, amp, delta0)
    edge = {"kind": "spherical", "radius": R, "amplitude": amp, "box_half": half}
    patch = czm_contact_patch(
        law=law,
        ceiling=2.0e6,
        edge=edge,
        total_force=ref["total_force_N"],
        panel=PANEL,
        centre=CENTRE,
        grid=grid,
    )
    prof = _profile_error(patch, ref["peak_pressure_Pa"], ref["contact_radius_m"], CENTRE)
    return {
        "grid": grid,
        "analytic": ref,
        "solved": {
            "total_force_N": patch.total_force,
            "peak_pressure_Pa": patch.peak_pressure,
            "contact_area_m2": patch.contact_area,
            "mean_pressure_Pa": patch.total_force / patch.contact_area,
            "peak_over_mean": patch.peak_pressure / (patch.total_force / patch.contact_area),
            "approach_m": patch.approach,
            "residual": patch.residual,
        },
        "errors": {
            "peak_pressure_rel": patch.peak_pressure / ref["peak_pressure_Pa"] - 1.0,
            "contact_area_rel": patch.contact_area / ref["contact_area_m2"] - 1.0,
            "force_rel": patch.total_force / ref["total_force_N"] - 1.0,
            "peak_over_mean_minus_2": patch.peak_pressure
            / (patch.total_force / patch.contact_area) - 2.0,
            **{f"profile_{kk}": vv for kk, vv in prof.items() if kk != "n_interior_points"},
        },
        "n_interior_points": prof["n_interior_points"],
    }


def run_capped_case(grid: int = 97) -> dict:
    """Ceiling-active case: R = 0.4, amp = 0.02, delta0 = 1e-3 -> p0 = 4 MPa > 2 MPa."""
    law = default_law()
    k = float(law.stiffness)
    R, amp, half, ceiling = 0.4, 0.02, 0.10, 2.0e6
    delta0 = 1.0e-3
    ref = analytic_capped(k, R, amp, delta0, ceiling)
    edge = {"kind": "spherical", "radius": R, "amplitude": amp, "box_half": half}
    patch = czm_contact_patch(
        law=law,
        ceiling=ceiling,
        edge=edge,
        total_force=ref["total_force_N"],
        panel=PANEL,
        centre=CENTRE,
        grid=grid,
    )
    xg = patch.x.numpy()
    yg = patch.y.numpy()
    dX, dY = np.meshgrid(xg - CENTRE[0], yg - CENTRE[1], indexing="ij")
    r = np.sqrt(dX**2 + dY**2)
    p_num = patch.pressure.numpy()
    capped = r < ref["capped_radius_m"] * 0.9  # strictly inside the flat top
    return {
        "grid": grid,
        "analytic": ref,
        "solved": {
            "total_force_N": patch.total_force,
            "peak_pressure_Pa": patch.peak_pressure,
            "contact_area_m2": patch.contact_area,
            "residual": patch.residual,
        },
        "errors": {
            "peak_pressure_rel": patch.peak_pressure / ceiling - 1.0,
            "force_rel": patch.total_force / ref["total_force_N"] - 1.0,
            "capped_flat_top_max_dev": float(np.abs(p_num[capped] - ceiling).max())
            if capped.any() else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[1]
            / "benchmarks" / "artifacts" / "GAP-PP-1-czm-contact-analytic"
        ),
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    grids = [49, 97, 193]
    rows = [run_case(g) for g in grids]
    capped = run_capped_case()

    base = rows[1]  # reference grid
    result = {
        "case": "GAP-PP-1-ICE-STRUCT-CONTACT-PATCH: CZM contact solve vs analytic paraboloid",
        "reference": (
            "paraboloid indenter on a Winkler (linear penalty) foundation: "
            "p(r) = p0*(1 - r^2/a^2), F = pi*k*R^2*delta0^2/(2*amp), mean = p0/2"
        ),
        "parameters": {
            "stiffness_Pa_per_m": float(default_law().stiffness),
            "ceiling_Pa": 2.0e6,
            "edge": {"kind": "spherical", "radius": 1.0, "amplitude": 0.02, "box_half": 0.10},
            "delta0_m": 1.5e-4,
            "panel_m": list(PANEL),
            "centre_m": list(CENTRE),
        },
        "grid_convergence": rows,
        "capped_case": capped,
        "pass": all(
            abs(r["errors"]["peak_pressure_rel"]) < 0.01
            and abs(r["errors"]["contact_area_rel"]) < 0.02
            and abs(r["errors"]["peak_over_mean_minus_2"]) < 0.05
            for r in rows
        )
        and abs(capped["errors"]["peak_pressure_rel"]) < 1e-9
        and abs(capped["errors"]["force_rel"]) < 1e-6,
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nPASS={result['pass']}  artifact: {out / 'verification.json'}")


if __name__ == "__main__":
    main()
