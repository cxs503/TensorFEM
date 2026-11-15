"""Nonlinear-FE cross-validation of the closed-form ice-grillage ultimate.

Why this module exists
----------------------
:mod:`tensorfem.grillage_ice_ultimate` closes the ``GAP-PP-5-ICE-STRUCT-
GRILLAGE-ULTIMATE`` "last jump" (plate-field bending stress -> structural
ultimate) with a **closed-form / empirical** chain: the plate local-bending
capacity is the plastic-hinge moment ``m_p = sigma_y t^2/4`` (degraded by the
coexisting membrane compression through the von Mises factor
``sqrt(1 - eta^2)``), and the ultimate ice pressure is the value at which the
Navier-series peak bending moment equals that capacity.  The acceptance item
explicitly allows *either* a nonlinear solve *or* an empirical formula, but the
closed-form branch has never been cross-checked against an independent
**nonlinear finite-element** collapse of the same plate.  That is the
verification debt this module repays.

What it does
------------
For each of the three documented combined cases (light / design / severe ice
patch + hydrostatic head), it builds a **bare, simply-supported plate** of the
representative loaded panel (``frame_spacing x bay_spacing``, ``t = 20 mm``,
EH36 ``sigma_y = 355 MPa``) with the layered elasto-plastic von-Karman
``PlateModel`` of :mod:`tensorfem.ice_belt_collapse`, applies the **combined
lateral** load (centred ice patch + uniform hydrostatic) as the arc-length
reference, and traces the plastic collapse path with the Crisfield spherical
arc-length continuation.  The FE peak load factor ``lam_peak`` is the load
multiplier at which the plate actually collapses.

The cross-check asserts the **conservatism** of the closed-form "last jump":
the closed-form ``plate_local_bending`` capacity is a *first-yield* criterion
(the Navier peak bending moment reaches the plastic-hinge moment), which is a
**lower bound** on the true yield-line collapse.  The FE collapse must therefore
be **at or above** the closed-form prediction (``lam_peak >= 1/u_plate_eta0``);
an over-prediction would be an unsafe design formula.  The ratio
``lam_peak / (1/u_plate_eta0)`` is the quantified conservatism margin.

Scope and honesty
-----------------
* The FE model is **lateral-only**: the coexisting membrane compression
  (``eta = sigma_x / sigma_y``) is *not* applied as an in-plane pre-load in the
  FE trace, so ``lam_peak`` is compared against the closed-form **undegraded**
  capacity ``m_p = sigma_y t^2/4`` (``u_plate_eta0``).  The axial degradation
  ``sqrt(1 - eta^2)`` is a documented von-Mises interaction of the plastic
  hinge and is verified *separately* (``T11b``) as a closed-form
  self-consistency check -- the FE lateral-only cross-check is the cleanest
  possible "last jump" and does not entangle the two load branches.
* Small-strain, moderate-rotation (von Karman) plate; no stiffener web
  tripping, no residual stress, no strain-rate effects -- the same scope as the
  closed-form module.
* The FE collapse is a *plastic-hinge* mechanism (the plate is stocky,
  ``beta = 0.73``), not a buckling path; the arc-length driver is used to
  resolve the limit point, not a bifurcation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    case_study_geometry,
    grillage_utilisations,
    plate_patch_response,
)
from .ice_belt_collapse import (
    PlateModel,
    elastic_perfectly_plastic,
    patch_pressure_load,
    simply_supported_edges,
    solve_arc_length,
    arc_length_limit,
    arc_length_status,
    structured_mesh,
    uniform_pressure_load,
)

__all__ = [
    "build_fe_panel",
    "fe_lateral_collapse",
    "yield_line_uniform_pressure",
    "crosscheck_plate_local_bending",
    "run_fe_crosscheck",
    "main",
]


def build_fe_panel(
    geom: GrillageGeometry,
    *,
    nx: int = 14,
    ny: int = 4,
    n_layers: int = 5,
) -> PlateModel:
    """Build the bare SSSS plate of the representative loaded panel.

    The representative panel is ``frame_spacing`` (x, along the longitudinal
    stiffeners) by ``bay_spacing`` (y, across) with the case-study plate
    thickness, on EH36 elastic-perfectly-plastic steel.  ``nx``/``ny`` resolve
    the plastic-hinge field; the panel is 6:1 so ``nx`` is the long direction.
    """
    geom.validate()
    if nx < 4 or ny < 2:
        raise ValueError("nx >= 4 and ny >= 2 required to resolve the hinge field")
    grade = elastic_perfectly_plastic(sigma_0=geom.yield_stress)
    nodes, elems = structured_mesh(geom.frame_spacing, geom.bay_spacing, nx, ny)
    model = PlateModel(nodes, elems, geom.plate_thickness, grade, n_layers=n_layers)
    return model


def _combined_lateral_reference(
    model: PlateModel,
    geom: GrillageGeometry,
    case: CombinedLoadCase,
) -> tuple[np.ndarray, float, float]:
    """Build the combined lateral reference load (ice patch + hydrostatic).

    Returns ``(fext, p_ice_ref, p_h_ref)`` where ``fext`` is the nodal load
    vector for the *sum* of the centred ice patch and the uniform hydrostatic
    head, both at their case values (so the arc-length load factor scales the
    combined lateral load together).
    """
    lx, ly = case.patch_extents(geom)
    have_patch = case.ice_pressure > 0.0 and lx > 0.0 and ly > 0.0
    fext = np.zeros(model.ndof)
    if have_patch:
        # patch area on the *representative* panel (one bay x one frame space)
        patch_area = min(lx, geom.frame_spacing) * min(ly, geom.bay_spacing)
        fext += patch_pressure_load(model, case.ice_pressure, patch_area)
    if case.hydrostatic_pressure > 0.0:
        fext += uniform_pressure_load(model, case.hydrostatic_pressure)
    return fext, case.ice_pressure, case.hydrostatic_pressure


def fe_lateral_collapse(
    geom: GrillageGeometry,
    case: CombinedLoadCase,
    *,
    nx: int = 14,
    ny: int = 4,
    n_steps: int = 80,
    dlam0: float = 0.05,
) -> dict[str, object]:
    """Trace the FE plastic collapse of the panel under the combined lateral.

    Returns the peak load factor ``lam_peak`` (the multiplier on the combined
    lateral reference at which the plate collapses), the collapse pressure and
    the arc-length status.
    """
    geom.validate()
    case.validate(geom)
    model = build_fe_panel(geom, nx=nx, ny=ny)
    fixed = simply_supported_edges(model)
    fext, p_ice_ref, p_h_ref = _combined_lateral_reference(model, geom, case)
    path = solve_arc_length(model, fext, fixed, n_steps=n_steps, dlam0=dlam0)
    limit = arc_length_limit(path)
    lam_peak = float(limit["peak_load_factor"])
    combined_ref = p_ice_ref + p_h_ref
    return {
        "lam_peak": lam_peak,
        "collapse_combined_pressure": lam_peak * combined_ref,
        "ice_ref": p_ice_ref,
        "hydro_ref": p_h_ref,
        "combined_ref": combined_ref,
        "arc_status": arc_length_status(limit),
        "energy_residual": float(path.energy_residual()),
        "mesh": [nx, ny],
        "path_lam": list(path.lam[1:]),
        "path_w": list(path.centre_w),
    }


def yield_line_uniform_pressure(geom: GrillageGeometry) -> float:
    """Rigid-plastic (yield-line) collapse pressure for a uniform lateral load.

    The representative panel is 6:1 (``frame_spacing``/``bay_spacing``), so it
    spans essentially one-way across the short direction ``b = bay_spacing``;
    the yield-line collapse is ``p_u = 8 M_p / b^2`` with ``M_p = sigma_y t^2/4``
    (the classical one-way strip mechanism).  This is the physically correct
    *ultimate* for a plate under uniform lateral pressure and is the reference
    the FE collapse is checked against for the uniform branch.
    """
    geom.validate()
    mp = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    return 8.0 * mp / geom.bay_spacing ** 2


def _closed_form_eta0_utilisation(
    geom: GrillageGeometry,
    case: CombinedLoadCase,
    *,
    terms: int = 61,
) -> float:
    """Closed-form ``plate_local_bending`` utilisation at ``eta = 0``.

    The undegraded plastic-hinge capacity ``m_p = sigma_y t^2/4``; the applied
    bending moment is the Navier-series peak of the combined ice patch +
    hydrostatic on the single panel (the same ``m_plate`` the closed-form
    module computes, but without the ``sqrt(1 - eta^2)`` membrane factor).
    """
    geom.validate()
    case.validate(geom)
    lx, ly = case.patch_extents(geom)
    have_patch = case.ice_pressure > 0.0 and lx > 0.0 and ly > 0.0
    m_plate = 0.0
    if have_patch:
        pr = plate_patch_response(
            span_x=geom.frame_spacing, span_y=geom.bay_spacing,
            d1=geom.plate_rigidity, d2=geom.plate_rigidity,
            d3=geom.plate_rigidity, pressure=case.ice_pressure,
            patch=(0.5 - 0.5 * lx / geom.frame_spacing,
                   0.5 + 0.5 * lx / geom.frame_spacing,
                   0.5 - 0.5 * ly / geom.bay_spacing,
                   0.5 + 0.5 * ly / geom.bay_spacing),
            terms=terms, poisson=geom.poisson, sample=15)
        m_plate += max(pr["m_x_max"], pr["m_y_max"])
    if case.hydrostatic_pressure > 0.0:
        pr = plate_patch_response(
            span_x=geom.frame_spacing, span_y=geom.bay_spacing,
            d1=geom.plate_rigidity, d2=geom.plate_rigidity,
            d3=geom.plate_rigidity, pressure=case.hydrostatic_pressure,
            terms=terms, poisson=geom.poisson, sample=15)
        m_plate += max(pr["m_x_max"], pr["m_y_max"])
    cap = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    return m_plate / cap


def crosscheck_plate_local_bending(
    geom: GrillageGeometry,
    case: CombinedLoadCase,
    *,
    nx: int = 14,
    ny: int = 4,
    n_steps: int = 80,
    dlam0: float = 0.05,
    terms: int = 61,
) -> dict[str, object]:
    """Cross-check the FE collapse against the closed-form capacity.

    ``lam_peak`` (FE) is compared against ``1 / u_plate_eta0`` (closed-form
    undegraded capacity).  The closed-form is a *first-yield* lower bound, so
    the FE collapse must be **at or above** it (``lam_peak >= 1/u_plate_eta0``);
    the ratio is the quantified conservatism margin.  An over-prediction
    (``lam_peak < 1/u_plate_eta0``) would flag an unsafe design formula.
    """
    fe = fe_lateral_collapse(geom, case, nx=nx, ny=ny, n_steps=n_steps, dlam0=dlam0)
    u_eta0 = _closed_form_eta0_utilisation(geom, case, terms=terms)
    closed_multiplier = 1.0 / u_eta0 if u_eta0 > 0.0 else float("inf")
    rel_error = fe["lam_peak"] / closed_multiplier - 1.0
    # conservatism gate: the closed-form must not over-predict the collapse
    conservative = bool(fe["lam_peak"] >= closed_multiplier - 1e-9)
    return {
        "lam_peak_fe": fe["lam_peak"],
        "closed_multiplier": closed_multiplier,
        "u_plate_eta0": u_eta0,
        "conservatism_ratio": fe["lam_peak"] / closed_multiplier,
        "rel_error": rel_error,
        "abs_error_pct": abs(rel_error) * 100.0,
        "pass_conservative": conservative,
        "arc_status": fe["arc_status"],
        "energy_residual": fe["energy_residual"],
        "mesh": fe["mesh"],
        "ice_ref": fe["ice_ref"],
        "hydro_ref": fe["hydro_ref"],
        "combined_ref": fe["combined_ref"],
        "collapse_combined_pressure": fe["collapse_combined_pressure"],
    }


def _scenarios() -> list[tuple[str, float, float, float, float, float]]:
    """The three documented combined cases (same as ``combined_case_table``)."""
    return [
        ("light-ice + ballast head", 0.15, 1.0e6, 0.35, 0.35, 5.0e4),
        ("design-ice + design head", 0.30, 2.0e6, 0.70, 0.35, 1.0e5),
        ("severe-ice + tank head", 0.45, 3.0e6, 1.05, 0.35, 1.5e5),
    ]


def run_fe_crosscheck(
    geom: GrillageGeometry | None = None,
    *,
    nx: int = 14,
    ny: int = 4,
    n_steps: int = 80,
    dlam0: float = 0.05,
) -> dict[str, object]:
    """Run the FE-vs-closed-form cross-check over the three combined cases.

    Also runs the uniform-branch yield-line anchor (``T11a``): the FE collapse
    of the panel under a *uniform* lateral head is compared against the
    rigid-plastic ``p_u = 8 M_p / b^2``.
    """
    geom = (case_study_geometry() if geom is None else geom).validate()
    rows: list[dict[str, object]] = []
    for name, eta, p_ice, lx, ly, p_h in _scenarios():
        case = CombinedLoadCase(
            axial_stress=eta * geom.yield_stress, ice_pressure=p_ice,
            patch_length=lx, patch_width=ly, hydrostatic_pressure=p_h)
        cc = crosscheck_plate_local_bending(
            geom, case, nx=nx, ny=ny, n_steps=n_steps, dlam0=dlam0)
        cc["scenario"] = name
        cc["sigma_x_over_yield"] = eta
        rows.append(cc)

    # ---- T11a: uniform-branch yield-line anchor ---------------------------
    # A uniform 1 MPa head (no patch) on the same panel: the FE collapse
    # pressure is compared against the rigid-plastic p_u = 8 M_p / b^2.
    uniform_case = CombinedLoadCase(
        axial_stress=0.0, ice_pressure=0.0, patch_length=0.0, patch_width=0.0,
        hydrostatic_pressure=1.0e6)
    fe_u = fe_lateral_collapse(geom, uniform_case, nx=nx, ny=ny,
                               n_steps=n_steps, dlam0=max(dlam0, 0.05))
    p_u = yield_line_uniform_pressure(geom)
    fe_u_pressure = fe_u["lam_peak"] * fe_u["combined_ref"]
    yield_line_err = fe_u_pressure / p_u - 1.0
    rows.append({
        "scenario": "uniform-head yield-line anchor (T11a)",
        "sigma_x_over_yield": 0.0,
        "lam_peak_fe": fe_u["lam_peak"],
        "closed_multiplier": 1.0,
        "u_plate_eta0": 1.0,
        "conservatism_ratio": fe_u["lam_peak"],
        "rel_error": yield_line_err,
        "abs_error_pct": abs(yield_line_err) * 100.0,
        "pass_conservative": bool(fe_u_pressure >= p_u - 1e-9),
        "arc_status": fe_u["arc_status"],
        "energy_residual": fe_u["energy_residual"],
        "mesh": fe_u["mesh"],
        "ice_ref": 0.0,
        "hydro_ref": 1.0e6,
        "combined_ref": 1.0e6,
        "collapse_combined_pressure": fe_u_pressure,
        "yield_line_pressure": p_u,
    })

    passed = all(r["pass_conservative"] for r in rows)
    return {
        "suite": "grillage_ice_fe_crosscheck",
        "geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "yield_stress_MPa": geom.yield_stress / 1e6,
        },
        "cases": rows,
        "passed": passed,
        "n_cases": len(rows),
        "gate": ("FE collapse >= closed-form first-yield capacity "
                 "(conservative lower bound); uniform branch vs p_u = 8 M_p / b^2"),
    }


def main(out_dir: str | Path = "outputs/grillage_ice_fe_crosscheck") -> dict[str, object]:
    """Run the cross-check and write the machine-readable artifact."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = run_fe_crosscheck()
    (out / "fe_crosscheck.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":  # pragma: no cover
    payload = main()
    print(json.dumps({"passed": payload["passed"], "n_cases": payload["n_cases"]}))
