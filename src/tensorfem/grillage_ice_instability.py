"""Stiffener tripping integration for the ice-grillage ultimate strength module.

Why this module exists
----------------------
``grillage_ice_ultimate`` provides three limit states (plate local bending,
longitudinal stiffener beam-column, transverse frame rigid-plastic collapse)
but explicitly does *not* model stiffener web local buckling or tripping --
those are supplied by :mod:`tensorfem.stiffener_tripping`, which was kept
separate to preserve the 40 published verification gates.  The independent
audit of anchor ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE`` listed "stiffener
tripping" as ``not_run``.  This module closes that gap by integrating the
tripping and web local buckling checks as a **fourth limit state** alongside
the existing three, without modifying any of the verified functions.

What is added
-------------
1. ``stiffener_instability_utilisation`` -- computes the utilisation of the
   longitudinal stiffener against tripping and web local buckling, using the
   total extreme-fibre stress (axial + amplified bending) from the grillage
   load model.  The limiting stress is the minimum of the tripping stress
   (lower bound: ``k_theta = 0``) and the web local buckling stress (lower
   bound: both edges simply supported, ``k = 4``).
2. ``grillage_utilisations_with_instability`` -- calls the existing
   ``grillage_utilisations`` and augments the result with the instability
   utilisation.  The existing three keys are untouched.
3. ``ultimate_ice_pressure_with_instability`` -- bisection on the ice pressure
   using all four limit states.  The existing ``ultimate_ice_pressure`` is not
   modified.
4. ``tripping_governed_geometry`` -- a grillage with a deep, thin-webbed
   stiffener (T350x7+70x10) at wider spacing (0.50 m) and longer frame span
   (3.0 m) where the tripping check governs over the beam-column check.

Verification gates
------------------
* ``T14``: for the case-study geometry (stocky stiffener), the instability
  utilisation is strictly less than the beam-column utilisation (tripping does
  not govern for stocky sections).
* ``T15``: for the tripping-governed geometry, the instability utilisation
  exceeds the beam-column utilisation at moderate axial stress.
* ``T16``: the instability utilisation is monotone increasing in the ice
  pressure (same monotonicity property as the other three checks).
* ``T17``: the tripping-governed geometry's ultimate ice pressure with
  instability is strictly lower than without (tripping reduces capacity).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from .grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    grillage_utilisations,
    ultimate_ice_pressure,
    _governing,
)
from .stiffener_tripping import (
    TeeTorsionSection,
    stiffener_instability_check,
)

__all__ = [
    "stiffener_instability_utilisation",
    "grillage_utilisations_with_instability",
    "ultimate_ice_pressure_with_instability",
    "tripping_governed_geometry",
]


def stiffener_instability_utilisation(
    geom: GrillageGeometry,
    case: CombinedLoadCase,
    ice_pressure: float | None = None,
    terms: int = 61,
) -> dict[str, object]:
    """Stiffener instability (tripping + web buckling) utilisation.

    The total extreme-fibre stress in the stiffener (axial + amplified bending)
    is compared against the minimum of the tripping stress (lower bound,
    ``k_theta = 0``) and the web local buckling stress (lower bound, both edges
    simply supported, ``k = 4``).  The utilisation is the ratio of the total
    stress to this limiting stress.

    Returns a dict with keys:
    - ``stiffener_instability``: the utilisation (0 = no stress, 1 = at limit)
    - ``limiting_stress_Pa``: the governing instability stress
    - ``governing_instability``: "tripping" or "web_local_buckling"
    - ``tripping_sigma_free_Pa``: tripping stress with k_theta = 0
    - ``web_buckling_sigma_Pa``: web local buckling stress (k = 4)
    - ``total_stress_Pa``: axial + amplified bending stress in the stiffener
    """
    # Get the existing utilisations to extract the total stiffener stress
    u_base = grillage_utilisations(geom, case, ice_pressure=ice_pressure, terms=terms)

    # Total stress in the stiffener = axial + amplified bending
    sigma_axial = case.axial_stress
    sigma_bending = u_base.get("stiffener_bending_stress", 0.0)
    beta_m = u_base.get("moment_amplification", 1.0)
    sigma_total = sigma_axial + sigma_bending  # bending_stress already includes beta_m

    # Build the torsion section from the stiffener geometry
    stiff = geom.stiffener
    torsion_section = TeeTorsionSection(
        web_height=stiff.web_height,
        web_thickness=stiff.web_thickness,
        flange_width=stiff.flange_width,
        flange_thickness=stiff.flange_thickness,
        young=geom.young,
        poisson=geom.poisson,
    )

    # Run the instability check
    inst = stiffener_instability_check(
        torsion_section,
        span=geom.frame_spacing,
        stiffener_spacing=geom.bay_spacing,
        plate_thickness=geom.plate_thickness,
        yield_stress=geom.yield_stress,
        lateral_pressure=case.hydrostatic_pressure,
    )

    sigma_tripping_free = float(inst["tripping_sigma_free_Pa"])
    sigma_web_buckling = float(inst["web_local_buckling"]["min_Pa"])
    sigma_limiting = float(inst["limiting_stress_Pa"])
    governing = str(inst["governing"])

    u_instability = sigma_total / sigma_limiting if sigma_limiting > 0.0 else float("inf")

    return {
        "stiffener_instability": u_instability,
        "limiting_stress_Pa": sigma_limiting,
        "governing_instability": governing,
        "tripping_sigma_free_Pa": sigma_tripping_free,
        "web_buckling_sigma_Pa": sigma_web_buckling,
        "total_stress_Pa": sigma_total,
        "axial_stress_Pa": sigma_axial,
        "bending_stress_Pa": sigma_bending,
    }


def grillage_utilisations_with_instability(
    geom: GrillageGeometry,
    case: CombinedLoadCase,
    ice_pressure: float | None = None,
    terms: int = 61,
    member_model: str = "smeared",
    member_sub: int = 8,
    member_load_split: float = 0.5,
) -> dict[str, float]:
    """Member utilisations including stiffener instability (tripping + web buckling).

    Calls the existing ``grillage_utilisations`` and augments the result with
    the ``stiffener_instability`` utilisation.  The existing three keys
    (``plate_local_bending``, ``longitudinal_stiffener``, ``transverse_frame``)
    are untouched, preserving the published verification gates.
    """
    u_base = grillage_utilisations(
        geom, case, ice_pressure=ice_pressure, terms=terms,
        member_model=member_model, member_sub=member_sub,
        member_load_split=member_load_split,
    )
    u_inst = stiffener_instability_utilisation(
        geom, case, ice_pressure=ice_pressure, terms=terms,
    )
    return {
        **u_base,
        "stiffener_instability": u_inst["stiffener_instability"],
        "instability_limiting_stress_Pa": u_inst["limiting_stress_Pa"],
        "instability_governing": u_inst["governing_instability"],
        "instability_tripping_sigma_Pa": u_inst["tripping_sigma_free_Pa"],
        "instability_web_buckling_sigma_Pa": u_inst["web_buckling_sigma_Pa"],
        "instability_total_stress_Pa": u_inst["total_stress_Pa"],
    }


def _governing_with_instability(u: Mapping[str, float]) -> str:
    keys = ("plate_local_bending", "longitudinal_stiffener",
            "transverse_frame", "stiffener_instability")
    return max(keys, key=lambda k: (u.get(k, 0.0), k))


def ultimate_ice_pressure_with_instability(
    geom: GrillageGeometry,
    case: CombinedLoadCase,
    tol: float = 1e-6,
    max_iter: int = 200,
    terms: int = 61,
    member_model: str = "smeared",
    member_sub: int = 8,
    member_load_split: float = 0.5,
) -> dict[str, object]:
    """Largest ice patch pressure keeping every member utilisation <= 1,
    including stiffener instability (tripping + web buckling).

    Same bisection algorithm as ``ultimate_ice_pressure`` but the peak
    utilisation includes the ``stiffener_instability`` key.
    """
    geom.validate()
    case.validate(geom)
    if case.patch_length <= 0.0 or case.patch_width <= 0.0:
        raise ValueError("ultimate_ice_pressure_with_instability needs a positive patch size")

    _FOUR = ("plate_local_bending", "longitudinal_stiffener",
             "transverse_frame", "stiffener_instability")

    def peak(p: float) -> tuple[float, dict[str, float]]:
        u = grillage_utilisations_with_instability(
            geom, case, ice_pressure=p, terms=terms,
            member_model=member_model, member_sub=member_sub,
            member_load_split=member_load_split,
        )
        return max(u[k] for k in _FOUR), u

    u0, base = peak(0.0)
    if u0 >= 1.0:
        return {
            "ice_pressure_ultimate": 0.0,
            "governing_member": _governing_with_instability(base),
            "utilisation_at_limit": u0,
            "utilisation_without_ice": u0,
            "base_utilisations": base,
            "utilisations_at_limit": base,
            "margin_vs_design": 0.0 if case.design_pressure else None,
            "design_provenance": case.design_provenance,
            "converged": True,
        }

    lo = 0.0
    hi = max(1.0e5, 10.0 * (case.design_pressure or 1.0e6))
    u_hi, _ = peak(hi)
    guard = 0
    while u_hi < 1.0 and guard < 60:
        lo = hi
        hi *= 2.0
        u_hi, _ = peak(hi)
        guard += 1
    if u_hi < 1.0:
        return {
            "ice_pressure_ultimate": float("inf"),
            "governing_member": "none",
            "utilisation_at_limit": u_hi,
            "utilisation_without_ice": u0,
            "base_utilisations": base,
            "margin_vs_design": None,
            "design_provenance": case.design_provenance,
            "converged": False,
        }

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        u_mid, _ = peak(mid)
        if u_mid < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(hi, 1.0):
            break

    p_ult = 0.5 * (lo + hi)
    _, u_lim = peak(p_ult)
    u_peak = max(u_lim[k] for k in _FOUR)
    return {
        "ice_pressure_ultimate": p_ult,
        "governing_member": _governing_with_instability(u_lim),
        "utilisation_at_limit": u_peak,
        "utilisation_without_ice": u0,
        "base_utilisations": base,
        "utilisations_at_limit": u_lim,
        "margin_vs_design": (p_ult / case.design_pressure) if case.design_pressure else None,
        "design_provenance": case.design_provenance,
        "converged": True,
    }


def tripping_governed_geometry() -> GrillageGeometry:
    """Grillage with a deep, thin-webbed stiffener where tripping governs.

    T350x7+70x10 longitudinal stiffeners at b = 0.50 m spacing,
    T500x10+120x14 transverse frames at a = 2.5 m,
    6 bays spanning 3.0 m between girders, t = 18 mm plate,
    EH36 steel (sigma_y = 355 MPa).

    The h_w/t_w = 50 ratio and the long frame spacing (2.5 m) make the
    tripping stress lower than the beam-column limiting stress, so the
    stiffener_instability check governs over longitudinal_stiffener.
    """
    from .grillage_ice_ultimate import MemberSection
    return GrillageGeometry(
        frame_spacing=2.5, bay_spacing=0.50, n_bays=6, n_spans=4,
        plate_thickness=0.018,
        stiffener=MemberSection(0.350, 0.007, 0.070, 0.010),
        frame=MemberSection(0.500, 0.010, 0.120, 0.014),
        young=2.06e11, poisson=0.3, yield_stress=3.55e8,
    ).validate()
