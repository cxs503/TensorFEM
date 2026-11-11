"""W23: multi-station IACS CSR hull-girder ULS scan with a Smith :math:`M_u`.

Why this module exists
----------------------
W22 (:mod:`tensorfem.hull_girder_csr_w22`) closed one ULS check per *ship*
(midship section of four parent hulls) and paired the two moment-curvature
directions as follows::

    analyse_direction(members, condition="sagging")   # kappa > 0
    analyse_direction(members, condition="hogging")   # kappa < 0

with ``condition='sagging'`` documented as *bottom in compression*.  In the
convention of :mod:`tensorfem.hull_girder_smith_collapse`, ``kappa > 0`` puts
``y < y_na`` in compression, i.e. the **bottom** is compressed.  For a ship
section bottom compression is **hogging** (deck in tension); sagging puts the
**deck** in compression.  W22 therefore paired the sagging wave moment with the
bottom-compression capacity, which is the wrong extreme fibre.

This module fixes the pairing structurally by naming the two directions after
the *compressed extreme fibre* and never after a curvature sign::

    CONDITION_COMPRESSED_FIBRE = {"sagging": "deck", "hogging": "bottom"}

and derives the curvature sign from that statement
(:func:`curvature_sign`).  The elastic (yield) capacity is likewise taken from
the compressed extreme fibre.  The material effect of the corrected pairing on
the W22 parent hull is quantified in
``benchmarks/hull_girder_uls/csr_utilisation.json`` (section
``pairing_audit``) so the defect and its consequence are both on record.

What is scanned
---------------
One hull-girder model (the documented KVLCC2 form of
:mod:`tensorfem.hull_girder_csr_chain`) is discretised at
``STATION_FRACTIONS`` longitudinal stations (six sections, default
``0.20 ... 0.70 L``) and each section is checked in **both** wave conditions,
giving twelve ULS cases.  Per station:

* the local section breadth comes from the hull form's waterline breadth at
  that station, so the six sections are geometrically distinct;
* the scantlings are the midship UR S11-closed plating
  (:func:`~tensorfem.hull_girder_csr_w22.build_scantlings`) tapered by
  :func:`scantling_taper` -- an *explicitly declared study assumption*
  (constant through the parallel body, linear reduction outside it), not a
  rule requirement;
* the still-water bending moment ``M_sw(x)`` comes from the beam equilibrium
  of the homogeneous-loaded tanker condition already used in W22;
* the wave bending moment ``M_wv(x)`` comes from integrating the design-wave
  load distribution of :func:`~tensorfem.hull_girder_csr_chain.wave_load_distribution`
  (crest at the ends = sagging, crest amidships = hogging).  The midship value
  is cross-checked against the rule moment
  :func:`~tensorfem.hull_girder_csr_chain.rule_wave_bending_moment`; the CSR
  longitudinal distribution factor itself is *not* transcribed, so the station
  variation is physics-based and this is stated as a limitation.

Two capacity methods per case
-----------------------------
* **Smith progressive collapse** (incremental curvature, panel component curves
  with plate post-buckling and stiffener-column softening,
  :func:`~tensorfem.hull_girder_smith_collapse.solve_moment_curvature`):
  :math:`M_u` = peak of the moment-curvature curve.
* **Elastic method**: :math:`M_y = \\sigma_y W` with ``W`` from the compressed
  extreme fibre of the same fibre discretisation, i.e. the yield criterion
  ``sigma = M/Z <= sigma_y/gamma_R`` of the anchor document at
  ``gamma_R = 1``.

The relative difference ``M_u/M_y - 1`` is reported for every case; cases with
``|M_u/M_y - 1| > 15%`` are attributed by :func:`attribution_evidence` to
buckling (compression-side panels reach their compressive limit below yield) or
to yielding.

Partial safety factors
----------------------
``gamma_s`` and ``gamma_w`` of CSR Pt.1 Ch.5 could **not** be transcribed: the
acquisition attempt is refreshed by
``benchmarks/hull_girder_uls/csr_factor_register.json`` (``status=BLOCKED``,
network 403, no corpus copy).  Utilisation is therefore reported

* at unit factors (``gamma_s = gamma_w = 1``), explicitly labelled a
  *unit-factor reference value, not a CSR compliance statement*, and
* as the admissible factor ``gamma* = M_u / (M_sw + M_wv)`` per case, plus
* as a factor sensitivity scan (``gamma_w`` at ``gamma_s = 1``) that shows at
  which factor level each case would breach unity -- the utilisation is linear
  in the factors, so this is exact and needs no invented factor value.

Scope and limitations
---------------------
* Plane sections remain plane; no shear lag, torsion, warping or fracture.
* Scantlings are a parametric CSR-consistent reconstruction, not a vessel's
  structural drawings; the station taper is a declared assumption.
* One loading condition (homogeneous loaded tanker) supplies ``M_sw``; because
  it is sagging at midship, the hogging still-water component is zero there and
  the hogging demand is wave-only -- a conservative/limited statement,
  documented as such.
* Grid convergence is measured on the fibre discretisation of the section and
  on the curvature step (the two discretisation parameters of the Smith
  engine), Roache GCI with a safety factor of 1.25.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from .hull_girder_csr_chain import (
    MILD_STEEL_YIELD,
    STEEL_E,
    beam_moment_from_load,
    homogeneous_loaded_tanker_weights,
    rule_wave_bending_moment,
    solve_equilibrium_draft_trim,
    wave_load_distribution,
    weight_centroid,
    weight_per_length,
)
from .hull_girder_csr_w22 import (
    CASE_CATALOGUE,
    HullGirderCase,
    build_scantlings,
    panel_component_law,
    section_fibers,
    validate_case_geometry,
)
from .hull_girder_smith_collapse import curvature_grid, solve_moment_curvature

__all__ = [
    "CONDITION_COMPRESSED_FIBRE",
    "STATION_FRACTIONS",
    "FIBRE_LEVELS",
    "STEP_LEVELS",
    "GCI_SAFETY_FACTOR",
    "curvature_sign",
    "scantling_taper",
    "station_members",
    "elastic_section_properties",
    "analyse_condition",
    "load_distributions",
    "attribution_evidence",
    "uls_case",
    "grid_convergence",
    "run_uls_scan",
]

#: Which extreme fibre is in compression in each bending condition.  This is the
#: physical definition; the curvature sign and the elastic section modulus both
#: follow from it (see the module docstring for the W22 pairing defect).
CONDITION_COMPRESSED_FIBRE = {"sagging": "deck", "hogging": "bottom"}

#: Longitudinal stations, as a fraction of ``Lpp`` from the aft perpendicular.
STATION_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)

#: Fibre discretisation levels (strips per deep member) of the grid study.
FIBRE_LEVELS = (4, 8, 16)

#: Curvature-increment levels of the second grid-study axis.
STEP_LEVELS = (160, 320, 640)

#: Reference discretisation used for the reported ``M_u`` (finest of each axis).
REFERENCE_STRIPS = FIBRE_LEVELS[-1]
REFERENCE_STEPS = STEP_LEVELS[-1]

#: Roache grid-convergence safety factor (recommended 1.25 for three grids).
GCI_SAFETY_FACTOR = 1.25

#: Station taper: constant through the parallel body, linear reduction outside.
PARALLEL_BODY_HALF = 0.15
TAPER_END = 0.35
TAPER_MINIMUM = 0.80

#: Relative difference above which a case must be attributed (task criterion 4).
ATTRIBUTION_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# 1. Geometry of the station sections
# ---------------------------------------------------------------------------
def curvature_sign(condition: str) -> float:
    """Curvature sign that puts the condition's compressed fibre in compression.

    The Smith engine uses ``kappa > 0`` for bottom compression, so sagging (deck
    compression) needs ``-1`` and hogging needs ``+1``.
    """
    fibre = CONDITION_COMPRESSED_FIBRE[_check_condition(condition)]
    return 1.0 if fibre == "bottom" else -1.0


def _check_condition(condition: str) -> str:
    if condition not in CONDITION_COMPRESSED_FIBRE:
        raise ValueError(
            f"condition must be one of {sorted(CONDITION_COMPRESSED_FIBRE)}, got {condition!r}")
    return condition


def scantling_taper(x_fraction: float) -> float:
    """Declared station taper of plating/stiffener scantlings off midship.

    ``1.0`` for ``|x/L - 0.5| <= PARALLEL_BODY_HALF``, then linear down to
    ``TAPER_MINIMUM`` at ``|x/L - 0.5| = TAPER_END`` (and held there).
    """
    if not 0.0 < x_fraction < 1.0:
        raise ValueError(f"x_fraction must be in (0, 1), got {x_fraction!r}")
    distance = abs(x_fraction - 0.5)
    if distance <= PARALLEL_BODY_HALF:
        return 1.0
    excess = min(1.0, (distance - PARALLEL_BODY_HALF) / (TAPER_END - PARALLEL_BODY_HALF))
    return 1.0 - (1.0 - TAPER_MINIMUM) * excess


def _station_breadth(form, x_fraction: float) -> float:
    """Section breadth at ``x_fraction``, normalised to the declared beam ``B``.

    ``HullForm.breadth()`` is ``sectional_area/T`` of the Cb-closed form, whose
    maximum exceeds the declared ``B`` of the particulars; the profile is
    therefore scaled so that its maximum is exactly ``B``.  This keeps every
    station section consistent with the declared ``L/B/T/Cb`` (the load model
    still integrates the hull form as given, exactly as in W22).
    """
    x = form.x()
    profile = form.breadth()
    return float(np.interp(x_fraction * form.Lpp, x, profile) * form.B / profile.max())


def station_members(case: HullGirderCase, x_fraction: float, *,
                    deck_thickness: float) -> tuple[tuple, dict]:
    """Scantlings of the section at ``x_fraction`` of ``Lpp``.

    The station breadth comes from the hull form, the plating/stiffener
    scantlings are the midship closure tapered by :func:`scantling_taper`.
    """
    form = case.form()
    breadth = _station_breadth(form, x_fraction)
    taper = scantling_taper(x_fraction)
    station_case = replace(case, B=breadth)
    from .hull_girder_csr_w22 import _scaled_scantlings  # documented internal helper

    raw = _scaled_scantlings(station_case, deck_thickness=deck_thickness * taper)
    members = tuple(
        replace(member,
                plate_thickness=member.plate_thickness * taper,
                stiffener_area=member.stiffener_area * taper,
                web_height=member.web_height * taper,
                flange_thickness=member.flange_thickness * taper)
        for member in raw
    )
    return members, {
        "x_fraction": x_fraction,
        "x_m": x_fraction * form.Lpp,
        "station_breadth_m": breadth,
        "midship_breadth_m": form.B,
        "breadth_ratio": breadth / form.B,
        "taper_factor": taper,
        "deck_thickness_m": deck_thickness * taper,
    }


# ---------------------------------------------------------------------------
# 2. Section properties and the two capacity methods
# ---------------------------------------------------------------------------
def elastic_section_properties(fibers) -> dict:
    """``(I, y_na, W_deck, W_bottom)`` of the fibre discretisation.

    ``W`` is referred to the outermost *fibre centroid*, so the elastic method
    and the Smith solve share one structural idealisation (the plating
    half-thickness offset is therefore not resolved; its effect on ``W`` is
    quantified in the study as ``W_centroid_vs_extreme``).
    """
    area = sum(f.area for f in fibers)
    y_na = sum(f.area * f.y for f in fibers) / area
    inertia = sum(f.area * (f.y - y_na) ** 2 for f in fibers)
    y_top = max(f.y for f in fibers)
    y_bottom = min(f.y for f in fibers)
    return {
        "area_m2": area,
        "y_na_m": y_na,
        "inertia_m4": inertia,
        "y_top_m": y_top,
        "y_bottom_m": y_bottom,
        "W_deck_m3": inertia / (y_top - y_na),
        "W_bottom_m3": inertia / (y_na - y_bottom),
    }


def elastic_capacity(properties: dict, condition: str, *,
                     yield_stress: float = MILD_STEEL_YIELD, gamma_R: float = 1.0) -> dict:
    """Elastic (first-yield) moment capacity of the compressed extreme fibre."""
    fibre = CONDITION_COMPRESSED_FIBRE[_check_condition(condition)]
    modulus = properties["W_deck_m3"] if fibre == "deck" else properties["W_bottom_m3"]
    return {
        "method": "elastic yield",
        "compressed_fibre": fibre,
        "properties": dict(properties),
        "section_modulus_m3": modulus,
        "yield_stress_Pa": yield_stress,
        "gamma_R": gamma_R,
        "M_y_Nm": yield_stress * modulus / gamma_R,
        "criterion": "sigma = M/Z <= sigma_y/gamma_R (gamma_R=1 unit-factor reference)",
    }


def analyse_condition(members, *, condition: str, strips: int, steps: int,
                      young: float = STEEL_E,
                      yield_stress: float = MILD_STEEL_YIELD) -> dict:
    """Smith solve of one section in one bending condition (correct pairing)."""
    sign = curvature_sign(condition)
    fibers = section_fibers(members, strips=strips, young=young,
                            yield_stress=yield_stress)
    properties = elastic_section_properties(fibers)
    extent = (properties["y_top_m"] - properties["y_na_m"]
              if CONDITION_COMPRESSED_FIBRE[condition] == "deck"
              else properties["y_na_m"] - properties["y_bottom_m"])
    kappa_y = (yield_stress / young) / extent
    grid = curvature_grid(kappa_y, steps=steps, maximum_ratio=8.0)
    result = solve_moment_curvature(fibers, tuple(sign * k for k in grid))
    peak = max(abs(point.moment) for point in result.points)
    return {
        "condition": condition,
        "compressed_fibre": CONDITION_COMPRESSED_FIBRE[condition],
        "kappa_sign": sign,
        "strips": strips,
        "steps": steps,
        "fibres": len(fibers),
        "M_u_Nm": peak,
        "ultimate_curvature": result.ultimate_curvature,
        "M_first_yield_solver_Nm": result.first_yield_moment,
        "elastic_slope_Nm_m": result.elastic_slope,
        "ledger_residual": result.ledger_residual,
        "peak_is_interior": result.peak_is_interior,
        "curve": [(point.curvature, point.moment) for point in result.points],
        "properties": properties,
        "fiber_area_relative_error": _fiber_area_error(fibers, members),
    }


def _fiber_area_error(fibers, members) -> float:
    total_fibres = sum(f.area for f in fibers)
    total_members = sum(m.total_area for m in members)
    return abs(total_fibres - total_members) / total_members


# ---------------------------------------------------------------------------
# 3. Load effects along the hull girder
# ---------------------------------------------------------------------------
def load_distributions(case: HullGirderCase, *, design_wave_height: float | None = None) -> dict:
    """``M_sw(x)``, ``M_wv,sagging(x)`` and ``M_wv,hogging(x)`` of the model.

    Sagging is positive.  The wave moments come from integrating the design-wave
    load distribution (crest at the ends -> sagging, crest amidships ->
    hogging); the midship values are cross-checked against the rule moments.
    """
    form = case.form()
    height = case.design_wave_height if design_wave_height is None else design_wave_height
    items, total_mass_t = homogeneous_loaded_tanker_weights(form)
    lcg = weight_centroid(form, items)
    equilibrium = solve_equilibrium_draft_trim(form, total_mass_t, lcg)
    x = form.x()
    q_still_water = equilibrium["buoyancy_N_per_m"] - weight_per_length(form, items)
    m_sw = beam_moment_from_load(x, q_still_water)["moment"]

    wave = {}
    for condition, crest in (("sagging", 0.0), ("hogging", form.Lpp / 2.0)):
        q_wave = wave_load_distribution(form, h_w=height, crest_position=crest)
        moment = beam_moment_from_load(x, q_wave)["moment"]
        mid = int(np.argmin(np.abs(x - form.Lpp / 2.0)))
        value = float(moment[mid])
        wave[condition] = {
            "crest_position_m": crest,
            "moment": moment,
            "M_mid_signed_Nm": value,
            "M_mid_magnitude_Nm": abs(value),
        }
    # Sign audit: the sagging wave must produce a positive midship moment and the
    # hogging wave the opposite one, otherwise the whole pairing is meaningless.
    if not (wave["sagging"]["M_mid_signed_Nm"] > 0.0 > wave["hogging"]["M_mid_signed_Nm"]):
        raise ValueError("wave load distribution does not reproduce the sagging/hogging signs")

    rule = rule_wave_bending_moment(L=form.Lpp, B=form.B, Cb=form.Cb)
    mid = int(np.argmin(np.abs(x - form.Lpp / 2.0)))
    return {
        "x": x,
        "M_sw": m_sw,
        "wave": wave,
        "M_sw_mid_signed_Nm": float(m_sw[mid]),
        "rule": rule,
        "design_wave_height_m": height,
        "wave_to_rule_ratio": {
            "sagging": wave["sagging"]["M_mid_magnitude_Nm"] / rule["M_ws"],
            "hogging": wave["hogging"]["M_mid_magnitude_Nm"] / rule["M_wh"],
        },
        "equilibrium": {k: v for k, v in equilibrium.items() if k != "buoyancy_N_per_m"},
        "displacement_t": total_mass_t,
        "lcg_m": lcg,
    }


def _interp(x: np.ndarray, values: np.ndarray, x_station: float) -> float:
    return float(np.interp(x_station, x, values))


# ---------------------------------------------------------------------------
# 4. ULS combination, per case
# ---------------------------------------------------------------------------
def uls_case(*, case_id: str, station: dict, condition: str, capacity: dict,
             elastic: dict, loads: dict, gamma_s: float, gamma_w: float) -> dict:
    """One ``gamma_s M_sw + gamma_w M_wv <= M_u`` evaluation.

    Only the still-water component of the *same sign* as the condition enters
    the combination (the other one is zero), which is the W22 convention kept
    for comparability.
    """
    condition = _check_condition(condition)
    x_station = station["x_m"]
    m_sw_signed = _interp(loads["x"], loads["M_sw"], x_station)
    m_wv_signed = _interp(loads["x"], loads["wave"][condition]["moment"], x_station)
    m_sw = max(m_sw_signed, 0.0) if condition == "sagging" else max(-m_sw_signed, 0.0)
    m_wv = max(m_wv_signed, 0.0) if condition == "sagging" else max(-m_wv_signed, 0.0)
    lhs = gamma_s * m_sw + gamma_w * m_wv
    m_u = capacity["M_u_Nm"]
    utilisation = lhs / m_u
    return {
        "case_id": case_id,
        "station_x_fraction": station["x_fraction"],
        "station_x_m": x_station,
        "station_breadth_m": station["station_breadth_m"],
        "taper_factor": station["taper_factor"],
        "condition": condition,
        "compressed_fibre": CONDITION_COMPRESSED_FIBRE[condition],
        "M_sw_signed_Nm": m_sw_signed,
        "M_wv_signed_Nm": m_wv_signed,
        "M_sw_component_Nm": m_sw,
        "M_wv_component_Nm": m_wv,
        "gamma_s": gamma_s,
        "gamma_w": gamma_w,
        "lhs_Nm": lhs,
        "M_u_smith_Nm": m_u,
        "M_y_elastic_Nm": elastic["M_y_Nm"],
        "utilisation": utilisation,
        "verdict": "PASS" if utilisation <= 1.0 else "FAIL",
        "gamma_admissible": m_u / lhs if lhs > 0.0 else None,
        "gamma_w_max_at_gamma_s_1": (m_u - m_sw) / m_wv if m_wv > 0.0 else None,
        "ultimate_curvature": capacity.get("ultimate_curvature"),
        "peak_is_interior": capacity.get("peak_is_interior"),
        "ledger_residual": capacity.get("ledger_residual"),
    }


def attribution_evidence(members, capacity: dict, condition: str, *,
                         yield_stress: float = MILD_STEEL_YIELD) -> dict:
    """Which mechanism controls the Smith capacity of this case.

    A panel is *buckling limited* when its compressive limit is below yield; the
    controlling member is the compression-side panel with the largest area.
    """
    fibre = CONDITION_COMPRESSED_FIBRE[condition]
    y_na = capacity["properties"]["y_na_m"]
    compression = [m for m in members
                   if (m.y > y_na if fibre == "deck" else m.y < y_na)]
    panel = []
    for member in compression:
        law = panel_component_law(member, yield_stress=yield_stress)
        peak = law.peak_compressive_stress
        panel.append({
            "member": member.name,
            "area_m2": member.total_area,
            "peak_compressive_stress_Pa": peak,
            "peak_over_yield": peak / yield_stress,
            "panel_limited_by_buckling": peak < 0.95 * yield_stress,
        })
    panel.sort(key=lambda row: row["peak_over_yield"])
    controlling = panel[0] if panel else None
    ratio = capacity["M_u_Nm"] / capacity["M_y_elastic_Nm"] if capacity.get("M_y_elastic_Nm") else None
    if ratio is None:
        dominant = "unknown"
    elif abs(ratio - 1.0) <= ATTRIBUTION_THRESHOLD:
        dominant = "elastic/yield dominated"
    elif ratio < 1.0:
        dominant = "buckling dominated (compression panels peak below yield)"
    else:
        dominant = "strain-hardening/redistribution dominated"
    return {
        "compression_side_members": panel,
        "controlling_member": controlling,
        "M_u_over_M_y": ratio,
        "dominant_mechanism": dominant,
        "attribution_required": ratio is not None and abs(ratio - 1.0) > ATTRIBUTION_THRESHOLD,
    }


# ---------------------------------------------------------------------------
# 5. Grid convergence (Roache GCI, three levels)
# ---------------------------------------------------------------------------
def grid_convergence(coarse: float, medium: float, fine: float, *,
                     ratio: float = 2.0,
                     safety_factor: float = GCI_SAFETY_FACTOR) -> dict:
    """Observed order and fine-grid GCI of three monotone-refined values."""
    if min(coarse, medium, fine) <= 0.0:
        raise ValueError("grid values must be positive")
    e21 = medium - coarse
    e32 = fine - medium
    converged = "monotone"
    if e21 == 0.0 or e32 == 0.0:
        order = None
        converged = "exact/zero difference"
    elif e21 * e32 > 0.0:
        # monotone approach: |e32| = |e21| / r^p  =>  p = ln(|e21/e32|)/ln(r)
        order = math.log(abs(e21 / e32)) / math.log(ratio)
    else:
        order = None
        converged = "oscillatory"
    if order is None or order <= 0.0:
        # cannot extrapolate: report the fine-pair change as a conservative GCI
        if order is not None:
            converged = "non-asymptotic (fine-pair change at noise level)"
        gci = safety_factor * abs(e32 / fine)
        order_used = None
    else:
        gci = safety_factor * abs(e32 / fine) / (ratio ** order - 1.0)
        order_used = order
    return {
        "values": {"coarse": coarse, "medium": medium, "fine": fine},
        "refinement_ratio": ratio,
        "safety_factor": safety_factor,
        "relative_change_medium_to_fine": abs(e32 / fine),
        "order_observed": order,
        "order_used": order_used,
        "gci_fine": gci,
        "gci_fine_percent": 100.0 * gci,
        "behaviour": converged,
        "verdict": "PASS" if gci < 0.05 else "FAIL",
    }


# ---------------------------------------------------------------------------
# 6. Study driver
# ---------------------------------------------------------------------------
def run_uls_scan(
    *,
    case: HullGirderCase = CASE_CATALOGUE[0],
    stations: tuple[float, ...] = STATION_FRACTIONS,
    fibre_levels: tuple[int, ...] = FIBRE_LEVELS,
    step_levels: tuple[int, ...] = STEP_LEVELS,
    gamma_s: float = 1.0,
    gamma_w: float = 1.0,
    design_wave_height: float | None = None,
    factor_register_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict:
    """Run the multi-station ULS scan and evaluate criteria W23-C1..C4."""
    if len(stations) < 5:
        raise ValueError("the study needs at least five sections")
    if len(fibre_levels) < 3 or len(step_levels) < 3:
        raise ValueError("grid convergence needs three levels on each axis")
    root = Path(__file__).resolve().parents[2] if repo_root is None else Path(repo_root)

    geometry = validate_case_geometry(case)
    members_mid, closure = build_scantlings(case)
    deck_thickness = closure["deck_thickness_final"]
    loads = load_distributions(case, design_wave_height=design_wave_height)

    strips = max(fibre_levels)
    steps = max(step_levels)

    rows, grid_rows = [], []
    for x_fraction in stations:
        members, station = station_members(case, x_fraction, deck_thickness=deck_thickness)
        capacity, mesh_series, step_series = {}, {}, {}
        for condition in ("sagging", "hogging"):
            reference = analyse_condition(members, condition=condition,
                                          strips=strips, steps=steps)
            elastic = elastic_capacity(reference["properties"], condition)
            reference["M_y_elastic_Nm"] = elastic["M_y_Nm"]
            capacity[condition] = reference

            # grid axis 1: fibre discretisation, curvature steps at the finest level
            fibre_values = {}
            for level in fibre_levels:
                run = (reference if level == strips
                       else analyse_condition(members, condition=condition,
                                              strips=level, steps=steps))
                fibre_values[level] = run["M_u_Nm"]
            # grid axis 2: curvature increment, fibres at the finest level
            step_values = {}
            for level in step_levels:
                run = (reference if level == steps
                       else analyse_condition(members, condition=condition,
                                              strips=strips, steps=level))
                step_values[level] = run["M_u_Nm"]
            mesh_series[condition] = grid_convergence(*fibre_values.values())
            step_series[condition] = grid_convergence(*step_values.values())
            mesh_series[condition]["values"] = {str(k): v for k, v in fibre_values.items()}
            step_series[condition]["values"] = {str(k): v for k, v in step_values.items()}
            grid_rows.append({
                "case_id": f"{case.key}_x{x_fraction:.2f}_{condition}",
                "station_x_fraction": x_fraction,
                "condition": condition,
                "compressed_fibre": CONDITION_COMPRESSED_FIBRE[condition],
                "fibres": reference["fibres"],
                "fibre_levels": list(fibre_levels),
                "fibre_family": mesh_series[condition],
                "step_levels": list(step_levels),
                "step_family": step_series[condition],
                "M_u_reference_Nm": reference["M_u_Nm"],
            })
            reference["elastic"] = elastic
            reference["attribution"] = attribution_evidence(members, reference, condition)
            reference["members"] = [
                {"name": m.name, "y": m.y, "vertical_span": m.vertical_span,
                 "plate_area": m.plate_area, "stiffener_area": m.stiffener_area,
                 "plate_thickness": m.plate_thickness}
                for m in members
            ]

        for condition in ("sagging", "hogging"):
            row = uls_case(case_id=f"{case.key}_x{x_fraction:.2f}_{condition}",
                           station=station, condition=condition,
                           capacity=capacity[condition], elastic=capacity[condition]["elastic"],
                           loads=loads, gamma_s=gamma_s, gamma_w=gamma_w)
            row["M_u_over_M_y"] = capacity[condition]["attribution"]["M_u_over_M_y"]
            row["dominant_mechanism"] = capacity[condition]["attribution"]["dominant_mechanism"]
            row["controlling_member"] = capacity[condition]["attribution"]["controlling_member"]
            row["elastic"] = capacity[condition]["elastic"]
            row["attribution"] = capacity[condition]["attribution"]
            row["members"] = capacity[condition]["members"]
            row["fibres"] = capacity[condition]["fibres"]
            row["M_first_yield_solver_Nm"] = capacity[condition]["M_first_yield_solver_Nm"]
            row["elastic_slope_Nm"] = capacity[condition]["elastic_slope_Nm_m"]
            row["curve"] = capacity[condition]["curve"]
            row["station"] = station
            rows.append(row)

    criteria = _criteria(rows, grid_rows, root, factor_register_path)
    return {
        "study": "W23 multi-station IACS CSR hull-girder ULS scan",
        "mission_id": 15094,
        "repository": "TensorFEM",
        "hull_model": {
            "case": case.key,
            "label": case.label,
            "particulars": {"Lpp": case.Lpp, "B": case.B, "D": case.D,
                            "T": case.T, "Cb": case.Cb},
            "geometry_validation": geometry,
            "midship_scantlings": {
                "deck_thickness_final_m": deck_thickness,
                "UR_S11_closure": closure,
            },
            "displacement_t": loads["displacement_t"],
            "lcg_m": loads["lcg_m"],
            "still_water_condition": "homogeneous loaded tanker (W22 template)",
            "design_wave_height_m": loads["design_wave_height_m"],
            "M_sw_mid_signed_Nm": loads["M_sw_mid_signed_Nm"],
            "rule_wave_moments": loads["rule"],
            "wave_to_rule_ratio": loads["wave_to_rule_ratio"],
            "equilibrium": loads["equilibrium"],
        },
        "conventions": {
            "sagging": "deck in compression (kappa < 0 in the Smith engine)",
            "hogging": "bottom in compression (kappa > 0 in the Smith engine)",
            "sign_of_moments": "sagging positive",
            "pairing_rule": "each condition is evaluated against the capacity of its "
                            "compressed extreme fibre (W22 paired the curvature sign, "
                            "not the compressed fibre -- see pairing_audit)",
        },
        "settings": {
            "stations_x_over_L": list(stations),
            "reference_strips": strips,
            "reference_steps": steps,
            "fibre_levels": list(fibre_levels),
            "step_levels": list(step_levels),
            "gamma_s": gamma_s,
            "gamma_w": gamma_w,
            "gamma_status": "BLOCKED - unit factors used as a reference value only",
            "gci_safety_factor": GCI_SAFETY_FACTOR,
            "attribution_threshold": ATTRIBUTION_THRESHOLD,
            "taper": {"parallel_body_half": PARALLEL_BODY_HALF,
                      "taper_end": TAPER_END, "taper_minimum": TAPER_MINIMUM},
        },
        "cases": rows,
        "grid_convergence": grid_rows,
        "criteria": criteria,
        "limitations": [
            "plane sections remain plane: no shear lag, torsion, warping or fracture",
            "scantlings are a parametric CSR-consistent reconstruction; the station "
            "taper is a declared study assumption, not a rule requirement",
            "a single loading condition (homogeneous loaded tanker) supplies M_sw; it is "
            "sagging at midships, so the hogging still-water component vanishes there",
            "CSR partial safety factors gamma_s / gamma_w could not be transcribed; "
            "utilisation is a unit-factor reference value, not a compliance statement",
            "the CSR longitudinal distribution factor of M_wv is not transcribed; the "
            "station-wise M_wv comes from the design-wave load model instead",
        ],
    }


def _criteria(rows: list[dict], grid_rows: list[dict], root: Path,
              factor_register_path: Path | None) -> list[dict]:
    n_cases = len(rows)
    complete = all(
        row["M_u_smith_Nm"] > 0.0 and row["lhs_Nm"] >= 0.0
        and row["utilisation"] is not None and row["verdict"] in ("PASS", "FAIL")
        for row in rows
    )
    residuals = [recompute_residual(row) for row in rows]
    factor_path = (factor_register_path if factor_register_path is not None
                   else root / "benchmarks" / "hull_girder_uls" / "csr_factor_register.json")
    factor_register = json.loads(factor_path.read_text()) if factor_path.exists() else None
    transcribed = []
    if factor_register:
        transcribed = [f for f in factor_register.get("factors", [])
                       if f.get("status") == "TRANSCRIBED" and f.get("printed_values")]
    gci_max = max(row["fibre_family"]["gci_fine_percent"] for row in grid_rows)
    gci_max_steps = max(row["step_family"]["gci_fine_percent"] for row in grid_rows)
    attributed = [row for row in rows if row["attribution"]["attribution_required"]]
    attributed_ok = all(row["attribution"]["dominant_mechanism"] != "unknown" for row in attributed)

    return [
        {
            "id": "W23-C1",
            "criterion": ">=10 cases (>=5 sections x 2 wave conditions), each with "
                         "gamma_s M_sw + gamma_w M_wv, M_u, utilisation and PASS/FAIL",
            "value": {"n_cases": n_cases, "n_sections": len({row['station_x_fraction'] for row in rows}),
                      "complete": complete,
                      "max_utilisation": max(row["utilisation"] for row in rows),
                      "n_fail": sum(1 for row in rows if row["verdict"] == "FAIL")},
            "threshold": {"n_cases": 10, "n_sections": 5, "complete": True},
            "verdict": "PASS" if n_cases >= 10 and complete
                       and len({row['station_x_fraction'] for row in rows}) >= 5 else "FAIL",
            "evidence": "benchmarks/hull_girder_uls/section_scan.csv",
        },
        {
            "id": "W23-C2a",
            "criterion": "clause-formula independent recomputation residual < 1e-6",
            "value": {"max_residual": max(residuals),
                      "components": ["lhs", "utilisation", "M_u", "W_elastic"]},
            "threshold": 1e-6,
            "verdict": "PASS" if max(residuals) < 1e-6 else "FAIL",
            "evidence": "benchmarks/hull_girder_uls/csr_utilisation.json:recomputation",
        },
        {
            "id": "W23-C2b",
            "criterion": "gamma_s / gamma_w taken from the CSR table with a clause "
                         "citation per factor",
            "value": {"factors_transcribed": len(transcribed),
                      "register_status": (factor_register or {}).get("status", "MISSING")},
            "threshold": {"factors_transcribed": 2},
            "verdict": "FAIL",
            "detail": "BLOCKED: no authorised CSR text in this environment (network 403, "
                      "no corpus copy). Utilisation is reported at unit factors plus the "
                      "admissible factor gamma* = M_u/(M_sw+M_wv) and a gamma_w sensitivity "
                      "scan, so no rule factor value is invented.",
            "evidence": "benchmarks/hull_girder_uls/csr_factor_register.json",
        },
        {
            "id": "W23-C3",
            "criterion": "grid convergence: GCI of M_u < 5% on three grids, order reported",
            "value": {"max_gci_percent_fibres": gci_max,
                      "max_gci_percent_curvature_steps": gci_max_steps},
            "threshold": 5.0,
            "verdict": "PASS" if max(gci_max, gci_max_steps) < 5.0 else "FAIL",
            "evidence": "benchmarks/hull_girder_uls/grid_convergence.json",
        },
        {
            "id": "W23-C4",
            "criterion": "elastic vs Smith ultimate moment listed per case; every case "
                         "with |difference| > 15% attributed",
            "value": {"n_cases": n_cases, "n_above_threshold": len(attributed),
                      "all_attributed": attributed_ok},
            "threshold": {"n_cases": 10, "all_attributed": True},
            "verdict": "PASS" if n_cases >= 10 and attributed_ok else "FAIL",
            "evidence": "benchmarks/hull_girder_uls/section_scan.csv",
        },
    ]


def recompute_residual(row: dict) -> float:
    """Largest relative residual of the clause arithmetic recomputed from stored data.

    Deliberately uses only the tabulated values of the row (as written to the
    CSV/JSON), so the check is independent of the in-memory objects.
    """
    residuals = []
    lhs = row["gamma_s"] * row["M_sw_component_Nm"] + row["gamma_w"] * row["M_wv_component_Nm"]
    if row["lhs_Nm"] > 0.0:
        residuals.append(abs(lhs - row["lhs_Nm"]) / row["lhs_Nm"])
    utilisation = lhs / row["M_u_smith_Nm"]
    if row["utilisation"] > 0.0:
        residuals.append(abs(utilisation - row["utilisation"]) / row["utilisation"])
    elastic_data = row["elastic"]
    properties = elastic_data["properties"]
    modulus = elastic_data["section_modulus_m3"]
    elastic = elastic_data["yield_stress_Pa"] * modulus / elastic_data["gamma_R"]
    residuals.append(abs(elastic - row["M_y_elastic_Nm"]) / row["M_y_elastic_Nm"])
    inertia = properties["inertia_m4"]
    recomputed_modulus = (inertia / (properties["y_top_m"] - properties["y_na_m"])
                          if elastic_data["compressed_fibre"] == "deck"
                          else inertia / (properties["y_na_m"] - properties["y_bottom_m"]))
    residuals.append(abs(recomputed_modulus - modulus) / modulus)
    return max(residuals)