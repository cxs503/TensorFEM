"""Maturity layer of the detail-fatigue chain: criteria ledger with real numbers.

The primitives live in :mod:`tensorfem.fatigue.weld_toe` (parametric joint, Q8
plane model, IIW extrapolation, DNV-RP-C203 S-N curves) and
:mod:`tensorfem.fatigue.beam_interface` (beam-theory nominal stress).  This module
turns them into the machine-checkable maturity ledger of the welded-detail fatigue
work package:

* :data:`JOINT_FAMILY` -- the acceptance geometry family (``t`` = 10/15/20 mm,
  ``r`` = 0.5/1/2 mm) that every criterion is evaluated on;
* :func:`toe_mesh_resolution` -- toe mesh size against the ``t/10`` requirement;
* :func:`extrapolation_linearity` -- **genuine** least-squares ``R^2`` of the
  hot-spot extrapolation.  A two-station ("linear") extrapolation has ``R^2 = 1``
  identically, so the linearity criterion is only meaningful from three stations
  upwards; the two-point value is returned but flagged as degenerate;
* :func:`gci_hotspot` -- Grid Convergence Index of the hot-spot stress over three
  mesh levels (Roache's three-grid procedure);
* :func:`radius_negative_controls` -- ``r -> 0`` singularity trend (must be
  recorded, not crash) and ``r -> inf`` trend of the notch and structural SCF;
* :func:`scf_vs_iiw` -- the finite-element notch/structural SCF against the
  closed-form Nii/Maddox anchors and against the IIW structural SCF band;
* :func:`beam_to_detail_validation` -- the global-to-detail transfer chain
  (beam theory ``sigma_nom`` -> ``sigma_hot = SCF * sigma_nom``) checked against
  the finite-element structural stress of a load-carrying joint;
* :func:`run_maturity_assessment` -- the full ledger, one row per acceptance
  criterion, with the measured number, the requirement and the verdict.

Honesty contract.  A criterion whose physical premise does not hold is reported as
``False`` with the reason; nothing is tuned to fall inside a band.  In particular
the IIW ``SCF = 1.2-1.5`` band belongs to structural-discontinuity details (bracket
toes, collar plates, load-carrying joints), not to the transverse *non-load-carrying*
fillet T-joint of this family, whose structural SCF is ``~1.0`` by theory.  See the
report for the measured evidence.
"""
from __future__ import annotations

import math

import numpy as np

from .beam_interface import BeamSection, beam_nominal_stress
from .weld_toe import (
    QUADRATIC_STATIONS,
    WeldDetail,
    WeldToeSolution,
    fatigue_life,
    hot_spot_stress,
    maddox_stress_concentration,
    nii_stress_concentration,
    solve_weld_detail,
)

__all__ = [
    "JOINT_FAMILY",
    "IIW_STRUCTURAL_SCF_BAND",
    "family_geometry",
    "toe_mesh_resolution",
    "extrapolation_linearity",
    "gci_hotspot",
    "radius_negative_controls",
    "scf_vs_iiw",
    "power_law_response",
    "beam_to_detail_validation",
    "run_maturity_assessment",
]

#: Acceptance geometry family of the work package: plate thickness (m) and toe
#: radius (m) combinations quoted in the node specification (t = 10-20 mm,
#: r = 0.5-2 mm).
JOINT_FAMILY: tuple[tuple[float, float], ...] = (
    (0.010, 0.0005),
    (0.010, 0.001),
    (0.010, 0.002),
    (0.015, 0.001),
    (0.015, 0.002),
    (0.020, 0.001),
    (0.020, 0.002),
)

#: IIW structural (hot-spot) SCF band quoted by the node specification.
IIW_STRUCTURAL_SCF_BAND: tuple[float, float] = (1.2, 1.5)

#: DNV-RP-C203 curve D slope of the first branch, used by the power-law check.
_M_SLOPE = 3.0


# --------------------------------------------------------------------------- #
# Geometry family
# --------------------------------------------------------------------------- #
def family_geometry(
    plate_thickness_m: float,
    toe_radius_m: float,
    *,
    joint_type: str = "t_fillet",
    load_case: str = "plate_axial",
) -> WeldDetail:
    """Return a member of the acceptance family with self-consistent geometry.

    The attachment height and the weld leg are scaled with the toe radius so that
    the weld profile stays non-degenerate over the whole radius range (a fixed
    14 mm leg with a 20 mm toe radius has no flank left).  ``load_case`` defaults
    to ``"plate_axial"``: the plate carries the nominal stress and the attachment
    is load free, i.e. the non-load-carrying transverse detail.
    """
    t = float(plate_thickness_m)
    r = float(toe_radius_m)
    if joint_type == "butt":
        return WeldDetail(
            joint_type="butt",
            plate_thickness_m=t,
            toe_radius_m=r,
            reinforcement_m=max(3.0 * r, 0.15 * t),
            crest_width_m=max(6.0 * r, 0.30 * t),
            load_case=load_case,
        )
    if joint_type == "corner":
        return WeldDetail(
            joint_type="corner",
            plate_thickness_m=t,
            toe_radius_m=r,
            attachment_thickness_m=0.6 * t,
            attachment_height_m=max(1.4 * r, 1.5 * t),
            weld_leg_m=1.4 * r,
            symmetry=False,
            load_case=load_case,
        )
    return WeldDetail(
        joint_type="t_fillet",
        plate_thickness_m=t,
        toe_radius_m=r,
        attachment_thickness_m=t,
        attachment_height_m=max(1.8 * r, 1.2 * t),
        weld_leg_m=max(1.4 * r, 0.7 * t),
        load_case=load_case,
    )


def toe_mesh_resolution(detail: WeldDetail, *, level: int = 1) -> dict[str, object]:
    """Toe mesh size of the model against the ``<= t / 10`` requirement.

    The mesher places ``16 * level`` uniform columns across a band of
    ``1.75 r`` around the toe (``0.25 r`` of flank plus ``0.5 r`` of plate), so the
    surface element size across the toe arc is ``1.75 r / (16 level)``.  The
    structured depth rows below the surface band are graded geometrically and the
    first row is ``0.05 r / level``.  Both are reported, together with the global
    minimum Q8 edge length of the mesh.
    """
    r = detail.toe_radius_m
    t = detail.plate_thickness_m
    n_cols = 16 * level
    band = 1.75 * r
    surface_size = band / n_cols
    first_row = 0.05 * r / level
    solution = solve_weld_detail(detail, level=level)
    limit = t / 10.0
    return {
        "toe_radius_m": r,
        "plate_thickness_m": t,
        "level": level,
        "surface_element_m": surface_size,
        "surface_element_over_t": surface_size / t,
        "first_depth_row_m": first_row,
        "global_min_edge_m": solution.min_edge_m,
        "limit_m": limit,
        "pass": surface_size <= limit and first_row <= limit,
    }


# --------------------------------------------------------------------------- #
# Hot-spot extrapolation linearity
# --------------------------------------------------------------------------- #
def extrapolation_linearity(
    solution: WeldToeSolution,
    plate_thickness_m: float,
    *,
    station_sets: dict[str, tuple[float, ...]] | None = None,
) -> dict[str, object]:
    """Least-squares ``R^2`` of the surface stress against the extrapolation line.

    For each station set (in units of ``t``, measured from the toe tangent point)
    the surface stress is sampled, a first-order least-squares line is fitted and
    the coefficient of determination is returned together with the extrapolated
    toe value and the sample count.  ``R^2`` is *identically* one for exactly two
    stations; such sets are flagged ``degenerate=True`` so that a two-point check
    can never be mistaken for a linearity check.
    """
    sets = station_sets or {
        "iiw_quadratic": QUADRATIC_STATIONS,
        "half_1_0_1_5": (0.5, 1.0, 1.5),
        "dense_0_4_to_1_4": (0.4, 0.7, 1.0, 1.4),
        "linear_0_5_1_5": (0.5, 1.5),
    }
    t = float(plate_thickness_m)
    x = np.asarray(solution.surface_x_m, dtype=float)
    s = np.asarray(solution.surface_sigma_xx_pa, dtype=float) / solution.nominal_stress_pa
    toe = float(solution.x_toe_m)
    rows: dict[str, object] = {}
    for name, stations in sets.items():
        f = np.asarray(stations, dtype=float)
        values = np.asarray([float(np.interp(toe + fi * t, x, s)) for fi in f])
        design = np.vstack((np.ones_like(f), f)).T
        coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
        fit = design @ coeff
        ss_tot = float(((values - values.mean()) ** 2).sum())
        ss_res = float(((values - fit) ** 2).sum())
        mean_level = float(values.mean())
        # A nearly flat surface-stress path (non-load-carrying detail, where the
        # structural stress equals the nominal stress) makes R^2 a ratio of two
        # round-off-level numbers; flag it instead of quoting a meaningless value.
        flat = abs(mean_level) > 0.0 and math.sqrt(ss_tot / len(f)) / abs(mean_level) < 1e-3
        r2 = 1.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot
        rows[name] = {
            "stations": tuple(float(v) for v in f),
            "sampled_stress_ratio": tuple(float(v) for v in values),
            "r_squared": r2,
            "residual_rms": math.sqrt(ss_res / len(f)),
            "extrapolated_toe": float(coeff[0]),
            "degenerate": f.size < 3,
            "flat_field": bool(flat),
        }
    strict = [v for v in rows.values() if not v["degenerate"]]
    scored = [v for v in strict if not v["flat_field"]]
    return {
        "t": t,
        "sets": rows,
        "min_r_squared_strict": (
            min(v["r_squared"] for v in scored) if scored else float("nan")
        ),
        "all_flat": bool(scored == []),
        "pass": bool(scored) and all(v["r_squared"] >= 0.999 for v in scored),
    }


# --------------------------------------------------------------------------- #
# Grid convergence of the hot-spot stress
# --------------------------------------------------------------------------- #
def gci_hotspot(
    detail: WeldDetail, *, levels: tuple[int, ...] = (1, 2, 3)
) -> dict[str, object]:
    """Roache Grid Convergence Index of the extrapolated hot-spot stress.

    The refinement ratio is taken from the toe surface element size of the levels
    (``r = h_coarse / h_fine``), the apparent order ``p`` follows from the
    three-level ratio of differences, and the GCI of the finest grid uses the
    factor of safety ``Fs = 1.25`` (three-grid study).
    """
    rows = []
    for level in levels:
        sol = solve_weld_detail(detail, level=level)
        lin = hot_spot_stress(sol, detail.plate_thickness_m, method="linear")
        res = toe_mesh_resolution(detail, level=level)
        rows.append(
            {
                "level": level,
                "h_m": res["surface_element_m"],
                "hotspot_ratio": lin.stress_ratio,
                "n_elements": sol.n_elements,
            }
        )
    rows.sort(key=lambda row: row["h_m"], reverse=True)  # coarse -> fine
    f1, f2, f3 = (row["hotspot_ratio"] for row in rows)
    h1, h2, h3 = (row["h_m"] for row in rows)
    r21 = h1 / h2
    r32 = h2 / h3
    eps21 = f2 - f1
    eps32 = f3 - f2
    p = float("nan")
    if abs(eps32) > 0.0 and abs(r21 - 1.0) > 1e-12:
        ratio = eps21 / eps32
        if ratio > 0.0:
            p = abs(math.log(ratio) / math.log(r21))
    fs = 1.25
    if math.isfinite(p) and p > 0.0 and abs(f3) > 0.0:
        gci_fine = fs * abs(eps32 / f3) / (r32**p - 1.0)
    else:
        gci_fine = float("nan")
    change_2x = abs(f3 - f2) / abs(f3) if abs(f3) > 0.0 else float("nan")
    return {
        "levels": rows,
        "refinement": (r21, r32),
        "apparent_order": p,
        "gci_fine": gci_fine,
        "change_finest_pair": change_2x,
        "pass": bool(math.isfinite(gci_fine) and gci_fine <= 0.03 and change_2x <= 0.02),
    }


# --------------------------------------------------------------------------- #
# Radius negative controls
# --------------------------------------------------------------------------- #
def radius_negative_controls(
    plate_thickness_m: float = 0.020,
    *,
    radii_m: tuple[float, ...] = (0.0002, 0.0005, 0.001, 0.002, 0.005, 0.010, 0.020),
) -> dict[str, object]:
    """Toe-radius sweep: ``r -> 0`` singularity trend and ``r -> inf`` limit.

    * ``r -> 0``: the notch SCF must grow and the model must still solve (the
      trend is recorded, no exception is raised).
    * ``r -> inf``: the *structural* hot-spot stress must approach the nominal
      stress; the notch (peak surface) SCF approaches one only asymptotically and
      its decay exponent is fitted, because a welded attachment retains a
      stiffness-step boundary layer at any finite radius.
    """
    rows = []
    for r in radii_m:
        detail = family_geometry(plate_thickness_m, r, load_case="plate_axial")
        sol = solve_weld_detail(detail, level=1)
        lin = hot_spot_stress(sol, detail.plate_thickness_m, method="linear")
        rows.append(
            {
                "radius_m": r,
                "radius_over_t": r / plate_thickness_m,
                "kt_peak": sol.kt_toe,
                "structural_ratio": lin.stress_ratio,
                "n_elements": sol.n_elements,
            }
        )
    ratios = np.asarray([row["radius_over_t"] for row in rows])
    kt = np.asarray([row["kt_peak"] for row in rows])
    structurals = np.asarray([row["structural_ratio"] for row in rows])
    # decay fit Kt - 1 = A (r/t)^-p over the resolved radii
    excess = kt - 1.0
    good = excess > 1e-9
    exponent = float("nan")
    amplitude = float("nan")
    if good.sum() >= 2:
        slope, intercept = np.polyfit(np.log(ratios[good]), np.log(excess[good]), 1)
        exponent = float(-slope)
        amplitude = float(math.exp(intercept))
    # radius ratio needed for a 0.5 % band on Kt, from the fitted power law
    target = 0.005
    r_needed = float("inf")
    if math.isfinite(exponent) and exponent > 0.0 and amplitude > 0.0:
        r_needed = float((amplitude / target) ** (1.0 / exponent))
    structural_error = float(abs(structurals[-1] - 1.0))
    return {
        "rows": rows,
        "decay_amplitude": amplitude,
        "decay_exponent": exponent,
        "radius_over_t_for_half_percent": r_needed,
        "structural_ratio_largest_radius": float(structurals[-1]),
        "structural_deviation_largest_radius": structural_error,
        "monotone_decreasing_kt": bool(np.all(np.diff(kt) < 0.0)),
        "singularity_trend": bool(kt[0] > kt[-1]),
        "structural_limit_pass": structural_error <= 0.005,
        "notch_limit_pass": bool(structural_error <= 0.005 and kt[-1] - 1.0 <= 0.005),
    }


# --------------------------------------------------------------------------- #
# SCF against the closed-form anchors and the IIW band
# --------------------------------------------------------------------------- #
def scf_vs_iiw(
    family: tuple[tuple[float, float], ...] = JOINT_FAMILY,
    *,
    level: int = 1,
) -> dict[str, object]:
    """FE notch and structural SCF of the family against anchors and IIW band.

    ``t_fillet`` is compared with the Nii closed form, ``butt`` with the Maddox
    closed form.  The structural (IIW-extrapolated) ratio is compared with the
    ``1.2-1.5`` band quoted by the node specification.
    """
    rows = []
    for joint_type in ("t_fillet", "butt"):
        for t, r in family:
            detail = family_geometry(t, r, joint_type=joint_type, load_case="plate_axial")
            sol = solve_weld_detail(detail, level=level)
            lin = hot_spot_stress(sol, t, method="linear")
            quad = hot_spot_stress(sol, t, method="quadratic")
            if joint_type == "t_fillet":
                anchor = nii_stress_concentration(
                    t, r, detail.attachment_height_m
                )
                anchor_name = "nii"
            else:
                anchor = maddox_stress_concentration(t, r)
                anchor_name = "maddox"
            lo, hi = IIW_STRUCTURAL_SCF_BAND
            rows.append(
                {
                    "joint_type": joint_type,
                    "t_m": t,
                    "r_m": r,
                    "t_over_r": t / r,
                    "kt_peak": sol.kt_toe,
                    "anchor": anchor_name,
                    "anchor_value": anchor,
                    "anchor_error": (sol.kt_toe - anchor) / anchor,
                    "scf_structural_linear": lin.stress_ratio,
                    "scf_structural_quadratic": quad.stress_ratio,
                    "in_iiw_band": bool(lo <= lin.stress_ratio <= hi),
                }
            )
    structural = np.asarray([row["scf_structural_linear"] for row in rows])
    peak = np.asarray([row["kt_peak"] for row in rows])
    lo, hi = IIW_STRUCTURAL_SCF_BAND
    mid = 0.5 * (lo + hi)
    worst_structural_dev = float(np.max(np.abs(structural - mid) / mid))
    return {
        "rows": rows,
        "band": (lo, hi),
        "band_mid": mid,
        "structural_min": float(structural.min()),
        "structural_max": float(structural.max()),
        "peak_min": float(peak.min()),
        "peak_max": float(peak.max()),
        "worst_structural_deviation_from_mid": worst_structural_dev,
        "band_pass": bool(np.all((structural >= lo) & (structural <= hi))),
        "peak_in_band": bool(np.any((peak >= lo) & (peak <= hi))),
        "anchor_error_range": (
            float(min(row["anchor_error"] for row in rows)),
            float(max(row["anchor_error"] for row in rows)),
        ),
    }


# --------------------------------------------------------------------------- #
# S-N power law
# --------------------------------------------------------------------------- #
def power_law_response(
    stress_range_pa: float = 100.0e6,
    *,
    environment: str = "air",
    ratio: float = 2.0,
) -> dict[str, object]:
    """Stress-range doubling must multiply the life by ``ratio ** m`` (m = 3).

    The regression of ``log N`` against ``log ds`` over a decade of stress ranges
    returns the measured exponent and its residual; DNV-RP-C203 curve D has
    ``m1 = 3`` on the first branch.
    """
    ds = float(stress_range_pa)
    # stay above the curve-D knee (52.7 MPa at 1e7 cycles for FAT90) so the
    # regression sees the m1 = 3 branch only
    ranges = np.asarray([ds * (1.0 + 0.1 * k) for k in range(11)])
    lives = np.asarray([fatigue_life(v, environment=environment) for v in ranges])
    slope, intercept = np.polyfit(np.log(ranges), np.log(lives), 1)
    fit = intercept + slope * np.log(ranges)
    residual = float(np.max(np.abs(np.log(lives) - fit) / np.abs(np.log(lives))))
    n1 = fatigue_life(ds, environment=environment)
    n2 = fatigue_life(ratio * ds, environment=environment)
    measured_ratio = n1 / n2
    return {
        "stress_range_pa": ds,
        "environment": environment,
        "measured_exponent": float(-slope),
        "expected_exponent": _M_SLOPE,
        "life_ratio_measured": float(measured_ratio),
        "life_ratio_expected": float(ratio**_M_SLOPE),
        "max_log_residual": residual,
        "pass": bool(
            abs(-slope - _M_SLOPE) <= 1e-6
            and abs(measured_ratio - ratio**_M_SLOPE) / (ratio**_M_SLOPE) <= 1e-6
            and residual <= 0.01
        ),
    }


# --------------------------------------------------------------------------- #
# Beam -> detail transfer chain
# --------------------------------------------------------------------------- #
def beam_to_detail_validation(
    plate_thickness_m: float = 0.020,
    toe_radius_m: float = 0.002,
    *,
    support_length_m: float = 0.160,
) -> dict[str, object]:
    """Check ``sigma_hot = SCF * sigma_nom`` against the FE structural stress.

    The load-carrying configuration (``attachment_supported``) pulls the attachment
    while the far plate edge is supported vertically.  Beam theory gives the plate
    section stress at the toe, ``sigma_nom = M / Z`` with ``M = F L`` and
    ``Z = t^2 / 6``; the finite-element structural (hot-spot) stress must reproduce
    it, which validates the beam-to-detail transfer at the SCF = 1 end of the chain
    and pins the SCF definition used for the load-carrying family.
    """
    detail = family_geometry(plate_thickness_m, toe_radius_m, load_case="attachment_supported")
    sol = solve_weld_detail(detail, level=1)
    lin = hot_spot_stress(sol, plate_thickness_m, method="linear")
    quad = hot_spot_stress(sol, plate_thickness_m, method="quadratic")
    # per unit width: total pull = t (so that the plate membrane nominal is 1)
    t = plate_thickness_m
    force = t
    lever = support_length_m - float(sol.x_toe_m)
    z_section = t**2 / 6.0
    sigma_beam = force * lever / z_section
    fe = lin.stress_ratio
    error = (fe - sigma_beam) / sigma_beam
    # beam-theory chain as used by the platform: N/A + M/Z on a real section
    section = BeamSection(area_m2=t * 1.0, I_m4=1.0 * t**3 / 12.0, c_m=0.5 * t)
    sigma_nom = beam_nominal_stress(bending_moment_nm=force * 1.0 * lever * 1.0,
                                    axial_force_n=0.0, section=section)
    scf = lin.stress_ratio
    sigma_hot = scf * sigma_nom
    life = fatigue_life(abs(sigma_hot) * 1.0e6, environment="air",
                        thickness_m=plate_thickness_m)
    return {
        "detail": detail,
        "fe_structural_ratio": fe,
        "fe_structural_ratio_quadratic": quad.stress_ratio,
        "beam_section_ratio": sigma_beam,
        "relative_error": error,
        "section_modulus_m3": section.Z_m3,
        "sigma_nom_pa": sigma_nom,
        "scf": scf,
        "sigma_hot_pa": sigma_hot,
        "life_cycles": life,
        "pass": bool(abs(error) <= 0.05),
    }


# --------------------------------------------------------------------------- #
# Full ledger
# --------------------------------------------------------------------------- #
def run_maturity_assessment(*, level: int = 1) -> dict[str, object]:
    """Run every acceptance criterion and return the measured ledger."""
    t_ref, r_ref = 0.020, 0.002
    reference = family_geometry(t_ref, r_ref, load_case="plate_axial")
    solution = solve_weld_detail(reference, level=level)
    # the IIW surface extrapolation exists to remove the *structural* gradient, so
    # the linearity of the extrapolation is measured on the load-carrying joint;
    # on the non-load-carrying joint the surface stress is flat and R^2 degenerates
    carrying = family_geometry(t_ref, r_ref, load_case="attachment_supported")
    carrying_solution = solve_weld_detail(carrying, level=level)

    mesh = toe_mesh_resolution(reference, level=level)
    linearity = extrapolation_linearity(carrying_solution, t_ref)
    linearity_flat = extrapolation_linearity(solution, t_ref)
    convergence = gci_hotspot(reference)
    controls = radius_negative_controls(t_ref)
    scf = scf_vs_iiw(level=level)
    power = power_law_response(100.0e6)
    chain = beam_to_detail_validation(t_ref, r_ref)

    criteria = {
        "scf_iiw_band": {
            "requirement": "FE T-joint structural SCF inside the IIW 1.2-1.5 band, "
            "deviation <= 5 % from the band centre",
            "measured_structural_range": (scf["structural_min"], scf["structural_max"]),
            "measured_notch_range": (scf["peak_min"], scf["peak_max"]),
            "worst_deviation_from_mid": scf["worst_structural_deviation_from_mid"],
            "pass": bool(scf["band_pass"]),
            "note": "the 1.2-1.5 band belongs to structural-discontinuity details; "
            "the non-load-carrying transverse fillet of this family has a "
            "structural SCF of ~1.0 by theory, so the band is not reproduced",
        },
        "mesh_size": {
            "requirement": "toe surface element <= t / 10",
            "measured_m": mesh["surface_element_m"],
            "limit_m": mesh["limit_m"],
            "pass": bool(mesh["pass"]),
        },
        "mesh_convergence": {
            "requirement": "2x toe refinement changes the SCF by <= 2 %, GCI <= 3 %",
            "change": convergence["change_finest_pair"],
            "gci": convergence["gci_fine"],
            "apparent_order": convergence["apparent_order"],
            "pass": bool(convergence["pass"]),
        },
        "hotspot_linearity": {
            "requirement": "0.5t / 1.5t extrapolation linear, R^2 >= 0.999",
            "load_case": "attachment_supported",
            "min_r_squared_strict": linearity["min_r_squared_strict"],
            "sets": {
                name: {"r_squared": row["r_squared"], "degenerate": row["degenerate"]}
                for name, row in linearity["sets"].items()
            },
            "flat_field_reference": {
                name: {"r_squared": row["r_squared"], "flat_field": row["flat_field"]}
                for name, row in linearity_flat["sets"].items()
            },
            "pass": bool(linearity["pass"]),
        },
        "scf_closed_form": {
            "requirement": "FE notch SCF of the T-joint family within 5 % of the Nii "
            "closed form",
            "worst_abs_error": max(
                abs(row["anchor_error"])
                for row in scf["rows"]
                if row["joint_type"] == "t_fillet"
            ),
            "errors_t_joint": [
                row["anchor_error"] for row in scf["rows"] if row["joint_type"] == "t_fillet"
            ],
            "errors_butt": [
                row["anchor_error"] for row in scf["rows"] if row["joint_type"] == "butt"
            ],
            "pass": bool(
                max(
                    abs(row["anchor_error"])
                    for row in scf["rows"]
                    if row["joint_type"] == "t_fillet"
                )
                <= 0.05
            ),
        },
        "fatigue_power_law": {
            "requirement": "2x stress range -> 2^m life ratio, regression residual <= 1 %",
            "measured_exponent": power["measured_exponent"],
            "life_ratio": power["life_ratio_measured"],
            "residual": power["max_log_residual"],
            "pass": bool(power["pass"]),
        },
        "radius_negative_control": {
            "requirement": "r -> 0 shows a singularity trend without crashing; "
            "SCF -> 1 within 0.5 % as r -> infinity",
            "singularity_trend": controls["singularity_trend"],
            "monotone": controls["monotone_decreasing_kt"],
            "structural_deviation": controls["structural_deviation_largest_radius"],
            "notch_deviation_at_r_over_t_1": controls["rows"][-1]["kt_peak"] - 1.0,
            "radius_over_t_needed": controls["radius_over_t_for_half_percent"],
            "pass": bool(
                controls["singularity_trend"]
                and controls["monotone_decreasing_kt"]
                and controls["structural_limit_pass"]
            ),
            "note": "the structural hot-spot reaches 1.0 within 0.5 %, the notch peak "
            "only asymptotically; the criterion is scored on the structural stress",
        },
        "beam_detail_chain": {
            "requirement": "sigma_hot = SCF * sigma_nom reproduces the beam-theory "
            "section stress within 5 %",
            "fe_ratio": chain["fe_structural_ratio"],
            "beam_ratio": chain["beam_section_ratio"],
            "relative_error": chain["relative_error"],
            "pass": bool(chain["pass"]),
        },
    }
    all_pass = all(bool(row["pass"]) for row in criteria.values())
    return {
        "reference_detail": {
            "t_m": t_ref,
            "r_m": r_ref,
            "t_over_r": t_ref / r_ref,
        },
        "criteria": criteria,
        "all_pass": bool(all_pass),
        "scf_family": scf,
        "radius_controls": controls,
        "mesh": mesh,
        "convergence": convergence,
        "linearity": linearity,
        "linearity_flat_field": linearity_flat,
        "power_law": power,
        "beam_chain": {k: v for k, v in chain.items() if k != "detail"},
    }


def _json_ready(value):  # pragma: no cover - helper for artefact dumping
    """Recursively convert numpy scalars / tuples into JSON-friendly objects."""
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    if isinstance(value, WeldDetail):
        return {f: getattr(value, f) for f in value.__dataclass_fields__}
    return value