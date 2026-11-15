"""Verification face for the ice-grillage ultimate strength module.

Every capability claimed by :mod:`tensorfem.grillage_ice_ultimate` is bound to an
executable anchor here; the numbers are computed, never transcribed.  The gates
are those of the ice-grillage acceptance item: a benchmark comparable with
literature/rule within 15 %, >= 3 combined load cases and a machine-readable
panel-stress -> ultimate-load map.

Anchors
-------
``T1a`` / ``T1b``
    Isotropic SSSS plate under a uniform load versus the classical series
    coefficients ``alpha = 0.00406`` (deflection) and ``beta = 0.0479`` (bending
    moment) of Timoshenko & Woinowsky-Krieger, *Theory of Plates and Shells*,
    Table 30.  Gate 3 %.
``T2``
    Patch-load Navier convergence, ``terms = 31`` versus ``terms = 121``.
    Gate 0.5 %.
``T3``
    Isotropic load-share symmetry (``f_x == f_y``) and unit sum of the modal
    partition.
``T4``
    Orthotropy limit: ``Dx/Dy = 1e6`` drives the modal share onto the
    x-direction.
``T5``
    Exact beam-column amplification ``2(sec u - 1)/u^2`` against the
    ``1/(1 - P/Pe)`` approximation; the exact factor is never the smaller one.
``T6``
    Rigid-plastic frame collapse: the uniform-load closed form ``16 Mp / L`` and
    the point-load limit ``8 Mp / L`` are reproduced, and the capacity grows
    monotonically with the loaded length.
``T7``
    Axial-only grillage ultimate stress against the Paik-Thayamballi empirical
    stiffened-panel strength already verified in
    ``tensorfem.stiffened_panel_ultimate`` for three plate slenderness ratios.
    Gate 15 % (the literature/rule comparison the acceptance item demands).
``T8``
    Monotonicity: the ultimate ice patch pressure falls as the coexisting
    compression grows and as the hydrostatic pressure grows.
``T9``
    Load-path equilibrium: the force handed to the stiffener system plus the
    force handed to the frame system equals the applied patch + hydrostatic
    force to machine precision.
``T10``
    The closed-form central-patch beam moment ``F(2L - c)/8`` agrees with an
    independent numeric integration of the shear/moment field.
``T11``
    The panel plate field that *binds* every case-study ultimate is recomputed
    with an independently discretised solver (``orthotropic_plate_fd``, 13-point
    biharmonic stencil, exact patch integration, antisymmetric ghost nodes) on
    the real 2.1 m x 0.35 m panel and the real patch of each acceptance case.
    Two gates per case: peak moment (series vs FD) and the resulting ultimate
    pressure (module bisection vs the FD algebra).  Gate 0.5 %.
``T12``
    A stiffened-panel scantling on which the **longitudinal stiffener** (not the
    plate field) governs the ultimate, so that the member-limit branch of the
    grillage layer is actually exercised by an acceptance-tier combined case.
    The ultimate pressure is recomputed from independent algebra (own section
    properties, own Perry-Robertson / Johnson-Ostenfeld, own secant
    magnification and own closed-form solve with the bisection removed) and
    gated at 0.5 %; the plate field must stay below the member at the limit.
``T13``
    Cross-method invariant on that member-governed scantling: the rigid-plastic
    grillage collapse of ``grillage_plastic_collapse`` (successive plastic hinges,
    a method that shares no member-limit logic with the elastic checks) may not
    fall below the elastic member limit -- plastic redistribution can only add
    reserve.  A violation would falsify one of the two methods.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    beam_column_amplification_approx,
    beam_column_amplification_exact,
    beam_moment_centre_patch,
    case_study_geometry,
    frame_plastic_collapse_force,
    grillage_utilisations,
    panel_stress_ultimate_map,
    plate_patch_response,
    stiffener_axial_capacity,
    ultimate_ice_pressure,
    write_ultimate_map,
)
from .orthotropic_plate_fd import ssss_orthotropic_plate_fd

__all__ = [
    "case_study_geometry_slender",
    "member_governed_geometry",
    "member_governed_case_table",
    "plate_field_fd_check",
    "run_grillage_verification",
    "combined_case_table",
    "CASE_STUDY_SCENARIOS",
    "MEMBER_GOVERNED_SCENARIOS",
    "main",
]


def case_study_geometry_slender() -> GrillageGeometry:
    """Thinner-plate variant used for the 15 % panel-strength comparison (T7).

    Measured correction (independent audit): this scantling does **not** reach a
    stiffener-governed ultimate under any combined load tried -- the plate field
    still binds (stiffener utilisation <= 0.76 at the limit, see
    ``plate_field_fd_check`` and the audit note in
    ``docs/AUDIT-GAP-PP-5-GRILLAGE-ULTIMATE-independent-review.md``).  The
    member-governed branch is exercised by :func:`member_governed_geometry`
    instead.
    """
    return GrillageGeometry(
        frame_spacing=2.1, bay_spacing=0.45, n_bays=6, n_spans=4,
        plate_thickness=0.013,
        stiffener=MemberSection(0.20, 0.009, 0.060, 0.010),
        frame=MemberSection(0.50, 0.010, 0.100, 0.012),
        young=2.06e11, poisson=0.3, yield_stress=3.55e8,
    ).validate()




CASE_STUDY_SCENARIOS: tuple[tuple[str, float, float, float, float, float], ...] = (
    # name, sigma_x / sigma_y, ice patch pressure [Pa], patch length [m],
    # patch width [m], hydrostatic pressure [Pa]
    ("light-ice + ballast head", 0.15, 1.0e6, 0.35, 0.35, 5.0e4),
    ("design-ice + design head", 0.30, 2.0e6, 0.70, 0.35, 1.0e5),
    ("severe-ice + tank head", 0.45, 3.0e6, 1.05, 0.35, 1.5e5),
)

MEMBER_GOVERNED_SCENARIOS: tuple[tuple[str, float, float, float, float, float], ...] = (
    # same layout as CASE_STUDY_SCENARIOS, for the member-governed scantling
    ("light ice + ballast head", 0.00, 1.0e6, 0.55, 0.55, 5.0e4),
    ("design ice + design head", 0.15, 1.5e6, 1.10, 0.55, 1.0e5),
    ("severe ice + tank head", 0.30, 2.0e6, 1.65, 0.55, 1.5e5),
)


def member_governed_geometry() -> GrillageGeometry:
    """Scantling on which the longitudinal stiffener, not the plate, governs.

    Documentation example, not a rule table: 26 mm plating over a 0.55 m bay
    keeps the plate field above the member level, while the shallow 160 x 6 web
    makes one stiffener line the weakest link.  The measured utilisation split at
    the limit is reported by :func:`member_governed_case_table`; the branch is
    what the acceptance item means by a *grillage* ultimate rather than a
    plate-panel ultimate.

    The plating was re-set from 25 mm to 26 mm in round 8 so that the plate
    utilisation at the member limit stays below 0.9 on every combined case.  The
    round-7 web-origin correction (web welded to the plate underside) had pushed
    the light-ice case's plate utilisation to 0.911, which forced the regression
    lock in ``tests/test_grillage_ultimate_independent_audit.py`` to 0.95; the
    +1 mm re-set restores the 0.9-level margin (0.838 / 0.759 / 0.719 on the
    three combined cases) while keeping the longitudinal stiffener the governing
    member -- see ``docs/CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round8.md``.
    """
    return GrillageGeometry(
        frame_spacing=2.4, bay_spacing=0.55, n_bays=4, n_spans=4,
        plate_thickness=0.026,
        stiffener=MemberSection(0.16, 0.006, 0.05, 0.007),
        frame=MemberSection(0.55, 0.010, 0.120, 0.014),
    ).validate()


def _effective_section(geom: GrillageGeometry, plate_width: float
                       ) -> tuple[float, float, float, float]:
    """``(A_e, I_e, c, Z)`` of the stiffener plus a ``plate_width`` strip.

    Independent of ``MemberSection``: the laminae are laid out explicitly
    (attached plate centred at ``z = 0``, web welded to the plate *underside*
    at ``z = -t/2``, flange below the web) so that a defect in the
    composite-property algebra cannot hide in both places.  The junction
    convention (web from the plate underside, no double-counted overlap) is the
    only thing shared with the module, and it is stated here on purpose -- see
    ``docs/CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round7.md``.
    """
    t = geom.plate_thickness
    h = geom.stiffener.web_height
    tw = geom.stiffener.web_thickness
    bf = geom.stiffener.flange_width
    tf = geom.stiffener.flange_thickness
    z0 = -0.5 * t  # web root: plate underside, not the plate mid-plane
    parts = [
        (plate_width * t, 0.0, plate_width * t ** 3 / 12.0),
        (h * tw, z0 - 0.5 * h, tw * h ** 3 / 12.0),
        (bf * tf, z0 - h - 0.5 * tf, bf * tf ** 3 / 12.0),
    ]
    area = sum(a for a, _, _ in parts)
    z_bar = sum(a * z for a, z, _ in parts) / area
    inertia = sum(a * (z - z_bar) ** 2 + i0 for a, z, i0 in parts)
    c_ext = max(max(abs(z - z_bar) for _, z, _ in parts), abs(z_bar))
    return area, inertia, c_ext, inertia / c_ext


def hand_stiffener_limit(geom: GrillageGeometry, case: CombinedLoadCase, *,
                         terms: int = 61) -> dict[str, float]:
    """Ultimate ice pressure from the stiffener branch, by explicit algebra.

    The module solves the same limit by bisecting a utilisation function; here
    the whole chain is written out and solved in closed form, which makes the
    bisection, the utilisation composition and the section algebra checkable
    against something that shares none of them:

    ``A_e = be t + h tw + bf tf``,  ``Z = I_e / c``  (own laminae above, web
    root at the plate underside),
    ``sigma_e = pi^2 E I_e / (A_e L^2)``, ``eta_P = e0 c A_e / I_e``,
    ``sigma_PR`` from the Perry-Robertson quadratic, ``sigma_JO`` by
    Johnson-Ostenfeld, ``sigma_L = min`` thereof,
    ``beta_m = 2(sec u - 1)/u^2`` with ``u = (L/2) sqrt(P / E I_e)``,
    ``M(p) = (p A_patch f_x / n_ovl + M_h)(2L - c_p)/8`` and finally
    ``p_ult = ((sigma_L - sigma_x) Z / beta_m - M_h_per) / k`` with
    ``k = M`` per unit ice pressure.

    The one ingredient taken from the module is the elastic modal load share
    ``f_x`` (a plate partition, gated by T3/T4/T9); everything downstream is
    independent.
    """
    geom.validate()
    case.validate(geom)
    sigma_y = geom.yield_stress
    span = geom.frame_spacing
    lx, ly = case.patch_extents(geom)
    if lx <= 0.0 or ly <= 0.0:
        raise ValueError("hand_stiffener_limit needs a patch")

    # Faulkner effective width (own transcription)
    beta = geom.bay_spacing / geom.plate_thickness * math.sqrt(sigma_y / geom.young)
    ratio = 1.0 if beta <= 1.0 else min(1.0, 2.0 / beta - 1.0 / beta ** 2)
    be = ratio * geom.bay_spacing

    area_e, inertia_e, c_ext, z_section = _effective_section(geom, be)
    sigma_e = math.pi ** 2 * geom.young * inertia_e / (area_e * span ** 2)
    e0 = geom.imperfection_ratio * span
    eta_p = e0 * c_ext * area_e / inertia_e
    s = sigma_y + sigma_e * (1.0 + eta_p)
    sigma_pr = (s - math.sqrt(max(s * s - 4.0 * sigma_y * sigma_e, 0.0))) / 2.0
    sigma_jo = (sigma_y * (1.0 - sigma_y / (4.0 * sigma_e))
                if sigma_e >= 0.5 * sigma_y else sigma_e)
    sigma_l = min(sigma_pr, sigma_jo)

    axial_force = case.axial_stress * area_e
    pe = math.pi ** 2 * geom.young * inertia_e / span ** 2
    if 0.0 < axial_force < pe:
        u = 0.5 * math.sqrt(axial_force * span ** 2 / (geom.young * inertia_e))
        beta_m = 2.0 * (1.0 / math.cos(u) - 1.0) / u ** 2
    elif axial_force >= pe:
        beta_m = 10.0
    else:
        beta_m = 1.0

    # load share: elastic modal partition of the smeared orthotropic plate
    dx, dy, d3 = geom.plate_rigidities()
    from .grillage_ice_ultimate import system_load_shares
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    patch = (0.5 - 0.5 * lx / span_x, 0.5 + 0.5 * lx / span_x,
             0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y)
    resp = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3,
                                pressure=1.0, patch=patch, terms=terms,
                                poisson=geom.poisson)
    f_x, _ = system_load_shares(resp)
    n_ovl = min(max(2, int(round(ly / geom.bay_spacing)) + 1), geom.n_stiffeners)

    k = (lx * ly * f_x / n_ovl) * (2.0 * span - min(lx, span)) / 8.0
    if case.hydrostatic_pressure > 0.0:
        resp_h = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                      d3=d3, pressure=1.0,
                                      patch=(0.0, 1.0, 0.0, 1.0), terms=terms,
                                      poisson=geom.poisson)
        f_x_h, _ = system_load_shares(resp_h)
        per_stiff_h = case.hydrostatic_pressure * geom.panel_area * f_x_h \
            / geom.n_stiffeners
        m_h = per_stiff_h * span / 8.0  # (2L - L) / 8 with a full-span load
    else:
        m_h = 0.0

    p_ult = ((sigma_l - case.axial_stress) * z_section / beta_m - m_h) / k
    return {
        "p_ult": p_ult,
        "effective_width": be,
        "area_effective": area_e,
        "inertia_effective": inertia_e,
        "section_modulus": z_section,
        "limiting_stress": sigma_l,
        "perry_robertson_stress": sigma_pr,
        "johnson_ostenfeld_stress": sigma_jo,
        "euler_stress": sigma_e,
        "moment_amplification": beta_m,
        "load_share_x": f_x,
        "stiffeners_touched": float(n_ovl),
        "moment_per_pascal": k,
        "hydrostatic_moment": m_h,
    }


def plate_field_fd_check(geom: GrillageGeometry | None = None, *,
                         nx: int = 160, terms: int = 61) -> list[dict[str, object]]:
    """Recompute the binding plate field of the case study with an FD solver.

    For every acceptance case the panel field ``a x b`` (frame spacing x bay
    spacing) is solved twice: by the module's Navier series (the field the
    ``plate_local_bending`` utilisation is built from) and by
    ``orthotropic_plate_fd``, which discretises the biharmonic operator with a
    13-point stencil, integrates the patch exactly over each cell and enforces
    ``w = w,nn = 0`` with antisymmetric ghost nodes -- a different method, a
    different discretisation and no shared code.  The ultimate pressure is then
    recomputed from the FD moments in closed form (``p_ult = (m_p_eff - p_h
    m_h) / m_1``) and compared with the module's bisection.
    """
    geom = (case_study_geometry() if geom is None else geom).validate()
    a, b = geom.frame_spacing, geom.bay_spacing
    d = geom.plate_rigidity
    cap = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    hydro_fd = ssss_orthotropic_plate_fd(
        span_x=a, span_y=b, d1=d, d2=d, d3=d, pressure=1.0, nx=nx,
        poisson=geom.poisson)
    m_h_fd = max(float(hydro_fd["m_x_max"]), float(hydro_fd["m_y_max"]))
    rows: list[dict[str, object]] = []
    for name, eta, p_ice, lx, ly, p_h in CASE_STUDY_SCENARIOS:
        patch = (0.5 - 0.5 * lx / a, 0.5 + 0.5 * lx / a,
                 0.5 - 0.5 * ly / b, 0.5 + 0.5 * ly / b)
        series = plate_patch_response(span_x=a, span_y=b, d1=d, d2=d, d3=d,
                                      pressure=1.0, patch=patch, terms=terms,
                                      sample=41, poisson=geom.poisson)
        fd = ssss_orthotropic_plate_fd(span_x=a, span_y=b, d1=d, d2=d, d3=d,
                                       pressure=1.0, patch=patch, nx=nx,
                                       poisson=geom.poisson)
        m_series = max(float(series["m_x_max"]), float(series["m_y_max"]))
        m_fd = max(float(fd["m_x_max"]), float(fd["m_y_max"]))
        cap_eff = cap * math.sqrt(max(1.0 - eta ** 2, 0.0))
        p_fd = (cap_eff - p_h * m_h_fd) / m_fd
        case = CombinedLoadCase(
            axial_stress=eta * geom.yield_stress, ice_pressure=p_ice,
            patch_length=lx, patch_width=ly, hydrostatic_pressure=p_h,
            design_pressure=p_ice, design_provenance="scenario design pressure")
        res = ultimate_ice_pressure(geom, case, terms=terms)
        u_lim = res["utilisations_at_limit"]
        rows.append({
            "scenario": name,
            "sigma_x_over_yield": eta,
            "patch_length_m": lx,
            "patch_width_m": ly,
            "hydrostatic_MPa": p_h / 1e6,
            "peak_moment_series_per_Pa": m_series,
            "peak_moment_fd_per_Pa": m_fd,
            "peak_moment_rel_error": m_fd / m_series - 1.0,
            "p_ult_module_MPa": res["ice_pressure_ultimate"] / 1e6,
            "p_ult_fd_MPa": p_fd / 1e6,
            "p_ult_rel_error": p_fd / res["ice_pressure_ultimate"] - 1.0,
            "governing_member": res["governing_member"],
            "utilisation_plate_at_limit": u_lim["plate_local_bending"],
            "utilisation_stiffener_at_limit": u_lim["longitudinal_stiffener"],
            "utilisation_frame_at_limit": u_lim["transverse_frame"],
            "fd_grid": [int(nx), int(round(nx * b / a))],
        })
    return rows


def member_governed_case_table(geom: GrillageGeometry | None = None, *,
                               terms: int = 61) -> list[dict[str, object]]:
    """Three combined cases on the scantling where the stiffener governs.

    Each row carries the member utilisations at the limit, the governing member
    (``longitudinal_stiffener``), the ultimate patch pressure, the safety margin
    against the scenario design pressure and the closed-form recomputation of
    :func:`hand_stiffener_limit` for comparison.
    """
    geom = (member_governed_geometry() if geom is None else geom).validate()
    rows: list[dict[str, object]] = []
    for name, eta, p_ice, lx, ly, p_h in MEMBER_GOVERNED_SCENARIOS:
        case = CombinedLoadCase(
            axial_stress=eta * geom.yield_stress, ice_pressure=p_ice,
            patch_length=lx, patch_width=ly, hydrostatic_pressure=p_h,
            design_pressure=p_ice, design_provenance="scenario design pressure")
        res = ultimate_ice_pressure(geom, case, terms=terms)
        u_lim = res["utilisations_at_limit"]
        hand = hand_stiffener_limit(geom, case, terms=terms)
        rows.append({
            "scenario": name,
            "sigma_x_over_yield": eta,
            "sigma_x_MPa": eta * geom.yield_stress / 1e6,
            "ice_patch_MPa": p_ice / 1e6,
            "patch_length_m": lx,
            "patch_width_m": ly,
            "hydrostatic_MPa": p_h / 1e6,
            "utilisation_plate_at_limit": float(u_lim["plate_local_bending"]),
            "utilisation_stiffener_at_limit": float(u_lim["longitudinal_stiffener"]),
            "utilisation_frame_at_limit": float(u_lim["transverse_frame"]),
            "ice_pressure_ultimate_MPa": float(res["ice_pressure_ultimate"]) / 1e6,
            "governing_member": res["governing_member"],
            "safety_margin": res["margin_vs_design"],
            "p_ult_hand_closed_form_MPa": hand["p_ult"] / 1e6,
            "rel_error_hand_vs_module": hand["p_ult"] / res["ice_pressure_ultimate"] - 1.0,
            "stiffener_limiting_stress_MPa": hand["limiting_stress"] / 1e6,
            "moment_amplification": hand["moment_amplification"],
            "stiffeners_touched": hand["stiffeners_touched"],
        })
    return rows


def _record(checks: list[dict[str, object]], name: str, value: float,
            reference: float, tol: float, note: str) -> None:
    err = abs(value - reference) / abs(reference) if reference else abs(value)
    checks.append({"id": name, "value": float(value), "reference": float(reference),
                   "rel_error": float(err), "tolerance": float(tol),
                   "passed": bool(err <= tol), "note": note})


def _assert(checks: list[dict[str, object]], name: str, ok: bool,
            detail: str) -> None:
    checks.append({"id": name, "value": float(ok), "reference": 1.0,
                   "rel_error": 0.0 if ok else 1.0, "tolerance": 0.0,
                   "passed": bool(ok), "note": detail})


def run_grillage_verification(*, terms: int = 61, fd_nx: int = 120,
                              plastic_check: bool = True) -> dict[str, object]:
    """Run the T1-T13 anchors and return a structured evidence bundle."""
    checks: list[dict[str, object]] = []

    # ---------------- T1: isotropic SSSS plate vs Timoshenko Table 30 ------
    side, thickness, load, e, nu = 1.0, 0.010, 1.0e4, 2.06e11, 0.3
    d_plate = e * thickness ** 3 / (12.0 * (1.0 - nu ** 2))
    resp = plate_patch_response(span_x=side, span_y=side, d1=d_plate, d2=d_plate,
                                d3=d_plate, pressure=load, terms=121, sample=21,
                                poisson=nu)
    _record(checks, "T1a-w_max", resp["w_max"],
            0.00406 * load * side ** 4 / d_plate, 0.03,
            "Timoshenko & Woinowsky-Krieger Table 30, alpha = 0.00406")
    _record(checks, "T1b-M_max", resp["m_x_max"], 0.0479 * load * side ** 2, 0.03,
            "Timoshenko & Woinowsky-Krieger Table 30, beta = 0.0479")

    # ---------------- T2: patch series convergence -------------------------
    coarse = plate_patch_response(span_x=side, span_y=side, d1=d_plate, d2=d_plate,
                                  d3=d_plate, pressure=load,
                                  patch=(0.3, 0.7, 0.3, 0.7), terms=31, sample=21,
                                  poisson=nu)
    fine = plate_patch_response(span_x=side, span_y=side, d1=d_plate, d2=d_plate,
                                d3=d_plate, pressure=load,
                                patch=(0.3, 0.7, 0.3, 0.7), terms=121, sample=21,
                                poisson=nu)
    _record(checks, "T2-w_convergence", coarse["w_max"], fine["w_max"], 0.005,
            "Navier patch series, terms 31 vs 121")
    _record(checks, "T2-M_convergence", coarse["m_x_max"], fine["m_x_max"], 0.02,
            "Navier patch series, terms 31 vs 121 (moment converges slower)")

    # ---------------- T3: isotropic share symmetry -------------------------
    uniform = plate_patch_response(span_x=side, span_y=side, d1=d_plate, d2=d_plate,
                                   d3=d_plate, pressure=load, terms=121,
                                   poisson=nu)
    _record(checks, "T3-fx_eq_fy", uniform["load_share_x"],
            uniform["load_share_y"], 1e-9, "isotropic plate symmetry")
    _assert(checks, "T3-share_sum",
            abs(uniform["load_share_x"] + uniform["load_share_y"]
                + uniform["load_share_torsion"] - 1.0) < 1e-9,
            "modal partition sums to one")

    # ---------------- T4: orthotropy limit ---------------------------------
    shares = {}
    for ratio in (1e2, 1e4, 1e6, 1e8):
        shares[ratio] = plate_patch_response(
            span_x=side, span_y=side, d1=d_plate * ratio, d2=d_plate,
            d3=d_plate, pressure=load, terms=61, poisson=nu)["load_share_x"]
    # the modal partition approaches 1 as (Dx/Dy)^{1/4}, not linearly, because the
    # high-n modes stay y-dominated up to n ~ (Dx/Dy)^{1/4}
    _assert(checks, "T4-orthotropy_limit",
            shares[1e8] > 0.99 and all(shares[b] > shares[a]
                                       for a, b in zip((1e2, 1e4, 1e6), (1e4, 1e6, 1e8))),
            "load_share_x vs Dx/Dy: " + ", ".join(
                f"{k:.0e}:{v:.4f}" for k, v in shares.items()))

    # ---------------- T5: beam-column amplification ------------------------
    for ratio in (0.2, 0.5):
        exact = beam_column_amplification_exact(ratio, 1.0)
        approx = beam_column_amplification_approx(ratio, 1.0)
        _assert(checks, f"T5-exact_ge_approx_P/Pe={ratio}", exact >= approx - 1e-12,
                f"exact {exact:.6f} vs 1/(1-r) {approx:.6f}")
        _record(checks, f"T5-amplification_P/Pe={ratio}", exact, approx, 0.15,
                "secant magnification vs the standard approximation (15 % gate)")

    # ---------------- T6: rigid-plastic frame collapse ---------------------
    mp, span = 1.0e6, 2.8
    _record(checks, "T6a-uniform_16Mp/L",
            frame_plastic_collapse_force(mp, span, span), 16.0 * mp / span, 1e-12,
            "uniform-load collapse of a fixed-fixed beam")
    _record(checks, "T6b-point_8Mp/L",
            frame_plastic_collapse_force(mp, span, 0.0), 8.0 * mp / span, 1e-12,
            "point-load limit of the patch formula")
    capacities = [frame_plastic_collapse_force(mp, span, c)
                  for c in (0.1, 0.5, 1.0, 2.0, span)]
    _assert(checks, "T6c-monotone_in_patch_length",
            all(b > a for a, b in zip(capacities, capacities[1:])),
            f"capacities {['%.3f' % c for c in capacities]}")

    # ---------------- T7: versus Paik-Thayamballi empirical strength -------
    try:
        from .stiffened_panel_ultimate import StiffenedPanel
    except Exception as exc:  # pragma: no cover - import guard
        StiffenedPanel = None
        t7_note = f"import failed: {exc!r}"
    else:
        t7_note = "Paik-Thayamballi stiffened-panel strength (empirical, 15 % gate)"
    t7_rows: list[dict[str, float]] = []
    if StiffenedPanel is not None:
        base = case_study_geometry()
        for spacing, thickness in ((0.45, 0.013), (0.35, 0.020), (0.50, 0.012)):
            geom = GrillageGeometry(
                frame_spacing=base.frame_spacing, bay_spacing=spacing, n_bays=6,
                n_spans=4, plate_thickness=thickness, stiffener=base.stiffener,
                frame=base.frame, young=base.young, poisson=base.poisson,
                yield_stress=base.yield_stress).validate()
            mine = stiffener_axial_capacity(geom)["panel_ultimate_stress"] / 1e6
            reference = float(StiffenedPanel(
                span=geom.frame_spacing, spacing=geom.bay_spacing,
                plate_thickness=geom.plate_thickness,
                web_height=geom.stiffener.web_height,
                web_thickness=geom.stiffener.web_thickness,
                flange_width=geom.stiffener.flange_width,
                flange_thickness=geom.stiffener.flange_thickness,
                young=geom.young, poisson=geom.poisson,
                yield_stress=geom.yield_stress).validate().panel_ultimate_stress) / 1e6
            tag = f"b/t={spacing / thickness:.1f}, beta={geom.plate_slenderness:.2f}"
            _record(checks, f"T7-panel_ultimate[{tag}]", mine, reference, 0.15, t7_note)
            t7_rows.append({"bay_spacing_m": spacing, "plate_thickness_mm": thickness * 1e3,
                            "beta": geom.plate_slenderness,
                            "sigma_u_effective_width_MPa": mine,
                            "sigma_u_paik_MPa": reference,
                            "rel_error": abs(mine - reference) / reference})

    # ---------------- T8: monotonicity in the combined loads ---------------
    geom = case_study_geometry()
    patch_l, patch_w = 0.6, 0.35
    p_axial: list[float] = []
    for eta in (0.0, 0.15, 0.30, 0.45):
        case = CombinedLoadCase(axial_stress=eta * geom.yield_stress,
                                patch_length=patch_l, patch_width=patch_w,
                                hydrostatic_pressure=5.0e4)
        p_axial.append(float(ultimate_ice_pressure(geom, case)["ice_pressure_ultimate"]))
    _assert(checks, "T8a-monotone_in_axial",
            all(b < a for a, b in zip(p_axial, p_axial[1:])),
            f"p_ult [MPa] {['%.4f' % (p / 1e6) for p in p_axial]}")
    p_hydro: list[float] = []
    for p_h in (0.0, 5.0e4, 1.0e5):
        case = CombinedLoadCase(axial_stress=0.3 * geom.yield_stress,
                                patch_length=patch_l, patch_width=patch_w,
                                hydrostatic_pressure=p_h)
        p_hydro.append(float(ultimate_ice_pressure(geom, case)["ice_pressure_ultimate"]))
    _assert(checks, "T8b-monotone_in_hydrostatic",
            all(b < a for a, b in zip(p_hydro, p_hydro[1:])),
            f"p_ult [MPa] {['%.4f' % (p / 1e6) for p in p_hydro]}")

    # ---------------- T9: load-path equilibrium ----------------------------
    worst = 0.0
    for eta in (0.0, 0.3, 0.6):
        for p_h in (0.0, 1.0e5):
            case = CombinedLoadCase(axial_stress=eta * geom.yield_stress,
                                    ice_pressure=1.5e6, patch_length=patch_l,
                                    patch_width=patch_w, hydrostatic_pressure=p_h)
            u = grillage_utilisations(geom, case)
            total = u["force_ice"] + u["force_hydrostatic"]
            carried = u["force_stiffener_system"] + u["force_frame_system"]
            worst = max(worst, abs(carried - total) / total)
    _assert(checks, "T9-load_path_equilibrium", worst < 1e-12,
            f"max relative residual {worst:.3e} over 6 combined cases")

    # ---------------- T10: beam patch moment by independent statics --------
    F, L, c = 1.0e5, 2.1, 0.7
    xs = np.linspace(0.0, L, 20001)
    q_intensity = np.where(np.abs(xs - 0.5 * L) <= 0.5 * c, F / c, 0.0)
    shear = 0.5 * F - np.concatenate(([0.0], np.cumsum(
        0.5 * (q_intensity[1:] + q_intensity[:-1]) * np.diff(xs))))
    moment = np.concatenate(([0.0], np.cumsum(
        0.5 * (shear[1:] + shear[:-1]) * np.diff(xs))))
    numeric = float(np.max(moment))
    _record(checks, "T10-centre_patch_moment", beam_moment_centre_patch(F, L, c),
            numeric, 1e-4, "closed form F(2L-c)/8 vs numeric integration of M(x)")

    # ---------------- T11: independent FD recomputation of the plate field -
    fd_rows = plate_field_fd_check(geom, nx=fd_nx, terms=terms)
    for row in fd_rows:
        _record(checks, f"T11-peak_moment_FD[{row['scenario']}]",
                float(row["peak_moment_fd_per_Pa"]),
                float(row["peak_moment_series_per_Pa"]), 0.005,
                "Navier series vs independent FD (13-point biharmonic stencil), "
                f"grid {row['fd_grid']}")
        _record(checks, f"T11-p_ult_FD[{row['scenario']}]",
                float(row["p_ult_fd_MPa"]), float(row["p_ult_module_MPa"]), 0.005,
                "module bisection vs closed-form solve on the FD moments")
        _assert(checks, f"T11-plate_binds[{row['scenario']}]",
                abs(float(row["utilisation_plate_at_limit"]) - 1.0) < 1e-3,
                "the case study is plate-field governed: "
                f"U_plate = {row['utilisation_plate_at_limit']:.6f}, "
                f"U_stiff = {row['utilisation_stiffener_at_limit']:.4f}, "
                f"U_frame = {row['utilisation_frame_at_limit']:.4f}")

    # ---------------- T12: member-governed branch, independent algebra ------
    member_rows = member_governed_case_table(terms=terms)
    for row in member_rows:
        _assert(checks, f"T12-stiffener_governs[{row['scenario']}]",
                row["governing_member"] == "longitudinal_stiffener",
                "governing member " + str(row["governing_member"])
                + f"; U_stiff = {row['utilisation_stiffener_at_limit']:.6f}"
                + f", U_plate = {row['utilisation_plate_at_limit']:.4f}"
                + f", U_frame = {row['utilisation_frame_at_limit']:.4f}")
        _record(checks, f"T12-p_ult_hand[{row['scenario']}]",
                float(row["p_ult_hand_closed_form_MPa"]),
                float(row["ice_pressure_ultimate_MPa"]), 0.005,
                "closed-form stiffener limit (own section algebra, own "
                "Perry-Robertson/JO, bisection removed) vs module")
        _assert(checks, f"T12-plate_below_member[{row['scenario']}]",
                float(row["utilisation_plate_at_limit"]) < 1.0,
                f"U_plate = {row['utilisation_plate_at_limit']:.4f} < 1 at the "
                "member-limited ultimate")

    # ---------------- T13: cross-method invariant (plastic >= elastic) ------
    cross: dict[str, object] = {}
    if plastic_check:
        try:
            from .grillage_plastic_collapse import grillage_plastic_collapse
        except Exception as exc:  # pragma: no cover - import guard
            _assert(checks, "T13-cross_method_available", False,
                    f"rigid-plastic module unavailable: {exc!r}")
        else:
            modal = member_governed_geometry()
            name, eta, p_ice, lx, ly, p_h = MEMBER_GOVERNED_SCENARIOS[1]
            case = CombinedLoadCase(
                axial_stress=eta * modal.yield_stress, ice_pressure=p_ice,
                patch_length=lx, patch_width=ly, hydrostatic_pressure=p_h,
                design_pressure=p_ice, design_provenance="scenario design pressure")
            elastic = ultimate_ice_pressure(modal, case, terms=terms)
            plastic = grillage_plastic_collapse(modal, case, elements_per_span=4,
                                                terms=terms)
            p_el = float(elastic["ice_pressure_ultimate"])
            p_pl = float(plastic["ice_pressure_grillage_plastic"])
            cross = {
                "scenario": name,
                "elastic_member_limit_MPa": p_el / 1e6,
                "rigid_plastic_grillage_MPa": p_pl / 1e6,
                "plastic_over_elastic": p_pl / p_el,
                "plastic_governing_system": plastic.get("governing_member_line"),
                "note": "the plastic limit is a redistribution upper bound on the "
                        "elastic member limit; it must not be smaller",
            }
            _assert(checks, "T13-plastic_ge_elastic", p_pl >= p_el,
                    f"rigid-plastic {p_pl / 1e6:.4f} MPa vs elastic member limit "
                    f"{p_el / 1e6:.4f} MPa (ratio {p_pl / p_el:.3f})")

    passed = all(bool(c["passed"]) for c in checks)
    return {
        "suite": "grillage_ice_ultimate",
        "terms": int(terms),
        "fd_nx": int(fd_nx),
        "checks": checks,
        "passed": passed,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "paik_comparison": t7_rows,
        "plate_field_fd": fd_rows,
        "member_governed_cases": member_rows,
        "cross_method": cross,
    }


def combined_case_table(geom: GrillageGeometry | None = None) -> list[dict[str, object]]:
    """Three combined ice-patch + lateral-pressure cases with the ultimate load.

    The acceptance item asks for >= 3 combined load cases; these are the
    documented ice-belt scenario set (light / design / severe) and each row
    carries the governing member, the member utilisations at the limit and the
    safety margin.
    """
    geom = (case_study_geometry() if geom is None else geom).validate()
    rows: list[dict[str, object]] = []
    for name, eta, p_ice, lx, ly, p_h in CASE_STUDY_SCENARIOS:
        case = CombinedLoadCase(
            axial_stress=eta * geom.yield_stress, ice_pressure=p_ice,
            patch_length=lx, patch_width=ly, hydrostatic_pressure=p_h,
            design_pressure=p_ice, design_provenance="scenario design pressure")
        u = grillage_utilisations(geom, case)
        res = ultimate_ice_pressure(geom, case)
        rows.append({
            "scenario": name,
            "sigma_x_over_yield": eta,
            "sigma_x_MPa": eta * geom.yield_stress / 1e6,
            "ice_patch_MPa": p_ice / 1e6,
            "patch_area_m2": min(lx, geom.frame_spacing) * min(ly, geom.bay_spacing),
            "hydrostatic_MPa": p_h / 1e6,
            "utilisation_plate": float(u["plate_local_bending"]),
            "utilisation_stiffener": float(u["longitudinal_stiffener"]),
            "utilisation_frame": float(u["transverse_frame"]),
            "ice_pressure_ultimate_MPa": float(res["ice_pressure_ultimate"]) / 1e6,
            "governing_member": res["governing_member"],
            "safety_margin": res["margin_vs_design"],
            "force_stiffener_system_kN": float(u["force_stiffener_system"]) / 1e3,
            "force_frame_system_kN": float(u["force_frame_system"]) / 1e3,
        })
    return rows


def main(out_dir: str | Path = "outputs/grillage_ice_ultimate") -> dict[str, object]:
    """Run the anchors + case study and write the machine-readable artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    geom = case_study_geometry()
    verification = run_grillage_verification()
    (out / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8")
    table = combined_case_table(geom)
    (out / "combined_cases.json").write_text(
        json.dumps({"geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "n_bays": geom.n_bays,
            "n_spans": geom.n_spans,
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "yield_stress_MPa": geom.yield_stress / 1e6,
        }, "cases": table}, indent=2), encoding="utf-8")
    (out / "member_governed_cases.json").write_text(
        json.dumps({"geometry": {
            "frame_spacing_m": member_governed_geometry().frame_spacing,
            "bay_spacing_m": member_governed_geometry().bay_spacing,
            "n_bays": member_governed_geometry().n_bays,
            "n_spans": member_governed_geometry().n_spans,
            "plate_thickness_mm": member_governed_geometry().plate_thickness * 1e3,
            "stiffener": "T160x6 + 50x7",
            "frame": "T550x10 + 120x14",
            "yield_stress_MPa": member_governed_geometry().yield_stress / 1e6,
        }, "cases": verification["member_governed_cases"],
           "cross_method": verification["cross_method"]}, indent=2),
        encoding="utf-8")
    (out / "plate_field_fd_check.json").write_text(
        json.dumps({"panel_m": [geom.frame_spacing, geom.bay_spacing],
                    "fd_nx": verification["fd_nx"],
                    "rows": verification["plate_field_fd"]}, indent=2),
        encoding="utf-8")
    mapping = panel_stress_ultimate_map(
        geom, design_pressure=2.0e6,
        design_provenance="documentation example (2.0 MPa at the bay area); "
                          "replace by the applicable Polar Class rule value")
    write_ultimate_map(mapping, out / "ultimate_map.json")
    return {"verification": verification, "cases": table, "map": mapping}


if __name__ == "__main__":  # pragma: no cover
    payload = main()
    print(json.dumps({"passed": payload["verification"]["passed"],
                      "n_checks": payload["verification"]["n_checks"],
                      "n_failed": payload["verification"]["n_failed"]}, indent=2))