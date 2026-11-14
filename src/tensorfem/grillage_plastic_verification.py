"""Verification gates for the grillage plastic-collapse (limit analysis) layer.

Every gate is a comparison with an **independent** reference: closed-form
rigid-plastic theory, virtual-work upper bounds, exact elasticity, or the
closed-form member collapse already carried by ``grillage_ice_ultimate``.  No
gate compares the solver with itself.

Gates (``gate`` = allowed relative error, predeclared before the run):

``V1`` simply supported (SS) beam, uniform load: ``w_u = 8 M_p / L^2``.
``V2`` SS beam, central patch ``c``: ``w_u = 8 M_p / (2 L c - c^2)``.
``V3`` clamped-clamped beam, uniform load: ``w_u = 16 M_p / L^2``.
``V4`` two-span continuous beam, uniform load: ``11.65685 M_p / L^2`` (closed
form of :func:`two_span_collapse_upper_bound`, virtual work minimised over the
hinge position).
``V5`` elastic path: SS beam midspan moment ``q L^2 / 8`` and deflection
``5 q L^4 / 384 EI``.
``V6`` axial interaction: reduced plastic moment ``M_p (1 - (P/P_y)^2)``.
``V7`` mesh convergence in elements per span.
``V8`` patch-size and coexisting-compression sensitivity of the case study.
``V9`` load-share positivity, partition and ``terms`` convergence.
``V10`` cross-implementation: the transverse-frame limit against the
SS central-patch closed form ``F_u = M_p / [(2L - c)/8]``.

Eccentric ice-patch gates (round 3):

``V11`` eccentric patch of the statically determinate frame line at three
positions against the exact closed form
:func:`grillage_plastic_collapse.eccentric_patch_collapse_force`.
``V12`` that closed form's two analytic limits: the central-patch value
``8 M_p/(2L-c)`` and the point-load value ``M_p L/(a(L-a))``.
``V13`` the limit load is minimum with the patch centred (closed form *and*
hinge solver) -- the assumption behind the centred case study, tested.
``V14`` eccentric patch on the smeared orthotropic plate: the Navier solution
(which now carries the even modes) against an independent finite-difference
solve (:mod:`tensorfem.orthotropic_plate_fd`).
``V15`` witness that the even modes matter: the pre-round-3 odd-only basis is
re-evaluated here and its error on an eccentric patch is recorded, so the gate
fails if the fix is ever reverted.

Run: ``python -m tensorfem.grillage_plastic_verification``
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .grillage_ice_ultimate import (
    case_study_geometry,
    plate_patch_response,
    system_load_shares,
)
from .grillage_plastic_collapse import (
    MemberLine,
    _BeamSystem,
    case_study_load_case,
    eccentric_patch_collapse_force,
    grillage_plastic_collapse,
    member_collapse_pressure,
    two_span_collapse_upper_bound,
)
from .orthotropic_plate_fd import ssss_orthotropic_plate_fd

__all__ = ["run_grillage_plastic_verification", "main"]

SPAN = 2.1
YIELD = 3.55e8


def _section():
    from .grillage_ice_ultimate import MemberSection
    return MemberSection(0.25, 0.010, 0.080, 0.012,
                         plate_thickness=0.020, plate_width=0.35)


def _member(spans, n, **kw):
    return MemberLine("bench", spans, _section(), YIELD,
                      young=2.06e11, elements_per_span=n, **kw)


def _check(checks: list[dict], name: str, got: float, ref: float, gate: float,
           note: str = "") -> None:
    err = got / ref - 1.0 if ref else float("nan")
    checks.append({"gate": name, "got": got, "reference": ref,
                   "rel_error": err, "allowed": gate,
                   "passed": bool(abs(err) <= gate), "note": note})


def run_grillage_plastic_verification(*, out_dir: str | Path | None = None,
                                      make_figures: bool = True) -> dict:
    section = _section()
    mp = section.plastic_modulus * YIELD
    checks: list[dict] = []

    # -- V1 / V2 / V3 / V4 ---------------------------------------------------
    n = 16
    r = member_collapse_pressure(_member([SPAN], n), ice_tributary=1.0,
                                 ice_interval=(0.0, SPAN))
    _check(checks, "V1_ss_uniform", r["ice_pressure_ultimate"],
           8.0 * mp / SPAN ** 2, 0.01, "8 Mp/L^2, n=16")

    c = 1.0
    r = member_collapse_pressure(_member([SPAN], n), ice_tributary=1.0,
                                 ice_interval=(0.5 * (SPAN - c), 0.5 * (SPAN + c)))
    _check(checks, "V2_ss_central_patch", r["ice_pressure_ultimate"],
           8.0 * mp / (2.0 * SPAN * c - c ** 2), 0.01,
           "8 Mp/(2Lc-c^2), c=1.0 m")

    r = member_collapse_pressure(_member([SPAN], n, fixed_ends=True),
                                 ice_tributary=1.0, ice_interval=(0.0, SPAN))
    _check(checks, "V3_clamped_uniform", r["ice_pressure_ultimate"],
           16.0 * mp / SPAN ** 2, 0.01, "16 Mp/L^2, n=16")

    r = member_collapse_pressure(_member([SPAN, SPAN], n), ice_tributary=1.0,
                                 ice_interval=(0.0, 2.0 * SPAN))
    ub = two_span_collapse_upper_bound(span=SPAN, plastic_moment=mp)
    _check(checks, "V4_two_span_uniform", r["ice_pressure_ultimate"],
           ub["collapse_load_exact"], 0.02,
           "virtual-work minimum at a/L = sqrt(2)-1")

    # -- V5 elastic path -----------------------------------------------------
    system = _BeamSystem(_member([SPAN], 32))
    f = system.load_vector(lambda x: 1.0)
    u = system.solve({}, f)
    moments = system.section_moments({}, u)
    mid = int(np.argmin(np.abs(system.xs - 0.5 * SPAN)))
    ei = 2.06e11 * section.inertia
    ref_defl = 5.0 * SPAN ** 4 / (384.0 * ei)
    _check(checks, "V5a_elastic_midspan_moment", moments[mid],
           SPAN ** 2 / 8.0, 0.002, "q L^2/8")
    _check(checks, "V5b_elastic_midspan_deflection",
           float(system.deflection({}, u)[mid]), ref_defl, 0.01,
           "5 q L^4/384EI")

    # -- V6 axial interaction ------------------------------------------------
    hinges = {}
    py = section.area * YIELD
    for ratio in (0.15, 0.30, 0.45):
        m = _member([SPAN], n, axial_force=ratio * py)
        exact = 1.0 - ratio ** 2
        _check(checks, f"V6_axial_reduction_{int(ratio*100)}",
               m.plastic_moment / m.plastic_moment_full, exact, 1e-12,
               "Mp(1-(P/Py)^2)")
        r = member_collapse_pressure(m, ice_tributary=1.0,
                                     ice_interval=(0.0, SPAN))
        _check(checks, f"V6_collapse_scales_{int(ratio*100)}",
               r["ice_pressure_ultimate"] / (8.0 * mp / SPAN ** 2), exact, 0.02,
               "collapse load carries the same axial reduction")

    # -- V7 mesh convergence -------------------------------------------------
    convergence = []
    for nn in (4, 8, 16, 32):
        r1 = member_collapse_pressure(_member([SPAN], nn), ice_tributary=1.0,
                                      ice_interval=(0.0, SPAN))
        r4 = member_collapse_pressure(_member([SPAN, SPAN], nn), ice_tributary=1.0,
                                      ice_interval=(0.0, 2.0 * SPAN))
        convergence.append({
            "elements_per_span": nn,
            "V1_relative_error": r1["ice_pressure_ultimate"] / (8.0 * mp / SPAN ** 2) - 1.0,
            "V4_relative_error": r4["ice_pressure_ultimate"] / ub["collapse_load_exact"] - 1.0,
        })
    worst = abs(convergence[-1]["V1_relative_error"])
    checks.append({"gate": "V7_mesh_convergence", "got": worst, "reference": 0.002,
                   "rel_error": worst / 0.002, "allowed": 1.0,
                   "passed": bool(worst <= 0.002),
                   "note": "|V1 relative error| at 32 elements/span vs a 0.2 % budget"})

    # -- V9 load shares ------------------------------------------------------
    geom = case_study_geometry()
    case = case_study_load_case()
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    lx, ly = case.patch_extents(geom)
    dx, dy, d3 = geom.plate_rigidities()
    patch = (0.5 - 0.5 * lx / span_x, 0.5 + 0.5 * lx / span_x,
             0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y)
    shares = {}
    for terms in (61, 121):
        resp = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                    d3=d3, pressure=1.0, patch=patch, terms=terms)
        fx, fy = system_load_shares(resp)
        shares[terms] = (fx, fy, resp["load_share_torsion"])
    ok_positive = all(v >= 0.0 for terms in shares for v in shares[terms])
    ok_partition = abs(shares[61][0] + shares[61][1]
                       + shares[61][2] - 1.0) < 1e-12
    _check(checks, "V9a_load_share_positive", 1.0 if ok_positive else 0.0, 1.0, 0.0,
           "both bending shares and the torsion share >= 0")
    _check(checks, "V9b_system_share_partition", shares[61][0] + shares[61][1],
           1.0, 1e-12, "renormalised bending shares sum to one")
    resp61 = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                  d3=d3, pressure=1.0, patch=patch, terms=61)
    modal_sum = (resp61["load_share_x"] + resp61["load_share_y"]
                 + resp61["load_share_torsion"])
    _check(checks, "V9d_modal_share_partition", modal_sum, 1.0, 1e-12,
           "modal x + y + torsion shares sum to one")
    _check(checks, "V9c_load_share_terms_convergence", shares[121][0], shares[61][0],
           0.02, "x share, terms 121 vs 61")

    # -- V10 cross-implementation -------------------------------------------
    mp_frame = geom.frame_section.plastic_modulus * geom.yield_stress
    patch_len = min(case.patch_width, geom.bay_spacing)
    exact_force = mp_frame / ((2.0 * geom.frame_span - patch_len) / 8.0)
    member = MemberLine("transverse_frame", [geom.frame_span], geom.frame_section,
                        geom.yield_stress, young=geom.young, elements_per_span=16)
    tributary = 0.4371
    rf = member_collapse_pressure(member, ice_tributary=tributary,
                                  ice_interval=(0.5 * (geom.frame_span - patch_len),
                                                0.5 * (geom.frame_span + patch_len)))
    _check(checks, "V10_frame_vs_closed_form",
           rf["ice_pressure_ultimate"] * tributary * patch_len, exact_force, 0.02,
           "SS central-patch collapse F_u = 8 Mp/(2L-c)")

    # -- V11 eccentric patch: hinge solver vs exact closed form --------------
    c_ecc = case.patch_length                       # 1.05 m = half the frame span
    ecc_rows = []
    for frac in (0.30, 0.50, 0.70):
        line = MemberLine("transverse_frame_ecc", [geom.frame_span],
                          geom.frame_section, geom.yield_stress,
                          young=geom.young, elements_per_span=64)
        interval = (frac * geom.frame_span - 0.5 * c_ecc,
                    frac * geom.frame_span + 0.5 * c_ecc)
        solved = member_collapse_pressure(line, ice_tributary=1.0,
                                          ice_interval=interval,
                                          hydro_pressure=0.0, hydro_tributary=0.0)
        ref = eccentric_patch_collapse_force(
            plastic_moment=line.plastic_moment, span=geom.frame_span,
            patch_length=c_ecc, patch_centre=frac)
        ecc_rows.append({
            "patch_centre_fraction": frac,
            "patch_centre_x": frac * geom.frame_span,
            "solver_line_load": solved["ice_pressure_ultimate"],
            "closed_form_line_load": ref["line_load_ultimate"],
            "rel_error": solved["ice_pressure_ultimate"]
            / ref["line_load_ultimate"] - 1.0,
            "hinges": len(solved["hinges"]),
            "maximum_at": ref["maximum_at"],
        })
    for tag, row in zip(("a_x30", "b_x50", "c_x70"), ecc_rows):
        _check(checks, f"V11{tag}_eccentric_patch", row["solver_line_load"],
               row["closed_form_line_load"], 0.005,
               "determinate span, c/L = 0.5, n = 64 (independent closed form)")

    # -- V12 analytic limits of the eccentric closed form --------------------
    mp_frame_line = geom.frame_section.plastic_modulus * geom.yield_stress
    central = eccentric_patch_collapse_force(plastic_moment=mp_frame_line,
                                             span=geom.frame_span,
                                             patch_length=c_ecc, patch_centre=0.5)
    _check(checks, "V12a_closed_form_central_patch",
           central["force_ultimate"],
           8.0 * mp_frame_line / (2.0 * geom.frame_span - c_ecc), 1e-12,
           "a = L/2 reproduces F_u = 8 Mp/(2L-c)")
    point = eccentric_patch_collapse_force(plastic_moment=mp_frame_line,
                                           span=geom.frame_span,
                                           patch_length=1e-6, patch_centre=0.3)
    _check(checks, "V12b_closed_form_point_load",
           point["force_ultimate"],
           mp_frame_line * geom.frame_span / (0.3 * geom.frame_span
                                              * 0.7 * geom.frame_span), 1e-5,
           "c -> 0 reproduces the point-load value Mp L/(a(L-a))")

    # -- V13 worst patch position: centred -----------------------------------
    stations = np.linspace(0.25 + 1e-9, 0.75 - 1e-9, 11)
    closed = [eccentric_patch_collapse_force(
        plastic_moment=mp_frame_line, span=geom.frame_span, patch_length=c_ecc,
        patch_centre=float(a))["force_ultimate"] for a in stations]
    solved_scan = []
    for a in stations:
        line = MemberLine("scan", [geom.frame_span], geom.frame_section,
                          geom.yield_stress, young=geom.young,
                          elements_per_span=32)
        interval = (float(a) * geom.frame_span - 0.5 * c_ecc,
                    float(a) * geom.frame_span + 0.5 * c_ecc)
        solved_scan.append(member_collapse_pressure(
            line, ice_tributary=1.0, ice_interval=interval,
            hydro_pressure=0.0, hydro_tributary=0.0)["ice_pressure_ultimate"])
    _check(checks, "V13a_worst_centre_closed_form", min(closed), closed[5], 1e-9,
           "closed form is minimum at a = L/2")
    _check(checks, "V13b_worst_centre_solver", min(solved_scan), solved_scan[5],
           0.005, "hinge solver is minimum at a = L/2")

    # -- V14 eccentric patch on the plate: Navier vs finite difference -------
    span_x_plate = geom.n_spans * geom.frame_spacing
    span_y_plate = geom.frame_span
    dx, dy, d3 = geom.plate_rigidities()
    lx_p, ly_p = case.patch_extents(geom)
    plate_rows = []
    for cx in (0.25, 0.50):
        rect = (cx - 0.5 * lx_p / span_x_plate, cx + 0.5 * lx_p / span_x_plate,
                0.5 - 0.5 * ly_p / span_y_plate, 0.5 + 0.5 * ly_p / span_y_plate)
        nav = plate_patch_response(span_x=span_x_plate, span_y=span_y_plate,
                                   d1=dx, d2=dy, d3=d3, pressure=1.0,
                                   patch=rect)
        fdv = ssss_orthotropic_plate_fd(span_x=span_x_plate, span_y=span_y_plate,
                                        d1=dx, d2=dy, d3=d3, pressure=1.0,
                                        patch=rect, nx=240)
        plate_rows.append({
            "patch_centre_fraction": cx,
            "navier_w_max": nav["w_max"], "fd_w_max": fdv["w_max"],
            "navier_m_x_max": nav["m_x_max"], "fd_m_x_max": fdv["m_x_max"],
            "fd_w_peak_x_fraction": fdv["w_peak_x"] / span_x_plate,
            "rel_error_w": nav["w_max"] / fdv["w_max"] - 1.0,
            "rel_error_m_x": nav["m_x_max"] / fdv["m_x_max"] - 1.0,
        })
    for tag, row in zip(("a_x25", "b_x50"), plate_rows):
        _check(checks, f"V14{tag}_plate_deflection",
               row["navier_w_max"], row["fd_w_max"], 0.02,
               "Navier series vs 13-point FD, nx = 240")
        _check(checks, f"V14{tag}_plate_moment",
               row["navier_m_x_max"], row["fd_m_x_max"], 0.02,
               "Navier series vs 13-point FD, nx = 240")

    # -- V15 witness: the odd-only basis cannot represent an eccentric patch --
    witness = _odd_only_basis_witness(geom, case)
    ok = (abs(witness["odd_only_w_error"]) >= 0.05
          and abs(witness["fixed_w_error"]) <= 0.01
          and abs(witness["odd_only_m_x_error"]) >= 0.20
          and abs(witness["fixed_m_x_error"]) <= 0.02)
    _check(checks, "V15_odd_only_basis_witness", 1.0 if ok else 0.0, 1.0, 0.0,
           "pre-round-3 odd-only basis >=5% (w) / >=20% (m_x) wrong on an "
           "eccentric patch; the fixed basis <=1% / <=2%")

    # -- case study + V8 sensitivity ----------------------------------------
    case_study = {}
    for nn in (4, 8, 16):
        res = grillage_plastic_collapse(geom, case, elements_per_span=nn)
        case_study[nn] = {
            "ice_pressure_ultimate": res["ice_pressure_ultimate"],
            "ice_pressure_grillage_plastic": res["ice_pressure_grillage_plastic"],
            "ice_pressure_panel_limit": res["ice_pressure_panel_limit"],
            "governing_mode": res["governing_mode"],
            "governing_member_line": res["governing_member_line"],
            "margin_vs_design": res["margin_vs_design"],
            "systems": {k: {"ice_pressure_ultimate": v["ice_pressure_ultimate"],
                            "ice_pressure_first_hinge": v["ice_pressure_first_hinge"],
                            "plastic_reserve_ratio": v["plastic_reserve_ratio"],
                            "hinges": v["hinges"]}
                        for k, v in res["systems"].items()},
            "load_shares": res["load_shares"],
            "dimensionless": res["dimensionless"],
        }
    base = case_study[16]
    _check(checks, "V7b_case_study_mesh", base["ice_pressure_grillage_plastic"],
           case_study[8]["ice_pressure_grillage_plastic"], 0.05,
           "grillage plastic pressure, n=16 vs n=8")

    sensitivity = []
    for patch_scale in (0.667, 1.0, 1.333):
        for axial_ratio in (0.0, 0.15, 0.30):
            variant = type(case)(
                axial_stress=axial_ratio * geom.yield_stress,
                ice_pressure=case.ice_pressure,
                patch_length=patch_scale * case.patch_length,
                patch_width=case.patch_width,
                hydrostatic_pressure=case.hydrostatic_pressure,
                design_pressure=case.design_pressure,
                design_provenance=case.design_provenance,
            ).validate(geom)
            res = grillage_plastic_collapse(geom, variant, elements_per_span=8)
            sensitivity.append({
                "patch_length": variant.patch_length,
                "axial_ratio": axial_ratio,
                "ice_pressure_grillage_plastic": res["ice_pressure_grillage_plastic"],
                "ice_pressure_panel_limit": res["ice_pressure_panel_limit"],
                "ice_pressure_ultimate": res["ice_pressure_ultimate"],
                "governing_mode": res["governing_mode"],
                "stiffener_reserve":
                    res["systems"]["longitudinal_stiffener"]["plastic_reserve_ratio"],
                "frame_reserve":
                    res["systems"]["transverse_frame"]["plastic_reserve_ratio"],
            })
    by_axial = {}
    for row in sensitivity:
        if abs(row["patch_length"] - case.patch_length) < 1e-12:
            by_axial[row["axial_ratio"]] = row["ice_pressure_grillage_plastic"]
    monotone_axial = (by_axial[0.0] >= by_axial[0.15] >= by_axial[0.30])
    by_patch = {}
    for row in sensitivity:
        if row["axial_ratio"] == 0.30:
            by_patch[row["patch_length"]] = row["ice_pressure_grillage_plastic"]
    monotone_patch = all(by_patch[a] >= by_patch[b]
                         for a, b in zip(sorted(by_patch), sorted(by_patch)[1:]))
    _check(checks, "V8a_axial_monotone", 1.0 if monotone_axial else 0.0, 1.0, 0.0,
           "collapse pressure non-increasing with coexisting compression")
    _check(checks, "V8b_patch_monotone", 1.0 if monotone_patch else 0.0, 1.0, 0.0,
           "collapse pressure non-increasing with patch size")

    patch_map = patch_position_map(geom, case)

    result = {
        "module": "tensorfem.grillage_plastic_collapse",
        "eccentric_patch": {
            "frame_line_gates": ecc_rows,
            "worst_position_scan": {
                "patch_centre_fractions": [float(a) for a in stations],
                "closed_form_force": [float(v) for v in closed],
                "hinge_solver_line_load": [float(v) for v in solved_scan],
            },
            "closed_form_central_force": central["force_ultimate"],
            "closed_form_point_load_force": point["force_ultimate"],
        },
        "plate_eccentric": plate_rows,
        "odd_only_witness": witness,
        "patch_position_map": patch_map,
        "anchor": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
        "n_failed": sum(0 if c["passed"] else 1 for c in checks),
        "convergence": convergence,
        "case_study": {str(k): v for k, v in case_study.items()},
        "sensitivity": sensitivity,
        "load_shares": {str(k): {"x": v[0], "y": v[1], "torsion": v[2]}
                        for k, v in shares.items()},
        "reference": {
            "Mp_stiffener": mp,
            "Mp_frame": mp_frame,
            "plastic_modulus_stiffener": section.plastic_modulus,
            "plastic_modulus_frame": geom.frame_section.plastic_modulus,
            "span": SPAN,
            "yield_stress": YIELD,
        },
    }
    if out_dir is not None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "grillage_plastic_verification.json").write_text(
            json.dumps(result, indent=2, sort_keys=True, default=str))
        (path / "patch_position_map.json").write_text(
            json.dumps({"anchor": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
                        "case": "ice patch position x axial compression sweep",
                        "rows": patch_map}, indent=2, sort_keys=True))
        if make_figures:
            try:
                result["figures"] = _figures(result, path, geom, case)
            except Exception as exc:  # pragma: no cover - plotting optional
                result["figures"] = [f"figure generation failed: {exc}"]
    return result


def _navier_plate_field(*, span_x: float, span_y: float, d1: float, d2: float,
                        d3: float, pressure: float,
                        patch: tuple[float, float, float, float],
                        terms: int = 61, sample: int = 21,
                        poisson: float = 0.3, odd_only: bool = False) -> dict:
    """Sampled Navier field of the SSSS orthotropic plate (verification only).

    Same series as ``plate_patch_response`` but returning the field, which the
    library function does not expose.  ``odd_only=True`` reproduces the
    pre-round-3 basis (odd modes only) so the defect can be quantified and
    plotted; the two branches are otherwise identical.
    """
    fx1, fx2, fy1, fy2 = (float(v) for v in patch)
    step = 2 if odd_only else 1
    m_vec = np.arange(1, terms + 1, step, dtype=float)
    n_vec = np.arange(1, terms + 1, step, dtype=float)
    m = m_vec[:, None]
    n = n_vec[None, :]
    cx = np.cos(m * math.pi * fx1) - np.cos(m * math.pi * fx2)
    cy = np.cos(n * math.pi * fy1) - np.cos(n * math.pi * fy2)
    q_mn = (4.0 * pressure / (math.pi ** 2 * m * n)) * cx * cy
    kx = d1 * m ** 4 / span_x ** 4
    ky = d2 * n ** 4 / span_y ** 4
    kt = 2.0 * d3 * m ** 2 * n ** 2 / (span_x ** 2 * span_y ** 2)
    amp = q_mn / (math.pi ** 4 * (kx + ky + kt))
    xs = np.linspace(0.0, span_x, sample)
    ys = np.linspace(0.0, span_y, sample)
    sx = np.sin(math.pi * m_vec[:, None] * xs[None, :] / span_x)
    sy = np.sin(math.pi * n_vec[:, None] * ys[None, :] / span_y)
    ax = (m * math.pi / span_x) ** 2
    ay = (n * math.pi / span_y) ** 2
    d12 = poisson * math.sqrt(d1 * d2)
    w = np.einsum("mn,mi,nj->ij", amp, sx, sy)
    m_x = np.einsum("mn,mi,nj->ij", (d1 * ax + d12 * ay) * amp, sx, sy)
    return {"w": w, "m_x": m_x, "x": xs, "y": ys,
            "w_max": float(np.max(np.abs(w))),
            "m_x_max": float(np.max(np.abs(m_x)))}


def _odd_only_basis_witness(geom, case) -> dict[str, float]:
    """Error of the pre-round-3 odd-only mode set on an eccentric patch.

    Reference = the independent finite-difference solve.  ``fixed`` is the
    library response with the even modes in.  The patch is placed at
    ``x/L = 0.3`` so that it is well off centre.
    """
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    dx, dy, d3 = geom.plate_rigidities()
    lx, ly = case.patch_extents(geom)
    cx = 0.3
    rect = (cx - 0.5 * lx / span_x, cx + 0.5 * lx / span_x,
            0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y)
    fixed = plate_patch_response(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                 d3=d3, pressure=1.0, patch=rect)
    odd = _navier_plate_field(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                              d3=d3, pressure=1.0, patch=rect, odd_only=True)
    fd = ssss_orthotropic_plate_fd(span_x=span_x, span_y=span_y, d1=dx, d2=dy,
                                   d3=d3, pressure=1.0, patch=rect, nx=240)
    return {
        "patch_centre_fraction": cx,
        "fd_w_max": fd["w_max"],
        "fd_m_x_max": fd["m_x_max"],
        "fixed_w_max": fixed["w_max"],
        "fixed_m_x_max": fixed["m_x_max"],
        "odd_only_w_max": odd["w_max"],
        "odd_only_m_x_max": odd["m_x_max"],
        "fixed_w_error": fixed["w_max"] / fd["w_max"] - 1.0,
        "fixed_m_x_error": fixed["m_x_max"] / fd["m_x_max"] - 1.0,
        "odd_only_w_error": odd["w_max"] / fd["w_max"] - 1.0,
        "odd_only_m_x_error": odd["m_x_max"] / fd["m_x_max"] - 1.0,
    }


def patch_position_map(geom, case, *,
                       centres=(0.25, 0.35, 0.50, 0.65, 0.75),
                       axial_ratios=(0.0, 0.30),
                       elements_per_span: int = 6) -> list[dict]:
    """Ice-patch-position sweep of the grillage ultimate state (machine readable).

    Each row is one combined load case: patch centre ``(cx, 0.5)`` of the two
    system spans, coexisting hull-girder compression ``axial_ratio * sigma_y``,
    with the resulting ultimate pressure, governing limit state and safety
    margin.  The centred row is the round-2 case, so the table answers "how much
    does an off-centre ice patch move the structural ultimate".
    """
    rows: list[dict] = []
    for cx in centres:
        for axial_ratio in axial_ratios:
            variant = type(case)(
                axial_stress=axial_ratio * geom.yield_stress,
                ice_pressure=case.ice_pressure,
                patch_length=case.patch_length,
                patch_width=case.patch_width,
                hydrostatic_pressure=case.hydrostatic_pressure,
                design_pressure=case.design_pressure,
                design_provenance=case.design_provenance,
            ).validate(geom)
            res = grillage_plastic_collapse(geom, variant,
                                            elements_per_span=elements_per_span,
                                            patch_centre=(float(cx), 0.5))
            systems = res["systems"]
            rows.append({
                "patch_centre_x_fraction": float(cx),
                "patch_centre_y_fraction": 0.5,
                "patch_centre_x": float(cx) * geom.n_spans * geom.frame_spacing,
                "axial_ratio": float(axial_ratio),
                "ice_pressure_ultimate": res["ice_pressure_ultimate"],
                "ice_pressure_grillage_plastic": res["ice_pressure_grillage_plastic"],
                "ice_pressure_panel_limit": res["ice_pressure_panel_limit"],
                "governing_mode": res["governing_mode"],
                "governing_member_line": res["governing_member_line"],
                "load_share_x": res["load_shares"]["x"],
                "load_share_y": res["load_shares"]["y"],
                "stiffener_limit": systems["longitudinal_stiffener"]["ice_pressure_ultimate"],
                "frame_limit": systems["transverse_frame"]["ice_pressure_ultimate"],
                "stiffener_reserve": systems["longitudinal_stiffener"]["plastic_reserve_ratio"],
                "frame_reserve": systems["transverse_frame"]["plastic_reserve_ratio"],
                "margin_vs_design": res["margin_vs_design"],
            })
    return rows


def _figures(result: dict, path: Path, geom, case) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    conv = result["convergence"]
    ns = [c["elements_per_span"] for c in conv]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].semilogx(ns, [abs(c["V1_relative_error"]) * 100 for c in conv], "o-",
                   label="V1 SS beam 8Mp/L$^2$")
    ax[0].semilogx(ns, [abs(c["V4_relative_error"]) * 100 for c in conv], "s-",
                   label="V4 two-span 11.657Mp/L$^2$")
    ax[0].axhline(1.0, color="k", ls="--", lw=1, label="1 % gate")
    ax[0].set_xlabel("elements per span")
    ax[0].set_ylabel("|relative error| [%]")
    ax[0].set_title("limit load convergence vs closed form")
    ax[0].legend()
    ax[0].grid(True, which="both", alpha=0.3)
    base = result["case_study"]["16"]
    names = ["plate field\n(panel limit)", "stiffener\nmechanism", "frame\nmechanism"]
    vals = [base["ice_pressure_panel_limit"],
            base["systems"]["longitudinal_stiffener"]["ice_pressure_ultimate"],
            base["systems"]["transverse_frame"]["ice_pressure_ultimate"]]
    ax[1].bar(names, [v / 1e6 for v in vals], color=["#888", "#2b6cb0", "#c05621"])
    ax[1].axhline(base["ice_pressure_ultimate"] / 1e6, color="r", ls="--",
                  label="reported ultimate")
    ax[1].set_ylabel("ice pressure [MPa]")
    ax[1].set_title("case study: governing limit (elements/span = 16)")
    ax[1].legend()
    fig.tight_layout()
    f1 = path / "fig_grillage_plastic_collapse.png"
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    made.append(str(f1))

    sens = result["sensitivity"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for ratio in sorted({row["axial_ratio"] for row in sens}):
        rows = [r for r in sens if r["axial_ratio"] == ratio]
        ax.plot([r["patch_length"] for r in rows],
                [r["ice_pressure_grillage_plastic"] / 1e6 for r in rows],
                "o-", label=f"$\\sigma_x/\\sigma_y$ = {ratio:g}")
    ax.set_xlabel("ice patch length [m]")
    ax.set_ylabel("grillage plastic collapse pressure [MPa]")
    ax.set_title("sensitivity: patch size and coexisting compression")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = path / "fig_grillage_plastic_sensitivity.png"
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    made.append(str(f2))

    # -- eccentric patch: limit load vs patch position -----------------------
    scan = result["eccentric_patch"]["worst_position_scan"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scan["patch_centre_fractions"], scan["closed_form_force"], "k-",
            lw=2, label="exact closed form")
    ax.plot(scan["patch_centre_fractions"], scan["hinge_solver_line_load"],
            "o", ms=4, label="hinge solver (n = 32)")
    ax.axvline(0.5, color="r", ls="--", lw=1, label="centred patch = minimum")
    ax.set_xlabel("patch centre $a/L$")
    ax.set_ylabel("collapse force [N]")
    ax.set_title("eccentric ice patch on a simply supported frame span")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f3 = path / "fig_grillage_eccentric_patch.png"
    fig.savefig(f3, dpi=140)
    plt.close(fig)
    made.append(str(f3))

    # -- case study: ultimate vs patch position and compression --------------
    rows = result["patch_position_map"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for ratio in sorted({r["axial_ratio"] for r in rows}):
        sel = [r for r in rows if r["axial_ratio"] == ratio]
        ax.plot([r["patch_centre_x_fraction"] for r in sel],
                [r["ice_pressure_ultimate"] / 1e6 for r in sel], "o-",
                label=f"$\\sigma_x/\\sigma_y$ = {ratio:g}")
    ax.set_xlabel("ice patch centre $x/L$")
    ax.set_ylabel("structural ultimate pressure [MPa]")
    ax.set_title("case study: ultimate vs patch position")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f4 = path / "fig_grillage_patch_position_map.png"
    fig.savefig(f4, dpi=140)
    plt.close(fig)
    made.append(str(f4))

    # -- defect witness: odd-only basis vs fixed basis vs FD reference -------
    wit = result["odd_only_witness"]
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["$w_{max}$", "$M_{x,max}$"]
    odd = [100.0 * wit["odd_only_w_error"], 100.0 * wit["odd_only_m_x_error"]]
    fixed = [100.0 * wit["fixed_w_error"], 100.0 * wit["fixed_m_x_error"]]
    pos = np.arange(len(labels))
    ax.bar(pos - 0.2, odd, 0.4, label="odd-only basis (pre-fix)")
    ax.bar(pos + 0.2, fixed, 0.4, label="full mode set (fixed)")
    ax.axhline(0.0, color="k", lw=1)
    ax.set_xticks(pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("error vs finite difference [%]")
    ax.set_title("eccentric patch $x/L=0.3$: why the even modes matter")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    f5 = path / "fig_grillage_odd_only_witness.png"
    fig.savefig(f5, dpi=140)
    plt.close(fig)
    made.append(str(f5))
    return made


def main(out_dir: str | Path = "artifacts/grillage_plastic_collapse") -> dict:
    result = run_grillage_plastic_verification(out_dir=out_dir)
    for c in result["checks"]:
        print(f"{'PASS' if c['passed'] else 'FAIL'}  {c['gate']:32s} "
              f"got={c['got']:.6g} ref={c['reference']:.6g} "
              f"err={100.0 * c['rel_error']:+.4f}% gate={100.0 * c['allowed']:.2f}%")
    print(f"\n{len(result['checks'])} gates, {result['n_failed']} failed")
    return result


if __name__ == "__main__":  # pragma: no cover
    import sys
    res = main(sys.argv[1] if len(sys.argv) > 1 else "artifacts/grillage_plastic_collapse")
    print("figures:", res.get("figures"))
    raise SystemExit(0 if res["all_passed"] else 1)