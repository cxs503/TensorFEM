"""Evidence run: stiffener local instability (web buckling + tripping) as limit
states of the ice-patch + lateral-pressure grillage.

Anchor: GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE (declared limitation: "stiffener
web local buckling / tripping and frame tripping are not modelled").

Everything below is computed; nothing is transcribed.  Gates T14a-T14g compare
the module against *independent* oracles (closed-form torsional buckling, a
Hermite beam finite element for the plate rotational spring, closed-form web
strip buckling, monotonicity, and the mode-summation check).  The application
section then answers the question the grillage chain could not answer before:
*how much plate rotational restraint must remain for the column state to be the
governing stiffener limit state?*

Usage::

    PYTHONPATH=src python benchmarks/grillage_local_instability.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from tensorfem.grillage_ice_ultimate import case_study_geometry, stiffener_axial_capacity
from tensorfem.stiffener_tripping import (
    TeeTorsionSection,
    plate_buckling_coefficient,
    plate_rotational_restraint,
    stiffener_instability_check,
    tripping_stress,
    web_local_buckling_stress,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "grillage_local_instability"
FIG = ROOT / "docs" / "assets" / "grillage_local_instability"


# --------------------------------------------------------------------------- #
# independent oracles
# --------------------------------------------------------------------------- #
def _hermite_matrices(length: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bending stiffness, axial-geometric and consistent-mass 4x4 Hermite blocks.

    ``B``  multiplies ``EI`` for ``int EI w''^2``,
    ``Gc`` multiplies ``c``  for ``int c w'^2``,
    ``M``  multiplies ``k``  for ``int k w^2``.
    """
    h = float(length)
    b = EI = np.array([
        [12.0, 6 * h, -12.0, 6 * h],
        [6 * h, 4 * h * h, -6 * h, 2 * h * h],
        [-12.0, -6 * h, 12.0, -6 * h],
        [6 * h, 2 * h * h, -6 * h, 4 * h * h],
    ]) / h ** 3
    g = np.array([
        [36.0, 3 * h, -36.0, 3 * h],
        [3 * h, 4 * h * h, -3 * h, -h * h],
        [-36.0, -3 * h, 36.0, -3 * h],
        [3 * h, -h * h, -3 * h, 4 * h * h],
    ]) / (30.0 * h)
    m = np.array([
        [156.0, 22 * h, 54.0, -13 * h],
        [22 * h, 4 * h * h, 13 * h, -3 * h * h],
        [54.0, 13 * h, 156.0, -22 * h],
        [-13 * h, -3 * h * h, -22 * h, 4 * h * h],
    ]) * h / 420.0
    return b, g, m


def _assemble(n_elements: int, length: float, blocks: list[np.ndarray],
              offsets: list[int], ndof: int | None = None) -> np.ndarray:
    """Assemble 4x4 Hermite blocks; ``offsets`` positions each field in the DOF vector."""
    n_nodes = n_elements + 1
    if ndof is None:
        ndof = 2 * n_nodes * len(blocks)
    total = np.zeros((ndof, ndof))
    for f, (block, off) in enumerate(zip(blocks, offsets)):
        for e in range(n_elements):
            dofs = [off + 2 * e, off + 2 * e + 1, off + 2 * e + 2, off + 2 * e + 3]
            for i in range(4):
                for j in range(4):
                    total[dofs[i], dofs[j]] += block[i, j]
    return total


def _assemble_mixed(n_elements: int, specs: list[tuple[np.ndarray, int, int]],
                    ndof: int) -> np.ndarray:
    """Assemble 4x4 Hermite blocks at (row offset, column offset) positions."""
    total = np.zeros((ndof, ndof))
    for block, row_off, col_off in specs:
        for e in range(n_elements):
            rows = [row_off + 2 * e, row_off + 2 * e + 1,
                    row_off + 2 * e + 2, row_off + 2 * e + 3]
            cols = [col_off + 2 * e, col_off + 2 * e + 1,
                    col_off + 2 * e + 2, col_off + 2 * e + 3]
            for i in range(4):
                for j in range(4):
                    total[rows[i], cols[j]] += block[i, j]
    return total


def beam_rotational_spring_fem(*, span: float, rigidity: float,
                               far_edge: str = "clamped",
                               elements: int = 2) -> float:
    """``M(theta)/theta`` of a plate strip by an independent Hermite beam FEM.

    Node 0 is the stiffener line with a prescribed unit rotation and zero
    deflection; the far node is clamped (``far_edge='clamped'``) or simply
    supported / rotation free (``far_edge='pinned'``).  The end moment follows
    from the recovered curvature ``w''(0)`` of the first element, so this oracle
    shares nothing with the closed-form ``c D / (b/2)`` used by the module.
    """
    n = int(elements)
    h = span / n
    b, _, _ = _hermite_matrices(h)
    k = _assemble(n, span, [rigidity * b], [0])
    ndof = 2 * (n + 1)
    prescribed = {0: 0.0, 1: 1.0, 2 * n: 0.0}
    if far_edge == "clamped":
        prescribed[2 * n + 1] = 0.0
    elif far_edge != "pinned":
        raise ValueError("far_edge must be 'clamped' or 'pinned'")
    free = [d for d in range(ndof) if d not in prescribed]
    fixed = sorted(prescribed)
    u = np.zeros(ndof)
    for d, v in prescribed.items():
        u[d] = v
    rhs = -k[np.ix_(free, fixed)] @ u[fixed]
    u[free] = np.linalg.solve(k[np.ix_(free, free)], rhs)
    hh = h
    w0, t0, w1, t1 = u[0], u[1], u[2], u[3]
    curv0 = (-6.0 / hh ** 2 * w0 - 4.0 / hh * t0
             + 6.0 / hh ** 2 * w1 - 2.0 / hh * t1)
    return float(abs(rigidity * curv0 / u[1]))


def tripping_eigen_fem(section: TeeTorsionSection, *, span: float,
                       rotational_restraint: float = 0.0,
                       lateral_translation: str = "restrained",
                       elements: int = 16) -> float:
    """Tripping stress by a Hermite finite-element eigenvalue solve.

    Independent discretisation of the same energy functional (piecewise cubic
    fields instead of the module's global sine): with ``phi = 0`` at the ends
    (warping free, rotations free) the smallest positive generalised eigenvalue
    of ``K v = P G v`` is the buckling axial force, returned as a stress.
    """
    n = int(elements)
    h = span / n
    b, g, m = _hermite_matrices(h)
    j = section.saint_venant_constant
    e = section.young
    r2 = section.polar_radius_squared
    d = section.centroid_z
    n1 = 2 * (n + 1)
    ndof = n1 if lateral_translation == "restrained" else 2 * n1
    if lateral_translation == "restrained":
        k_el = _assemble(n, span, [e * section.warping_constant * b,
                                   j * section.shear_modulus_value * g,
                                   rotational_restraint * m], [0, 0, 0], ndof=ndof)
        k_geo = _assemble(n, span, [r2 * g], [0], ndof=ndof)
        u_end = {0: 0.0, 2 * n: 0.0}
        dof_map = list(range(n1))
    else:
        k_el = _assemble_mixed(n, [
            (e * section.inertia_lateral * b, 0, 0),
            (e * section.warping_constant * b, n1, n1),
            (j * section.shear_modulus_value * g, n1, n1),
            (rotational_restraint * m, n1, n1),
        ], ndof)
        k_geo = _assemble_mixed(n, [
            (g, 0, 0),                                  # int u'^2
            (r2 * g, n1, n1),                           # R^2 int phi'^2
            (d * g, 0, n1), (d * g, n1, 0),             # 2 d int u' phi'
        ], ndof)
        u_end = {0: 0.0, 2 * n: 0.0, n1: 0.0, n1 + 2 * n: 0.0}
        dof_map = list(range(ndof))
    free = [dof for dof in dof_map if dof not in u_end]
    k_ff = k_el[np.ix_(free, free)]
    g_ff = k_geo[np.ix_(free, free)]
    vals = np.linalg.eigvals(np.linalg.solve(g_ff, k_ff))
    positive = [float(np.real(v)) for v in vals
                if abs(np.imag(v)) < 1e-6 * max(abs(v), 1.0) and np.real(v) > 0.0]
    if not positive:
        raise RuntimeError("the FE eigenproblem produced no positive buckling load")
    p = min(positive)
    return p / section.area


def analytic_free_torsion(section: TeeTorsionSection) -> float:
    """``sigma = G J A / I_p`` -- classic pure-torsion buckling of a strut."""
    return (section.shear_modulus_value * section.saint_venant_constant
            * section.area / section.polar_inertia)


def analytic_warping_torsion(section: TeeTorsionSection, span: float) -> float:
    """``sigma = (GJ + pi^2 E I_w / L^2) A / (I_p + A d^2)``."""
    k = math.pi / span
    return ((section.shear_modulus_value * section.saint_venant_constant
             + section.young * section.warping_constant * k ** 2)
            / (section.area * section.polar_radius_squared))


def flat_bar(height: float, thickness: float, **kw) -> TeeTorsionSection:
    """Bare flat bar: no flange outstand, so ``I_w -> 0`` (pure-torsion gate)."""
    return TeeTorsionSection(web_height=height, web_thickness=thickness,
                             flange_width=thickness, flange_thickness=thickness, **kw)


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def run_gates() -> dict[str, object]:
    tee = TeeTorsionSection(0.25, 0.010, 0.080, 0.012)
    span = 2.1
    checks: list[dict[str, object]] = []

    def record(name: str, value: float, reference: float, tol: float,
               note: str) -> None:
        value = float(value)
        reference = float(reference)
        err = abs(value - reference) / abs(reference) if reference else abs(value)
        checks.append({"gate": name, "value": value, "reference": reference,
                       "rel_error": err, "tolerance": tol,
                       "passed": bool(err <= tol), "note": note})

    def assert_gate(name: str, ok: bool, note: str) -> None:
        checks.append({"gate": name, "passed": bool(ok), "note": note})

    # ---- T14a/b: tripping vs an independent Hermite-FE eigenvalue solve ---- #
    got = float(tripping_stress(tee, span=span)["sigma_cr_Pa"])
    ref = tripping_eigen_fem(tee, span=span, elements=16)
    record("T14a-tripping_free_vs_FE", got, ref, 1e-4,
           "closed-form sine Ritz vs Hermite FE eigensolution, k_theta = 0")

    # weak restraint (1 % of the clamped spring): boundary layer is long, so the
    # sine Ritz is accurate here -- this is the range the design bracket lives in
    k_weak = 0.01 * plate_rotational_restraint(plate_thickness=0.012,
                                              stiffener_spacing=0.60)
    got = float(tripping_stress(tee, span=span, rotational_restraint=k_weak)["sigma_cr_Pa"])
    ref = tripping_eigen_fem(tee, span=span, rotational_restraint=k_weak, elements=24)
    record("T14b-tripping_weak_restraint_vs_FE", got, ref, 0.02,
           "k_theta = 1 % of the clamped spring (long boundary layer): sine Ritz "
           "vs converged Hermite FE")

    got = float(tripping_stress(tee, span=span, lateral_translation="free")["sigma_cr_Pa"])
    ref = tripping_eigen_fem(tee, span=span, lateral_translation="free", elements=16)
    record("T14b2-free_strut_vs_FE", got, ref, 1e-4,
           "coupled lateral--torsional problem vs the FE eigensolution")

    # ---- T14b3: known bias of the sine Ritz for k_theta > 0 ---------------- #
    k_rich = plate_rotational_restraint(plate_thickness=0.020,
                                        stiffener_spacing=0.35, far_edge="clamped")
    sine_ritz = float(tripping_stress(tee, span=span,
                                      rotational_restraint=k_rich)["sigma_cr_Pa"])
    fe = tripping_eigen_fem(tee, span=span, rotational_restraint=k_rich, elements=24)
    record("T14b3-sine_ritz_vs_FE_restrained", sine_ritz, fe, 5.0,
           "single-mode sine Ritz vs converged Hermite FE: the sine value is an "
           "upper bound (torsional boundary layer), documented and quantified, "
           "not used as the design value")
    assert_gate("T14b3-nonconservative_direction", sine_ritz > fe,
                "the sine Ritz must overestimate the restrained tripping stress")

    # ---- T14c: warping constant, numerical vs closed form ------------------ #
    record("T14c-warping_constant", tee.warping_constant,
           tee.warping_constant_closed_form, 0.001,
           "int omega^2 dA (64 samples) vs h_e^2 t_f b_f^3 / 12")

    # ---- T14d: mode summation ---------------------------------------------- #
    sigmas = [float(tripping_stress(tee, span=span, modes=m)["sigma_cr_Pa"])
              for m in range(1, 7)]
    record("T14d-mode_summation", (max(sigmas) - min(sigmas)) / sigmas[0], 0.0, 1e-12,
           "m = 1..6 minima must coincide for a prismatic member")

    # ---- T14e: full restraint suppresses tripping -------------------------- #
    huge = 1.0e14
    got = float(tripping_stress(tee, span=span, rotational_restraint=huge)["sigma_cr_Pa"])
    assert_gate("T14e-restraint_suppression", got > 1e3 * tee.young,
                "k_theta = 1e14 N: sigma_cr must exceed 1e3 E (tripping suppressed)")

    # ---- T14f: plate rotational spring vs independent beam FEM ------------- #
    for far, c in (("clamped", 4.0), ("pinned", 3.0)):
        for t_p, b_s in ((0.020, 0.35), (0.008, 1.0)):
            d = tee.young * t_p ** 3 / (12.0 * (1.0 - 0.3 ** 2))
            fem = beam_rotational_spring_fem(span=0.5 * b_s, rigidity=d,
                                             far_edge=far, elements=2)
            mod = plate_rotational_restraint(plate_thickness=t_p,
                                             stiffener_spacing=b_s,
                                             young=tee.young, poisson=0.3,
                                             far_edge=far)
            record(f"T14f-spring_closedform[{far},t={t_p},b={b_s}]",
                   mod, c * d / (0.5 * b_s), 1e-12,
                   "module vs c D / (b_s/2) written out independently")
            record(f"T14f-spring_FEM[{far},t={t_p},b={b_s}]", fem, mod, 1e-9,
                   "Hermite beam FEM end moment vs the module spring constant")

    # ---- T14g: web buckling ------------------------------------------------ #
    w1 = web_local_buckling_stress(web_height=0.25, web_thickness=0.010)
    w2 = web_local_buckling_stress(web_height=0.25, web_thickness=0.020)
    record("T14g-web_scaling_t", w2 / w1, 4.0, 1e-12,
           "sigma_w ~ t^2 -> doubling t must quadruple sigma_w")
    w3 = web_local_buckling_stress(web_height=0.50, web_thickness=0.010)
    record("T14g-web_scaling_h", w1 / w3, 4.0, 1e-12,
           "sigma_w ~ 1/h^2 -> doubling h must quarter sigma_w")
    record("T14g-buckling_coefficient_exact",
           plate_buckling_coefficient(web_height=0.55, panel_length=2.75), 4.0, 1e-3,
           "a/h = 5 exactly -> k = (m h/a + a/(m h))^2 minimum must be 4")
    k_used = (web_local_buckling_stress(web_height=0.55, web_thickness=0.010,
                                        panel_length=2.8)
              / web_local_buckling_stress(web_height=0.55, web_thickness=0.010))
    record("T14g-aspect_ratio_effect", k_used,
           plate_buckling_coefficient(web_height=0.55, panel_length=2.8) / 4.0, 1e-12,
           "the check's k must be the exact aspect-ratio coefficient")

    # ---- T14h: monotonicity ------------------------------------------------ #
    sig_long = [float(tripping_stress(tee, span=L)["sigma_cr_Pa"])
                for L in (5.0, 2.1, 1.0)]
    assert_gate("T14h-monotone_in_span", sig_long[0] < sig_long[1] < sig_long[2],
                "shorter unsupported length must raise the tripping stress")
    wide = TeeTorsionSection(0.25, 0.010, 0.160, 0.012)
    assert_gate("T14h-monotone_in_flange",
                float(tripping_stress(wide, span=span)["sigma_cr_Pa"])
                > float(tripping_stress(tee, span=span)["sigma_cr_Pa"]),
                "a wider flange raises I_w and therefore the tripping stress")
    thin_web = TeeTorsionSection(0.25, 0.006, 0.080, 0.012)
    assert_gate("T14h-thin_web_lower",
                float(tripping_stress(thin_web, span=span)["sigma_cr_Pa"])
                < float(tripping_stress(tee, span=span)["sigma_cr_Pa"]),
                "a thinner web (same depth and flange) must trip at a lower "
                "stress (J ~ t_w^3)")

    return {
        "schema": "tensorfem.grillage_local_instability.gates/1",
        "anchor": "GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE",
        "checks": checks,
        "all_passed": all(c["passed"] for c in checks),
    }


# --------------------------------------------------------------------------- #
# application: the anchor case-study geometry
# --------------------------------------------------------------------------- #
def _restraint_fraction_for(section: TeeTorsionSection, *, span: float,
                            spacing: float, plate_thickness: float,
                            target_stress: float) -> dict[str, float]:
    """Restraint fraction ``eta`` at which tripping reaches ``target_stress``.

    ``sigma_tr(eta) = (GJ k^2 + EI_w k^4 + eta k_theta,max) / (A R^2 k^2)`` is
    linear in ``eta``; the fraction is solved directly (no search).
    """
    k = math.pi / span
    k_torsion = (section.saint_venant_constant * section.shear_modulus_value * k * k
                 + section.young * section.warping_constant * k ** 4)
    need = target_stress * section.area * section.polar_radius_squared * k * k
    k_theta_max = plate_rotational_restraint(plate_thickness=plate_thickness,
                                             stiffener_spacing=spacing,
                                             young=section.young,
                                             poisson=section.poisson,
                                             far_edge="clamped")
    eta = (need - k_torsion) / k_theta_max if k_theta_max > 0 else math.inf
    return {"k_theta_max_N": k_theta_max, "torsional_stiffness_Nm2": k_torsion,
            "required_k_theta_N": need, "required_fraction": eta}


def anchor_case() -> dict[str, object]:
    geom = case_study_geometry()
    cap = stiffener_axial_capacity(geom)
    sec = TeeTorsionSection(geom.stiffener.web_height, geom.stiffener.web_thickness,
                            geom.stiffener.flange_width, geom.stiffener.flange_thickness,
                            young=geom.young, poisson=geom.poisson)
    chk = stiffener_instability_check(
        sec, span=geom.frame_spacing, stiffener_spacing=geom.bay_spacing,
        plate_thickness=geom.plate_thickness, yield_stress=geom.yield_stress)

    frame = geom.frame
    sec_f = TeeTorsionSection(frame.web_height, frame.web_thickness,
                              frame.flange_width, frame.flange_thickness,
                              young=geom.young, poisson=geom.poisson)
    chk_f = stiffener_instability_check(
        sec_f, span=geom.frame_span, stiffener_spacing=geom.frame_spacing,
        plate_thickness=geom.plate_thickness, yield_stress=geom.yield_stress)

    column = cap["limiting_stress"]
    target = min(column, chk["web_local_buckling"]["min_Pa"])
    frac = _restraint_fraction_for(sec, span=geom.frame_spacing,
                                   spacing=geom.bay_spacing,
                                   plate_thickness=geom.plate_thickness,
                                   target_stress=target)
    # converged FE values replace the single-mode sine Ritz for the restrained ends
    fe_pinned = tripping_eigen_fem(sec, span=geom.frame_spacing,
                                   rotational_restraint=chk["plate_rotational_restraint_pinned_N"],
                                   elements=24)
    fe_clamped = tripping_eigen_fem(sec, span=geom.frame_spacing,
                                    rotational_restraint=chk["plate_rotational_restraint_clamped_N"],
                                    elements=24)
    fe_at_required = tripping_eigen_fem(sec, span=geom.frame_spacing,
                                        rotational_restraint=max(frac["required_k_theta_N"], 0.0),
                                        elements=24)
    return {
        "geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "longitudinal": {"web_mm": geom.stiffener.web_height * 1e3,
                             "web_thickness_mm": geom.stiffener.web_thickness * 1e3,
                             "flange_mm": geom.stiffener.flange_width * 1e3,
                             "flange_thickness_mm": geom.stiffener.flange_thickness * 1e3},
            "frame": {"web_mm": frame.web_height * 1e3,
                      "web_thickness_mm": frame.web_thickness * 1e3,
                      "flange_mm": frame.flange_width * 1e3,
                      "flange_thickness_mm": frame.flange_thickness * 1e3,
                      "span_m": geom.frame_span},
            "yield_stress_MPa": geom.yield_stress / 1e6,
        },
        "longitudinal": {
            "column_limiting_stress_MPa": column / 1e6,
            "johnson_ostenfeld_MPa": cap["johnson_ostenfeld_stress"] / 1e6,
            "perry_robertson_MPa": cap["perry_robertson_stress"] / 1e6,
            "web_local_buckling_MPa": chk["web_local_buckling"]["min_Pa"] / 1e6,
            "tripping_free_MPa": chk["tripping_sigma_free_Pa"] / 1e6,
            "tripping_pinned_single_mode_ritz_MPa": chk["tripping_sigma_pinned_Pa"] / 1e6,
            "tripping_clamped_single_mode_ritz_MPa": chk["tripping_sigma_clamped_Pa"] / 1e6,
            "tripping_pinned_FE_MPa": fe_pinned / 1e6,
            "tripping_clamped_FE_MPa": fe_clamped / 1e6,
            "tripping_at_required_restraint_FE_MPa": fe_at_required / 1e6,
            "FE_over_single_mode_ritz_pinned": fe_pinned / chk["tripping_sigma_pinned_Pa"],
            "FE_over_single_mode_ritz_clamped": fe_clamped / chk["tripping_sigma_clamped_Pa"],
            "tripping_free_strut_MPa": chk["tripping_sigma_free_strut_Pa"] / 1e6,
            "local_limiting_MPa": chk["limiting_stress_Pa"] / 1e6,
            "local_governing": chk["governing"],
            "local_over_column": chk["limiting_stress_Pa"] / column,
            "restraint_fraction_needed": frac,
        },
        "frame": {
            "web_local_buckling_MPa": chk_f["web_local_buckling"]["min_Pa"] / 1e6,
            "tripping_free_MPa": chk_f["tripping_sigma_free_Pa"] / 1e6,
            "tripping_pinned_single_mode_ritz_MPa": chk_f["tripping_sigma_pinned_Pa"] / 1e6,
            "tripping_clamped_single_mode_ritz_MPa": chk_f["tripping_sigma_clamped_Pa"] / 1e6,
            "tripping_free_strut_MPa": chk_f["tripping_sigma_free_strut_Pa"] / 1e6,
            "local_limiting_MPa": chk_f["limiting_stress_Pa"] / 1e6,
            "local_governing": chk_f["governing"],
            "local_over_yield": chk_f["limiting_over_yield"],
        },
        "verdict": (
            "With intact plating the plate rotational spring is "
            f"{frac['k_theta_max_N']:.3e} N against a torsional stiffness of "
            f"{frac['torsional_stiffness_Nm2']:.3e} N m^2, i.e. only "
            f"{100.0 * max(frac['required_fraction'], 0.0):.2f} % of the full "
            "mid-bay restraint is needed for the column state to stay critical; "
            "the existing chain is therefore valid for an intact plate but "
            "non-conservative by up to "
            f"{column / chk['limiting_stress_Pa']:.2f}x if that restraint is lost."),
    }


# --------------------------------------------------------------------------- #
# sweep: where does the local state govern?
# --------------------------------------------------------------------------- #
def sweep() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    young, nu, sy = 2.06e11, 0.3, 3.55e8
    for t_p_mm in (6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
        for b_s in (0.35, 0.60, 0.90, 1.20):
            for hw, tw in ((0.25, 0.010), (0.35, 0.010), (0.45, 0.010),
                           (0.55, 0.010), (0.65, 0.010), (0.55, 0.008),
                           (0.65, 0.007)):
                sec = TeeTorsionSection(hw, tw, 0.120, 0.014, young=young,
                                        poisson=nu)
                chk = stiffener_instability_check(
                    sec, span=2.8, stiffener_spacing=b_s,
                    plate_thickness=t_p_mm * 1e-3, yield_stress=sy)
                rows.append({
                    "plate_thickness_mm": t_p_mm,
                    "stiffener_spacing_m": b_s,
                    "web_height_mm": hw * 1e3,
                    "web_thickness_mm": tw * 1e3,
                    "web_depth_ratio": hw / tw,
                    "web_local_buckling_MPa": chk["web_local_buckling"]["min_Pa"] / 1e6,
                    "tripping_free_MPa": chk["tripping_sigma_free_Pa"] / 1e6,
                    "tripping_pinned_MPa": chk["tripping_sigma_pinned_Pa"] / 1e6,
                    "tripping_clamped_MPa": (chk["tripping_sigma_clamped_Pa"] / 1e6
                                             if math.isfinite(chk["tripping_sigma_clamped_Pa"])
                                             else None),
                    "local_limiting_MPa": chk["limiting_stress_Pa"] / 1e6,
                    "local_limiting_over_yield": chk["limiting_over_yield"],
                    "governing_free": ("web_local_buckling"
                                       if chk["web_local_buckling"]["min_Pa"]
                                       <= chk["tripping_sigma_free_Pa"] else "tripping"),
                })
    return {
        "schema": "tensorfem.grillage_local_instability.sweep/1",
        "span_m": 2.8,
        "yield_stress_MPa": sy / 1e6,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figures(sweep_data: dict[str, object], anchor: dict[str, object]) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    rows = sweep_data["rows"]

    # fig1: limiting stress vs web depth ratio for several plate thicknesses
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for t_p in (6.0, 8.0, 10.0, 12.0, 16.0, 20.0):
        pts = sorted((r["web_depth_ratio"], r["local_limiting_over_yield"])
                     for r in rows if r["plate_thickness_mm"] == t_p
                     and r["stiffener_spacing_m"] == 0.35)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-",
                label=f"t_p = {t_p:.0f} mm")
    ax.axhline(1.0, color="k", ls=":", lw=0.8, label="yield")
    ax.set_xlabel(r"web depth ratio $h_w/t_w$")
    ax.set_ylabel(r"$\sigma_{lim}/\sigma_Y$  (min of tripping, web buckling)")
    ax.set_title("Local instability limits, $b_s$ = 0.35 m, span 2.8 m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p1 = FIG / "fig1_local_limit_vs_web_ratio.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    out.append({"name": "fig1_local_limit_vs_web_ratio",
                "path": str(p1),
                "caption": "Limiting local-instability stress (lower bound of the "
                           "tripping bracket and web local buckling) normalised by "
                           "yield versus web depth ratio."})

    # fig2: tripping bracket vs plate restraint (for the anchor stiffener & frame)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    geom = case_study_geometry()
    for label, section, span, spacing in (
            ("longitudinal 250x10+80x12", anchor and TeeTorsionSection(
                geom.stiffener.web_height, geom.stiffener.web_thickness,
                geom.stiffener.flange_width, geom.stiffener.flange_thickness),
             geom.frame_spacing, geom.bay_spacing),
            ("frame 550x10+120x14", TeeTorsionSection(
                geom.frame.web_height, geom.frame.web_thickness,
                geom.frame.flange_width, geom.frame.flange_thickness),
             geom.frame_span, geom.frame_spacing)):
        etas = np.linspace(0.0, 0.02, 41)
        k_max = plate_rotational_restraint(plate_thickness=geom.plate_thickness,
                                           stiffener_spacing=spacing,
                                           far_edge="clamped")
        sig = [tripping_stress(section, span=span,
                               rotational_restraint=float(e * k_max))["sigma_cr_Pa"] / 1e6
               for e in etas]
        ax.plot(etas * 100.0, sig, "-", label=f"tripping, {label}")
    ax.axhline(geom.yield_stress / 1e6, color="k", ls=":", lw=0.8, label="yield")
    ax.set_xlabel("plate rotational restraint used [% of full mid-bay restraint]")
    ax.set_ylabel("tripping stress [MPa]")
    ax.set_title("Tripping is destroyed by even a few percent of plate restraint")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p2 = FIG / "fig2_tripping_vs_restraint.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    out.append({"name": "fig2_tripping_vs_restraint",
                "path": str(p2),
                "caption": "Tripping stress of the two anchor members versus the "
                           "fraction of the full mid-bay plate rotational restraint "
                           "actually available."})

    # fig3: governing-branch map over (plate thickness, stiffener spacing)
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    t_list = sorted({r["plate_thickness_mm"] for r in rows})
    b_list = sorted({r["stiffener_spacing_m"] for r in rows})
    grid = np.zeros((len(t_list), len(b_list)))
    for r in rows:
        i = t_list.index(r["plate_thickness_mm"])
        j = b_list.index(r["stiffener_spacing_m"])
        grid[i, j] = max(grid[i, j], r["local_limiting_over_yield"])
    im = ax.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(b_list)))
    ax.set_xticklabels([f"{b:.2f}" for b in b_list])
    ax.set_yticks(range(len(t_list)))
    ax.set_yticklabels([f"{t:.0f}" for t in t_list])
    ax.set_xlabel("stiffener spacing $b_s$ [m]")
    ax.set_ylabel("plate thickness $t_p$ [mm]")
    ax.set_title("Worst local-instability utilisation $\\sigma_{lim}/\\sigma_Y$\n"
                 "(lower bound: no plate rotational restraint)")
    for i in range(len(t_list)):
        for j in range(len(b_list)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label=r"$\sigma_{lim}/\sigma_Y$")
    p3 = FIG / "fig3_local_limit_map.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    out.append({"name": "fig3_local_limit_map",
                "path": str(p3),
                "caption": "Worst local-instability utilisation of the swept envelope "
                           "as a function of plate thickness and stiffener spacing."})
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    gates = run_gates()
    anchor = anchor_case()
    sweep_data = sweep()
    figs = figures(sweep_data, anchor)

    (ART / "gates.json").write_text(json.dumps(gates, indent=1))
    (ART / "anchor_instability.json").write_text(json.dumps(anchor, indent=1))
    (ART / "sweep.json").write_text(json.dumps(sweep_data, indent=1))

    print(f"gates: {sum(1 for c in gates['checks'] if c['passed'])}/"
          f"{len(gates['checks'])} passed, all_passed={gates['all_passed']}")
    lg = anchor["longitudinal"]
    print(f"longitudinal: column {lg['column_limiting_stress_MPa']:.1f} MPa, "
          f"local low {lg['local_limiting_MPa']:.1f} MPa "
          f"(tripping free {lg['tripping_free_MPa']:.1f}, web "
          f"{lg['web_local_buckling_MPa']:.1f}), "
          f"ratio {lg['local_over_column']:.3f}")
    fr = anchor["frame"]
    print(f"frame: local {fr['local_limiting_MPa']:.1f} MPa "
          f"({fr['local_governing']}), bracket up to "
          f"{fr['tripping_clamped_single_mode_ritz_MPa']:.1f} MPa (single-mode Ritz)")
    print(f"restraint fraction needed: "
          f"{100.0 * lg['restraint_fraction_needed']['required_fraction']:.3f} %; "
          f"FE tripping there {lg['tripping_at_required_restraint_FE_MPa']:.1f} MPa "
          f"(column state {lg['column_limiting_stress_MPa']:.1f} MPa); "
          f"FE/Ritz clamped {lg['FE_over_single_mode_ritz_clamped']:.3f}")
    print(f"figures: {len(figs)}")
    return 0 if gates["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())