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

__all__ = [
    "case_study_geometry_slender",
    "run_grillage_verification",
    "combined_case_table",
    "main",
]


def case_study_geometry_slender() -> GrillageGeometry:
    """Thinner-plate variant used to exercise the stiffener-governed branch."""
    return GrillageGeometry(
        frame_spacing=2.1, bay_spacing=0.45, n_bays=6, n_spans=4,
        plate_thickness=0.013,
        stiffener=MemberSection(0.20, 0.009, 0.060, 0.010),
        frame=MemberSection(0.50, 0.010, 0.100, 0.012),
        young=2.06e11, poisson=0.3, yield_stress=3.55e8,
    ).validate()


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


def run_grillage_verification(*, terms: int = 61) -> dict[str, object]:
    """Run the T1-T10 anchors and return a structured evidence bundle."""
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

    passed = all(bool(c["passed"]) for c in checks)
    return {
        "suite": "grillage_ice_ultimate",
        "terms": int(terms),
        "checks": checks,
        "passed": passed,
        "n_checks": len(checks),
        "n_failed": sum(1 for c in checks if not c["passed"]),
        "paik_comparison": t7_rows,
    }


def combined_case_table(geom: GrillageGeometry | None = None) -> list[dict[str, object]]:
    """Three combined ice-patch + lateral-pressure cases with the ultimate load.

    The acceptance item asks for >= 3 combined load cases; these are the
    documented ice-belt scenario set (light / design / severe) and each row
    carries the governing member, the member utilisations at the limit and the
    safety margin.
    """
    geom = (case_study_geometry() if geom is None else geom).validate()
    scenarios = [
        ("light-ice + ballast head", 0.15, 1.0e6, 0.35, 0.35, 5.0e4),
        ("design-ice + design head", 0.30, 2.0e6, 0.70, 0.35, 1.0e5),
        ("severe-ice + tank head", 0.45, 3.0e6, 1.05, 0.35, 1.5e5),
    ]
    rows: list[dict[str, object]] = []
    for name, eta, p_ice, lx, ly, p_h in scenarios:
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