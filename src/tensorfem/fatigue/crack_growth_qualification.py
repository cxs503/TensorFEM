"""Acceptance evidence for the geometry-aware fatigue crack-growth face.

The generator returns one JSON-serialisable record per acceptance criterion:

(a) SIF anchors -- infinite plate, edge crack ``1.12``, semicircular surface
    crack ``0.6637`` (Newman-Raju), hole-edge Kirsch limit ``1.12 K_t``, and a
    finite-width centre-crack cross-check against Isida's tabulated correction;
(b) Paris integration closed-form vs the Runge-Kutta ODE integrator;
(c) negative control ``Delta K = 0 => da/dN = 0`` (and no growth, infinite life);
(d) residual-life monotonicity in the initial crack size;
(e) Paris life vs IIW S-N life for the same detail (within one order of magnitude);
(f) discretisation (GCI) convergence of the growth integral, ``<= 5%``;
(g) Paris-exponent sweep ``m = 2, 3, 4`` -> monotonically decreasing life;
plus the legacy-vs-geometry baseline table that quantifies the defect this
module removes.

Every record carries the numbers that decide the criterion, so the report can
be regenerated and re-audited without re-deriving anything by hand.
"""
from __future__ import annotations

import math

from ..fatigue_crack_propagation import (
    ParisLaw,
    k_edge_crack_finite_width,
    k_edge_crack_semi_infinite,
    k_hole_edge_crack,
    k_infinite_plate,
    k_semi_elliptical_surface_crack,
    paris_integrate_closed_form,
)
from .crack_growth_geometry import (
    SNCurveIIW,
    compare_paris_vs_iiw,
    convergence_study,
    fillet_weld_toe_life,
    hole_edge_crack_life,
    integrate_paris_ode,
    weld_toe_surface_crack_life,
)

__all__ = [
    "run_geometry_aware_qualification",
    "residual_life_scan",
    "growth_curve",
]

#: Reference values quoted from the open literature (used as oracles, never
#: computed from the implementation under test).
NEWMAN_RAJU_SEMICIRCULAR_F = 0.6637  # K_I = F sigma sqrt(pi a), a/c = 1, a/t -> 0
KIRSCH_K_T = 3.0  # circular hole, uniaxial tension
EDGE_CRACK_F = 1.12  # edge crack, semi-infinite plate
ISIDA_CENTRE_CRACK_F = {0.2: 1.0254, 0.4: 1.1118, 0.6: 1.2977}  # F(2a/W)


def _record(criterion: str, name: str, **payload: object) -> dict[str, object]:
    row = {"criterion": criterion, "id": name}
    row.update(payload)
    return row


def _rel(actual: float, oracle: float) -> float:
    return abs(actual - oracle) / abs(oracle)


def run_geometry_aware_qualification() -> dict[str, object]:
    """Run the full (a)-(g) acceptance suite and return the evidence bundle."""
    evidence: list[dict[str, object]] = []

    # ---------------------------------------------------------------- (a) SIF
    sigma = 100.0e6

    # (a1) finite-width correction -> 1 as a/W -> 0 (exact infinite-plate limit)
    a_tiny, width = 1.0e-8, 0.2
    k_fw = k_infinite_plate(sigma, a_tiny).k_i_pa_sqrt_m  # reference
    from .crack_growth_geometry import centre_crack_finite_width_delta_k

    dk_fw = centre_crack_finite_width_delta_k(sigma, width)
    k_fw_actual = dk_fw(a_tiny)
    err = _rel(k_fw_actual, k_fw)
    evidence.append(
        _record(
            "a",
            "finite_width_reduces_to_infinite_plate",
            actual=k_fw_actual,
            oracle=k_fw,
            relative_error=err,
            tolerance=1.0e-4,
            passed=err < 1.0e-4,
        )
    )

    # (a2) edge crack in a semi-infinite plate: published F = 1.12
    a_edge = 2.0e-3
    k_edge = k_edge_crack_semi_infinite(sigma, a_edge).k_i_pa_sqrt_m
    oracle_edge = EDGE_CRACK_F * sigma * math.sqrt(math.pi * a_edge)
    err = _rel(k_edge, oracle_edge)
    # the finite-width polynomial must reproduce the same limit as a/W -> 0
    y_poly_limit = k_edge_crack_finite_width(sigma, a_tiny, width).geometry_factor
    err_poly = _rel(y_poly_limit, EDGE_CRACK_F)
    evidence.append(
        _record(
            "a",
            "edge_crack_F_1.12",
            actual=k_edge,
            oracle=oracle_edge,
            relative_error=err,
            tolerance=0.02,
            finite_width_polynomial_limit=y_poly_limit,
            polynomial_limit_error=err_poly,
            passed=err < 0.02 and err_poly < 0.02,
        )
    )

    # (a3) semicircular surface crack, deepest point, a/t -> 0: F -> 0.6637
    t_ref = 1.0  # thick section so that a/t -> 0
    a_sc = 1.0e-5
    sif = k_semi_elliptical_surface_crack(sigma, a_sc, a_sc, t_ref, front_location="deepest")
    f_sc = sif.k_i_pa_sqrt_m / (sigma * math.sqrt(math.pi * a_sc))
    err_sc = _rel(f_sc, NEWMAN_RAJU_SEMICIRCULAR_F)
    evidence.append(
        _record(
            "a",
            "newman_raju_semicircular_surface_crack",
            actual=f_sc,
            oracle=NEWMAN_RAJU_SEMICIRCULAR_F,
            relative_error=err_sc,
            tolerance=0.02,
            a_over_t=a_sc / t_ref,
            passed=err_sc < 0.02,
        )
    )

    # (a4) hole-edge crack, a/R -> 0: F -> 1.12 * K_t = 3.36 (Kirsch)
    radius = 5.0e-3
    a_hole = 1.0e-7
    f_hole = k_hole_edge_crack(sigma, a_hole, radius).geometry_factor
    oracle_hole = EDGE_CRACK_F * KIRSCH_K_T
    err_hole = _rel(f_hole, oracle_hole)
    evidence.append(
        _record(
            "a",
            "hole_edge_kirsch_limit",
            actual=f_hole,
            oracle=oracle_hole,
            relative_error=err_hole,
            tolerance=0.02,
            passed=err_hole < 0.02,
        )
    )

    # (a5) finite-width centre crack vs Isida's tabulated correction F(2a/W).
    #      Reference basis: handbook table values; the secant model is validated
    #      in the stated range only (a model-range statement, not an
    #      implementation-error statement -- (a1) bounds the implementation).
    isida_rows = []
    for ratio, f_ref in ISIDA_CENTRE_CRACK_F.items():
        a_test = 0.5 * ratio * width
        y = k_infinite_plate(sigma, a_test).k_i_pa_sqrt_m
        y_model = dk_fw(a_test) / y
        isida_rows.append(
            {
                "twice_a_over_W": ratio,
                "model_factor": y_model,
                "isida_factor": f_ref,
                "relative_error": _rel(y_model, f_ref),
            }
        )
    worst = max(row["relative_error"] for row in isida_rows)
    evidence.append(
        _record(
            "a",
            "finite_width_centre_crack_vs_isida_table",
            rows=isida_rows,
            worst_relative_error=worst,
            tolerance=0.02,
            passed=worst < 0.02,
        )
    )

    # ------------------------------------------------------- (b) Paris integral
    law = ParisLaw(coefficient_c=1.0e-32, exponent_m=3.0)
    a0_b, ac_b, dsigma_b = 1.0e-4, 5.0e-3, 80.0e6
    n_closed = paris_integrate_closed_form(a0_b, ac_b, dsigma_b, law)
    dk_b = centre_crack_finite_width_delta_k(dsigma_b, 1.0e3)  # ~infinite plate
    n_rk4 = integrate_paris_ode(dk_b, law, a0_b, ac_b, n_steps=200)
    err_b = _rel(n_rk4, n_closed)
    # truly constant Delta K: N = (a_c - a_0) / (C Delta K^m)
    dk_const = 25.0e6
    n_const_closed = (ac_b - a0_b) / (law.coefficient_c * dk_const**law.exponent_m)
    n_const_rk4 = integrate_paris_ode(lambda _a: dk_const, law, a0_b, ac_b, n_steps=100)
    err_const = _rel(n_const_rk4, n_const_closed)
    evidence.append(
        _record(
            "b",
            "paris_closed_form_vs_rk4",
            actual=n_rk4,
            oracle=n_closed,
            relative_error=err_b,
            constant_delta_k_actual=n_const_rk4,
            constant_delta_k_oracle=n_const_closed,
            constant_delta_k_error=err_const,
            tolerance=0.01,
            passed=err_b < 0.01 and err_const < 0.01,
        )
    )

    # ---------------------------------------------- (c) negative control
    zero_law = ParisLaw(coefficient_c=1.0e-32, exponent_m=3.0)
    rate_at_zero = zero_law.da_dN(0.0)
    zero_life = integrate_paris_ode(lambda _a: 0.0, zero_law, 1.0e-4, 1.0e-3)
    below_threshold = ParisLaw(
        coefficient_c=1.0e-32, exponent_m=3.0, threshold_dk=5.0e6
    )
    below_life = integrate_paris_ode(lambda _a: 3.0e6, below_threshold, 1.0e-4, 1.0e-3)
    passed_c = (
        rate_at_zero == 0.0
        and math.isinf(zero_life)
        and math.isinf(below_life)
        and below_threshold.da_dN(5.0e6) == 0.0
    )
    evidence.append(
        _record(
            "c",
            "negative_control_zero_delta_k",
            da_dN_zero=rate_at_zero,
            cycles_at_zero_delta_k=zero_life,
            cycles_below_threshold=below_life,
            tolerance=0.0,
            passed=passed_c,
        )
    )

    # --------------------------------------------------- (d) monotonicity
    # Barsom-type ferritic-pearlitic steel, SI units: C = 1.5e-29
    # (m/cycle)/(Pa*sqrt(m))^3 == 1.5e-11 (m/cycle)/(MPa*sqrt(m))^3.
    law_c = ParisLaw(coefficient_c=1.5e-29, exponent_m=3.0)
    k_ic = 60.0e6
    scan = residual_life_scan(
        detail="hole_edge",
        initial_cracks_m=(5.0e-5, 1.0e-4, 2.0e-4, 5.0e-4, 1.0e-3),
        **{
            "hole_radius_m": 0.01,
            "plate_width_m": 0.15,
            "sigma_range_pa": 100.0e6,
            "law": law_c,
            "fracture_toughness_pa_sqrt_m": k_ic,
        },
    )
    monotone = all(
        scan[i]["n_f"] > scan[i + 1]["n_f"] for i in range(len(scan) - 1)
    )
    evidence.append(
        _record(
            "d",
            "residual_life_monotonic_in_initial_crack",
            rows=scan,
            monotone_decreasing=monotone,
            passed=monotone,
        )
    )

    # ------------------------------------------------- (e) S-N cross-check
    weld_life = weld_toe_surface_crack_life(
        thickness_m=0.020,
        crack_depth_m=2.0e-4,
        crack_half_length_m=1.0e-3,
        sigma_range_pa=100.0e6,
        law=law_c,
        fracture_toughness_pa_sqrt_m=k_ic,
        plate_half_width_m=1.0,
    )
    sn = SNCurveIIW(detail_category_pa=71.0e6)  # IIW FAT 71 weld toe
    scf = 1.2  # hot-spot / nominal at the weld toe
    agreement = compare_paris_vs_iiw(weld_life.n_f, sn, 100.0e6 * scf)
    evidence.append(
        _record(
            "e",
            "paris_vs_iiw_sn_weld_toe",
            n_f_paris=weld_life.n_f,
            n_f_sn=agreement.n_f_sn,
            log10_ratio=agreement.log10_ratio,
            hot_spot_stress_range_pa=100.0e6 * scf,
            detail_category_pa=sn.detail_category_pa,
            within_one_order=agreement.within_one_order,
            tolerance_orders=1.0,
            passed=agreement.within_one_order,
        )
    )

    # -------------------------------------------------------- (f) GCI
    study = convergence_study(
        centre_crack_finite_width_delta_k(80.0e6, 0.2), law, 1.0e-4, 5.0e-3, base_steps=50
    )
    gci_rows = [
        {"n_steps": study.step_counts[i], "cycles": study.values[i]} for i in range(3)
    ]
    evidence.append(
        _record(
            "f",
            "discretisation_gci_growth_integral",
            rows=gci_rows,
            observed_order=study.observed_order,
            extrapolated_cycles=study.extrapolated,
            gci_fine=study.gci_fine,
            tolerance=0.05,
            passed=study.gci_fine <= 0.05,
        )
    )

    # ------------------------------------------------ (g) Paris exponent sweep
    sweep_rows = []
    for m in (2.0, 3.0, 4.0):
        law_m = ParisLaw(coefficient_c=1.5e-29, exponent_m=m)
        n_closed_m = paris_integrate_closed_form(1.0e-4, 5.0e-3, 80.0e6, law_m)
        n_rk4_m = integrate_paris_ode(
            centre_crack_finite_width_delta_k(80.0e6, 1.0e3), law_m, 1.0e-4, 5.0e-3
        )
        sweep_rows.append(
            {
                "m": m,
                "n_f_rk4": n_rk4_m,
                "n_f_closed": n_closed_m,
                "relative_error": _rel(n_rk4_m, n_closed_m),
            }
        )
    monotone_m = (
        sweep_rows[0]["n_f_rk4"] > sweep_rows[1]["n_f_rk4"] > sweep_rows[2]["n_f_rk4"]
    )
    closed_ok = all(row["relative_error"] < 0.01 for row in sweep_rows)
    evidence.append(
        _record(
            "g",
            "paris_exponent_sweep",
            rows=sweep_rows,
            monotone_decreasing=monotone_m,
            closed_form_within_1pct=closed_ok,
            passed=monotone_m and closed_ok,
        )
    )

    # ------------------------------------- baseline: the defect being removed
    baseline_rows = [_baseline("weld_toe", law_c, k_ic), _baseline("hole_edge", law_c, k_ic),
                     _baseline("fillet_weld", law_c, k_ic)]
    evidence.append(
        _record(
            "baseline",
            "legacy_uniform_Y_vs_geometry_aware",
            rows=baseline_rows,
            passed=True,
            note="non-conservative legacy life / geometry-aware life",
        )
    )

    all_passed = all(bool(row.get("passed", False)) for row in evidence)
    return {
        "scope": (
            "TensorFEM geometry-aware Paris crack growth: SIF anchors, residual "
            "life, IIW S-N cross-check, GCI convergence"
        ),
        "certification": "Verification evidence only; not class-society certification",
        "units": {"stress": "Pa", "length": "m", "cycles": "-"},
        "evidence": evidence,
        "passed": all_passed,
    }


def _baseline(detail: str, law: ParisLaw, k_ic: float) -> dict[str, object]:
    common = {"law": law, "fracture_toughness_pa_sqrt_m": k_ic}
    if detail == "weld_toe":
        life = weld_toe_surface_crack_life(
            thickness_m=0.020,
            crack_depth_m=2.0e-4,
            crack_half_length_m=1.0e-3,
            sigma_range_pa=100.0e6,
            **common,
        )
    elif detail == "hole_edge":
        life = hole_edge_crack_life(
            hole_radius_m=0.01,
            crack_length_m=2.0e-4,
            plate_width_m=0.15,
            sigma_range_pa=100.0e6,
            **common,
        )
    elif detail == "fillet_weld":
        life = fillet_weld_toe_life(
            plate_thickness_m=0.02,
            weld_leg_m=0.008,
            crack_length_m=2.0e-4,
            sigma_range_pa=100.0e6,
            **common,
        )
    else:  # pragma: no cover - programming error
        raise ValueError(detail)
    return {
        "detail": detail,
        "n_f_geometry_aware": life.n_f,
        "n_f_legacy_uniform_Y": life.n_f_legacy,
        "ratio_legacy_over_geometry": life.ratio_legacy_to_geometry,
        "a_c_geometry_m": life.critical_crack_m,
        "delta_k_initial_pa_sqrt_m": life.delta_k_initial_pa_sqrt_m,
        "Y_initial": life.geometry_factor_initial,
        "gci_fine": life.gci_fine,
    }


def residual_life_scan(
    detail: str, initial_cracks_m: tuple[float, ...], **kwargs: object
) -> list[dict[str, float]]:
    """Residual life over a vector of initial crack sizes (same detail)."""
    rows: list[dict[str, float]] = []
    builders = {
        "weld_toe": weld_toe_surface_crack_life,
        "hole_edge": hole_edge_crack_life,
        "fillet_weld": fillet_weld_toe_life,
    }
    if detail not in builders:
        raise ValueError(f"unknown detail {detail!r}")
    builder = builders[detail]
    key = {
        "weld_toe": "crack_depth_m",
        "hole_edge": "crack_length_m",
        "fillet_weld": "crack_length_m",
    }[detail]
    for a0 in initial_cracks_m:
        life = builder(**{key: a0}, **kwargs)  # type: ignore[arg-type]
        rows.append(
            {
                "a_0_m": a0,
                "a_c_m": life.critical_crack_m,
                "n_f": life.n_f,
                "n_f_legacy": life.n_f_legacy,
                "gci_fine": life.gci_fine,
            }
        )
    return rows


def growth_curve(
    delta_k_of_a, law: ParisLaw, a0: float, a_c: float, n_samples: int = 60
) -> tuple[list[float], list[float]]:
    """Crack-size history ``(a, N)`` for plotting, obtained by repeated solves."""
    alphas = [
        a0 * (a_c / a0) ** (i / (n_samples - 1)) for i in range(n_samples)
    ]
    cycles = [0.0]
    for i in range(1, n_samples):
        cycles.append(
            integrate_paris_ode(delta_k_of_a, law, a0, alphas[i], n_steps=200)
        )
    return alphas, cycles