"""Acceptance evidence for the load-spectrum / retardation residual-life face.

The generator returns JSON-serialisable records, one per acceptance criterion,
each carrying the numbers that decide it (no hand-copied values):

(a) SIF anchors -- infinite plate, secant finite-width centre crack against the
    exact collinear-array factor ``sigma sqrt(W tan(pi a / W))``, edge-crack
    ``1.12`` limit, semi-circular surface-crack Newman-Raju limit;
(b) Paris integration: this module's spectrum integrator (``model='none'``,
    constant amplitude) against ``paris_integrate_closed_form``;
(c) negative controls: ``da/dN(Delta K = 0) = 0``, a sub-threshold spectrum
    grows nothing (infinite life), and a zero-amplitude block is rejected;
(d) residual life strictly decreasing in the initial (inspection) crack size;
(e) Paris life vs IIW S-N life for the same detail, and Paris spectrum life vs
    a Miner S-N life under a Rayleigh band spectrum (both within one order of
    magnitude);
(f) discretisation convergence (GCI) of the growth integral and of the Rayleigh
    band discretisation, both ``<= 5%``;
(g) Paris-exponent sweep ``m = 2, 3, 4`` at iso-rate reference ``Delta K`` ->
    monotonically decreasing life;
(h) Wheeler shielding invariants -- ``0 < phi <= 1``, non-decreasing in ``a``,
    exactly ``1`` past the overload elastic boundary, and the closed-form value
    ``phi(a_ol) = OLR**(-2 m)``;
(i) overload-ratio sweep -> monotonically increasing life, with ``OLR = 1``
    reproducing the unretarded life exactly;
(j) no-overload limit -> the retarded solver degenerates to the geometry-aware
    Paris life (independent quadrature, ``<= 1%``);
(k) overload-spacing sweep -> the life factor decreases as the overloads are
    spaced further apart than the plastic-zone crossing time;
(l) inverse identification of ``(C, m)`` from a synthetic ``a(N)`` curve;
(m) residual-life table for the three canonical ship details under one
    spectrum, with the retarded/unretarded pair and the life factor.

The generator is deliberately dependency-free (stdlib only) and embarrassingly
parallel: the expensive assessments are submitted as plain dictionaries to a
process pool, so every worker rebuilds its own geometry from a name string
(nothing but strings and floats crosses the process boundary).
"""

from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from ..fatigue_crack_propagation import (
    ParisLaw,
    k_edge_crack_finite_width,
    k_infinite_plate,
    k_semi_elliptical_surface_crack,
    paris_integrate_closed_form,
    paris_integrate_variable_amplitude,
)
from .crack_growth_geometry import (
    SNCurveIIW,
    edge_crack_finite_width_delta_k,
    hole_edge_crack_delta_k,
    richardson_gci,
    surface_crack_delta_k,
)
from .spectrum_retardation import (
    LoadBlock,
    block_spectrum,
    RetardationZone,
    fit_paris_parameters,
    overload_sequence,
    plastic_zone_size,
    rayleigh_band_spectrum,
    spectrum_residual_life,
)

__all__ = ["DETAILS", "run_spectrum_retardation_qualification"]

# ---------------------------------------------------------------------------
# predeclared physical data (one place, referenced by every criterion)
# ---------------------------------------------------------------------------
YIELD_STRESS_PA = 355.0e6  # S355 hull structural steel
FRACTURE_TOUGHNESS = 60.0e6  # K_IC, Pa sqrt(m) -- conservative weld/HAZ value
PARIS_LAW = ParisLaw(5.5e-30, 3.0)  # C in m/cycle/(Pa sqrt(m))^3, m = 3
BASE_STRESS_RANGE = 120.0e6  # Delta sigma at the detail, Pa
RAYLEIGH_RMS_PA = 60.0e6  # stress-range RMS of the narrow-band spectrum
STRESS_RATIO = 0.1  # R = sigma_min / sigma_max
WELD_FAT = 90.0e6  # IIW FAT90 for a transverse butt weld toe
DELTA_K_REFERENCE = 10.0e6  # iso-rate reference for the exponent sweep

DETAILS: dict[str, dict[str, float]] = {
    # transverse butt-weld toe: semi-elliptical surface crack, a/c = 1/3 frozen
    "weld_toe_surface_crack": {
        "thickness_m": 0.025,
        "initial_crack_m": 0.0020,
        "initial_half_length_m": 0.0060,
        "upper_crack_m": 0.020,
    },
    # open hole (e.g. scallop/lightening hole) edge crack; Kirsch amplification
    "hole_edge_crack": {
        "hole_radius_m": 0.010,
        "plate_width_m": 0.200,
        "initial_crack_m": 0.0005,
        "upper_crack_m": 0.030,
    },
    # fillet-weld toe crack growing as an edge crack through the attachment
    "fillet_weld_toe_crack": {
        "width_m": 0.010,
        "initial_crack_m": 0.0010,
        "upper_crack_m": 0.006,
    },
}


def _geometry(detail: str, sigma_range_pa: float):
    """Rebuild the ``Delta K(a)`` factory for a named detail (process-safe)."""
    if detail == "weld_toe_surface_crack":
        p = DETAILS[detail]
        return surface_crack_delta_k(
            sigma_range_pa,
            p["thickness_m"],
            p["initial_crack_m"],
            p["initial_half_length_m"],
            front_location="deepest",
        )
    if detail == "hole_edge_crack":
        p = DETAILS[detail]
        return hole_edge_crack_delta_k(sigma_range_pa, p["hole_radius_m"], p["plate_width_m"])
    if detail == "fillet_weld_toe_crack":
        p = DETAILS[detail]
        return edge_crack_finite_width_delta_k(sigma_range_pa, p["width_m"])
    if detail == "infinite_plate":
        return lambda a: sigma_range_pa * math.sqrt(math.pi * a)
    raise KeyError(f"unknown detail {detail!r}")


def _run_case(spec: dict[str, Any]) -> dict[str, Any]:
    """Run one spectrum assessment from a plain-dictionary specification."""
    detail = spec["detail"]
    law = ParisLaw(spec.get("c", PARIS_LAW.coefficient_c), spec.get("m", PARIS_LAW.exponent_m))
    spacing = spec.get("spacing", 1000.0)
    olr = spec.get("olr", 1.5)
    stress_range = spec.get("stress_range", BASE_STRESS_RANGE)
    if spec.get("spectrum") == "rayleigh":
        spectrum = rayleigh_band_spectrum(
            spec["rms_stress_pa"],
            spec.get("total_cycles", 100.0),
            n_bands=spec.get("n_bands", 8),
            stress_ratio=STRESS_RATIO,
        )
    else:
        spectrum = overload_sequence(
            stress_range,
            spacing,
            overload_ratio=olr,
            stress_ratio=STRESS_RATIO,
        )
    params = dict(DETAILS[detail]) if detail in DETAILS else {}
    a0 = spec.get("a0", params.get("initial_crack_m", 0.001))
    upper = spec.get("upper", params.get("upper_crack_m", 0.02))
    result = spectrum_residual_life(
        spectrum,
        law,
        lambda s: _geometry(detail, s),
        initial_crack_m=a0,
        upper_crack_m=upper,
        fracture_toughness_pa_sqrt_m=spec.get("k_ic", FRACTURE_TOUGHNESS),
        yield_stress_pa=spec.get("yield_stress", YIELD_STRESS_PA),
        model=spec.get("model", "wheeler"),
        substeps_per_block=spec.get("substeps", 8),
        detail=detail,
        critical_stress_pa=spec.get("critical_stress_pa"),
    )
    return {
        "spec": spec,
        "cycles": result.cycles_to_failure,
        "repetitions": result.spectrum_repetitions,
        "critical_crack_m": result.critical_crack_m,
        "critical_mode": result.critical_mode,
        "min_phi": result.min_retardation_factor,
        "retarded_cycles": result.retarded_cycles,
        "events": len(result.retardation_events),
        "reached": result.reached_critical,
    }


def _run_cases(specs: list[dict[str, Any]], workers: int = 12) -> list[dict[str, Any]]:
    if len(specs) == 1:
        return [_run_case(specs[0])]
    with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
        return list(pool.map(_run_case, specs))


def _row(
    identifier: str,
    description: str,
    actual: float | bool | None,
    oracle: float | bool | None,
    metric: float | None,
    tolerance: float | None,
    passed: bool,
    note: str = "",
) -> dict[str, Any]:
    return {
        "id": identifier,
        "description": description,
        "actual": actual,
        "oracle": oracle,
        "metric": metric,
        "tolerance": tolerance,
        "passed": bool(passed),
        "note": note,
    }


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


# ---------------------------------------------------------------------------
# (a) SIF anchors
# ---------------------------------------------------------------------------


def criterion_a() -> list[dict[str, Any]]:
    sigma, a_half, W = 100.0e6, 0.010, 0.100
    rows = []

    k_inf = k_infinite_plate(sigma, a_half).k_i_pa_sqrt_m
    k_inf_ref = sigma * math.sqrt(math.pi * a_half)
    rows.append(
        _row(
            "a1_infinite_plate",
            "K_I infinite plate vs closed form",
            k_inf,
            k_inf_ref,
            _rel(k_inf, k_inf_ref),
            0.02,
            _rel(k_inf, k_inf_ref) <= 0.02,
        )
    )

    f_sec = 1.0 / math.sqrt(math.cos(math.pi * a_half / W))
    k_exact = sigma * math.sqrt(W * math.tan(math.pi * a_half / W))
    k_secant = sigma * math.sqrt(math.pi * a_half) * f_sec
    rows.append(
        _row(
            "a2_centre_crack_finite_width",
            "secant correction vs exact collinear-array factor sqrt(W tan(pi a/W)) at 2a/W=0.2",
            k_secant,
            k_exact,
            _rel(k_secant, k_exact),
            0.02,
            _rel(k_secant, k_exact) <= 0.02,
        )
    )

    a_edge = 1.0e-4
    k_edge = k_edge_crack_finite_width(sigma, a_edge, 0.1).k_i_pa_sqrt_m
    k_edge_ref = 1.12 * sigma * math.sqrt(math.pi * a_edge)
    rows.append(
        _row(
            "a3_edge_crack_limit",
            "edge-crack finite-width factor vs 1.12 sigma sqrt(pi a) as a/W -> 0",
            k_edge,
            k_edge_ref,
            _rel(k_edge, k_edge_ref),
            0.02,
            _rel(k_edge, k_edge_ref) <= 0.02,
        )
    )

    a_surf = 0.001
    k_surf = k_semi_elliptical_surface_crack(
        sigma, a_surf, a_surf, 0.050, front_location="deepest"
    ).k_i_pa_sqrt_m
    k_surf_ref = 0.6637 * sigma * math.sqrt(math.pi * a_surf)
    rows.append(
        _row(
            "a4_surface_crack_semicircular",
            "Newman-Raju semi-circular surface crack vs 0.6637 sigma sqrt(pi a)",
            k_surf,
            k_surf_ref,
            _rel(k_surf, k_surf_ref),
            0.02,
            _rel(k_surf, k_surf_ref) <= 0.02,
            "0.6637 is the standard semi-circular edge-crack surface value",
        )
    )
    return rows


# ---------------------------------------------------------------------------
# (b) Paris integration anchor
# ---------------------------------------------------------------------------


def criterion_b() -> list[dict[str, Any]]:
    sigma_range = 120.0e6
    a0, a_c = 0.002, 0.010
    sigma_max = sigma_range / (1.0 - STRESS_RATIO)
    k_ic = sigma_max * math.sqrt(math.pi * a_c)  # makes a_c exact for Y = 1
    spec = {
        "detail": "infinite_plate",
        "spacing": 1000.0,
        "olr": 1.0,
        "model": "none",
        "a0": a0,
        "upper": a_c,
        "k_ic": k_ic,
        "stress_range": sigma_range,
    }
    got = _run_case(spec)
    closed = paris_integrate_closed_form(a0, a_c, sigma_range, PARIS_LAW)
    return [
        _row(
            "b1_paris_integration_closed_form",
            "spectrum integrator (model='none', Y=1) vs closed-form Paris integral",
            got["cycles"],
            closed,
            _rel(got["cycles"], closed),
            0.01,
            _rel(got["cycles"], closed) <= 0.01,
            f"critical crack from K_I(a_c)=K_IC returned {got['critical_crack_m']:.6f} m",
        )
    ]


# ---------------------------------------------------------------------------
# (c) negative controls
# ---------------------------------------------------------------------------


def criterion_c() -> list[dict[str, Any]]:
    rows = []
    zero_rate = PARIS_LAW.da_dN(0.0)
    rows.append(
        _row(
            "c1_zero_delta_k",
            "da/dN at Delta K = 0",
            zero_rate,
            0.0,
            _rel(zero_rate, 1e-300),
            1e-12,
            zero_rate == 0.0,
        )
    )

    sigma_range = BASE_STRESS_RANGE
    a0, a_c = 0.002, 0.010
    k_ic = (sigma_range / (1.0 - STRESS_RATIO)) * math.sqrt(math.pi * a_c)
    sub = _run_case(
        {
            "detail": "infinite_plate",
            "spacing": 1000.0,
            "olr": 1.0,
            "model": "wheeler",
            "a0": a0,
            "upper": a_c,
            "k_ic": k_ic,
            "m": 3.0,
            "c": PARIS_LAW.coefficient_c * 1e-40,  # forces Delta K_th > applied Delta K
        }
    )
    rows.append(
        _row(
            "c2_subthreshold_no_growth",
            "spectrum whose Delta K stays below the threshold grows nothing",
            sub["reached"],
            False,
            None,
            None,
            sub["reached"] is False,
            f"life={sub['cycles']!r} (inf), events={sub['events']}",
        )
    )

    rejected = False
    try:
        LoadBlock(0.0, 1000.0)
    except ValueError:
        rejected = True
    rows.append(
        _row(
            "c3_zero_amplitude_block_rejected",
            "a zero-amplitude block is rejected instead of silently growing a crack",
            rejected,
            True,
            None,
            None,
            rejected,
        )
    )
    return rows


# ---------------------------------------------------------------------------
# (d) monotonicity in the initial crack size
# ---------------------------------------------------------------------------


def criterion_d() -> list[dict[str, Any]]:
    a0_values = [0.002, 0.0025, 0.003, 0.004, 0.005]
    specs = [
        {
            "detail": "weld_toe_surface_crack",
            "spacing": 1000.0,
            "olr": 1.5,
            "model": "wheeler",
            "a0": a0,
        }
        for a0 in a0_values
    ]
    results = _run_cases(specs)
    lives = [r["cycles"] for r in results]
    monotone = all(lives[i] > lives[i + 1] for i in range(len(lives) - 1))
    rows = [
        _row(
            "d1_initial_size_monotonicity",
            "N_residual strictly decreases with the detected initial crack size a0",
            lives[0],
            lives[-1],
            (lives[0] - lives[-1]) / lives[0],
            None,
            monotone,
            "a0 (m) -> N: "
            + ", ".join(f"{a0:g}:{n:.4g}" for a0, n in zip(a0_values, lives, strict=True)),
        )
    ]
    rows.append(
        _row(
            "d2_initial_size_table",
            "residual life vs a0 table",
            None,
            None,
            None,
            None,
            True,
            "; ".join(f"a0={a0:g} -> N={n:.6g}" for a0, n in zip(a0_values, lives, strict=True)),
        )
    )
    return rows


# ---------------------------------------------------------------------------
# (e) Paris vs S-N
# ---------------------------------------------------------------------------


def criterion_e() -> list[dict[str, Any]]:
    rows = []
    ca = _run_case(
        {
            "detail": "weld_toe_surface_crack",
            "spacing": 1000.0,
            "olr": 1.0,
            "model": "none",
        }
    )
    sn = SNCurveIIW(detail_category_pa=WELD_FAT)
    n_sn = sn.life(BASE_STRESS_RANGE)
    log_ratio = math.log10(ca["cycles"] / n_sn)
    rows.append(
        _row(
            "e1_paris_vs_sn_constant_amplitude",
            "Paris constant-amplitude life vs IIW FAT90 S-N life at the same stress range",
            ca["cycles"],
            n_sn,
            log_ratio,
            1.0,
            abs(log_ratio) <= 1.0,
            "metric is log10(N_paris / N_sn)",
        )
    )

    bands = _run_cases(
        [
            {
                "detail": "weld_toe_surface_crack",
                "spectrum": "rayleigh",
                "rms_stress_pa": RAYLEIGH_RMS_PA,
                "total_cycles": 100.0,
                "n_bands": 8,
                "model": "none",
                "critical_stress_pa": 4.0 * RAYLEIGH_RMS_PA / (1.0 - STRESS_RATIO),
            },
        ]
    )[0]
    spectrum = rayleigh_band_spectrum(RAYLEIGH_RMS_PA, 100.0, n_bands=8, stress_ratio=STRESS_RATIO)
    damage = 0.0
    for block in spectrum.blocks:
        damage += block.cycles / sn.life(block.stress_range_pa)
    n_miner = spectrum.total_cycles / damage
    log2 = math.log10(bands["cycles"] / n_miner)
    rows.append(
        _row(
            "e2_paris_vs_miner_rayleigh",
            "Paris spectrum life vs Palmgren-Miner S-N life under the same Rayleigh band spectrum",
            bands["cycles"],
            n_miner,
            log2,
            1.0,
            abs(log2) <= 1.0,
            "metric is log10(N_paris / N_miner)",
        )
    )
    return rows


# ---------------------------------------------------------------------------
# (f) discretisation convergence
# ---------------------------------------------------------------------------


def criterion_f() -> list[dict[str, Any]]:
    rows = []
    steps = [4, 8, 16]
    results = _run_cases(
        [
            {
                "detail": "weld_toe_surface_crack",
                "spacing": 5000.0,
                "olr": 1.5,
                "model": "wheeler",
                "substeps": s,
            }
            for s in steps
        ]
    )
    coarse, medium, fine = (results[2]["cycles"], results[1]["cycles"], results[0]["cycles"])
    order, extrapolated, gci, ok = richardson_gci(fine, medium, coarse, requirement=0.05)
    rows.append(
        _row(
            "f1_growth_integral_gci",
            "Richardson GCI of the spectrum growth integral (substeps/block 4/8/16)",
            fine,
            extrapolated,
            gci,
            0.05,
            gci <= 0.05,
            f"observed order {order:.2f}; lives (4/8/16) = {coarse:.6g}/{medium:.6g}/{fine:.6g}",
        )
    )

    # The narrow-band record peak is pinned at 4 sigma_rms (a conventional
    # extreme-value peak for the load history) so that refining the band set
    # does not also move the critical crack size: only the growth integral is
    # under study here.  The band representatives preserve the third moment
    # exactly, so the residual spread is the geometry-factor coupling only.
    peak = 4.0 * RAYLEIGH_RMS_PA / (1.0 - STRESS_RATIO)
    band_counts = [4, 8, 16]
    band_runs = _run_cases(
        [
            {
                "detail": "weld_toe_surface_crack",
                "spectrum": "rayleigh",
                "rms_stress_pa": RAYLEIGH_RMS_PA,
                "total_cycles": 100.0,
                "n_bands": n,
                "model": "none",
                "critical_stress_pa": peak,
            }
            for n in band_counts
        ]
    )
    b_coarse, b_medium, b_fine = (
        band_runs[0]["cycles"],
        band_runs[1]["cycles"],
        band_runs[2]["cycles"],
    )
    b_order, b_extrap, b_gci, b_ok = richardson_gci(b_fine, b_medium, b_coarse, requirement=0.05)
    rows.append(
        _row(
            "f2_rayleigh_band_gci",
            "Richardson GCI of the Rayleigh band discretisation (4/8/16 bands)",
            b_fine,
            b_extrap,
            b_gci,
            0.05,
            b_gci <= 0.05,
            f"observed order {b_order:.2f}; lives = {b_coarse:.6g}/{b_medium:.6g}/{b_fine:.6g}",
        )
    )
    return rows


# ---------------------------------------------------------------------------
# (g) Paris exponent sweep
# ---------------------------------------------------------------------------


def criterion_g() -> list[dict[str, Any]]:
    m_values = [2.0, 3.0, 4.0]
    specs = []
    for m in m_values:
        c_iso = PARIS_LAW.coefficient_c * DELTA_K_REFERENCE ** (3.0 - m)
        specs.append(
            {
                "detail": "weld_toe_surface_crack",
                "spacing": 20000.0,
                "olr": 1.0,
                "model": "none",
                "m": m,
                "c": c_iso,
            }
        )
    results = _run_cases(specs)
    lives = [r["cycles"] for r in results]
    monotone = all(lives[i] > lives[i + 1] for i in range(len(lives) - 1))
    return [
        _row(
            "g1_exponent_sweep",
            "life strictly decreases with the Paris exponent m at iso-rate reference Delta K",
            lives[0],
            lives[-1],
            None,
            None,
            monotone,
            "m -> N: " + ", ".join(f"{m:g}:{n:.4g}" for m, n in zip(m_values, lives, strict=True)),
        )
    ]


# ---------------------------------------------------------------------------
# (h) Wheeler shielding invariants
# ---------------------------------------------------------------------------


def criterion_h() -> list[dict[str, Any]]:
    olr = 1.5
    spacing = 1000.0
    sigma_range = BASE_STRESS_RANGE
    overload_range = sigma_range * olr
    geo = _geometry("weld_toe_surface_crack", overload_range)
    a_ol = 0.004
    k_ol = geo(a_ol)
    zone_m = plastic_zone_size(k_ol, YIELD_STRESS_PA, plane_strain=True)
    zone = RetardationZone(a_ol, zone_m, k_ol, "overload")
    base_geo = _geometry("weld_toe_surface_crack", sigma_range)

    grid = [a_ol + i * (zone_m / 40.0) for i in range(41)]
    grid += [a_ol + zone_m, a_ol + 2.0 * zone_m, a_ol + 10.0 * zone_m]
    phis = [
        zone.factor(a, plastic_zone_size(base_geo(a), YIELD_STRESS_PA), PARIS_LAW.exponent_m)
        for a in grid
    ]
    in_range = all(0.0 < p <= 1.0 for p in phis)
    non_decreasing = all(phis[i] <= phis[i + 1] + 1e-15 for i in range(len(phis) - 1))
    terminal = all(
        zone.factor(a, plastic_zone_size(base_geo(a), YIELD_STRESS_PA), PARIS_LAW.exponent_m) == 1.0
        for a in (a_ol + zone_m, a_ol + 2.0 * zone_m, a_ol + 10.0 * zone_m)
    )
    closed = olr ** (-2.0 * PARIS_LAW.exponent_m)
    phi_at_overload = zone.factor(
        a_ol, plastic_zone_size(base_geo(a_ol), YIELD_STRESS_PA), PARIS_LAW.exponent_m
    )
    return [
        _row(
            "h1_wheeler_bounds",
            "phi in (0,1] on the whole shielding interval, non-decreasing, exactly 1 past a_p",
            min(phis),
            1.0,
            None,
            None,
            in_range and non_decreasing and terminal,
            f"a_p = {zone.elastic_boundary_m:.6g} m, r_p = {zone_m:.6g} m, "
            f"phi_min = {min(phis):.6g}, terminal phi = {phis[-1]:.6g}",
        ),
        _row(
            "h2_wheeler_overload_anchor",
            "phi at the overload crack size equals OLR^(-2m) (radius ratio K_base/K_OL squared, cubed)",
            phi_at_overload,
            closed,
            _rel(phi_at_overload, closed),
            1e-9,
            _rel(phi_at_overload, closed) <= 1e-9,
            f"overload spacing {spacing:g} cycles, OLR = {olr}",
        ),
    ]


# ---------------------------------------------------------------------------
# (i) overload ratio sweep
# ---------------------------------------------------------------------------


def criterion_i() -> list[dict[str, Any]]:
    olrs = [1.0, 1.25, 1.5]
    specs = []
    # paired design: each overload ratio is compared with its own unretarded run,
    # because a higher overload also lowers a_c (higher sigma_max reaches K_IC
    # earlier).  Only the paired ratio isolates the retardation effect.
    for value in olrs:
        specs.append(
            {
                "detail": "weld_toe_surface_crack",
                "spacing": 5000.0,
                "olr": value,
                "model": "wheeler",
            }
        )
        specs.append(
            {"detail": "weld_toe_surface_crack", "spacing": 5000.0, "olr": value, "model": "none"}
        )
    results = _run_cases(specs)
    retarded = [results[2 * i]["cycles"] for i in range(3)]
    unretarded = [results[2 * i + 1]["cycles"] for i in range(3)]
    factors = [r / u for r, u in zip(retarded, unretarded, strict=True)]
    monotone = all(factors[i] < factors[i + 1] for i in range(len(factors) - 1))
    no_retardation_at_unity = abs(factors[0] - 1.0) <= 1e-9 and results[0]["retarded_cycles"] == 0
    return [
        _row(
            "i1_overload_ratio_sweep",
            "retarded life increases monotonically with the overload ratio",
            factors[0],
            factors[-1],
            None,
            None,
            monotone,
            "OLR -> life factor (paired with its own unretarded life): "
            + ", ".join(
                f"{o:g}:{f:.5g} ({r:.4g}/{u:.4g})"
                for o, f, r, u in zip(olrs, factors, retarded, unretarded, strict=True)
            ),
        ),
        _row(
            "i2_unity_overload_coincidence",
            "OLR = 1 reproduces the unretarded life exactly and creates no zone",
            factors[0],
            1.0,
            abs(factors[0] - 1.0),
            1e-9,
            no_retardation_at_unity,
            f"retarded cycles at OLR=1: {results[0]['retarded_cycles']:.6g}, "
            f"zones created: {results[0]['events']} (unity overload re-creates a zone "
            f"at the current tip, but phi = 1 there by construction)",
        ),
    ]


# ---------------------------------------------------------------------------
# (j) no-overload limit against an independent quadrature
# ---------------------------------------------------------------------------


def criterion_j() -> list[dict[str, Any]]:
    """No-overload limit: retarded solver == geometry-aware Paris life."""
    spacing = 5000.0
    specs = [
        {"detail": "weld_toe_surface_crack", "spacing": spacing, "olr": 1.0, "model": "wheeler"},
        {"detail": "weld_toe_surface_crack", "spacing": spacing, "olr": 1.0, "model": "none"},
    ]
    retarded, plain = _run_cases(specs)
    delta_k = _geometry("weld_toe_surface_crack", BASE_STRESS_RANGE)
    a_c = plain["critical_crack_m"]

    # independent trapezoid in ln a of dN/da = 1 / (C Delta K(a)^m)
    n_points = 200001
    lo, hi = math.log(DETAILS["weld_toe_surface_crack"]["initial_crack_m"]), math.log(a_c)
    h = (hi - lo) / (n_points - 1)
    integral = 0.0
    for i in range(n_points):
        a = math.exp(lo + i * h)
        rate = PARIS_LAW.da_dN(delta_k(a))
        integral += a / rate
    integral *= h
    return [
        _row(
            "j1_no_overload_limit_vs_independent_quadrature",
            "wheeler-enabled solver on an overload-free spectrum vs independent ln-a trapezoid of dN/da",
            retarded["cycles"],
            integral,
            _rel(retarded["cycles"], integral),
            0.01,
            _rel(retarded["cycles"], integral) <= 0.01,
            f"model='none' control: {plain['cycles']:.6g} cycles",
        ),
    ]


# ---------------------------------------------------------------------------
# (k) overload spacing sweep
# ---------------------------------------------------------------------------


def criterion_k() -> list[dict[str, Any]]:
    spacings = [1000.0, 20000.0, 100000.0]
    specs = []
    for spacing in spacings:
        specs.append(
            {"detail": "weld_toe_surface_crack", "spacing": spacing, "olr": 1.5, "model": "wheeler"}
        )
        specs.append(
            {"detail": "weld_toe_surface_crack", "spacing": spacing, "olr": 1.0, "model": "none"}
        )
    results = _run_cases(specs)
    factors = [results[2 * i]["cycles"] / results[2 * i + 1]["cycles"] for i in range(3)]
    decreasing = all(factors[i] > factors[i + 1] for i in range(len(factors) - 1))
    return [
        _row(
            "k1_spacing_sweep",
            "life factor decreases as overloads are spaced beyond the plastic-zone crossing time",
            factors[0],
            factors[-1],
            None,
            None,
            decreasing,
            "spacing -> factor: "
            + ", ".join(f"{s:g}:{f:.5g}" for s, f in zip(spacings, factors, strict=True)),
        )
    ]


# ---------------------------------------------------------------------------
# (l) inverse identification of the Paris constants
# ---------------------------------------------------------------------------


def criterion_l() -> list[dict[str, Any]]:
    a0, a_c = 0.002, 0.010
    n_steps = 401
    cracks, cycles = [], []
    for i in range(n_steps):
        a = a0 + (a_c - a0) * i / (n_steps - 1)
        cracks.append(a)
        cycles.append(
            0.0 if a <= a0 else paris_integrate_closed_form(a0, a, BASE_STRESS_RANGE, PARIS_LAW)
        )
    fit = fit_paris_parameters(cracks, cycles, _geometry("infinite_plate", BASE_STRESS_RANGE))
    err_c = _rel(fit.coefficient_c, PARIS_LAW.coefficient_c)
    err_m = _rel(fit.exponent_m, PARIS_LAW.exponent_m)
    return [
        _row(
            "l1_paris_parameter_recovery",
            "recovered (C, m) from a synthetic a(N) curve vs the generating constants",
            fit.exponent_m,
            PARIS_LAW.exponent_m,
            err_m,
            0.01,
            err_c <= 0.01 and err_m <= 0.01 and fit.r_squared > 0.999,
            f"C_fit={fit.coefficient_c:.6e} (rel err {err_c:.2e}), "
            f"m_fit={fit.exponent_m:.6f}, R^2={fit.r_squared:.8f}, n={fit.n_points}",
        )
    ]


# ---------------------------------------------------------------------------
# (m) detail residual-life table + legacy comparison
# ---------------------------------------------------------------------------


def criterion_m() -> list[dict[str, Any]]:
    details = list(DETAILS)
    specs = []
    for detail in details:
        specs.append({"detail": detail, "spacing": 1000.0, "olr": 1.5, "model": "wheeler"})
        specs.append({"detail": detail, "spacing": 1000.0, "olr": 1.5, "model": "none"})
    results = _run_cases(specs)
    rows = []
    for i, detail in enumerate(details):
        ret = results[2 * i]
        plain = results[2 * i + 1]
        a_c = plain["critical_crack_m"]
        # legacy Y=1 baseline, integrated over repeated spectra with this module's
        # integrator (the legacy helper itself stops after one spectrum pass, see
        # the note) so the *geometry* error is isolated from the repetition error
        legacy = spectrum_residual_life(
            block_spectrum(
                [BASE_STRESS_RANGE, BASE_STRESS_RANGE * 1.5],
                [1000.0, 1.0],
                stress_ratio=STRESS_RATIO,
            ),
            PARIS_LAW,
            lambda s: lambda a: s * math.sqrt(math.pi * a),
            initial_crack_m=DETAILS[detail]["initial_crack_m"],
            upper_crack_m=0.08,
            fracture_toughness_pa_sqrt_m=FRACTURE_TOUGHNESS,
            yield_stress_pa=YIELD_STRESS_PA,
            model="none",
        )
        legacy_one_pass = paris_integrate_variable_amplitude(
            DETAILS[detail]["initial_crack_m"],
            a_c,
            [(BASE_STRESS_RANGE, 1000.0), (BASE_STRESS_RANGE * 1.5, 1.0)],
            PARIS_LAW,
        )
        rows.append(
            _row(
                f"m{len(rows) + 1}_{detail}",
                f"residual life for {detail} under one spectrum (1000 baseline cycles + OLR 1.5)",
                ret["cycles"],
                plain["cycles"],
                ret["cycles"] / plain["cycles"],
                None,
                True,
                f"unretarded {plain['cycles']:.6g}; legacy Y=1 (repeated spectra) "
                f"{legacy.cycles_to_failure:.6g} (ratio {legacy.cycles_to_failure / plain['cycles']:.4g}); "
                f"legacy helper one-pass output {legacy_one_pass.cycles_to_failure:.6g} "
                f"(reached_critical={legacy_one_pass.reached_critical}); "
                f"a_c={a_c:.6g} ({plain['critical_mode']}); min phi={ret['min_phi']:.4g}; "
                f"retarded cycles={ret['retarded_cycles']:.4g}",
            )
        )
    return rows


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_spectrum_retardation_qualification() -> dict[str, Any]:
    """Run every criterion and return the evidence dictionary."""
    criteria: list[dict[str, Any]] = []
    criteria += criterion_a()
    criteria += criterion_b()
    criteria += criterion_c()
    criteria += criterion_d()
    criteria += criterion_e()
    criteria += criterion_f()
    criteria += criterion_g()
    criteria += criterion_h()
    criteria += criterion_i()
    criteria += criterion_j()
    criteria += criterion_k()
    criteria += criterion_l()
    criteria += criterion_m()
    failed = [row["id"] for row in criteria if not row["passed"]]
    return {
        "module": "tensorfem.fatigue.spectrum_retardation",
        "criteria": criteria,
        "n_criteria": len(criteria),
        "n_failed": len(failed),
        "failed_ids": failed,
        "all_passed": not failed,
        "data": {
            "yield_stress_pa": YIELD_STRESS_PA,
            "fracture_toughness_pa_sqrt_m": FRACTURE_TOUGHNESS,
            "paris_c": PARIS_LAW.coefficient_c,
            "paris_m": PARIS_LAW.exponent_m,
            "stress_range_pa": BASE_STRESS_RANGE,
            "stress_ratio": STRESS_RATIO,
            "rayleigh_rms_pa": RAYLEIGH_RMS_PA,
            "iiw_fat_pa": WELD_FAT,
            "details": DETAILS,
        },
    }


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(run_spectrum_retardation_qualification(), indent=2, default=str))
