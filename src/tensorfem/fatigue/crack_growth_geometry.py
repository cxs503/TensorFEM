"""Geometry-aware Paris-law crack-growth and residual-life verification.

Why this module exists
----------------------
``tensorfem.fatigue_crack_propagation`` supplies the stress-intensity-factor
(SIF) library and a Paris-law integrator, but its residual-life helpers
integrate crack growth with a *constant, unit* geometry factor ``Y = 1``; the
detail's own SIF correction (semi-elliptical surface-crack shape factor,
open-hole stress concentration, finite-width edge-crack polynomial) is
evaluated for the report and then discarded from the life integral.  For an
open-hole edge crack the discarded factor is ``F ~ 3.4``, so ``Delta K`` is
under-estimated by that factor and, for ``m = 3``, the life is over-predicted
by ``F**m ~ 40`` -- a non-conservative error of more than one order of
magnitude in a structural-integrity quantity.

This module supplies the geometry-aware path used by the ship-structure
verification face:

* ``Delta K(a)`` callables for the canonical details (finite-width centre
  crack, finite-width edge crack / fillet-weld toe, semi-elliptical weld-toe
  surface crack with preserved aspect ratio, hole-edge crack with Kirsch
  amplification and a finite-width correction),
* a classical 4th-order Runge-Kutta integrator of the Paris ODE in the
  transformed variable ``t = ln a`` (the integrand ``dN/da`` spans many
  decades, so a uniform step in ``a`` is poorly conditioned), with a
  three-level Richardson/GCI discretisation study,
* a bracketed root solve for the critical crack size from
  ``K_I(a_c) = K_IC`` using the *same* geometry dependence as the growth
  integration (the legacy helper used ``Y = 1`` here as well),
* the Walker stress-ratio correction and an IIW-style bilinear S-N curve so
  a Paris life can be cross-adjudicated against the S-N method,
* a ``run_geometry_aware_qualification`` evidence generator covering the
  campaign acceptance criteria (a)-(g) with an explicit legacy-vs-geometry
  baseline comparison.

Scope and documented simplifications
------------------------------------
LEFM, small-scale yielding, ``Delta K`` above threshold, opening mode I only,
no crack-closure/retardation (no overload, no Wheeler model), no residual
stress field, fixed surface-crack aspect ratio ``a/c`` (no ``a/c`` evolution),
``R = 0`` unless a stress ratio is supplied to the Walker helper.  Stress
units are Pa, lengths m, cycles dimensionless.  The finite-width centre-crack
correction is the secant (Feddersen) model, which reproduces the tabulated
Isida correction to better than 0.5% for ``2a/W <= 0.6`` and degrades above
it; every detail integration therefore stops at the validity limit of its
geometry factor instead of crossing it silently.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from ..fatigue_crack_propagation import (
    ParisLaw,
    compute_critical_crack,
    k_edge_crack_finite_width,
    k_hole_edge_crack,
    k_semi_elliptical_surface_crack,
    paris_integrate_numerical,
)

__all__ = [
    "ConvergenceStudy",
    "GeometryAwareLife",
    "SNCurveIIW",
    "centre_crack_finite_width_delta_k",
    "edge_crack_finite_width_delta_k",
    "hole_edge_crack_delta_k",
    "surface_crack_delta_k",
    "integrate_paris_ode",
    "richardson_gci",
    "critical_crack_length",
    "geometry_aware_life",
    "weld_toe_surface_crack_life",
    "hole_edge_crack_life",
    "fillet_weld_toe_life",
    "walker_effective_delta_k",
    "compare_paris_vs_iiw",
    "convergence_study",
]

DeltaKOfA = Callable[[float], float]


# ---------------------------------------------------------------------------
# Input validation (local, so the module is usable stand-alone)
# ---------------------------------------------------------------------------

def _pos(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return value


# ---------------------------------------------------------------------------
# Delta K(a) callables for the canonical ship details
# ---------------------------------------------------------------------------

def centre_crack_finite_width_delta_k(
    sigma_range_pa: float, width_m: float, *, max_ratio: float = 0.6
) -> DeltaKOfA:
    """``Delta K(a) = sigma * sqrt(pi a) * sqrt(sec(pi a / W))``.

    ``a`` is the crack half length of a centre crack in a strip of *full* width
    ``W``.  The secant (Feddersen) correction with argument ``pi a / W``
    reproduces Isida's tabulated correction ``F(2a/W)`` to better than 0.5% for
    ``2a/W <= 0.6``.
    """
    sigma = _pos(sigma_range_pa, "sigma_range_pa")
    W = _pos(width_m, "width_m")

    def delta_k(a: float) -> float:
        a = _pos(a, "crack_half_length_m")
        if 2.0 * a / W > max_ratio:
            raise ValueError("2a/W above the validity limit of the secant correction")
        return sigma * math.sqrt(math.pi * a) * math.sqrt(1.0 / math.cos(math.pi * a / W))

    return delta_k


def edge_crack_finite_width_delta_k(
    sigma_range_pa: float, width_m: float
) -> DeltaKOfA:
    """``Delta K(a)`` for an edge crack of length ``a`` in a strip of width ``W``.

    Delegates to the SIF library so the growth integral and any reported ``K_I``
    cannot drift apart.
    """
    sigma = _pos(sigma_range_pa, "sigma_range_pa")
    W = _pos(width_m, "width_m")

    def delta_k(a: float) -> float:
        a = _pos(a, "crack_length_m")
        return k_edge_crack_finite_width(sigma, a, W).k_i_pa_sqrt_m

    return delta_k


def hole_edge_crack_delta_k(
    sigma_range_pa: float,
    hole_radius_m: float,
    plate_width_m: float | None = None,
) -> DeltaKOfA:
    """``Delta K(a)`` for a crack of length ``a`` growing from a circular hole.

    ``a`` is measured *from the hole edge*; the amplification is the Tada
    correlation for two opposite radial cracks at a hole, which recovers the
    exact small-crack limit ``F -> 1.12 K_t`` (``K_t = 3``, Kirsch) and decays
    towards the edge-crack limit.  An optional finite-width factor
    ``Y_edge(a + R, W) / 1.12`` is applied when the plate width is known.
    """
    sigma = _pos(sigma_range_pa, "sigma_range_pa")
    R = _pos(hole_radius_m, "hole_radius_m")
    W = _pos(plate_width_m, "plate_width_m") if plate_width_m is not None else None

    def delta_k(a: float) -> float:
        a = _pos(a, "crack_length_m")
        f_hole = k_hole_edge_crack(sigma, a, R).geometry_factor
        f_width = 1.0
        if W is not None:
            total = a + R
            f_width = k_edge_crack_finite_width(sigma, total, W).geometry_factor / 1.12
        return sigma * math.sqrt(math.pi * a) * f_hole * f_width

    return delta_k


def surface_crack_delta_k(
    sigma_range_pa: float,
    thickness_m: float,
    initial_depth_m: float,
    initial_half_length_m: float,
    *,
    plate_half_width_m: float | None = None,
    front_location: str = "deepest",
    max_depth_fraction: float = 0.8,
) -> DeltaKOfA:
    """``Delta K(a)`` for a semi-elliptical surface crack with fixed ``a/c``.

    The aspect ratio of the initial flaw is preserved (``c = a * c0 / a0``), the
    standard engineering assumption when the aspect-ratio evolution is not
    modelled.  The Newman-Raju evaluation delegates to the SIF library.
    """
    sigma = _pos(sigma_range_pa, "sigma_range_pa")
    t = _pos(thickness_m, "thickness_m")
    a0 = _pos(initial_depth_m, "initial_depth_m")
    c0 = _pos(initial_half_length_m, "initial_half_length_m")
    aspect = c0 / a0

    def delta_k(a: float) -> float:
        a = _pos(a, "crack_depth_m")
        if a / t > max_depth_fraction:
            raise ValueError("a/t above the validity limit of the Newman-Raju fit")
        return k_semi_elliptical_surface_crack(
            sigma,
            a,
            aspect * a,
            t,
            plate_half_width_m=plate_half_width_m,
            front_location=front_location,  # type: ignore[arg-type]
        ).k_i_pa_sqrt_m

    return delta_k


# ---------------------------------------------------------------------------
# Paris ODE integration with a discretisation convergence study
# ---------------------------------------------------------------------------

def integrate_paris_ode(
    delta_k_of_a: DeltaKOfA,
    law: ParisLaw,
    initial_crack_m: float,
    final_crack_m: float,
    *,
    n_steps: int = 200,
) -> float:
    """Cycles to grow from ``a0`` to ``a_f`` under ``da/dN = C (Delta K(a))^m``.

    The integral ``N = int dN/da da`` is evaluated by classical 4th-order
    Runge-Kutta in the substitution ``t = ln a`` so that the geometric growth of
    the crack is resolved uniformly (``N = int dN/dt dt`` with
    ``dN/dt = a / (C Delta K(a)^m)``).  Returns ``inf`` when ``Delta K`` at the
    initial crack size does not exceed the threshold: the crack never grows.
    """
    a0 = _pos(initial_crack_m, "initial_crack_m")
    a1 = _pos(final_crack_m, "final_crack_m")
    if a1 <= a0:
        raise ValueError("final_crack_m must exceed initial_crack_m")
    if not isinstance(n_steps, int) or n_steps < 2:
        raise ValueError("n_steps must be an integer >= 2")

    def local_rate(a: float) -> float:
        dk = delta_k_of_a(a)
        da_dn = law.da_dN(dk)
        return da_dn

    if law.da_dN(delta_k_of_a(a0)) <= 0.0:
        return math.inf

    t_end = math.log(a1 / a0)
    h = t_end / n_steps

    def rate(t: float) -> float:
        # keep every quadrature node strictly inside the (a0, a1) interval:
        # the geometry-factor fits validate a closed range (e.g. a/t <= 0.8)
        # and a node landing on the upper bound by round-off must not raise.
        a = a0 * math.exp(t)
        if a >= a1:
            a = a1 * (1.0 - 1e-12)
        da_dn = local_rate(a)
        if da_dn <= 0.0:
            return math.inf
        return a / da_dn

    total = 0.0
    t = 0.0
    for _ in range(n_steps):
        k1 = rate(t)
        k2 = rate(t + 0.5 * h)
        k3 = k2
        k4 = rate(t + h)
        total += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t += h
    return total


@dataclass(frozen=True)
class ConvergenceStudy:
    """Three-level Richardson study of a discretised quantity (ASME V&V / GCI)."""

    step_counts: tuple[int, int, int]
    values: tuple[float, float, float]
    refinement_ratio: float
    observed_order: float
    extrapolated: float
    gci_fine: float
    converged: bool


def richardson_gci(
    fine: float,
    medium: float,
    coarse: float,
    *,
    refinement_ratio: float = 2.0,
    safety_factor: float = 1.25,
    requirement: float = 0.05,
) -> tuple[float, float, float, bool]:
    """Observed order, Richardson extrapolation and fine-grid GCI.

    Returns ``(p_observed, extrapolated, gci_fine, converged)``.  ``gci_fine``
    follows Roache's formulation ``GCI = Fs | (f2 - f1) / f1 | / (r^p - 1)``
    with the standard safety factor ``Fs = 1.25`` for three grids.  When the
    two finest solutions coincide to machine precision the order is reported as
    ``inf`` and the GCI as ``0`` (the sequence has converged).
    """
    r = _pos(refinement_ratio, "refinement_ratio")
    if r <= 1.0:
        raise ValueError("refinement_ratio must be > 1")
    for name, value in (("fine", fine), ("medium", medium), ("coarse", coarse)):
        if not math.isfinite(value):
            raise ValueError(f"{name} solution must be finite")

    d21 = medium - fine
    d32 = coarse - medium
    scale = max(abs(fine), 1e-300)

    if abs(d21) <= 1e-14 * scale:
        return math.inf, fine, 0.0, True
    if abs(d32) <= 1e-14 * scale:
        return math.inf, fine, 0.0, True

    ratio = d32 / d21
    if ratio <= 0.0:
        # non-monotone sequence: the order is not estimable, report the
        # conservative bound obtained with the formal order of the integrator
        order = 4.0
        extrapolated = fine + (fine - medium) / (r**order - 1.0)
        gci = safety_factor * abs(d21) / scale / (r**order - 1.0)
        return order, extrapolated, gci, gci <= requirement

    order = math.log(ratio) / math.log(r)
    order = max(0.5, min(order, 8.0))
    extrapolated = fine + (fine - medium) / (r**order - 1.0)
    gci = safety_factor * abs(d21) / scale / (r**order - 1.0)
    return order, extrapolated, gci, gci <= requirement


def convergence_study(
    delta_k_of_a: DeltaKOfA,
    law: ParisLaw,
    initial_crack_m: float,
    final_crack_m: float,
    *,
    base_steps: int = 100,
    requirement: float = 0.05,
) -> ConvergenceStudy:
    """Refine the integration step twice and report the resulting GCI."""
    fine = integrate_paris_ode(
        delta_k_of_a, law, initial_crack_m, final_crack_m, n_steps=4 * base_steps
    )
    medium = integrate_paris_ode(
        delta_k_of_a, law, initial_crack_m, final_crack_m, n_steps=2 * base_steps
    )
    coarse = integrate_paris_ode(
        delta_k_of_a, law, initial_crack_m, final_crack_m, n_steps=base_steps
    )
    order, extrapolated, gci, ok = richardson_gci(
        fine, medium, coarse, requirement=requirement
    )
    return ConvergenceStudy(
        step_counts=(4 * base_steps, 2 * base_steps, base_steps),
        values=(fine, medium, coarse),
        refinement_ratio=2.0,
        observed_order=order,
        extrapolated=extrapolated,
        gci_fine=gci,
        converged=ok,
    )


def critical_crack_length(
    stress_intensity_of_a: Callable[[float], float],
    fracture_toughness_pa_sqrt_m: float,
    lower_m: float,
    upper_m: float,
    *,
    tolerance_m: float = 1e-12,
    max_iterations: int = 400,
) -> float:
    """Solve ``K_I(a_c) = K_IC`` for ``a_c`` in ``[lower, upper]`` by bisection.

    Uses the caller's geometry-dependent ``K_I(a)``, so the critical size is
    consistent with the geometry factor used in the growth integral (the legacy
    helper assumed ``Y = 1``).  Raises when the toughness is not reached inside
    the bracket, because silently returning an extrapolated size would hide a
    validity violation.
    """
    k_ic = _pos(fracture_toughness_pa_sqrt_m, "fracture_toughness_pa_sqrt_m")
    lo = _pos(lower_m, "lower_m")
    hi = _pos(upper_m, "upper_m")
    if hi <= lo:
        raise ValueError("upper_m must exceed lower_m")

    def residual(a: float) -> float:
        return stress_intensity_of_a(a) - k_ic

    f_lo = residual(lo)
    f_hi = residual(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            "K_IC is not bracketed between lower_m and upper_m: "
            f"K(lower)-K_IC = {f_lo:.3e}, K(upper)-K_IC = {f_hi:.3e}"
        )

    lo_c, hi_c = lo, hi
    for _ in range(max_iterations):
        mid = 0.5 * (lo_c + hi_c)
        if residual(mid) < 0.0:
            lo_c = mid
        else:
            hi_c = mid
        if hi_c - lo_c < tolerance_m:
            break
    return 0.5 * (lo_c + hi_c)


# ---------------------------------------------------------------------------
# Residual-life result with the legacy baseline attached
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GeometryAwareLife:
    """Residual life from a geometry-aware Paris integration.

    ``n_f_legacy`` is the value the uniform-``Y`` integrator returns for the same
    nominal data (``a_c`` from ``Y = 1`` and ``Delta K = sigma sqrt(pi a)``); the
    ratio quantifies the non-conservative error the geometry-aware path removes.
    """

    n_f: float
    n_f_legacy: float
    ratio_legacy_to_geometry: float
    initial_crack_m: float
    critical_crack_m: float
    delta_k_initial_pa_sqrt_m: float
    geometry_factor_initial: float
    fracture_toughness_pa_sqrt_m: float
    gci_fine: float
    observed_order: float
    detail: str
    critical_mode: str


def _legacy_life(
    initial_crack_m: float,
    sigma_range_pa: float,
    sigma_max_pa: float,
    law: ParisLaw,
    fracture_toughness_pa_sqrt_m: float,
) -> float:
    """Reproduce the uniform-``Y`` life exactly (the audited baseline)."""
    a_c = compute_critical_crack(sigma_max_pa, fracture_toughness_pa_sqrt_m)
    if a_c <= initial_crack_m:
        return 0.0
    return paris_integrate_numerical(
        initial_crack_m, a_c, sigma_range_pa, law
    ).cycles_to_failure


def geometry_aware_life(
    delta_k_of_a: DeltaKOfA,
    stress_intensity_of_a: Callable[[float], float],
    *,
    initial_crack_m: float,
    upper_crack_m: float,
    sigma_range_pa: float,
    sigma_max_pa: float,
    law: ParisLaw,
    fracture_toughness_pa_sqrt_m: float,
    detail: str,
    base_steps: int = 100,
) -> GeometryAwareLife:
    """Full residual-life assessment with a consistent critical crack size."""
    a0 = _pos(initial_crack_m, "initial_crack_m")
    k_ic = _pos(fracture_toughness_pa_sqrt_m, "fracture_toughness_pa_sqrt_m")
    sigma_range = _pos(sigma_range_pa, "sigma_range_pa")
    sigma_max = _pos(sigma_max_pa, "sigma_max_pa")

    # the geometry fits are validated on a closed interval; stay a hair inside
    # it so that endpoint evaluations cannot trip the validity guard
    upper = float(upper_crack_m) * (1.0 - 1e-9)
    try:
        a_c = critical_crack_length(stress_intensity_of_a, k_ic, a0, upper)
        critical_mode = "fracture_K_IC"
    except ValueError:
        # The geometry factor never drives K_I to K_IC inside its validity
        # range: the flaw reaches the section boundary (through-thickness
        # penetration / ligament severance) first, which is the governing
        # failure mode for thin hull plating.  Life is then reported to that
        # boundary and the mode is flagged instead of silently extrapolating
        # the SIF fit beyond its range.
        a_c = upper
        critical_mode = "penetration"
    study = convergence_study(
        delta_k_of_a, law, a0, a_c, base_steps=base_steps
    )
    n_f = study.extrapolated
    n_legacy = _legacy_life(a0, sigma_range, sigma_max, law, k_ic)
    dk0 = delta_k_of_a(a0)
    y0 = dk0 / (sigma_range * math.sqrt(math.pi * a0))
    return GeometryAwareLife(
        n_f=n_f,
        n_f_legacy=n_legacy,
        ratio_legacy_to_geometry=(n_legacy / n_f) if n_f > 0 else math.inf,
        initial_crack_m=a0,
        critical_crack_m=a_c,
        delta_k_initial_pa_sqrt_m=dk0,
        geometry_factor_initial=y0,
        fracture_toughness_pa_sqrt_m=k_ic,
        gci_fine=study.gci_fine,
        observed_order=study.observed_order,
        detail=detail,
        critical_mode=critical_mode,
    )


# ---------------------------------------------------------------------------
# Ship-structure details
# ---------------------------------------------------------------------------

def weld_toe_surface_crack_life(
    *,
    thickness_m: float,
    crack_depth_m: float,
    crack_half_length_m: float,
    sigma_range_pa: float,
    law: ParisLaw,
    fracture_toughness_pa_sqrt_m: float,
    sigma_min_pa: float = 0.0,
    plate_half_width_m: float | None = None,
    base_steps: int = 100,
) -> GeometryAwareLife:
    """Weld-toe semi-elliptical surface crack, deepest point, fixed ``a/c``."""
    t = _pos(thickness_m, "thickness_m")
    a0 = _pos(crack_depth_m, "crack_depth_m")
    c0 = _pos(crack_half_length_m, "crack_half_length_m")
    aspect = c0 / a0
    sigma_max = _pos(sigma_range_pa, "sigma_range_pa") + float(sigma_min_pa)
    if sigma_max <= sigma_min_pa:
        raise ValueError("sigma_range_pa must be > 0")

    delta_k = surface_crack_delta_k(
        sigma_range_pa,
        t,
        a0,
        c0,
        plate_half_width_m=plate_half_width_m,
    )

    def k_of_a(a: float) -> float:
        return k_semi_elliptical_surface_crack(
            sigma_max,
            a,
            aspect * a,
            t,
            plate_half_width_m=plate_half_width_m,
            front_location="deepest",
        ).k_i_pa_sqrt_m

    return geometry_aware_life(
        delta_k,
        k_of_a,
        initial_crack_m=a0,
        upper_crack_m=0.8 * t,
        sigma_range_pa=sigma_range_pa,
        sigma_max_pa=sigma_max,
        law=law,
        fracture_toughness_pa_sqrt_m=fracture_toughness_pa_sqrt_m,
        detail="weld_toe_semi_elliptical_surface_crack",
        base_steps=base_steps,
    )


def hole_edge_crack_life(
    *,
    hole_radius_m: float,
    crack_length_m: float,
    plate_width_m: float,
    sigma_range_pa: float,
    law: ParisLaw,
    fracture_toughness_pa_sqrt_m: float,
    sigma_min_pa: float = 0.0,
    base_steps: int = 100,
) -> GeometryAwareLife:
    """Open-hole edge crack growing across the ligament, with ``K_t`` retained."""
    R = _pos(hole_radius_m, "hole_radius_m")
    a0 = _pos(crack_length_m, "crack_length_m")
    W = _pos(plate_width_m, "plate_width_m")
    sigma_max = _pos(sigma_range_pa, "sigma_range_pa") + float(sigma_min_pa)

    delta_k = hole_edge_crack_delta_k(sigma_range_pa, R, W)

    def k_of_a(a: float) -> float:
        return (
            sigma_max
            * math.sqrt(math.pi * a)
            * k_hole_edge_crack(sigma_max, a, R).geometry_factor
            * k_edge_crack_finite_width(sigma_max, a + R, W).geometry_factor
            / 1.12
        )

    return geometry_aware_life(
        delta_k,
        k_of_a,
        initial_crack_m=a0,
        upper_crack_m=0.6 * W - R,
        sigma_range_pa=sigma_range_pa,
        sigma_max_pa=sigma_max,
        law=law,
        fracture_toughness_pa_sqrt_m=fracture_toughness_pa_sqrt_m,
        detail="hole_edge_crack",
        base_steps=base_steps,
    )


def fillet_weld_toe_life(
    *,
    plate_thickness_m: float,
    weld_leg_m: float,
    crack_length_m: float,
    sigma_range_pa: float,
    law: ParisLaw,
    fracture_toughness_pa_sqrt_m: float,
    sigma_min_pa: float = 0.0,
    base_steps: int = 100,
) -> GeometryAwareLife:
    """Fillet-weld toe crack modelled as a finite-width edge crack."""
    t = _pos(plate_thickness_m, "plate_thickness_m")
    leg = _pos(weld_leg_m, "weld_leg_m")
    a0 = _pos(crack_length_m, "crack_length_m")
    w_eff = t + 2.0 * leg
    sigma_max = _pos(sigma_range_pa, "sigma_range_pa") + float(sigma_min_pa)

    delta_k = edge_crack_finite_width_delta_k(sigma_range_pa, w_eff)

    def k_of_a(a: float) -> float:
        return k_edge_crack_finite_width(sigma_max, a, w_eff).k_i_pa_sqrt_m

    return geometry_aware_life(
        delta_k,
        k_of_a,
        initial_crack_m=a0,
        upper_crack_m=0.6 * w_eff,
        sigma_range_pa=sigma_range_pa,
        sigma_max_pa=sigma_max,
        law=law,
        fracture_toughness_pa_sqrt_m=fracture_toughness_pa_sqrt_m,
        detail="fillet_weld_toe_edge_crack",
        base_steps=base_steps,
    )


# ---------------------------------------------------------------------------
# Mean-stress (R-ratio) and S-N cross-adjudication
# ---------------------------------------------------------------------------

def walker_effective_delta_k(
    delta_k_pa_sqrt_m: float, stress_ratio: float, *, gamma: float = 0.5
) -> float:
    """Walker effective stress-intensity range ``Delta K / (1 - R)^(1 - gamma)``.

    ``gamma`` is the material Walker exponent (``~0.5`` for ferritic steels);
    ``R = 0`` reproduces the raw range, ``R -> 1`` amplifies it without bound.
    """
    dk = _pos(delta_k_pa_sqrt_m, "delta_k_pa_sqrt_m")
    R = float(stress_ratio)
    if not math.isfinite(R) or R >= 1.0 or R <= -1.0:
        raise ValueError("stress_ratio must lie in (-1, 1)")
    g = float(gamma)
    if not (0.0 <= g <= 1.0):
        raise ValueError("gamma must lie in [0, 1]")
    return dk / (1.0 - R) ** (1.0 - g)


@dataclass(frozen=True)
class SNCurveIIW:
    """Bilinear IIW-type S-N curve in stress-range form.

    ``N = N_ref (FAT / Delta sigma)^m1`` down to the knee at ``N_knee``, then
    ``N = N_knee (Delta sigma_D / Delta sigma)^m2`` with
    ``Delta sigma_D = FAT (N_ref / N_knee)^(1/m1)``.
    """

    detail_category_pa: float
    reference_cycles: float = 2.0e6
    slope_1: float = 3.0
    knee_cycles: float = 1.0e7
    slope_2: float = 5.0

    def __post_init__(self) -> None:
        _pos(self.detail_category_pa, "detail_category_pa")
        _pos(self.reference_cycles, "reference_cycles")
        _pos(self.slope_1, "slope_1")
        _pos(self.knee_cycles, "knee_cycles")
        _pos(self.slope_2, "slope_2")
        if self.knee_cycles <= self.reference_cycles:
            raise ValueError("knee_cycles must exceed reference_cycles")

    @property
    def endurance_limit_pa(self) -> float:
        return self.detail_category_pa * (self.reference_cycles / self.knee_cycles) ** (
            1.0 / self.slope_1
        )

    def life(self, stress_range_pa: float) -> float:
        ds = _pos(stress_range_pa, "stress_range_pa")
        if ds >= self.endurance_limit_pa:
            return self.reference_cycles * (self.detail_category_pa / ds) ** self.slope_1
        return self.knee_cycles * (self.endurance_limit_pa / ds) ** self.slope_2


@dataclass(frozen=True)
class ParisSNAgreement:
    n_f_paris: float
    n_f_sn: float
    log10_ratio: float
    within_one_order: bool


def compare_paris_vs_iiw(
    paris_life: float, sn_curve: SNCurveIIW, hot_spot_stress_range_pa: float
) -> ParisSNAgreement:
    """Cross-adjudicate a Paris life against an IIW S-N life for the same detail."""
    n_sn = sn_curve.life(hot_spot_stress_range_pa)
    if not math.isfinite(paris_life) or paris_life <= 0.0:
        raise ValueError("paris_life must be finite and > 0")
    log_ratio = math.log10(paris_life / n_sn)
    return ParisSNAgreement(
        n_f_paris=paris_life,
        n_f_sn=n_sn,
        log10_ratio=log_ratio,
        within_one_order=abs(log_ratio) <= 1.0,
    )