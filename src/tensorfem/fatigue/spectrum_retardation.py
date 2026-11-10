"""Load-spectrum residual life with crack-growth retardation (Wheeler model).

Why this module exists
----------------------
``tensorfem.fatigue.crack_growth_geometry`` closed the *geometry* gap of the
Paris-law chain (the legacy residual-life helpers integrated with a constant
``Y = 1``).  Its documented scope statement still excludes the *load-history*
gap:

    "no crack-closure/retardation (no overload, no Wheeler model), no residual
    stress field"

and the only spectrum-capable helper in the legacy library,
``fatigue_crack_propagation.paris_integrate_variable_amplitude``, applies a
single constant geometry factor ``Y`` to every block.  Both omissions matter for
a ship detail under a real hull load spectrum: a constant-amplitude integration
cannot represent the tensile overloads that a welded detail actually sees, so a
residual life coupled to a spectrum with overloads is biased, and the size of
that bias cannot be quantified without a spectrum/retardation capability.

This module supplies it:

* ``LoadBlock`` / ``LoadSpectrum`` -- an ordered block history with stress
  range, cycle count, stress ratio (``sigma_max`` is *derived*, not assumed) and
  an explicit overload flag;
* ``plastic_zone_size`` -- Irwin plane-strain / plane-stress yield-zone radius
  ``r_y = kappa (K_max / sigma_y)**2``;
* ``RetardationZone`` / ``wheeler_factor`` -- Wheeler plastic-zone shielding
  ``phi = (r_y,i / (a_p - a_i))**m`` clipped to ``(0, 1]``, returning exactly
  ``1`` once the tip has grown past the overload elastic boundary
  ``a_p = a_ol + r_y,ol``;
* ``spectrum_residual_life`` -- block-wise RK4 integration of the retarded
  Paris ODE in ``t = ln a`` (the natural variable when the growth rate spans
  decades), with a bracketed bisection for the exact failure cycle inside the
  step that crosses ``a_c``; ``a_c`` comes from ``K_I(a_c) = K_IC`` with the
  *same* geometry factor as the growth integration (never ``Y = 1``);
* ``fit_paris_parameters`` -- inverse identification of ``(C, m)`` from a
  measured ``a(N)`` curve, so a spectrum prediction can be re-anchored on
  inspection data instead of handbook constants;
* ``rayleigh_band_spectrum`` -- narrow-band Rayleigh range discretisation into
  equal-probability bands (band-cycle conservation exact by construction).

Scope and documented simplifications
------------------------------------
LEFM, small-scale yielding, opening mode I, ``Delta K`` above ``Delta K_th``.
The Wheeler zone is created only by cycles flagged as overloads and is never
erased by compressive cycles; compressive cycles carry ``Delta K = 0`` and
therefore grow nothing.  Closure is represented only through Wheeler shielding --
no Elber ``U``, no Willenborg residual-stress field, no crack kinking, no
overload-interaction decay, no ``a/c`` evolution for surface flaws (the initial
aspect ratio is frozen, matching the geometry module).  Blocks are
constant-amplitude within themselves; a cycle-by-cycle history is emulated with
one-cycle blocks.  Because Wheeler has no closure-memory decay, a densely
repeated overload sequence keeps the plastic zone in place by re-creating it at
the current tip every repetition; the model therefore *upper-bounds* the
retardation of a periodic overload sequence (documented and measured in the
qualification report, never presented as a validated closure model).
Stress in Pa, length in m, cycles dimensionless.

Every quantity needed for an audit is returned in a frozen dataclass -- no
hidden print or plot state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..fatigue_crack_propagation import ParisLaw
from .crack_growth_geometry import DeltaKOfA, critical_crack_length

__all__ = [
    "PLANE_STRAIN_ZONE_COEFFICIENT",
    "PLANE_STRESS_ZONE_COEFFICIENT",
    "LoadBlock",
    "LoadSpectrum",
    "ParisFit",
    "RetardationEvent",
    "RetardationZone",
    "SpectrumLifeResult",
    "block_spectrum",
    "fit_paris_parameters",
    "overload_sequence",
    "plastic_zone_size",
    "rayleigh_band_moment",
    "rayleigh_band_spectrum",
    "spectrum_residual_life",
    "wheeler_factor",
]

# Irwin plastic-zone coefficients: r_p = kappa (K / sigma_y)^2.
PLANE_STRAIN_ZONE_COEFFICIENT = 1.0 / (6.0 * math.pi)
PLANE_STRESS_ZONE_COEFFICIENT = 1.0 / (2.0 * math.pi)

# A Wheeler factor this close to unity is round-off, not shielding: the zone is
# re-created at the current tip, so ``r_y,i`` and ``r_y,overload`` are the same
# number and the ratio is 1 up to a few ulp.
_PHI_TOLERANCE = 1e-12


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------


def _real(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _pos(value: float, name: str) -> float:
    value = _real(value, name)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def _nonneg(value: float, name: str) -> float:
    value = _real(value, name)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return value


# ---------------------------------------------------------------------------
# plastic zone and Wheeler shielding
# ---------------------------------------------------------------------------


def plastic_zone_size(
    stress_intensity_pa_sqrt_m: float,
    yield_stress_pa: float,
    *,
    plane_strain: bool = True,
) -> float:
    """Irwin crack-tip plastic-zone radius ``r_p = kappa (K / sigma_y)**2``.

    ``kappa = 1/(6 pi)`` (plane strain) or ``1/(2 pi)`` (plane stress).
    ``K = 0`` returns ``0``: the zero/compressive load limit has no zone.
    """
    k = _nonneg(stress_intensity_pa_sqrt_m, "stress_intensity_pa_sqrt_m")
    sigma_y = _pos(yield_stress_pa, "yield_stress_pa")
    kappa = PLANE_STRAIN_ZONE_COEFFICIENT if plane_strain else PLANE_STRESS_ZONE_COEFFICIENT
    return kappa * (k / sigma_y) ** 2


def wheeler_factor(
    *,
    crack_m: float,
    current_zone_m: float,
    overload_crack_m: float,
    overload_zone_m: float,
    exponent: float,
) -> float:
    """Wheeler retardation factor ``phi`` in ``(0, 1]``.

    ``phi = (r_y,i / (a_p - a_i))**m`` for ``a_i < a_p`` with
    ``a_p = a_overload + r_y,overload``; ``phi = 1`` once the tip has grown past
    ``a_p``.  Clipping at ``1`` keeps the factor a purely retarding multiplier:
    it can never accelerate growth.
    """
    a = _pos(crack_m, "crack_m")
    r_i = _nonneg(current_zone_m, "current_zone_m")
    a_ol = _pos(overload_crack_m, "overload_crack_m")
    r_ol = _nonneg(overload_zone_m, "overload_zone_m")
    m = _pos(exponent, "exponent")

    remaining = (a_ol + r_ol) - a
    if remaining <= 0.0:
        return 1.0
    if r_i <= 0.0:
        return 0.0
    return min(1.0, (r_i / remaining) ** m)


@dataclass(frozen=True)
class RetardationZone:
    """A Wheeler zone created by one overload at ``crack_m``."""

    crack_m: float
    zone_m: float
    stress_intensity_pa_sqrt_m: float
    block_label: str = ""

    def __post_init__(self) -> None:
        _pos(self.crack_m, "crack_m")
        _nonneg(self.zone_m, "zone_m")
        _nonneg(self.stress_intensity_pa_sqrt_m, "stress_intensity_pa_sqrt_m")

    @property
    def elastic_boundary_m(self) -> float:
        """``a_p = a_overload + r_y,overload``: where the shielding dies out."""
        return self.crack_m + self.zone_m

    def factor(self, crack_m: float, current_zone_m: float, exponent: float) -> float:
        return wheeler_factor(
            crack_m=crack_m,
            current_zone_m=current_zone_m,
            overload_crack_m=self.crack_m,
            overload_zone_m=self.zone_m,
            exponent=exponent,
        )


# ---------------------------------------------------------------------------
# load spectrum
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadBlock:
    """A constant-amplitude block of ``cycles`` cycles at ``stress_range_pa``.

    ``stress_ratio`` is ``R = sigma_min / sigma_max``; the absolute maximum
    ``sigma_max = stress_range / (1 - R)`` is derived, never assumed, because the
    Wheeler zone is driven by ``K_max``, not by the range.
    """

    stress_range_pa: float
    cycles: float
    stress_ratio: float = 0.0
    is_overload: bool = False
    label: str = ""

    def __post_init__(self) -> None:
        _pos(self.stress_range_pa, "stress_range_pa")
        _pos(self.cycles, "cycles")
        R = _real(self.stress_ratio, "stress_ratio")
        if not (-1.0 < R < 1.0):
            raise ValueError("stress_ratio must lie in (-1, 1)")

    @property
    def stress_max_pa(self) -> float:
        return self.stress_range_pa / (1.0 - self.stress_ratio)

    @property
    def stress_min_pa(self) -> float:
        return self.stress_max_pa - self.stress_range_pa


@dataclass(frozen=True)
class LoadSpectrum:
    """An ordered, repeatable block load history."""

    blocks: tuple[LoadBlock, ...]
    name: str = ""

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("a spectrum needs at least one block")
        for block in self.blocks:
            if not isinstance(block, LoadBlock):
                raise TypeError("spectrum blocks must be LoadBlock instances")

    @property
    def total_cycles(self) -> float:
        return math.fsum(block.cycles for block in self.blocks)

    @property
    def max_stress_range_pa(self) -> float:
        return max(block.stress_range_pa for block in self.blocks)

    @property
    def max_stress_pa(self) -> float:
        return max(block.stress_max_pa for block in self.blocks)

    @property
    def overload_count(self) -> int:
        return sum(1 for block in self.blocks if block.is_overload)


def block_spectrum(
    stress_ranges_pa: Sequence[float],
    cycles: Sequence[float],
    *,
    stress_ratio: float = 0.0,
    name: str = "",
) -> LoadSpectrum:
    """Build a baseline spectrum from parallel range/cycle sequences."""
    ranges = tuple(stress_ranges_pa)
    counts = tuple(cycles)
    if len(ranges) != len(counts):
        raise ValueError("stress_ranges_pa and cycles must have equal length")
    blocks = tuple(
        LoadBlock(float(ds), float(n), stress_ratio=stress_ratio, label=f"band{i}")
        for i, (ds, n) in enumerate(zip(ranges, counts, strict=True))
    )
    return LoadSpectrum(blocks, name=name or "block_spectrum")


def overload_sequence(
    baseline_range_pa: float,
    baseline_cycles: float,
    *,
    overload_ratio: float,
    overload_cycles: float = 1.0,
    stress_ratio: float = 0.0,
    name: str = "",
) -> LoadSpectrum:
    """``baseline_cycles`` baseline cycles followed by one overload block.

    The overload block carries ``overload_ratio * baseline_range`` at the same
    stress ratio, so ``overload_ratio = 1`` degenerates exactly to the
    constant-amplitude history -- the no-retardation limit used as an anchor.
    """
    ds_b = _pos(baseline_range_pa, "baseline_range_pa")
    n_b = _pos(baseline_cycles, "baseline_cycles")
    olr = _pos(overload_ratio, "overload_ratio")
    if olr < 1.0:
        raise ValueError("overload_ratio must be >= 1")
    n_ol = _pos(overload_cycles, "overload_cycles")
    baseline = LoadBlock(ds_b, n_b, stress_ratio=stress_ratio, label="baseline")
    overload = LoadBlock(
        ds_b * olr,
        n_ol,
        stress_ratio=stress_ratio,
        is_overload=True,
        label=f"overload_olr{olr:g}",
    )
    return LoadSpectrum((baseline, overload), name=name or "baseline_plus_overload")


def _gammp(a: float, x: float) -> float:
    """Regularised lower incomplete gamma ``P(a, x)`` (series / continued fraction).

    Textbook Numerical-Recipes split: converging series for ``x < a + 1`` and a
    modified-Lentz continued fraction for the complement otherwise, both to
    ~1e-15 relative.  Kept local so the module stays dependency-free (stdlib
    only) and importable in constrained environments.
    """
    if x <= 0.0:
        return 0.0
    log_prefactor = a * math.log(x) - x - math.lgamma(a)
    if x < a + 1.0:
        ap = a
        term = 1.0 / a
        total = term
        for _ in range(1000):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-16:
                break
        return total * math.exp(log_prefactor)
    # complement via continued fraction (Lentz)
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return 1.0 - math.exp(log_prefactor) * h


def rayleigh_band_moment(rms_stress_pa: float, n_bands: int, exponent: float) -> tuple[float, ...]:
    """Band representatives ``Delta sigma_rep`` that preserve the ``exponent``-th moment.

    ``sum_k p_k Delta sigma_rep,k**exponent == E[Delta sigma**exponent]`` holds to
    round-off (lower incomplete gamma, not quadrature), so the band set carries
    the exact moment the Paris growth law integrates.
    """
    sigma = _pos(rms_stress_pa, "rms_stress_pa")
    e = _pos(exponent, "exponent")
    if isinstance(n_bands, bool) or not isinstance(n_bands, int) or n_bands < 2:
        raise ValueError("n_bands must be an integer >= 2")
    shape = 0.5 * e + 1.0
    gamma_shape = math.exp(math.lgamma(shape))
    scale = (2.0 * math.sqrt(2.0) * sigma) ** e
    reps = []
    for k in range(n_bands):
        u1 = -math.log1p(-k / n_bands)
        u2 = math.inf if k + 1 >= n_bands else -math.log1p(-(k + 1) / n_bands)
        p_upper = 1.0 if math.isinf(u2) else _gammp(shape, u2)
        p_lower = 0.0 if u1 <= 0.0 else _gammp(shape, u1)
        band_moment = scale * gamma_shape * (p_upper - p_lower)  # unconditional E[s^e 1_band]
        reps.append((n_bands * band_moment) ** (1.0 / e))
    return tuple(reps)


def rayleigh_band_spectrum(
    rms_stress_pa: float,
    total_cycles: float,
    *,
    n_bands: int = 8,
    stress_ratio: float = 0.0,
    moment_exponent: float = 3.0,
    name: str = "",
) -> LoadSpectrum:
    """Discretise a narrow-band Rayleigh range distribution into equal-probability bands.

    For a narrow-band Gaussian process the range is Rayleigh distributed,
    ``F(s) = 1 - exp(-s^2 / (8 sigma_rms^2))``.  The bands are chosen with equal
    probability ``1/n`` and each band is represented by the stress range that
    preserves the ``moment_exponent``-th moment of the band,

        ``Delta sigma_rep = ( n * E[Delta sigma^e 1_band] ) ** (1/e)``,

    which is the moment the Paris growth law actually integrates.  With equal
    probability bands ``E[Delta sigma^e 1_band]`` follows from a truncated
    incomplete gamma integral, evaluated in closed form with the regularised
    lower incomplete gamma ``P(e/2 + 1, u)``.  Consequences: band-cycle conservation is
    exact by construction and, for a pure power-law growth law, the life is
    exact for *any* band count -- the discretisation error only enters through
    the crack-geometry factor, which is why an earlier midpoint (CDF-centre)
    representation needed a 26% band-refinement correction while the
    moment-matched one converges immediately (both measured in
    ``spectrum_retardation_qualification``).
    """
    n_total = _pos(total_cycles, "total_cycles")
    _real(stress_ratio, "stress_ratio")
    reps = rayleigh_band_moment(rms_stress_pa, n_bands, moment_exponent)
    blocks = tuple(
        LoadBlock(ds, n_total / n_bands, stress_ratio=stress_ratio, label=f"band{k}")
        for k, ds in enumerate(reps)
    )
    return LoadSpectrum(blocks, name=name or f"rayleigh_bands_{n_bands}")


# ---------------------------------------------------------------------------
# spectrum residual life
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetardationEvent:
    """A new overload plastic zone created during the integration."""

    crack_m: float
    zone_m: float
    stress_intensity_pa_sqrt_m: float
    cycle: float
    block_label: str


@dataclass(frozen=True)
class SpectrumLifeResult:
    """Residual life (and its audit trail) for one spectrum/detail pair."""

    cycles_to_failure: float
    final_crack_m: float
    critical_crack_m: float
    reached_critical: bool
    critical_mode: str
    spectrum_cycles: float
    spectrum_repetitions: float
    model: str
    min_retardation_factor: float
    retarded_cycles: float
    retardation_events: tuple[RetardationEvent, ...]
    n_baseline_paris: float
    retardation_life_factor: float
    initial_crack_m: float
    yield_stress_pa: float
    detail: str


def _rk4_ln_rate(ln_a: float, delta_n: float, ln_rate: Callable[[float], float]) -> float:
    """One classical RK4 step in ``t = ln a`` across ``delta_n`` cycles."""
    k1 = ln_rate(ln_a)
    k2 = ln_rate(ln_a + 0.5 * delta_n * k1)
    k3 = ln_rate(ln_a + 0.5 * delta_n * k2)
    k4 = ln_rate(ln_a + delta_n * k3)
    return ln_a + delta_n * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def _critical_crack(
    stress_intensity_of_a: Callable[[float], float],
    toughness_pa_sqrt_m: float,
    initial_crack_m: float,
    upper_crack_m: float,
) -> tuple[float, str]:
    upper = float(upper_crack_m) * (1.0 - 1e-9)
    try:
        a_c = critical_crack_length(
            stress_intensity_of_a, toughness_pa_sqrt_m, initial_crack_m, upper
        )
        return a_c, "fracture_K_IC"
    except ValueError:
        # the geometry factor never drives K_I to K_IC inside its validity
        # range: the flaw reaches the section boundary first (penetration)
        return upper, "penetration"


def _make_ln_rate(
    dk_fn: DeltaKOfA,
    kmax_fn: DeltaKOfA,
    *,
    a_cap: float,
    coefficient_c: float,
    exponent_m: float,
    threshold: float,
    zone: RetardationZone | None,
    zone_exponent: float,
    zone_coefficient: float,
    inverse_yield_squared: float,
) -> Callable[[float], float]:
    """Build ``d ln a / dN`` with the (optional) Wheeler factor inlined.

    Kept free of the boundary-validation helpers on purpose: this callable sits
    in the innermost quadrature loop and is evaluated 1e5-1e7 times per
    assessment, so the validation cost would dominate run time.
    """
    if zone is None:

        def ln_rate(ln_x: float) -> float:
            x = math.exp(ln_x)
            if x > a_cap:
                x = a_cap
            dk = dk_fn(x)
            if dk <= threshold:
                return 0.0
            return coefficient_c * dk**exponent_m / x

        return ln_rate

    boundary = zone.crack_m + zone.zone_m

    def ln_rate(ln_x: float) -> float:
        x = math.exp(ln_x)
        if x > a_cap:
            x = a_cap
        dk = dk_fn(x)
        if dk <= threshold:
            return 0.0
        remaining = boundary - x
        if remaining <= 0.0:
            phi = 1.0
        else:
            k_max = kmax_fn(x)
            r_current = zone_coefficient * k_max * k_max * inverse_yield_squared
            ratio = r_current / remaining
            phi = 1.0 if ratio >= 1.0 else ratio**zone_exponent
        return coefficient_c * dk**exponent_m * phi / x

    return ln_rate


def spectrum_residual_life(
    spectrum: LoadSpectrum,
    law: ParisLaw,
    geometry_delta_k: Callable[[float], DeltaKOfA],
    *,
    initial_crack_m: float,
    upper_crack_m: float,
    fracture_toughness_pa_sqrt_m: float,
    yield_stress_pa: float,
    model: str = "wheeler",
    plane_strain: bool = True,
    wheeler_exponent: float | None = None,
    substeps_per_block: int = 8,
    max_repetitions: int = 5_000_000,
    detail: str = "",
    n_baseline_paris: float | None = None,
    critical_stress_pa: float | None = None,
) -> SpectrumLifeResult:
    """Residual life from ``initial_crack_m`` to ``a_c`` under ``spectrum``.

    ``geometry_delta_k(sigma_range)`` must return ``Delta K(a)`` for that range.
    The same factory is evaluated at each block's ``sigma_max`` for ``K_max``,
    which keeps the critical-crack root solve and the growth integral on one
    geometry definition.  ``model='none'`` disables shielding and recovers plain
    geometry-aware Paris growth (the consistency anchor).

    ``n_baseline_paris`` optionally supplies the unretarded reference life (for
    example from ``crack_growth_geometry.geometry_aware_life``) so that
    ``retardation_life_factor`` is measured against an independent baseline
    instead of the module's own ``model='none'`` run.

    ``critical_stress_pa`` pins the absolute ``sigma_max`` used for the
    ``K_I(a_c) = K_IC`` check.  It defaults to the largest block stress, which is
    correct for a block history with explicit peaks; for a *distribution*
    spectrum (e.g. a moment-matched Rayleigh band set) the largest band range is
    a quadrature representative rather than a physical peak, so a band-refinement
    study must pin the physical record peak here -- otherwise the band count
    moves ``a_c`` as well as the growth integral and the two effects cannot be
    separated.
    """
    a0 = _pos(initial_crack_m, "initial_crack_m")
    k_ic = _pos(fracture_toughness_pa_sqrt_m, "fracture_toughness_pa_sqrt_m")
    sigma_y = _pos(yield_stress_pa, "yield_stress_pa")
    if model not in ("wheeler", "none"):
        raise ValueError("model must be 'wheeler' or 'none'")
    if isinstance(substeps_per_block, bool) or not isinstance(substeps_per_block, int):
        raise TypeError("substeps_per_block must be an int")
    if substeps_per_block < 1:
        raise ValueError("substeps_per_block must be >= 1")
    if isinstance(max_repetitions, bool) or not isinstance(max_repetitions, int):
        raise TypeError("max_repetitions must be an int")
    m_ret = (
        law.exponent_m if wheeler_exponent is None else _pos(wheeler_exponent, "wheeler_exponent")
    )

    dk_cache: dict[float, DeltaKOfA] = {}

    def dk_for(stress: float) -> DeltaKOfA:
        key = float(stress)
        fn = dk_cache.get(key)
        if fn is None:
            fn = geometry_delta_k(key)
            dk_cache[key] = fn
        return fn

    critical_stress = (
        spectrum.max_stress_pa
        if critical_stress_pa is None
        else _pos(critical_stress_pa, "critical_stress_pa")
    )
    a_c, critical_mode = _critical_crack(dk_for(critical_stress), k_ic, a0, upper_crack_m)
    if a_c <= a0:
        raise ValueError("initial_crack_m must be below the critical crack size")
    a_cap = a_c * (1.0 - 1e-12)

    coefficient_c = law.coefficient_c
    exponent_m = law.exponent_m
    threshold = law.threshold_dk
    zone_coefficient = (
        PLANE_STRAIN_ZONE_COEFFICIENT if plane_strain else PLANE_STRESS_ZONE_COEFFICIENT
    )
    inverse_yield_squared = 1.0 / (sigma_y * sigma_y)
    retard = model == "wheeler"

    # block descriptors are constant across repetitions: build once
    descriptors = []
    for block in spectrum.blocks:
        dk_fn = dk_for(block.stress_range_pa)
        kmax_fn = dk_for(block.stress_max_pa)
        descriptors.append(
            (
                block,
                dk_fn,
                kmax_fn,
                block.cycles / substeps_per_block,
                True,
            )
        )

    a = a0
    n_total = 0.0
    zone: RetardationZone | None = None
    rates: list[Callable[[float], float]] = []
    zone_signature: tuple[float, float] | None = None
    min_phi = 1.0
    retarded_cycles = 0.0
    events: list[RetardationEvent] = []
    max_events = 64
    reached = False
    repetitions = 0

    def rebuild_rates() -> None:
        rates.clear()
        for block, dk_fn, kmax_fn, _dn, _active in descriptors:
            rates.append(
                _make_ln_rate(
                    dk_fn,
                    kmax_fn,
                    a_cap=a_cap,
                    coefficient_c=coefficient_c,
                    exponent_m=exponent_m,
                    threshold=threshold,
                    zone=zone,
                    zone_exponent=m_ret,
                    zone_coefficient=zone_coefficient,
                    inverse_yield_squared=inverse_yield_squared,
                )
            )

    while repetitions < max_repetitions and not reached:
        a_before = a
        for index, (block, dk_fn, kmax_fn, dn, active) in enumerate(descriptors):
            if reached or not active:
                continue
            if retard and block.is_overload:
                k_now = kmax_fn(a)
                if zone is None or k_now > zone.stress_intensity_pa_sqrt_m * (1.0 + 1e-9):
                    zone = RetardationZone(
                        crack_m=a,
                        zone_m=zone_coefficient * k_now * k_now * inverse_yield_squared,
                        stress_intensity_pa_sqrt_m=k_now,
                        block_label=block.label,
                    )
                    zone_signature = None
                    if len(events) < max_events:
                        events.append(
                            RetardationEvent(
                                crack_m=a,
                                zone_m=zone.zone_m,
                                stress_intensity_pa_sqrt_m=k_now,
                                cycle=n_total,
                                block_label=block.label,
                            )
                        )

            signature = None if zone is None else (zone.crack_m, zone.zone_m)
            if not rates or signature != zone_signature:
                rebuild_rates()
                zone_signature = signature

            ln_rate = rates[index]
            for _ in range(substeps_per_block):
                ln_a = math.log(a)
                if retard and zone is not None:
                    phi_here = zone.factor(
                        a,
                        zone_coefficient * kmax_fn(a) ** 2 * inverse_yield_squared,
                        m_ret,
                    )
                else:
                    phi_here = 1.0
                if phi_here < min_phi:
                    min_phi = phi_here

                a_new = math.exp(_rk4_ln_rate(ln_a, dn, ln_rate))
                if a_new >= a_c:
                    lo, hi = 0.0, dn
                    for _ in range(80):
                        mid = 0.5 * (lo + hi)
                        if math.exp(_rk4_ln_rate(ln_a, mid, ln_rate)) >= a_c:
                            hi = mid
                        else:
                            lo = mid
                    n_total += hi
                    if phi_here < 1.0 - _PHI_TOLERANCE:
                        retarded_cycles += hi
                    a = a_c
                    reached = True
                    break

                n_total += dn
                a = a_new
                if phi_here < 1.0 - _PHI_TOLERANCE:
                    retarded_cycles += dn

        repetitions += 1
        if not reached and (a - a_before) <= 1e-15 * a:
            break

    cycles = n_total if reached else math.inf
    final = a

    baseline = n_baseline_paris
    if baseline is None:
        baseline = cycles
    factor = (cycles / baseline) if (math.isfinite(cycles) and baseline > 0) else math.inf

    return SpectrumLifeResult(
        cycles_to_failure=cycles,
        final_crack_m=final,
        critical_crack_m=a_c,
        reached_critical=reached,
        critical_mode=critical_mode,
        spectrum_cycles=spectrum.total_cycles,
        spectrum_repetitions=float(repetitions),
        model=model,
        min_retardation_factor=min_phi,
        retarded_cycles=retarded_cycles,
        retardation_events=tuple(events),
        n_baseline_paris=baseline,
        retardation_life_factor=factor,
        initial_crack_m=a0,
        yield_stress_pa=sigma_y,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# inverse identification of Paris constants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParisFit:
    coefficient_c: float
    exponent_m: float
    intercept: float
    r_squared: float
    n_points: int


def fit_paris_parameters(
    crack_m: Sequence[float],
    cycles: Sequence[float],
    delta_k_of_a: DeltaKOfA,
    *,
    min_delta_k_pa_sqrt_m: float = 0.0,
) -> ParisFit:
    """Least-squares identification of ``(C, m)`` from an ``a(N)`` curve.

    Local growth rates are central differences of the measured ``(a, N)`` pairs;
    ``Delta K`` is evaluated at the midpoint crack length.  The regression is
    linear in ``(ln Delta K, ln da/dN)``, so ``m`` is the slope and
    ``C = exp(intercept)``.  ``r_squared`` is reported so that a non-Paris
    dataset cannot be fitted silently.
    """
    a = [float(x) for x in crack_m]
    n = [float(x) for x in cycles]
    if len(a) != len(n):
        raise ValueError("crack_m and cycles must have equal length")
    if len(a) < 3:
        raise ValueError("need at least three (a, N) points for a central-difference fit")

    xs: list[float] = []
    ys: list[float] = []
    for i in range(1, len(a) - 1):
        da = a[i + 1] - a[i - 1]
        dn = n[i + 1] - n[i - 1]
        if dn <= 0.0 or da <= 0.0:
            continue
        dk = delta_k_of_a(a[i])
        if dk <= min_delta_k_pa_sqrt_m:
            continue
        xs.append(math.log(dk))
        ys.append(math.log(da / dn))

    if len(xs) < 2:
        raise ValueError("not enough valid growth-rate points to fit Paris parameters")

    mx = math.fsum(xs) / len(xs)
    my = math.fsum(ys) / len(ys)
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = math.fsum((y - my) ** 2 for y in ys)
    ss_res = math.fsum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys, strict=True))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    return ParisFit(
        coefficient_c=math.exp(intercept),
        exponent_m=slope,
        intercept=intercept,
        r_squared=r2,
        n_points=len(xs),
    )
