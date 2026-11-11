"""Ice-belt plate/frame ratcheting: IACS UR I PC ice-pressure spectrum, Bree
shakedown anchor and accumulated permanent deflection (P226W3-N05 / m14467).

Motivation
----------
:func:`tensorfem.stiffened_panel_ultimate.iso19906_ice_belt_check` answers a
single, *monotonic* question: does the ice-belt panel survive one design ice
pressure (elastic screening / ULS)?  It says nothing about what repeated ice
ramming does to the same panel -- the failure mode that governs an ice-classed
shell in service.  This module adds the missing *cyclic* layer:

1. **IACS UR I PC ice-pressure spectrum** (:class:`IcePressureModel`,
   :func:`ice_pressure`, :func:`ice_collision_spectrum`): the pressure-area
   relation of the Polar Class local ice load model,

   .. math::  p(A) = C_d C_f p_{ref} (A/A_{ref})^{-alpha},

   converted into a *block cycle spectrum* (pressure amplitude vs. number of
   impacts) from the voyage ice-collision statistics (impacts per voyage,
   Weibull-distributed amplitudes truncated at the design pressure).

   **Data provenance / honesty note:** the *structure* of the pressure-area
   law (power law, pressure decreasing with loaded area), the area exponent
   ``alpha = 0.3`` and the ``C_d``/``C_f`` multiplicative structure follow the
   IACS Polar Class local-load formulation; the *numerical* class pressure
   ``p_ref`` and the collision statistics are **inputs**.  No class constant is
   hard-coded as if it were certified: :func:`ice_class_pressure_table`
   returns an explicitly labelled ``ILLUSTRATIVE`` dataset that must be
   replaced by the values of the governing IACS UR I2 table for a real design
   (:func:`check_pressure_provenance` fails closed on illustrative data when
   ``require_design_data=True``).

2. **Bree shakedown anchor** (:func:`bree_cycles`, :func:`bree_region`,
   :func:`melan_shakedown`): the classical Bree (1967) plate problem -- a plate
   under a *steady primary membrane stress* ``sigma_p`` plus a *cyclic
   self-equilibrated through-thickness* stress of range ``Delta sigma_t`` --
   solved with two independent methods:

   * a **real incremental solver**: elastic-perfectly-plastic (EPP) fibres
     through the thickness with *restrained bending* (Bree's plane-section
     idealisation, ``eps(z) = e`` uniform, as for a thin tube wall or a
     stiffener-restrained panel), membrane equilibrium
     ``int sigma dz = 2 sigma_p`` enforced by an active-set Newton iteration
     followed by an exact 1-D return map.  The region (elastic / shakedown /
     plastic cycling / ratcheting) is thus an **output** of the mechanics, not
     an assumed formula;

   * the **Melan static (lower-bound) shakedown theorem** as an analytic
     linear-programming anchor (:func:`melan_shakedown`): find a
     time-independent residual field ``rho(z)`` with zero membrane
     (``int rho dz = 0``) such that
     ``-1 <= x + lambda y z + rho(z) <= 1`` for every load factor ``lambda`` in
     ``[0, 1]`` (all stresses normalised by the yield stress).

   Region labels: ``E`` elastic, ``S`` elastic shakedown (plasticity only in the
   first cycles), ``P`` plastic cycling (alternating plasticity, zero net
   growth), ``R`` ratcheting (non-zero membrane growth per cycle), ``COLLAPSE``
   (``x >= 1``: the primary load alone exhausts the section).

   The two methods are cross-checked; the Melan LP is a *sufficient*
   (lower-bound) condition, so the machine-checked criterion is that the
   incremental solver never shakes down where the LP is infeasible, and the two
   boundaries agree where the LP is tight.  The elastic limit is exact and
   analytic: ``x + y = 1``.

3. **Panel-level accumulated deflection** (:func:`ice_belt_deflection_history`,
   :func:`ice_belt_service_life`): the section-level growth of (2) is fed back
   into the panel through the von Karman membrane kinematics
   ``eps_m = (pi^2/4)(w0/L)^2``, i.e. ``w0 = (2L/pi) sqrt(eps_m)``, so the
   growing plastic membrane strain stiffens the panel (membrane action) while
   the ice pressure spectrum keeps driving it.  Output: deflection vs. cycle
   curve, permanent deflection limit (default ``L/150``) and cycles-to-limit.

Scope and limitations (explicit, do not over-claim)
---------------------------------------------------
* EPP material: no isotropic/kinematic hardening, so the *saturation* of
  ratcheting by hardening is not represented (hardening reduces, never
  increases, the growth; results are conservative in the growth direction).
* Plane-stress 1-D fibres through the thickness (Bree's idealisation): no
  two-way plate bending, no shear, no buckling, no weld residual field, no
  low-cycle-fatigue damage accumulation, no crack growth.
* The deflection feedback uses the von Karman first-order membrane relation and
  a fully restrained membrane; a real panel with flexible frames gives less
  membrane action and therefore *larger* deflection.
* Ice-pressure values are *inputs* (see the provenance note above); results are
  only as good as the supplied ``p_ref`` / spectrum.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "IcePressureModel",
    "IceCycleBin",
    "IceCollisionSpectrum",
    "illustrative_class_pressure",
    "ice_class_pressure_table",
    "check_pressure_provenance",
    "ice_pressure",
    "ice_collision_spectrum",
    "BreeCycleResult",
    "BreeRegionMap",
    "bree_cycles",
    "bree_region",
    "bree_region_map",
    "melan_shakedown",
    "melan_boundary",
    "analytic_elastic_x",
    "analytic_shakedown_x",
    "IceBeltPlate",
    "DeflectionHistory",
    "membrane_relief_pressure",
    "first_cycle_plastic_deflection",
    "ServiceLifeResult",
    "bending_stress_from_pressure",
    "membrane_deflection",
    "deflection_to_membrane_strain",
    "ice_belt_deflection_history",
    "ice_belt_service_life",
    "spectrum_total_cycles",
]

# --------------------------------------------------------------------------- #
# 1. IACS UR I PC ice pressure -> cycle spectrum
# --------------------------------------------------------------------------- #

#: Relative Polar Class pressure factors of the *illustrative* dataset.  The
#: ordering (PC1 = heaviest ice, PC7 = lightest) is the physical content; the
#: absolute numbers are placeholders (see the module docstring).
_ILLUSTRATIVE_CLASS_FACTOR = {1: 1.00, 2: 0.88, 3: 0.77, 4: 0.66,
                              5: 0.55, 6: 0.44, 7: 0.33}


@dataclass(frozen=True)
class IcePressureModel:
    """IACS-style local ice pressure ``p(A) = C_d C_f p_ref (A/A_ref)^-alpha``.

    ``p_ref`` is the average pressure on the reference area ``A_ref``; ``c_d``
    (displacement / engine-power influence) and ``c_f`` (frame / hull-form
    influence) are the multiplicative factors of the Polar Class local-load
    formulation; ``alpha`` is the pressure-area exponent.  ``provenance`` is
    free text and is gated by :func:`check_pressure_provenance`.
    """

    p_ref_mpa: float
    area_ref_m2: float = 1.0
    area_exponent: float = 0.3
    c_d: float = 1.0
    c_f: float = 1.0
    provenance: str = "user"

    def __post_init__(self) -> None:
        if not self.p_ref_mpa > 0.0:
            raise ValueError("p_ref_mpa must be positive")
        if not self.area_ref_m2 > 0.0:
            raise ValueError("area_ref_m2 must be positive")
        if not (0.0 <= self.area_exponent < 1.0):
            raise ValueError("area_exponent must lie in [0, 1)")
        if self.c_d <= 0.0 or self.c_f <= 0.0:
            raise ValueError("c_d and c_f must be positive")

    def pressure(self, area_m2) -> np.ndarray:
        a = np.asarray(area_m2, dtype=float)
        if np.any(a <= 0.0):
            raise ValueError("loaded area must be positive")
        return (self.c_d * self.c_f * self.p_ref_mpa
                * (a / self.area_ref_m2) ** (-self.area_exponent))


def ice_pressure(area_m2, p_ref_mpa: float, *, area_ref_m2: float = 1.0,
                 area_exponent: float = 0.3, c_d: float = 1.0,
                 c_f: float = 1.0) -> np.ndarray:
    """Convenience wrapper around :class:`IcePressureModel`."""
    return IcePressureModel(p_ref_mpa=p_ref_mpa, area_ref_m2=area_ref_m2,
                            area_exponent=area_exponent, c_d=c_d,
                            c_f=c_f).pressure(area_m2)


def illustrative_class_pressure(polar_class: int, p_ref_pc1_mpa: float = 1.0) -> float:
    """Illustrative class pressure ``p_ref`` for ``polar_class`` (NOT for design)."""
    if polar_class not in _ILLUSTRATIVE_CLASS_FACTOR:
        raise ValueError("polar_class must be an integer in 1..7")
    if p_ref_pc1_mpa <= 0.0:
        raise ValueError("p_ref_pc1_mpa must be positive")
    return float(_ILLUSTRATIVE_CLASS_FACTOR[polar_class] * p_ref_pc1_mpa)


def ice_class_pressure_table(p_ref_pc1_mpa: float = 1.0) -> dict:
    """Illustrative PC1..PC7 pressure table (``provenance='ILLUSTRATIVE'``)."""
    return {
        "provenance": "ILLUSTRATIVE",
        "note": ("replace with the governing IACS UR I2 / class-society table "
                 "values before any design use"),
        "p_ref_mpa": {pc: illustrative_class_pressure(pc, p_ref_pc1_mpa)
                      for pc in sorted(_ILLUSTRATIVE_CLASS_FACTOR)},
    }


def check_pressure_provenance(model: IcePressureModel,
                             require_design_data: bool = False) -> None:
    """Fail closed when illustrative (non-design) pressure data is used.

    With ``require_design_data=True`` a model whose ``provenance`` starts with
    ``ILLUSTRATIVE`` raises ``ValueError``; this is the machine gate that keeps
    demonstration numbers out of design claims.
    """
    if require_design_data and str(model.provenance).upper().startswith(
            "ILLUSTRATIVE"):
        raise ValueError(
            "ice pressure provenance is ILLUSTRATIVE; supply design data from "
            "the governing IACS UR I2 table (provenance='IACS-UR-I2 <rev>')")


@dataclass(frozen=True)
class IceCycleBin:
    """One block of ``count`` ice impacts of amplitude ``pressure_mpa``."""

    pressure_mpa: float
    count: int

    def __post_init__(self) -> None:
        if self.pressure_mpa < 0.0:
            raise ValueError("pressure_mpa must be non-negative")
        if self.count < 0:
            raise ValueError("count must be non-negative")


@dataclass(frozen=True)
class IceCollisionSpectrum:
    """Block cycle spectrum of ice impacts (design-pressure normalised)."""

    bins: tuple[IceCycleBin, ...]
    design_pressure_mpa: float
    loaded_area_m2: float
    weibull_shape: float
    provenance: str = "user"

    def __iter__(self):
        return iter(self.bins)

    def __len__(self) -> int:
        return len(self.bins)

    @property
    def total_cycles(self) -> int:
        return int(sum(b.count for b in self.bins))

    def amplitudes(self) -> np.ndarray:
        return np.array([b.pressure_mpa for b in self.bins], dtype=float)

    def counts(self) -> np.ndarray:
        return np.array([b.count for b in self.bins], dtype=int)

    def max_pressure(self) -> float:
        return float(self.amplitudes().max()) if self.bins else 0.0

    def scaled(self, factor: float) -> "IceCollisionSpectrum":
        """Scale all amplitudes (e.g. for another polar class)."""
        if factor < 0.0:
            raise ValueError("factor must be non-negative")
        return replace(self,
                       bins=tuple(IceCycleBin(b.pressure_mpa * factor, b.count)
                                  for b in self.bins),
                       design_pressure_mpa=self.design_pressure_mpa * factor)

    def bin_by_pressure(self, n_bins: int) -> "IceCollisionSpectrum":
        """Re-group the blocks into ``n_bins`` equal-count pressure bins.

        The representative amplitude is the count-weighted mean pressure of
        each group, the standard block-counting idealisation.
        """
        if n_bins <= 0:
            raise ValueError("n_bins must be positive")
        cycles = np.repeat(self.amplitudes(), self.counts())
        if cycles.size == 0:
            return replace(self, bins=())
        order = np.argsort(cycles)
        groups = np.array_split(cycles[order], min(int(n_bins), cycles.size))
        bins = []
        for g in groups:
            if g.size == 0:
                continue
            bins.append(IceCycleBin(pressure_mpa=float(g.mean()),
                                    count=int(g.size)))
        bins.sort(key=lambda b: -b.pressure_mpa)
        return replace(self, bins=tuple(bins))


def ice_collision_spectrum(design_pressure_mpa: float, *,
                           impacts_per_voyage: float, voyages: int = 1,
                           loaded_area_m2: float = 1.0,
                           weibull_shape: float = 1.0, n_bins: int = 10,
                           max_amplitude_ratio: float = 1.0,
                           provenance: str = "user") -> IceCollisionSpectrum:
    """Ice impact block spectrum from the voyage collision statistics.

    Ice-impact amplitudes are idealised as Weibull distributed,
    ``F(z) = 1 - exp(-(z/lambda)^k)`` on the design-normalised amplitude
    ``z = p/p_design``, truncated at ``max_amplitude_ratio`` (default: the design
    pressure itself).  ``impacts_per_voyage`` sets the expected number of
    collisions; the total count is ``round(impacts_per_voyage * voyages)``.
    ``lambda`` is calibrated so that the mean of the truncated distribution is
    ``0.5 * max_amplitude_ratio`` (documented neutral choice: the design
    pressure is reached by the rarest impacts, the bulk of the spectrum sits
    near half of it).

    Blocks are produced by equal-*probability* quantile slicing, which preserves
    the Weibull statistics exactly (each bin holds the same number of impacts).
    """
    if design_pressure_mpa <= 0.0:
        raise ValueError("design_pressure_mpa must be positive")
    if impacts_per_voyage <= 0.0:
        raise ValueError("impacts_per_voyage must be positive")
    if voyages <= 0:
        raise ValueError("voyages must be positive")
    if weibull_shape <= 0.0:
        raise ValueError("weibull_shape must be positive")
    if not (0.0 < max_amplitude_ratio <= 1.0):
        raise ValueError("max_amplitude_ratio must lie in (0, 1]")
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    k = float(weibull_shape)
    z_max = float(max_amplitude_ratio)
    target = 0.5 * z_max

    def truncated_mean(lam: float) -> float:
        s = (z_max / lam) ** k
        p_le = _lower_incomplete_gamma_regularised(1.0 + 1.0 / k, s)
        lower = lam * math.gamma(1.0 + 1.0 / k) * p_le
        p_cap = 1.0 - math.exp(-s)
        return lower + z_max * (1.0 - p_cap)

    lo, hi = 1e-4 * z_max, 50.0 * max(z_max, 1.0)
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if truncated_mean(mid) < target:
            lo = mid
        else:
            hi = mid
    lam = 0.5 * (lo + hi)

    n_impacts = int(round(impacts_per_voyage * voyages))
    bins: list[IceCycleBin] = []
    if n_impacts > 0:
        p_cap = 1.0 - math.exp(-(z_max / lam) ** k)
        counts = _split_counts(n_impacts, n_bins)
        probs = np.linspace(0.0, 1.0, n_bins + 1)
        # 8-node Gauss-Legendre rule inside each probability block: the
        # representative amplitude is the *conditional mean* of the block, not
        # its quantile midpoint (the latter would bias the spectrum low on a
        # heavy-tailed distribution).
        nodes, node_w = np.polynomial.legendre.leggauss(8)
        for c, pa, pb in zip(counts, probs[:-1], probs[1:]):
            if c == 0:
                continue
            half = 0.5 * (pb - pa)
            mid = 0.5 * (pb + pa)
            u = mid + half * nodes
            # quantile of the *truncated* variable: z = F^-1(u) clipped at the
            # cap, so the probability atom at z_max (mass 1 - p_cap) is included
            z_nodes = lam * np.power(
                -np.log(np.maximum(1e-300, 1.0 - u)), 1.0 / k)
            z = float(np.sum(node_w * np.minimum(z_nodes, z_max))
                      / np.sum(node_w))
            bins.append(IceCycleBin(pressure_mpa=float(z * design_pressure_mpa),
                                    count=int(c)))
    bins.sort(key=lambda b: -b.pressure_mpa)
    return IceCollisionSpectrum(bins=tuple(bins),
                                design_pressure_mpa=float(design_pressure_mpa),
                                loaded_area_m2=float(loaded_area_m2),
                                weibull_shape=k, provenance=provenance)


def _split_counts(total: int, parts: int) -> list[int]:
    parts = max(1, int(parts))
    base, rem = divmod(int(total), parts)
    return [base + (1 if i < rem else 0) for i in range(parts)]


def _lower_incomplete_gamma_regularised(a: float, x: float) -> float:
    """Regularised lower incomplete gamma ``P(a, x)`` (series / continued fraction)."""
    if x <= 0.0:
        return 0.0
    if x < a + 1.0:
        term = 1.0 / a
        total = term
        for n in range(1, 500):
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return total * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
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
        if abs(delta - 1.0) < 1e-15:
            break
    q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
    return 1.0 - q


def spectrum_total_cycles(spectra: Iterable[IceCollisionSpectrum]) -> int:
    """Total number of cycles of several spectra."""
    return int(sum(s.total_cycles for s in spectra))


# --------------------------------------------------------------------------- #
# 2. Bree shakedown problem: incremental EPP plate + Melan LP anchor
# --------------------------------------------------------------------------- #
# Normalisation: sigma_hat = sigma/sigma_y, eps_hat = E eps/sigma_y, through
# thickness z in [-1, 1], x = sigma_p/sigma_y (steady primary membrane),
# y = Delta sigma_t/sigma_y (range of the self-equilibrated linear stress).


@dataclass(frozen=True)
class BreeCycleResult:
    """Steady cyclic response of the Bree problem for one ``(x, y)`` pair."""

    x: float
    y: float
    region: str
    growth_per_cycle: float
    plastic_range: float
    first_cycle_growth: float
    growth_history: tuple[float, ...]
    cycles_run: int
    plastic_strain_peak: float = 0.0

    @property
    def is_ratcheting(self) -> bool:
        return self.region == "R"

    @property
    def shakes_down(self) -> bool:
        return self.region in ("E", "S")


def _thickness_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Through-thickness fibres and weights on ``[-1, 1]``.

    Composite trapezoidal rule **including the two surfaces** ``z = +-1``, which
    is essential: the cyclic thermal stress is extreme at the surfaces, so a
    quadrature whose nodes are all interior (e.g. Gauss-Legendre) would never
    see the surface yielding and would over-predict shakedown.
    """
    if n < 5:
        raise ValueError("at least 5 through-thickness fibres are required")
    z = np.linspace(-1.0, 1.0, int(n))
    h = 2.0 / (n - 1)
    w = np.full(int(n), h)
    w[0] = 0.5 * h
    w[-1] = 0.5 * h
    return z, w


def _return_map(sigma_trial: np.ndarray):
    """1-D EPP return map -> ``(sigma, d_eps_p, elastic_mask)``."""
    elastic = np.abs(sigma_trial) <= 1.0
    sigma = np.clip(sigma_trial, -1.0, 1.0)
    return sigma, sigma_trial - sigma, elastic


def _solve_increment(x: float, lam: float, y: float, z: np.ndarray,
                     w: np.ndarray, eps_p: np.ndarray, tol: float = 1e-13,
                     max_iter: int = 400):
    """Solve membrane equilibrium at load factor ``lam`` (bending restrained).

    Bree's idealisation: the through-thickness total strain is **uniform**
    (``kappa = 0``, plane sections / restrained bending, as for the wall of a
    thin tube or a stiffener-restrained ice-belt panel), so there is a single
    global unknown, the membrane strain ``e``, fixed by
    ``(1/2) int sigma dz = x``.  The fibre response is exact 1-D EPP with the
    algorithmic tangent ``H = 1`` (elastic) / ``0`` (plastic), i.e. an
    active-set Newton iteration.
    """
    e = x + 0.5 * float(np.sum(w * eps_p))      # elastic predictor
    for _ in range(max_iter):
        sigma_trial = e - eps_p - lam * y * z
        sigma, _, elastic = _return_map(sigma_trial)
        r = 0.5 * float(np.sum(w * sigma)) - x
        if abs(r) < tol:
            break
        stiffness = 0.5 * float(np.sum(w * elastic.astype(float)))
        if stiffness <= 1e-14:
            raise FloatingPointError(
                "Bree section has no elastic fibre left (plastic mechanism): "
                f"x={x}, y={y}, lam={lam}")
        e -= r / stiffness
    else:  # pragma: no cover - defensive
        raise RuntimeError(
            f"Bree increment did not converge: x={x}, y={y}, lam={lam}")
    sigma_trial = e - eps_p - lam * y * z
    sigma, d_eps, _ = _return_map(sigma_trial)
    return e, 0.0, eps_p + d_eps, sigma


def bree_cycles(x: float, y: float, *, cycles: int = 12, steps: int = 8,
                fibres: int = 40, growth_tol: float = 1e-9,
                plastic_tol: float = 1e-10) -> BreeCycleResult:
    """Run the Bree problem for up to ``cycles`` full load cycles.

    Load path per cycle: ``lambda: 0 -> 1 -> 0`` with ``steps`` increments each
    way.  ``x = sigma_p/sigma_y`` is the steady primary membrane stress and
    ``y = Delta sigma_t/sigma_y`` the range of the cyclic self-equilibrated
    through-thickness stress.
    """
    if x < 0.0:
        raise ValueError("x must be non-negative")
    if y < 0.0:
        raise ValueError("y must be non-negative")
    if cycles < 2:
        raise ValueError("at least 2 cycles are needed to measure a rate")
    if steps < 2:
        raise ValueError("at least 2 steps per half cycle are needed")
    z, w = _thickness_rule(fibres)
    eps_p = np.zeros_like(z)
    e = x + 0.5 * float(np.sum(w * eps_p))
    kappa = 0.0
    growth: list[float] = []
    max_plastic_inc = 0.0
    first_growth = 0.0
    collapsed = False
    for cyc in range(cycles):
        e_start = e
        cycle_plastic = 0.0
        try:
            for k in range(1, steps + 1):
                lam = k / steps
                e, kappa, eps_new, sigma = _solve_increment(x, lam, y, z, w, eps_p)
                cycle_plastic = max(cycle_plastic,
                                    float(np.max(np.abs(eps_new - eps_p))))
                eps_p = eps_new
            for k in range(steps - 1, -1, -1):
                lam = k / steps
                e, kappa, eps_new, sigma = _solve_increment(x, lam, y, z, w, eps_p)
                cycle_plastic = max(cycle_plastic,
                                    float(np.max(np.abs(eps_new - eps_p))))
                eps_p = eps_new
        except FloatingPointError:
            collapsed = True
            break
        g = e - e_start
        growth.append(g)
        if cyc == 0:
            first_growth = g
        max_plastic_inc = max(max_plastic_inc, cycle_plastic)
        if cyc >= 3 and abs(growth[-1]) <= growth_tol and \
                abs(growth[-1] - growth[-2]) <= growth_tol:
            break

    if collapsed or x >= 1.0:
        return BreeCycleResult(x=float(x), y=float(y), region="COLLAPSE",
                               growth_per_cycle=float("inf"),
                               plastic_range=float("inf"),
                               first_cycle_growth=float("inf"),
                               growth_history=tuple(float(g) for g in growth),
                               cycles_run=len(growth),
                               plastic_strain_peak=float(np.max(np.abs(eps_p))))

    steady_growth = float(np.mean(growth[-2:])) if len(growth) >= 2 else growth[-1]
    plastic_range = _plastic_strain_range(x, y, z, w, eps_p, steps)
    if max_plastic_inc <= plastic_tol:
        region = "E"
    elif abs(steady_growth) > growth_tol:
        region = "R"
    elif plastic_range > plastic_tol:
        region = "P"
    else:
        region = "S"
    return BreeCycleResult(x=float(x), y=float(y), region=region,
                           growth_per_cycle=float(steady_growth),
                           plastic_range=float(plastic_range),
                           first_cycle_growth=float(first_growth),
                           growth_history=tuple(float(g) for g in growth),
                           cycles_run=len(growth),
                           plastic_strain_peak=float(np.max(np.abs(eps_p))))


def _plastic_strain_range(x: float, y: float, z: np.ndarray, w: np.ndarray,
                          eps_p: np.ndarray, steps: int) -> float:
    """Plastic strain range of the last cycle, re-played from the final state."""
    eps = eps_p.copy()
    lo = eps.copy()
    hi = eps.copy()
    e, kappa = x, 0.0
    try:
        for k in range(1, steps + 1):
            e, kappa, eps, _ = _solve_increment(x, k / steps, y, z, w, eps)
            lo = np.minimum(lo, eps)
            hi = np.maximum(hi, eps)
        for k in range(steps - 1, -1, -1):
            e, kappa, eps, _ = _solve_increment(x, k / steps, y, z, w, eps)
            lo = np.minimum(lo, eps)
            hi = np.maximum(hi, eps)
    except FloatingPointError:
        return float("inf")
    return float(np.max(hi - lo))


def melan_shakedown(x: float, y: float, *, fibres: int = 61) -> bool:
    """Melan static-theorem shakedown test for the Bree problem (exact LP).

    A residual field ``rho(z)`` with zero membrane (``int rho dz = 0``) must
    keep ``x + lambda y z + rho(z)`` inside ``[-1, 1]`` for every ``lambda`` in
    ``[0, 1]``.  Since the
    constraints are linear in ``lambda`` at fixed ``z``, checking
    ``lambda = 0`` and ``lambda = 1`` is sufficient.  Feasibility of this LP is
    a rigorous *lower bound* on the shakedown limit (sufficient condition).
    """
    try:
        from scipy.optimize import linprog
    except Exception:  # pragma: no cover
        raise RuntimeError("scipy is required for the Melan LP anchor")
    if fibres < 5:
        raise ValueError("at least 5 fibres are required")
    z, w = _thickness_rule(fibres)
    n = z.size
    eye = np.eye(n)
    rows, rhs = [], []
    for lam in (0.0, 1.0):
        base = x + lam * y * z
        # per-fibre constraints:  rho_i <= 1 - base_i   and   -rho_i <= 1 + base_i
        rows.append(eye)
        rhs.append(1.0 - base)
        rows.append(-eye)
        rhs.append(1.0 + base)
    # residual field must carry no membrane load (zero mean); bending is
    # restrained in this idealisation, so no zero-moment condition applies.
    res = linprog(np.zeros(n), A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                  A_eq=np.atleast_2d(0.5 * w), b_eq=np.zeros(1),
                  bounds=[(None, None)] * n, method="highs")
    if res.status == 0:
        return True
    if res.status == 2:
        return False
    raise RuntimeError(f"Melan LP failed: {res.message}")


def melan_boundary(y: float, *, fibres: int = 41, tol: float = 1e-9) -> float:
    """Largest ``x`` that still shakes down (Melan) at the given ``y``."""
    if y < 0.0:
        raise ValueError("y must be non-negative")
    lo, hi = 0.0, 1.0
    if not melan_shakedown(lo, y, fibres=fibres):
        return 0.0
    if melan_shakedown(hi, y, fibres=fibres):
        return 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if melan_shakedown(mid, y, fibres=fibres):
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def analytic_elastic_x(y: float) -> float:
    """Exact elastic limit in primary-stress ratio at secondary range ``y``.

    The first peak yields where ``x + y z`` reaches unity at ``z = 1``, hence
    ``x_el = 1 - y`` (zero for ``y >= 1``); this is *exact*, no iteration.
    """
    if y < 0.0:
        raise ValueError("y must be non-negative")
    return max(0.0, 1.0 - y)


def analytic_shakedown_x(y: float) -> float:
    """Closed-form Melan shakedown limit ``x_sd = 1 - y/4`` (restrained bending).

    *Derivation.*  With restrained bending the residual field only has to carry
    no membrane load, ``int rho dz = 0``.  Pointwise containment of the two load
    extremes requires (normalised stresses, ``x, y >= 0``)

    .. math::  -1 - x - y\\max(-z,0) \\le \\rho(z) \\le 1 - x - y\\max(z,0),

    so a zero-mean ``rho`` exists **iff** the mean of the pointwise upper bound
    is non-negative::

        mean(1 - x - y max(z, 0)) = 1 - x - y/4 >= 0   <=>   x <= 1 - y/4.

    Averaging is enough for *necessity* (pointwise bounds imply the bound on the
    mean) and an ``rho`` attaining the bounds where they are active is enough
    for *sufficiency*, so the boundary is exact within the idealisation.  Note
    that it differs from the classical ``x + y/2 = 1`` line, which is obtained
    when the residual is restricted to a *linear* (pure-bending) field; the
    restrained-bending kinematics admits a general zero-mean residual and hence
    extends the shakedown region.  Machine checks: :func:`melan_shakedown`
    (LP bisection agrees to ``5e-3``) and the incremental solver (region
    agreement on the whole diagram, see the tests).
    """
    if y < 0.0:
        raise ValueError("y must be non-negative")
    # second, independent condition: at the surface z = 1 the two load extremes
    # must fit into the same yield interval, which requires y <= 2 (beyond that
    # no residual field can prevent reverse yielding: alternating plasticity).
    if y > 2.0:
        return 0.0
    return 1.0 - 0.25 * y


@dataclass(frozen=True)
class BreeRegionMap:
    """Region labels of the Bree diagram on an ``(x, y)`` grid."""

    xs: np.ndarray
    ys: np.ndarray
    regions: tuple[tuple[str, ...], ...]

    def fraction(self, name: str) -> float:
        total = sum(len(r) for r in self.regions)
        hit = sum(1 for row in self.regions for v in row if v == name)
        return hit / total if total else 0.0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for row in self.regions:
            for v in row:
                out[v] = out.get(v, 0) + 1
        return out


def bree_region_map(xs: Sequence[float], ys: Sequence[float], *, cycles: int = 8,
                    steps: int = 6, fibres: int = 24) -> BreeRegionMap:
    """Classify the Bree diagram on a grid (region = mechanical output)."""
    xs = np.asarray(list(xs), dtype=float)
    ys = np.asarray(list(ys), dtype=float)
    rows: list[tuple[str, ...]] = []
    for y in ys:
        row: list[str] = []
        for x in xs:
            row.append(bree_region(float(x), float(y), cycles=cycles,
                                   steps=steps, fibres=fibres))
        rows.append(tuple(row))
    return BreeRegionMap(xs=xs, ys=ys, regions=tuple(rows))


def bree_region(x: float, y: float, **kwargs) -> str:
    """Region label ``E``/``S``/``P``/``R``/``COLLAPSE`` at ``(x, y)``."""
    if x >= 1.0:
        return "COLLAPSE"
    return bree_cycles(x, y, **kwargs).region


# --------------------------------------------------------------------------- #
# 3. Panel / frame level accumulated deflection
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IceBeltPlate:
    """Ice-belt panel idealised as a plate strip of span ``span_m``."""

    span_m: float
    thickness_m: float
    young_pa: float
    yield_pa: float
    edge_condition: str = "simple"          # 'simple' | 'clamped'
    membrane_restraint: float = 1.0         # 1 = fully restrained, 0 = free
    deflection_limit_ratio: float = 1.0 / 150.0
    poisson: float = 0.3

    def __post_init__(self) -> None:
        if self.span_m <= 0.0 or self.thickness_m <= 0.0:
            raise ValueError("span and thickness must be positive")
        if self.young_pa <= 0.0 or self.yield_pa <= 0.0:
            raise ValueError("elastic constants must be positive")
        if self.edge_condition not in ("simple", "clamped"):
            raise ValueError("edge_condition must be 'simple' or 'clamped'")
        if not (0.0 <= self.membrane_restraint <= 1.0):
            raise ValueError("membrane_restraint must lie in [0, 1]")
        if not (0.0 < self.deflection_limit_ratio < 1.0):
            raise ValueError("deflection_limit_ratio must lie in (0, 1)")
        if not (0.0 <= self.poisson < 0.5):
            raise ValueError("poisson must lie in [0, 0.5)")

    @property
    def section_modulus(self) -> float:
        """Elastic section modulus per unit width, ``t^2/6`` (m^3 per m)."""
        return self.thickness_m ** 2 / 6.0

    @property
    def moment_coefficient(self) -> float:
        """``M = coeff * p * L^2`` (1/8 simply supported, 1/12 clamped)."""
        return 1.0 / 8.0 if self.edge_condition == "simple" else 1.0 / 12.0

    @property
    def deflection_limit_m(self) -> float:
        return self.deflection_limit_ratio * self.span_m

    def yield_pressure_pa(self) -> float:
        """First-yield pressure of the elastic plate strip (Pa)."""
        return (self.yield_pa * self.thickness_m ** 2
                / (6.0 * self.moment_coefficient * self.span_m ** 2))

    def elastic_deflection(self, pressure_pa: float) -> float:
        """Linear plate-strip deflection under ``pressure_pa`` (m)."""
        d = (self.young_pa * self.thickness_m ** 3
             / (12.0 * (1.0 - self.poisson ** 2)))
        c = 5.0 / 384.0 if self.edge_condition == "simple" else 1.0 / 384.0
        return float(c * pressure_pa * self.span_m ** 4 / d)


def bending_stress_from_pressure(pressure_pa: float, span_m: float,
                                thickness_m: float, *,
                                edge_condition: str = "simple") -> float:
    """Outer-fibre bending stress of a plate strip under ``pressure_pa`` (Pa)."""
    if edge_condition not in ("simple", "clamped"):
        raise ValueError("edge_condition must be 'simple' or 'clamped'")
    coeff = 1.0 / 8.0 if edge_condition == "simple" else 1.0 / 12.0
    moment = coeff * pressure_pa * span_m ** 2
    return float(6.0 * moment / thickness_m ** 2)


def membrane_deflection(plastic_membrane_strain: float, span_m: float) -> float:
    """Permanent central deflection from membrane strain (von Karman kinematics).

    For ``w = w0 sin(pi x/L)`` the membrane strain is
    ``eps_m = (pi^2/4)(w0/L)^2``, hence ``w0 = (2L/pi) sqrt(eps_m)``.
    """
    if plastic_membrane_strain < 0.0:
        raise ValueError("membrane strain must be non-negative")
    if span_m <= 0.0:
        raise ValueError("span_m must be positive")
    return float(2.0 * span_m / math.pi * math.sqrt(plastic_membrane_strain))


def deflection_to_membrane_strain(deflection_m: float, span_m: float) -> float:
    """Inverse of :func:`membrane_deflection`."""
    if deflection_m < 0.0:
        raise ValueError("deflection must be non-negative")
    if span_m <= 0.0:
        raise ValueError("span_m must be positive")
    return float((math.pi * deflection_m / (2.0 * span_m)) ** 2)


@dataclass(frozen=True)
class DeflectionHistory:
    """Per-cycle permanent-deflection accumulation at constant amplitude."""

    pressure_mpa: float
    cycles: np.ndarray
    deflection_m: np.ndarray
    membrane_strain: np.ndarray
    primary_ratio: np.ndarray
    secondary_ratio: np.ndarray
    region: tuple[str, ...]
    plateau_reached: bool

    def final_deflection(self) -> float:
        return float(self.deflection_m[-1]) if self.deflection_m.size else 0.0

    def final_ratio(self, span_m: float) -> float:
        return self.final_deflection() / span_m


def first_cycle_plastic_deflection(plate: "IceBeltPlate", pressure_pa: float,
                                  ) -> float:
    """Permanent central deflection after the first application of the load.

    Pure-bending response of the rectangular plate strip with an ideal
    elastic-plastic section.  With ``M_y = sigma_y t^2/6`` (first yield),
    ``M_p = sigma_y t^2/4`` (plastic moment) and ``kappa_y = 2 sigma_y/(E t)``,
    the moment-curvature law is

    .. math::  M = M_p \\left[1 - \\tfrac13 (\\kappa_y/\\kappa)^2\\right],

    so ``kappa = kappa_y/sqrt(3(1 - M/M_p))``.  The permanent deflection is the
    unloaded plastic curvature over the span, ``w_p = (kappa - M/EI) L^2/8``.

    Returns ``inf`` when ``M >= M_p`` (the bending-only mechanism), which the
    caller must treat as a collapse flag: a real panel then relies on membrane
    action, i.e. the large-deflection regime that this closed form excludes.
    """
    if pressure_pa <= 0.0:
        return 0.0
    t = plate.thickness_m
    m_y = plate.yield_pa * t ** 2 / 6.0
    m_p = plate.yield_pa * t ** 2 / 4.0
    moment = plate.moment_coefficient * pressure_pa * plate.span_m ** 2
    if moment <= m_y:
        return 0.0
    if moment >= m_p:
        return float("inf")
    kappa_y = 2.0 * plate.yield_pa / (plate.young_pa * t)
    kappa = kappa_y / math.sqrt(3.0 * (1.0 - moment / m_p))
    inertia = t ** 3 / 12.0
    kappa_elastic = moment / (plate.young_pa * inertia)
    kappa_plastic = kappa - kappa_elastic
    return float(kappa_plastic * plate.span_m ** 2 / 8.0)


def membrane_relief_pressure(plate: "IceBeltPlate", pressure_pa: float,
                             deflection_m: float,
                             primary_ratio: float) -> float:
    """Pressure left for *bending* after the membrane carries its share.

    von Karman plate strip, ``w = w0 sin(pi x/L)``: the membrane force
    ``N = x sigma_y h`` (per unit width) relieves the transverse load by
    ``N (pi^2 w0 / L^2)`` (the ``N nabla^2 w`` term of the von Karman equation),
    so only ``p_eff = p - N pi^2 w0 / L^2`` is left to be carried in bending.
    This is the *stabilising* feedback of membrane action: a growing permanent
    deflection sheds bending into membrane tension instead of amplifying it.
    """
    n_membrane = primary_ratio * plate.yield_pa * plate.thickness_m
    relief = n_membrane * math.pi ** 2 * deflection_m / plate.span_m ** 2         * plate.membrane_restraint
    return max(0.0, float(pressure_pa - relief))


def ice_belt_deflection_history(plate: IceBeltPlate, pressure_mpa: float, *,
                               cycles: int = 60, steps: int = 6,
                               fibres: int = 24, primary_cap: float = 0.98
                               ) -> DeflectionHistory:
    """Accumulate the permanent deflection over ``cycles`` equal-amplitude cycles.

    Each cycle solves the Bree problem at

    * ``y = sigma_b(p_eff)/sigma_y`` -- the secondary, self-equilibrated
      through-thickness bending of the *residual* pressure
      ``p_eff = p - N pi^2 w0/L^2`` (:func:`membrane_relief_pressure`), and
    * ``x = min(primary_cap, membrane_restraint E eps_m^p / sigma_y)`` -- the
      primary membrane stress generated by the accumulated permanent deflection
      through the von Karman kinematics ``eps_m = (pi^2/4)(w0/L)^2``,

    then adds the cycle's membrane-plastic-strain growth to ``eps_m^p``.  The
    deflection therefore saturates: growing ``w0`` raises ``x`` (more shakedown
    margin is consumed) but simultaneously *lowers* ``y`` (bending is shed into
    membrane action), which is the physically correct von Karman feedback.
    """
    if pressure_mpa < 0.0:
        raise ValueError("pressure_mpa must be non-negative")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")
    if not (0.0 < primary_cap <= 1.0):
        raise ValueError("primary_cap must lie in (0, 1]")
    pressure_pa = pressure_mpa * 1.0e6
    z, w = _thickness_rule(fibres)
    eps_p = np.zeros_like(z)
    # Bree membrane strain is *normalised* by eps_y = sigma_y/E; convert to the
    # true strain whenever the von Karman deflection kinematics is used.
    strain_scale = plate.yield_pa / plate.young_pa
    w_set = first_cycle_plastic_deflection(plate, pressure_pa)
    if not math.isfinite(w_set):
        w_set = plate.deflection_limit_m * 1.5      # bending mechanism: flag
    eps_m = deflection_to_membrane_strain(w_set, plate.span_m) / strain_scale
    e = 0.0
    kappa = 0.0
    out_cycles = [0]
    out_deflection = [w_set]
    out_strain = [eps_m]
    out_x = [min(primary_cap, plate.membrane_restraint * eps_m)]
    out_y = [bending_stress_from_pressure(
        pressure_pa, plate.span_m, plate.thickness_m,
        edge_condition=plate.edge_condition) / plate.yield_pa]
    out_region = ["E"]
    plateau = False
    while len(out_cycles) <= cycles:
        n = out_cycles[-1] + 1
        w0 = membrane_deflection(eps_m * strain_scale, plate.span_m)
        # eps_m is already normalised (eps E / sigma_y), so it *is* x directly
        x = min(primary_cap, plate.membrane_restraint * eps_m)
        p_eff = membrane_relief_pressure(plate, pressure_pa, w0, x)
        y = bending_stress_from_pressure(p_eff, plate.span_m, plate.thickness_m,
                                         edge_condition=plate.edge_condition) \
            / plate.yield_pa
        e_start = e
        plastic_this = 0.0
        collapsed = False
        try:
            for k in range(1, steps + 1):
                e, kappa, eps_new, _ = _solve_increment(x, k / steps, y, z, w, eps_p)
                plastic_this = max(plastic_this,
                                   float(np.max(np.abs(eps_new - eps_p))))
                eps_p = eps_new
            for k in range(steps - 1, -1, -1):
                e, kappa, eps_new, _ = _solve_increment(x, k / steps, y, z, w, eps_p)
                plastic_this = max(plastic_this,
                                   float(np.max(np.abs(eps_new - eps_p))))
                eps_p = eps_new
        except FloatingPointError:
            collapsed = True
        growth = e - e_start
        if growth > 0.0:
            eps_m += growth
        if collapsed:
            region = "COLLAPSE"
        elif plastic_this > 1e-10:
            region = "R" if abs(growth) > 1e-10 else "P"
        else:
            region = "S" if out_region[-1] == "E" and plastic_this <= 1e-10 \
                else out_region[-1]
        out_cycles.append(n)
        out_deflection.append(membrane_deflection(eps_m * strain_scale,
                                                 plate.span_m))
        out_strain.append(eps_m)
        out_x.append(x)
        out_y.append(y)
        out_region.append(region)
        if collapsed:
            break
        if n > 4 and abs(growth) <= 1e-13 and plastic_this <= 1e-11:
            plateau = True
            break
    return DeflectionHistory(pressure_mpa=float(pressure_mpa),
                             cycles=np.asarray(out_cycles, dtype=int),
                             deflection_m=np.asarray(out_deflection, dtype=float),
                             membrane_strain=np.asarray(out_strain, dtype=float),
                             primary_ratio=np.asarray(out_x, dtype=float),
                             secondary_ratio=np.asarray(out_y, dtype=float),
                             region=tuple(out_region),
                             plateau_reached=plateau)


@dataclass(frozen=True)
class ServiceLifeResult:
    """Panel-level spectrum accumulation result."""

    plate: IceBeltPlate
    spectrum: IceCollisionSpectrum
    histories: tuple[DeflectionHistory, ...]
    total_deflection_m: float
    limit_m: float
    cycles_to_limit: float
    utilisation: float
    max_secondary_ratio: float
    max_primary_ratio: float
    region_counts: dict

    @property
    def acceptable(self) -> bool:
        return self.utilisation <= 1.0


def ice_belt_service_life(plate: IceBeltPlate, spectrum: IceCollisionSpectrum, *,
                          cycles_per_bin: int | None = 40, steps: int = 6,
                          fibres: int = 24) -> ServiceLifeResult:
    """Accumulate the spectrum deflection and compare with the panel limit.

    ``total_deflection_m`` is the root-sum-square of the per-bin permanent
    deflections (the von Karman kinematics is non-linear, so the per-bin
    contributions are combined in quadrature -- a documented conservative/
    neutral approximation), and ``cycles_to_limit`` linearly extrapolates the
    steady growth rate of the dominant bin.

    ``cycles_per_bin`` caps the per-bin cycle count to keep the sweep
    affordable (``None`` = run every cycle).
    """
    if len(spectrum) == 0:
        return ServiceLifeResult(plate=plate, spectrum=spectrum, histories=(),
                                 total_deflection_m=0.0,
                                 limit_m=plate.deflection_limit_m,
                                 cycles_to_limit=float("inf"), utilisation=0.0,
                                 max_secondary_ratio=0.0, max_primary_ratio=0.0,
                                 region_counts={})
    histories: list[DeflectionHistory] = []
    sum_sq = 0.0
    max_p = 0.0
    max_x = 0.0
    counts: dict[str, int] = {}
    dominant_rate = 0.0
    for b in spectrum:
        n = b.count if cycles_per_bin is None else min(b.count, int(cycles_per_bin))
        hist = ice_belt_deflection_history(plate, b.pressure_mpa, cycles=n,
                                           steps=steps, fibres=fibres)
        histories.append(hist)
        sum_sq += hist.final_deflection() ** 2
        max_p = max(max_p, b.pressure_mpa)
        if hist.primary_ratio.size:
            max_x = max(max_x, float(np.max(hist.primary_ratio)))
        for r in hist.region:
            counts[r] = counts.get(r, 0) + 1
        if hist.cycles.size >= 2 and hist.deflection_m.size >= 2:
            g = hist.deflection_m[-1] - hist.deflection_m[-2]
            if g > dominant_rate:
                dominant_rate = g
    total = math.sqrt(sum_sq)
    limit = plate.deflection_limit_m
    if dominant_rate > 0.0:
        remaining = max(0.0, limit - total)
        cycles_to_limit = remaining / dominant_rate
    else:
        cycles_to_limit = float("inf") if total < limit else 0.0
    return ServiceLifeResult(plate=plate, spectrum=spectrum,
                             histories=tuple(histories),
                             total_deflection_m=float(total),
                             limit_m=float(limit),
                             cycles_to_limit=float(cycles_to_limit),
                             utilisation=float(total / limit),
                             max_secondary_ratio=float(
                                 bending_stress_from_pressure(
                                     max_p * 1.0e6, plate.span_m, plate.thickness_m,
                                     edge_condition=plate.edge_condition)
                                 / plate.yield_pa),
                             max_primary_ratio=float(max_x),
                             region_counts=counts)