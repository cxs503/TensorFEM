"""H4-ICE-BELT-FEM round 1: pressure-area law x plate-panel cross-validation x
design-pressure zones.

Why this module exists
----------------------
``tensorfem.ice_belt_collapse`` carries the *(pressure -> collapse)* chain for an
ice-belt plate field and ``tensorfem.ice_belt_ratcheting`` the *(pressure ->
cyclic accumulation)* chain.  Neither answers what has to be settled **before**
either chain may speak for a real panel:

1. is the local ice **pressure-area curve** implemented correctly -- point by
   point, on an independent evaluation route, not merely "it looks like a power
   law"?
2. does the **plate-panel response** to that pressure depend on the FE
   discretisation?  Two independent discretisations must agree to <10 % before
   either of them is allowed to represent the panel.
3. is the **design-pressure zone division** (longitudinal hull areas x structural
   load patches x ice-belt extent) internally consistent -- does it tile the hull
   without gaps or overlaps, does the pressure ordering follow from the law, and
   does it follow the class ranking of the ecosystem ice-class registry?

This module is the verification surface for those three questions.  It is
deliberately **rule-value free**: every IACS coefficient (``p_ref``, ``C_d``,
``C_f``, hull-area factors, belt offsets, class factors) is caller supplied and
carried together with a ``provenance`` string that
:func:`check_pressure_provenance` gates.  The rule text is not available offline
and a transcribed "looks about right" table is worse than an explicit WITHHELD.

Pressure-area law
-----------------
.. math::  p(A) = C_d C_f p_{ref} (A/A_{ref})^{-\\alpha},
           \\qquad F(A) = p(A) A \\propto A^{1-\\alpha}

The functional form, the decrease of pressure with loaded area and the
``alpha = 0.3`` local-pressure exponent are the documented IACS Polar Class
local-load structure (IACS UR I2 "Design ice pressure"; ISO 19906:2010 uses the
same pressure-area form for local ice action).  The **numeric anchor**
``p_ref`` is the caller's rule value -- never supplied here.

Three independent evaluation routes make the curve checkable without
circularity:

===========  ==================================================================
``direct``   ``p = c A**-alpha``
``log``      ``p = exp(log c - alpha (log A - log A_ref))`` -- never forms
             ``A**-alpha``
``inverse``  ``p = dF/dA`` by central differencing the *design force*
             ``F(A) = p A``; this route only sees ``F``, not ``p``
===========  ==================================================================

``F`` is in turn checked by Gauss-Legendre quadrature of ``p``, and the scaling
invariant ``p A**alpha = C_d C_f p_ref A_ref**alpha`` is asserted constant along
the whole curve.  Agreement of the routes to ~1e-9 is a statement about the
*implementation*; it is **not** an accuracy statement against IACS, which needs
the rule table (see the ``rule_table_comparison`` block of the run artifact,
status ``WITHHELD``).

Panel cross-validation
----------------------
The same ice-belt plate field is solved by three mutually independent routes:

* **MITC4** -- :class:`tensorfem.ice_belt_collapse.PlateModel`, the 5-DOF layered
  flat shell (numpy, assumed transverse-shear field, Dvorkin & Bathe 1984) used
  by the collapse chain;
* **Q4-SRI** -- :func:`tensorfem.plate_mindlin.solve_plate_on_winkler_foundation`,
  the independent torch implementation with the one-point shear rule (Hughes,
  Cohen & Haroun 1978), ``k = 0``;
* **Navier** -- the Kirchhoff double-series solution of the simply supported
  plate (:func:`navier_patch_centre`), sharing neither code nor discretisation
  with either FE route; uniform-load terms as in Timoshenko & Woinowsky-Krieger
  (1959) section 30, patch terms from the double Fourier coefficients of a
  centred rectangular patch (derivation in the docstring).

Acceptance: **pairwise** relative difference < 10 % on the peak deflection, for a
full-field pressure *and* for a local ice patch whose magnitude comes from the
pressure-area law itself, so both features are exercised together.

Design-pressure zones
---------------------
:class:`HullZone` (longitudinal hull areas), :class:`DesignPatch` (plate field /
stiffener / primary member -- each with its own loaded area, hence its own
pressure off the same curve) and :class:`IceBelt` (vertical extent from
caller-supplied rule offsets) are combined by :func:`design_pressure_zones` and
audited by :func:`check_design_pressure_zones` (checks Z1..Z9).  Every check is a
machine-checkable invariant of the division itself, not a comparison against a
transcribed rule number.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "PressureAreaLaw",
    "RulePressureTable",
    "check_pressure_provenance",
    "pressure_area_curve_rows",
    "check_curve_routes",
    "check_force_quadrature",
    "compare_with_rule_table",
    "HullZone",
    "partition_hull_zones",
    "check_zone_partition",
    "IceBelt",
    "ice_belt_extent",
    "check_belt_containment",
    "DesignPatch",
    "panel_patches",
    "design_pressure_zones",
    "check_design_pressure_zones",
    "check_class_ordering",
    "navier_patch_centre",
    "panel_cross_validation",
    "panel_mesh_study",
    "run_h4_verification",
]

_PA_PER_MPA = 1.0e6


# --------------------------------------------------------------------------- #
# 1. pressure-area law and its independent evaluation routes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PressureAreaLaw:
    """``p(A) = C_d C_f p_ref (A/A_ref)**-alpha`` local ice pressure.

    ``p_ref`` is the *average* pressure on the reference area ``A_ref`` for the
    governing ice class; ``c_d`` / ``c_f`` are the multiplicative displacement and
    hull-form factors of the Polar Class local-load formulation.  All are rule
    values: they are supplied together with ``provenance`` and gated by
    :func:`check_pressure_provenance`.
    """

    p_ref: float
    area_ref: float = 1.0
    exponent: float = 0.3
    c_d: float = 1.0
    c_f: float = 1.0
    provenance: str = "caller-supplied IACS UR I2 local-pressure anchor"

    def __post_init__(self) -> None:
        if not (self.p_ref > 0.0) or not math.isfinite(self.p_ref):
            raise ValueError("p_ref must be positive and finite")
        if not (self.area_ref > 0.0):
            raise ValueError("area_ref must be positive")
        if not (0.0 <= self.exponent < 1.0):
            raise ValueError("exponent must lie in [0, 1) so that F(A) grows")
        if self.c_d <= 0.0 or self.c_f <= 0.0:
            raise ValueError("c_d and c_f must be positive")

    # -- coefficients ------------------------------------------------------ #
    @property
    def scale(self) -> float:
        """Curve amplitude ``C_d C_f p_ref`` (Pa)."""
        return self.c_d * self.c_f * self.p_ref

    @property
    def p_ref_mpa(self) -> float:
        return self.p_ref / _PA_PER_MPA

    def scaled(self, factor: float, provenance: str | None = None) -> "PressureAreaLaw":
        """Same law with the amplitude multiplied by ``factor`` (zone / class)."""
        if not (factor > 0.0):
            raise ValueError("factor must be positive")
        return PressureAreaLaw(
            p_ref=self.p_ref * float(factor),
            area_ref=self.area_ref,
            exponent=self.exponent,
            c_d=self.c_d,
            c_f=self.c_f,
            provenance=self.provenance if provenance is None else provenance,
        )

    # -- the three evaluation routes --------------------------------------- #
    def pressure(self, area) -> np.ndarray:
        """Route ``direct``: ``p = C_d C_f p_ref (A/A_ref)**-alpha``."""
        a = np.asarray(area, dtype=float)
        if np.any(~np.isfinite(a)) or np.any(a <= 0.0):
            raise ValueError("loaded area must be positive and finite")
        return self.scale * np.power(a / self.area_ref, -self.exponent)

    def pressure_logspace(self, area) -> np.ndarray:
        """Route ``log``: exponentiate the log of the law (never forms ``A**-a``)."""
        a = np.asarray(area, dtype=float)
        if np.any(a <= 0.0):
            raise ValueError("loaded area must be positive and finite")
        return np.exp(math.log(self.scale)
                      - self.exponent * np.log(a / self.area_ref))

    def design_force(self, area) -> np.ndarray:
        """Patch force ``F(A) = p(A) A`` (N)."""
        a = np.asarray(area, dtype=float)
        return self.pressure(a) * a

    def line_load(self, area, span: float) -> np.ndarray:
        """Equivalent line load ``q = p(A) A / span`` (N/m)."""
        if not (span > 0.0):
            raise ValueError("span must be positive")
        return self.design_force(area) / float(span)

    def area_for_force(self, force, *, newton: int = 60) -> np.ndarray:
        """Inverse of :meth:`design_force`, closed form polished by Newton."""
        f = np.asarray(force, dtype=float)
        if np.any(f <= 0.0):
            raise ValueError("force must be positive")
        c = self.scale * self.area_ref ** self.exponent
        a = np.power(f / c, 1.0 / (1.0 - self.exponent))
        for _ in range(int(newton)):
            residual = self.design_force(a) - f
            slope = (1.0 - self.exponent) * self.design_force(a) / a
            step = residual / slope
            a = np.abs(a - step)
            if np.max(np.abs(step)) <= 1.0e-15 * np.max(a):
                break
        return a

    def invariant(self, area) -> np.ndarray:
        """``p A**alpha`` -- constant along the curve for a correct implementation."""
        a = np.asarray(area, dtype=float)
        return self.pressure(a) * np.power(a, self.exponent)

    def loglog_slope(self, area_lo: float, area_hi: float) -> float:
        """Numerical log-log slope of ``p(A)`` between two areas."""
        if not (0.0 < area_lo < area_hi):
            raise ValueError("need 0 < area_lo < area_hi")
        p_lo = float(self.pressure(area_lo))
        p_hi = float(self.pressure(area_hi))
        return math.log(p_hi / p_lo) / math.log(area_hi / area_lo)

    def force_derivative(self, area, rel_step: float = 1.0e-5) -> np.ndarray:
        """``dF/dA`` by central differences of :meth:`design_force` only.

        Note the identity ``dF/dA = (1 - alpha) p``: with ``F = p A`` and
        ``p ~ A**-alpha`` the derivative is *not* ``p``.  Recovering ``p`` from it
        therefore needs the ``(1 - alpha)`` division done by
        :meth:`marginal_pressure`.
        """
        a = np.atleast_1d(np.asarray(area, dtype=float))
        if np.any(a <= 0.0):
            raise ValueError("loaded area must be positive")
        h = rel_step * a
        lo = np.maximum(a - h, 0.5 * a)
        hi = a + h
        return (self.design_force(hi) - self.design_force(lo)) / (hi - lo)

    def marginal_pressure(self, area, rel_step: float = 1.0e-5) -> np.ndarray:
        """Route ``marginal``: recover ``p`` from ``dF/dA / (1 - alpha)``.

        Only :meth:`design_force` and the exponent are used; the analytic
        pressure path is never touched.
        """
        return self.force_derivative(area, rel_step) / (1.0 - self.exponent)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"PressureAreaLaw(p_ref={self.p_ref_mpa:g} MPa, "
                f"A_ref={self.area_ref:g} m2, alpha={self.exponent:g}, "
                f"c_d={self.c_d:g}, c_f={self.c_f:g})")


def check_pressure_provenance(law: PressureAreaLaw,
                              require_design_data: bool = False) -> None:
    """Fail closed when a demonstration (non-rule) curve is used for design.

    Provenance strings beginning with ``DEMONSTRATION`` / ``ILLUSTRATIVE`` /
    ``SYNTHETIC`` / ``PLACEHOLDER`` are mechanism-only data.  With
    ``require_design_data=True`` they raise, and the error text names what has to
    be supplied instead.
    """
    tag = str(law.provenance).strip().upper()
    if require_design_data and tag.startswith(("DEMONSTRATION", "ILLUSTRATIVE",
                                               "SYNTHETIC", "PLACEHOLDER")):
        raise ValueError(
            "pressure-area provenance is not rule data "
            f"({law.provenance!r}); supply the governing IACS UR I2 local "
            "pressure anchor (p_ref, C_d, C_f, A_ref, alpha) with a citable "
            "provenance before any design use")


# --------------------------------------------------------------------------- #
# 2. point-by-point curve verification
# --------------------------------------------------------------------------- #
def pressure_area_curve_rows(law: PressureAreaLaw,
                             areas: Sequence[float]) -> list[dict]:
    """Point-by-point rows of the pressure-area curve on all three routes."""
    a = np.asarray(sorted(float(x) for x in areas), dtype=float)
    if a.size == 0:
        raise ValueError("at least one area is required")
    p_direct = law.pressure(a)
    p_log = law.pressure_logspace(a)
    p_inv = law.marginal_pressure(a)
    p_rad = radial_field_mean_pressure(a, law)
    force = law.design_force(a)
    inv = law.invariant(a)
    area_back = law.area_for_force(force)
    rows = []
    for i, area in enumerate(a):
        rows.append({
            "area_m2": float(area),
            "pressure_mpa_direct": float(p_direct[i]) / _PA_PER_MPA,
            "pressure_mpa_logspace": float(p_log[i]) / _PA_PER_MPA,
            "pressure_mpa_marginal": float(p_inv[i]) / _PA_PER_MPA,
            "pressure_mpa_radial_mean": float(p_rad[i]) / _PA_PER_MPA,
            "rel_diff_logspace": float(p_log[i] / p_direct[i] - 1.0),
            "rel_diff_marginal": float(p_inv[i] / p_direct[i] - 1.0),
            "rel_diff_radial_mean": float(p_rad[i] / p_direct[i] - 1.0),
            "design_force_N": float(force[i]),
            "line_load_N_per_m": float(force[i] / math.sqrt(area)),
            "invariant_p_A_alpha": float(inv[i]),
            "invariant_rel_diff": float(inv[i] / law.scale - 1.0),
            "area_recovered_m2": float(area_back[i]),
            "rel_diff_area_inverse": float(area_back[i] / area - 1.0),
        })
    return rows


def check_curve_routes(law: PressureAreaLaw, areas: Sequence[float],
                       tolerance: float = 1.0e-9) -> dict:
    """Aggregate the three-route comparison into one machine-checkable verdict."""
    areas = [float(x) for x in areas]
    rows = pressure_area_curve_rows(law, areas)
    max_log = max(abs(r["rel_diff_logspace"]) for r in rows)
    max_inv = max(abs(r["rel_diff_marginal"]) for r in rows)
    max_rad = max(abs(r["rel_diff_radial_mean"]) for r in rows)
    max_scaling = max(abs(r["invariant_rel_diff"]) for r in rows)
    max_area = max(abs(r["rel_diff_area_inverse"]) for r in rows)
    slope = law.loglog_slope(min(areas), max(areas))
    fit = fit_loglog_exponent(law, areas)
    return {
        "rows": rows,
        "max_rel_diff_logspace": max_log,
        "max_rel_diff_marginal": max_inv,
        "max_rel_diff_radial_mean": max_rad,
        "max_rel_diff_scaling_invariant": max_scaling,
        "max_rel_diff_area_inverse": max_area,
        "loglog_slope_between_endpoints": slope,
        "loglog_slope_error_vs_exponent": slope + law.exponent,
        "fitted_exponent": fit["fitted_exponent"],
        "fitted_exponent_error": fit["fitted_exponent_error"],
        "fit_max_residual": fit["max_abs_log_residual"],
        "tolerance": tolerance,
        "passed": bool(max(max_log, max_inv, max_rad, max_scaling, max_area)
                       <= tolerance),
        "provenance": law.provenance,
    }


def check_force_quadrature(law: PressureAreaLaw, a_lo: float, a_hi: float,
                           order: int = 24, segments: int = 32) -> dict:
    """``int_{a_lo}^{a_hi} p dA`` by Gauss-Legendre vs ``Delta F / (1 - alpha)``.

    The interval is split geometrically (``segments`` panels, equal ratios) and a
    fixed-order Gauss rule is applied to each: a single rule over the full
    pressure-area range would be dominated by the ``A**-alpha`` curvature and is
    not a fair test of the force path.  ``segments = 1`` reproduces the naive
    single-panel rule and is reported alongside, so the refinement is visible.
    """
    if not (0.0 < a_lo < a_hi):
        raise ValueError("need 0 < a_lo < a_hi")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    # Because F(A) = p(A) A = c A**(1-alpha), the *accumulated* pressure integral
    # is int p dA = Delta F / (1 - alpha).  Same (1 - alpha) factor as the
    # marginal-pressure route; forgetting it is exactly the trap this check
    # documents.
    exact = float(law.design_force(a_hi) - law.design_force(a_lo)) \
        / (1.0 - law.exponent)

    ratio = (a_hi / a_lo) ** (1.0 / max(1, int(segments)))
    edges = a_lo * ratio ** np.arange(int(segments) + 1)
    edges[-1] = a_hi
    quad = 0.0
    for lo, hi in zip(edges, edges[1:]):
        half = 0.5 * (hi - lo)
        mid = 0.5 * (hi + lo)
        quad += float(np.sum(weights * law.pressure(mid + half * nodes)) * half)

    half = 0.5 * (a_hi - a_lo)
    mid = 0.5 * (a_hi + a_lo)
    single = float(np.sum(weights * law.pressure(mid + half * nodes)) * half)
    return {"a_lo": a_lo, "a_hi": a_hi, "order": int(order),
            "segments": int(segments),
            "quadrature_integral_N": quad, "force_difference_N": exact,
            "rel_error": quad / exact - 1.0,
            "single_panel_rel_error": single / exact - 1.0,
            "passed": abs(quad / exact - 1.0) <= 1.0e-10}


def fit_loglog_exponent(law: PressureAreaLaw, areas: Sequence[float]) -> dict:
    """Recover the exponent by least squares on ``(log A, log p)``.

    Catches the realistic implementation errors (sign, exponent, reference-area
    slip) that a self-consistency check with the same formula cannot.
    """
    a = np.asarray([float(x) for x in areas], dtype=float)
    if a.size < 2 or np.any(a <= 0.0):
        raise ValueError("need at least two positive areas")
    log_a = np.log(a)
    log_p = np.log(law.pressure(a))
    slope, intercept = np.polyfit(log_a, log_p, 1)
    residual = log_p - (slope * log_a + intercept)
    return {"fitted_exponent": float(slope),
            "fitted_exponent_error": float(slope + law.exponent),
            "fitted_amplitude": float(math.exp(intercept)),
            "amplitude_rel_error": float(math.exp(intercept) / law.scale
                                         * law.area_ref ** law.exponent - 1.0),
            "max_abs_log_residual": float(np.max(np.abs(residual)))}


def radial_field_mean_pressure(area, law: PressureAreaLaw, order: int = 40,
                               n_theta: int = 64) -> np.ndarray:
    """Route ``radial``: the law as the *mean* of a self-similar local field.

    The pressure-area relation is the area average of a local ice pressure field
    that decays with distance from the contact, ``p_local(r) = c r**-beta``.  On
    a circle of area ``A`` (radius ``R = sqrt(A/pi)``)

    .. math::  \\bar p = \\frac{1}{A}\\int_0^{2\\pi}\\!\\int_0^R
               c r^{-\\beta} r \\, dr \\, d\\theta
               = \\frac{2 c}{2-\\beta} R^{-\\beta}
               \\;\\propto\\; A^{-\\beta/2},

    so the law's exponent is ``alpha = beta/2`` and the same curve must come out of
    a two-dimensional quadrature of the local field.  The substitution
    ``r = R t^{1/(2-beta)}`` makes the integrand constant, so Gauss-Legendre is
    exact here and the comparison isolates calibration/implementation, not
    quadrature error.  ``c`` is calibrated so that the mean at ``law.area_ref``
    equals ``law.pressure(area_ref)`` (a calibration, not a free fit -- with one
    scale factor fixed the *shape* is a prediction).
    """
    beta = 2.0 * law.exponent
    if not (0.0 < beta < 2.0):
        raise ValueError("radial route needs 0 < beta < 2 (i.e. 0 <= alpha < 1)")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    t = 0.5 * (nodes + 1.0)
    w = 0.5 * weights
    power = 1.0 / (2.0 - beta)
    r_ref = math.sqrt(law.area_ref / math.pi)
    # mean at the reference area is c * 2/(2-beta) * r_ref**-beta
    c_local = float(law.pressure(law.area_ref)) * (2.0 - beta) / 2.0 * r_ref ** beta
    out = []
    for target in np.atleast_1d(np.asarray(area, dtype=float)):
        if target <= 0.0:
            raise ValueError("loaded area must be positive")
        radius = math.sqrt(target / math.pi)
        r = radius * np.power(t, power)
        integral = float(np.sum(w * np.power(r, 1.0 - beta)
                                * radius * power * np.power(t, power - 1.0)))
        mean = 2.0 * math.pi * c_local * integral / target
        out.append(mean)
    return np.array(out)


# --------------------------------------------------------------------------- #
# 3. comparison against an externally supplied rule table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RulePressureTable:
    """A rule table of ``(loaded area, average pressure)`` pairs, with provenance.

    The table is *data*: rule edition, clause and revision travel with it and are
    echoed into every comparison row.  Nothing here can invent a value.
    """

    rows: tuple[tuple[float, float], ...]        # (area_m2, pressure_MPa)
    provenance: str
    clause: str = ""
    is_rule_data: bool = False

    def __post_init__(self) -> None:
        if len(self.rows) < 2:
            raise ValueError("a rule table needs at least two points")
        a = [float(r[0]) for r in self.rows]
        if any(x <= 0.0 for x in a):
            raise ValueError("loaded areas must be positive")
        if any(b <= a0 for a0, b in zip(a, a[1:])):
            raise ValueError("table areas must be strictly increasing")
        if any(float(r[1]) <= 0.0 for r in self.rows):
            raise ValueError("table pressures must be positive")
        if not str(self.provenance).strip():
            raise ValueError("a rule table must carry its provenance")

    def areas(self) -> np.ndarray:
        return np.array([float(r[0]) for r in self.rows], dtype=float)

    def pressures(self) -> np.ndarray:
        return np.array([float(r[1]) for r in self.rows], dtype=float)


def compare_with_rule_table(law: PressureAreaLaw, table: RulePressureTable,
                            tolerance: float = 0.05,
                            require_rule_data: bool = True) -> dict:
    """Point-by-point ``ref vs got`` comparison of the law against ``table``.

    With ``require_rule_data=True`` a demonstration table is refused, so a
    mechanism check can never be mistaken for a rule-accuracy claim.
    """
    if require_rule_data and not table.is_rule_data:
        raise ValueError(
            f"table {table.provenance!r} is not flagged as rule data; a "
            "demonstration table cannot support a rule-accuracy claim")
    areas = table.areas()
    ref = table.pressures()
    got = law.pressure(areas) / _PA_PER_MPA
    rows = []
    for i, area in enumerate(areas):
        rows.append({"area_m2": float(area), "ref_mpa": float(ref[i]),
                     "got_mpa": float(got[i]),
                     "rel_error": float(got[i] / ref[i] - 1.0)})
    worst = max(abs(r["rel_error"]) for r in rows)
    return {"clause": table.clause, "table_provenance": table.provenance,
            "rows": rows, "max_abs_rel_error": worst, "tolerance": tolerance,
            "passed": bool(worst <= tolerance)}

# --------------------------------------------------------------------------- #
# 4. design-pressure zone division
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HullZone:
    """One longitudinal hull area with its own pressure factor."""

    key: str
    x_start: float
    x_end: float
    pressure_factor: float = 1.0
    provenance: str = "caller-supplied hull-area factor"

    def __post_init__(self) -> None:
        if not (self.x_end > self.x_start):
            raise ValueError(f"zone {self.key!r}: x_end must exceed x_start")
        if not (self.pressure_factor > 0.0):
            raise ValueError(f"zone {self.key!r}: pressure_factor must be positive")

    @property
    def length(self) -> float:
        return self.x_end - self.x_start

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.x_start + self.x_end)


def partition_hull_zones(length: float,
                         fractions: Iterable[tuple[str, float]],
                         factors: dict[str, float] | None = None,
                         provenance: str = "caller-supplied hull-area boundaries"
                         ) -> tuple[HullZone, ...]:
    """Build the longitudinal hull-area partition from end fractions.

    ``fractions`` is an ordered ``(key, end_fraction_of_length)`` sequence; the
    last fraction must be 1.0.  Boundaries are validated strictly increasing, so
    a gap or an overlap cannot be constructed by accident.
    """
    if not (length > 0.0):
        raise ValueError("length must be positive")
    fr = [(str(k), float(v)) for k, v in fractions]
    if not fr:
        raise ValueError("at least one zone is required")
    if abs(fr[-1][1] - 1.0) > 1.0e-12:
        raise ValueError("the last zone must end at the ship length (1.0)")
    for (_, f0), (k1, f1) in zip(fr, fr[1:]):
        if not (f1 > f0):
            raise ValueError(f"zone {k1!r}: fractions must strictly increase")
    fac = dict(factors or {})
    zones = []
    x0 = 0.0
    for key, f in fr:
        x1 = f * length
        zones.append(HullZone(key=key, x_start=x0, x_end=x1,
                              pressure_factor=float(fac.get(key, 1.0)),
                              provenance=provenance))
        x0 = x1
    return tuple(zones)


def check_zone_partition(zones: Sequence[HullZone], length: float,
                         tol: float = 1.0e-12) -> dict:
    """Z1 length tiling, Z2 contiguity, Z3 endpoint containment, Z4 factors."""
    zones = tuple(zones)
    total = sum(z.length for z in zones)
    gaps = [abs(z1.x_start - z0.x_end) for z0, z1 in zip(zones, zones[1:])]
    z1_tiling = abs(total - length) <= tol * max(1.0, length)
    z2_contiguous = all(g <= tol * max(1.0, length) for g in gaps)
    z3_inside = (abs(zones[0].x_start) <= tol
                 and abs(zones[-1].x_end - length) <= tol * max(1.0, length))
    z4_factors = all(z.pressure_factor > 0.0 for z in zones)
    return {
        "covered_length": total, "ship_length": length,
        "zone_lengths": {z.key: z.length for z in zones},
        "max_boundary_gap": max(gaps) if gaps else 0.0,
        "Z1_length_tiling": bool(z1_tiling),
        "Z2_contiguity": bool(z2_contiguous),
        "Z3_endpoint_containment": bool(z3_inside),
        "Z4_positive_factors": bool(z4_factors),
        "passed": bool(z1_tiling and z2_contiguous and z3_inside and z4_factors),
    }


@dataclass(frozen=True)
class IceBelt:
    """Vertical extent of the ice belt above the base line (m)."""

    z_lower: float
    z_upper: float
    provenance: str = "caller-supplied rule offsets"
    scantling_draught: float | None = None

    def __post_init__(self) -> None:
        if not (self.z_upper > self.z_lower):
            raise ValueError("z_upper must exceed z_lower")

    @property
    def height(self) -> float:
        return self.z_upper - self.z_lower

    def contains(self, z: float, tol: float = 1.0e-12) -> bool:
        return self.z_lower - tol <= z <= self.z_upper + tol


def ice_belt_extent(scantling_draught: float, lower_offset: float,
                    upper_offset: float,
                    provenance: str = "caller-supplied rule offsets") -> IceBelt:
    """Belt from ``T_s - lower_offset`` to ``T_s + upper_offset``.

    The offsets are rule quantities (they follow from the draught and the rule's
    belt definition), so they are inputs, not constants of this module.
    """
    if not (scantling_draught > 0.0):
        raise ValueError("scantling_draught must be positive")
    if lower_offset <= 0.0 or upper_offset <= 0.0:
        raise ValueError("belt offsets must be positive")
    z_lower = scantling_draught - lower_offset
    if z_lower <= 0.0:
        raise ValueError("the belt would start below the base line")
    return IceBelt(z_lower=z_lower, z_upper=scantling_draught + upper_offset,
                   provenance=provenance, scantling_draught=scantling_draught)


def check_belt_containment(belt: IceBelt, depth: float,
                           hull_zone_interval: tuple[float, float] | None = None
                           ) -> dict:
    """Z5: the belt lies inside the hull depth and (optionally) inside one zone."""
    checks = {"Z5b_belt_above_baseline": belt.z_lower >= 0.0,
              "Z5c_belt_below_deck": belt.z_upper <= depth + 1.0e-12,
              "depth": depth, "belt": [belt.z_lower, belt.z_upper]}
    if hull_zone_interval is not None:
        lo, hi = hull_zone_interval
        checks["Z5d_belt_inside_zone_interval"] = bool(lo <= belt.z_lower
                                                      and belt.z_upper <= hi)
    checks["passed"] = all(v for k, v in checks.items() if k.startswith("Z5"))
    return checks


@dataclass(frozen=True)
class DesignPatch:
    """One structural load patch: a *loaded area* and where it comes from."""

    label: str
    area: float
    basis: str = ""

    def __post_init__(self) -> None:
        if not (self.area > 0.0):
            raise ValueError(f"patch {self.label!r}: area must be positive")


def panel_patches(spacing: float, span_plate: float, span_stiffener: float,
                  span_primary: float) -> tuple[DesignPatch, ...]:
    """The three load patches of an ice-belt panel, each from its own geometry.

    plate field ``A = s l_plate``; stiffener ``A = s l_stiffener``; primary
    supporting member ``A = s l_primary``.  With ``l_plate <= l_stiffener <=
    l_primary`` the areas are ordered and the pressure-area law then gives a
    decreasing pressure down the hierarchy -- the physical reason the rule uses a
    different area for each member.
    """
    for name, val in (("spacing", spacing), ("span_plate", span_plate),
                      ("span_stiffener", span_stiffener),
                      ("span_primary", span_primary)):
        if not (val > 0.0):
            raise ValueError(f"{name} must be positive")
    if not (span_plate <= span_stiffener <= span_primary):
        raise ValueError("patch spans must be ordered plate <= stiffener <= primary")
    return (
        DesignPatch("plate_field", spacing * span_plate,
                    basis="s * l_plate (frame spacing x plate panel span)"),
        DesignPatch("stiffener", spacing * span_stiffener,
                    basis="s * l_stiffener (frame spacing x stiffener span)"),
        DesignPatch("primary_member", spacing * span_primary,
                    basis="s * l_primary (frame spacing x primary member span)"),
    )


def design_pressure_zones(law: PressureAreaLaw, zones: Sequence[HullZone],
                          patches: Sequence[DesignPatch]) -> dict:
    """Pressure, force and line load of every (hull zone x load patch) cell."""
    rows = []
    for zone in zones:
        for patch in patches:
            p = float(law.pressure(patch.area)) * zone.pressure_factor
            rows.append({
                "zone": zone.key,
                "zone_factor": zone.pressure_factor,
                "zone_length": zone.length,
                "zone_midpoint": zone.midpoint,
                "patch": patch.label,
                "patch_area_m2": patch.area,
                "patch_basis": patch.basis,
                "pressure_mpa": p / _PA_PER_MPA,
                "pressure_pa": p,
                "patch_force_N": p * patch.area,
                "line_load_N_per_m": p * math.sqrt(patch.area),
            })
    return {"rows": rows, "zone_count": len(zones), "patch_count": len(patches),
            "law": repr(law)}


def check_design_pressure_zones(table: dict, law: PressureAreaLaw,
                                zones: Sequence[HullZone],
                                patches: Sequence[DesignPatch],
                                tol: float = 1.0e-12) -> dict:
    """Audit the division: ordering, zone-factor linearity and force accounting.

    * **Z6 pressure ordering** -- inside one hull zone the pressure falls strictly
      as the loaded area grows (positive exponent) and the patch force grows.
    * **Z7 zone-factor linearity** -- the pressure ratio between two hull zones
      equals its factor ratio to machine precision (factors are multiplicative).
    * **Z8 force accounting** -- the per-row pressure recomputed from the law
      equals the tabulated one, and the length-weighted belt load equals the sum
      of the zone contributions.
    """
    rows = table["rows"]
    ordering_ok = True
    ordering_detail = []
    for zone in zones:
        sub = sorted([r for r in rows if r["zone"] == zone.key],
                     key=lambda r: r["patch_area_m2"])
        p = [r["pressure_pa"] for r in sub]
        f = [r["patch_force_N"] for r in sub]
        decreasing = all(b < a for a, b in zip(p, p[1:]))
        force_growing = all(b > a for a, b in zip(f, f[1:]))
        ordering_detail.append({"zone": zone.key,
                                "pressures_mpa": [x / _PA_PER_MPA for x in p],
                                "forces_N": f,
                                "pressure_strictly_decreasing": bool(decreasing),
                                "force_increasing": bool(force_growing)})
        ordering_ok = ordering_ok and decreasing and force_growing

    linearity_ok = True
    linearity_detail = []
    if len(zones) > 1:
        ref = zones[0]
        for zone in zones[1:]:
            for patch in patches:
                r_ref = next(r for r in rows
                             if r["zone"] == ref.key and r["patch"] == patch.label)
                r_z = next(r for r in rows
                           if r["zone"] == zone.key and r["patch"] == patch.label)
                ratio = r_z["pressure_pa"] / r_ref["pressure_pa"]
                expected = zone.pressure_factor / ref.pressure_factor
                err = abs(ratio / expected - 1.0)
                linearity_detail.append({"zone": zone.key, "patch": patch.label,
                                         "ratio": ratio, "expected": expected,
                                         "abs_rel_error": err})
                linearity_ok = linearity_ok and err <= tol

    force_ok = True
    force_detail = []
    for r in rows:
        recomputed = float(law.pressure(r["patch_area_m2"])) * r["zone_factor"]
        err = abs(recomputed / r["pressure_pa"] - 1.0)
        force_ok = force_ok and err <= 1.0e-10
        force_detail.append({"zone": r["zone"], "patch": r["patch"],
                             "rel_error": err})

    belt_load = sum(z.length * z.pressure_factor for z in zones)
    belt_load_check = (sum(r["zone_length"] * r["zone_factor"] for r in rows)
                       / max(1, len(patches)))
    belt_ok = abs(belt_load_check / belt_load - 1.0) <= tol

    return {
        "Z6_pressure_ordering": bool(ordering_ok),
        "Z6_detail": ordering_detail,
        "Z7_zone_factor_linearity": bool(linearity_ok),
        "Z7_detail": linearity_detail,
        "Z8_force_accounting": bool(force_ok and belt_ok),
        "Z8_detail": force_detail,
        "passed": bool(ordering_ok and linearity_ok and force_ok and belt_ok),
    }


def check_class_ordering(class_pressures: dict[str, float],
                         class_ranks: dict[str, int],
                         tol: float = 1.0e-12,
                         strict: bool = True) -> dict:
    """Z9: a heavier class (higher rank) must not carry a lower reference pressure.

    ``class_ranks`` is external ecosystem data (the ship-platform ice-class
    registry), so the ordering claim is checked against a registry rather than
    against an ordering this module wrote itself.
    """
    common = [k for k in class_pressures if k in class_ranks]
    if len(common) < 2:
        raise ValueError("need at least two classes present in both inputs")
    pairs = []
    ok = True
    for k0 in common:
        for k1 in common:
            if k0 == k1:
                continue
            if class_ranks[k0] > class_ranks[k1]:
                lhs, rhs = class_pressures[k0], class_pressures[k1]
                good = (lhs > rhs * (1.0 + tol) if strict
                        else lhs >= rhs * (1.0 - tol))
                pairs.append({"heavier": k0, "lighter": k1,
                              "p_heavier_mpa": lhs / _PA_PER_MPA,
                              "p_lighter_mpa": rhs / _PA_PER_MPA,
                              "non_decreasing_with_rank": bool(good)})
                ok = ok and good
    return {"Z9_class_ordering": bool(ok), "strict": bool(strict), "pairs": pairs,
            "passed": bool(ok), "ranks_source": "injected ice-class registry"}


# --------------------------------------------------------------------------- #
# 5. panel cross-validation: MITC4 vs Q4-SRI vs Navier series
# --------------------------------------------------------------------------- #
def navier_patch_centre(span_x: float, span_y: float, thickness: float,
                        young: float, poisson: float, pressure: float,
                        patch_area: float | None = None, terms: int = 199,
                        position: tuple[float, float] | None = None) -> float:
    """Kirchhoff Navier deflection of an SSSS plate under a centred ice patch.

    Load: uniform ``pressure`` over a centred rectangle whose area is
    ``patch_area`` and whose aspect ratio equals the plate's (the same patch
    convention as ``ice_belt_collapse.patch_pressure_load``); ``patch_area=None``
    means the whole plate.  Expanding the load in the plate's own eigenfunctions,

    .. math::

        q_{mn} = \\frac{4}{\\pi^2 m n}\\bigl(\\cos\\frac{m\\pi x_1}{a}
                 -\\cos\\frac{m\\pi x_2}{a}\\bigr)
                 \\bigl(\\cos\\frac{n\\pi y_1}{b}-\\cos\\frac{n\\pi y_2}{b}\\bigr) q,

        w(x,y) = \\frac{1}{\\pi^4 D}\\sum_{m,n}
                 \\frac{q_{mn}\\sin(m\\pi x/a)\\sin(n\\pi y/b)}
                      {\\bigl((m/a)^2+(n/b)^2\\bigr)^2},

    which reduces to the classical uniform-load series (Timoshenko &
    Woinowsky-Krieger 1959, section 30) when the patch is the whole plate.
    """
    if not (0.0 < span_x and 0.0 < span_y and 0.0 < thickness):
        raise ValueError("spans and thickness must be positive")
    if not (pressure >= 0.0):
        raise ValueError("pressure must be non-negative")
    rigidity = young * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    x, y = ((0.5 * span_x, 0.5 * span_y) if position is None else position)
    lx, ly = span_x, span_y
    if patch_area is None:
        x1, x2, y1, y2 = 0.0, span_x, 0.0, span_y
    else:
        if patch_area > span_x * span_y:
            raise ValueError("patch area exceeds the plate area")
        ratio = math.sqrt(patch_area / (span_x * span_y))
        hx, hy = 0.5 * ratio * span_x, 0.5 * ratio * span_y
        x1, x2 = 0.5 * span_x - hx, 0.5 * span_x + hx
        y1, y2 = 0.5 * span_y - hy, 0.5 * span_y + hy

    m = np.arange(1, int(terms) + 1, dtype=float)
    cx = np.cos(m * math.pi * x1 / lx) - np.cos(m * math.pi * x2 / lx)
    cy = np.cos(m * math.pi * y1 / ly) - np.cos(m * math.pi * y2 / ly)
    qx = 4.0 * pressure * cx / (math.pi ** 2 * m)
    qy = cy / m
    sx = np.sin(m * math.pi * x / lx)
    sy = np.sin(m * math.pi * y / ly)
    denom = (m[:, None] ** 2 / lx ** 2 + m[None, :] ** 2 / ly ** 2) ** 2
    terms_mn = (qx[:, None] * qy[None, :] * sx[:, None] * sy[None, :]) / denom
    return float(np.sum(terms_mn) / (math.pi ** 4 * rigidity))


def _mitc4_peak(model, load, fixed):
    """Elastic peak deflection of the ice-belt MITC4 model (independent route 1)."""
    from .ice_belt_collapse import solve_load_control

    path = solve_load_control(model, load, fixed, [1.0], tol=1e-9, max_iter=30)
    return abs(float(path.centre_w[-1])), int(path.iterations[-1])


def _q4sri_peak(side: float, thickness: float, young: float, poisson: float,
                function_pressure, nx: int) -> float:
    """Elastic peak deflection of the Q4-SRI torch plate (independent route 2)."""
    from .plate_mindlin import solve_plate_on_winkler_foundation

    res = solve_plate_on_winkler_foundation(
        nx=int(nx), side=float(side), young=float(young), poisson=float(poisson),
        thickness=float(thickness), winkler_k=0.0, pressure=function_pressure,
        boundary="simply_supported")
    w = res.displacement[0::3]
    return abs(float(w.max()))


def panel_cross_validation(law: PressureAreaLaw, *, span_x: float = 1.0,
                           span_y: float = 1.0, thickness: float = 0.010,
                           young: float = 2.06e11, poisson: float = 0.3,
                           patch_fraction: float | None = None,
                           load_factor: float = 1.0,
                           n_mitc: int = 15, n_sri: int = 16,
                           navier_terms: int = 199,
                           tolerance: float = 0.10) -> dict:
    """Cross-validate the ice-belt panel on three independent discretisations.

    ``patch_fraction=None`` loads the whole plate field with
    ``p = law.pressure(span_x*span_y)``; otherwise a centred patch of area
    ``patch_fraction * span_x * span_y`` is loaded with ``p = law.pressure(A)``,
    so the pressure magnitude itself comes from the pressure-area law.

    Anti-circularity: the Q4-SRI route builds its own mesh and load vector inside
    ``tensorfem.plate_mindlin``; the Navier route shares no mesh, element or
    solver code with either; the MITC4 route is the numpy layered shell the
    collapse chain uses.  Only the *problem* (geometry, material, pressure) is
    common.
    """
    from .ice_belt_collapse import (PlateModel, elastic_perfectly_plastic,
                                    inplane_free_bcs, patch_pressure_load,
                                    structured_mesh, uniform_pressure_load)

    young = float(young)
    poisson = float(poisson)
    if abs(span_x - span_y) > 1.0e-12:
        raise ValueError("span_x == span_y is required: the Q4-SRI route is a "
                         "square-mesh plate solver")

    area_full = span_x * span_y
    patch_area = None if patch_fraction is None else patch_fraction * area_full
    pressure = float(law.pressure(area_full if patch_area is None else patch_area)) \
        * float(load_factor)

    # D1: MITC4 layered shell (numpy)
    nodes, elems = structured_mesh(span_x, span_y, int(n_mitc), int(n_mitc))
    model = PlateModel(nodes, elems, thickness, elastic_perfectly_plastic())
    model.von_karman = False
    if patch_area is None:
        load = uniform_pressure_load(model, pressure)
    else:
        load = patch_pressure_load(model, pressure, patch_area)
    w_mitc, iterations = _mitc4_peak(model, load, inplane_free_bcs(model))

    # D2: Q4-SRI torch plate (separate implementation)
    if patch_area is None:
        def fn(x, y):
            return pressure
    else:
        ratio = math.sqrt(patch_area / area_full)
        hx, hy = 0.5 * ratio * span_x, 0.5 * ratio * span_y
        cx, cy = 0.5 * span_x, 0.5 * span_y

        def fn(x, y):
            inside = (abs(float(x) - cx) <= hx + 1e-12) and (abs(float(y) - cy) <= hy + 1e-12)
            return pressure if inside else 0.0
    w_sri = _q4sri_peak(span_x, thickness, young, poisson, fn, n_sri)

    # D3: Kirchhoff Navier double series (analytic, no mesh)
    w_nav = navier_patch_centre(span_x, span_y, thickness, young, poisson,
                                pressure, patch_area=patch_area,
                                terms=navier_terms)

    def rel(a, b):
        return abs(a / b - 1.0)

    pairs = {
        "mitc4_vs_q4sri": rel(w_mitc, w_sri),
        "mitc4_vs_navier": rel(w_mitc, w_nav),
        "q4sri_vs_navier": rel(w_sri, w_nav),
    }
    return {
        "case": "full_field" if patch_area is None else "centred_patch",
        "geometry": {"span_x": span_x, "span_y": span_y, "thickness": thickness,
                     "loaded_area_m2": area_full if patch_area is None else patch_area},
        "material": {"young": young, "poisson": poisson},
        "pressure_pa": pressure, "pressure_mpa": pressure / _PA_PER_MPA,
        "load_factor": float(load_factor),
        "w_over_t": (max(abs(w_mitc), abs(w_sri), abs(w_nav)) / thickness),
        "pressure_source": "load_factor x pressure-area law at the loaded area",
        "routes": {
            "mitc4_wpeak_m": w_mitc,
            "q4sri_wpeak_m": w_sri,
            "navier_wpeak_m": w_nav,
        },
        "mitc4_iterations": iterations,
        "meshes": {"mitc4": [int(n_mitc), int(n_mitc)], "q4sri": [int(n_sri), int(n_sri)]},
        "pairwise_rel_diff": pairs,
        "max_pairwise_rel_diff": max(pairs.values()),
        "tolerance": tolerance,
        "passed": bool(max(pairs.values()) <= tolerance),
    }


def panel_mesh_study(law: PressureAreaLaw, *, span_x: float = 1.0,
                     span_y: float = 1.0, thickness: float = 0.010,
                     young: float = 2.06e11, poisson: float = 0.3,
                     patch_fraction: float | None = None,
                     load_factor: float = 1.0,
                     meshes: Sequence[int] = (9, 15, 25, 35),
                     navier_terms: int = 199) -> dict:
    """Discretisation study of the MITC4 route against the Navier reference."""
    from .ice_belt_collapse import (PlateModel, elastic_perfectly_plastic,
                                    inplane_free_bcs, patch_pressure_load,
                                    structured_mesh, uniform_pressure_load)

    area_full = span_x * span_y
    patch_area = None if patch_fraction is None else patch_fraction * area_full
    pressure = float(law.pressure(area_full if patch_area is None else patch_area)) \
        * float(load_factor)
    w_nav = navier_patch_centre(span_x, span_y, thickness, young, poisson,
                                pressure, patch_area=patch_area,
                                terms=navier_terms)
    rows = []
    for n in meshes:
        nodes, elems = structured_mesh(span_x, span_y, int(n), int(n))
        model = PlateModel(nodes, elems, thickness, elastic_perfectly_plastic())
        model.von_karman = False
        load = (uniform_pressure_load(model, pressure) if patch_area is None
                else patch_pressure_load(model, pressure, patch_area))
        w, _ = _mitc4_peak(model, load, inplane_free_bcs(model))
        rows.append({"elements_per_side": int(n), "wpeak_m": w,
                     "w_navier_m": w_nav, "rel_error": w / w_nav - 1.0})
    return {"case": "full_field" if patch_area is None else "centred_patch",
            "navier_wpeak_m": w_nav, "rows": rows}


# --------------------------------------------------------------------------- #
# 6. the H4 acceptance bundle
# --------------------------------------------------------------------------- #
#: Areas at which the pressure-area curve is tabulated in the artifact (m2).
CURVE_AREAS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)

#: Hull-area end fractions of the demonstration ship (inputs, not rule values).
DEMO_ZONE_FRACTIONS = (("stern", 0.20), ("midbody", 0.60),
                       ("bow_intermediate", 0.85), ("bow", 1.00))

#: Demonstration hull-area factors (mechanism only; real values are rule data).
DEMO_ZONE_FACTORS = {"stern": 0.75, "midbody": 0.85,
                     "bow_intermediate": 1.00, "bow": 1.25}

#: Demonstration class factors relative to the heaviest class (mechanism only:
#: the *ordering* is checked against the injected registry, the numbers are not
#: rule values and must never be used for design).
DEMO_CLASS_FACTORS = {"PC1": 1.15, "PC2": 1.05, "PC3": 0.95, "PC4": 0.85,
                      "PC5": 0.75, "PC6": 0.65, "PC7": 0.55,
                      "IAS": 0.50, "IA": 0.40, "IB": 0.30, "IC": 0.22}


def elastic_load_factor(law: PressureAreaLaw, *, span_x: float, span_y: float,
                        thickness: float, young: float = 2.06e11,
                        poisson: float = 0.3,
                        patch_fraction: float | None = None,
                        target_w_over_t: float = 0.02) -> dict:
    """Load factor that puts the panel in the small-deflection (linear) range.

    The pressure-area law returns a *design* pressure; at that level an ice-belt
    plate field is well past first yield and the discretisation comparison would
    be measuring the nonlinear solution path instead of the discretisation.  The
    factor is estimated from the analytic Navier deflection per unit pressure, so
    the target ``w/t`` is reached by construction, and
    :func:`check_load_linearity` proves afterwards that the response really is
    linear at that level.
    """
    area_full = span_x * span_y
    patch_area = None if patch_fraction is None else patch_fraction * area_full
    p_design = float(law.pressure(area_full if patch_area is None else patch_area))
    w_per_pa = navier_patch_centre(span_x, span_y, thickness, young, poisson, 1.0,
                                   patch_area=patch_area)
    target = target_w_over_t * thickness
    factor = target / (w_per_pa * p_design)
    return {"load_factor": factor, "p_design_pa": p_design,
            "p_design_mpa": p_design / _PA_PER_MPA,
            "p_validation_pa": p_design * factor,
            "p_validation_mpa": p_design * factor / _PA_PER_MPA,
            "navier_w_per_pa": w_per_pa, "target_w_over_t": target_w_over_t,
            "note": ("design pressure governs the collapse chain; the "
                     "discretisation cross-check runs at the same law shape in "
                     "the linear range")}


def check_load_linearity(law: PressureAreaLaw, *, span_x: float, span_y: float,
                         thickness: float, load_factor: float,
                         patch_fraction: float | None = None,
                         young: float = 2.06e11, poisson: float = 0.3,
                         n_mitc: int = 15, tol: float = 1.0e-4) -> dict:
    """Prove the cross-validation case is linear: ``w(2p)/w(p) == 2``."""
    low = panel_cross_validation(law, span_x=span_x, span_y=span_y,
                                 thickness=thickness, patch_fraction=patch_fraction,
                                 load_factor=load_factor, n_mitc=n_mitc,
                                 n_sri=8, young=young, poisson=poisson)
    high = panel_cross_validation(law, span_x=span_x, span_y=span_y,
                                  thickness=thickness, patch_fraction=patch_fraction,
                                  load_factor=2.0 * load_factor, n_mitc=n_mitc,
                                  n_sri=8, young=young, poisson=poisson)
    ratio = high["routes"]["mitc4_wpeak_m"] / low["routes"]["mitc4_wpeak_m"]
    return {"w_low_m": low["routes"]["mitc4_wpeak_m"],
            "w_high_m": high["routes"]["mitc4_wpeak_m"],
            "ratio": ratio, "expected": 2.0, "tol": tol,
            "passed": bool(abs(ratio / 2.0 - 1.0) <= tol)}


def run_h4_verification(*, p_ref: float = 2.0e6, area_ref: float = 1.0,
                        exponent: float = 0.3, c_d: float = 1.0, c_f: float = 1.0,
                        provenance: str = (
                            "DEMONSTRATION anchor p_ref=2.0 MPa: mechanism check "
                            "only, NOT a rule value"),
                        class_ranks: dict[str, int] | None = None,
                        class_pressures_mpa: dict[str, float] | None = None,
                        rule_table: RulePressureTable | None = None,
                        ship_length: float = 120.0, depth: float = 12.0,
                        scantling_draught: float = 7.0,
                        belt_lower_offset: float = 2.9,
                        belt_upper_offset: float = 2.6,
                        spacing: float = 0.35, span_plate: float = 1.6,
                        span_stiffener: float = 3.2, span_primary: float = 6.4,
                        thickness: float = 0.010) -> dict:
    """Run the three H4 round-1 criteria and return the machine-check bundle.

    Criterion 1 -- pressure-area curve: point-by-point table on three independent
    evaluation routes plus scaling/quadrature/inverse-consistency checks.  The
    comparison against *IACS UR PC rule values* is reported as ``WITHHELD``
    unless a rule table is injected, because the rule text is not available
    offline and inventing values is not allowed.
    Criterion 2 -- plate-panel cross-validation: MITC4 vs Q4-SRI vs Navier for a
    full field and a local patch, <10 % pairwise.
    Criterion 3 -- design-pressure zone division: tiling, belt containment,
    ordering, factor linearity, force accounting and class-registry ordering.
    """
    law = PressureAreaLaw(p_ref=p_ref, area_ref=area_ref, exponent=exponent,
                          c_d=c_d, c_f=c_f, provenance=provenance)

    # -- criterion 1: the curve ------------------------------------------- #
    curve = check_curve_routes(law, CURVE_AREAS)
    quadrature = check_force_quadrature(law, CURVE_AREAS[0], CURVE_AREAS[-1])
    if rule_table is None:
        rule_cmp = {
            "status": "WITHHELD",
            "fail_stage": "rule-table ingestion",
            "failure_reason": (
                "IACS UR I2 local-pressure table (p_ref per Polar Class at the "
                "rule reference area) is not available offline in this "
                "worktree and is not exposed by the ship platform API; the "
                "repository policy forbids embedding invented rule constants, "
                "so no ref-vs-got numbers against the rule are claimed. "
                "compare_with_rule_table() is ready to consume the table as "
                "soon as it is injected as data (RulePressureTable with "
                "is_rule_data=True, provenance = rule edition + clause)."),
            "ready_api": "compare_with_rule_table(law, RulePressureTable(...))",
        }
    else:
        rule_cmp = compare_with_rule_table(law, rule_table)
        rule_cmp["status"] = "PASS" if rule_cmp["passed"] else "FAIL"
    crit1 = {
        "name": "pressure-area curve implementation",
        "curve_routes_passed": curve["passed"],
        "max_rel_diff_logspace": curve["max_rel_diff_logspace"],
        "max_rel_diff_marginal": curve["max_rel_diff_marginal"],
        "max_rel_diff_radial_mean": curve["max_rel_diff_radial_mean"],
        "max_rel_diff_scaling_invariant": curve["max_rel_diff_scaling_invariant"],
        "max_rel_diff_area_inverse": curve["max_rel_diff_area_inverse"],
        "loglog_slope_between_endpoints": curve["loglog_slope_between_endpoints"],
        "loglog_slope_error_vs_exponent": curve["loglog_slope_error_vs_exponent"],
        "fitted_exponent": curve["fitted_exponent"],
        "fitted_exponent_error": curve["fitted_exponent_error"],
        "fit_max_residual": curve["fit_max_residual"],
        "quadrature_passed": quadrature["passed"],
        "quadrature_rel_error": quadrature["rel_error"],
        "rule_table_comparison": rule_cmp,
        "status": ("PASS" if curve["passed"] and quadrature["passed"]
                   and rule_cmp["status"] == "PASS"
                   else "PARTIAL_WITHHELD" if curve["passed"] and quadrature["passed"]
                   else "FAIL"),
    }

    # -- criterion 2: panel cross-validation ------------------------------- #
    lf_full = elastic_load_factor(law, span_x=span_plate, span_y=span_plate,
                                  thickness=thickness)
    lf_patch = elastic_load_factor(law, span_x=span_plate, span_y=span_plate,
                                   thickness=thickness, patch_fraction=0.25)
    cv_full = panel_cross_validation(law, span_x=span_plate, span_y=span_plate,
                                     thickness=thickness,
                                     load_factor=lf_full["load_factor"])
    cv_patch = panel_cross_validation(law, span_x=span_plate, span_y=span_plate,
                                      thickness=thickness,
                                      patch_fraction=0.25,
                                      load_factor=lf_patch["load_factor"])
    lin_full = check_load_linearity(law, span_x=span_plate, span_y=span_plate,
                                    thickness=thickness,
                                    load_factor=lf_full["load_factor"])
    lin_patch = check_load_linearity(law, span_x=span_plate, span_y=span_plate,
                                     thickness=thickness,
                                     patch_fraction=0.25,
                                     load_factor=lf_patch["load_factor"])
    mesh_study = panel_mesh_study(law, span_x=span_plate, span_y=span_plate,
                                  thickness=thickness,
                                  load_factor=lf_full["load_factor"],
                                  meshes=(7, 11, 15, 21))
    crit2 = {
        "name": "plate-panel response, two independent discretisations",
        "loaded_panel": {"span_x": span_plate, "span_y": span_plate,
                         "thickness": thickness},
        "load_level_full_field": lf_full,
        "load_level_centred_patch": lf_patch,
        "full_field": cv_full,
        "centred_patch": cv_patch,
        "linearity_full_field": lin_full,
        "linearity_centred_patch": lin_patch,
        "mesh_study": mesh_study,
        "tolerance": 0.10,
        "status": "PASS" if (cv_full["passed"] and cv_patch["passed"]
                             and lin_full["passed"] and lin_patch["passed"])
                  else "FAIL",
    }

    # -- criterion 3: design-pressure zones -------------------------------- #
    zones = partition_hull_zones(ship_length, DEMO_ZONE_FRACTIONS,
                                 DEMO_ZONE_FACTORS)
    patches = panel_patches(spacing, span_plate, span_stiffener, span_primary)
    partition = check_zone_partition(zones, ship_length)
    belt = ice_belt_extent(scantling_draught, belt_lower_offset,
                           belt_upper_offset,
                           provenance="caller-supplied belt offsets (rule data)")
    belt_check = check_belt_containment(belt, depth)
    table = design_pressure_zones(law, zones, patches)
    zone_checks = check_design_pressure_zones(table, law, zones, patches)

    if class_ranks is None:
        class_check = {"status": "WITHHELD",
                       "failure_reason": "no ice-class registry injected"}
    else:
        cp = class_pressures_mpa
        if cp is None:
            cp = {k: law.p_ref_mpa * DEMO_ZONE_FACTORS.get("bow", 1.0)
                  for k in class_ranks}
        class_check = check_class_ordering(cp, class_ranks)
        class_check["status"] = "PASS" if class_check["passed"] else "FAIL"
    crit3 = {
        "name": "design pressure zone division",
        "ship_length": ship_length, "depth": depth,
        "zones": [{"key": z.key, "x_start": z.x_start, "x_end": z.x_end,
                   "length": z.length, "factor": z.pressure_factor}
                  for z in zones],
        "patches": [{"label": p.label, "area_m2": p.area, "basis": p.basis}
                    for p in patches],
        "belt": {"z_lower": belt.z_lower, "z_upper": belt.z_upper,
                 "height": belt.height, "provenance": belt.provenance},
        "partition_check": partition,
        "belt_check": belt_check,
        "zone_checks": zone_checks,
        "class_check": class_check,
        "status": ("FAIL" if not (partition["passed"] and belt_check["passed"]
                                  and zone_checks["passed"])
                   else "PASS" if class_check.get("status") == "PASS"
                   else "PARTIAL_WITHHELD"),
    }

    statuses = (crit1["status"], crit2["status"], crit3["status"])
    if all(st == "PASS" for st in statuses):
        overall = "PASS"
    elif any(st == "FAIL" for st in statuses):
        overall = "FAIL"
    else:
        overall = "PARTIAL_WITHHELD"
    withheld = [k for k, v in (("criterion_1.rule_table_comparison", crit1["rule_table_comparison"]),
                               ("criterion_3.class_check", class_check))
                if v.get("status") == "WITHHELD"]
    return {
        "mission": "H4-ICE-BELT-FEM",
        "round": 1,
        "law": {"p_ref_mpa": law.p_ref_mpa, "area_ref_m2": law.area_ref,
                "exponent": law.exponent, "c_d": law.c_d, "c_f": law.c_f,
                "provenance": law.provenance,
                "form": "p(A) = C_d C_f p_ref (A/A_ref)**-alpha"},
        "criterion_1_pressure_area_curve": crit1,
        "criterion_2_panel_cross_validation": crit2,
        "criterion_3_design_pressure_zones": crit3,
        "curve_table": curve["rows"],
        "pressure_zone_table": table["rows"],
        "withheld_items": withheld,
        "overall_status": overall,
    }


def fetch_ice_class_ranks(base_url: str | None = None, timeout: float = 5.0) -> dict:
    """Ice-class ranks from the ship-platform registry (ecosystem data injection).

    Domain knowledge (which class is heavier) belongs to the ship platform; this
    module only consumes it.  Any failure returns ``{}`` so the caller reports the
    class-ordering check as WITHHELD rather than guessing an order.
    """
    import os
    import urllib.error
    import urllib.request

    base = (base_url or os.environ.get("TENSOR_SHIP_API_BASE")
            or "http://127.0.0.1:8002")
    url = base.rstrip("/") + "/api/v1/polaris/ice-classes"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            blob = json.loads(resp.read())
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return {}
    ranks = {}
    for entry in blob.get("ice_classes", []):
        key, rank = entry.get("key"), entry.get("rank")
        if isinstance(key, str) and isinstance(rank, (int, float)):
            ranks[key] = float(rank)
    return ranks


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI
    """`python -m tensorfem.ice_belt_pressure_area [--out PATH]`."""
    import argparse

    ap = argparse.ArgumentParser(description="H4 ice-belt pressure-area / panel "
                                             "cross-validation / zone division")
    ap.add_argument("--out", default=None, help="write the JSON artifact here")
    ap.add_argument("--p-ref-mpa", type=float, default=2.0)
    ap.add_argument("--exponent", type=float, default=0.3)
    ap.add_argument("--no-registry", action="store_true",
                    help="skip the ship-platform ice-class registry query")
    args = ap.parse_args(argv)

    ranks = {} if args.no_registry else fetch_ice_class_ranks()
    p_ref = args.p_ref_mpa * _PA_PER_MPA
    class_pressures = ({k: p_ref * DEMO_CLASS_FACTORS[k]
                        for k in ranks if k in DEMO_CLASS_FACTORS}
                       if ranks else None)
    bundle = run_h4_verification(p_ref=p_ref, exponent=args.exponent,
                                 class_ranks=ranks or None,
                                 class_pressures_mpa=class_pressures)
    text = json.dumps(bundle, indent=2, sort_keys=False)
    if args.out:
        import pathlib

        path = pathlib.Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        print(f"wrote {path}")
    else:
        print(text)
    print(bundle["overall_status"])
    return 0 if bundle["overall_status"] != "FAIL" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
