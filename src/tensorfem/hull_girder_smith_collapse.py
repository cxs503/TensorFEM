"""Smith progressive-collapse analysis of a hull-girder cross-section.

Why this module exists
----------------------
``tensorfem.hull_girder_ultimate`` integrates an *ideal elastic-plastic*
rectangular section: every fibre yields at ``sigma_y`` and never softens, so
its moment-curvature curve is monotone and there is no ultimate state.
``tensorfem.plate_stiffener_ultimate.smith_mk_curve`` is an *element-level*
limiting-stress model: an element is elastic up to a supplied
``compression_strength`` and constant afterwards, again without softening and
therefore without a peak.

A genuine Smith (1977) progressive-collapse analysis needs three ingredients
that neither of those surfaces provides:

1. component stress-strain curves that buckle (plate field), buckle as a
   column (stiffener + attached plate) and then *soften* after their own peak;
2. an incremental section solve that keeps axial equilibrium while the neutral
   axis migrates, curvature step by curvature step, so that the *section*
   moment can peak and drop;
3. an energy ledger that closes against ``integral M dkappa``, which is the
   only independent check that the incremental path is consistent.

This module supplies exactly that: :class:`StressStrainLaw` (tabulated,
energy-consistent component curve), the incremental section solver
:func:`solve_moment_curvature` (neutral-axis equilibrium per increment, energy
ledger, peak detection), a documented double-bottom geometry template
(:class:`DoubleBottomGeometry`, ``ISSC2009_DOUBLE_BOTTOM``) and the benchmark
runner :func:`run_issc2009_double_bottom`.

Physical scope and conventions
------------------------------
* Plane sections remain plane; no shear lag, no torsion, no load-introduction
  effects, no fracture.
* ``y`` is measured upward from the outer-bottom plating mid-surface.
* Curvature ``kappa > 0`` puts ``y < y_na`` in compression, i.e. the bottom is
  compressed (sagging for a ship section).
* Strains are positive in tension.  Component curves are supplied in that
  convention; the plate-field curve is flipped from the
  compression-positive convention of
  :class:`~.smith_progressive_collapse.PlateStrip`.
* Loading is monotone, so the ledger tracks stored + dissipated energy
  (``sigma depsilon >= 0`` along the path) and therefore closes over the whole
  *including* the post-ultimate softening branch.

Verification surface
--------------------
* :func:`verify_ideal_plastic_section` compares the incremental engine with the
  closed-form ideal elastic-plastic rectangular solution (a code-independent
  oracle).
* The benchmark runner reports criteria C1..C6 of the campaign node, including
  an explicit ``NOT_VERIFIED`` status when no experimental anchor is available
  in the environment (honest-FAIL discipline: it is never fabricated).
* :data:`PROVENANCE` lists, item by item, what is derived here, what is an
  engineering assumption, and what is withheld.

Withheld anchors (explicit)
---------------------------
* The ISSC 2009 double-bottom *experimental* ultimate moment is **not**
  available in this repository or in this execution environment.  The runner
  therefore ships an injection point (:class:`UltimateMomentAnchor`,
  :func:`load_anchor`) and reports C1 as ``NOT_VERIFIED`` until a documented
  anchor file is supplied.  No number is invented.
* The geometry template beyond ``L x B x H = 6.0 x 3.0 x 2.0 m``,
  ``t = 8..12 mm`` and ``sigma_y = 315 MPa`` (the values fixed by the campaign
  node) is an **engineering assumption**, listed field by field in
  :class:`DoubleBottomGeometry` and varied in the thickness sweeps.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

from .smith_progressive_collapse import (
    IMPERFECTION_SENSITIVITY,
    PlateStrip,
    TeeStiffener,
)

__all__ = [
    "PROVENANCE",
    "CRITERIA",
    "StressStrainLaw",
    "SectionFiber",
    "CollapsePoint",
    "MomentCurvatureResult",
    "elastic_perfectly_plastic_law",
    "plate_field_law",
    "column_law",
    "interaction_law",
    "curvature_grid",
    "solve_moment_curvature",
    "DoubleBottomGeometry",
    "ISSC2009_DOUBLE_BOTTOM",
    "plastic_neutral_axis",
    "verify_ideal_plastic_section",
    "limiting_stress_envelope",
    "UltimateMomentAnchor",
    "load_anchor",
    "run_issc2009_double_bottom",
]

PROVENANCE = {
    "derived_here": (
        "incremental neutral-axis-equilibrium section solver; energy ledger; "
        "component-law composition; double-bottom section assembly; criteria "
        "aggregation"
    ),
    "reused_verified_curves": (
        "PlateStrip (Faulkner-anchored effective-width plate field with "
        "post-ultimate softening) and TeeStiffener (Perry-Robertson column + "
        "web/flange local buckling) from tensorfem.smith_progressive_collapse"
    ),
    "engineering_assumptions": (
        "geometry template fields other than L/B/H the t-range and sigma_y "
        "(girder count 3, floor spacing 1.5 m, flange 150x10 mm, member "
        "thickness split); column length of a longitudinal girder = model "
        "length (no intermediate transverse support inside the specimen); "
        "imperfection-reference convention (fixed w0/t = 0.1, see "
        "DoubleBottomGeometry.imperfection_reference); one panel-average "
        "component curve applied fibre-wise inside a member"
    ),
    "withheld": (
        "ISSC 2009 double-bottom experimental ultimate moment (not available "
        "in this environment -> criterion C1 stays NOT_VERIFIED, never "
        "invented); Gordo & Soares (1993) plate-strip relations (source "
        "unavailable); IACS CSR clause coefficients (caller supplied in the "
        "parent module)"
    ),
}

CRITERIA = {
    "C1": "M_u vs ISSC 2009 double-bottom experimental anchor <= 15%",
    "C2": "elastic M-kappa slope vs E*I <= 5%",
    "C3": "imperfection sensitivity w0/t=0.1 -> <=10%, w0/t=0.5 -> <=30% drop",
    "C4": "energy-ledger residual <= 5%",
    "C5": "mesh convergence 10/20/40 fibres: M_u spread <= 5%",
    "C6": "pytest suite >= 15 tests, all green",
}


# --------------------------------------------------------------------------
# 0. validation helpers
# --------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


# --------------------------------------------------------------------------
# 1. energy-consistent component stress-strain law
# --------------------------------------------------------------------------
class StressStrainLaw:
    """Tabulated component stress-strain curve with a consistent energy ledger.

    The law is tabulated once on a symmetric strain grid; the stress comes from
    ``sigma_of`` and the specific energy ``U(eps) = integral_0^eps sigma deps``
    is accumulated with the trapezoidal rule *on the same grid*.  Because the
    ledger, the section moment and the reported energies all read the same
    table, the only remaining ledger error is the curvature discretisation
    (checked in criterion C4).

    Parameters
    ----------
    sigma_of:
        Callable ``strain -> stress`` in Pa, with ``sigma_of(0) == 0``.
    young, yield_stress:
        Reference elastic modulus and yield stress of the component; used for
        the yielded-area diagnostic and for the default strain range.
    maximum_strain:
        Half-width of the tabulation range; defaults to twenty yield strains
        each way, which covers the curvature sweep of the section solver.
    points:
        Odd number of tabulation points (default 1501, i.e. a strain step of
        ``4e-5`` over the +-20 yield-strain range).
    """

    def __init__(self, sigma_of, *, young: float, yield_stress: float,
                 maximum_strain: float | None = None,
                 points: int = 1501) -> None:
        if not callable(sigma_of):
            raise TypeError("sigma_of must be callable")
        self.young = _positive("young", young)
        self.yield_stress = _positive("yield_stress", yield_stress)
        if isinstance(points, bool) or not isinstance(points, int):
            raise TypeError("points must be an integer")
        if points < 101 or points % 2 == 0:
            raise ValueError("points must be an odd integer >= 101")
        max_strain = (20.0 * self.yield_stress / self.young
                      if maximum_strain is None
                      else _positive("maximum_strain", maximum_strain))
        step = 2.0 * max_strain / (points - 1)
        centre = (points - 1) // 2
        strains = [-max_strain + i * step for i in range(points)]
        strains[centre] = 0.0
        stresses = []
        for strain in strains:
            value = sigma_of(strain)
            if not math.isfinite(value):
                raise ValueError("sigma_of returned a non-finite stress")
            stresses.append(float(value))
        if abs(stresses[centre]) > 1e-9 * self.yield_stress:
            raise ValueError("sigma_of(0) must be zero")
        energies = [0.0] * points
        for i in range(centre + 1, points):
            energies[i] = energies[i - 1] + 0.5 * (stresses[i] + stresses[i - 1]) * step
        for i in range(centre - 1, -1, -1):
            energies[i] = energies[i + 1] - 0.5 * (stresses[i] + stresses[i + 1]) * step
        self.maximum_strain = max_strain
        self._step = step
        self._strains = tuple(strains)
        self._stresses = tuple(stresses)
        self._energies = tuple(energies)

    # -- access -----------------------------------------------------------
    def stress(self, strain: float) -> float:
        """Stress at ``strain`` (clamped to the tabulated range)."""
        return self._interp(self._stresses, _finite("strain", strain))

    def energy(self, strain: float) -> float:
        """Specific energy ``integral_0^eps sigma deps`` at ``strain``.

        The integral of the *piecewise-linear interpolant* of the tabulated
        stress is evaluated exactly (quadratic in the local offset), so the
        energy is consistent with :meth:`stress` to machine precision and does
        not inherit the linear-interpolation error of the quadratic energy
        table near ``eps = 0``.
        """
        limit = self.maximum_strain
        eps = min(limit, max(-limit, _finite("strain", strain)))
        position = (eps + limit) / self._step
        index = int(math.floor(position))
        if index >= len(self._strains) - 1:
            index = len(self._strains) - 2
            offset = eps - self._strains[index]
            return (self._energies[index] + self._stresses[index] * offset
                    + 0.5 * (self._stresses[index + 1] - self._stresses[index])
                    / self._step * offset ** 2)
        offset = eps - self._strains[index]
        return (self._energies[index] + self._stresses[index] * offset
                + 0.5 * (self._stresses[index + 1] - self._stresses[index])
                / self._step * offset ** 2)

    def _interp(self, table: tuple[float, ...], strain: float) -> float:
        limit = self.maximum_strain
        eps = min(limit, max(-limit, strain))
        position = (eps + limit) / self._step
        index = int(math.floor(position))
        if index >= len(table) - 1:
            return table[-1]
        frac = position - index
        return table[index] * (1.0 - frac) + table[index + 1] * frac

    @property
    def yield_strain(self) -> float:
        return self.yield_stress / self.young

    @property
    def peak_compressive_stress(self) -> float:
        """Largest compressive stress magnitude reached by the curve (Pa)."""
        return max(-value for value in self._stresses if value < 0.0)

    def is_yielded(self, strain: float) -> bool:
        """True once the tabulated stress has reached the yield plateau."""
        return abs(self.stress(strain)) >= self.yield_stress * (1.0 - 1e-9)

    def table(self) -> dict[str, tuple[float, ...]]:
        return {"strain": self._strains, "stress": self._stresses,
                "energy": self._energies}


def elastic_perfectly_plastic_law(*, young: float, yield_stress: float,
                                  residual_ratio: float = 1.0) -> StressStrainLaw:
    """Symmetric elastic-perfectly plastic law (hard corners, compact flanges)."""
    e = _positive("young", young)
    sy = _positive("yield_stress", yield_stress)
    cap = sy * _positive("residual_ratio", residual_ratio)

    def sigma_of(strain: float) -> float:
        magnitude = min(abs(strain) * e, cap)
        return math.copysign(magnitude, strain)

    return StressStrainLaw(sigma_of, young=e, yield_stress=sy)


def plate_field_law(strip: PlateStrip, *, tension_ratio: float = 1.0) -> StressStrainLaw:
    """Section law of one plate field: buckling and softening in compression.

    Compression is taken from :meth:`PlateStrip.stress` (compression-positive;
    elastic -> effective-width rise -> peak -> softening); tension is
    elastic-perfectly plastic at ``tension_ratio * sigma_y``.
    """
    strip.validate()
    e = strip.young
    sy = strip.yield_stress
    cap = sy * _positive("tension_ratio", tension_ratio)

    def sigma_of(strain: float) -> float:
        if strain >= 0.0:
            return min(strain * e, cap)
        return -strip.stress(-strain)

    return StressStrainLaw(sigma_of, young=e, yield_stress=sy)


def column_law(stiffener: TeeStiffener, *, tension_ratio: float = 1.0) -> StressStrainLaw:
    """Section law of a stiffener unit (column + local buckling, softening)."""
    stiffener.validate()
    e = stiffener.young
    sy = stiffener.yield_stress
    cap = sy * _positive("tension_ratio", tension_ratio)

    def sigma_of(strain: float) -> float:
        if strain >= 0.0:
            return min(strain * e, cap)
        return -stiffener.stress(-strain)

    return StressStrainLaw(sigma_of, young=e, yield_stress=sy)


def interaction_law(laws, *, label: str = "") -> StressStrainLaw:
    """Pointwise minimum of several laws (independent mechanisms)."""
    laws = tuple(laws)
    if len(laws) < 2:
        raise ValueError("interaction_law needs at least two laws")
    young = laws[0].young
    sy = min(law.yield_stress for law in laws)
    limit = min(law.maximum_strain for law in laws)

    def sigma_of(strain: float) -> float:
        return min(law.stress(strain) for law in laws)

    law = StressStrainLaw(sigma_of, young=young, yield_stress=sy,
                          maximum_strain=limit)
    law.label = label
    return law


# --------------------------------------------------------------------------
# 2. section assembly and incremental solver
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionFiber:
    """One fibre: area, height above the section baseline, and its law."""

    name: str
    area: float
    y: float
    law: StressStrainLaw

    def validate(self) -> "SectionFiber":
        _positive("area", self.area)
        _finite("y", self.y)
        if not isinstance(self.law, StressStrainLaw):
            raise TypeError("law must be a StressStrainLaw")
        return self


@dataclass(frozen=True)
class CollapsePoint:
    """One equilibrium state on the moment-curvature path."""

    curvature: float
    neutral_axis: float
    moment: float
    axial_resultant: float
    yielded_fraction: float
    internal_energy: float
    external_work: float


@dataclass(frozen=True)
class MomentCurvatureResult:
    """Moment-curvature path plus the derived ultimate state."""

    points: tuple[CollapsePoint, ...]
    ultimate_moment: float
    ultimate_curvature: float
    first_yield_moment: float
    first_yield_curvature: float
    elastic_slope: float
    ledger_residual: float
    peak_is_interior: bool

    def curve(self, stride: int = 1) -> list[dict[str, float]]:
        rows = []
        for index in range(0, len(self.points), max(1, stride)):
            point = self.points[index]
            rows.append({
                "curvature": point.curvature,
                "moment": point.moment,
                "neutral_axis": point.neutral_axis,
                "yielded_fraction": point.yielded_fraction,
                "internal_energy": point.internal_energy,
                "external_work": point.external_work,
            })
        return rows


def curvature_grid(yield_curvature: float, *, steps: int = 320,
                   maximum_ratio: float = 8.0, exponent: float = 2.0) -> tuple[float, ...]:
    """Curvature sweep starting at zero with quadratic spacing near the origin.

    The quadratic law puts most points in the elastic range (where the moment
    rises fast) while still reaching ``maximum_ratio * kappa_y`` to capture the
    post-ultimate descent.
    """
    ky = _positive("yield_curvature", yield_curvature)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 20:
        raise ValueError("steps must be an integer >= 20")
    ratio = _positive("maximum_ratio", maximum_ratio)
    power = _positive("exponent", exponent)
    return tuple(ky * ratio * (i / steps) ** power for i in range(steps + 1))


def _axial_and_moment(fibers: tuple[SectionFiber, ...], curvature: float,
                      neutral_axis: float) -> tuple[float, float, float, float]:
    force = 0.0
    moment = 0.0
    yielded_area = 0.0
    energy = 0.0
    for fiber in fibers:
        eps = curvature * (fiber.y - neutral_axis)
        stress = fiber.law.stress(eps)
        force += fiber.area * stress
        moment += fiber.area * stress * (fiber.y - neutral_axis)
        if fiber.law.is_yielded(eps):
            yielded_area += fiber.area
        energy += fiber.area * fiber.law.energy(eps)
    return force, moment, yielded_area, energy


def solve_moment_curvature(fibers, curvatures, *, tolerance: float = 1e-10,
                           max_iterations: int = 200) -> MomentCurvatureResult:
    """Incremental Smith solve with neutral-axis equilibrium per increment.

    For every prescribed curvature the neutral axis is found by bisection on the
    axial resultant (the resultant is strictly decreasing in ``y_na`` because
    every component law is non-increasing in strain), then the section moment
    ``M = sum A sigma (y - y_na)`` is integrated and the energy ledger
    ``W = integral M dkappa`` compared with ``U = sum A U_component``.
    """
    fibers = tuple(fiber.validate() for fiber in fibers)
    if len(fibers) < 2:
        raise ValueError("at least two fibres are required")
    curvatures = tuple(_finite("curvature", value) for value in curvatures)
    if len(curvatures) < 3:
        raise ValueError("at least three curvatures are required")
    sign = 1.0 if curvatures[-1] >= 0.0 else -1.0
    if abs(curvatures[0]) > 1e-15:
        raise ValueError("the curvature sweep must start at zero")
    magnitudes = [sign * value for value in curvatures]
    if any(b <= a for a, b in zip(magnitudes, magnitudes[1:])):
        raise ValueError("curvature magnitudes must be strictly increasing")

    reference = sum(fiber.area * fiber.law.yield_stress for fiber in fibers)
    ys = [fiber.y for fiber in fibers]
    span = max(ys) - min(ys)
    lo, hi = min(ys) - span, max(ys) + span

    points: list[CollapsePoint] = []
    work = 0.0
    previous_moment = 0.0
    previous_curvature = 0.0
    centroid = sum(fiber.area * fiber.y for fiber in fibers) / sum(
        fiber.area for fiber in fibers)
    for kappa in curvatures:
        if kappa == 0.0:
            # zero curvature: every fibre is at zero strain, so any axis is in
            # equilibrium; report the elastic centroid and zero moment.
            points.append(CollapsePoint(
                curvature=0.0, neutral_axis=centroid, moment=0.0,
                axial_resultant=0.0, yielded_fraction=0.0,
                internal_energy=0.0, external_work=0.0))
            continue
        # g(y) = sign * N(y) is strictly decreasing in y for either curvature
        # sign (dN/dy_na has the sign of -kappa), so a bracketed Illinois
        # iteration converges monotonically and quickly.
        left, right = lo, hi
        g_left = sign * _axial_and_moment(fibers, kappa, left)[0]
        g_right = sign * _axial_and_moment(fibers, kappa, right)[0]
        if g_left <= 0.0 or g_right >= 0.0:
            raise ValueError("neutral-axis bracket does not contain the root")
        mid = 0.5 * (left + right)
        last_side = 0
        for _ in range(max_iterations):
            denominator = g_right - g_left
            mid = (left * g_right - right * g_left) / denominator if denominator else 0.5 * (left + right)
            if not (left < mid < right):
                mid = 0.5 * (left + right)
            g_mid = sign * _axial_and_moment(fibers, kappa, mid)[0]
            if abs(g_mid) <= tolerance * reference:
                break
            if g_mid > 0.0:
                left, g_left = mid, g_mid
                if last_side == 1:
                    g_right *= 0.5
                last_side = 1
            else:
                right, g_right = mid, g_mid
                if last_side == -1:
                    g_left *= 0.5
                last_side = -1
        neutral_axis = mid
        force, moment, yielded_area, energy = _axial_and_moment(fibers, kappa, neutral_axis)
        work += 0.5 * (previous_moment + moment) * (kappa - previous_curvature)
        total_area = sum(fiber.area for fiber in fibers)
        points.append(CollapsePoint(
            curvature=kappa,
            neutral_axis=neutral_axis,
            moment=moment,
            axial_resultant=force,
            yielded_fraction=yielded_area / total_area,
            internal_energy=energy,
            external_work=work,
        ))
        previous_moment, previous_curvature = moment, kappa

    moments = [sign * point.moment for point in points]
    peak_index = max(range(len(moments)), key=lambda i: moments[i])
    ultimate = points[peak_index]
    slope = _initial_slope(points)
    first_yield = next((point for point in points if point.yielded_fraction > 0.0), points[-1])
    reference_energy = max(abs(point.internal_energy) for point in points)
    residual = 0.0
    for point in points:
        residual = max(residual, abs(point.internal_energy - point.external_work)
                       / max(reference_energy, 1e-30))
    return MomentCurvatureResult(
        points=tuple(points),
        ultimate_moment=moments[peak_index],
        ultimate_curvature=ultimate.curvature,
        first_yield_moment=sign * first_yield.moment,
        first_yield_curvature=first_yield.curvature,
        elastic_slope=slope,
        ledger_residual=residual,
        peak_is_interior=0 < peak_index < len(points) - 1,
    )


def _initial_slope(points, *, tolerance: float = 5e-3) -> float:
    """Initial (elastic) slope of a moment-curvature path.

    The secant stiffness ``M/kappa`` is evaluated from the first non-zero
    curvature point, then the path is scanned up to the first point whose
    secant stiffness deviates by more than ``tolerance`` -- i.e. up to the
    onset of the first non-linearity (plate buckling, column buckling or
    yielding).  A least-squares line through the origin is fitted over the
    points before that onset, which is the stiffness the section shows in the
    limit ``kappa -> 0``.
    """
    usable = [point for point in points if point.curvature != 0.0]
    if not usable:
        return 0.0
    reference = usable[0].moment / usable[0].curvature
    elastic = []
    for point in usable:
        secant = point.moment / point.curvature
        if abs(secant - reference) > tolerance * abs(reference):
            break
        elastic.append(point)
    if len(elastic) >= 2:
        numerator = sum(point.curvature * point.moment for point in elastic)
        denominator = sum(point.curvature ** 2 for point in elastic)
        return numerator / denominator
    return reference


def plastic_neutral_axis(fibers) -> float:
    """Axis that splits the total area in half (plastic neutral axis)."""
    fibers = tuple(fibers)
    total = sum(fiber.area for fiber in fibers)
    if total <= 0.0:
        raise ValueError("fibres must carry positive area")
    ordered = sorted(fibers, key=lambda fiber: fiber.y)
    accumulated = 0.0
    for fiber in ordered:
        if accumulated + fiber.area >= 0.5 * total:
            return fiber.y
        accumulated += fiber.area
    return ordered[-1].y

# --------------------------------------------------------------------------
# 3. double-bottom (box girder) section template
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DoubleBottomGeometry:
    """Parametric double-bottom box-girder section used for the ISSC case.

    Only ``length``, ``breadth``, ``depth``, the thickness range and
    ``yield_stress`` are fixed by the campaign node
    (``L x B x H = 6.0 x 3.0 x 2.0 m``, ``t = 8..12 mm``,
    ``sigma_y = 315 MPa``).  Everything else is an **engineering assumption**
    and is exposed here so that it can be swept; :meth:`describe` flags each
    field as fixed or assumed.

    Section layout (all members longitudinally continuous, ``y`` upward from
    the outer-bottom plating mid-surface):

    * outer bottom plating, ``breadth`` wide, thickness ``bottom_thickness``;
    * inner bottom plating at ``depth - inner_bottom_thickness / 2``;
    * two side shells, ``depth`` high, thickness ``side_thickness``;
    * ``girder_count`` longitudinal girders: web of ``girder_thickness``
      spanning the clear double-bottom height plus a
      ``girder_flange_width x girder_flange_thickness`` flange;
    * transverse floors are **not** modelled as load-carrying members (they do
      not contribute to longitudinal bending); they only define the plate-field
      length ``floor_spacing`` of every longitudinal panel.

    Panel fields for the component curves: plate fields of width
    ``panel_width = breadth / (girder_count + 1)`` (bottom and inner bottom) and
    of width ``depth`` (side shells, girder webs) with field length
    ``floor_spacing``; the girder web additionally carries a Perry-Robertson
    column mechanism over the full model length.
    """

    length: float = 6.0
    breadth: float = 3.0
    depth: float = 2.0
    bottom_thickness: float = 0.012
    inner_bottom_thickness: float = 0.010
    side_thickness: float = 0.010
    girder_thickness: float = 0.008
    girder_flange_width: float = 0.150
    girder_flange_thickness: float = 0.010
    girder_count: int = 3
    floor_spacing: float = 1.5
    young: float = 2.06e11
    poisson: float = 0.3
    yield_stress: float = 315e6
    imperfection_ratio: float = 0.1
    imperfection_reference: str = "fixed"
    imperfection_sensitivity: float = IMPERFECTION_SENSITIVITY
    residual_ratio: float = 0.2
    residual_strength: float = 0.4
    subdivisions: int = 20
    include_girders: bool = True

    FIXED_FIELDS = ("length", "breadth", "depth", "yield_stress")
    ASSUMED_FIELDS = (
        "bottom_thickness", "inner_bottom_thickness", "side_thickness",
        "girder_thickness", "girder_flange_width", "girder_flange_thickness",
        "girder_count", "floor_spacing", "young", "poisson",
        "imperfection_ratio", "imperfection_reference",
        "imperfection_sensitivity", "residual_ratio", "residual_strength",
        "subdivisions", "include_girders")

    def validate(self) -> "DoubleBottomGeometry":
        for name in ("length", "breadth", "depth", "bottom_thickness",
                     "inner_bottom_thickness", "side_thickness",
                     "girder_thickness", "girder_flange_width",
                     "girder_flange_thickness", "floor_spacing", "young",
                     "yield_stress"):
            _positive(name, getattr(self, name))
        if not isinstance(self.girder_count, int) or isinstance(self.girder_count, bool):
            raise TypeError("girder_count must be an integer")
        if self.girder_count < 0:
            raise ValueError("girder_count must be non-negative")
        if not isinstance(self.subdivisions, int) or isinstance(self.subdivisions, bool):
            raise TypeError("subdivisions must be an integer")
        if self.subdivisions < 2:
            raise ValueError("subdivisions must be >= 2")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1, 0.5)")
        if not (0.0 < self.imperfection_ratio < 5.0):
            raise ValueError("imperfection_ratio (w0/t) must lie in (0, 5)")
        if self.imperfection_reference not in ("fixed", "beta"):
            raise ValueError("imperfection_reference must be 'fixed' or 'beta'")
        if self.bottom_thickness + self.inner_bottom_thickness >= self.depth:
            raise ValueError("double-bottom thicknesses must be smaller than the depth")
        return self

    # -- derived dimensions ----------------------------------------------
    @property
    def panel_width(self) -> float:
        return self.breadth / (self.girder_count + 1)

    @property
    def clear_web_height(self) -> float:
        return self.depth - self.bottom_thickness - self.inner_bottom_thickness

    # -- members (elastic / plastic properties) ---------------------------
    def members(self) -> tuple[dict[str, float], ...]:
        """Longitudinally continuous members: area, centroid and own inertia."""
        self.validate()
        members = [
            {"name": "bottom_plating", "area": self.breadth * self.bottom_thickness,
             "y": 0.5 * self.bottom_thickness,
             "inertia": self.breadth * self.bottom_thickness ** 3 / 12.0},
            {"name": "inner_bottom", "area": self.breadth * self.inner_bottom_thickness,
             "y": self.depth - 0.5 * self.inner_bottom_thickness,
             "inertia": self.breadth * self.inner_bottom_thickness ** 3 / 12.0},
        ]
        side_area = self.depth * self.side_thickness
        for index in range(2):
            members.append({
                "name": f"side_shell_{index}",
                "area": side_area,
                "y": 0.5 * self.depth,
                "inertia": self.side_thickness * self.depth ** 3 / 12.0,
            })
        if self.include_girders and self.girder_count:
            clear = self.clear_web_height
            web_area = clear * self.girder_thickness
            flange_area = self.girder_flange_width * self.girder_flange_thickness
            for index in range(self.girder_count):
                members.append({
                    "name": f"girder_{index}_web",
                    "area": web_area,
                    "y": self.bottom_thickness + 0.5 * clear,
                    "inertia": self.girder_thickness * clear ** 3 / 12.0,
                })
                members.append({
                    "name": f"girder_{index}_flange",
                    "area": flange_area,
                    "y": (self.depth - self.inner_bottom_thickness
                          - 0.5 * self.girder_flange_thickness),
                    "inertia": (self.girder_flange_width
                                * self.girder_flange_thickness ** 3 / 12.0),
                })
        return tuple(members)

    def elastic_properties(self) -> dict[str, float]:
        """Exact elastic section properties of :meth:`members`."""
        members = self.members()
        area = sum(member["area"] for member in members)
        centroid = sum(member["area"] * member["y"] for member in members) / area
        inertia = sum(member["inertia"] + member["area"] * (member["y"] - centroid) ** 2
                      for member in members)
        return {
            "area": area,
            "neutral_axis": centroid,
            "inertia": inertia,
            "stiffness": self.young * inertia,
            "z_bottom": inertia / centroid,
            "z_top": inertia / (self.depth - centroid),
            "first_yield_moment": self.yield_stress * inertia / centroid,
        }

    def plastic_properties(self) -> dict[str, float]:
        """Plastic moment from the *distributed* fibre areas.

        Lumping a member at its centroid would erase the lever arm of members
        that straddle the plastic neutral axis (side shells, girder webs), so
        the plastic moment is integrated over the same fibre discretisation
        that the collapse solver uses, with the full yield stress as the
        capacity (no buckling knockdown -- this is the plastic bound).
        """
        layout = self.fibre_layout()
        total = sum(area for _, area, _, _ in layout)
        ordered = sorted(layout, key=lambda row: row[2])
        accumulated = 0.0
        plastic_axis = ordered[-1][2]
        for _, area, y, _ in ordered:
            if accumulated + area >= 0.5 * total:
                plastic_axis = y
                break
            accumulated += area
        moment = self.yield_stress * sum(
            area * abs(y - plastic_axis) for _, area, y, _ in layout)
        return {"plastic_neutral_axis": plastic_axis, "plastic_moment": moment,
                "area": total}

    # -- component curves -------------------------------------------------
    def _strip(self, *, length: float, width: float, thickness: float) -> PlateStrip:
        beta = (width / thickness) * math.sqrt(self.yield_stress / self.young)
        if self.imperfection_reference == "fixed":
            # PlateStrip scales the ultimate with 1 - c (w0/t - ref)/ref where
            # ref = 0.1 beta^2; requiring a *fixed* reference 0.1 means
            # w0_eff = w0 * beta^2, which makes the multiplier
            # 1 - c (w0 - 0.1)/0.1 exactly.
            ratio = self.imperfection_ratio * beta ** 2
        else:
            ratio = self.imperfection_ratio
        return PlateStrip(
            length=length, width=width, thickness=thickness,
            young=self.young, poisson=self.poisson,
            yield_stress=self.yield_stress,
            imperfection_ratio=ratio,
            residual_ratio=self.residual_ratio,
            residual_strength=self.residual_strength,
            imperfection_sensitivity=self.imperfection_sensitivity,
        )

    def component_laws(self) -> dict[str, StressStrainLaw]:
        """Named component laws used by the fibre assembly."""
        bottom = plate_field_law(self._strip(
            length=self.floor_spacing, width=self.panel_width,
            thickness=self.bottom_thickness))
        inner = plate_field_law(self._strip(
            length=self.floor_spacing, width=self.panel_width,
            thickness=self.inner_bottom_thickness))
        side = plate_field_law(self._strip(
            length=self.floor_spacing, width=self.depth,
            thickness=self.side_thickness))
        laws = {"bottom_plating": bottom, "inner_bottom": inner, "side_shell": side}
        if self.include_girders and self.girder_count:
            clear = self.clear_web_height
            web_plate = plate_field_law(self._strip(
                length=self.floor_spacing, width=clear,
                thickness=self.girder_thickness))
            column = column_law(TeeStiffener(
                web_height=clear, web_thickness=self.girder_thickness,
                flange_width=self.girder_flange_width,
                flange_thickness=self.girder_flange_thickness,
                panel_length=self.length, plate_width=self.panel_width,
                plate_thickness=self.bottom_thickness,
                young=self.young, poisson=self.poisson,
                yield_stress=self.yield_stress,
                residual_ratio=self.residual_ratio,
                residual_strength=self.residual_strength))
            laws["girder_web"] = interaction_law(
                (web_plate, column), label="girder web: plate buckling vs column")
            laws["girder_flange"] = elastic_perfectly_plastic_law(
                young=self.young, yield_stress=self.yield_stress)
        return laws

    def fibre_layout(self) -> tuple[tuple[str, float, float, str], ...]:
        """Geometric fibre layout ``(name, area, y, law_key)`` without any law.

        Keeping the layout separate from the component curves lets the plastic
        bound and the area audit run without building any stress-strain table.
        """
        self.validate()
        n = self.subdivisions
        through = max(1, n // 10)
        panels = self.girder_count + 1
        layout: list[tuple[str, float, float, str]] = []

        panel_area = self.panel_width * self.bottom_thickness
        for panel in range(panels):
            for layer in range(through):
                y = self.bottom_thickness * (layer + 0.5) / through
                layout.append((f"bottom_p{panel}_l{layer}", panel_area / through,
                               y, "bottom_plating"))
        inner_area = self.panel_width * self.inner_bottom_thickness
        for panel in range(panels):
            for layer in range(through):
                y = (self.depth - self.inner_bottom_thickness
                     + self.inner_bottom_thickness * (layer + 0.5) / through)
                layout.append((f"inner_p{panel}_l{layer}", inner_area / through,
                               y, "inner_bottom"))
        side_area = self.depth * self.side_thickness / n
        for side in range(2):
            for index in range(n):
                y = self.depth * (index + 0.5) / n
                layout.append((f"side_{side}_f{index}", side_area, y, "side_shell"))
        if self.include_girders and self.girder_count:
            clear = self.clear_web_height
            web_area = clear * self.girder_thickness / n
            flange_area = self.girder_flange_width * self.girder_flange_thickness
            flange_y = (self.depth - self.inner_bottom_thickness
                        - 0.5 * self.girder_flange_thickness)
            for girder in range(self.girder_count):
                for index in range(n):
                    y = self.bottom_thickness + clear * (index + 0.5) / n
                    layout.append((f"girder{girder}_web_f{index}", web_area, y,
                                   "girder_web"))
                layout.append((f"girder{girder}_flange", flange_area, flange_y,
                               "girder_flange"))
        return tuple(layout)

    def fibers(self) -> tuple[SectionFiber, ...]:
        """Discretise the section into fibres (``subdivisions`` per deep member)."""
        laws = self.component_laws()
        return tuple(SectionFiber(name, area, y, laws[key])
                     for name, area, y, key in self.fibre_layout())

    def describe(self) -> dict[str, object]:
        """Serialisable description with the fixed/assumed provenance split."""
        self.validate()
        values = {name: getattr(self, name) for name in self.FIXED_FIELDS}
        assumed = {name: getattr(self, name) for name in self.ASSUMED_FIELDS}
        return {
            "fixed_by_campaign_node": values,
            "assumed_by_implementation": assumed,
            "derived": {
                "panel_width": self.panel_width,
                "clear_web_height": self.clear_web_height,
                "through_thickness_fibres": max(1, self.subdivisions // 10),
            },
        }


ISSC2009_DOUBLE_BOTTOM = DoubleBottomGeometry()
"""Template instance: the campaign-node dimensions with the assumed layout."""


# --------------------------------------------------------------------------
# 4. code-independent verification of the incremental engine
# --------------------------------------------------------------------------
def verify_ideal_plastic_section(*, width: float = 0.8, height: float = 2.4,
                                 young: float = 2.06e11,
                                 yield_stress: float = 315e6,
                                 fibers: int = 200,
                                 tolerance: float = 0.03) -> dict[str, object]:
    """Compare the incremental engine with the ideal plastic rectangular oracle.

    The oracle is the closed-form ideal elastic-plastic solution
    ``M = E I kappa`` up to ``kappa_y = 2 sigma_y/(E h)`` and
    ``M = b sigma_y h^2/4 - b sigma_y c^2/3`` with ``c = sigma_y/(E kappa)``
    beyond it.  This isolates the section solver from every buckling and
    imperfection assumption.
    """
    width = _positive("width", width)
    height = _positive("height", height)
    e = _positive("young", young)
    sy = _positive("yield_stress", yield_stress)
    if not isinstance(fibers, int) or isinstance(fibers, bool) or fibers < 20:
        raise ValueError("fibers must be an integer >= 20")
    law = elastic_perfectly_plastic_law(young=e, yield_stress=sy)
    area = width * height / fibers
    dy = height / fibers
    section = tuple(SectionFiber(f"fibre_{i}", area, (i + 0.5) * dy, law)
                    for i in range(fibers))
    ky = 2.0 * sy / (e * height)
    curvatures = tuple(ky * factor for factor in
                       (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0))
    result = solve_moment_curvature(section, curvatures)

    def oracle(kappa: float) -> float:
        inertia = width * height ** 3 / 12.0
        if kappa <= ky:
            return e * inertia * kappa
        core = sy / (e * kappa)
        return width * sy * height ** 2 / 4.0 - width * sy * core ** 2 / 3.0

    errors = []
    for point in result.points:
        reference = oracle(point.curvature)
        scale = max(abs(reference), 1e-30)
        errors.append(abs(point.moment - reference) / scale)
    plastic = sy * width * height ** 2 / 4.0
    elastic_inertia = width * height ** 3 / 12.0
    slope_error = abs(result.elastic_slope - e * elastic_inertia) / (e * elastic_inertia)
    moment_error = max(errors)
    return {
        "schema": "tensorfem.smith-section-oracle/1.0",
        "oracle": "closed-form ideal elastic-plastic rectangular section",
        "max_relative_moment_error": moment_error,
        "elastic_slope_relative_error": slope_error,
        "plastic_moment_relative_error": abs(result.ultimate_moment - plastic) / plastic,
        "full_plastic_moment": plastic,
        "computed_peak_moment": result.ultimate_moment,
        "tolerance": tolerance,
        "passed": moment_error < tolerance and slope_error < tolerance,
        "curve": [{"curvature": point.curvature, "moment": point.moment,
                   "oracle": oracle(point.curvature)} for point in result.points],
    }


def limiting_stress_envelope(geometry: DoubleBottomGeometry, *, steps: int = 160,
                             maximum_curvature_ratio: float = 8.0) -> dict[str, object]:
    """Independent cross-method check with the element-level limiting-stress solve.

    The parent module's :func:`~.plate_stiffener_ultimate.smith_mk_curve` is a
    different solver: each element is elastic up to a *limiting compression
    stress* and constant afterwards, with no post-ultimate softening, so its
    moment-curvature path is monotone and its peak is an **envelope** of the
    softening solution.  Feeding it the same fibres with the same component
    peak stresses gives a model-independent consistency check on the magnitude
    of the ultimate moment (the softening solution must stay below the
    envelope).
    """
    import numpy as np

    from .plate_stiffener_ultimate import SectionElement, smith_mk_curve

    geometry.validate()
    props = geometry.elastic_properties()
    eps_yield = geometry.yield_stress / geometry.young
    extreme = max(props["neutral_axis"], geometry.depth - props["neutral_axis"])
    kappa_yield = eps_yield / extreme
    curvatures = np.array(curvature_grid(kappa_yield, steps=steps,
                                         maximum_ratio=maximum_curvature_ratio))
    elements = [
        SectionElement(fibre.name, fibre.area, fibre.y, fibre.law.young,
                       fibre.law.yield_stress,
                       compression_strength=fibre.law.peak_compressive_stress)
        for fibre in geometry.fibers()
    ]
    result = smith_mk_curve(elements, curvatures)
    return {
        "method": "element-level limiting-stress Smith (no softening) -- envelope",
        "envelope_ultimate_moment": float(result["M_u"]),
        "envelope_curvature": float(result["kappa_at_Mu"]),
        "curvature_sign": "sagging (kappa > 0)",
    }


# --------------------------------------------------------------------------
# 5. experimental anchor injection (honest: never invented)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class UltimateMomentAnchor:
    """Externally supplied experimental ultimate moment.

    The anchor is *injected*, never embedded: this repository does not contain
    a transcribed ISSC 2009 double-bottom test value, so the benchmark runner
    reports ``NOT_VERIFIED`` until a caller supplies a documented anchor.
    """

    reference_ultimate_moment: float
    source: str
    tolerance: float = 0.15
    condition: str = "sagging"

    def validate(self) -> "UltimateMomentAnchor":
        _positive("reference_ultimate_moment", self.reference_ultimate_moment)
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must document where the anchor comes from")
        _positive("tolerance", self.tolerance)
        if self.condition not in ("sagging", "hogging"):
            raise ValueError("condition must be 'sagging' or 'hogging'")
        return self


def load_anchor(path: str | Path | None) -> UltimateMomentAnchor | None:
    """Load an anchor JSON; ``None`` when no path is given or the file is absent."""
    if path is None:
        return None
    file = Path(path)
    if not file.exists():
        return None
    payload = json.loads(file.read_text(encoding="utf-8"))
    anchor = UltimateMomentAnchor(
        reference_ultimate_moment=payload["reference_ultimate_moment"],
        source=payload["source"],
        tolerance=payload.get("tolerance", 0.15),
        condition=payload.get("condition", "sagging"),
    )
    return anchor.validate()


# --------------------------------------------------------------------------
# 6. ISSC 2009 double-bottom benchmark runner
# --------------------------------------------------------------------------
def _analyse_case(geometry: DoubleBottomGeometry, *, steps: int,
                  maximum_curvature_ratio: float,
                  condition: str) -> dict[str, object]:
    geometry.validate()
    if condition not in ("sagging", "hogging"):
        raise ValueError("condition must be 'sagging' or 'hogging'")
    props = geometry.elastic_properties()
    plastic = geometry.plastic_properties()
    eps_yield = geometry.yield_stress / geometry.young
    extreme = max(props["neutral_axis"], geometry.depth - props["neutral_axis"])
    kappa_yield = eps_yield / extreme
    sign = 1.0 if condition == "sagging" else -1.0
    grid = tuple(sign * value for value in curvature_grid(
        kappa_yield, steps=steps, maximum_ratio=maximum_curvature_ratio))
    fibers = geometry.fibers()
    result = solve_moment_curvature(fibers, grid)
    fiber_area = sum(fiber.area for fiber in fibers)
    return {
        "properties": props,
        "plastic": plastic,
        "kappas": {"yield_curvature": kappa_yield},
        "result": result,
        "fiber_area_relative_error": abs(fiber_area - props["area"]) / props["area"],
    }


def run_issc2009_double_bottom(*, geometry: DoubleBottomGeometry | None = None,
                               anchor: UltimateMomentAnchor | None = None,
                               subdivisions: tuple[int, ...] = (10, 20, 40),
                               imperfection_ratios: tuple[float, ...] = (0.05, 0.1, 0.2, 0.5),
                               curvature_steps: int = 320,
                               maximum_curvature_ratio: float = 8.0,
                               pytest_tests: int | None = None) -> dict[str, object]:
    """Run criteria C1..C6 for the ISSC 2009 double-bottom Smith case.

    Parameters
    ----------
    geometry:
        Section template; defaults to :data:`ISSC2009_DOUBLE_BOTTOM`.
    anchor:
        Experimental ultimate moment.  ``None`` (the default in this
        environment) leaves criterion C1 ``NOT_VERIFIED`` -- it is never
        replaced by a fabricated number.
    subdivisions, imperfection_ratios:
        Predeclared mesh and imperfection schedules (criteria C5 and C3).
    pytest_tests:
        Number of tests collected by the accompanying pytest suite; recorded
        for criterion C6 and left as ``NOT_RECORDED`` when not supplied.
    """
    from dataclasses import replace

    base = (ISSC2009_DOUBLE_BOTTOM if geometry is None else geometry).validate()
    if anchor is not None:
        anchor.validate()

    sagging = _analyse_case(base, steps=curvature_steps,
                            maximum_curvature_ratio=maximum_curvature_ratio,
                            condition="sagging")
    hogging = _analyse_case(base, steps=curvature_steps,
                            maximum_curvature_ratio=maximum_curvature_ratio,
                            condition="hogging")
    primary = sagging
    result = primary["result"]
    props = primary["properties"]
    plastic = primary["plastic"]

    ultimate = result.ultimate_moment
    elastic_slope_error = abs(result.elastic_slope - props["stiffness"]) / props["stiffness"]

    # C3: imperfection sensitivity against the smallest scheduled imperfection.
    baseline_ratio = min(imperfection_ratios)
    imperfection_rows = []
    for ratio in imperfection_ratios:
        case = replace(base, imperfection_ratio=ratio)
        analysis = _analyse_case(case, steps=curvature_steps,
                                 maximum_curvature_ratio=maximum_curvature_ratio,
                                 condition="sagging")
        imperfection_rows.append({
            "w0_over_t": ratio,
            "ultimate_moment": analysis["result"].ultimate_moment,
            "ultimate_curvature": analysis["result"].ultimate_curvature,
            "plate_ultimate_ratio_bottom": case.component_laws()["bottom_plating"].peak_compressive_stress / base.yield_stress,
        })
    reference_row = min(imperfection_rows, key=lambda row: abs(row["w0_over_t"] - baseline_ratio))
    for row in imperfection_rows:
        row["drop_vs_baseline"] = ((reference_row["ultimate_moment"] - row["ultimate_moment"])
                                   / reference_row["ultimate_moment"])

    def _drop(ratio: float) -> float:
        row = min(imperfection_rows, key=lambda item: abs(item["w0_over_t"] - ratio))
        return float(row["drop_vs_baseline"])

    # C5: fibre-count convergence.
    mesh_rows = []
    for count in subdivisions:
        case = replace(base, subdivisions=count)
        analysis = _analyse_case(case, steps=curvature_steps,
                                 maximum_curvature_ratio=maximum_curvature_ratio,
                                 condition="sagging")
        mesh_rows.append({"subdivisions": count,
                          "fibres": len(case.fibers()),
                          "ultimate_moment": analysis["result"].ultimate_moment})
    moments = [row["ultimate_moment"] for row in mesh_rows]
    mean_moment = sum(moments) / len(moments)
    mesh_spread = (max(moments) - min(moments)) / mean_moment

    # C1: experimental anchor.
    if anchor is None:
        anchor_block = {
            "status": "NOT_VERIFIED",
            "computed_ultimate_moment": ultimate,
            "reason": ("no documented ISSC 2009 double-bottom experimental "
                       "ultimate moment is available in this environment; the "
                       "number is not invented (honest-FAIL discipline)"),
            "how_to_close": ("supply a JSON {reference_ultimate_moment, "
                             "source, tolerance, condition} via load_anchor()"),
        }
    else:
        error = abs(ultimate - anchor.reference_ultimate_moment) / anchor.reference_ultimate_moment
        anchor_block = {
            "status": "PASS" if error <= anchor.tolerance else "FAIL",
            "computed_ultimate_moment": ultimate,
            "reference_ultimate_moment": anchor.reference_ultimate_moment,
            "source": anchor.source,
            "relative_error": error,
            "tolerance": anchor.tolerance,
        }

    engine_check = verify_ideal_plastic_section(young=base.young,
                                                yield_stress=base.yield_stress)
    envelope = limiting_stress_envelope(base, steps=max(40, curvature_steps // 2),
                                        maximum_curvature_ratio=maximum_curvature_ratio)
    envelope_ratio = ultimate / envelope["envelope_ultimate_moment"]
    envelope["softening_over_envelope"] = envelope_ratio
    envelope["consistent"] = 0.5 <= envelope_ratio <= 1.0 + 1e-9
    envelope["note"] = (
        "the softening solution must sit below the limiting-stress envelope; "
        "a ratio above 1 would mean the two solvers contradict each other")

    criteria = {
        "C1": {"status": anchor_block["status"], "tolerance": 0.15,
               "detail": anchor_block},
        "C2": {"status": "PASS" if elastic_slope_error <= 0.05 else "FAIL",
               "tolerance": 0.05, "relative_error": elastic_slope_error,
               "computed_slope": result.elastic_slope,
               "reference_stiffness": props["stiffness"]},
        "C3": {"status": "PASS" if (_drop(0.1) <= 0.10 and _drop(0.5) <= 0.30) else "FAIL",
               "tolerance": {"w0_over_t_0.1": 0.10, "w0_over_t_0.5": 0.30},
               "drop_at_0.1": _drop(0.1), "drop_at_0.5": _drop(0.5),
               "baseline_w0_over_t": baseline_ratio},
        "C4": {"status": "PASS" if result.ledger_residual <= 0.05 else "FAIL",
               "tolerance": 0.05, "residual": result.ledger_residual},
        "C5": {"status": "PASS" if mesh_spread <= 0.05 else "FAIL",
               "tolerance": 0.05, "spread": mesh_spread,
               "schedule": [row["subdivisions"] for row in mesh_rows]},
        "C6": {"status": ("PASS" if pytest_tests is not None and pytest_tests >= 15
                          else ("NOT_RECORDED" if pytest_tests is None else "FAIL")),
               "required_tests": 15, "collected_tests": pytest_tests},
    }
    verified = [name for name, value in criteria.items() if value["status"] == "PASS"]
    failed = [name for name, value in criteria.items() if value["status"] == "FAIL"]
    pending = [name for name, value in criteria.items()
               if value["status"] not in ("PASS", "FAIL")]
    overall = "FAIL" if failed else ("PASS" if not pending else "CONDITIONAL")

    return {
        "schema": "tensorfem.issc2009-double-bottom-smith/1.0",
        "case": "ISSC 2009 double-bottom box girder, Smith progressive collapse",
        "units": "SI (m, Pa, N m, 1/m)",
        "scope": ("plane-section progressive collapse; buckling/softening "
                  "component curves; no shear lag, torsion, fracture or "
                  "residual-stress gradient"),
        "geometry": base.describe(),
        "provenance": PROVENANCE,
        "primary_condition": "sagging",
        "section_properties": {
            "area": props["area"],
            "neutral_axis": props["neutral_axis"],
            "inertia": props["inertia"],
            "first_yield_moment": props["first_yield_moment"],
            "plastic_moment": plastic["plastic_moment"],
            "plastic_neutral_axis": plastic["plastic_neutral_axis"],
            "yield_curvature": primary["kappas"]["yield_curvature"],
            "fibre_area_relative_error": primary["fiber_area_relative_error"],
        },
        "ultimate": {
            "sagging_moment": ultimate,
            "sagging_curvature": result.ultimate_curvature,
            "hogging_moment": hogging["result"].ultimate_moment,
            "hogging_curvature": hogging["result"].ultimate_curvature,
            "first_yield_moment": result.first_yield_moment,
            "M_u_over_M_p": ultimate / plastic["plastic_moment"],
            "M_u_over_M_y": ultimate / props["first_yield_moment"],
            "peak_is_interior": result.peak_is_interior,
        },
        "energy_ledger": {
            "residual": result.ledger_residual,
            "internal_energy_at_peak": max(point.internal_energy for point in result.points),
            "external_work_at_peak": max(point.external_work for point in result.points),
            "note": ("W = integral M dkappa and U = sum A integral sigma deps use "
                     "the same tabulated component laws; the residual is the "
                     "curvature-discretisation error of the trapezoidal rule"),
        },
        "curves": {
            "sagging": result.curve(stride=max(1, len(result.points) // 80)),
            "hogging": hogging["result"].curve(stride=max(1, len(hogging["result"].points) // 80)),
        },
        "imperfection_study": imperfection_rows,
        "mesh_study": mesh_rows,
        "section_engine_verification": engine_check,
        "cross_method_envelope": envelope,
        "criteria": criteria,
        "criteria_passed": len(verified),
        "criteria_failed": failed,
        "criteria_pending": pending,
        "overall_status": overall,
        "limitations": (
            "Smith-type plane-section collapse with component curves; the "
            "component laws are panel-average surrogates, not shell-FE solutions. "
            "The experimental anchor for criterion C1 is withheld in this "
            "environment, so the ISSC comparison is NOT_VERIFIED here. "
            "M_u is a first-collapse-of-section value for monotone bending; "
            "unloading, cyclic and residual-stress-gradient effects are excluded."
        ),
    }
