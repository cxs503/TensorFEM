"""Moment--curvature collapse surface of a stiffened panel (Smith's method).

Why this module exists
----------------------
``tensorfem.stiffened_panel_ultimate`` supplies component limiting strengths
(Faulkner plate, Johnson--Ostenfeld column, St Venant tripping, flange local
buckling) and a "components-at-ultimate" area average.  Its progressive
collapse routine integrates a *hull-girder* section with the neutral axis held
at the elastic value and a limiting-stress plateau.  Neither is a panel-level
moment--curvature (``M``--``chi``) surface, which is what a hull-girder
progressive-collapse analysis actually consumes from a stiffened panel.

This module adds that surface, with the parts the earlier routines omit:

* **Section re-equilibration at every curvature increment.**  For every
  ``chi`` the reference strain ``eps0`` is re-solved so that the axial force
  of the cross-section equals the applied one (zero by default), so the
  neutral axis shifts into the tension side as the compressed side yields and
  sheds load -- the effect a fixed-axis integration cannot represent.
* **Per-fibre mode assignment and unit unloading.**  Each fibre of the section
  (plate strips, web, flange) carries its *own* limiting mode and its own
  softening branch, so a fibre that has passed its limit unloads
  monotonically while the applied curvature keeps growing (Smith's "strain
  increments with unloading units").
* **An energy ledger.**  ``W = integral M dchi`` (trapezoid on the sweep) is
  compared with the internal work ``U = sum A_i U_i(eps_i)`` of the fibre
  laws.  With the axial force zero the two must agree, so the residual is a
  direct check of the incremental integration and of the equilibrium solve,
  not a restatement of the assumptions.
* **Monotonicity / peak-uniqueness checks and increment-refinement studies**,
  so an ``M_u`` claim carries its convergence evidence.
* **Cross-verification surface against TensorFEM discretisations**:
  :func:`column_euler_fem_check` (beam eigenvalue vs Euler),
  :func:`plate_beta_fem_check` (finite-difference plate eigenvalue vs the
  von Karman slenderness anchor) and :func:`fe_panel_compression_path`
  (elasto-plastic Shell4 Riks path of the panel unit cell) which brackets the
  axial ultimate stress that the fibre laws predict.

Scope and honesty statement
---------------------------
* The fibre law is a limiting-stress model (elastic -> limit -> optional
  linear softening -> residual plateau).  It is a *deformation-theory*
  approximation of the load-end shortening of a plate/stiffener, not a
  large-deflection elasto-plastic solution.  ``softening_strain`` and
  ``residual_ratio`` are **caller supplied**: the defaults are the plateau
  model (no post-peak reserve and no drop), which is the same assumption as
  the existing screening chain, so the defaults are *not* calibrated numbers
  presented as measurements.
* Because the fibre law is a function of total strain, the internal work is
  path independent.  The energy residual therefore measures the quadrature and
  equilibrium error only; it is not evidence that the fibre law is right.
* The ``M``--``chi`` peak is a section analysis result.  It is bracketed here
  by the first-yield moment (lower) and the lumped-fibre fully plastic moment
  (upper); those bounds are properties of the discretisation and are reported
  as such.
* The elasto-plastic Riks path in :func:`fe_panel_compression_path` is
  *axial* compression of a half-bay unit cell.  It cross-checks the axial
  ``sigma_u`` and the shape of the load-end shortening, not the bending
  ``M``--``chi`` curve itself; an elasto-plastic shell *bending* path is not
  executed here and that gap is stated in the report rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .stiffened_panel_ultimate import StiffenedPanel

__all__ = [
    "LoadShorteningCurve",
    "PanelFibre",
    "PanelSection",
    "CurvaturePoint",
    "PanelCollapse",
    "section_from_panel",
    "moment_curvature",
    "axial_ultimate",
    "increment_refinement_study",
    "column_euler_fem_check",
    "plate_beta_fem_check",
    "fe_panel_compression_path",
]


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _finite(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _fraction(name: str, value: float) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return number


# --------------------------------------------------------------------------
# 1. fibre load-end shortening law
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class LoadShorteningCurve:
    """One fibre's stress--strain law in compression and tension.

    Tension is linear elastic capped at yield.  Compression is

    * linear elastic while ``eps <= eps_lim = limit_stress / E``;
    * a linear softening branch over ``softening_strain`` absolute strain,
      falling from ``limit_stress`` to ``residual_ratio * limit_stress``;
    * a constant residual plateau afterwards (``residual_ratio`` defaults to
      1, which is the limiting-stress plateau of the screening chain).

    ``limit_stress`` may not exceed the yield stress: a fibre cannot carry more
    than first yield in a *compression* limit state without an explicit strain
    hardening model, which this module does not offer.
    """

    young: float
    yield_stress: float
    limit_stress: float
    residual_ratio: float = 1.0
    softening_strain: float = 0.0

    def __post_init__(self) -> None:
        e = _positive("young", self.young)
        sy = _positive("yield_stress", self.yield_stress)
        limit = _positive("limit_stress", self.limit_stress)
        if limit > sy * (1.0 + 1e-12):
            raise ValueError("limit_stress must not exceed the yield stress")
        _fraction("residual_ratio", self.residual_ratio)
        softening = _finite("softening_strain", self.softening_strain)
        if softening < 0.0:
            raise ValueError("softening_strain must be non-negative")
        if softening > 0.0 and self.residual_ratio >= 1.0:
            raise ValueError(
                "a softening branch with residual_ratio >= 1 does not soften; "
                "leave softening_strain at zero for a plateau law"
            )
        if softening == 0.0 and self.residual_ratio != 1.0:
            raise ValueError("residual_ratio < 1 needs a positive softening_strain to reach it")
        # Touch the attributes so a NaN slips through no earlier than here.
        _finite("young*factor", e * sy)

    @property
    def limit_strain(self) -> float:
        return self.limit_stress / self.young

    @property
    def residual_stress(self) -> float:
        return self.limit_stress * self.residual_ratio

    def stress(self, strain: float) -> float:
        """Signed stress (tension positive) at a signed strain."""
        eps = float(strain)
        if eps >= 0.0:
            return min(self.young * eps, self.yield_stress)
        magnitude = -eps
        if magnitude <= self.limit_strain:
            return -self.young * magnitude
        if self.softening_strain <= 0.0:
            return -self.limit_stress
        fraction = min(1.0, (magnitude - self.limit_strain) / self.softening_strain)
        return -(self.limit_stress - (self.limit_stress - self.residual_stress) * fraction)

    def energy(self, strain: float) -> float:
        """Internal energy density ``int_0^eps sigma deps`` (path independent)."""
        eps = float(strain)
        if eps >= 0.0:
            yield_strain = self.yield_stress / self.young
            if eps <= yield_strain:
                return 0.5 * self.young * eps * eps
            return 0.5 * self.yield_stress * yield_strain + self.yield_stress * (eps - yield_strain)
        magnitude = -eps
        limit = self.limit_strain
        if magnitude <= limit:
            return 0.5 * self.young * magnitude * magnitude
        energy = 0.5 * self.limit_stress * limit
        if self.softening_strain <= 0.0:
            return energy + self.limit_stress * (magnitude - limit)
        span = min(self.softening_strain, magnitude - limit)
        drop = self.limit_stress - self.residual_stress
        energy += self.limit_stress * span - 0.5 * drop * span * span / self.softening_strain
        if magnitude > limit + self.softening_strain:
            energy += self.residual_stress * (magnitude - limit - self.softening_strain)
        return energy


# --------------------------------------------------------------------------
# 2. section definition
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PanelFibre:
    """One lumped fibre of the panel cross-section."""

    name: str
    area: float
    offset: float
    curve: LoadShorteningCurve
    mode: str = "plate"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("fibre name must be non-empty")
        _positive(f"area[{self.name}]", self.area)
        _finite(f"offset[{self.name}]", self.offset)

    def stress(self, strain: float) -> float:
        return self.curve.stress(strain)

    def energy(self, strain: float) -> float:
        return self.area * self.curve.energy(strain)


@dataclass(frozen=True)
class PanelSection:
    """A stiffened-panel cross-section as a set of strain-loaded fibres.

    ``offset`` is measured from the plate mid-surface, positive towards the
    stiffener.  Bending is about the transverse (``y``) axis, so a curvature
    ``chi > 0`` compresses the stiffener side.
    """

    fibres: tuple[PanelFibre, ...]
    young: float
    yield_stress: float
    label: str = "panel-section"

    def __post_init__(self) -> None:
        if len(self.fibres) < 2:
            raise ValueError("a section needs at least two fibres")
        _positive("young", self.young)
        _positive("yield_stress", self.yield_stress)

    # -- geometric properties ---------------------------------------------
    @property
    def total_area(self) -> float:
        return sum(f.area for f in self.fibres)

    @property
    def first_moment(self) -> float:
        return sum(f.area * f.offset for f in self.fibres)

    @property
    def elastic_neutral_axis(self) -> float:
        return self.first_moment / self.total_area

    @property
    def inertia(self) -> float:
        axis = self.elastic_neutral_axis
        return sum(f.area * (f.offset - axis) ** 2 for f in self.fibres)

    @property
    def polar_second_moment(self) -> float:
        return sum(f.area * f.offset * f.offset for f in self.fibres)

    def plastic_neutral_axis(self) -> float:
        """Area-bisecting axis of the lumped fibre set.

        With lumped fibres the axis is exact only to within the fibre spacing;
        it is reported with the fully plastic moment as a discretisation
        property, not as a continuum result.
        """
        total = self.total_area
        ordered = sorted(self.fibres, key=lambda f: f.offset)
        running = 0.0
        for index, fibre in enumerate(ordered):
            running += fibre.area
            if running >= 0.5 * total:
                if index == 0:
                    return fibre.offset
                previous = ordered[index - 1]
                if running == 0.5 * total:
                    return 0.5 * (previous.offset + fibre.offset)
                return fibre.offset
        return ordered[-1].offset

    def fully_plastic_moment(self) -> float:
        """Lumped-fibre plastic moment about the area-bisecting axis."""
        axis = self.plastic_neutral_axis()
        return sum(self.yield_stress * f.area * abs(f.offset - axis) for f in self.fibres)

    def first_yield_moment(self, curvature: float) -> float:
        """Moment at which the most strained fibre reaches its limit."""
        return self.young * self.inertia * _positive("curvature", curvature)

    def section_modulus(self) -> tuple[float, float]:
        """Elastic section moduli for the plate side and the stiffener side."""
        axis = self.elastic_neutral_axis
        upper = max(f.offset for f in self.fibres) - axis
        lower = axis - min(f.offset for f in self.fibres)
        inertia = self.inertia
        return inertia / upper, inertia / lower

    def summary(self) -> dict[str, object]:
        plate_modulus, stiffener_modulus = self.section_modulus()
        return {
            "label": self.label,
            "fibres": len(self.fibres),
            "total_area": self.total_area,
            "elastic_neutral_axis": self.elastic_neutral_axis,
            "plastic_neutral_axis": self.plastic_neutral_axis(),
            "inertia": self.inertia,
            "section_modulus_plate_side": plate_modulus,
            "section_modulus_stiffener_side": stiffener_modulus,
            "fully_plastic_moment": self.fully_plastic_moment(),
            "limits": {
                f.name: {"mode": f.mode, "limit_stress": f.curve.limit_stress} for f in self.fibres
            },
        }


def section_from_panel(
    panel: StiffenedPanel,
    *,
    plate_strips: int = 8,
    web_layers: int = 4,
    flange_layers: int = 2,
    include_tripping: bool = False,
    softening_strain: float = 0.0,
    residual_ratio: float = 1.0,
    label: str | None = None,
) -> PanelSection:
    """Discretise a :class:`StiffenedPanel` into plate and stiffener fibres.

    The plate field is split into ``plate_strips`` equal strips across the
    stiffener spacing (all at the plate mid-surface, all with the Faulkner
    plate limiting stress) so that a coarse or a fine section discretisation
    can be compared.  The web is lumped at ``-h_w/2`` with the
    Johnson--Ostenfeld column limit; the flange at ``-h_w`` with
    ``min(column, flange local buckling)`` and, when ``include_tripping`` is
    set, the St Venant tripping stress as well.  Assigning the column limit to
    the web/flange fibres is the components-at-ultimate reduction: the
    stiffener-plus-plate column buckles as one member, and the fibre closest to
    the extreme compression fibre governs the section moment.

    ``web_layers`` / ``flange_layers`` subdivide those plates through their
    thickness so that the lumped section recovers their own bending inertia
    (``h^3 t / 12`` for the web).  A single lumped fibre at the centroid loses
    it, which costs about 11 % of the section inertia for a typical tee
    stiffener -- more than the modelling error this module is meant to resolve.
    The plate field itself needs no such subdivision (its own inertia is
    ``b t^3 / 12``, a few tenths of a percent) but ``plate_strips`` is kept as
    the width discretisation knob.
    """
    for name, value in (
        ("plate_strips", plate_strips),
        ("web_layers", web_layers),
        ("flange_layers", flange_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    panel.validate()
    young = panel.young
    yield_stress = panel.yield_stress
    curve_kwargs = {"softening_strain": softening_strain, "residual_ratio": residual_ratio}

    plate_area = panel.plate_area / plate_strips
    plate_curve = LoadShorteningCurve(
        young=young,
        yield_stress=yield_stress,
        limit_stress=panel.plate_limiting_stress,
        **curve_kwargs,
    )
    fibres = [
        PanelFibre(
            name=f"plate[{index}]", area=plate_area, offset=0.0, curve=plate_curve, mode="plate"
        )
        for index in range(plate_strips)
    ]
    column_limit = panel.column_limiting_stress
    web_curve = LoadShorteningCurve(
        young=young,
        yield_stress=yield_stress,
        limit_stress=min(column_limit, yield_stress),
        **curve_kwargs,
    )
    web_area = panel.web_height * panel.web_thickness / web_layers
    for index in range(web_layers):
        # The web spans z in [-h_w, 0]; layer 0 is the layer next to the plate.
        offset = -panel.web_height * (index + 0.5) / web_layers
        fibres.append(
            PanelFibre(
                name=f"web[{index}]", area=web_area, offset=offset, curve=web_curve, mode="column"
            )
        )
    flange_candidates = {"column": column_limit, "flange_local": panel.flange_local_buckling_stress}
    if include_tripping:
        flange_candidates["tripping"] = panel.tripping_stress
    flange_mode, flange_limit = min(flange_candidates.items(), key=lambda item: item[1])
    flange_curve = LoadShorteningCurve(
        young=young,
        yield_stress=yield_stress,
        limit_stress=min(flange_limit, yield_stress),
        **curve_kwargs,
    )
    flange_area = panel.flange_width * panel.flange_thickness / flange_layers
    for index in range(flange_layers):
        # The flange mid-surface is at z = -h_w, thickness t_f through it.
        offset = -panel.web_height + (index + 0.5 - 0.5 * flange_layers) * (
            panel.flange_thickness / flange_layers
        )
        fibres.append(
            PanelFibre(
                name=f"flange[{index}]",
                area=flange_area,
                offset=offset,
                curve=flange_curve,
                mode=flange_mode,
            )
        )
    return PanelSection(
        fibres=tuple(fibres),
        young=young,
        yield_stress=yield_stress,
        label=label or "stiffened-panel",
    )


# --------------------------------------------------------------------------
# 3. incremental moment--curvature integration (Smith's method)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CurvaturePoint:
    """One converged point of the ``M``--``chi`` path."""

    curvature: float
    moment: float
    reference_strain: float
    axial_force: float
    axial_residual: float
    fibres_at_limit: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "curvature": self.curvature,
            "moment": self.moment,
            "reference_strain": self.reference_strain,
            "axial_force": self.axial_force,
            "axial_residual": self.axial_residual,
            "fibres_at_limit": list(self.fibres_at_limit),
        }


@dataclass(frozen=True)
class PanelCollapse:
    """Result of :func:`moment_curvature` with its verification ledger."""

    section: PanelSection
    points: tuple[CurvaturePoint, ...]
    ultimate_moment: float
    ultimate_curvature: float
    ultimate_index: int
    internal_work: float
    external_work: float
    peak_refined: bool
    local_maxima: tuple[int, ...]
    plateau_at_peak: bool = False

    @property
    def energy_residual(self) -> float:
        scale = max(abs(self.internal_work), abs(self.external_work), 1e-30)
        return abs(self.internal_work - self.external_work) / scale

    @property
    def moment_ratio(self) -> float:
        plastic = self.section.fully_plastic_moment()
        return self.ultimate_moment / plastic if plastic else float("inf")

    @property
    def monotonic_to_peak(self) -> bool:
        moments = [p.moment for p in self.points[: self.ultimate_index + 1]]
        return all(b >= a - 1e-12 * abs(a) - 1e-15 for a, b in zip(moments, moments[1:]))

    @property
    def post_peak(self) -> bool:
        if self.ultimate_index == len(self.points) - 1:
            return False
        peak = self.points[self.ultimate_index].moment
        return self.points[-1].moment < peak * (1.0 - 1e-9)

    @property
    def peak_unique(self) -> bool:
        return len(self.local_maxima) == 1

    @property
    def saturating(self) -> bool:
        """True when ``M(chi)`` ends within 1 % of its ultimate value.

        A saturating path means the fibre laws have no post-buckling
        degradation: the moment asymptotes to the plastic moment instead of
        reaching a limit point.  Reported so an ``M_u`` is never presented as a
        limit point when the model cannot produce one.
        """
        return abs(self.points[-1].moment - self.ultimate_moment) <= 0.01 * abs(
            self.ultimate_moment
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ultimate_moment": self.ultimate_moment,
            "ultimate_curvature": self.ultimate_curvature,
            "ultimate_index": self.ultimate_index,
            "fully_plastic_moment": self.section.fully_plastic_moment(),
            "moment_over_plastic": self.moment_ratio,
            "internal_work": self.internal_work,
            "external_work": self.external_work,
            "energy_residual": self.energy_residual,
            "monotonic_to_peak": self.monotonic_to_peak,
            "post_peak": self.post_peak,
            "peak_unique": self.peak_unique,
            "plateau_at_peak": self.plateau_at_peak,
            "saturating": self.saturating,
            "local_maxima": list(self.local_maxima),
            "points": [p.as_dict() for p in self.points],
        }


def _axial_force(section: PanelSection, strain_shift: float, curvature: float) -> float:
    total = 0.0
    for fibre in section.fibres:
        total += fibre.area * fibre.stress(strain_shift + curvature * fibre.offset)
    return total


def _has_softening(section: PanelSection) -> bool:
    return any(fibre.curve.softening_strain > 0.0 for fibre in section.fibres)


def _reference_strain_bracket(
    section: PanelSection, curvature: float, axial_force: float
) -> tuple[float, float, float]:
    axis = section.elastic_neutral_axis
    extent = max(abs(f.offset - axis) for f in section.fibres)
    strain_scale = max(f.curve.limit_strain for f in section.fibres)
    scale = max(strain_scale + curvature * extent, strain_scale, 1e-30)
    for _ in range(40):
        low, high = -4.0 * scale, 4.0 * scale
        f_low = _axial_force(section, low, curvature) - axial_force
        f_high = _axial_force(section, high, curvature) - axial_force
        if f_low <= 0.0 <= f_high:
            return low, high, 0.0
        scale *= 2.0
    raise ValueError(
        f"axial equilibrium cannot be bracketed at curvature {curvature:g}: "
        "the applied axial force exceeds the section capacity"
    )


def _solve_reference_strain(
    section: PanelSection,
    curvature: float,
    axial_force: float,
    *,
    previous: float | None = None,
    tolerance: float = 1e-12,
    max_iterations: int = 400,
) -> tuple[float, float]:
    """Solve for the strain at the plate mid-surface that equilibrates the axial force.

    With a *plateau* law every fibre stress is a non-decreasing function of the
    strain, so the section force is non-decreasing and plain bisection is
    bracketed and unique.  A *softening* branch breaks that: past the limit the
    stress falls as the compressive strain grows, so the section force can have
    several roots in the bracket and the root is selected by continuation from
    the previous curvature increment (:func:`_solve_reference_strain`).  A flat
    range (all fibres on their residual plateau) leaves the residual as the only
    evidence that the point is admissible; it is reported for every point.
    """
    low, high, _ = _reference_strain_bracket(section, curvature, axial_force)
    residual = tolerance * max(section.total_area * section.yield_stress, 1e-30)
    if _has_softening(section):
        # A softening branch makes the section force non-monotone in the
        # reference strain, so the bracket can hold several roots.  Smith's
        # method is a continuation along the curvature path: select the root
        # nearest the previous increment, which keeps the neutral-axis history
        # continuous instead of jumping branches.
        samples = 257
        step = (high - low) / (samples - 1)
        values = [
            _axial_force(section, low + step * i, curvature) - axial_force for i in range(samples)
        ]
        intervals = []
        for index in range(samples - 1):
            if values[index] == 0.0:
                intervals.append((low + step * index, low + step * index))
            elif values[index] * values[index + 1] < 0.0:
                intervals.append((low + step * index, low + step * (index + 1)))
        target = 0.0 if previous is None else float(previous)
        if intervals:
            chosen = min(intervals, key=lambda pair: abs(0.5 * (pair[0] + pair[1]) - target))
            low, high = chosen
            if low == high:
                return low, 0.0
    middle = 0.5 * (low + high)
    for _ in range(max_iterations):
        middle = 0.5 * (low + high)
        f_middle = _axial_force(section, middle, curvature) - axial_force
        if abs(f_middle) <= residual or 0.5 * (high - low) <= 1e-18:
            return middle, f_middle
        if f_middle < 0.0:
            low = middle
        else:
            high = middle
    return middle, _axial_force(section, middle, curvature) - axial_force


def _moment_at(section: PanelSection, curvature: float, strain_shift: float) -> float:
    moment = 0.0
    for fibre in section.fibres:
        stress = fibre.stress(strain_shift + curvature * fibre.offset)
        moment += stress * fibre.area * fibre.offset
    return moment


def _fibres_at_limit(
    section: PanelSection, curvature: float, strain_shift: float
) -> tuple[str, ...]:
    names = []
    for fibre in section.fibres:
        strain = strain_shift + curvature * fibre.offset
        if strain < 0.0 and -strain >= fibre.curve.limit_strain * (1.0 - 1e-12):
            names.append(fibre.name)
    return tuple(names)


def _local_maxima(moments: list[float], *, tolerance: float = 1e-9) -> tuple[int, ...]:
    """Strict local maxima, with an equal-valued run counted once.

    A limiting-stress *plateau* law has no limit point: ``M(chi)`` increases
    monotonically and saturates at the plastic moment, so a naive
    ``M[k] >= M[k-1] and M[k] >= M[k+1]`` test reports every point of the
    plateau as a peak.  Runs within ``tolerance`` of the running maximum are
    therefore collapsed into a single peak, whose index is the first point of
    the run (the knee).  A genuine second maximum, above the first by more than
    ``tolerance``, is reported separately.
    """
    if not moments:
        return ()
    scale = max(abs(value) for value in moments)
    band = tolerance * max(scale, 1e-30)
    peaks: list[int] = []
    index = 0
    count = len(moments)
    while index < count - 1:
        if moments[index + 1] > moments[index] + band:
            index += 1
            continue
        peaks.append(index)
        level = moments[index]
        index += 1
        while index < count - 1 and moments[index + 1] <= moments[index] + band:
            if moments[index] > level + band:
                peaks.append(index)
                level = moments[index]
            index += 1
    if not peaks:
        # Strictly rising over the whole sweep: the ultimate point is the last
        # increment, and there is exactly one of them.
        return (count - 1,)
    return tuple(peaks)


def _is_plateau(moments: list[float], index: int, *, tolerance: float = 1e-9) -> bool:
    """True when the peak sits on a flat run of at least three points."""
    if index + 1 >= len(moments) or index == 0:
        return index + 1 >= len(moments)
    scale = max(abs(value) for value in moments)
    band = tolerance * max(scale, 1e-30)
    return abs(moments[index + 1] - moments[index]) <= band


def moment_curvature(
    section: PanelSection,
    *,
    steps: int = 200,
    maximum_strain_factor: float = 6.0,
    axial_force: float = 0.0,
    tolerance: float = 1e-10,
) -> PanelCollapse:
    """Integrate the ``M``--``chi`` path in equal curvature increments.

    ``maximum_strain_factor`` sets the sweep end: the curvature at which the
    extreme fibre reaches ``maximum_strain_factor * eps_y``.  For every
    curvature the reference strain is re-solved for axial equilibrium before
    the moment is integrated, which is Smith's method with the neutral axis
    free to migrate.  The energy ledger assumes ``axial_force = 0`` (no work
    of an external axial load is accounted for).
    """
    if not isinstance(steps, bool) and not isinstance(steps, int):
        raise ValueError("steps must be an integer")
    if steps < 8:
        raise ValueError("steps must be at least 8")
    factor = _positive("maximum_strain_factor", maximum_strain_factor)
    force = _finite("axial_force", axial_force)
    if abs(force) > 1e-9 * section.total_area * section.yield_stress:
        raise ValueError(
            "the energy ledger is only balanced for axial_force = 0; integrate a "
            "preloaded section through the bending term explicitly instead"
        )
    axis = section.elastic_neutral_axis
    extent = max(abs(f.offset - axis) for f in section.fibres)
    yield_strain = section.yield_stress / section.young
    chi_end = factor * yield_strain / extent
    points: list[CurvaturePoint] = []
    previous_shift = 0.0
    for index in range(1, steps + 1):
        curvature = chi_end * index / steps
        shift, residual = _solve_reference_strain(
            section, curvature, force, previous=previous_shift, tolerance=tolerance
        )
        previous_shift = shift
        moment = _moment_at(section, curvature, shift)
        points.append(
            CurvaturePoint(
                curvature=curvature,
                moment=moment,
                reference_strain=shift,
                axial_force=_axial_force(section, shift, curvature),
                axial_residual=residual,
                fibres_at_limit=_fibres_at_limit(section, curvature, shift),
            )
        )
    moments = [point.moment for point in points]
    peak_index = max(range(len(moments)), key=lambda i: moments[i])
    # With a strain-driven (deformation-theory) fibre law the internal work is
    # path independent, so the ledger compares the work of the bending moment
    # with the internal energy of the *final* state only.  Dividing the energy
    # of every point by the number of points (the obvious mistake) destroys the
    # identity dU = M dchi and produces a meaningless residual.
    final = points[-1]
    internal = 0.0
    for fibre in section.fibres:
        internal += fibre.energy(final.reference_strain + final.curvature * fibre.offset)
    external = 0.0
    previous_curvature = 0.0
    previous_moment = 0.0
    for point in points:
        external += 0.5 * (point.moment + previous_moment) * (point.curvature - previous_curvature)
        previous_curvature = point.curvature
        previous_moment = point.moment
    return PanelCollapse(
        section=section,
        points=tuple(points),
        ultimate_moment=moments[peak_index],
        ultimate_curvature=points[peak_index].curvature,
        ultimate_index=peak_index,
        internal_work=internal,
        external_work=external,
        peak_refined=False,
        local_maxima=_local_maxima(moments),
        plateau_at_peak=_is_plateau(moments, peak_index),
    )


def axial_ultimate(
    section: PanelSection, *, steps: int = 400, maximum_strain_factor: float = 6.0
) -> dict[str, float]:
    """Uniform-strain compression ultimate of the same fibre set.

    ``sigma(eps) = sum A_i sigma_i(eps) / sum A_i`` -- the area-weighted limiting
    stress chain, but computed from the fibre laws of this section (including
    any softening) and maximised over the sweep.  This is the quantity an
    elasto-plastic unit-cell Riks path reports, so it is the bridge between the
    closed forms and :func:`fe_panel_compression_path`.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 8:
        raise ValueError("steps must be an integer of at least 8")
    factor = _positive("maximum_strain_factor", maximum_strain_factor)
    yield_strain = section.yield_stress / section.young
    strain_end = factor * yield_strain
    area = section.total_area
    best = (-1.0, 0.0)
    for index in range(1, steps + 1):
        strain = -strain_end * index / steps
        stress = sum(f.area * f.stress(strain) for f in section.fibres) / area
        magnitude = -stress
        if magnitude > best[0]:
            best = (magnitude, -strain)
    return {
        "average_ultimate_stress": best[0],
        "ultimate_over_yield": best[0] / section.yield_stress,
        "ultimate_compressive_strain": best[1],
        "ultimate_over_yield_strain": best[1] / yield_strain,
    }


def increment_refinement_study(
    section: PanelSection,
    *,
    steps: tuple[int, ...] = (50, 100, 200, 400),
    maximum_strain_factor: float = 6.0,
) -> dict[str, object]:
    """Refine the curvature sweep and report peak and ledger convergence."""
    if len(steps) < 2:
        raise ValueError("at least two step counts are required")
    rows = []
    for count in steps:
        collapse = moment_curvature(
            section, steps=count, maximum_strain_factor=maximum_strain_factor
        )
        rows.append(
            {
                "steps": int(count),
                "ultimate_moment": collapse.ultimate_moment,
                "ultimate_curvature": collapse.ultimate_curvature,
                "energy_residual": collapse.energy_residual,
                "peak_unique": collapse.peak_unique,
                "monotonic_to_peak": collapse.monotonic_to_peak,
            }
        )
    finest = rows[-1]
    for row in rows[:-1]:
        row["moment_relative_difference"] = (
            abs(row["ultimate_moment"] - finest["ultimate_moment"]) / finest["ultimate_moment"]
        )
        row["curvature_relative_difference"] = (
            abs(row["ultimate_curvature"] - finest["ultimate_curvature"])
            / finest["ultimate_curvature"]
        )
    finest["moment_relative_difference"] = 0.0
    finest["curvature_relative_difference"] = 0.0
    residuals = [row["energy_residual"] for row in rows]
    return {
        "rows": rows,
        "finest_ultimate_moment": finest["ultimate_moment"],
        "finest_ultimate_curvature": finest["ultimate_curvature"],
        "energy_residual_decreasing": all(b <= a for a, b in zip(residuals, residuals[1:])),
        "all_peaks_unique": all(row["peak_unique"] for row in rows),
        "all_monotonic_to_peak": all(row["monotonic_to_peak"] for row in rows),
    }


# --------------------------------------------------------------------------
# 4. cross-verification against TensorFEM discretisations
# --------------------------------------------------------------------------
def column_euler_fem_check(
    panel: StiffenedPanel,
    *,
    elements: tuple[int, ...] = (8, 16, 32, 64),
    modes: int = 1,
) -> dict[str, object]:
    """Compare the Johnson--Ostenfeld anchor with a beam eigenvalue solve.

    The FE eigenproblem uses :func:`tensorfem.buckling.pinned_pinned_column_buckling`
    on the stiffener-plus-attached-plate ``EI``.  Its Rayleigh-quotient
    convergence is quadratic from above, so the error sequence itself is the
    mesh-convergence evidence for the closed-form Euler stress.
    """
    from .buckling import euler_pinned_critical_load, pinned_pinned_column_buckling

    panel.validate()
    train = []
    for count in elements:
        result = pinned_pinned_column_buckling(
            panel.span, int(count), panel.young * panel.column_inertia, modes=1
        )
        critical_load = float(result.load_factors[0])
        closed = euler_pinned_critical_load(panel.span, panel.young * panel.column_inertia)
        train.append(
            {
                "elements": int(count),
                "fe_critical_load": critical_load,
                "euler_critical_load": closed,
                "relative_error": abs(critical_load / closed - 1.0),
            }
        )
    closed_stress = panel.column_euler_stress
    return {
        "panel": panel.summary(),
        "column_euler_stress": closed_stress,
        "johnson_ostenfeld_stress": panel.column_limiting_stress,
        "johnson_ostenfeld_over_yield": panel.column_limiting_stress / panel.yield_stress,
        "mesh_train": train,
        "finest_relative_error": train[-1]["relative_error"],
    }


def plate_beta_fem_check(
    panel: StiffenedPanel,
    *,
    grids: tuple[tuple[int, int], ...] = ((6, 6), (12, 12), (24, 24)),
) -> dict[str, object]:
    """Compare the von Karman slenderness anchor ``beta`` with a plate eigen-solve.

    ``beta = (b/t) sqrt(sigma_y / E)`` is defined from the elastic plate
    buckling stress with ``k = 4``.  The finite-difference eigenvalue of
    :func:`tensorfem.marine_plate_buckling.finite_difference_uniaxial_buckling`
    supplies an independent critical stress; the buckling coefficient it
    implies and the slenderness it implies are reported per grid.
    """
    from .marine_plate_buckling import (
        finite_difference_uniaxial_buckling,
        isotropic_plate_rigidities,
        navier_uniaxial_buckling_load,
    )

    panel.validate()
    rigidity_x, rigidity_y, coupling = isotropic_plate_rigidities(
        panel.young, panel.poisson, panel.plate_thickness
    )
    train = []
    for grid_x, grid_y in grids:
        result = finite_difference_uniaxial_buckling(
            length=panel.span,
            width=panel.spacing,
            thickness=panel.plate_thickness,
            rigidity_x=rigidity_x,
            rigidity_y=rigidity_y,
            coupling_rigidity=coupling,
            grid_x=int(grid_x),
            grid_y=int(grid_y),
        )
        critical = float(result.critical_stress)
        reference_load, _m, _n = navier_uniaxial_buckling_load(
            length=panel.span,
            width=panel.spacing,
            rigidity_x=rigidity_x,
            rigidity_y=rigidity_y,
            coupling_rigidity=coupling,
            maximum_half_waves=64,
        )
        reference = float(reference_load) / panel.plate_thickness
        sigma_e = math.pi**2 * panel.plate_rigidity / (panel.spacing**2 * panel.plate_thickness)
        train.append(
            {
                "grid": [int(grid_x), int(grid_y)],
                "fe_critical_stress": critical,
                "navier_critical_stress": reference,
                "fe_over_navier_error": abs(critical / reference - 1.0),
                "buckling_coefficient_fe": critical / sigma_e,
                "coefficient_error_vs_k4": abs(critical / sigma_e / 4.0 - 1.0),
            }
        )
    return {
        "panel_slenderness_beta": panel.plate_slenderness,
        "plate_buckling_stress_k4": panel.plate_buckling_stress,
        "mesh_train": train,
        "finest_coefficient_error_vs_k4": train[-1]["coefficient_error_vs_k4"],
        "note": (
            "beta = (b/t) sqrt(sigma_y/E) is a geometric/material parameter; the "
            "closed-form elastic anchor sigma_E = 4 pi^2 D / (b^2 t) behind the "
            "Faulkner fit is what this eigen-solve verifies, through the buckling "
            "coefficient k -> 4 with mesh refinement."
        ),
    }


def fe_panel_compression_path(
    panel: StiffenedPanel,
    *,
    nx: int = 3,
    ny: int = 2,
    layers: int = 2,
    steps: int = 24,
    step_size: float | None = None,
    load_fraction: float = 0.4,
    max_iterations: int = 10,
    include_points: bool = True,
) -> dict[str, object]:
    """Elasto-plastic Shell4 Riks path of the half-bay unit cell in compression.

    Thin wrapper around :func:`tensorfem.stiffened_panel_unit_fe.trace_panel_unit_path`
    that returns the acceptance quantities the section analysis needs: the
    average-stress ultimate, its end shortening and the post-peak drop.  The
    arc-length load factor is the average applied stress, so the result is
    directly comparable with :func:`axial_ultimate`.
    """
    from .stiffened_panel_unit_fe import build_panel_unit, trace_panel_unit_path, trace_summary

    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 4:
        raise ValueError("steps must be an integer of at least 4")
    fraction = _positive("load_fraction", load_fraction)
    case = build_panel_unit(panel, nx=nx, ny=ny, layers=layers)
    ds = 0.02 * panel.plate_thickness if step_size is None else _positive("step_size", step_size)
    load_scale = ds / (fraction * panel.yield_stress)
    trace = trace_panel_unit_path(
        case, steps=steps, step_size=ds, load_scale=load_scale, max_iterations=max_iterations
    )
    summary = trace_summary(case, trace)
    result: dict[str, object] = {
        "mesh": {"nx": int(nx), "ny": int(ny), "layers": int(layers)},
        "element_counts": list(case.element_counts),
        "free_dofs": int(len(case.free_dofs())),
        "steps": int(steps),
        "step_size": ds,
        "load_fraction": fraction,
        "ultimate_stress": summary["ultimate_stress"],
        "ultimate_over_yield": summary["ultimate_over_yield"],
        "ultimate_end_shortening": summary["ultimate_end_shortening"],
        "has_peak": summary["has_peak"],
        "post_peak_drop": summary["post_peak_drop"],
        "converged": summary["converged"],
        "points_completed": summary["points_completed"],
        "closed_form_property_based_over_yield": panel.panel_ultimate_stress / panel.yield_stress,
    }
    if include_points:
        result["points"] = [
            {
                "step": point["step"],
                "stress_over_yield": point["stress_over_yield"],
                "end_shortening": point["end_shortening"],
                "maximum_plastic_strain": point["maximum_plastic_strain"],
            }
            for point in summary["points"]
        ]
    return result
