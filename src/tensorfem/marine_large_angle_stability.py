"""Exact large-angle righting-arm oracle for wall-sided box sections.

Verification debt this module closes
-----------------------------------
``tensorfem.marine_hydrostatics`` implements the righting arm as ``GM*sin(phi)``
and refuses heel angles beyond 10 degrees.  ``docs/marine-hydrostatics.md``
states that "the sine form implemented by the solver differs by less than 3%
over the declared range"; that claim was only ever checked against the same
linear relation, so no test could have detected its failure.  The true
righting arm of a wall-sided body is

    GZ(phi) = sin(phi) * [ GM + 0.5 * BM0 * tan(phi)**2 ],

which is exact only while the free surface still cuts the two vertical sides,
i.e. up to the starboard deck-edge immersion angle

    phi_deck = atan(2 * (D - T) / B).

This module supplies that exact reference *from first principles* (immersed
section geometry, not the textbook formula) so the merged linear model can be
measured against an independent oracle instead of against itself.

Conventions
-----------
Right-handed axes, midship section in the ``(y, z)`` plane, ``y`` to
starboard, ``z`` up from the keel plane, unit length along the ship.
A point at body coordinates ``(y, z)`` sits at space height
``h = z*cos(phi) - y*sin(phi)`` once the section is heeled ``phi`` to
starboard, so the free surface is the line ``h = h0`` and the immersed region
is ``{z*cos(phi) - y*sin(phi) <= h0}``.

The free-floating equilibrium is solved, not assumed: ``h0`` is found by
bisecting the immersed-area residual ``A(phi, h0) - B*T``.  The closed form
``h0 = T*cos(phi)`` then becomes an independent check of that solve rather
than an input to it.

``GZ`` is the space-horizontal distance from ``G`` to the buoyancy line,
``GZ = (Y_B - Y_G)`` with ``(Y, Z) = (y*cos(phi) + z*sin(phi),
z*cos(phi) - y*sin(phi))``; positive heeling to starboard, positive righting
arm.  ``B`` is the centroid of the immersed section (unit length, so area and
volume coincide) and ``G`` is ``KG`` on the centreline.

Scope: one wall-sided rectangular section, rigid body, no free-surface liquid
effects, no trim, no flooding.  Beyond ``phi_deck`` water over the deck makes
the solution a geometric submergence exercise, not a stability claim; the
envelope is therefore declared closed there and callers can inspect
:meth:`BoxBarge.deck_immersed`.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

__all__ = [
    # main (m3275): exact-arm oracle
    "BoxBarge",
    "ImmersedSection",
    # branch (m3495): independent wall-sided implementation
    "SCHEMA",
    "SMALL_ANGLE_LIMIT_DEG",
    "WallSidedBarge",
    "gz_closed_form",
    "polygon_section",
]

DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665


@dataclass(frozen=True)
class ImmersedSection:
    """Result of the exact section clip at one heel angle."""

    heel_deg: float
    free_surface_height: float
    area: float
    centroid_y: float
    centroid_z: float
    waterline_width: float
    deck_immersed: bool
    vertices: tuple[tuple[float, float], ...]

    def relative_area_error(self, target: float) -> float:
        return abs(self.area - target) / target


def _finite(value: float, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        raise ValueError(f"{name} must be finite{' and positive' if positive else ''}")
    return number


def _polygon_area_centroid(points: list[tuple[float, float]]):
    """Signed area and centroid of a simple polygon (shoelace)."""
    count = len(points)
    if count < 3:
        return 0.0, 0.0, 0.0
    twice_area = 0.0
    cy = 0.0
    cz = 0.0
    for index in range(count):
        y0, z0 = points[index]
        y1, z1 = points[(index + 1) % count]
        cross = y0 * z1 - y1 * z0
        twice_area += cross
        cy += (y0 + y1) * cross
        cz += (z0 + z1) * cross
    area = 0.5 * twice_area
    if abs(area) < 1e-300:
        return 0.0, 0.0, 0.0
    return area, cy / (6.0 * area), cz / (6.0 * area)


def _clip_below(points: list[tuple[float, float]], sin_phi: float,
                cos_phi: float, h0: float) -> list[tuple[float, float]]:
    """Keep the part of a convex polygon with ``z*cos - y*sin <= h0``.

    ``points`` must be counter-clockwise.  Sutherland-Hodgman against a single
    half-plane: exact for convex input, no tolerance needed.
    """
    def level(point: tuple[float, float]) -> float:
        return point[1] * cos_phi - point[0] * sin_phi - h0

    clipped: list[tuple[float, float]] = []
    count = len(points)
    for index in range(count):
        current = points[index]
        following = points[(index + 1) % count]
        current_level = level(current)
        following_level = level(following)
        current_inside = current_level <= 0.0
        following_inside = following_level <= 0.0
        if current_inside:
            clipped.append(current)
        if current_inside != following_inside:
            ratio = current_level / (current_level - following_level)
            clipped.append((current[0] + ratio * (following[0] - current[0]),
                            current[1] + ratio * (following[1] - current[1])))
    return clipped


def _waterline_chord(points: list[tuple[float, float]], sin_phi: float,
                     cos_phi: float, h0: float) -> float:
    """Length of the free-surface chord through the section boundary."""
    crossings: list[tuple[float, float]] = []
    count = len(points)
    for index in range(count):
        a = points[index]
        b = points[(index + 1) % count]
        level_a = a[1] * cos_phi - a[0] * sin_phi - h0
        level_b = b[1] * cos_phi - b[0] * sin_phi - h0
        if abs(level_a) < 1e-13:
            crossings.append(a)
        if level_a * level_b < 0.0:
            ratio = level_a / (level_a - level_b)
            crossings.append((a[0] + ratio * (b[0] - a[0]),
                              a[1] + ratio * (b[1] - a[1])))
    unique: list[tuple[float, float]] = []
    for point in crossings:
        if not any(math.dist(point, other) < 1e-12 for other in unique):
            unique.append(point)
    if len(unique) < 2:
        return 0.0
    return max(math.dist(unique[i], unique[j])
               for i in range(len(unique)) for j in range(i + 1, len(unique)))


@dataclass(frozen=True)
class BoxBarge:
    """Rectangular, wall-sided barge section with exact large-angle geometry."""

    length: float
    beam: float
    depth: float
    draft: float
    kg: float
    density: float = DEFAULT_DENSITY
    gravity: float = DEFAULT_GRAVITY

    def __post_init__(self) -> None:
        _finite(self.length, "length", positive=True)
        _finite(self.beam, "beam", positive=True)
        _finite(self.depth, "depth", positive=True)
        _finite(self.draft, "draft", positive=True)
        _finite(self.kg, "kg")
        _finite(self.density, "density", positive=True)
        _finite(self.gravity, "gravity", positive=True)
        if not 0.0 < float(self.draft) < float(self.depth):
            raise ValueError("equilibrium draft must lie between 0 and the depth")

    # -- upright hydrostatics (identical to the merged linear model) -------- #
    @property
    def volume(self) -> float:
        return self.length * self.beam * self.draft

    @property
    def displacement_mass(self) -> float:
        return self.density * self.volume

    @property
    def kb(self) -> float:
        return 0.5 * self.draft

    @property
    def bm(self) -> float:
        return self.beam**2 / (12.0 * self.draft)

    @property
    def km(self) -> float:
        return self.kb + self.bm

    @property
    def gm(self) -> float:
        return self.km - self.kg

    @property
    def deck_edge_angle_deg(self) -> float:
        """Heel at which the starboard deck edge touches the free surface."""
        return math.degrees(math.atan2(2.0 * (self.depth - self.draft), self.beam))

    @property
    def bottom_exposure_angle_deg(self) -> float:
        """Heel at which the rising-side bilge leaves the water."""
        return math.degrees(math.atan2(2.0 * self.draft, self.beam))

    @property
    def wall_sided_envelope_deg(self) -> float:
        """Largest heel for which the textbook closed form stays exact.

        The closed form assumes the free surface cuts both vertical sides, so
        it needs the starboard deck edge still dry *and* the rising-side bilge
        still submerged: ``min(phi_deck, atan(2T/B))``.
        """
        return min(self.deck_edge_angle_deg, self.bottom_exposure_angle_deg)

    def section_polygon(self) -> list[tuple[float, float]]:
        half = 0.5 * self.beam
        return [(-half, 0.0), (half, 0.0), (half, self.depth), (-half, self.depth)]

    # -- exact immersed geometry ------------------------------------------- #
    def free_surface_height_closed_form(self, heel_deg: float) -> float:
        """Free-surface height from wall-sided volume closure, ``T*cos(phi)``."""
        return self.draft * math.cos(math.radians(heel_deg))

    def _immersed_area(self, sin_phi: float, cos_phi: float, h0: float) -> float:
        clipped = _clip_below(self.section_polygon(), sin_phi, cos_phi, h0)
        area, _, _ = _polygon_area_centroid(clipped)
        return abs(area)

    def solve_free_surface_height(self, heel_deg: float,
                                 tolerance: float = 1e-14,
                                 max_iterations: int = 400) -> float:
        """Bisect ``h0`` so that the immersed area equals ``B*T``."""
        _finite(heel_deg, "heel_deg")
        if abs(math.radians(heel_deg)) >= 0.5 * math.pi:
            raise ValueError("heel angle must stay inside (-90, 90) degrees")
        radians = math.radians(heel_deg)
        sin_phi = math.sin(radians)
        cos_phi = math.cos(radians)
        target = self.beam * self.draft
        radius = math.hypot(self.beam, self.depth)
        low, high = -radius, self.depth + radius
        if self._immersed_area(sin_phi, cos_phi, low) > target:
            raise ValueError("no flotation solution below the section")
        if self._immersed_area(sin_phi, cos_phi, high) < target:
            raise ValueError("no flotation solution above the section")
        for _ in range(max_iterations):
            middle = 0.5 * (low + high)
            if self._immersed_area(sin_phi, cos_phi, middle) < target:
                low = middle
            else:
                high = middle
            if high - low < tolerance * max(1.0, self.depth):
                break
        return 0.5 * (low + high)

    def immersed_section(self, heel_deg: float, *, offset: float | None = None) -> ImmersedSection:
        """Exact immersed section at ``heel_deg`` (degrees, starboard positive)."""
        _finite(heel_deg, "heel_deg")
        radians = math.radians(heel_deg)
        if abs(radians) >= 0.5 * math.pi:
            raise ValueError("heel angle must stay inside (-90, 90) degrees")
        sin_phi = math.sin(radians)
        cos_phi = math.cos(radians)
        h0 = self.solve_free_surface_height(heel_deg) if offset is None else float(offset)
        clipped = _clip_below(self.section_polygon(), sin_phi, cos_phi, h0)
        area, centroid_y, centroid_z = _polygon_area_centroid(clipped)
        deck_immersed = abs(heel_deg) > self.deck_edge_angle_deg + 1e-12
        return ImmersedSection(
            heel_deg=float(heel_deg),
            free_surface_height=h0,
            area=abs(area),
            centroid_y=centroid_y,
            centroid_z=centroid_z,
            waterline_width=_waterline_chord(self.section_polygon(), sin_phi, cos_phi, h0),
            deck_immersed=deck_immersed,
            vertices=tuple(clipped),
        )

    # -- righting arms ----------------------------------------------------- #
    def gz(self, heel_deg: float, *, offset: float | None = None) -> float:
        """Exact righting arm from the immersed-section centroid."""
        section = self.immersed_section(heel_deg, offset=offset)
        radians = math.radians(section.heel_deg)
        y_g = 0.0
        y_b, z_b = section.centroid_y, section.centroid_z
        return (y_b - y_g) * math.cos(radians) + (z_b - self.kg) * math.sin(radians)

    def restoring_moment(self, heel_deg: float) -> float:
        """``displacement * g * GZ`` using the exact arm (N m)."""
        return self.displacement_mass * self.gravity * self.gz(heel_deg)

    def wall_sided_gz(self, heel_deg: float) -> float:
        """Closed form ``sin(phi)*(GM + 0.5*BM0*tan(phi)^2)``."""
        radians = math.radians(heel_deg)
        return math.sin(radians) * (self.gm + 0.5 * self.bm * math.tan(radians) ** 2)

    def linear_gz(self, heel_deg: float) -> float:
        """Merged linear model arm ``GM*sin(phi)``."""
        return self.gm * math.sin(math.radians(heel_deg))

    def linear_arm_relative_error(self, heel_deg: float) -> float:
        """Relative error of the merged linear arm against the exact arm."""
        exact = self.gz(heel_deg)
        if abs(exact) < 1e-14:
            return 0.0
        return abs(self.linear_gz(heel_deg) - exact) / abs(exact)

    # -- envelope diagnostics ---------------------------------------------- #
    def zero_arm_angle_deg(self) -> float | None:
        """Non-zero ``GZ = 0`` root of an unstable wall-sided section.

        For ``GM < 0`` the arm starts negative at upright and changes sign at
        ``tan(phi) = sqrt(-2*GM/BM)``: the angle at which the section would
        lurch.  Stable sections (``GM >= 0``) have no positive root inside the
        wall-sided envelope, hence ``None``.
        """
        if self.gm >= 0.0:
            return None
        return math.degrees(math.atan(math.sqrt(-2.0 * self.gm / self.bm)))

    def dynamic_stability(self, heel_deg: float, panels: int = 1024) -> float:
        """``integral(0, phi) GZ dphi`` by composite Simpson on the exact arm.

        The free-surface height is supplied from the closed form ``T*cos(phi)``
        to avoid a bisection per sample; that identity is itself verified
        against the solved equilibrium in the test suite, and the integrand
        remains the geometric centroid of the clipped section.
        """
        if panels % 2:
            raise ValueError("panels must be even")
        upper = float(heel_deg)
        if upper == 0.0:
            return 0.0
        # Sample angles are in degrees; the quadrature step must be in radians.
        angle_step = upper / panels
        weight_step = math.radians(upper) / panels

        def arm(angle: float) -> float:
            return self.gz(angle, offset=self.free_surface_height_closed_form(angle))

        total = arm(0.0) + arm(upper)
        for index in range(1, panels):
            weight = 4.0 if index % 2 else 2.0
            total += weight * arm(index * angle_step)
        return total * weight_step / 3.0

    def wall_sided_dynamic_stability(self, heel_deg: float) -> float:
        """Closed form ``GM*(1-cos) + 0.5*BM*(sec+cos-2)``."""
        radians = math.radians(heel_deg)
        return (self.gm * (1.0 - math.cos(radians))
                + 0.5 * self.bm * (1.0 / math.cos(radians) + math.cos(radians) - 2.0))

    # -- sampling ---------------------------------------------------------- #
    def gz_curve(self, heel_max_deg: float | None = None,
                 samples: int = 181) -> list[tuple[float, float, float]]:
        """``(heel_deg, gz, linear_gz)`` triples used by the audit report."""
        if samples < 2:
            raise ValueError("samples must be at least 2")
        upper = self.deck_edge_angle_deg if heel_max_deg is None else float(heel_max_deg)
        if upper < 0.0:
            raise ValueError("heel_max_deg must be non-negative")
        return [
            (upper * index / (samples - 1),
             self.gz(upper * index / (samples - 1)),
             self.linear_gz(upper * index / (samples - 1)))
            for index in range(samples)
        ]


# ============================================================================
# Exact large-angle intact stability of a wall-sided (rectangular) barge.
#
# Why this module exists
# ----------------------
# ``tensorfem.marine_hydrostatics`` deliberately stops at ``|heel| <= 10 deg`` and
# evaluates the righting arm as ``GZ = GM * sin(phi)`` there.  That model is the
# small-angle limit only: it discards the second-order rise of the centre of
# buoyancy that a wall-sided hull produces once the waterplane tilts.  For a
# *rectangular* section, however, the intact righting arm has an exact closed form
# over the whole upright range, the classical wall-sided formula
#
#     GZ(phi) = sin(phi) * [ GM + 0.5 * BM * tan^2(phi) ],          (1)
#
# with ``GM = KB + BM - KG``, ``KB = T/2``, ``BM = I_wl / nabla``,
# ``I_wl = L B^3 / 12`` and ``nabla = L B T`` (Biran & Lopez-Pulido, *Ship
# Hydrostatics and Stability*, 2nd ed., wall-sided righting-lever identity; the
# same identity is quoted in Derrett, *Ship Stability for Masters and Mates*).
# Equation (1) follows from the standard lever decomposition
#
#     GZ = y_B cos(phi) + (KB' - KG) sin(phi),
#     y_B  = BM tan(phi),   KB' = T/2 + 0.5 BM tan^2(phi),           (2)
#
# which is the form used here as the *independent* oracle: this module recovers
# ``y_B`` and ``KB'`` from polygon geometry (clip the section with the tilted
# waterplane, take the shoelace area/centroid, and solve for the sinkage that keeps
# displacement constant) and never evaluates equation (1) on that path.
#
# Validity of equation (1) is bounded twice: the deck edge must stay dry
# (``tan|phi| < 2 (D - T) / B``) **and** the keel must stay submerged
# (``tan|phi| < 2 T / B``).  Whichever binds first ends the closed form.  Beyond
# a keel emergence the section becomes triangular at the fixed draft and the
# floating body sinks; :meth:`WallSidedBarge.gz_numerical` still returns the
# correct arm there because it solves for the equilibrium waterplane, which is
# exactly the regime where equation (1) must not be extrapolated.
#
# Scope boundary: rectangular, wall-sided, intact, upright-to-deck-edge only.  No
# deck-immersion/downflooding, cross-curves, free surface, damage or trim; this is
# not a rules-compliance or approval tool.
# """

SCHEMA = "tensorfem.marine-large-angle-stability.v1"
SMALL_ANGLE_LIMIT_DEG = 10.0

#: Relative tolerance used by the polygon clipping predicates.
_CLIP_TOL = 1e-14


def _positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return value


def _clip_polygon(
    polygon: Sequence[tuple[float, float]], tan_phi: float, offset: float
) -> list[tuple[float, float]]:
    """Clip a polygon with the half plane ``z <= draft + y*tan_phi + offset``.

    ``offset`` is the unknown sinkage of the equilibrium waterplane; the
    waterline itself is ``z = draft + y*tan(phi) + offset``.
    """

    def signed(y: float, z: float) -> float:
        return z - (tan_phi * y + offset)

    out: list[tuple[float, float]] = []
    count = len(polygon)
    for index in range(count):
        y1, z1 = polygon[index]
        y2, z2 = polygon[(index + 1) % count]
        f1 = signed(y1, z1)
        f2 = signed(y2, z2)
        inside1 = f1 <= _CLIP_TOL
        inside2 = f2 <= _CLIP_TOL
        if inside1:
            out.append((y1, z1))
        if inside1 != inside2:
            denominator = f1 - f2
            if denominator != 0.0:
                t = f1 / denominator
                out.append((y1 + t * (y2 - y1), z1 + t * (z2 - z1)))
    return out


def _area_centroid(
    polygon: Sequence[tuple[float, float]],
) -> tuple[float, float, float]:
    """Return ``(area, y_centroid, z_centroid)`` of a simple polygon."""

    if len(polygon) < 3:
        return 0.0, float("nan"), float("nan")
    twice_area = 0.0
    y_moment = 0.0
    z_moment = 0.0
    count = len(polygon)
    for index in range(count):
        y1, z1 = polygon[index]
        y2, z2 = polygon[(index + 1) % count]
        cross = y1 * z2 - y2 * z1
        twice_area += cross
        y_moment += (y1 + y2) * cross
        z_moment += (z1 + z2) * cross
    area = 0.5 * twice_area
    if area == 0.0:
        return 0.0, float("nan"), float("nan")
    return abs(area), y_moment / (3.0 * twice_area), z_moment / (3.0 * twice_area)


def polygon_section(
    beam: float,
    depth: float,
    draft: float,
    heel_deg: float,
    *,
    strict: bool = True,
) -> dict[str, float | bool]:
    """Submerged box section at heel, with the equilibrium waterplane solved.

    Returns the clipped area, its centroid (``y_B``, ``KB'``), the sinkage that
    conserves displacement, and the validity flags for deck immersion and keel
    emergence.  This is the independent geometric oracle used to cross-check the
    wall-sided closed form.
    """

    beam = _positive(beam, "beam")
    depth = _positive(depth, "depth")
    draft = _positive(draft, "draft")
    if not draft < depth:
        raise ValueError("draft must be below molded depth")
    heel = _finite(heel_deg, "heel_deg")
    phi = math.radians(heel)
    tan_phi = math.tan(phi)
    if abs(tan_phi) > 1e6:
        raise ValueError("heel angle is too large for a section calculation")

    target = beam * draft
    rectangle = [
        (-0.5 * beam, 0.0),
        (0.5 * beam, 0.0),
        (0.5 * beam, depth),
        (-0.5 * beam, depth),
    ]

    def area_at(offset: float) -> tuple[float, float, float]:
        clipped = _clip_polygon(rectangle, tan_phi, draft + offset)
        return _area_centroid(clipped)

    def area_of(offset: float) -> float:
        return area_at(offset)[0]

    scale = beam + depth
    lo, hi = -scale, scale
    for _ in range(200):
        if area_of(lo) <= target:
            break
        lo *= 2.0
    else:  # pragma: no cover - defensive
        raise ValueError("could not bracket a submerged section")
    for _ in range(200):
        if area_of(hi) >= target:
            break
        hi *= 2.0
    else:  # pragma: no cover - defensive
        raise ValueError("could not bracket a submerged section")
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if area_of(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-15 * scale:
            break
    offset = 0.5 * (lo + hi)
    area, y_b, z_b = area_at(offset)

    half_tilt = 0.5 * beam * abs(tan_phi)
    highest = draft + half_tilt + offset
    lowest = draft - half_tilt + offset
    deck_immersed = highest > depth * (1.0 + 1e-9)
    keel_emerged = lowest < -1e-9 * depth
    if strict and deck_immersed:
        raise ValueError(
            "deck edge is immersed at this heel "
            f"(waterline reaches {highest:.6g} m of a {depth:.6g} m deep hull); "
            "the intact wall-sided model does not apply"
        )
    return {
        "heel_angle_deg": heel,
        "sinkage": offset,
        "area": area,
        "area_error": area - target,
        "y_b": y_b,
        "z_b": z_b,
        "highest_waterline": highest,
        "lowest_waterline": lowest,
        "deck_immersed": deck_immersed,
        "keel_emerged": keel_emerged,
    }


@dataclass(frozen=True)
class WallSidedBarge:
    """Rectangular wall-sided barge in SI units.

    ``draft`` is the upright equilibrium draft and ``kg`` the upright centre of
    gravity height; both are fixed properties of the trial condition.
    """

    name: str
    length: float
    beam: float
    depth: float
    draft: float
    kg: float
    fluid_density: float = DEFAULT_DENSITY
    gravity: float = DEFAULT_GRAVITY

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        _positive(self.length, "length")
        _positive(self.beam, "beam")
        _positive(self.depth, "depth")
        _positive(self.draft, "draft")
        _finite(self.kg, "kg")
        _positive(self.fluid_density, "fluid_density")
        _positive(self.gravity, "gravity")
        if not self.draft < self.depth:
            raise ValueError("draft must be positive and below molded depth")

    # -- hydrostatics of the upright condition -------------------------------
    @property
    def displacement_volume(self) -> float:
        return self.length * self.beam * self.draft

    @property
    def displacement_mass(self) -> float:
        return self.fluid_density * self.displacement_volume

    @property
    def kb(self) -> float:
        return 0.5 * self.draft

    @property
    def waterplane_inertia(self) -> float:
        return self.length * self.beam**3 / 12.0

    @property
    def bm(self) -> float:
        return self.waterplane_inertia / self.displacement_volume

    @property
    def km(self) -> float:
        return self.kb + self.bm

    @property
    def gm(self) -> float:
        return self.km - self.kg

    @property
    def stability_state(self) -> str:
        if self.gm > 0.0:
            return "positive_gm"
        if self.gm < 0.0:
            return "negative_gm"
        return "neutral_gm"

    # -- validity limits -----------------------------------------------------
    @property
    def deck_immersion_limit_deg(self) -> float:
        """Geometric deck-edge immersion heel evaluated **at the upright draft**.

        This is the bound that ends the wall-sided closed form: equation (1)
        keeps the waterplane pinned at ``z = T + y tan(phi)``.  Once the keel
        emerges the floating body also sinks or rises, so the *equilibrium*
        deck-immersion heel differs; use
        :meth:`equilibrium_deck_immersion_deg` for that value.
        """

        return math.degrees(math.atan2(2.0 * (self.depth - self.draft), self.beam))

    def equilibrium_deck_immersion_deg(self, *, tol_deg: float = 1e-10) -> float:
        """Heel at which the low-side deck edge reaches the solved waterplane.

        The waterplane here conserves displacement (sinkage solved section by
        section), so this is the physical end of a wall-sided intact trial
        beyond the closed-form range.
        """

        lo, hi = 0.0, 89.0
        if self.section_state(hi, strict=False)["deck_immersed"] is not True:
            return hi
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if self.section_state(mid, strict=False)["deck_immersed"]:
                hi = mid
            else:
                lo = mid
            if hi - lo <= tol_deg:
                break
        return 0.5 * (lo + hi)

    @property
    def keel_emergence_limit_deg(self) -> float:
        """Heel at which the high-side keel corner leaves the water."""

        return math.degrees(math.atan2(2.0 * self.draft, self.beam))

    @property
    def closed_form_limit_deg(self) -> float:
        """End of the wall-sided closed-form range (whichever limit binds first)."""

        return min(self.deck_immersion_limit_deg, self.keel_emergence_limit_deg)

    @property
    def closed_form_limit_reason(self) -> str:
        if self.keel_emergence_limit_deg <= self.deck_immersion_limit_deg:
            return "keel_emergence"
        return "deck_edge_immersion"

    @property
    def angle_of_loll_deg(self) -> float | None:
        """Classical wall-sided angle of loll ``tan^2(phi) = -2 GM / BM``.

        Only defined for negative metacentric height; ``None`` for ``GM >= 0``.
        The identity follows from equation (1) having a non-trivial root.
        """

        if self.gm >= 0.0:
            return None
        return math.degrees(math.atan(math.sqrt(-2.0 * self.gm / self.bm)))

    @property
    def loll_inside_closed_form_range(self) -> bool:
        angle = self.angle_of_loll_deg
        return angle is None or angle <= self.closed_form_limit_deg

    # -- righting arms -------------------------------------------------------
    def gz_closed_form(self, heel_deg: float, *, strict: bool = True) -> float:
        """Equation (1): exact wall-sided righting arm inside the stated range."""

        heel = _finite(heel_deg, "heel_deg")
        limit = self.closed_form_limit_deg
        if strict and abs(heel) > limit * (1.0 + 1e-12):
            raise ValueError(
                f"|heel| = {abs(heel):.6g} deg exceeds the wall-sided closed-form "
                f"limit {limit:.6g} deg ({self.closed_form_limit_reason}); "
                "use gz_numerical() or reduce the heel"
            )
        phi = math.radians(heel)
        return math.sin(phi) * (self.gm + 0.5 * self.bm * math.tan(phi) ** 2)

    def section_state(self, heel_deg: float, *, strict: bool = True) -> dict[str, Any]:
        return polygon_section(
            self.beam, self.depth, self.draft, heel_deg, strict=strict
        )

    def gz_numerical(self, heel_deg: float, *, strict: bool = True) -> float:
        """Equation (2) evaluated from polygon geometry (independent oracle)."""

        state = self.section_state(heel_deg, strict=strict)
        phi = math.radians(float(state["heel_angle_deg"]))
        return float(state["y_b"]) * math.cos(phi) + (
            float(state["z_b"]) - self.kg
        ) * math.sin(phi)

    def gz_small_angle(self, heel_deg: float, *, strict: bool = True) -> float:
        """The ``GM * sin(phi)`` model shipped by ``marine_hydrostatics``."""

        heel = _finite(heel_deg, "heel_deg")
        if strict and abs(heel) > SMALL_ANGLE_LIMIT_DEG:
            raise ValueError(
                f"|heel| = {abs(heel):.6g} deg exceeds the small-angle limit of "
                f"{SMALL_ANGLE_LIMIT_DEG:g} degrees"
            )
        return self.gm * math.sin(math.radians(heel))

    def curve(self, angles_deg: Iterable[float]) -> list[dict[str, Any]]:
        """Tabulate closed form, oracle and small-angle arm for each heel angle."""

        limit = self.closed_form_limit_deg
        rows: list[dict[str, Any]] = []
        for heel in angles_deg:
            heel = float(heel)
            state = self.section_state(heel)
            exact = self.gz_numerical(heel)
            inside = abs(heel) <= limit * (1.0 + 1e-12)
            closed = self.gz_closed_form(heel) if inside else None
            small = self.gz_small_angle(heel) if abs(heel) <= SMALL_ANGLE_LIMIT_DEG else None
            rows.append(
                {
                    "heel_angle_deg": heel,
                    "gz_closed_form": closed,
                    "gz_numerical": exact,
                    "gz_small_angle": small,
                    "closed_form_inside_range": inside,
                    "sinkage": float(state["sinkage"]),
                    "keel_emerged": bool(state["keel_emerged"]),
                    "displacement_error": float(state["area_error"]),
                    "small_angle_relative_error": (
                        None
                        if small is None or exact == 0.0
                        else abs(small - exact) / abs(exact)
                    ),
                }
            )
        return rows

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "schema": SCHEMA,
                "displacement_volume": self.displacement_volume,
                "displacement_mass": self.displacement_mass,
                "kb": self.kb,
                "waterplane_inertia": self.waterplane_inertia,
                "bm": self.bm,
                "km": self.km,
                "gm": self.gm,
                "stability_state": self.stability_state,
                "deck_immersion_limit_deg": self.deck_immersion_limit_deg,
                "keel_emergence_limit_deg": self.keel_emergence_limit_deg,
                "closed_form_limit_deg": self.closed_form_limit_deg,
                "closed_form_limit_reason": self.closed_form_limit_reason,
                "equilibrium_deck_immersion_deg": self.equilibrium_deck_immersion_deg(),
                "angle_of_loll_deg": self.angle_of_loll_deg,
                "loll_inside_closed_form_range": self.loll_inside_closed_form_range,
            }
        )
        return data


def gz_closed_form(beam: float, draft: float, kg: float, heel_deg: float) -> float:
    """Convenience wrapper around equation (1) for a unit-length section.

    ``beam`` is the section beam; the length cancels from ``BM`` so only the
    section geometry enters.
    """

    beam = _positive(beam, "beam")
    draft = _positive(draft, "draft")
    kg = _finite(kg, "kg")
    bm = beam**2 / (12.0 * draft)
    gm = 0.5 * draft + bm - kg
    phi = math.radians(_finite(heel_deg, "heel_deg"))
    return math.sin(phi) * (gm + 0.5 * bm * math.tan(phi) ** 2)
