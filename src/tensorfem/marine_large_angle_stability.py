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
from dataclasses import dataclass

__all__ = ["ImmersedSection", "BoxBarge"]

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