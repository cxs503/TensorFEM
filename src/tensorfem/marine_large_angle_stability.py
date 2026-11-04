"""Exact large-angle intact stability of a wall-sided (rectangular) barge.

Why this module exists
----------------------
``tensorfem.marine_hydrostatics`` deliberately stops at ``|heel| <= 10 deg`` and
evaluates the righting arm as ``GZ = GM * sin(phi)`` there.  That model is the
small-angle limit only: it discards the second-order rise of the centre of
buoyancy that a wall-sided hull produces once the waterplane tilts.  For a
*rectangular* section, however, the intact righting arm has an exact closed form
over the whole upright range, the classical wall-sided formula

    GZ(phi) = sin(phi) * [ GM + 0.5 * BM * tan^2(phi) ],          (1)

with ``GM = KB + BM - KG``, ``KB = T/2``, ``BM = I_wl / nabla``,
``I_wl = L B^3 / 12`` and ``nabla = L B T`` (Biran & Lopez-Pulido, *Ship
Hydrostatics and Stability*, 2nd ed., wall-sided righting-lever identity; the
same identity is quoted in Derrett, *Ship Stability for Masters and Mates*).
Equation (1) follows from the standard lever decomposition

    GZ = y_B cos(phi) + (KB' - KG) sin(phi),
    y_B  = BM tan(phi),   KB' = T/2 + 0.5 BM tan^2(phi),           (2)

which is the form used here as the *independent* oracle: this module recovers
``y_B`` and ``KB'`` from polygon geometry (clip the section with the tilted
waterplane, take the shoelace area/centroid, and solve for the sinkage that keeps
displacement constant) and never evaluates equation (1) on that path.

Validity of equation (1) is bounded twice: the deck edge must stay dry
(``tan|phi| < 2 (D - T) / B``) **and** the keel must stay submerged
(``tan|phi| < 2 T / B``).  Whichever binds first ends the closed form.  Beyond
a keel emergence the section becomes triangular at the fixed draft and the
floating body sinks; :meth:`WallSidedBarge.gz_numerical` still returns the
correct arm there because it solves for the equilibrium waterplane, which is
exactly the regime where equation (1) must not be extrapolated.

Scope boundary: rectangular, wall-sided, intact, upright-to-deck-edge only.  No
deck-immersion/downflooding, cross-curves, free surface, damage or trim; this is
not a rules-compliance or approval tool.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

__all__ = [
    "SCHEMA",
    "SMALL_ANGLE_LIMIT_DEG",
    "WallSidedBarge",
    "gz_closed_form",
    "polygon_section",
]

SCHEMA = "tensorfem.marine-large-angle-stability.v1"
SMALL_ANGLE_LIMIT_DEG = 10.0
DEFAULT_DENSITY = 1025.0
DEFAULT_GRAVITY = 9.80665

#: Relative tolerance used by the polygon clipping predicates.
_CLIP_TOL = 1e-14


def _positive(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return value


def _finite(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
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