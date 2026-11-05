"""Large-angle stability (`GZ` curve) of a wall-sided rectangular barge.

Why this module exists
----------------------
``tensorfem.marine_hydrostatics`` deliberately stops at |heel| <= 10 deg and
computes the righting arm from the small-angle (metacentric) relation
``GZ = GM sin(phi)``.  Its own documentation lists "large-angle GZ curve /
cross-curves" as an unsupported capability.  This module supplies that missing
capability for the geometry where an *exact* answer exists, so that the
capability can be verified to machine precision rather than calibrated against
itself.

Theory
------
For a prismatic barge of beam ``B``, moulded depth ``D`` and a symmetric
loading (no trim, longitudinal centre of gravity on the section centreline) the
heeled displacement is a purely two-dimensional problem: the immersed part of
the transverse section is the rectangle

    R = [-B/2, B/2] x [0, D]

cut by the waterline plane.  In body axes (x toward starboard, z up from the
keel) a heel of ``phi`` about the longitudinal axis puts the waterline at

    -x sin(phi) + z cos(phi) = d,

so the immersed set is ``{r : n . r <= d}`` with ``n = (-sin phi, cos phi)``.
Constant displacement (``V = L B T`` at every heel, which is exact for a
symmetric prismatic hull) fixes ``d`` through

    Area(clipped section) = B * T.

The section area and its centroid are then evaluated by exact convex-polygon
clipping and the shoelace centroid formula - no quadrature, no mesh.  The
righting arm is the horizontal (world) offset between the buoyancy centroid
``B = (xB, zB)`` and the centre of gravity ``G = (0, KG)``:

    GZ(phi) = xB cos(phi) + (zB - KG) sin(phi).

Wall-sided closed form (the verification anchor)
------------------------------------------------
As long as the waterline stays off the deck and the keel edge stays wetted
("wall-sided" range) the exact result reduces to the classical closed form

    GZ(phi) = sin(phi) * (GM + BM tan^2(phi) / 2),
    GM = KB + BM - KG,  KB = T/2,  BM = B^2 / (12 T),

which is the wall-sided stability formula of Biran, *Ship Hydrostatics and
Stability* (Butterworth-Heinemann, 2003), Sec. 4.9, and of Lewis (ed.),
*Principles of Naval Architecture* Vol. I (SNAME, 1988), Sec. 2.6.  The
validity band follows from the waterline height at the section centre
(``h = T`` in that range):

    tan(phi) < 2 T / B          (keel edge emerges  -> bilge-emergence branch)
    tan(phi) <= 2 (D - T) / B   (low deck edge immerses -> deck-immersion branch)

Outside the wall-sided band the geometry solver keeps working (it is exact for
any heel in [0, 90] deg, including deck immersion and bilge emergence); only the
closed form stops being valid, and :func:`wall_sided_validity_deg` reports the
boundary.

Scope and honest limitations
----------------------------
Supported: prismatic rectangular sections, free heel with fixed displacement
and zero trim, heels 0..90 deg, arbitrary ``KG``, cross-curves (``KN`` arms) and
dynamic-stability areas.

Not supported: arbitrary hull forms, trim / free-trim coupling, free-surface
correction, deck immersion *with* watertight-closure change (once the deck
submerges the real vessel would flood; the geometry below keeps a closed box so
the branch is a mathematical continuation, not a survivability prediction),
damage stability, seakeeping, and any classification-society regulation.

All quantities are SI (m, kg, s, N).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch  # noqa: F401  (kept for package-wide dtype/device consistency)

SCHEMA = "tensorfem.marine-large-angle-gz.v1"
MAX_HEEL_DEG = 90.0
DEFAULT_ANGLES_DEG = tuple(float(a) for a in range(0, 91, 5))

__all__ = [
    "SCHEMA",
    "MAX_HEEL_DEG",
    "DEFAULT_ANGLES_DEG",
    "HeeledSection",
    "GZCurve",
    "section_at_heel",
    "wall_sided_gz",
    "wall_sided_validity_deg",
    "wall_sided_dynamic_area",
    "gz_curve",
    "cross_curves",
    "dynamic_stability",
    "analyze_large_angle_case",
    "write_gz_report",
]


# ----------------------------------------------------------------------
# exact section geometry
# ----------------------------------------------------------------------
def _rectangle(beam: float, depth: float) -> list[tuple[float, float]]:
    """Section polygon, counter-clockwise, x centred on the centreline, z from keel."""
    half = 0.5 * beam
    return [(-half, 0.0), (half, 0.0), (half, depth), (-half, depth)]


def _clip_below(
    polygon: Sequence[tuple[float, float]],
    nx: float,
    nz: float,
    offset: float,
) -> list[tuple[float, float]]:
    """Sutherland-Hodgman clip of ``polygon`` against ``{p : nx*px + nz*pz <= offset}``."""
    out: list[tuple[float, float]] = []
    count = len(polygon)
    for index in range(count):
        p = polygon[index]
        q = polygon[(index + 1) % count]
        fp = nx * p[0] + nz * p[1] - offset
        fq = nx * q[0] + nz * q[1] - offset
        if fp <= 0.0:
            out.append(p)
        if (fp < 0.0 < fq) or (fq < 0.0 < fp):
            ratio = fp / (fp - fq)
            out.append((p[0] + ratio * (q[0] - p[0]), p[1] + ratio * (q[1] - p[1])))
    return out


def _area_centroid(polygon: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    """Signed area and centroid of a simple polygon (shoelace / section formula)."""
    if len(polygon) < 3:
        return 0.0, 0.0, 0.0
    twice_area = 0.0
    cx = 0.0
    cz = 0.0
    count = len(polygon)
    for index in range(count):
        x0, z0 = polygon[index]
        x1, z1 = polygon[(index + 1) % count]
        cross = x0 * z1 - x1 * z0
        twice_area += cross
        cx += (x0 + x1) * cross
        cz += (z0 + z1) * cross
    area = 0.5 * twice_area
    if abs(area) < 1e-300:
        return 0.0, 0.0, 0.0
    return area, cx / (3.0 * twice_area), cz / (3.0 * twice_area)


@dataclass(frozen=True)
class HeeledSection:
    """Exact immersed section at one heel angle."""

    heel_deg: float
    immersed_area: float
    xb: float
    zb: float
    waterline_offset: float
    waterline_height_centre: float
    bilge_emerged: bool
    deck_immersed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def section_at_heel(
    beam: float,
    depth: float,
    immersed_area: float,
    heel_deg: float,
    *,
    max_iterations: int = 200,
) -> HeeledSection:
    """Exact immersed section carrying ``immersed_area`` at ``heel_deg``.

    The waterline offset ``d`` is the monotone root of
    ``Area(d) = immersed_area``; it is bracketed on the corner extrema of
    ``n . r`` and refined by bisection to machine precision.
    """
    for label, value, positive in (
        ("beam", beam, True),
        ("depth", depth, True),
        ("immersed_area", immersed_area, True),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number")
        if not math.isfinite(float(value)) or (positive and float(value) <= 0.0):
            raise ValueError(f"{label} must be finite{' and positive' if positive else ''}")
    if isinstance(heel_deg, bool) or not isinstance(heel_deg, (int, float)):
        raise ValueError("heel_deg must be a number")
    if not math.isfinite(float(heel_deg)) or abs(float(heel_deg)) > MAX_HEEL_DEG:
        raise ValueError(f"heel_deg must lie in [-{MAX_HEEL_DEG:g}, {MAX_HEEL_DEG:g}]")
    if immersed_area >= beam * depth:
        raise ValueError("immersed_area must be smaller than the full section area")

    phi = math.radians(float(heel_deg))
    nx, nz = -math.sin(phi), math.cos(phi)
    polygon = _rectangle(beam, depth)

    projections = [nx * x + nz * z for x, z in polygon]
    low, high = min(projections), max(projections)

    def area_at(offset: float) -> float:
        return abs(_area_centroid(_clip_below(polygon, nx, nz, offset))[0])

    if area_at(high) < immersed_area:
        raise ValueError("immersed_area is not reachable at this heel angle")
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        if mid <= low or mid >= high:
            break
        if area_at(mid) < immersed_area:
            low = mid
        else:
            high = mid
    offset = 0.5 * (low + high)
    clipped = _clip_below(polygon, nx, nz, offset)
    area, xb, zb = _area_centroid(clipped)
    area = abs(area)

    # Waterline height measured along the section: z = (d + x sin phi) / cos phi,
    # which is linear in x, so the centre/high/low values follow from the normal.
    if abs(nz) > 0.0:
        h_centre = offset / nz
    else:  # pragma: no cover - guarded by MAX_HEEL_DEG but keep the branch safe
        h_centre = float("nan")
    tangent = math.tan(phi) if abs(math.cos(phi)) > 1e-15 else float("inf")
    if math.isfinite(tangent):
        h_high = h_centre + 0.5 * beam * tangent
        h_low = h_centre - 0.5 * beam * tangent
    else:  # exact 90 deg: the waterline is vertical, neither side is wetted out
        h_high = h_low = float("nan")
    # Flags are geometric diagnostics only; a small relative tolerance keeps the
    # knife-edge case (waterline exactly through a corner at the wall-sided
    # limit) from flipping on floating-point noise.
    flag_tol = 1e-12 * max(beam, depth, 1.0)
    bilge_emerged = math.isfinite(tangent) and h_low < -flag_tol
    deck_immersed = math.isfinite(tangent) and h_high > depth + flag_tol

    return HeeledSection(
        heel_deg=float(heel_deg),
        immersed_area=area,
        xb=xb,
        zb=zb,
        waterline_offset=offset,
        waterline_height_centre=h_centre,
        bilge_emerged=bool(bilge_emerged),
        deck_immersed=bool(deck_immersed),
    )


# ----------------------------------------------------------------------
# closed-form anchors
# ----------------------------------------------------------------------
def _check_positive_finite(label: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def wall_sided_gz(gm: float, bm: float, heel_deg: float) -> float:
    """Wall-sided closed form ``GZ = sin(phi) (GM + BM tan^2(phi)/2)``."""
    for label, value in (("gm", gm), ("bm", bm)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be a number")
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} must be finite")
    if abs(float(heel_deg)) > MAX_HEEL_DEG:
        raise ValueError("heel_deg outside [0, 90]")
    phi = math.radians(float(heel_deg))
    return math.sin(phi) * (float(gm) + 0.5 * float(bm) * math.tan(phi) ** 2)


def wall_sided_dynamic_area(gm: float, bm: float, heel_deg: float) -> float:
    """Exact ``int_0^phi GZ dphi`` of the wall-sided form.

    ``int sin = 1 - cos`` and ``int tan^2 sin = sec + cos - 2``, so

        A(phi) = GM (1 - cos phi) + BM (sec phi + cos phi - 2) / 2.
    """
    phi = math.radians(float(heel_deg))
    c = math.cos(phi)
    return float(gm) * (1.0 - c) + 0.5 * float(bm) * (1.0 / c + c - 2.0)


def wall_sided_validity_deg(beam: float, depth: float, draft: float) -> tuple[float, float]:
    """``(bilge_limit, deck_limit)`` closed-form heel limits, in degrees.

    ``tan phi_bilge = 2 T / B`` (keel edge emerges) and
    ``tan phi_deck = 2 (D - T) / B`` (low deck edge would immerse), both derived
    under the wall-sided assumption ``h = T`` (constant waterline height at the
    section centre).  The closed form therefore holds for
    ``0 <= phi <= min(phi_bilge, phi_deck)``; both limits are clamped to 90 deg.

    Caveat, verified in the tests: only the binding (smaller) limit is
    physical.  When ``D > 2 T`` the keel edge dries out first and the waterline
    height then drops *below* ``T``, so the actual deck-immersion angle is
    larger than ``phi_deck``.  Use the ``deck_immersed`` flags on
    :class:`HeeledSection` / :class:`GZCurve` for the true event.
    """
    beam = _check_positive_finite("beam", beam)
    depth = _check_positive_finite("depth", depth)
    draft = _check_positive_finite("draft", draft)
    if draft >= depth:
        raise ValueError("draft must be smaller than depth")
    bilge = math.degrees(math.atan2(2.0 * draft, beam))
    deck = math.degrees(math.atan2(2.0 * (depth - draft), beam))
    return min(bilge, MAX_HEEL_DEG), min(deck, MAX_HEEL_DEG)


# ----------------------------------------------------------------------
# curve building
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class GZCurve:
    """Large-angle stability result for one loading condition."""

    beam: float
    depth: float
    draft: float
    kg: float
    length: float
    fluid_density: float
    gravity: float
    displacement_volume: float
    displacement_mass: float
    kb: float
    bm: float
    gm: float
    angles_deg: tuple[float, ...]
    gz: tuple[float, ...]
    kn: tuple[float, ...]
    xb: tuple[float, ...]
    zb: tuple[float, ...]
    section_area: tuple[float, ...]
    bilge_emerged: tuple[bool, ...]
    deck_immersed: tuple[bool, ...]
    wall_sided_limit_deg: float
    max_gz: float
    angle_of_max_gz_deg: float
    dynamic_stability_area: float
    schema: str = SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_angles(angles_deg: Sequence[float]) -> tuple[float, ...]:
    if not isinstance(angles_deg, Sequence) or isinstance(angles_deg, (str, bytes)):
        raise ValueError("angles_deg must be a sequence of numbers")
    values = []
    for index, angle in enumerate(angles_deg):
        if isinstance(angle, bool) or not isinstance(angle, (int, float)):
            raise ValueError(f"angles_deg[{index}] must be a number")
        if not math.isfinite(float(angle)) or abs(float(angle)) > MAX_HEEL_DEG:
            raise ValueError(f"angles_deg[{index}] must lie in [-90, 90]")
        values.append(float(angle))
    if len(values) < 2:
        raise ValueError("at least two heel angles are required")
    return tuple(values)


def dynamic_stability(angles_deg: Sequence[float], gz: Sequence[float]) -> float:
    """Cumulative ``int GZ dphi`` over the tabulated curve (composite Simpson/trapezoid)."""
    angles = tuple(float(a) for a in angles_deg)
    arms = tuple(float(g) for g in gz)
    if len(angles) != len(arms) or len(angles) < 2:
        raise ValueError("angles and GZ must have equal length >= 2")
    total = 0.0
    for index in range(len(angles) - 1):
        a0, a1 = math.radians(angles[index]), math.radians(angles[index + 1])
        total += 0.5 * (arms[index] + arms[index + 1]) * (a1 - a0)
    return total


def gz_curve(
    beam: float,
    depth: float,
    draft: float,
    kg: float,
    angles_deg: Sequence[float] = DEFAULT_ANGLES_DEG,
    *,
    length: float | None = None,
    fluid_density: float = 1025.0,
    gravity: float = 9.80665,
) -> GZCurve:
    """Exact large-angle ``GZ`` curve of a prismatic rectangular barge."""
    beam = _check_positive_finite("beam", beam)
    depth = _check_positive_finite("depth", depth)
    draft = _check_positive_finite("draft", draft)
    density = _check_positive_finite("fluid_density", fluid_density)
    grav = _check_positive_finite("gravity", gravity)
    if isinstance(kg, bool) or not isinstance(kg, (int, float)) or not math.isfinite(float(kg)):
        raise ValueError("kg must be a finite number")
    if draft >= depth:
        raise ValueError("draft must be smaller than depth")
    section_length = 1.0 if length is None else _check_positive_finite("length", length)

    angles = _validate_angles(angles_deg)
    area_target = beam * draft
    kb = 0.5 * draft
    bm = beam**2 / (12.0 * draft)
    gm = kb + bm - float(kg)
    bilge_limit, deck_limit = wall_sided_validity_deg(beam, depth, draft)
    wall_limit = min(bilge_limit, deck_limit)
    volume = section_length * area_target
    mass = density * volume

    gz_values: list[float] = []
    kn_values: list[float] = []
    xb_values: list[float] = []
    zb_values: list[float] = []
    areas: list[float] = []
    bilge_flags: list[bool] = []
    deck_flags: list[bool] = []
    for angle in angles:
        sign = 1.0 if angle >= 0.0 else -1.0
        section = section_at_heel(beam, depth, area_target, abs(angle))
        phi = math.radians(abs(angle))
        arm = section.xb * math.cos(phi) + (section.zb - float(kg)) * math.sin(phi)
        gz_values.append(sign * arm)
        kn_values.append(sign * (section.xb * math.cos(phi) + section.zb * math.sin(phi)))
        xb_values.append(sign * section.xb)
        zb_values.append(section.zb)
        areas.append(section.immersed_area)
        bilge_flags.append(section.bilge_emerged)
        deck_flags.append(section.deck_immersed)

    best = max(range(len(angles)), key=lambda i: gz_values[i])
    return GZCurve(
        beam=beam,
        depth=depth,
        draft=draft,
        kg=float(kg),
        length=section_length,
        fluid_density=density,
        gravity=grav,
        displacement_volume=volume,
        displacement_mass=mass,
        kb=kb,
        bm=bm,
        gm=gm,
        angles_deg=angles,
        gz=tuple(gz_values),
        kn=tuple(kn_values),
        xb=tuple(xb_values),
        zb=tuple(zb_values),
        section_area=tuple(areas),
        bilge_emerged=tuple(bilge_flags),
        deck_immersed=tuple(deck_flags),
        wall_sided_limit_deg=wall_limit,
        max_gz=gz_values[best],
        angle_of_max_gz_deg=angles[best],
        dynamic_stability_area=dynamic_stability(angles, gz_values),
    )


def cross_curves(
    beam: float,
    depth: float,
    draft: float,
    kg_values: Sequence[float],
    angles_deg: Sequence[float] = DEFAULT_ANGLES_DEG,
    **kwargs: Any,
) -> dict[str, Any]:
    """``KN`` arms and ``GZ = KN - KG sin(phi)`` for several centres of gravity.

    The ``KN`` arm depends on geometry and displacement only, so one geometric
    sweep yields a whole cross-curve family - the standard naval-architecture
    decomposition (``GZ = KN - KG sin phi``).
    """
    if not isinstance(kg_values, Sequence) or isinstance(kg_values, (str, bytes)) or not kg_values:
        raise ValueError("kg_values must be a non-empty sequence")
    reference = gz_curve(beam, depth, draft, float(kg_values[0]), angles_deg, **kwargs)
    angles = reference.angles_deg
    sines = tuple(math.sin(math.radians(angle)) for angle in angles)
    table = {}
    for kg in kg_values:
        arm = float(kg)
        table[f"{arm:g}"] = tuple(kn - arm * s for kn, s in zip(reference.kn, sines))
    return {
        "angles_deg": list(angles),
        "kn": list(reference.kn),
        "gz_by_kg": {key: list(value) for key, value in table.items()},
    }


def analyze_large_angle_case(
    case: Mapping[str, Any],
    angles_deg: Sequence[float] = DEFAULT_ANGLES_DEG,
) -> GZCurve:
    """Build a GZ curve from the shared marine-hydrostatics case schema.

    The case file supplies geometry, fluid and condition (``specified_draft`` or
    ``load_case``) exactly as ``tensorfem.marine_hydrostatics`` expects.  Because
    the sweep replaces the single heel angle, ``condition.heel_angle_deg`` must
    be ``0.0``; anything else is rejected rather than silently ignored.
    """
    from .marine_hydrostatics import analyze_box_barge

    if not isinstance(case, Mapping):
        raise ValueError("case must be an object")
    condition = case.get("condition")
    if isinstance(condition, Mapping):
        heel = condition.get("heel_angle_deg", 0.0)
        if isinstance(heel, bool) or not isinstance(heel, (int, float)):
            raise ValueError("condition.heel_angle_deg must be a number")
        if float(heel) != 0.0:
            raise ValueError(
                "large-angle sweep requires condition.heel_angle_deg = 0; "
                "the sweep supplies the heel angles"
            )
    upright = analyze_box_barge(case)
    return gz_curve(
        beam=upright.beam,
        depth=upright.depth,
        draft=upright.draft,
        kg=upright.kg,
        angles_deg=angles_deg,
        length=upright.length,
        fluid_density=upright.fluid_density,
        gravity=upright.gravity,
    )


def write_gz_report(result: GZCurve, path: str | Path) -> Path:
    """Write the GZ curve as deterministic, sorted JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = result.to_dict()
    payload["angles_deg"] = list(result.angles_deg)
    payload["gz"] = list(result.gz)
    payload["kn"] = list(result.kn)
    payload["xb"] = list(result.xb)
    payload["zb"] = list(result.zb)
    payload["section_area"] = list(result.section_area)
    payload["bilge_emerged"] = list(result.bilge_emerged)
    payload["deck_immersed"] = list(result.deck_immersed)
    output.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return output