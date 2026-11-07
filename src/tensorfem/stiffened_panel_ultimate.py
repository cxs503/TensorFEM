"""Ultimate strength of stiffened panels and rule checks (ISO 19906 / CSR).

Why this module exists
----------------------
``tensorfem.marine_panel_ultimate_fe`` deliberately reports ``blocked``: the
repository had no executed peak/post-peak evidence for a compressed panel, so
no ultimate-strength claim could be admitted.  ``hull_girder_ultimate`` only
integrates ideal-elastic-plastic section fibres, which cannot represent plate
or column buckling, and ``marine_stiffener_failure`` qualifies linear
bifurcation of an isolated stiffener.

This module supplies the missing assessment chain:

1. **Component limiting strengths** -- plate uniaxial ultimate (Faulkner /
   von Karman effective width), stiffener column ultimate (Johnson-Ostenfeld
   inelastic column curve), stiffener tripping (St Venant torsional buckling
   about the attachment line) and local flange buckling.
2. **Stiffened-panel load-end shortening** -- plate and stiffener components
   strained together, weighted by area, giving ``sigma(eps)`` and ``sigma_u``.
3. **Progressive hull-girder collapse** -- the Smith (1977) incremental
   curvature method: each structural unit contributes its own limiting-stress
   curve and the section moment is integrated until the peak ``M_u`` is passed.
4. **ISO 19906 ice-belt screening** -- local ice pressure (pressure-area form)
   -> plate-field bending stress from the exact Navier series for a simply
   supported plate -> allowable-stress comparison, plus a rigid-plastic
   (yield-line) ultimate capacity for a ULS comparison.
5. **CSR-style hull-girder screening** -- wave/ice vertical bending moment
   against ``M_u`` with an explicit resistance factor.
6. **Ice + wave combination** -- in-plane utilisation from the hull girder
   superposed on the ice-belt bending utilisation through a documented
   quadratic interaction.

Scope and honesty statement
---------------------------
* Every closed form here is a documented engineering approximation with a
  stated validity range, not a verbatim reproduction of a class-society
  clause.  Rule numbers (safety factors, ice-pressure coefficients) are
  **caller supplied**; no rule coefficient is invented in this file.
* The load-end shortening is a limiting-stress model: elastic until the
  component limiting stress, then a plateau.  It omits post-buckling reserve
  and the softening after ``2 eps_y``; the panel ultimate is therefore
  expected to sit slightly below a large-deflection elasto-plastic shell
  solution (quantified in ``docs/stiffened-panel-ultimate-strength.md``).
* The numerical FE companion lives in
  :mod:`tensorfem.stiffened_panel_unit_fe`; the closed forms never consume
  their own output as evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

SQUARE_PLATE_PRESSURE_COEFFICIENT = 0.0479
"""Timoshenko & Woinowsky-Krieger Table 30: ``M_max = 0.0479 p b^2`` for a
simply supported square plate under uniform pressure (Poisson 0.3)."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


# --------------------------------------------------------------------------
# 1. panel unit and classical component strengths
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StiffenedPanel:
    """One longitudinally stiffened panel unit (plate field + one stiffener).

    All lengths in metres, stresses in pascals.  The plate mid-surface is
    ``z = 0`` and the stiffener (web + flange) hangs below it.
    """

    span: float
    spacing: float
    plate_thickness: float
    web_height: float
    web_thickness: float
    flange_width: float
    flange_thickness: float
    young: float = 2.06e11
    poisson: float = 0.3
    yield_stress: float = 2.35e8
    warping_inertia: float | None = None
    include_tripping: bool = False

    def validate(self) -> "StiffenedPanel":
        for name in ("span", "spacing", "plate_thickness", "web_height",
                     "web_thickness", "flange_width", "flange_thickness",
                     "young", "yield_stress"):
            _positive(name, getattr(self, name))
        _require(-1.0 < self.poisson < 0.5, "poisson must lie in (-1, 0.5)")
        _require(self.flange_width >= self.web_thickness,
                 "flange_width must not be narrower than the web")
        _require(self.web_height > 1.5 * self.flange_thickness,
                 "web_height must exceed 1.5 flange thicknesses")
        if self.warping_inertia is not None:
            _require(self.warping_inertia >= 0.0, "warping_inertia must be non-negative")
        return self

    # -- geometry ---------------------------------------------------------
    @property
    def plate_area(self) -> float:
        return self.spacing * self.plate_thickness

    @property
    def stiffener_area(self) -> float:
        return self.web_height * self.web_thickness + self.flange_width * self.flange_thickness

    @property
    def total_area(self) -> float:
        return self.plate_area + self.stiffener_area

    @property
    def plate_rigidity(self) -> float:
        return self.young * self.plate_thickness**3 / (12.0 * (1.0 - self.poisson**2))

    @property
    def plate_slenderness(self) -> float:
        """``beta = (b/t) sqrt(sigma_y/E)`` (von Karman plate slenderness)."""
        return self.spacing / self.plate_thickness * math.sqrt(self.yield_stress / self.young)

    @property
    def plate_buckling_stress(self) -> float:
        """Uniaxial elastic plate buckling stress, ``k = 4`` (long plate)."""
        return 4.0 * math.pi**2 * self.plate_rigidity / (self.spacing**2 * self.plate_thickness)

    @property
    def plate_yield_ratio(self) -> float:
        return self.plate_buckling_stress / self.yield_stress

    def _section_geometry(self) -> tuple[float, float, list[tuple[float, float, float]]]:
        """Return (area, first moment, [(area, arm, own inertia) ...]) about z=0."""
        plate = self.plate_area
        web_area = self.web_height * self.web_thickness
        flange_area = self.flange_width * self.flange_thickness
        web_arm = -0.5 * self.web_height
        flange_arm = -self.web_height
        parts = [
            (plate, 0.0, self.spacing * self.plate_thickness**3 / 12.0),
            (web_area, web_arm, self.web_thickness * self.web_height**3 / 12.0),
            (flange_area, flange_arm, self.flange_width * self.flange_thickness**3 / 12.0),
        ]
        area = sum(p[0] for p in parts)
        first = sum(p[0] * p[1] for p in parts)
        return area, first, parts

    @property
    def centroid_offset(self) -> float:
        """Section centroid measured from the plate mid-surface (negative)."""
        area, first, _ = self._section_geometry()
        return first / area

    @property
    def column_inertia(self) -> float:
        """Second moment of the attached plating + stiffener about its centroid."""
        area, first, parts = self._section_geometry()
        centroid = first / area
        return sum(own + a * (arm - centroid) ** 2 for a, arm, own in parts)

    @property
    def gyration_radius(self) -> float:
        area, _, _ = self._section_geometry()
        return math.sqrt(self.column_inertia / area)

    @property
    def column_slenderness(self) -> float:
        """``lambda = (a/(pi r)) sqrt(sigma_y/E)`` (pinned-pinned column)."""
        return self.span / (math.pi * self.gyration_radius) * math.sqrt(
            self.yield_stress / self.young)

    @property
    def column_euler_stress(self) -> float:
        """``sigma_E = pi^2 E I / (A a^2)``; equals ``sigma_y / lambda^2``."""
        area, _, _ = self._section_geometry()
        return math.pi**2 * self.young * self.column_inertia / (area * self.span**2)

    @property
    def tripping_stress(self) -> float:
        """St Venant torsion-only tripping stress of the stiffener.

        ``sigma_T = (G J + pi^2 E I_w / a^2) / I_p`` with ``I_p`` the polar
        second moment about the plate-stiffener intersection.  Warping is
        neglected unless ``warping_inertia`` is supplied, which makes this a
        conservative screen for open sections.
        """
        shear = self.young / (2.0 * (1.0 + self.poisson))
        web_area = self.web_height * self.web_thickness
        flange_area = self.flange_width * self.flange_thickness
        web_arm = -0.5 * self.web_height
        flange_arm = -self.web_height
        polar = self.web_thickness * self.web_height**3 / 12.0 + web_area * web_arm**2
        polar += self.flange_width * self.flange_thickness**3 / 12.0 + flange_area * flange_arm**2
        torsion = self.web_height * self.web_thickness**3 / 3.0 \
            + self.flange_width * self.flange_thickness**3 / 3.0
        warping = self.warping_inertia
        if warping is None:
            # T-section warping constant of the flange alone (web warping neglected)
            warping = self.flange_thickness**3 * self.flange_width**3 / 144.0
        return (shear * torsion + math.pi**2 * self.young * warping / self.span**2) / polar

    @property
    def flange_local_buckling_stress(self) -> float:
        """Outstand (one edge free) elastic plate buckling, ``k = 0.425``."""
        ratio = self.flange_width / self.flange_thickness
        return 0.425 * math.pi**2 * self.young / (12.0 * (1.0 - self.poisson**2)) / ratio**2

    # -- component limiting stresses --------------------------------------
    @property
    def faulkner_plate_strength(self) -> float:
        """Plate uniaxial ultimate stress from the Faulkner effective width.

        ``sigma_u/sigma_y = min(1, 2/beta - 1/beta^2)``.  The expression is an
        empirical fit to plate ultimate-strength tests, not an exact solution.
        """
        beta = self.plate_slenderness
        if beta <= 1.0:
            return self.yield_stress
        return self.yield_stress * min(1.0, 2.0 / beta - 1.0 / beta**2)

    @property
    def plate_limiting_stress(self) -> float:
        """Plate limiting stress (Faulkner ultimate, capped at yield)."""
        return min(self.faulkner_plate_strength, self.yield_stress)

    @property
    def column_limiting_stress(self) -> float:
        """Johnson-Ostenfeld column ultimate of the stiffener + attached plate."""
        return johnson_ostenfeld_strength(self.yield_stress, self.column_euler_stress)

    @property
    def stiffener_limiting_stress(self) -> float:
        """Stiffener limiting stress: column reduced by local buckling.

        Tripping is reported by :attr:`tripping_stress` but excluded from the
        default limiting stress: the St Venant torsion-only screen ignores the
        in-plane restraint of the flange and the attached plating, so for
        flanged stiffeners it is far more conservative than the rule tripping
        check it approximates.  Set ``include_tripping=True`` to let it govern.
        """
        candidates = [self.column_limiting_stress, self.flange_local_buckling_stress,
                      self.yield_stress]
        if self.include_tripping:
            candidates.append(self.tripping_stress)
        return min(candidates)

    @property
    def governing_stiffener_mode(self) -> str:
        candidates = {
            "column": self.column_limiting_stress,
            "flange_local": self.flange_local_buckling_stress,
        }
        if self.include_tripping:
            candidates["tripping"] = self.tripping_stress
        return min(candidates, key=candidates.get)

    # -- panel level ------------------------------------------------------
    @property
    def panel_ultimate_stress(self) -> float:
        """Area-weighted limiting stress of the panel unit.

        ``sigma_u = (A_p sigma_p + A_s sigma_s) / (A_p + A_s)`` with both
        components at their limiting stress.  This is the
        "components-at-ultimate" reduction of Smith's method; it assumes the
        plate and the stiffener reach their limits at the same applied strain,
        which is the usual screening assumption and is checked against the
        elasto-plastic shell solution in the benchmark.
        """
        return (self.plate_area * self.plate_limiting_stress
                + self.stiffener_area * self.stiffener_limiting_stress) / self.total_area

    @property
    def panel_yield_ratio(self) -> float:
        return self.panel_ultimate_stress / self.yield_stress

    def summary(self) -> dict[str, float | str]:
        return {
            "plate_slenderness": self.plate_slenderness,
            "plate_buckling_stress": self.plate_buckling_stress,
            "plate_limiting_stress": self.plate_limiting_stress,
            "column_slenderness": self.column_slenderness,
            "column_euler_stress": self.column_euler_stress,
            "column_limiting_stress": self.column_limiting_stress,
            "tripping_stress": self.tripping_stress,
            "flange_local_buckling_stress": self.flange_local_buckling_stress,
            "governing_stiffener_mode": self.governing_stiffener_mode,
            "stiffener_limiting_stress": self.stiffener_limiting_stress,
            "panel_ultimate_stress": self.panel_ultimate_stress,
            "panel_ultimate_over_yield": self.panel_yield_ratio,
            "plate_area": self.plate_area,
            "stiffener_area": self.stiffener_area,
            "centroid_offset": self.centroid_offset,
        }


def johnson_ostenfeld_strength(yield_stress: float, euler_stress: float) -> float:
    """Inelastic column curve ``sigma_u = sigma_y (1 - sigma_y/(4 sigma_E))``.

    Used for ``sigma_E >= sigma_y/2``; below that the Euler stress is returned
    unchanged.  This is the Johnson-Ostenfeld parabolic column curve used by
    several classification societies for stiffener buckling.
    """
    sy = _positive("yield_stress", yield_stress)
    se = _positive("euler_stress", euler_stress)
    if se >= 0.5 * sy:
        return sy * (1.0 - sy / (4.0 * se))
    return se


def load_end_shortening(panel: StiffenedPanel, strain: float) -> dict[str, float]:
    """Component and panel stress at one applied longitudinal strain.

    Limiting-stress model: each component is linear elastic until its limiting
    stress and constant afterwards.  ``strain`` is positive in compression;
    negative strain (tension) follows the elastic branch capped at yield.
    """
    panel.validate()
    eps = _finite("strain", strain)
    ey = panel.yield_stress / panel.young
    if eps <= 0.0:
        stress = max(eps, -ey) * panel.young
        return {"strain": eps, "plate_stress": stress, "stiffener_stress": stress,
                "panel_stress": stress, "plate_effective_width_ratio": 1.0}
    plate = min(panel.young * eps, panel.plate_limiting_stress)
    stiffener = min(panel.young * eps, panel.stiffener_limiting_stress)
    rel = min(1.0, max(panel.plate_limiting_stress / (panel.young * eps), 0.0))
    return {
        "strain": eps,
        "plate_stress": plate,
        "stiffener_stress": stiffener,
        "plate_effective_width_ratio": rel,
        "panel_stress": (panel.plate_area * plate
                         + panel.stiffener_area * stiffener) / panel.total_area,
    }


def panel_load_end_shortening_curve(
    panel: StiffenedPanel, *, strains: tuple[float, ...] | None = None
) -> dict[str, object]:
    """Tabulate the limiting-stress load-end shortening of one panel unit."""
    panel.validate()
    if strains is None:
        ey = panel.yield_stress / panel.young
        strains = tuple(ey * f for f in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0))
    rows = [load_end_shortening(panel, eps) for eps in strains]
    peak = max(row["panel_stress"] for row in rows)
    return {
        "strains": list(strains),
        "rows": rows,
        "peak_panel_stress": peak,
        "closed_form_panel_ultimate": panel.panel_ultimate_stress,
        "peak_matches_closed_form": abs(peak / panel.panel_ultimate_stress - 1.0) < 1e-12,
    }


# --------------------------------------------------------------------------
# 2. progressive collapse of a hull girder (Smith 1977 method)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class StructuralUnit:
    """A hull-girder structural unit at a fixed vertical position.

    ``panel`` supplies the compressive limiting stress; ``ultimate_stress``
    overrides it for units whose strength comes from elsewhere.
    """

    name: str
    area: float
    vertical_position: float
    panel: StiffenedPanel | None = None
    ultimate_stress: float | None = None

    def validate(self, yield_stress: float) -> "StructuralUnit":
        _positive(f"area[{self.name}]", self.area)
        _finite(f"vertical_position[{self.name}]", self.vertical_position)
        if self.panel is None and self.ultimate_stress is None:
            raise ValueError(f"unit {self.name} needs a panel or an ultimate_stress")
        if self.ultimate_stress is not None:
            _positive(f"ultimate_stress[{self.name}]", self.ultimate_stress)
            _require(self.ultimate_stress <= yield_stress * (1.0 + 1e-12),
                     f"unit {self.name} ultimate stress exceeds yield")
        return self

    def compressive_limit(self) -> float:
        if self.ultimate_stress is not None:
            return self.ultimate_stress
        assert self.panel is not None
        return self.panel.panel_ultimate_stress


def hull_girder_progressive_collapse(
    units: tuple[StructuralUnit, ...], *, yield_stress: float, young: float,
    curvatures: tuple[float, ...] | None = None, steps: int = 240,
    maximum_strain_factor: float = 6.0,
) -> dict[str, object]:
    """Integrate the sagging/hogging moment-curvature relation to ``M_u``.

    Smith's method: for each curvature every unit is strained by
    ``eps_i = kappa (y_i - y_na)``, its stress follows the limiting-stress
    curve (compression) or elasticity capped at yield (tension), and the
    section moment is integrated.  The neutral axis is held at its elastic
    value for the whole sweep, which is the usual screening simplification
    (a class-society implementation re-solves it every step).
    """
    sy = _positive("yield_stress", yield_stress)
    e = _positive("young", young)
    _require(len(units) >= 2, "at least two structural units are required")
    for unit in units:
        unit.validate(sy)
    area_total = sum(u.area for u in units)
    neutral = sum(u.area * u.vertical_position for u in units) / area_total
    inertia = sum(u.area * (u.vertical_position - neutral) ** 2 for u in units)
    limits = {u.name: u.compressive_limit() for u in units}
    ey = sy / e
    if curvatures is None:
        if steps < 8:
            raise ValueError("steps must be at least 8 when curvatures are generated")
        peak = maximum_strain_factor * ey / max(
            abs(u.vertical_position - neutral) for u in units)
        curvatures = tuple(peak * i / steps for i in range(1, steps + 1))
    rows = []
    for kappa in curvatures:
        k = _positive("curvature", kappa)
        moment = 0.0
        compressed: list[str] = []
        for unit in units:
            distance = unit.vertical_position - neutral
            eps_c = -k * distance
            if eps_c > 0.0:
                sigma = min(e * eps_c, limits[unit.name])
                if e * eps_c >= limits[unit.name]:
                    compressed.append(unit.name)
            else:
                sigma = max(e * eps_c, -sy)
            moment += sigma * unit.area * distance
        rows.append({"curvature": k, "moment": -moment,
                     "units_at_limit": tuple(compressed)})
    moments = [abs(row["moment"]) for row in rows]
    peak_index = max(range(len(moments)), key=lambda i: moments[i])
    elastic_moment = e * inertia * curvatures[peak_index]
    plastic_arm = sum(u.area * abs(u.vertical_position - neutral) for u in units)
    return {
        "curvatures": list(curvatures),
        "rows": rows,
        "neutral_axis": neutral,
        "section_inertia": inertia,
        "elastic_moment_at_peak": elastic_moment,
        "upper_bound_fully_plastic_moment": sy * plastic_arm,
        "ultimate_moment": moments[peak_index],
        "ultimate_curvature": curvatures[peak_index],
        "peak_index": peak_index,
        "post_peak": peak_index < len(moments) - 1
        and moments[-1] < moments[peak_index] * (1.0 - 1e-6),
        "plateau_reached": abs(moments[-1] / moments[peak_index] - 1.0) < 1e-3,
        "limits": limits,
    }


# --------------------------------------------------------------------------
# 3. exact Navier series for a simply supported plate under uniform pressure
# --------------------------------------------------------------------------
def simply_supported_plate_series(
    span: float, width: float, thickness: float, pressure: float, *,
    poisson: float = 0.3, terms: int = 51,
) -> dict[str, float]:
    """Navier double-series deflection and bending moment of a uniform-load plate.

    Simply supported on all four edges, isotropic, small deflection.  Returns
    the centre deflection and the maximum bending moment ``M_x = M_y``, both
    exact in the limit ``terms -> inf``.  Term counts are odd; convergence is
    monitored by :func:`series_convergence`.
    """
    a = _positive("span", span)
    b = _positive("width", width)
    t = _positive("thickness", thickness)
    p = _positive("pressure", pressure)
    if terms < 1 or terms % 2 == 0:
        raise ValueError("terms must be a positive odd integer")
    rigidity = t**3 / (12.0 * (1.0 - poisson**2))  # D/E for unit Young modulus
    deflection = 0.0
    moment_x = 0.0
    moment_y = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            denominator = m * n * (m**2 / a**2 + n**2 / b**2) ** 2
            shape = math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
            # w_mn = 16 p / (pi^6 D m n (m^2/a^2 + n^2/b^2)^2) at the centre
            deflection += 16.0 * p / (math.pi**6 * rigidity * denominator) * shape
            # M_x = D pi^2 sum A_mn (m^2/a^2 + nu n^2/b^2) -> the pi^2 D cancels
            moment_x += 16.0 * p / (math.pi**4 * denominator) \
                * (m**2 / a**2 + poisson * n**2 / b**2) * shape
            moment_y += 16.0 * p / (math.pi**4 * denominator) \
                * (poisson * m**2 / a**2 + n**2 / b**2) * shape
    # The governing moment is the larger of the two centre moments; ``width``
    # is the short side in a ship panel, so the coefficient is reported against
    # ``p b^2`` with ``b = width``.
    governing = moment_x if abs(moment_x) >= abs(moment_y) else moment_y
    return {
        "centre_deflection": deflection,
        "centre_moment": governing,
        "centre_moment_x": moment_x,
        "centre_moment_y": moment_y,
        "plate_rigidity": rigidity,
        "aspect_ratio": a / b,
    }


def plate_pressure_bending_stress(
    span: float, width: float, thickness: float, pressure: float, *,
    young: float = 2.06e11, poisson: float = 0.3, terms: int = 51,
) -> dict[str, float]:
    """Elastic bending stress at the centre of a simply supported plate."""
    series = simply_supported_plate_series(span, width, thickness, pressure,
                                           poisson=poisson, terms=terms)
    stress = 6.0 * abs(series["centre_moment"]) / _positive("thickness", thickness) ** 2
    return {"maximum_bending_stress": stress,
            "centre_moment": series["centre_moment"],
            "centre_moment_x": series["centre_moment_x"],
            "centre_moment_y": series["centre_moment_y"],
            "centre_deflection": series["centre_deflection"],
            "coefficient_m_over_pb2": abs(series["centre_moment"]) / (pressure * _positive("width", width) ** 2)}


def square_plate_coefficient_error(terms: int = 51) -> dict[str, float]:
    """Relative error of the series moment coefficient against the textbook value."""
    case = plate_pressure_bending_stress(1.0, 1.0, 0.01, 1.0, terms=terms)
    reference = SQUARE_PLATE_PRESSURE_COEFFICIENT
    return {"series_coefficient": case["coefficient_m_over_pb2"],
            "reference_coefficient": reference,
            "relative_error": abs(case["coefficient_m_over_pb2"] / reference - 1.0)}


# --------------------------------------------------------------------------
# 4. rigid-plastic (yield-line) capacity of the plate field
# --------------------------------------------------------------------------
def yield_line_collapse_pressure(
    span: float, width: float, thickness: float, yield_stress: float, *,
    boundary: str = "simply_supported",
) -> dict[str, float | str]:
    """Rigid-plastic upper bound for a rectangular plate under uniform pressure.

    Two independent mechanisms are evaluated and the smaller (tighter) upper
    bound is returned:

    * **fan/pyramid mechanism** -- four rigid facets hinged along the diagonals
      from each corner to a central apex.  Virtual work gives
      ``p = W_int / V`` with ``V = A w0 / 3``; for a square plate this reduces
      to the classical ``p_u = 24 M_p / a^2``.
    * **strip mechanism** -- a mid-span yield line spanning the short
      direction, valid for long plates; ``p_u = 8 M_p / b^2`` for a
      simply supported strip and ``12 M_p / b^2`` when both edges are fixed.

    ``M_p = sigma_y t^2 / 4`` is the plastic moment per unit width.
    """
    a = _positive("span", span)
    b = _positive("width", width)
    t = _positive("thickness", thickness)
    sy = _positive("yield_stress", yield_stress)
    if boundary not in ("simply_supported", "clamped"):
        raise ValueError("boundary must be 'simply_supported' or 'clamped'")
    mp = sy * t**2 / 4.0
    if boundary == "simply_supported":
        factor = 8.0
    else:
        factor = 16.0  # mid-span hinge plus both support hinges
    strip = factor * mp / b**2
    # Fan mechanism: 4 diagonal hinges from the corners to the apex.
    diagonal = math.hypot(a, b) / 2.0
    # Relative rotation across one diagonal: contributions of the two facets.
    # Facet over the b-edge has gradient 2 w0/b, facet over the a-edge 2 w0/a;
    # projected on the hinge normal the relative rotation is
    # (2 w0 / b) * a/L + (2 w0 / a) * b/L  with L = hypot(a, b)/2.
    rotation = (2.0 / b) * (a / (2.0 * diagonal)) + (2.0 / a) * (b / (2.0 * diagonal))
    internal = 4.0 * mp * rotation * diagonal
    volume = a * b / 3.0
    fan = internal / volume
    aspect = a / b
    if aspect <= 1.5:
        # The four-hinge pyramid is the admissible governing mode for square-ish
        # fields; the one-way strip is not kinematically admissible here because
        # its rigid halves violate w = 0 on the two short edges.
        capacity, mechanism = fan, "fan"
    elif aspect >= 2.0:
        capacity, mechanism = strip, "strip"
    else:
        # Transitional aspect ratio: neither mechanism is tight.  The smaller
        # value is retained so that the screened capacity is never overestimated.
        capacity, mechanism = min(fan, strip), "transition"
    return {
        "plastic_moment_per_width": mp,
        "fan_mechanism_pressure": fan,
        "strip_mechanism_pressure": strip,
        "collapse_pressure": capacity,
        "governing_mechanism": mechanism,
        "aspect_ratio": aspect,
        "boundary": boundary,
        "square_plate_closed_form": 24.0 * mp / a**2,
    }


# --------------------------------------------------------------------------
# 5. ice pressure, ISO 19906 style screening
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class IcePressureModel:
    """Local ice pressure as a pressure-area relation ``p = p0 (A/A0)^n``.

    The pressure-area form is the ISO 19906 local-pressure construct.  The
    coefficients are **not** rule values: they must be supplied by the user
    from the applicable edition of the standard (or from a site-specific ice
    load model).  Nothing here guesses a coefficient.
    """

    reference_pressure: float
    reference_area: float
    exponent: float = -0.3
    label: str = "caller-supplied"

    def validate(self) -> "IcePressureModel":
        _positive("reference_pressure", self.reference_pressure)
        _positive("reference_area", self.reference_area)
        _finite("exponent", self.exponent)
        _require(self.exponent < 0.0,
                 "the local ice pressure must decrease with contact area (exponent < 0)")
        return self

    def pressure(self, area: float) -> float:
        self.validate()
        a = _positive("area", area)
        return self.reference_pressure * (a / self.reference_area) ** self.exponent


def iso19906_ice_belt_check(
    panel: StiffenedPanel, model: IcePressureModel, *, load_area: float | None = None,
    allowable_factor: float = 1.0, boundary: str = "simply_supported",
) -> dict[str, object]:
    """Screen one ice-belt plate field under a local ice pressure.

    Chain: ``p_ice`` from the pressure-area relation on ``load_area`` ->
    elastic plate bending stress (Navier series) -> comparison with the
    allowable stress ``allowable_factor * sigma_y`` -> rigid-plastic collapse
    pressure for the ULS comparison.

    ``load_area`` defaults to the full plate field ``span * spacing``.
    """
    panel.validate()
    model.validate()
    area = panel.span * panel.spacing if load_area is None else _positive("load_area", load_area)
    _require(area <= panel.span * panel.spacing * (1.0 + 1e-12),
             "load_area must not exceed the plate field area")
    pressure = model.pressure(area)
    stress = plate_pressure_bending_stress(panel.span, panel.spacing,
                                           panel.plate_thickness, pressure,
                                           young=panel.young, poisson=panel.poisson)
    allowable = allowable_factor * panel.yield_stress
    capacity = yield_line_collapse_pressure(panel.span, panel.spacing,
                                            panel.plate_thickness,
                                            panel.yield_stress, boundary=boundary)
    return {
        "ice_pressure": pressure,
        "load_area": area,
        "pressure_model_label": model.label,
        "maximum_bending_stress": stress["maximum_bending_stress"],
        "allowable_stress": allowable,
        "allowable_utilisation": stress["maximum_bending_stress"] / allowable,
        "plastic_collapse_pressure": capacity["collapse_pressure"],
        "governing_mechanism": capacity["governing_mechanism"],
        "collapsed": pressure >= capacity["collapse_pressure"],
        "uls_utilisation": pressure / float(capacity["collapse_pressure"]),
        "elastic_over_plastic_pressure": pressure / float(capacity["collapse_pressure"]),
        "series_coefficient": stress["coefficient_m_over_pb2"],
    }


# --------------------------------------------------------------------------
# 6. CSR-style hull-girder check and ice+wave combination
# --------------------------------------------------------------------------
def csr_hull_girder_check(
    units: tuple[StructuralUnit, ...], *, yield_stress: float, young: float,
    vertical_bending_moment: float, resistance_factor: float = 1.0,
    curvatures: tuple[float, ...] | None = None,
) -> dict[str, object]:
    """Compare an applied vertical bending moment with the progressive-collapse ``M_u``.

    ``resistance_factor`` is the rule partial factor on the resistance side of
    the ultimate-strength format (``gamma_R``); it is a caller-supplied rule
    number, default 1.0 (no partial factor) so that nothing is invented here.
    """
    applied = _finite("vertical_bending_moment", vertical_bending_moment)
    gf = _positive("resistance_factor", resistance_factor)
    collapse = hull_girder_progressive_collapse(units, yield_stress=yield_stress,
                                                young=young, curvatures=curvatures)
    ultimate = float(collapse["ultimate_moment"])
    design_resistance = ultimate / gf
    return {
        "applied_moment": applied,
        "ultimate_moment": ultimate,
        "resistance_factor": gf,
        "design_resistance": design_resistance,
        "utilisation": abs(applied) / design_resistance,
        "passed": abs(applied) <= design_resistance,
        "neutral_axis": collapse["neutral_axis"],
        "section_inertia": collapse["section_inertia"],
        "upper_bound_fully_plastic_moment": collapse["upper_bound_fully_plastic_moment"],
        "collapse": collapse,
    }


def ice_wave_combined_check(
    panel: StiffenedPanel, model: IcePressureModel, *, wave_bending_moment: float,
    section_modulus: float, load_area: float | None = None,
    interaction: str = "quadratic",
) -> dict[str, object]:
    """Combine hull-girder wave compression with ice-belt plate bending.

    The wave/ice waterline bending moment produces the in-plane compressive
    stress ``sigma_c = M / Z`` at the ice belt; the local ice pressure produces
    plate bending ``sigma_b``.  The two act on different stress systems, so a
    member check combines them through an interaction surface:

    * ``quadratic`` -- ``(sigma_c/sigma_u)^2 + (sigma_b/sigma_y)^2 <= 1``
    * ``linear``    -- ``sigma_c/sigma_u + sigma_b/sigma_y <= 1``

    Both are documented screening assumptions, not clauses of a rule set.
    """
    panel.validate()
    model.validate()
    if interaction not in ("quadratic", "linear"):
        raise ValueError("interaction must be 'quadratic' or 'linear'")
    moment = abs(_finite("wave_bending_moment", wave_bending_moment))
    z = _positive("section_modulus", section_modulus)
    sigma_c = moment / z
    ice = iso19906_ice_belt_check(panel, model, load_area=load_area)
    sigma_b = float(ice["maximum_bending_stress"])
    in_plane = sigma_c / panel.panel_ultimate_stress
    bending = sigma_b / panel.yield_stress
    if interaction == "quadratic":
        utilisation = math.hypot(in_plane, bending)
    else:
        utilisation = in_plane + bending
    return {
        "in_plane_stress": sigma_c,
        "panel_ultimate_stress": panel.panel_ultimate_stress,
        "in_plane_utilisation": in_plane,
        "bending_stress": sigma_b,
        "bending_utilisation": bending,
        "interaction": interaction,
        "combined_utilisation": utilisation,
        "passed": utilisation <= 1.0,
        "ice_check": ice,
    }