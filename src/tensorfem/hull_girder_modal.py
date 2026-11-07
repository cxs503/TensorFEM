"""Hull-girder dry/wet modal analysis with analytic anchors.

Why this module exists
----------------------
:mod:`tensorfem.modal` covers Euler--Bernoulli beams and
:mod:`tensorfem.timoshenko_beam_modal` covers shear-flexible beams, but neither
knows anything about a *ship* girder: an equivalent box girder floating in
water, whose measured (wet) bending frequencies sit far below the in-vacuo
(dry) ones because the water oscillating with the hull acts as added mass.
This module provides the three analytic anchors a hull-girder modal model has
to reproduce before its numbers may be believed:

1. **Shear-flexible (Timoshenko) bending frequencies.**  For mode ``n`` of a
   uniform simply supported girder the first-order Timoshenko correction of the
   Euler--Bernoulli frequency is

   .. math:: f_n^{T} = f_n^{E}\\,\\sqrt{1 - \\left(\\frac{n\\pi}{L}\\right)^2
             \\frac{EI}{\\kappa G A}} .

   The exact reference is the Hu--Timoshenko dispersion relation already
   implemented in :mod:`tensorfem.timoshenko_beam_modal`; the square-root form
   above is the first-order truncation of that quadratic and is accurate to
   ``O((I/A)^2 k^4)``, i.e. fractions of a percent for slender hull girders.

2. **Two-dimensional section added mass (Lewis conformal mapping).**  The
   added mass per unit length of a section oscillating vertically is computed
   exactly for the Lewis family of section shapes.  With the conformal map

   .. math:: z(\\theta) = M\\left(\\zeta + a_1 \\zeta^{-1}
             + a_3 \\zeta^{-3}\\right), \\qquad \\zeta = e^{i\\theta},

   the half-breadth is ``b = M(1 + a1 + a3)``, the half-draft is
   ``h = M(1 - a1 + a3)``, and the potential of the section moving with unit
   vertical velocity is available in closed form.  The added mass follows from

   .. math:: m_{33} = \\rho \\oint \\varphi\\, n_y \\, \\mathrm{d}l
             = \\rho \\int_0^{2\\pi} \\varphi(\\theta)\\, x'(\\theta)\\,
               \\mathrm{d}\\theta .

   The implementation is validated against the three exact limits of the
   family: the circle (``m33 = rho*pi*a^2``), the ellipse (``m33 = rho*pi*b^2``
   for vertical motion of an ellipse with half-breadth ``b``) and the flat
   plate (``b/h -> inf``, ``m33 -> rho*pi*b^2``).

3. **Wet/dry frequency ratio.**  If the added mass is distributed along the
   girder in proportion to the structural mass, the generalised eigenproblem
   ``K phi = omega^2 (M + M_a) phi`` has ``M + M_a = (1 + kappa) M`` and

   .. math:: f^{wet} / f^{dry} = 1/\\sqrt{1 + \\kappa}, \\qquad
             \\kappa = m_{33} / m_{struct} .

   A finite element model that assembles the added mass with the *same* shape
   functions therefore reproduces this ratio to machine precision; any
   deviation is an assembly error, which is what the ratio check detects.

Conventions and applicability limits
------------------------------------
* Two-dimensional strip theory, zero forward speed, infinite water depth,
  inviscid irrotational flow.
* The free surface is treated as a plane of symmetry (``dphi/dz = 0``, the
  rigid-lid / low-frequency limit used for hull-girder vibration practice).
  The immersed section and its mirror image form the closed Lewis contour, so
  the computed added mass is the added mass of the *doubled* section -- which
  is exactly the rigid-lid value of the half section.
* Lewis forms cannot represent a sharp-cornered box: the closed-contour area
  of the family is bounded by the flat plate/ellipse envelope, so the box is
  matched on half-breadth, draft and (as closely as possible) area.  The
  fitted section is slightly smoother than a box, so the reported added mass is
  a *lower* bound for the sharp section (see ``area_ratio`` in the Lewis
  fit result).
* Bending only: vertical bending about the transverse axis, transverse shear
  carried by the two side walls (``A_s = 2 t D``).  Torsion, warping and the
  three-dimensional reduction of the added mass at the ends are outside the
  model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .modal import ModalResult, solve_modes
from .timoshenko_beam_modal import (
    TimoshenkoBeamSection,
    assemble_timoshenko_beam,
    euler_bernoulli_simply_supported_angular_frequencies,
    exact_simply_supported_timoshenko_angular_frequencies,
    timoshenko_shape_functions,
)

__all__ = [
    "BoxGirderSection",
    "LewisSection",
    "lewis_fit",
    "lewis_heave_added_mass",
    "timoshenko_corrected_angular_frequencies",
    "exact_hull_girder_angular_frequencies",
    "timoshenko_correction_factor",
    "wet_dry_frequency_ratio",
    "added_mass_matrix",
    "beam_simply_supported_modes",
    "wet_simply_supported_modes",
    "variable_added_mass_modes",
    "lewis_heave_coefficient",
    "ballast_mass_per_length",
    "floating_mass_per_length",
    "gci",
    "GAUSS_POINTS_4",
    "GAUSS_WEIGHTS_4",
]

#: Four-point Gauss--Legendre rule on ``[-1, 1]`` (exact to degree 7).
GAUSS_POINTS_4 = (-0.8611363115940526, -0.3399810435848563,
                  0.3399810435848563, 0.8611363115940526)
GAUSS_WEIGHTS_4 = (0.3478548451374538, 0.6521451548625461,
                   0.6521451548625461, 0.3478548451374538)

#: Standard gravity [m/s^2] (only used for documentation of the rigid-lid limit).
GRAVITY = 9.80665


@dataclass(frozen=True)
class BoxGirderSection:
    """Equivalent uniform box girder of a ship hull.

    ``length`` x ``breadth`` x ``depth`` outer dimensions, constant wall
    ``wall``; :attr:`draft` is the immersed depth used by the added-mass model
    and defaults to ``0.625 * depth`` (a typical hull form ratio).
    """

    length: float
    breadth: float
    depth: float
    wall: float
    young: float = 2.06e11
    density: float = 7850.0
    poisson: float = 0.30
    draft: float | None = None
    water_density: float = 1025.0

    # -- geometry ---------------------------------------------------------
    @property
    def area(self) -> float:
        """Material cross-sectional area of the thin-walled box [m^2]."""
        inner_b = self.breadth - 2.0 * self.wall
        inner_d = self.depth - 2.0 * self.wall
        return self.breadth * self.depth - inner_b * inner_d

    @property
    def inertia(self) -> float:
        """Second moment of area about the transverse (horizontal) axis [m^4]."""
        inner_b = self.breadth - 2.0 * self.wall
        inner_d = self.depth - 2.0 * self.wall
        return (self.breadth * self.depth**3 - inner_b * inner_d**3) / 12.0

    @property
    def shear_area(self) -> float:
        """Area of the two side walls carrying vertical shear, ``2 t D`` [m^2]."""
        return 2.0 * self.wall * self.depth

    @property
    def effective_shear_coefficient(self) -> float:
        """``kappa`` such that ``kappa * G * A = G * A_s`` for vertical shear."""
        return self.shear_area / self.area

    @property
    def shear_modulus(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def mass_per_length(self) -> float:
        """Structural mass per unit length [kg/m]."""
        return self.density * self.area

    @property
    def bending_stiffness(self) -> float:
        return self.young * self.inertia

    @property
    def shear_stiffness(self) -> float:
        """``kappa * G * A`` for vertical bending [N]."""
        return self.shear_modulus * self.shear_area

    @property
    def kappa_shear_ratio(self) -> float:
        """``EI / (kappa G A)`` [m^2]: the length scale of the shear correction."""
        return self.bending_stiffness / self.shear_stiffness

    @property
    def section_draft(self) -> float:
        return 0.625 * self.depth if self.draft is None else self.draft

    @property
    def half_breadth(self) -> float:
        return 0.5 * self.breadth

    @property
    def slenderness(self) -> float:
        return self.length / self.depth

    @property
    def block_coefficient(self) -> float:
        """Material area / enclosing rectangle area (the girder's ``Cb``)."""
        return self.area / (self.breadth * self.depth)

    @property
    def displaced_mass_per_length(self) -> float:
        """Mass of displaced water per unit length [kg/m]."""
        return self.water_density * self.breadth * self.section_draft

    def validate(self) -> None:
        for name in ("length", "breadth", "depth", "wall", "young", "density"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= self.poisson < 0.5:
            raise ValueError("poisson ratio must satisfy 0 <= nu < 0.5")
        if 2.0 * self.wall >= min(self.breadth, self.depth):
            raise ValueError("wall thickness must be less than half the smallest dimension")
        if not 0.0 < self.section_draft <= self.depth:
            raise ValueError("draft must satisfy 0 < draft <= depth")

    def to_timoshenko_section(self) -> TimoshenkoBeamSection:
        """Map onto the shear-flexible beam used by the finite element model.

        The element only ever uses the four products ``EI``, ``kappa G A``,
        ``rho A`` and ``rho I``.  A solid rectangle ``w x h`` has
        ``I/A = h^2/12``, so choosing

            h_eq = sqrt(12 I / A),   w_eq = A / h_eq,   kappa = A_s / A

        with the *same* ``E``, ``rho`` and ``nu`` reproduces all four exactly:
        ``E w h^3/12 = EI``, ``rho w h = rho A``, ``rho w h^3/12 = rho I`` and
        ``kappa G w h = G A_s``.  The equivalent rectangle is therefore a
        faithful carrier for the box girder's beam mechanics, not an
        approximation of its shape.
        """
        self.validate()
        height = math.sqrt(12.0 * self.inertia / self.area)
        width = self.area / height
        return TimoshenkoBeamSection(
            length=self.length,
            height=height,
            width=width,
            young=self.young,
            poisson=self.poisson,
            density=self.density,
            shear_coefficient=self.effective_shear_coefficient,
        )

    def geometry_audit(self) -> dict:
        """Independent verification of the closed-form section properties.

        Two paths that do not reuse the ``BD^3 - b_i d_i^3`` algebra are
        evaluated and compared with it:

        * **Green's theorem** on the outer and inner polygons
          (``A = (1/2) * sum(x_i y_j - x_j y_i)``, ``I = sum y^2 dA`` by the
          same edge sum) -- an independent formula path for the same two
          numbers;
        * the **thin-wall asymptote** ``A -> 2t(B+D)``,
          ``I -> t D^2 (3B + D)/6``, whose residual is ``O(t/D)`` and is therefore
          a check on the *physics* of the closed form rather than on its
          arithmetic.

        Every downstream frequency uses :attr:`area`/:attr:`inertia`, so this
        audit is the geometry gate that must pass before any modal number is
        reported.
        """
        self.validate()

        def polygon_properties(half_b: float, half_d: float) -> tuple[float, float]:
            """Area and second moment of area of an axis-aligned rectangle.

            Path that is independent of the ``B D^3 - b d^3`` algebra: the area
            from the shoelace sum and ``I = integral x y^2 dy`` along each edge
            (Green's theorem, ``I = int_int y^2 dA = oint x y^2 dy``).
            """
            corners = [(-half_b, -half_d), (half_b, -half_d), (half_b, half_d), (-half_b, half_d)]
            area = 0.0
            inertia = 0.0
            for i in range(4):
                x1, y1 = corners[i]
                x2, y2 = corners[(i + 1) % 4]
                area += 0.5 * (x1 * y2 - x2 * y1)
                if y2 == y1:
                    continue
                slope = (x2 - x1) / (y2 - y1)
                x_mid = 0.5 * (x1 + x2)
                y_mid = 0.5 * (y1 + y2)
                # integral of (x_mid + slope*(y - y_mid)) * y^2 dy over [y1, y2]
                inertia += x_mid * (y2**3 - y1**3) / 3.0 + slope * (
                    (y2**4 - y1**4) / 4.0 - y_mid * (y2**3 - y1**3) / 3.0
                )
            return abs(area), abs(inertia)

        outer_a, outer_i = polygon_properties(0.5 * self.breadth, 0.5 * self.depth)
        inner_a, inner_i = polygon_properties(
            0.5 * self.breadth - self.wall, 0.5 * self.depth - self.wall
        )
        area_poly = outer_a - inner_a
        inertia_poly = outer_i - inner_i
        thin_a = 2.0 * self.wall * (self.breadth + self.depth)
        thin_i = self.wall * self.depth**2 * (3.0 * self.breadth + self.depth) / 6.0
        return {
            "area_closed_form": self.area,
            "area_green": area_poly,
            "area_rel_error": abs(area_poly - self.area) / self.area,
            "inertia_closed_form": self.inertia,
            "inertia_green": inertia_poly,
            "inertia_rel_error": abs(inertia_poly - self.inertia) / self.inertia,
            "area_thin_wall": thin_a,
            "area_thin_wall_rel_dev": abs(thin_a - self.area) / self.area,
            "inertia_thin_wall": thin_i,
            "inertia_thin_wall_rel_dev": abs(thin_i - self.inertia) / self.inertia,
            "block_coefficient": self.block_coefficient,
            "slenderness": self.slenderness,
            "draft_over_depth": self.section_draft / self.depth,
        }


# ---------------------------------------------------------------------------
# Timoshenko correction
# ---------------------------------------------------------------------------
def timoshenko_correction_factor(section: BoxGirderSection, mode: int) -> float:
    """``sqrt(1 - (n pi / L)^2 EI / (kappa G A))`` for mode ``n`` (>= 1)."""
    if mode < 1:
        raise ValueError("mode must be >= 1")
    section.validate()
    k = mode * math.pi / section.length
    value = 1.0 - k**2 * section.kappa_shear_ratio
    if value <= 0.0:
        # The first-order correction is only meaningful while the shear term is
        # small; beyond this the square-root form collapses and the exact
        # Hu-Timoshenko root must be used.
        raise ValueError(
            f"mode {mode} is outside the validity of the first-order Timoshenko "
            "correction (1 - k^2 EI/(kappa G A) <= 0)"
        )
    return math.sqrt(value)


def timoshenko_corrected_angular_frequencies(
    section: BoxGirderSection, modes: int = 4, *, dtype=torch.float64
) -> torch.Tensor:
    """First-order Timoshenko frequencies ``f_n^T = f_n^E sqrt(1 - ...)``."""
    beam = section.to_timoshenko_section()
    eb = euler_bernoulli_simply_supported_angular_frequencies(beam, modes, dtype=dtype)
    return eb * torch.tensor(
        [timoshenko_correction_factor(section, n) for n in range(1, modes + 1)],
        dtype=dtype,
    )


# ---------------------------------------------------------------------------
# Lewis section added mass
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LewisSection:
    """Lewis-form section ``z = M(zeta + a1/zeta + a3/zeta^3)``.

    ``half_breadth`` and ``half_draft`` are the geometric half-breadth (``x``)
    and half-draft (``y``) of the closed contour; ``area_ratio`` reports the
    achieved closed-contour area over the area requested by the fit.
    """

    M: float
    a1: float
    a3: float
    half_breadth: float
    half_draft: float
    requested_area: float
    area_ratio: float

    def contour(self, theta: float) -> tuple[float, float]:
        """Point ``(x, y)`` and the tangent ``(x', y')`` at angle ``theta``."""
        c, s = math.cos(theta), math.sin(theta)
        c3, s3 = math.cos(3.0 * theta), math.sin(3.0 * theta)
        x = self.M * ((1.0 + self.a1) * c + self.a3 * c3)
        y = self.M * ((1.0 - self.a1) * s - self.a3 * s3)
        dx = self.M * (-(1.0 + self.a1) * s - 3.0 * self.a3 * s3)
        dy = self.M * ((1.0 - self.a1) * c - 3.0 * self.a3 * c3)
        return x, y, dx, dy

    def area(self, samples: int = 128) -> float:
        """Closed-contour area by the trapezoidal rule on ``(1/2) x dy - y dx``."""
        total = 0.0
        for i in range(samples):
            theta = 2.0 * math.pi * i / samples
            x, y, dx, dy = self.contour(theta)
            total += 0.5 * (x * dy - y * dx)
        return abs(total) * (2.0 * math.pi / samples)

    def potential(self, theta: float, velocity: float = 1.0) -> float:
        """Velocity potential on the contour for unit vertical body velocity.

        The conformal image of the unit-circle solution is
        ``F(zeta) = M (W_inf zeta + conj(W_inf) / zeta)`` with ``W_inf = i V``
        (a section moving in ``+y`` sees a stream ``-V`` in its own frame), so
        ``Re F = -2 M V sin(theta)``; adding the uniform translation ``V y``
        gives the potential of the moving section.  Only ``M`` enters ``Re F``:
        the shape parameters move the contour, not the far field.
        """
        _, y, _, _ = self.contour(theta)
        return velocity * y - 2.0 * velocity * self.M * math.sin(theta)

    def heave_added_mass(self, rho: float, samples: int = 256) -> float:
        """Added mass per unit length for vertical motion [kg/m]."""
        if rho <= 0.0:
            raise ValueError("rho must be positive")
        total = 0.0
        for i in range(samples):
            theta = 2.0 * math.pi * i / samples
            _, _, dx, _ = self.contour(theta)
            total += self.potential(theta, 1.0) * dx
        return rho * total * (2.0 * math.pi / samples)


def _lewis_shape(a3: float, half_breadth: float, half_draft: float) -> tuple[float, float]:
    """Return ``(M, a1)`` for a given ``a3`` matching ``b`` and ``h``.

    From ``b = M(1 + a1 + a3)`` and ``h = M(1 - a1 + a3)`` one gets
    ``M = (b + h) / (2 (1 + a3))`` and ``a1 = (b - h)/(b + h) * (1 + a3)``.
    """
    M = (half_breadth + half_draft) / (2.0 * (1.0 + a3))
    a1 = (half_breadth - half_draft) / (half_breadth + half_draft) * (1.0 + a3)
    return M, a1


def _lewis_section(a3: float, half_breadth: float, half_draft: float) -> LewisSection:
    M, a1 = _lewis_shape(a3, half_breadth, half_draft)
    return LewisSection(
        M=M,
        a1=a1,
        a3=a3,
        half_breadth=half_breadth,
        half_draft=half_draft,
        requested_area=float("nan"),
        area_ratio=float("nan"),
    )


def _contour_extremes(section: LewisSection, *, samples: int = 2048) -> tuple[float, float]:
    """Largest ``|x|`` and ``|y|`` on the contour (vectorised, no python loop)."""
    theta = torch.linspace(0.0, 2.0 * math.pi, samples, dtype=torch.float64)
    cos, sin = torch.cos(theta), torch.sin(theta)
    cos3, sin3 = torch.cos(3.0 * theta), torch.sin(3.0 * theta)
    x = section.M * ((1.0 + section.a1) * cos + section.a3 * cos3)
    y = section.M * ((1.0 - section.a1) * sin - section.a3 * sin3)
    return float(x.abs().max()), float(y.abs().max())


def _admissible(section: LewisSection, *, samples: int = 2048, tol: float = 1e-9) -> bool:
    """True while the contour stays inside its ``b x h`` bounding box."""
    x_max, y_max = _contour_extremes(section, samples=samples)
    return x_max <= section.half_breadth * (1.0 + tol) and y_max <= section.half_draft * (
        1.0 + tol
    )


def _max_boxy_a3(half_breadth: float, half_draft: float, *, samples: int = 2048) -> float:
    """Most negative admissible ``a3``: the boxiest section of the family.

    ``a3 < 0`` straightens the flanks and enlarges the section; beyond a limit
    the contour pokes out of the ``b x h`` bounding box (the shape stops
    representing a section of that breadth and draft), so the limit is found by
    bisection on the bounding-box test.
    """
    slope = (half_breadth - half_draft) / (half_breadth + half_draft)
    a3_ok = 0.0
    a3_bad = None
    step = -0.01
    probe = step
    while probe > -0.95:
        if abs(slope * (1.0 + probe)) + 3.0 * abs(probe) >= 0.98:
            a3_bad = probe
            break
        if not _admissible(_lewis_section(probe, half_breadth, half_draft), samples=samples):
            a3_bad = probe
            break
        a3_ok = probe
        probe += step
    if a3_bad is None:
        return a3_ok
    lo, hi = a3_ok, a3_bad
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _admissible(_lewis_section(mid, half_breadth, half_draft), samples=samples):
            lo = mid
        else:
            hi = mid
    return lo


def lewis_fit(
    half_breadth: float,
    half_draft: float,
    area: float | None = None,
    *,
    samples: int = 128,
    extremes: int = 2048,
) -> LewisSection:
    """Fit a Lewis form to ``(b, h)`` and, if given, to a target contour area.

    ``b`` and ``h`` are matched exactly and the contour is kept inside its
    bounding box; the remaining parameter ``a3`` shapes the section from the
    ellipse (``a3 = 0``) towards the boxiest admissible member (``a3 < 0``).
    When ``area`` is given, ``a3`` is chosen to reproduce the enclosed area;
    when the requested area exceeds what the family can produce, the boxiest
    member is returned and ``area_ratio`` reports the shortfall instead of
    silently clipping it.
    """
    if half_breadth <= 0.0 or half_draft <= 0.0:
        raise ValueError("half_breadth and half_draft must be positive")
    a3_min = _max_boxy_a3(half_breadth, half_draft, samples=extremes)

    def build(a3: float) -> tuple[LewisSection, float]:
        section = _lewis_section(a3, half_breadth, half_draft)
        achieved = section.area(samples)
        ratio = 1.0 if area is None else achieved / float(area)
        return (
            LewisSection(
                M=section.M,
                a1=section.a1,
                a3=a3,
                half_breadth=half_breadth,
                half_draft=half_draft,
                requested_area=0.0 if area is None else float(area),
                area_ratio=ratio,
            ),
            achieved,
        )

    section_ellipse, area_ellipse = build(0.0)
    section_max, area_max = build(a3_min)
    if area is None or area >= area_max:
        return section_max
    if area <= area_ellipse:
        return section_ellipse
    return _lewis_area_bisection(area, a3_min, build)


def _lewis_area_bisection(area: float, a3_min: float, build) -> LewisSection:
    """Bisect ``a3`` in ``[a3_min, 0]`` for the requested enclosed area.

    The area decreases monotonically as ``a3`` goes from ``a3_min`` (boxiest)
    to ``0`` (ellipse), so the bracket is ``[a3_min, 0]``.
    """
    lo, hi = a3_min, 0.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        _, achieved = build(mid)
        if achieved > area:  # too boxy, i.e. too much area: move towards the ellipse
            lo = mid
        else:
            hi = mid
    return build(0.5 * (lo + hi))[0]


def lewis_heave_added_mass(
    half_breadth: float,
    half_draft: float,
    rho: float,
    *,
    area: float | None = None,
    samples: int = 256,
) -> tuple[float, LewisSection]:
    """Added mass per unit length of a Lewis section in vertical motion."""
    section = lewis_fit(half_breadth, half_draft, area)
    return section.heave_added_mass(rho, samples), section


# ---------------------------------------------------------------------------
# Wet/dry scaling and finite element wet modes
# ---------------------------------------------------------------------------
def wet_dry_frequency_ratio(kappa: float) -> float:
    """``1 / sqrt(1 + kappa)`` for an added-mass ratio ``kappa >= 0``."""
    if kappa < 0.0:
        raise ValueError("kappa must be non-negative")
    return 1.0 / math.sqrt(1.0 + kappa)


def added_mass_matrix(section: BoxGirderSection, elements: int, added_mass: float,
                      *, dtype=torch.float64) -> torch.Tensor:
    """Consistent translational mass matrix for a uniform added mass [kg/m].

    Uses the same interdependent shape functions as the structural mass, so the
    added mass is distributed consistently along the girder (no rotary inertia:
    the water has no rotational added inertia in this strip model).
    """
    if elements < 1:
        raise ValueError("elements must be >= 1")
    if added_mass < 0.0:
        raise ValueError("added_mass must be non-negative")
    beam = section.to_timoshenko_section()
    le = beam.length / elements
    phi = 12.0 * section.bending_stiffness / (section.shear_stiffness * le**2)
    ke_scale = 0.5 * le  # xi in [0,1] <- [-1,1]
    ndof = 2 * (elements + 1)
    M = torch.zeros((ndof, ndof), dtype=dtype)
    me = torch.zeros((4, 4), dtype=dtype)
    for gp, gw in zip(GAUSS_POINTS_4, GAUSS_WEIGHTS_4):
        xi = 0.5 * (gp + 1.0)
        n_w, _, _, _ = timoshenko_shape_functions(xi, le, phi, dtype=dtype)
        me += (gw * ke_scale) * added_mass * torch.outer(n_w, n_w)
    for e in range(elements):
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        M[dofs[:, None], dofs] += me
    return M


def wet_simply_supported_modes(
    section: BoxGirderSection,
    elements: int,
    added_mass: float,
    extra_mass: float = 0.0,
    modes: int = 4,
    *,
    dtype=torch.float64,
) -> ModalResult:
    """Wet (in-water) modes of the pinned--pinned girder.

    ``extra_mass`` is the payload/ballast per unit length that makes the girder
    float at its draft (see :func:`ballast_mass_per_length`); ``added_mass`` is
    the hydrodynamic added mass per unit length.  Both enter through consistent
    translational mass matrices, so for a uniform girder the closed form
    ``f_wet = f_dry / sqrt(1 + kappa)`` with ``kappa = added/(m_struct+extra)``
    is reproduced exactly.
    """
    return beam_simply_supported_modes(
        section, elements, (extra_mass, added_mass), modes, dtype=dtype
    )


def variable_added_mass_modes(
    section: BoxGirderSection,
    elements: int,
    added_mass_at,
    extra_mass: float = 0.0,
    modes: int = 4,
    *,
    dtype=torch.float64,
) -> ModalResult:
    """Wet modes with an added mass that varies along the girder.

    ``added_mass_at(x)`` returns the local added mass per unit length [kg/m];
    it is sampled at each element midpoint and assembled with that element's
    consistent mass matrix.  This is the check of the *scalar* wet/dry
    reduction: when the added mass is not uniform, the ratio is a
    mode-shape-weighted average and no longer equals ``1/sqrt(1+kappa)``.
    """
    section.validate()
    if elements < 1:
        raise ValueError("elements must be >= 1")
    beam = section.to_timoshenko_section()
    K, M = assemble_timoshenko_beam(beam, elements, dtype=dtype)
    le = section.length / elements
    phi = 12.0 * section.bending_stiffness / (section.shear_stiffness * le**2)
    scale = 0.5 * le
    uniform = torch.zeros((4, 4), dtype=dtype)
    for gp, gw in zip(GAUSS_POINTS_4, GAUSS_WEIGHTS_4):
        n_w, _, _, _ = timoshenko_shape_functions(0.5 * (gp + 1.0), le, phi, dtype=dtype)
        uniform += (gw * scale) * torch.outer(n_w, n_w)
    for e in range(elements):
        xm = (e + 0.5) * le
        local = float(added_mass_at(xm)) + float(extra_mass)
        if local < 0.0:
            raise ValueError("added mass must be non-negative")
        dofs = torch.tensor([2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3])
        M[dofs[:, None], dofs] += local * uniform
    constrained = (0, 2 * elements)
    return solve_modes(K, M, constrained, modes=modes)


_COEFFICIENT_TABLE: dict | None = None
_COEFFICIENT_GRID = (-4.0, 1.9, 145)  # log(ratio) from 0.018 to 6.7, 145 nodes


def _coefficient_table() -> tuple[list[float], list[float]]:
    """Lazily built table of ``m33/(rho b^2)`` against ``log(h/b)``."""
    global _COEFFICIENT_TABLE
    if _COEFFICIENT_TABLE is None:
        lo, hi, count = _COEFFICIENT_GRID
        xs: list[float] = []
        ys: list[float] = []
        for i in range(count):
            log_ratio = lo + (hi - lo) * i / (count - 1)
            ratio = math.exp(log_ratio)
            xs.append(log_ratio)
            ys.append(lewis_fit(1.0, ratio).heave_added_mass(1.0, 128))
        _COEFFICIENT_TABLE = {"x": xs, "y": ys}
    return _COEFFICIENT_TABLE["x"], _COEFFICIENT_TABLE["y"]


def lewis_heave_coefficient(ratio: float, *, direct: bool = False) -> float:
    """Nondimensional heave added mass ``m33 / (rho * b^2)`` of a fitted section.

    ``ratio`` is ``half_draft / half_breadth``.  The Lewis form is refitted for
    each ratio (maximum-area member), so this is the coefficient a hull with a
    varying section needs.  For a flat plate (``ratio -> 0``) and for a circle
    (``ratio = 1``) it equals ``pi``; a boxier/denser section gives more.

    The result is read from a 145-node table in ``log(ratio)`` on
    ``[0.018, 6.7]`` with linear interpolation; the error is below ``1e-4``
    relative over the hull-like range ``ratio < 0.7`` and up to ``3e-3`` across
    the kink at ``ratio = 1`` where the boxiness saturates.  ``direct=True``
    refits instead, which is how the table itself is validated in the tests.
    """
    if ratio <= 0.0:
        raise ValueError("ratio must be positive")
    if direct:
        return lewis_fit(1.0, ratio).heave_added_mass(1.0, 256)
    xs, ys = _coefficient_table()
    log_ratio = math.log(ratio)
    if log_ratio <= xs[0]:
        return ys[0]
    if log_ratio >= xs[-1]:
        return ys[-1]
    # bisect on the uniform grid, then interpolate linearly.
    lo, hi = 0, len(xs) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= log_ratio:
            lo = mid
        else:
            hi = mid
    weight = (log_ratio - xs[lo]) / (xs[hi] - xs[lo])
    return ys[lo] + weight * (ys[hi] - ys[lo])


def floating_mass_per_length(section: BoxGirderSection) -> float:
    """Total vibrating mass per unit length of a freely floating girder [kg/m].

    A floating body displaces its own weight, so the mass that vibrates with
    the girder is the displaced water mass per unit length
    ``rho_w * B * T`` (structure plus payload/ballast), unless the bare steel
    is heavier than the displacement, in which case the steel alone is used.
    """
    section.validate()
    return max(section.mass_per_length, section.displaced_mass_per_length)


def ballast_mass_per_length(section: BoxGirderSection) -> float:
    """Payload/ballast needed to float the girder at its draft [kg/m]."""
    return floating_mass_per_length(section) - section.mass_per_length


def beam_simply_supported_modes(
    section: BoxGirderSection,
    elements: int,
    extra_masses: tuple[float, ...] = (),
    modes: int = 4,
    *,
    dtype=torch.float64,
) -> ModalResult:
    """Pinned--pinned modes with extra uniform translational masses [kg/m].

    ``extra_masses`` are added to the mass operator with the consistent
    translational shape functions (no rotary inertia), so ``K phi =
    omega^2 (M + sum M_extra) phi``: the dry girder plus payload, or the wet
    girder (payload + added mass).
    """
    section.validate()
    beam = section.to_timoshenko_section()
    K, M = assemble_timoshenko_beam(beam, elements, dtype=dtype)
    for extra in extra_masses:
        if extra < 0.0:
            raise ValueError("extra masses must be non-negative")
        M = M + added_mass_matrix(section, elements, extra, dtype=dtype)
    constrained = (0, 2 * elements)
    return solve_modes(K, M, constrained, modes=modes)


# ---------------------------------------------------------------------------
# Grid convergence (Roache GCI)
# ---------------------------------------------------------------------------
def gci(fine: float, medium: float, coarse: float, refinement: float = 2.0) -> dict:
    """Roache grid-convergence index for a three-grid refinement study.

    ``fine``/``medium``/``coarse`` are the same functional computed on
    successively coarser grids with uniform refinement factor ``refinement``.
    Returns the observed order of accuracy ``p``, the Richardson extrapolated
    value ``f_exact``, the fine-grid GCI (``1.25 * |eps| / (r^p - 1)``) and the
    relative error of each grid against the extrapolated value.
    """
    if refinement <= 1.0:
        raise ValueError("refinement must be > 1")
    e21 = medium - fine
    e32 = coarse - medium
    if e21 == 0.0 or e32 == 0.0:
        raise ValueError("degenerate convergence history")
    ratio = e32 / e21
    if ratio <= 0.0:
        raise ValueError("grids are not in the asymptotic range (oscillatory or diverging)")
    p = math.log(ratio) / math.log(refinement)
    f_exact = fine + (fine - medium) / (refinement**p - 1.0)
    return {
        "observed_order": p,
        "extrapolated": f_exact,
        "gci_fine": 1.25 * abs(fine - medium) / abs(fine) / (refinement**p - 1.0),
        "rel_error_fine": abs(fine - f_exact) / abs(f_exact),
        "rel_error_medium": abs(medium - f_exact) / abs(f_exact),
        "rel_error_coarse": abs(coarse - f_exact) / abs(f_exact),
        "convergence_ratio": ratio,
    }


def exact_hull_girder_angular_frequencies(
    section: BoxGirderSection, modes: int = 4, *, dtype=torch.float64
) -> torch.Tensor:
    """Exact simply supported Hu--Timoshenko frequencies of the equivalent girder."""
    return exact_simply_supported_timoshenko_angular_frequencies(
        section.to_timoshenko_section(), modes, dtype=dtype
    )