"""Physical calibration of the plate edge rotational restraint: scantlings -> beta.

Motivation
----------
:mod:`tensorfem.plate_elastic_restraint` parameterises the rotational fixity of a
plate edge with the embedding coefficient ``beta in [0, 1]`` and the distributed
spring coefficient

    ``c_r = zeta * D / side``,   ``zeta = beta / (1 - beta)``,   ``D = E t^3/(12(1-nu^2))``

i.e. ``c_r`` has units ``N*m/rad`` per metre of edge and ``side`` is the plate
dimension normal to the edge.  Everything that had been verified
(:mod:`tensorfem.plate_levy_restraint`, the beta band, the mesh invariance of the
tributary spring) was verified *for a given beta*.  The remaining hole - flagged
in ``docs/GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT.md`` §9.5 - is that ``beta``
itself was an arbitrary knob: nothing tied it to the longitudinals and frames
that actually bound an ice-belt panel, so the paper could not state the beta of a
real panel.

This module closes that hole.  It converts real scantlings into ``c_r`` and hence
into ``beta``, by

1. the **torsion-beam model** of the bounding member: the stiffener attached to
   the plate edge resists the edge rotation by twisting about its line of
   attachment, over one structural span (frame spacing for a longitudinal,
   longitudinal spacing for a frame), with end fixity supplied by the members it
   crosses;
2. homogenising that 1-D torsion problem into a *distributed* (Winkler-type)
   rotational spring ``c_r [N*m/rad/m]``, either by mean-rotation equivalence or
   by energy equivalence (both are returned; they coincide when the end fixity is
   symmetric);
3. the dimensionless group ``zeta = c_r * side / D`` which is the physically
   meaningful fixity measure - ``beta = zeta/(1+zeta)`` is only a monotone
   re-parameterisation of it (a fact that matters when quoting results: the
   *response* is strongly non-linear in beta, see the cases file).

Model (units: SI)
-----------------
A torque of intensity ``m [N*m/m]`` is applied along the member by the rotating
plate edge.  The member is a St. Venant torsion bar, ``GJ theta'' = -m``, with
rotational end springs ``K0``, ``K1 [N*m/rad]``:

    ``T(0) = K0 theta(0)``,   ``T(L) = -K1 theta(L)``,   ``T = GJ theta'``.

With ``theta(x) = -m x^2/(2 GJ) + C1 x + C0`` this is a 2x2 system; its closed
form solution is used here (no discretisation).  ``K = inf`` is a torsionally
clamped end and is handled as the analytic limit.

Two reference values follow analytically and are asserted in the tests:

* torsionally fixed at both ends: ``c_r = 12 GJ / L^2``
* torsionally fixed at one end, free at the other: ``c_r = 3 GJ / L^2``

Both are the classic stiffnesses of a bar ("12" / "3") transferred from a
concentrated end spring to a distributed rotational foundation.

Verification (see ``tests/test_plate_restraint_calibration.py``)
--------------------------------------------------------------
* the closed form is checked against an **independent 1-D finite-element**
  torsion-beam solve (quadratic elements + end springs) at several ``(K0, K1)``,
* ``J`` is checked against :mod:`tensorfem.open_section_torsion` (thin-walled,
  independent quadrature) and against the exact rectangle series
  ``J = b t^3/3 (1 - 0.63 t/b)``,
* the calibrated ``c_r`` fed into the exact Kirchhoff references
  (:func:`tensorfem.plate_levy_restraint.levy_restrained_plate`,
  :func:`~tensorfem.plate_levy_restraint.ritz_restrained_plate`) reproduces the
  Mindlin FE panel, which proves the calibration is expressed in the same
  convention the solver uses,
* ``beta = 0`` from a zero-stiffness member still reproduces the JOUR-ICE-STRUCT
  regression baseline (the pre-existing guard is untouched).

Assumptions / limits
--------------------
* St. Venant torsion only; warping (``E C_w``) is neglected.  For an open
  section with a free-to-warp end this *over*-estimates the restraint; for
  ``L/h > 10`` the error is a few percent (see the cases file §notes).
* The plate edge is assumed to rotate about the stiffener attachment line, and
  the stiffener is assumed to be held at the members it crosses.  The
  "attached plating" can optionally be included in ``J`` (see
  ``include_attached_plating``); the default excludes it, because the plating is
  the very thing whose rotation is being restrained.
* Linear elastic, small deflection, uniform member properties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

__all__ = [
    "shear_modulus",
    "torsion_constant",
    "torsion_constant_rectangle",
    "TorsionBeamRestraint",
    "torsion_beam_restraint",
    "beta_from_k_rot",
    "k_rot_from_beta",
    "StiffenerSection",
    "panel_restraint",
    "PanelRestraint",
]


# --------------------------------------------------------------------------- #
# section properties
# --------------------------------------------------------------------------- #
def shear_modulus(young: float, poisson: float) -> float:
    """Isotropic shear modulus ``G = E / (2 (1 + nu))``."""
    if young <= 0.0:
        raise ValueError("young must be positive")
    if not -1.0 < poisson < 0.5:
        raise ValueError("poisson must be in (-1, 0.5)")
    return young / (2.0 * (1.0 + poisson))


def torsion_constant(segments: Iterable[tuple[float, float]],
                     *, correction: bool = True) -> float:
    """St. Venant torsion constant ``J`` of an open thin-walled section.

    ``segments`` are ``(wall length, thickness)`` pairs.  Each wall is a
    rectangle, so its contribution is

    * ``correction=True`` (default):
      ``J = l t^3/3 (1 - 0.63 t/l)``, the exact rectangle series to first order
      (see :func:`torsion_constant_rectangle`);
    * ``correction=False``: ``J = l t^3/3``, the bare thin-wall limit used
      elsewhere in the library
      (:func:`tensorfem.grillage_beam_grid._tee_torsion_constant`).

    The two differ by the thickness-ratio penalty, which is *not* small for
    ship stiffener flanges (a 40x22 bulb flange loses ~35%).  Stiffener
    scantlings are chunky, so the corrected value is the default: it is also the
    conservative choice here, because a smaller ``J`` means less edge fixity and
    hence higher plate stress.
    """
    total = 0.0
    used = False
    for length, thickness in segments:
        length = float(length)
        thickness = float(thickness)
        if length < 0.0 or thickness < 0.0:
            raise ValueError("wall length and thickness must be non-negative")
        if length == 0.0 or thickness == 0.0:
            continue
        used = True
        if correction:
            total += torsion_constant_rectangle(length, thickness)
        else:
            total += length * thickness**3 / 3.0
    if not used:
        raise ValueError("at least one wall with positive length and thickness")
    return total


def torsion_constant_rectangle(width: float, thickness: float) -> float:
    """Exact (series) St. Venant constant of a solid rectangle.

    ``J = b t^3/3 (1 - 0.63 t/b)`` for ``b >= t`` (Timoshenko, *Theory of
    Elasticity*, §117; the single-term form is within 0.1% for ``t/b <= 0.5``).
    Used to bound the thin-wall approximation above.
    """
    b, t = max(width, thickness), min(width, thickness)
    if t <= 0.0:
        raise ValueError("thickness must be positive")
    return b * t**3 / 3.0 * (1.0 - 0.63 * t / b)


# --------------------------------------------------------------------------- #
# 1-D torsion beam between elastically restrained ends
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TorsionBeamRestraint:
    """Homogenised distributed rotational restraint of one member.

    Attributes
    ----------
    k_rot : float
        Distributed rotational spring stiffness ``[N*m/rad/m]`` defined by
        mean-rotation equivalence, ``k_rot = m / theta_mean``.
    k_rot_energy : float
        Same quantity defined by energy equivalence,
        ``k_rot = m^2 L / (2 U)`` where ``U`` is the strain energy of the member
        under the distributed torque ``m``.  Equal to ``k_rot`` for symmetric
        end fixity; the pair brackets the definition ambiguity otherwise.
    theta_mean : float
        Mean rotation of the member under unit distributed torque
        ``[rad/(N*m/m)]`` - i.e. ``1/k_rot``.
    gj, span, k0, k1 : float
        Inputs echoed back (``k0``/``k1`` may be ``inf``).
    clamped_both : float
        ``12 GJ / L^2`` - the analytic both-ends-fixed value, for reference.
    """

    k_rot: float
    k_rot_energy: float
    theta_mean: float
    gj: float
    span: float
    k0: float
    k1: float

    @property
    def clamped_both(self) -> float:
        return 12.0 * self.gj / self.span**2

    @property
    def free_mechanism(self) -> bool:
        """True when both ends are torsionally free (no restraint at all)."""
        return self.k_rot == 0.0


def _rotation_coeffs(m: float, gj: float, span: float,
                     k0: float, k1: float) -> tuple[float, float]:
    """``(C0, C1)`` of ``theta(x) = -m x^2/(2 GJ) + C1 x + C0``.

    Solves ``T(0) = K0 theta(0)``, ``T(L) = -K1 theta(L)`` with ``T = GJ theta'``,
    treating ``inf`` as the analytic clamped limit.  Raises ``ValueError`` when
    both ends are free (rigid-body rotation: there is no restraint).
    """
    L, phi = span, gj
    inf = math.inf
    if k0 == inf and k1 == inf:                     # clamped - clamped
        return 0.0, m * L / (2.0 * phi)
    if k0 == inf:                                   # clamped - spring k1
        c1 = m * (L + k1 * L * L / (2.0 * phi)) / (phi + k1 * L)
        return 0.0, c1
    if k1 == inf:                                   # spring k0 - clamped
        c1 = m * L * L / (2.0 * phi) / (L + phi / k0)
        return phi * c1 / k0, c1
    det = k0 * (phi + k1 * L) + phi * k1
    if det <= 0.0:
        raise ValueError(
            "both member ends are torsionally free: the member rotates rigidly "
            "and provides no rotational restraint (k_rot = 0)")
    rhs = m * (L + k1 * L * L / (2.0 * phi))
    # k0 C0 - phi C1 = 0 ;  k1 C0 + (phi + k1 L) C1 = rhs
    c1 = k0 * rhs / det
    c0 = phi * c1 / k0
    return c0, c1


def torsion_beam_restraint(gj: float, span: float,
                           k0: float = math.inf, k1: Optional[float] = None,
                           *, m: float = 1.0) -> TorsionBeamRestraint:
    """Homogenise a torsion bar (span ``L``, ends ``K0``/``K1``) into ``k_rot``.

    Parameters
    ----------
    gj : float
        Torsional rigidity ``G * J [N*m^2]`` of the bounding member.
    span : float
        Structural span of that member between the members that hold it
        (frame spacing for a longitudinal) [m].
    k0, k1 : float
        Rotational end fixity ``[N*m/rad]``; ``math.inf`` = torsionally clamped
        (the usual assumption at a frame), ``0`` = free to rotate.
        ``k1`` defaults to ``k0``.
    m : float
        Distributed torque used to probe the response; any positive value gives
        the same ``k_rot`` (the problem is linear).  ``[N*m/m]``

    Returns
    -------
    TorsionBeamRestraint

    Notes
    -----
    ``k_rot = 12 GJ/L^2`` for clamped-clamped and ``3 GJ/L^2`` for
    clamped-free, exactly; both limits are asserted in the tests.
    """
    if gj <= 0.0:
        raise ValueError("gj must be positive")
    if span <= 0.0:
        raise ValueError("span must be positive")
    if m <= 0.0:
        raise ValueError("m must be positive")
    if k1 is None:
        k1 = k0
    if k0 < 0.0 or k1 < 0.0:
        raise ValueError("end restraints must be non-negative")
    if k0 == 0.0 and k1 == 0.0:
        # rigid-body rotation: no restraint at all
        return TorsionBeamRestraint(0.0, 0.0, math.inf, gj, span, k0, k1)

    c0, c1 = _rotation_coeffs(m, gj, span, k0, k1)
    theta_mean = (-m * span**2 / (6.0 * gj) + c1 * span / 2.0 + c0)
    if theta_mean <= 0.0:  # pragma: no cover - would mean a sign bug
        raise RuntimeError("non-positive mean rotation; check sign conventions")
    k_rot = m / theta_mean

    # energy: U = 1/2 GJ int(theta'^2) + 1/2 K0 theta(0)^2 + 1/2 K1 theta(L)^2
    int_d1sq = (m * m * span**3 / (3.0 * gj**2)
                - m * c1 * span**2 / gj + c1 * c1 * span)
    theta_L = -m * span**2 / (2.0 * gj) + c1 * span + c0
    energy = 0.5 * gj * int_d1sq
    if k0 != math.inf:
        energy += 0.5 * k0 * c0 * c0
    if k1 != math.inf:
        energy += 0.5 * k1 * theta_L * theta_L
    k_rot_energy = m * m * span / (2.0 * energy)
    return TorsionBeamRestraint(k_rot, k_rot_energy, theta_mean,
                                gj, span, k0, k1)


# --------------------------------------------------------------------------- #
# restraint parameter: c_r  <->  zeta  <->  beta
# --------------------------------------------------------------------------- #
def beta_from_k_rot(k_rot: float, D: float, span: float) -> tuple[float, float]:
    """``(zeta, beta)`` of a distributed edge spring ``k_rot [N*m/rad/m]``.

    ``zeta = k_rot * span / D`` and ``beta = zeta / (1 + zeta)``.  This is the
    exact inverse of
    :func:`tensorfem.plate_elastic_restraint.beta_to_zeta`, so a calibrated
    ``k_rot`` can be fed to the solver without any re-fitting.
    """
    if D <= 0.0 or span <= 0.0:
        raise ValueError("D and span must be positive")
    if k_rot < 0.0:
        raise ValueError("k_rot must be non-negative")
    zeta = k_rot * span / D
    if math.isinf(zeta):
        return zeta, 1.0
    return zeta, zeta / (1.0 + zeta)


def k_rot_from_beta(beta: float, D: float, span: float) -> float:
    """Distributed edge spring ``[N*m/rad/m]`` of embedding coefficient beta."""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    return beta / (1.0 - beta) * D / span


# --------------------------------------------------------------------------- #
# scantlings -> beta for a real panel
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StiffenerSection:
    """Idealised stiffener: a flat bar, or a T with an optional flange.

    All dimensions in metres.  ``plate_flange_width``/``plate_thickness`` add an
    attached-plating wall of that breadth (only used when
    ``include_attached_plating`` is requested at calibration time).
    """

    web_height: float
    web_thickness: float
    flange_width: float = 0.0
    flange_thickness: float = 0.0
    name: str = "stiffener"

    def __post_init__(self) -> None:
        if self.web_height <= 0.0 or self.web_thickness <= 0.0:
            raise ValueError("web must have positive height and thickness")
        if (self.flange_width > 0.0) != (self.flange_thickness > 0.0):
            raise ValueError("flange needs both width and thickness, or neither")

    @property
    def is_flat_bar(self) -> bool:
        return self.flange_width == 0.0

    def walls(self, *, flange_split: bool = False) -> list[tuple[float, float]]:
        """``(wall length, thickness)`` of every wall of the section.

        ``flange_split`` represents the flange as the two half-width walls of the
        library's tree model (:func:`tensorfem.open_section_torsion.tee_section`,
        which puts a node at the web/flange junction).  That matters only with
        ``correction=True``: a 40 x 22 bulb flange as one rectangle and as two
        20 x 22 rectangles differ by ~25%, which is a genuine idealisation
        uncertainty of a chunky flange, not a coding choice - the calibration
        band reports both.
        """
        walls = [(self.web_height, self.web_thickness)]
        if self.flange_width > 0.0:
            if flange_split:
                walls.append((0.5 * self.flange_width, self.flange_thickness))
                walls.append((0.5 * self.flange_width, self.flange_thickness))
            else:
                walls.append((self.flange_width, self.flange_thickness))
        return walls

    def torsion_constant(self, *, attached_plate_width: float = 0.0,
                         attached_plate_thickness: float = 0.0,
                         correction: bool = True, flange_split: bool = False) -> float:
        walls = self.walls(flange_split=flange_split)
        if attached_plate_width > 0.0 and attached_plate_thickness > 0.0:
            walls = walls + [(attached_plate_width, attached_plate_thickness)]
        return torsion_constant(walls, correction=correction)


@dataclass(frozen=True)
class PanelRestraint:
    """Calibrated restraint of one rectangular panel (results of the model).

    ``beta_x`` applies to the edges ``x = 0, a`` (bound by the *frame*),
    ``beta_y`` to the edges ``y = 0, b`` (bound by the *longitudinals*), matching
    the argument order of
    :func:`tensorfem.plate_elastic_restraint.solve_restrained_panel`.
    """

    beta_x: float
    beta_y: float
    zeta_x: float
    zeta_y: float
    k_rot_x: float
    k_rot_y: float
    D: float
    side_x: float
    side_y: float
    frame: TorsionBeamRestraint = field(repr=False)
    longitudinal: TorsionBeamRestraint = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "beta_x": self.beta_x,
            "beta_y": self.beta_y,
            "zeta_x": self.zeta_x,
            "zeta_y": self.zeta_y,
            "k_rot_x_Nm_per_rad_per_m": self.k_rot_x,
            "k_rot_y_Nm_per_rad_per_m": self.k_rot_y,
            "k_rot_x_energy": self.frame.k_rot_energy,
            "k_rot_y_energy": self.longitudinal.k_rot_energy,
            "D_Nm": self.D,
            "side_x_m": self.side_x,
            "side_y_m": self.side_y,
            "longitudinal_gj_Nm2": self.longitudinal.gj,
            "frame_gj_Nm2": self.frame.gj,
            "longitudinal_span_m": self.longitudinal.span,
            "frame_span_m": self.frame.span,
        }


def panel_restraint(
    *,
    young: float,
    poisson: float,
    plate_thickness: float,
    side_x: float,
    side_y: float,
    longitudinal: StiffenerSection,
    frame: StiffenerSection,
    longitudinal_span: Optional[float] = None,
    frame_span: Optional[float] = None,
    longitudinal_ends: float | tuple[float, float] = math.inf,
    frame_ends: float | tuple[float, float] = math.inf,
    include_attached_plating: bool = False,
    correction: bool = True,
) -> PanelRestraint:
    """Calibrate ``beta_x``/``beta_y`` of a panel from its real scantlings.

    Parameters
    ----------
    young, poisson, plate_thickness
        Plate material and thickness.
    side_x, side_y
        Panel dimensions along x and y [m].  ``side_x`` is the frame spacing
        (the span of the longitudinals), ``side_y`` the longitudinal spacing.
    longitudinal, frame
        :class:`StiffenerSection` of the members bounding the ``y`` and ``x``
        edges respectively.
    longitudinal_span
        Span of the longitudinal between the members that hold it; defaults to
        ``side_x`` (the longitudinal spans the frame spacing).
    frame_span
        Free torsional span of the frame in way of the panel; defaults to
        ``side_y`` (the frame is held at the longitudinals it crosses).  Pass
        half the girder spacing for the "frame twists as a whole" model.
    longitudinal_ends, frame_ends
        Rotational end fixity of the bounding member, scalar or ``(k0, k1)``,
        ``math.inf`` = torsionally clamped.  Default both clamped.
    include_attached_plating
        Add the attached plating (breadth = the transverse spacing) to the
        torsion constant.  Off by default: the plating is the member whose
        rotation is being restrained, so counting it would double-count.
    correction
        Use the exact rectangle series for each wall of the section (default) or
        the bare ``l t^3/3`` thin-wall limit.

    Returns
    -------
    PanelRestraint
    """
    def _ends(value):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError("end fixity must be a scalar or a (k0, k1) pair")
            return float(value[0]), float(value[1])
        return float(value), float(value)

    D = young * plate_thickness**3 / (12.0 * (1.0 - poisson**2))
    G = shear_modulus(young, poisson)

    l_span = float(side_x if longitudinal_span is None else longitudinal_span)
    f_span = float(side_y if frame_span is None else frame_span)
    if l_span <= 0.0 or f_span <= 0.0:
        raise ValueError("member spans must be positive")

    l_plate_w = float(side_y) if include_attached_plating else 0.0
    f_plate_w = float(side_x) if include_attached_plating else 0.0
    j_l = longitudinal.torsion_constant(
        attached_plate_width=l_plate_w, attached_plate_thickness=plate_thickness,
        correction=correction)
    j_f = frame.torsion_constant(
        attached_plate_width=f_plate_w, attached_plate_thickness=plate_thickness,
        correction=correction)

    k0_l, k1_l = _ends(longitudinal_ends)
    k0_f, k1_f = _ends(frame_ends)
    long_res = torsion_beam_restraint(G * j_l, l_span, k0_l, k1_l)
    frame_res = torsion_beam_restraint(G * j_f, f_span, k0_f, k1_f)

    zeta_y, beta_y = beta_from_k_rot(long_res.k_rot, D, side_y)
    zeta_x, beta_x = beta_from_k_rot(frame_res.k_rot, D, side_x)
    return PanelRestraint(
        beta_x=beta_x, beta_y=beta_y,
        zeta_x=zeta_x, zeta_y=zeta_y,
        k_rot_x=frame_res.k_rot, k_rot_y=long_res.k_rot,
        D=D, side_x=float(side_x), side_y=float(side_y),
        frame=frame_res, longitudinal=long_res,
    )