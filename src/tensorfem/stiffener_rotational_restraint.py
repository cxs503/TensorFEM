"""Physical calibration of the plate-edge rotational restraint ``beta``.

Anchor: ``GAP-PP-4-ICE-STRUCT-ELASTIC-RESTRAINT`` (JOUR-ICE-STRUCT Sec. 4.2).

Why this module exists
----------------------
:mod:`tensorfem.plate_elastic_restraint` can solve a rectangular panel whose
four edges carry a *distributed* rotational spring of stiffness
``K_rot = zeta * D / side``, with ``zeta`` obtained from a dimensionless
"embedding coefficient" ``beta`` via ``zeta = beta / (1 - beta)``.  That is a
complete numerical capability, but ``beta`` is a **free parameter**: nothing in
the library says what value a real ice-belt plate field has, so the manuscript
can only quote a band, not a design point.

This module supplies the missing physics — the distributed rotational stiffness
that a longitudinal stiffener (or a transverse frame) actually offers to the
plate edge it supports — and converts it into exactly the ``zeta`` / ``beta``
the existing solver consumes.  It answers, for real scantlings, the question
"is the panel nearer simply supported or nearer clamped?", and quantifies how
much of the clamping the real structure delivers.

Model
-----
The plate edge rotates by ``theta`` about the edge line, and the attached
stiffener must **twist** by the same angle (it is welded to the plate along its
length).  The stiffener is torsionally held at its ends by the members running
perpendicular to it (frames hold the longitudinals, longitudinals hold the
frames) at a spacing ``L``.  For a uniform distributed torque ``m`` per unit
length applied over a shaft of torsional rigidity ``GJ`` whose twist is fixed at
both ends,

    GJ phi'' = -m ,  phi(0) = phi(L) = 0
    =>  phi(x) = m x (L - x) / (2 GJ),   mean(phi) = m L^2 / (12 GJ)

so the equivalent *distributed* rotational spring constants per unit edge
length is

    k_theta = m / mean(phi) = 12 GJ / L^2        [N*m/m/rad = N/rad]

and the dimensionless restraint and embedding coefficient follow as

    zeta = k_theta * a / D ,   beta = zeta (1 + eps) / (1 + zeta)

with ``D`` the plate flexural rigidity and ``a`` the plate span normal to that
edge.  The ``(1 + eps)`` factor is the exact algebraic inverse of
:func:`tensorfem.plate_elastic_restraint.beta_to_zeta` (which regularises
``beta -> 1`` with ``eps``), so the round trip
``k_theta -> beta -> K_rot`` returns the input stiffness to machine precision
for every ``zeta < 1/eps``.

**Saturation is a real limit of the ``beta`` parameterisation, not of this
module**: the solver's ``zeta = beta/(1 - beta + eps)`` cannot exceed ``1/eps``
(``1e6`` by default), so a restraint stiffer than ``zeta = 1/eps`` maps to
``beta = 1`` and its excess stiffness is *not* representable.  That regime is
reported (see :attr:`PanelRestraintCalibration.warnings`) rather than silently
truncated.

Structural subtlety worth stating explicitly
--------------------------------------------
The **support span of a member is the spacing of the members perpendicular to
it**, which is in general *not* the panel span in the same direction.  A
longitudinal stiffener is torsionally restrained by the transverse frames, so
its ``L`` is the *frame* spacing; a transverse frame is restrained by the
longitudinals, so its ``L`` is the *longitudinal* spacing.  Using the wrong span
is a common error and it enters the stiffness quadratically.

Critically, ``beta`` is **not** a moment-fixity ratio.  ``beta`` is a
monotone reparameterisation of ``zeta`` chosen by the solver; the physically
interpretable quantity is the fraction of the clamped edge moment that the real
restraint mobilises.  :func:`moment_fixity_strip` returns that fraction from the
exact cylindrical-bending (beam-strip) closed form already verified in
:mod:`tensorfem.plate_levy_restraint`.

Limitations (see also the anchor case file)
------------------------------------------
* St Venant torsion only: warping restraint of the stiffener web is neglected
  (conservative — it makes the real restraint stiffer, so ``beta`` is a lower
  bound).
* One-sided rotational restraint from a *single* member; eccentricity of the
  stiffener neutral axis and the resulting coupling with plate membrane action
  are ignored.
* Linear elasticity, small deflection; no plate/stiffener local buckling, which
  would soften the restraint near the elastic limit.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

from .plate_levy_restraint import flexural_rigidity, kirchhoff_strip_end_moment

__all__ = [
    "shear_modulus",
    "flexural_rigidity",
    "flat_bar_torsion_constant",
    "thin_walled_torsion_constant",
    "torsional_distributed_stiffness",
    "beta_from_rotational_stiffness",
    "moment_fixity_strip",
    "EdgeRestraint",
    "PanelRestraintCalibration",
    "calibrate_panel_restraint",
    "sensitivity_sweep",
]

# Regularisation used by ``beta_to_zeta``; kept identical so the inverse is exact.
_DEFAULT_EPS = 1e-6


# ---------------------------------------------------------------------------
# Section and material properties
# ---------------------------------------------------------------------------

def shear_modulus(young: float, poisson: float) -> float:
    """Isotropic shear modulus ``G = E / (2 (1 + nu))`` [Pa]."""
    if young <= 0.0:
        raise ValueError(f"young must be positive, got {young}")
    if not -1.0 < poisson < 0.5:
        raise ValueError(f"poisson must lie in (-1, 0.5), got {poisson}")
    return young / (2.0 * (1.0 + poisson))


def flat_bar_torsion_constant(width: float, thickness: float) -> float:
    """St Venant torsion constant of a solid rectangle [m^4].

    ``J = b t^3 / 3 * (1 - 0.63 t/b + 0.052 (t/b)^5)`` for ``b >= t``
    (Timoshenko & Goodier, *Theory of Elasticity*, 3rd ed., Art. 117).  The
    series is accurate to better than 0.5 % over the whole rectangle range: it
    gives ``0.1407 b t^3`` for a square, against the exact ``0.1406 b t^3``.

    Parameters
    ----------
    width, thickness : float
        Section dimensions [m]; the larger one is used as ``b``.
    """
    for name, value in (("width", width), ("thickness", thickness)):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    b = max(width, thickness)
    t = min(width, thickness)
    r = t / b
    return b * t**3 / 3.0 * (1.0 - 0.63 * r + 0.052 * r**5)


def thin_walled_torsion_constant(rectangles: Iterable[Sequence[float]]) -> float:
    """Torsion constant of a thin-walled open section, ``J = sum(b t^3 / 3)``.

    This is the sum over rectangular wall elements (``[width, thickness]``
    pairs) and is the convention used by ship classification rules for
    bulb/flat-bar/L/T profiles, where the connecting fillets are neglected.
    It tends to slightly under-predict ``J`` (conservative for a restraint
    stiffness).

    Parameters
    ----------
    rectangles : iterable of (width, thickness)
        All wall elements of the section [m].
    """
    rects = [(float(b), float(t)) for b, t in rectangles]
    if not rects:
        raise ValueError("need at least one (width, thickness) pair")
    for b, t in rects:
        if b <= 0.0 or t <= 0.0:
            raise ValueError(f"wall dimensions must be positive, got ({b}, {t})")
    return math.fsum(b * t**3 / 3.0 for b, t in rects)


def torsional_distributed_stiffness(
    torsion_rigidity: float,
    support_span: float,
) -> float:
    """Distributed rotational spring from a torsionally end-fixed member [N/rad].

    ``k_theta = 12 GJ / L^2`` for a member of torsional rigidity ``GJ``[N*m^2]
    whose twist is fixed at supports a distance ``L`` [m] apart, loaded by a
    uniform distributed torque (see the module docstring for the derivation).

    This is a *distributed* stiffness (moment per unit length per radian), with
    the dimension of a force.  It must **not** be confused with ``GJ / L``,
    which is a *nodal* rotational spring [N*m/rad] relating the total torque to
    an end rotation — different dimension, different physical situation.

    Parameters
    ----------
    torsion_rigidity : float
        ``GJ`` of the member [N*m^2], e.g. from
        :func:`flat_bar_torsion_constant` times
        :func:`shear_modulus`.
    support_span : float
        Spacing of the members that torsionally fix the ends [m].

    Raises
    ------
    ValueError
        For non-positive input.  A torsionally *free* member cannot react
        torque (its end twist would follow the plate and the restraint would be
        zero), so only the fixed-ended case is defined here.
    """
    if torsion_rigidity <= 0.0:
        raise ValueError(f"torsion_rigidity must be positive, got {torsion_rigidity}")
    if support_span <= 0.0:
        raise ValueError(f"support_span must be positive, got {support_span}")
    return 12.0 * torsion_rigidity / support_span**2


# ---------------------------------------------------------------------------
# Mapping to the solver's restraint parameter
# ---------------------------------------------------------------------------

def beta_from_rotational_stiffness(
    k_theta: float,
    D: float,
    span: float,
    eps: float = _DEFAULT_EPS,
) -> float:
    """Embedding coefficient ``beta`` for a distributed rotational stiffness.

    Exact inverse of
    :func:`tensorfem.plate_elastic_restraint.beta_to_zeta`, i.e.
    ``beta_to_zeta(beta_from_rotational_stiffness(k, D, a)) * D / a == k``.

    Parameters
    ----------
    k_theta : float
        Distributed edge rotational stiffness [N/rad]; ``0`` is a simple
        support.  Must be finite and non-negative; pass a large finite value
        (not ``inf``) for a clamped idealisation.
    D : float
        Plate flexural rigidity [N*m].
    span : float
        Plate span normal to the restrained edge [m].
    eps : float
        Regularisation of ``beta -> 1``; must match the solver's value.

    Returns
    -------
    beta : float
        Value in ``[0, 1]``.  Exactly ``1.0`` once ``zeta >= 1/eps``, where the
        solver's parameterisation saturates (the excess stiffness cannot be
        represented; :func:`calibrate_panel_restraint` reports that case in
        ``warnings`` instead of pretending the map is exact).
    """
    if not math.isfinite(k_theta):
        raise ValueError(f"k_theta must be finite, got {k_theta}")
    if k_theta < 0.0:
        raise ValueError(f"k_theta must be non-negative, got {k_theta}")
    if D <= 0.0 or span <= 0.0:
        raise ValueError("D and span must be positive")
    if not 0.0 < eps < 1.0:
        raise ValueError(f"eps must lie in (0, 1), got {eps}")
    zeta = k_theta * span / D
    if zeta >= 1.0 / eps:  # solver parameterisation saturates here
        return 1.0
    return zeta * (1.0 + eps) / (1.0 + zeta)


def moment_fixity_strip(
    k_theta: float,
    D: float,
    span: float,
    q: float = 1.0,
) -> float:
    """Fraction of the clamped edge moment mobilised by a spring [dimensionless].

    ``K_rot`` here is the *distributed* stiffness used by the plate solver, and
    the reference is the exact cylindrical-bending strip result of
    :func:`tensorfem.plate_levy_restraint.kirchhoff_strip_end_moment`:

        ``M_end = q L^2 / 12 / (1 + 2 D / (K L))``

    so the ratio is ``1 / (1 + 2 D / (K L))``: ``0`` when the edge is free to
    rotate, ``1`` for a clamped edge.  This — not ``beta`` — is the physical
    measure of how much clamping the structure delivers.

    Parameters
    ----------
    k_theta : float
        Distributed rotational stiffness [N/rad].
    D : float
        Plate flexural rigidity [N*m].
    span : float
        Strip span (the plate dimension in the bending direction) [m].
    q : float
        Uniform load used for normalisation only [Pa]; cancels in the ratio.
    """
    if k_theta < 0.0:
        raise ValueError(f"k_theta must be non-negative, got {k_theta}")
    if D <= 0.0 or span <= 0.0:
        raise ValueError("D and span must be positive")
    clamped = q * span**2 / 12.0
    if k_theta == 0.0:
        return 0.0
    return kirchhoff_strip_end_moment(q, span, D, k_theta) / clamped


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EdgeRestraint:
    """Calibrated restraint of one pair of opposite panel edges."""

    member: str
    torsion_rigidity: float   # GJ [N*m^2]
    support_span: float       # spacing of the torsionally fixing members [m]
    k_theta: float            # distributed stiffness [N/rad]
    panel_span: float         # plate span normal to the edge [m]
    D: float                  # plate flexural rigidity [N*m]
    zeta: float               # dimensionless solver restraint
    beta: float               # embedding coefficient handed to the solver
    moment_fixity: float      # mobilised fraction of the clamped edge moment

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PanelRestraintCalibration:
    """Physical restraint of a rectangular panel, ready to feed the solver."""

    young: float
    poisson: float
    thickness: float
    side_x: float
    side_y: float
    D: float
    longitudinal: EdgeRestraint
    frame: EdgeRestraint
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def solver_kwargs(self) -> dict:
        """Keyword arguments for :func:`solve_restrained_panel`."""
        return {"beta_x": self.longitudinal.beta, "beta_y": self.frame.beta}

    def to_dict(self) -> dict:
        out = {
            "young": self.young,
            "poisson": self.poisson,
            "thickness": self.thickness,
            "side_x": self.side_x,
            "side_y": self.side_y,
            "D": self.D,
            "longitudinal": self.longitudinal.to_dict(),
            "frame": self.frame.to_dict(),
            "warnings": list(self.warnings),
        }
        return out


# ---------------------------------------------------------------------------
# High-level calibration
# ---------------------------------------------------------------------------

def _member_rigidity(
    name: str,
    torsion_rigidity: float | None,
    rectangles: Iterable[Sequence[float]] | None,
    young: float,
    poisson: float,
) -> float:
    if torsion_rigidity is not None and rectangles is not None:
        raise ValueError(f"give either torsion_rigidity or rectangles for {name}, not both")
    if torsion_rigidity is None and rectangles is None:
        raise ValueError(f"need torsion_rigidity or rectangles for {name}")
    if torsion_rigidity is not None:
        if torsion_rigidity <= 0.0:
            raise ValueError(f"{name} torsion_rigidity must be positive")
        return float(torsion_rigidity)
    J = thin_walled_torsion_constant(rectangles)
    return shear_modulus(young, poisson) * J


def _edge_restraint(
    member: str,
    gj: float,
    support_span: float,
    panel_span: float,
    D: float,
    eps: float,
) -> EdgeRestraint:
    k_theta = torsional_distributed_stiffness(gj, support_span)
    zeta = k_theta * panel_span / D
    beta = beta_from_rotational_stiffness(k_theta, D, panel_span, eps=eps)
    fixity = moment_fixity_strip(k_theta, D, panel_span)
    return EdgeRestraint(
        member=member,
        torsion_rigidity=gj,
        support_span=support_span,
        k_theta=k_theta,
        panel_span=panel_span,
        D=D,
        zeta=zeta,
        beta=beta,
        moment_fixity=fixity,
    )


def calibrate_panel_restraint(
    *,
    young: float,
    poisson: float,
    thickness: float,
    side_x: float,
    side_y: float,
    longitudinal_torsion_rigidity: float | None = None,
    longitudinal_rectangles: Iterable[Sequence[float]] | None = None,
    frame_torsion_rigidity: float | None = None,
    frame_rectangles: Iterable[Sequence[float]] | None = None,
    longitudinal_support_span: float | None = None,
    frame_support_span: float | None = None,
    eps: float = _DEFAULT_EPS,
) -> PanelRestraintCalibration:
    """Calibrate both edge families of a rectangular panel from real scantlings.

    The panel occupies ``[0, side_x] x [0, side_y]``.  Edges ``x = 0`` and
    ``x = side_x`` are supported by **longitudinals**; edges ``y = 0`` and
    ``y = side_y`` by **transverse frames**.

    Default support spans follow the physical situation: a longitudinal is
    torsionally fixed by the transverse frames, so its support span defaults to
    ``side_y`` (frame spacing); a frame is fixed by the longitudinals, so its
    support span defaults to ``side_x`` (longitudinal spacing).  Override them
    only if the real member layout differs (e.g. girders).

    Parameters
    ----------
    young, poisson : float
        Plate material [Pa, -].
    thickness : float
        Plate thickness [m].
    side_x, side_y : float
        Panel spans [m]; ``side_x`` is the spacing of the longitudinals.
    longitudinal_torsion_rigidity, frame_torsion_rigidity : float, optional
        ``GJ`` of one stiffener/frame [N*m^2], if already known.
    longitudinal_rectangles, frame_rectangles : iterable, optional
        Wall elements ``(width, thickness)`` of one member [m], from which
        ``GJ`` is computed with :func:`thin_walled_torsion_constant`.
    longitudinal_support_span, frame_support_span : float, optional
        Override the default (structurally implied) support spans [m].
    eps : float
        Regularisation matching the solver.

    Returns
    -------
    PanelRestraintCalibration
        With ``solver_kwargs()`` giving ``beta_x`` (longitudinals) and
        ``beta_y`` (frames) for
        :func:`tensorfem.plate_elastic_restraint.solve_restrained_panel`.
    """
    for name, value in (
        ("thickness", thickness),
        ("side_x", side_x),
        ("side_y", side_y),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}")

    D = flexural_rigidity(young, poisson, thickness)
    gj_long = _member_rigidity(
        "longitudinal", longitudinal_torsion_rigidity, longitudinal_rectangles, young, poisson
    )
    gj_frame = _member_rigidity(
        "frame", frame_torsion_rigidity, frame_rectangles, young, poisson
    )

    span_long = float(longitudinal_support_span) if longitudinal_support_span is not None else float(side_y)
    span_frame = float(frame_support_span) if frame_support_span is not None else float(side_x)

    longitudinal = _edge_restraint(
        "longitudinal", gj_long, span_long, float(side_x), D, eps
    )
    frame = _edge_restraint("frame", gj_frame, span_frame, float(side_y), D, eps)

    warnings: list[str] = []
    for edge in (longitudinal, frame):
        if edge.zeta >= 1.0 / eps:
            warnings.append(
                f"{edge.member}: zeta={edge.zeta:.4g} saturates the beta "
                f"parameterisation (limit 1/eps={1.0 / eps:.0e}); beta pinned to 1 "
                f"and the excess stiffness is not representable"
            )
        elif edge.beta > 0.95:
            warnings.append(
                f"{edge.member}: beta={edge.beta:.4f} -> effectively clamped "
                f"(moment fixity {edge.moment_fixity:.4f})"
            )
        if edge.zeta < 0.05:
            warnings.append(
                f"{edge.member}: zeta={edge.zeta:.4g} -> effectively simply supported "
                f"(moment fixity {edge.moment_fixity:.3f})"
            )
    for edge in (longitudinal, frame):
        for attr in ("torsion_rigidity", "k_theta", "zeta", "beta", "moment_fixity"):
            if not math.isfinite(getattr(edge, attr)):
                raise ValueError(f"{edge.member}.{attr} is not finite")

    return PanelRestraintCalibration(
        young=young,
        poisson=poisson,
        thickness=thickness,
        side_x=side_x,
        side_y=side_y,
        D=D,
        longitudinal=longitudinal,
        frame=frame,
        warnings=tuple(warnings),
    )


def sensitivity_sweep(
    *,
    young: float,
    poisson: float,
    thickness: float,
    side_x: float,
    side_y: float,
    longitudinal_variants: Sequence[tuple[str, Sequence[Sequence[float]]]],
    frame_variants: Sequence[tuple[str, Sequence[Sequence[float]]]],
    support_spans: Sequence[float],
    eps: float = _DEFAULT_EPS,
) -> list[dict]:
    """Calibration over stiffener sizes and support spacings (machine readable).

    Every combination of ``support_spans`` (applied to *both* members),
    ``longitudinal_variants`` and ``frame_variants`` is evaluated; the flat
    output is convenient for a table or a JSON artifact.

    Parameters
    ----------
    longitudinal_variants, frame_variants : sequence of (label, rectangles)
        Named wall-element sets, e.g. ``("FB 150x12", [(0.15, 0.012)])``.
    support_spans : sequence of float
        Candidate spacings of the torsionally fixing members [m].

    Returns
    -------
    rows : list of dict
        One row per combination with ``beta_x``/``beta_y`` and the mobilised
        moment fixities.
    """
    rows: list[dict] = []
    for span in support_spans:
        if span <= 0.0:
            raise ValueError(f"support span must be positive, got {span}")
        for long_label, long_rects in longitudinal_variants:
            for frame_label, frame_rects in frame_variants:
                cal = calibrate_panel_restraint(
                    young=young,
                    poisson=poisson,
                    thickness=thickness,
                    side_x=side_x,
                    side_y=side_y,
                    longitudinal_rectangles=long_rects,
                    frame_rectangles=frame_rects,
                    longitudinal_support_span=span,
                    frame_support_span=span,
                    eps=eps,
                )
                rows.append(
                    {
                        "support_span": float(span),
                        "longitudinal": long_label,
                        "frame": frame_label,
                        "k_theta_x": cal.longitudinal.k_theta,
                        "zeta_x": cal.longitudinal.zeta,
                        "beta_x": cal.longitudinal.beta,
                        "moment_fixity_x": cal.longitudinal.moment_fixity,
                        "k_theta_y": cal.frame.k_theta,
                        "zeta_y": cal.frame.zeta,
                        "beta_y": cal.frame.beta,
                        "moment_fixity_y": cal.frame.moment_fixity,
                    }
                )
    return rows