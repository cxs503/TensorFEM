"""Stiffener local instability limit states: web local buckling and tripping.

Why this module exists
----------------------
``grillage_ice_ultimate`` states in its own scope block that *"stiffener web
local buckling / tripping and frame tripping are not modelled"*, and
``CASE-GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE-round6`` §5 note 2 repeats it.
Every limit state of the ice-patch + lateral-pressure + hull-girder-axial chain
is therefore a *plate* state or a *column* state.  A deep slender web under a
localised ice patch is exactly the case in which the stiffener can rotate about
its line of attachment (tripping) or buckle locally before the column strength
is reached.  This module supplies those two states from first principles; no
rule coefficient is transcribed and no catalogue value is quoted.

Section idealisation
--------------------
The **bare tee** (web + flange) is the tripping section.  The continuous plating
is *not* part of it: the plating is the restraining medium -- it blocks lateral
translation of the joint and supplies a rotational spring ``k_theta`` -- so
including its area or its St Venant torsion in the stiffener section would
double-count the restraint and inflate the tripping stress (for a 20 mm plate
at 0.35 m spacing the plate's ``b t^3/3`` would be ~7x the tee's own ``J``).
The plating therefore enters only through ``k_theta`` (and through the web
buckling edge condition).  This is the conservative and the conventional
choice; it is stated here so the reader can see it is not an oversight.

Web local buckling
------------------
The web is a long rectangular plate strip of depth ``h_w``, thickness ``t_w``,
compressed in its long direction, simply supported along the flange and along
the plate::

    sigma_w = k pi^2 D_w / (t_w h_w^2),   D_w = E t_w^3 / (12 (1 - nu^2))

``k = 4`` (both edges simply supported) is the lower bound used as the capacity;
``k = 6.97`` (one edge clamped) is reported as the upper bound of the bracket.
Both are the textbook eigenvalues of the simply-supported / clamped-simply
supported strip.

Tripping
--------
The stiffener is a beam of length ``L`` (frame spacing for a longitudinal,
girder spacing for a frame).  Its joint with the plate cannot translate
laterally -- the plating carries the in-plane shear -- so the only free
kinematic is the rotation ``phi(x)`` about the line of attachment.  With the
axial force ``P`` applied at the section centroid, a distance ``d`` from that
line, the energy per unit length and the work of ``P`` are

    U = 1/2 GJ phi'^2 + 1/2 EI_w phi''^2 + 1/2 k_theta phi^2
    W = P/2 (I_p/A + d^2) phi'^2

with

* ``J``    = St Venant constant of the open section, ``sum b t^3 / 3``,
* ``I_w``  = warping constant about the attachment line.  Taking the sectorial
             coordinate ``omega = int h_perp ds`` with the pole on the joint
             (so ``omega = 0`` along the web midline and along the plating,
             ``omega = h_e y`` along the flange), ``I_w = int (omega - omega_bar)^2 dA``
             is evaluated numerically and cross-checked against the closed form
             ``h_e^2 t_f b_f^3 / 12`` for a monosymmetric tee (gate T14c),
* ``d``    = centroid offset below the plate mid-plane,
* ``I_p``  = ``I_lat + I_strong`` about the centroid; ``I_lat = int y^2 dA`` is
             the second moment resisting the (restrained) lateral motion and
             ``I_strong = int (z - z_c)^2 dA`` the depth bending inertia;
* ``k_theta`` = rotational restraint of the plating per unit length, derived
             (not tabulated) as the edge rotation stiffness of a plate strip of
             tributary width ``b_s/2`` whose far edge is the mid-bay symmetry
             line::

                 k_theta = c D / (b_s / 2),  c = 4 (far edge clamped),
                                             c = 3 (far edge simply supported)

Ritz solution.  With ``phi = b sin(m pi x / L)`` (joint simply supported at the
frames, warping free there) the eigenvalue is closed form, no eigensolver::

    sigma_tr(m) = (GJ k^2 + EI_w k^4 + k_theta) / (A (I_p/A + d^2) k^2),
    k = m pi / L

and the tripping stress is the minimum over ``m``.  Because the basis is exact
for the operator, the ``m = 1`` root reproduces the classical limits exactly --
those limits are this module's verification gates:

* ``k_theta = 0, EI_w = 0``            -> ``sigma = G J A / I_p,total`` (pure torsion),
* ``EI_w`` only                        -> ``sigma = (GJ + pi^2 E I_w / L^2) A / R^2``,
* ``k_theta -> inf``                   -> tripping suppressed (returns ``inf``).

The lateral translation is *not* free; for completeness the module also reports
the free-strut lateral--torsional state (``lateral_translation='free'``, the
2 x 2 generalised problem written out in :func:`tripping_stress`).  For a tee
that state is far softer than tripping (its ``I_lat`` is only the flange's own
``t_f b_f^3 / 12``) and it is **not** reachable by a stiffener welded to
continuous plating -- it is a comparison value, labelled as such everywhere.

Honest scope
------------
* Elastic bifurcation only: no plasticity, no post-buckling reserve, no
  interaction with plate buckling or with the beam-column mode.  The output is a
  *limit state* to be compared with the column state, not a collapse load.
* Warping is taken about the attachment line; the warping/lateral-bending cross
  terms of a full Vlasov solution are neglected (for a monosymmetric tee the
  neglected terms are of order ``(t_f/b_f)^2``).
* End conditions: simply supported, warping free.  Transverse frames provide
  partial warping restraint, so the reported tripping stress is on the low side.
* The ``k_theta`` bracket ``[0, 8D/b_s]`` is reported in full, never collapsed
  into a single "the" tripping stress without the bracket.
* **Known bias of the single-mode sine Ritz for ``k_theta > 0``.** With a plate
  spring the torsion equation acquires exponentially decaying boundary layers of
  length ``(EI_w / k_theta)^(1/4)`` at the ends, which the global sine basis
  cannot represent; the sine Ritz therefore *overestimates* the restrained
  tripping stress (non-conservative).  The ``k_theta = 0`` value is exact (no
  boundary layer) and is the value the capacity check uses; the restrained ends of
  the bracket are labelled ``single_mode_ritz`` in the output and are cross-checked
  in ``benchmarks/grillage_local_instability.py`` (gate T14b3) against a converged
  Hermite finite-element eigenvalue solve, which is the value to quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "TeeTorsionSection",
    "plate_rotational_restraint",
    "web_local_buckling_stress",
    "plate_buckling_coefficient",
    "tripping_stress",
    "stiffener_instability_check",
]

_K_SIMPLY_SUPPORTED = 4.0
_K_ONE_EDGE_CLAMPED = 6.97


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def _non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite")
    return value


# --------------------------------------------------------------------------- #
# section properties needed by the torsional limit states
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TeeTorsionSection:
    """Bare tee stiffener (web + flange) with properties about its joint.

    The tee hangs below the plate (negative ``z``); the attachment line is the
    plate mid-plane ``z = 0``.  This matches the sign convention of
    ``grillage_ice_ultimate.MemberSection`` so a section can be built from the
    same parameters, but the attached plate strip is deliberately **not** part of
    the tripping section (see the module docstring).
    """

    web_height: float
    web_thickness: float
    flange_width: float
    flange_thickness: float
    young: float = 2.06e11
    poisson: float = 0.3
    shear_modulus: float | None = None
    warping_samples: int = 64

    def __post_init__(self) -> None:
        _positive("web_height", self.web_height)
        _positive("web_thickness", self.web_thickness)
        _positive("flange_width", self.flange_width)
        _positive("flange_thickness", self.flange_thickness)
        _positive("young", self.young)
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1, 0.5)")
        if self.shear_modulus is not None:
            _positive("shear_modulus", self.shear_modulus)
        if self.flange_width < self.web_thickness:
            raise ValueError("flange_width must not be narrower than web_thickness")
        if int(self.warping_samples) < 4:
            raise ValueError("warping_samples must be >= 4")

    # ---- elastic properties ---------------------------------------------- #
    @property
    def shear_modulus_value(self) -> float:
        if self.shear_modulus is not None:
            return float(self.shear_modulus)
        return self.young / (2.0 * (1.0 + self.poisson))

    @property
    def area(self) -> float:
        return (self.web_height * self.web_thickness
                + self.flange_width * self.flange_thickness)

    @property
    def centroid_z(self) -> float:
        """Centroid offset below the plate mid-plane (positive downward)."""
        web = self.web_height * self.web_thickness
        flange = self.flange_width * self.flange_thickness
        z_web = 0.5 * self.web_height
        z_flange = self.web_height + 0.5 * self.flange_thickness
        return (web * z_web + flange * z_flange) / (web + flange)

    @property
    def inertia_lateral(self) -> float:
        """``I_lat = int y^2 dA`` about the centroidal axis normal to the web."""
        t_w, b_f, t_f = self.web_thickness, self.flange_width, self.flange_thickness
        return (self.web_height * t_w ** 3 + t_f * b_f ** 3) / 12.0

    @property
    def inertia_strong(self) -> float:
        """``I_strong = int (z - z_c)^2 dA`` (bending through the section depth)."""
        zc = self.centroid_z
        web = self.web_height * self.web_thickness
        flange = self.flange_width * self.flange_thickness
        own_web = self.web_thickness * self.web_height ** 3 / 12.0
        own_flange = self.flange_width * self.flange_thickness ** 3 / 12.0
        arm_web = (0.5 * self.web_height - zc) ** 2
        arm_flange = (self.web_height + 0.5 * self.flange_thickness - zc) ** 2
        return own_web + web * arm_web + own_flange + flange * arm_flange

    @property
    def polar_inertia(self) -> float:
        return self.inertia_lateral + self.inertia_strong

    @property
    def polar_radius_squared(self) -> float:
        """``I_p/A + d^2`` -- the radius of gyration about the attachment line."""
        return self.polar_inertia / self.area + self.centroid_z ** 2

    @property
    def saint_venant_constant(self) -> float:
        """``J = sum b t^3 / 3`` over the two open rectangles."""
        return (self.web_height * self.web_thickness ** 3
                + self.flange_width * self.flange_thickness ** 3) / 3.0

    @property
    def flange_depth_below_joint(self) -> float:
        """``h_e``: depth of the flange midline below the plate mid-plane."""
        return self.web_height + 0.5 * self.flange_thickness

    @property
    def warping_constant(self) -> float:
        """``I_w = int (omega - omega_bar)^2 dA`` about the attachment line.

        ``omega = int h_perp ds`` with the pole at the joint: the web midline and
        the plating pass through the pole (``h_perp = 0``), the flange midline is
        at depth ``h_e`` so ``omega = h_e y``.  Integrated numerically over the
        flange so that a future non-rectangular part can be substituted without
        re-deriving; the monosymmetric closed form is
        :attr:`warping_constant_closed_form` and the two are gated (T14c).
        """
        n = int(self.warping_samples)
        b, t = self.flange_width, self.flange_thickness
        dA = b * t / n
        h_e = self.flange_depth_below_joint
        s1 = s2 = 0.0
        for i in range(n):
            y = -0.5 * b + (i + 0.5) * b / n
            w = h_e * y
            s1 += w * dA
            s2 += w * w * dA
        return max(s2 - s1 * s1 / self.area, 0.0)

    @property
    def warping_constant_closed_form(self) -> float:
        """``h_e^2 t_f b_f^3 / 12`` (monosymmetric tee, pole on the joint)."""
        return (self.flange_depth_below_joint ** 2
                * self.flange_thickness * self.flange_width ** 3 / 12.0)

    def report(self) -> dict[str, float]:
        return {
            "area_m2": self.area,
            "centroid_below_plate_m": self.centroid_z,
            "inertia_lateral_m4": self.inertia_lateral,
            "inertia_strong_m4": self.inertia_strong,
            "polar_inertia_m4": self.polar_inertia,
            "polar_radius_squared_m2": self.polar_radius_squared,
            "saint_venant_constant_m4": self.saint_venant_constant,
            "warping_constant_m6": self.warping_constant,
            "warping_constant_closed_form_m6": self.warping_constant_closed_form,
            "shear_modulus_Pa": self.shear_modulus_value,
            "web_depth_thickness_ratio": self.web_height / self.web_thickness,
        }


# --------------------------------------------------------------------------- #
# plate rotational restraint
# --------------------------------------------------------------------------- #
def plate_rotational_restraint(*, plate_thickness: float, stiffener_spacing: float,
                               young: float = 2.06e11, poisson: float = 0.3,
                               far_edge: str = "clamped") -> float:
    """Edge rotation stiffness ``k_theta`` [N] the plating gives a stiffener.

    A strip of tributary width ``b_s/2`` bends about its transverse axis: the near
    edge is the stiffener line (rotation ``theta``), the far edge is the mid-bay
    symmetry line, which cannot rotate (``far_edge='clamped'``) or, as a lower
    bound, is simply supported (``far_edge='pinned'``).  Beam solution per unit
    length::

        M = c D theta / (b_s / 2),  c = 4 (clamped) / 3 (pinned),
        D = E t_p^3 / (12 (1 - nu^2))
    """
    t = _positive("plate_thickness", plate_thickness)
    b_s = _positive("stiffener_spacing", stiffener_spacing)
    e = _positive("young", young)
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")
    if far_edge not in ("clamped", "pinned"):
        raise ValueError("far_edge must be 'clamped' or 'pinned'")
    d = e * t ** 3 / (12.0 * (1.0 - poisson ** 2))
    c = 4.0 if far_edge == "clamped" else 3.0
    return c * d / (0.5 * b_s)


# --------------------------------------------------------------------------- #
# web local buckling
# --------------------------------------------------------------------------- #
def plate_buckling_coefficient(*, web_height: float, panel_length: float,
                               modes: int = 200) -> float:
    """``k = min_m (m h/a + a/(m h))^2`` for a long simply supported strip.

    The classical buckling coefficient of a rectangular plate (width ``h`` =
    web depth, length ``a`` = unsupported stiffener length) in longitudinal
    compression.  For ``a/h`` an integer this returns exactly ``4``; otherwise it
    is the (slightly larger) exact minimum, which is what
    :func:`web_local_buckling_stress` uses when the panel length is supplied.
    """
    h = _positive("web_height", web_height)
    a = _positive("panel_length", panel_length)
    if int(modes) < 1:
        raise ValueError("modes must be >= 1")
    ratio = a / h
    return min((m / ratio + ratio / m) ** 2 for m in range(1, int(modes) + 1))


def web_local_buckling_stress(*, web_height: float, web_thickness: float,
                              young: float = 2.06e11, poisson: float = 0.3,
                              edge_condition: str = "simply_supported",
                              panel_length: float | None = None) -> float:
    """Elastic local buckling stress of the stiffener web strip.

    ``edge_condition='simply_supported'`` uses ``k = 4``; when the unsupported
    length is known (``panel_length``) the exact aspect-ratio coefficient
    :func:`plate_buckling_coefficient` is used instead (still >= 4).
    ``edge_condition='one_edge_clamped'`` uses ``k = 6.97`` (upper bound).
    """
    h = _positive("web_height", web_height)
    t = _positive("web_thickness", web_thickness)
    e = _positive("young", young)
    if not (-1.0 < poisson < 0.5):
        raise ValueError("poisson must lie in (-1, 0.5)")
    if edge_condition not in ("simply_supported", "one_edge_clamped"):
        raise ValueError("edge_condition must be 'simply_supported' or 'one_edge_clamped'")
    if edge_condition == "simply_supported":
        k = _K_SIMPLY_SUPPORTED
        if panel_length is not None:
            k = plate_buckling_coefficient(web_height=h, panel_length=panel_length)
            k = max(k, _K_SIMPLY_SUPPORTED)
    else:
        k = _K_ONE_EDGE_CLAMPED
    d = e * t ** 3 / (12.0 * (1.0 - poisson ** 2))
    return k * math.pi ** 2 * d / (t * h ** 2)


# --------------------------------------------------------------------------- #
# tripping
# --------------------------------------------------------------------------- #
def _torsional_reference(section: TeeTorsionSection, span: float,
                         k_theta: float) -> float:
    """Closed-form torsional branch -- the independent oracle for gate T14a/b."""
    k = math.pi / span
    num = (section.saint_venant_constant * section.shear_modulus_value
           + section.young * section.warping_constant * k ** 2 + k_theta)
    return num / (section.area * section.polar_radius_squared)


def _free_strut_roots(section: TeeTorsionSection, k: float, k22: float) -> list[float]:
    """Positive axial-force roots of the free-strut 2 x 2 problem.

    ``K = [[EI_lat k^4, 0], [0, k22]]``,
    ``G = k^2 [[1, d], [d, R^2]]`` with ``R^2 = I_p/A + d^2``;
    ``det(K - P G) = 0`` is the quadratic ``A P^2 + B P + C = 0`` solved here.
    """
    e = section.young
    k2, k4 = k * k, k ** 4
    k11 = e * section.inertia_lateral * k4
    d = section.centroid_z
    r2 = section.polar_radius_squared
    g11, g22, g12 = k2, r2 * k2, d * k2
    qa = g11 * g22 - g12 * g12
    qb = -(k11 * g22 + k22 * g11)
    qc = k11 * k22
    if abs(qa) < 1e-300:
        return [-qc / qb] if abs(qb) > 1e-300 else []
    disc = qb * qb - 4.0 * qa * qc
    if disc < 0.0:
        return []
    sq = math.sqrt(disc)
    return [(-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa)]


def tripping_stress(section: TeeTorsionSection, *, span: float,
                    rotational_restraint: float = 0.0,
                    lateral_translation: str = "restrained",
                    modes: int = 1) -> dict[str, object]:
    """Tripping stress of one stiffener, minimum positive root over the modes.

    Parameters
    ----------
    span
        Unsupported length (frame spacing for a longitudinal, girder spacing for
        a frame).
    rotational_restraint
        ``k_theta`` [N] from :func:`plate_rotational_restraint`; ``0`` means the
        plating gives no rotational restraint, which is the lower bound on the
        tripping stress and the value the design check uses.
    lateral_translation
        ``'restrained'`` (default): the plating blocks lateral translation, only
        the rotation is free -- the classical tripping mode.
        ``'free'``: lateral--torsional buckling of a free strut (comparison
        value only, see the module docstring).
    modes
        Number of half-wave modes searched; with the exact sine basis the
        fundamental governs for a prismatic member under uniform compression, so
        ``1`` is the default and larger values exist so a caller can demonstrate
        that (gate T14d).
    """
    L = _positive("span", span)
    k_theta = _non_negative("rotational_restraint", rotational_restraint)
    if int(modes) < 1:
        raise ValueError("modes must be >= 1")
    if lateral_translation not in ("restrained", "free"):
        raise ValueError("lateral_translation must be 'restrained' or 'free'")

    e = section.young
    area = section.area
    reference = _torsional_reference(section, L, k_theta)
    best: dict[str, object] | None = None
    for m in range(1, int(modes) + 1):
        k = m * math.pi / L
        k22 = (section.saint_venant_constant * section.shear_modulus_value * k * k
               + e * section.warping_constant * k ** 4 + k_theta)
        if lateral_translation == "restrained":
            roots = [k22 / (k * k * section.polar_radius_squared)]
        else:
            roots = _free_strut_roots(section, k, k22)
        positive = [r for r in roots if math.isfinite(r) and r > 0.0]
        if not positive:
            continue
        p = min(positive)
        if best is None or p < float(best["axial_force_N"]):
            best = {"mode": m, "axial_force_N": p, "half_wave_length_m": L / m}
    if best is None:
        return {
            "sigma_cr_Pa": math.inf,
            "mode": None,
            "axial_force_N": math.inf,
            "tripping_suppressed": True,
            "rotational_restraint_N": k_theta,
            "reference_torsional_sigma_Pa": reference,
            "span_m": L,
            "modes_searched": int(modes),
            "lateral_translation": lateral_translation,
            "sigma_cr_over_yield": None,
        }
    return {
        "sigma_cr_Pa": float(best["axial_force_N"]) / area,
        "mode": best["mode"],
        "axial_force_N": best["axial_force_N"],
        "half_wave_length_m": best["half_wave_length_m"],
        "tripping_suppressed": False,
        "rotational_restraint_N": k_theta,
        "reference_torsional_sigma_Pa": reference,
        "span_m": L,
        "modes_searched": int(modes),
        "lateral_translation": lateral_translation,
        "lateral_euler_sigma_Pa": (math.pi ** 2 * e * section.inertia_lateral
                                   / (area * L ** 2)),
    }


# --------------------------------------------------------------------------- #
# combined local-instability limit state
# --------------------------------------------------------------------------- #
def stiffener_instability_check(section: TeeTorsionSection, *, span: float,
                                stiffener_spacing: float,
                                plate_thickness: float = 0.0,
                                yield_stress: float | None = None,
                                lateral_pressure: float = 0.0) -> dict[str, object]:
    """Minimum of web local buckling and the bracketed tripping stress.

    The plate rotational restraint is bracketed: ``k_theta = 0`` (plating fully
    yielded locally / no restraint) up to ``8 D / b_s`` (mid-bay symmetry
    restraint at the caller's stiffener spacing), with ``6 D / b_s`` (far edge
    simply supported) in between.  The **lower** end controls the capacity; the
    whole bracket is reported so its width is visible.  ``plate_thickness = 0``
    means "no plating", i.e. ``k_theta = 0`` throughout.

    ``lateral_pressure`` is recorded for interface symmetry with the combined
    load case only: the elastic bifurcation stress of a prismatic member does not
    depend on a uniform lateral load (the pressure raises the compression through
    the panel bending, and that rise comes from the grillage load model).
    """
    L = _positive("span", span)
    spacing = _positive("stiffener_spacing", stiffener_spacing)
    t_p = _non_negative("plate_thickness", plate_thickness)
    p = _non_negative("lateral_pressure", lateral_pressure)

    web = {
        "simply_supported_Pa": web_local_buckling_stress(
            web_height=section.web_height, web_thickness=section.web_thickness,
            young=section.young, poisson=section.poisson, panel_length=L),
        "one_edge_clamped_Pa": web_local_buckling_stress(
            web_height=section.web_height, web_thickness=section.web_thickness,
            young=section.young, poisson=section.poisson,
            edge_condition="one_edge_clamped", panel_length=L),
        "buckling_coefficient": plate_buckling_coefficient(
            web_height=section.web_height, panel_length=L),
    }
    web["min_Pa"] = min(web["simply_supported_Pa"], web["one_edge_clamped_Pa"])

    if t_p > 0.0:
        k_theta_max = plate_rotational_restraint(
            plate_thickness=t_p, stiffener_spacing=spacing,
            young=section.young, poisson=section.poisson, far_edge="clamped")
        k_theta_pinned = plate_rotational_restraint(
            plate_thickness=t_p, stiffener_spacing=spacing,
            young=section.young, poisson=section.poisson, far_edge="pinned")
    else:
        k_theta_max = 0.0
        k_theta_pinned = 0.0

    free = tripping_stress(section, span=L, rotational_restraint=0.0)
    pinned = tripping_stress(section, span=L, rotational_restraint=k_theta_pinned)
    clamped = tripping_stress(section, span=L, rotational_restraint=k_theta_max)
    free_strut = tripping_stress(section, span=L, rotational_restraint=0.0,
                                lateral_translation="free")

    sigma_free = float(free["sigma_cr_Pa"])
    sigma_pinned = float(pinned["sigma_cr_Pa"])
    sigma_clamped = float(clamped["sigma_cr_Pa"])
    sigma_low = min(sigma_free, sigma_pinned)
    sigma_high = max(sigma_free, sigma_pinned, sigma_clamped)
    limiting = min(web["min_Pa"], sigma_low)

    out: dict[str, object] = {
        "span_m": L,
        "stiffener_spacing_m": spacing,
        "plate_thickness_m": t_p,
        "lateral_pressure_Pa": p,
        "plate_rotational_restraint_free_N": 0.0,
        "plate_rotational_restraint_pinned_N": k_theta_pinned,
        "plate_rotational_restraint_clamped_N": k_theta_max,
        "tripping_sigma_free_Pa": sigma_free,
        "tripping_sigma_pinned_Pa": sigma_pinned,
        "tripping_sigma_clamped_Pa": sigma_clamped,
        "tripping_sigma_low_Pa": sigma_low,
        "tripping_sigma_high_Pa": sigma_high,
        "restrained_values_are_single_mode_ritz": True,
        "restrained_bias_note": (
            "tripping_sigma_pinned/clamped come from the single-mode sine Ritz and "
            "are UPPER bounds: the true restrained value is lower because of the "
            "torsional boundary layer at the ends (see the module docstring and "
            "the Hermite FE cross-check in benchmarks/grillage_local_instability.py)"),
        "tripping_mode": free.get("mode"),
        "tripping_suppressed_when_clamped": bool(clamped["tripping_suppressed"]),
        "tripping_sigma_free_strut_Pa": float(free_strut["sigma_cr_Pa"]),
        "tripping_sigma_free_strut_note": (
            "lateral--torsional buckling with lateral translation free; the "
            "continuous plating blocks that translation, so this is a comparison "
            "value, not a design value"),
        "web_local_buckling": web,
        "limiting_stress_Pa": limiting,
        "governing": ("web_local_buckling" if web["min_Pa"] <= sigma_low
                      else "tripping"),
        "section": section.report(),
    }
    if yield_stress is not None:
        sy = _positive("yield_stress", yield_stress)
        out["yield_stress_Pa"] = sy
        out["limiting_over_yield"] = limiting / sy
        out["tripping_free_over_yield"] = sigma_free / sy
        out["tripping_clamped_over_yield"] = (sigma_clamped / sy
                                              if math.isfinite(sigma_clamped) else None)
    return out