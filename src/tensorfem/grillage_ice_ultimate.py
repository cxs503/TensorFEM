"""Grillage ultimate strength under a combined ice pressure patch + lateral pressure.

Why this module exists
----------------------
``tensorfem.stiffened_panel_ice_ultimate`` answers "what *uniform* ice pressure
does one stiffened panel bay carry before it collapses".  An ice-class
structural assessment does not stop there: the ice load is a **local patch**
spilling over several bays and carried partly by the longitudinal stiffeners and
partly by the transverse frames, while the **hydrostatic lateral pressure** acts
over the whole shell and the hull girder supplies a coexisting **in-plane
compression**.  That is a grillage problem, not a panel problem, and it is the
"panel stress -> structural limit" link missing from the ice-belt chain.

This module is the semi-analytical grillage layer.  It is deliberately *not*
another nonlinear FE solver: the heavy elastoplastic FE already exists in
``ice_belt_collapse`` / ``stiffened_panel_ice_ultimate`` and is used as the
independent cross-check.  What is added is

1. an **orthotropic-plate (Navier) response** for a rectangular ice pressure
   patch superimposed on a uniform lateral pressure, giving plate bending
   moments and the modal **load share between the x-members (longitudinal
   stiffeners) and the y-members (transverse frames)**;
2. a **member-level ultimate criterion set**: (a) local plate bending
   interaction, (b) longitudinal stiffener beam-column with lateral bending
   amplified by the exact secant factor, (c) transverse frame rigid-plastic
   collapse under the patch;
3. a **panel stress -> ultimate patch pressure map**, solved by bisection on the
   ice pressure, with governing mode, member utilisations and an optional safety
   margin against a caller-supplied rule design pressure.

Verification anchors (see ``grillage_ice_verification.run_grillage_verification``)
----------------------------------------------------------------------------------
* ``T1`` isotropic SSSS plate, uniform load: ``w_max = 0.00406 q a^4 / D`` and
  ``M_max = 0.0479 q a^2`` (Timoshenko & Woinowsky-Krieger, *Theory of Plates
  and Shells*, Table 30) -- gate 3 %.
* ``T2`` patch-load Navier series convergence (terms 31 vs 121) -- gate 0.5 %.
* ``T3`` isotropic plate load share ``f_x = f_y`` and ``f_x + f_y + f_t = 1``.
* ``T4`` orthotropy limit: ``Dx/Dy = 1e6`` drives ``f_x -> 1``.
* ``T5`` exact beam-column amplification ``2(sec u - 1)/u^2`` versus the
  ``1/(1-P/Pe)`` approximation.
* ``T6`` transverse frame rigid-plastic collapse: uniform load reproduces the
  closed-form ``16 Mp / L``; the patch formula degenerates to ``8 Mp / L`` as
  ``c -> 0``.
* ``T7`` axial-only grillage ultimate stress versus the Paik-Thayamballi
  stiffened-panel strength already verified in ``stiffened_panel_ultimate``
  (the 15 % literature/empirical gate).
* ``T8`` monotonicity/limits: ultimate ice pressure decreases with coexisting
  compression and with hydrostatic pressure.

Model content and provenance
----------------------------
* Lengths in metres, stresses and pressures in pascals, plate moments in N m per
  metre, member moments in N m.  No rule table is transcribed: the design
  pressure entering the margin output is caller supplied with a provenance
  string (same convention as ``ice_belt_collapse.IACSPressureArea``).
* Material: elastic-perfectly-plastic steel (``yield_stress``); von Mises enters
  through the section interaction expressions documented below.

Scope and limitations (explicit)
--------------------------------
* Small-deflection plate theory for the lateral response; the *axial* capacity
  comes from the effective-width beam-column law, not from a full
  load-shortening path, so post-buckling reserve beyond the peak is absent.
* Stiffener web local buckling / tripping and frame tripping are not modelled.
* Welding residual stress and initial deflection enter only through the
  effective-width law and ``imperfection_ratio``.
* The ice patch is quasi-static and uniform over its rectangle; other patch
  positions are available through the fractional extents argument.
* Load sharing is the *elastic* modal partition: once a member yields, load
  redistribution is not traced.  For ductile grillages this makes the ultimate
  load conservative (quantified in the case study).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .ice_belt_collapse import IACSPressureArea

__all__ = [
    "MemberSection",
    "GrillageGeometry",
    "CombinedLoadCase",
    "plate_patch_response",
    "beam_column_amplification_exact",
    "beam_column_amplification_approx",
    "beam_moment_centre_patch",
    "frame_plastic_collapse_force",
    "johnson_ostenfeld",
    "stiffener_axial_capacity",
    "system_load_shares",
    "grillage_utilisations",
    "ultimate_ice_pressure",
    "panel_stress_ultimate_map",
    "write_ultimate_map",
    "ice_pressure_area_rule",
    "case_study_geometry",
]


# --------------------------------------------------------------------------- #
# validation helpers (same style as the other marine modules)
# --------------------------------------------------------------------------- #
def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def _finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    return value


def _fraction(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must lie in (0, 1], got {value!r}")
    return value


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MemberSection:
    """Tee section (web + flange) built on an attached plate strip.

    The attached plate strip lies at ``z = 0``; the tee hangs below it
    (negative ``z``), matching the convention of ``stiffened_panel_ultimate``.
    """

    web_height: float
    web_thickness: float
    flange_width: float
    flange_thickness: float
    plate_thickness: float = 0.0
    plate_width: float = 0.0

    def validate(self) -> "MemberSection":
        _positive("web_height", self.web_height)
        _positive("web_thickness", self.web_thickness)
        _positive("flange_width", self.flange_width)
        _positive("flange_thickness", self.flange_thickness)
        _non_negative("plate_thickness", self.plate_thickness)
        _non_negative("plate_width", self.plate_width)
        if self.flange_width < self.web_thickness:
            raise ValueError("flange_width must not be narrower than web_thickness")
        if self.plate_thickness > 0.0 and self.plate_width <= 0.0:
            raise ValueError("an attached plate needs a positive plate_width")
        return self

    @property
    def web_area(self) -> float:
        return self.web_height * self.web_thickness

    @property
    def flange_area(self) -> float:
        return self.flange_width * self.flange_thickness

    @property
    def attached_area(self) -> float:
        return self.plate_width * self.plate_thickness

    @property
    def area(self) -> float:
        return self.web_area + self.flange_area + self.attached_area

    def _parts(self) -> list[tuple[float, float, float]]:
        """``[(area, z_from_plate_mid_plane, own_inertia)]``."""
        parts: list[tuple[float, float, float]] = []
        if self.attached_area > 0.0:
            parts.append((self.attached_area, 0.0,
                          self.plate_width * self.plate_thickness ** 3 / 12.0))
        parts.append((self.web_area, -0.5 * self.web_height,
                      self.web_thickness * self.web_height ** 3 / 12.0))
        parts.append((self.flange_area, -self.web_height - 0.5 * self.flange_thickness,
                      self.flange_width * self.flange_thickness ** 3 / 12.0))
        return parts

    @property
    def centroid_offset(self) -> float:
        """Centroid below the plate mid-plane (positive downward)."""
        parts = self._parts()
        total = sum(a for a, _, _ in parts)
        return -sum(a * z for a, z, _ in parts) / total

    @property
    def inertia(self) -> float:
        """Second moment about the composite centroid."""
        parts = self._parts()
        total = sum(a for a, _, _ in parts)
        zc = sum(a * z for a, z, _ in parts) / total
        return sum(a * (z - zc) ** 2 + i0 for a, z, i0 in parts)

    @property
    def extreme_fibre(self) -> float:
        parts = self._parts()
        total = sum(a for a, _, _ in parts)
        zc = sum(a * z for a, z, _ in parts) / total
        return max(max(abs(z - zc) for _, z, _ in parts), abs(zc))

    @property
    def elastic_modulus_section(self) -> float:
        return self.inertia / self.extreme_fibre

    @property
    def plastic_modulus(self) -> float:
        """Rigid-plastic modulus about the equal-area axis (stacked rectangles)."""
        strips: list[tuple[float, float, float]] = []  # (area, z_lo, z_hi)
        if self.attached_area > 0.0:
            t = self.plate_thickness
            strips.append((self.attached_area, -0.5 * t, 0.5 * t))
        n_sub = 100
        dz = self.web_height / n_sub
        for i in range(n_sub):
            z_hi = -i * dz
            strips.append((self.web_thickness * dz, z_hi - dz, z_hi))
        z_hi = -self.web_height
        strips.append((self.flange_area, z_hi - self.flange_thickness, z_hi))

        total = sum(a for a, _, _ in strips)
        z_lo = min(lo for _, lo, _ in strips)
        z_up = max(hi for _, _, hi in strips)
        for _ in range(80):
            mid = 0.5 * (z_lo + z_up)
            below = 0.0
            for a, lo, hi in strips:
                if hi <= mid:
                    below += a
                elif lo < mid:
                    below += a * (mid - lo) / (hi - lo)
            if below < 0.5 * total:
                z_lo = mid
            else:
                z_up = mid
        z_p = 0.5 * (z_lo + z_up)
        moment = 0.0
        for a, lo, hi in strips:
            if hi <= z_p or lo >= z_p:
                moment += a * abs(0.5 * (lo + hi) - z_p)
            else:
                h1 = z_p - lo
                h2 = hi - z_p
                moment += 0.5 * a * (h1 ** 2 + h2 ** 2) / (hi - lo)
        return moment


# --------------------------------------------------------------------------- #
# geometry and load case
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GrillageGeometry:
    """Orthogonally stiffened grillage: plate + longitudinal stiffeners + frames.

    Coordinates: ``x`` runs fore-aft along the longitudinal stiffeners, ``y``
    across.  Longitudinal stiffeners span ``frame_spacing`` in ``x`` between
    transverse frames; transverse frames span ``n_bays * bay_spacing`` in ``y``
    between the longitudinal girders.  The representative loaded panel is one bay
    wide (``bay_spacing``) and one frame space long (``frame_spacing``).
    """

    frame_spacing: float
    bay_spacing: float
    n_bays: int
    n_spans: int
    plate_thickness: float
    stiffener: MemberSection = field(
        default_factory=lambda: MemberSection(0.25, 0.010, 0.080, 0.012))
    frame: MemberSection = field(
        default_factory=lambda: MemberSection(0.55, 0.010, 0.120, 0.014))
    young: float = 2.06e11
    poisson: float = 0.3
    yield_stress: float = 3.55e8
    imperfection_ratio: float = 1.0 / 750.0

    def validate(self) -> "GrillageGeometry":
        _positive("frame_spacing", self.frame_spacing)
        _positive("bay_spacing", self.bay_spacing)
        _positive("plate_thickness", self.plate_thickness)
        _positive("young", self.young)
        _positive("yield_stress", self.yield_stress)
        _non_negative("imperfection_ratio", self.imperfection_ratio)
        if int(self.n_bays) < 1 or int(self.n_spans) < 1:
            raise ValueError("n_bays and n_spans must be >= 1")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must lie in (-1, 0.5)")
        self.stiffener.validate()
        self.frame.validate()
        if self.stiffener.web_height >= self.frame_spacing:
            raise ValueError("stiffener web_height must be smaller than frame_spacing")
        if self.frame.web_height >= self.n_bays * self.bay_spacing:
            raise ValueError("frame web_height must be smaller than the frame span")
        return self

    @property
    def n_stiffeners(self) -> int:
        return int(self.n_bays) + 1

    @property
    def n_frames(self) -> int:
        return int(self.n_spans) + 1

    @property
    def panel_area(self) -> float:
        return self.frame_spacing * self.bay_spacing

    @property
    def frame_span(self) -> float:
        return self.n_bays * self.bay_spacing

    @property
    def grillage_area(self) -> float:
        return self.frame_spacing * self.n_spans * self.frame_span

    @property
    def plate_rigidity(self) -> float:
        return self.young * self.plate_thickness ** 3 / (12.0 * (1.0 - self.poisson ** 2))

    @property
    def torsional_rigidity(self) -> float:
        g = self.young / (2.0 * (1.0 + self.poisson))
        return g * self.plate_thickness ** 3 / 12.0

    def with_plate(self, section: MemberSection, tributary: float) -> MemberSection:
        """Attach a plate strip of width ``tributary`` to ``section``."""
        return MemberSection(
            web_height=section.web_height,
            web_thickness=section.web_thickness,
            flange_width=section.flange_width,
            flange_thickness=section.flange_thickness,
            plate_thickness=self.plate_thickness,
            plate_width=tributary,
        ).validate()

    @property
    def stiffener_section(self) -> MemberSection:
        return self.with_plate(self.stiffener, self.bay_spacing)

    @property
    def frame_attached_width(self) -> float:
        """Effective plating attached to a frame: ``min(a, 0.4 L_frame)``.

        The ``0.4 L`` shear-lag limit is the standard effective-width cap for a
        transverse member; without it a deep frame in a narrow grillage would be
        credited with its whole frame spacing of plating.
        """
        return min(self.frame_spacing, 0.4 * self.frame_span)

    @property
    def frame_section(self) -> MemberSection:
        return self.with_plate(self.frame, self.frame_attached_width)

    def plate_rigidities(self) -> tuple[float, float, float]:
        """``(Dx, Dy, D3)`` of the smeared orthotropic plate equivalent.

        ``D3 = nu sqrt(Dx Dy) + 2 Dxy`` reduces to ``D`` for the isotropic plate
        (checked by T1/T3).
        """
        dx = self.young * self.stiffener_section.inertia / self.bay_spacing \
            + self.plate_rigidity
        dy = self.young * self.frame_section.inertia / self.frame_spacing \
            + self.plate_rigidity
        d3 = self.poisson * math.sqrt(dx * dy) + 2.0 * self.torsional_rigidity
        return dx, dy, d3

    @property
    def plate_slenderness(self) -> float:
        """``beta = (b/t) sqrt(sigma_y/E)`` of the plate field between stiffeners."""
        return self.bay_spacing / self.plate_thickness * math.sqrt(
            self.yield_stress / self.young)

    @property
    def faulkner_plate_strength(self) -> float:
        """Faulkner effective-width plate strength ratio ``sigma_u / sigma_y``."""
        beta = self.plate_slenderness
        if beta <= 1.0:
            return 1.0
        return min(1.0, 2.0 / beta - 1.0 / beta ** 2)

    @property
    def effective_plate_width(self) -> float:
        return self.faulkner_plate_strength * self.bay_spacing


@dataclass(frozen=True)
class CombinedLoadCase:
    """Ice patch + hydrostatic lateral pressure + in-plane compression.

    ``patch_length``/``patch_width`` are the patch extents in ``x``/``y``,
    clamped to the representative panel.  ``axial_stress`` is the coexisting
    longitudinal compression from the global hull girder (positive in
    compression).
    """

    axial_stress: float = 0.0
    ice_pressure: float = 0.0
    patch_length: float = 0.0
    patch_width: float = 0.0
    hydrostatic_pressure: float = 0.0
    design_pressure: float | None = None
    design_provenance: str = "caller supplied"

    def validate(self, geom: "GrillageGeometry") -> "CombinedLoadCase":
        _non_negative("axial_stress", self.axial_stress)
        _non_negative("ice_pressure", self.ice_pressure)
        _non_negative("patch_length", self.patch_length)
        _non_negative("patch_width", self.patch_width)
        _non_negative("hydrostatic_pressure", self.hydrostatic_pressure)
        if self.design_pressure is not None:
            _positive("design_pressure", self.design_pressure)
        if self.ice_pressure > 0.0 and (self.patch_length <= 0.0 or self.patch_width <= 0.0):
            raise ValueError("a non-zero ice_pressure needs a positive patch size")
        if self.axial_stress >= geom.yield_stress:
            raise ValueError("axial_stress must stay below the yield stress")
        return self

    def patch_extents(self, geom: "GrillageGeometry") -> tuple[float, float]:
        return (min(self.patch_length, geom.frame_spacing),
                min(self.patch_width, geom.bay_spacing))


# --------------------------------------------------------------------------- #
# orthotropic plate: patch response and modal load share (Navier)
# --------------------------------------------------------------------------- #
def plate_patch_response(*, span_x: float, span_y: float, d1: float, d2: float,
                         d3: float, pressure: float,
                         patch: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
                         terms: int = 61, sample: int = 21,
                         poisson: float = 0.3) -> dict[str, float]:
    """Navier solution of an SSSS orthotropic plate under a rectangular patch load.

    ``patch = (fx1, fx2, fy1, fy2)`` are fractional extents of the plate spans.
    Returns sampled peak deflection/moments plus the **modal load share**:
    ``load_share_x`` is the fraction of the patch work stored in the
    ``Dx m^4/a^4`` bending term, ``load_share_y`` in ``Dy n^4/b^4`` and
    ``load_share_torsion`` in the coupling/torsion term.  The shares are
    positive and sum to one for any patch; they are converged with ``terms`` for
    the centred patches used here (see ``grillage_plastic_verification``).  The
    patch may sit anywhere on the plate: the mode set is ``1 .. 2 terms`` so the
    even modes an off-centre patch excites are carried (they are exactly zero
    for a patch centred on the mid-lines).  Peaks are sampled on a
    ``sample x sample`` grid placed on ``linspace(0, span, sample)``; the
    deflection field of a rectangular patch is broad enough that 21 samples
    resolve the peak to well under 1 %, and the grid always contains the plate
    centre.  Pass a larger ``sample`` for a sharper patch.  The three sum to 1 by
    construction; for the isotropic square plate under a uniform load
    ``load_share_x == load_share_y`` (T3) and ``Dx >> Dy`` drives it to 1 (T4).
    """
    _positive("span_x", span_x)
    _positive("span_y", span_y)
    _positive("d1", d1)
    _positive("d2", d2)
    _positive("d3", d3)
    _finite("pressure", pressure)
    if abs(pressure) < 1e-30:
        raise ValueError("pressure must be non-zero")
    fx1, fx2, fy1, fy2 = (float(v) for v in patch)
    for name, v in (("fx1", fx1), ("fx2", fx2), ("fy1", fy1), ("fy2", fy2)):
        if not (-1e-12 <= v <= 1.0 + 1e-12):
            raise ValueError(f"{name} must be a fraction in [0, 1], got {v!r}")
    if not (fx2 > fx1 and fy2 > fy1):
        raise ValueError("patch extents must satisfy fx2 > fx1 and fy2 > fy1")
    terms = int(terms)
    if terms < 3:
        raise ValueError("terms must be >= 3")
    sample = int(sample)
    if sample < 3:
        raise ValueError("sample must be >= 3")

    x1, x2 = fx1 * span_x, fx2 * span_x
    y1, y2 = fy1 * span_y, fy2 * span_y
    load_area = (x2 - x1) * (y2 - y1)

    # Mode set = the odd modes 1 .. 2*terms-1 (the historical symmetric
    # half-basis, so every centred/flat patch keeps its old truncation) *plus*
    # the even modes.  The odd-only basis forces a field symmetric about the
    # plate mid-lines and therefore cannot represent an eccentric patch; the
    # even modes are what an off-centre patch excites (they vanish identically
    # for a patch centred on the mid-lines, so symmetric cases are unchanged).
    m_vec = np.arange(1.0, 2.0 * terms + 1.0, 1.0)
    n_vec = np.arange(1.0, 2.0 * terms + 1.0, 1.0)
    m = m_vec[:, None]
    n = n_vec[None, :]
    cx = np.cos(m * math.pi * x1 / span_x) - np.cos(m * math.pi * x2 / span_x)
    cy = np.cos(n * math.pi * y1 / span_y) - np.cos(n * math.pi * y2 / span_y)
    q_mn = (4.0 * pressure / (math.pi ** 2 * m * n)) * cx * cy

    kx = d1 * m ** 4 / span_x ** 4
    ky = d2 * n ** 4 / span_y ** 4
    kt = 2.0 * d3 * m ** 2 * n ** 2 / (span_x ** 2 * span_y ** 2)
    k_sum = kx + ky + kt
    amp = q_mn / (math.pi ** 4 * k_sum)

    # Load share = share of the patch work stored in the Dx, Dy and D3 terms of
    # the orthotropic plate strain energy.  The modal weight q_mn^2 / k_sum is
    # positive by construction and the three shares sum to one exactly.  (A
    # generalised-force weighting was tried first and is not usable: for a small
    # patch it returns negative, mode-count dependent values -- see the
    # verification note in ``grillage_plastic_collapse``.)
    w_modal = q_mn ** 2 / k_sum
    w_total = float(np.sum(w_modal))
    if w_total <= 0.0:  # pragma: no cover - defensive
        w_total = 1.0
    f_x = float(np.sum(w_modal * (kx / k_sum))) / w_total
    f_y = float(np.sum(w_modal * (ky / k_sum))) / w_total
    f_t = float(np.sum(w_modal * (kt / k_sum))) / w_total

    xs = np.linspace(0.0, span_x, sample)
    ys = np.linspace(0.0, span_y, sample)
    sx = np.sin(np.pi * m_vec[:, None] * xs[None, :] / span_x)
    sy = np.sin(np.pi * n_vec[:, None] * ys[None, :] / span_y)
    ax = (m * math.pi / span_x) ** 2
    ay = (n * math.pi / span_y) ** 2
    d12 = poisson * math.sqrt(d1 * d2)
    w_field = np.einsum("mn,mi,nj->ij", amp, sx, sy)
    m_x = np.einsum("mn,mi,nj->ij", (d1 * ax + d12 * ay) * amp, sx, sy)
    m_y = np.einsum("mn,mi,nj->ij", (d2 * ay + d12 * ax) * amp, sx, sy)

    return {
        "w_max": float(np.max(np.abs(w_field))),
        "m_x_max": float(np.max(np.abs(m_x))),
        "m_y_max": float(np.max(np.abs(m_y))),
        "load_share_x": f_x,
        "load_share_y": f_y,
        "load_share_torsion": f_t,
        "load_area": load_area,
        "total_force": pressure * load_area,
    }


# --------------------------------------------------------------------------- #
# beam-column and frame collapse closed forms
# --------------------------------------------------------------------------- #
def beam_column_amplification_exact(axial_force: float, euler_force: float) -> float:
    """Exact mid-span magnification ``2(sec u - 1)/u^2``, ``u = pi/2 sqrt(P/Pe)``."""
    _non_negative("axial_force", axial_force)
    _positive("euler_force", euler_force)
    ratio = axial_force / euler_force
    if ratio >= 1.0:
        raise ValueError("axial_force must stay below the Euler load")
    u = 0.5 * math.pi * math.sqrt(ratio)
    if u < 1e-8:
        return 1.0
    return 2.0 * (1.0 / math.cos(u) - 1.0) / u ** 2


def beam_column_amplification_approx(axial_force: float, euler_force: float) -> float:
    """Standard ``1 / (1 - P/Pe)`` magnification (approximation to the exact one)."""
    _non_negative("axial_force", axial_force)
    _positive("euler_force", euler_force)
    ratio = axial_force / euler_force
    if ratio >= 1.0:
        raise ValueError("axial_force must stay below the Euler load")
    return 1.0 / (1.0 - ratio)


def beam_moment_centre_patch(total_force: float, span: float, patch_length: float) -> float:
    """SS beam with a central patch: ``M_max = F (2L - c) / 8``."""
    _non_negative("total_force", total_force)
    _positive("span", span)
    _non_negative("patch_length", patch_length)
    c = min(patch_length, span)
    return total_force * (2.0 * span - c) / 8.0


def frame_plastic_collapse_force(plastic_moment: float, span: float,
                                 patch_length: float = 0.0) -> float:
    """Rigid-plastic collapse force of a fixed-fixed beam under a central patch.

    Mechanism: hinges at both ends and at mid-span; virtual work gives
    ``F_u = 8 Mp / (L (1 - c/(2L)))``.  ``c = 0`` reproduces ``8 Mp / L`` (point
    load) and ``c = L`` reproduces ``16 Mp / L`` (uniform load).
    """
    _positive("plastic_moment", plastic_moment)
    _positive("span", span)
    _non_negative("patch_length", patch_length)
    c = min(patch_length, span)
    return 8.0 * plastic_moment / (span * (1.0 - c / (2.0 * span)))


def johnson_ostenfeld(yield_stress: float, euler_stress: float) -> float:
    """Inelastic column limiting stress (Johnson-Ostenfeld form)."""
    _positive("yield_stress", yield_stress)
    _positive("euler_stress", euler_stress)
    if euler_stress >= 0.5 * yield_stress:
        return yield_stress * (1.0 - yield_stress / (4.0 * euler_stress))
    return euler_stress


# --------------------------------------------------------------------------- #
# member capacities
# --------------------------------------------------------------------------- #
def stiffener_axial_capacity(geom: "GrillageGeometry") -> dict[str, float]:
    """Effective-width beam-column capacity of one longitudinal stiffener.

    The plate contributes its Faulkner effective width; the column spans
    ``frame_spacing`` between transverse frames.  ``sigma_u = sigma_L A_e /
    A_gross`` is the axial ultimate stress of the panel strip (compared with the
    Paik empirical strength in T7).
    """
    geom.validate()
    be = geom.effective_plate_width
    section = geom.with_plate(geom.stiffener, be)
    gross = geom.with_plate(geom.stiffener, geom.bay_spacing)
    area_e = section.area
    inertia_e = section.inertia
    sigma_e = math.pi ** 2 * geom.young * inertia_e / (area_e * geom.frame_spacing ** 2)
    # Perry-Robertson imperfection reduction next to Johnson-Ostenfeld
    e0 = geom.imperfection_ratio * geom.frame_spacing
    c = section.extreme_fibre
    sigma_y = geom.yield_stress
    eta = e0 * c * area_e / inertia_e
    disc = (sigma_y + sigma_e * (1.0 + eta)) ** 2 - 4.0 * sigma_y * sigma_e
    sigma_pr = (sigma_y + sigma_e * (1.0 + eta) - math.sqrt(max(disc, 0.0))) / 2.0
    sigma_jo = johnson_ostenfeld(sigma_y, sigma_e)
    # plate buckling enters through the effective width A_e; capping sigma_L by
    # the plate strength ratio as well would knock the plate down twice
    sigma_l = min(sigma_pr, sigma_jo)
    return {
        "effective_width": be,
        "effective_width_ratio": be / geom.bay_spacing,
        "area_effective": area_e,
        "area_gross": gross.area,
        "inertia_effective": inertia_e,
        "euler_stress": sigma_e,
        "perry_robertson_stress": sigma_pr,
        "johnson_ostenfeld_stress": sigma_jo,
        "limiting_stress": sigma_l,
        "panel_ultimate_stress": sigma_l * area_e / gross.area,
        "plate_strength_ratio": geom.faulkner_plate_strength,
        "slenderness_beta": geom.plate_slenderness,
    }


def system_load_shares(resp: Mapping[str, float]) -> tuple[float, float]:
    """Renormalise the modal partition onto the two member systems.

    The torsion term of the orthotropic-plate partition is not a member system,
    so it is redistributed between the two bending directions in proportion to
    their stiffness share.  The result sums to 1 exactly, which is what keeps the
    member load path in equilibrium with the applied patch force.
    """
    fx = max(resp["load_share_x"], 0.0)
    fy = max(resp["load_share_y"], 0.0)
    total = fx + fy
    if total <= 1e-12:
        return 0.5, 0.5
    return fx / total, fy / total


_system_shares = system_load_shares  # backwards-compatible private alias


def grillage_utilisations(geom: "GrillageGeometry", case: "CombinedLoadCase",
                          ice_pressure: float | None = None,
                          terms: int = 61) -> dict[str, float]:
    """Member utilisations for one combined load case.

    Load path (two-system smeared orthotropic grillage, classical):

    ``plate`` -> ``longitudinal stiffener system`` (x, spans the frames) and
    ``transverse frame system`` (y, spans the girders).  The split is the modal
    stiffness partition of an equivalent orthotropic plate whose ``Dx`` comes
    from the longitudinal stiffener section and ``Dy`` from the frame section,
    renormalised so the two systems carry exactly the applied force
    (``load_share_x_system`` + ``load_share_y_system`` = 1).  The patch is then
    concentrated on the members it actually touches, which is conservative
    relative to the smeared distribution.

    Three independent checks (1.0 = at the member limit):

    ``plate_local_bending``
        the plate field ``a x b`` as an SSSS plate under the patch plus the
        uniform lateral pressure; capacity is the plate strip plastic moment
        reduced by the coexisting membrane compression,
        ``m_p sqrt(1 - (sigma_x/sigma_y)^2)``.
    ``longitudinal_stiffener``
        beam-column across the frames: extreme-fibre stress
        ``sigma_x + M beta_m / Z`` against the effective-width limiting stress,
        with the exact secant magnification ``beta_m``.
    ``transverse_frame``
        rigid-plastic collapse of the frame under the patch share of the ice load
        plus the hydrostatic share.
    """
    geom.validate()
    case.validate(geom)
    p_ice = case.ice_pressure if ice_pressure is None else float(ice_pressure)
    _non_negative("ice_pressure", p_ice)
    lx, ly = case.patch_extents(geom)
    have_patch = p_ice > 0.0 and lx > 0.0 and ly > 0.0

    span_x = geom.n_spans * geom.frame_spacing      # grillage length along x
    span_y = geom.frame_span                        # frame span along y
    patch = ((0.5 - 0.5 * lx / span_x, 0.5 + 0.5 * lx / span_x,
              0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y)
             if have_patch else (0.0, 1.0, 0.0, 1.0))

    dx, dy, d3 = geom.plate_rigidities()
    if have_patch:
        patch_resp = plate_patch_response(
            span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3, pressure=p_ice,
            patch=patch, terms=terms, poisson=geom.poisson)
        f_x_ice, f_y_ice = system_load_shares(patch_resp)
        force_ice = patch_resp["total_force"]
    else:
        patch_resp = None
        f_x_ice = f_y_ice = force_ice = 0.0

    if case.hydrostatic_pressure > 0.0:
        hydro_resp = plate_patch_response(
            span_x=span_x, span_y=span_y, d1=dx, d2=dy, d3=d3,
            pressure=case.hydrostatic_pressure, patch=(0.0, 1.0, 0.0, 1.0),
            terms=terms, poisson=geom.poisson)
        f_x_h, f_y_h = system_load_shares(hydro_resp)
        force_h = case.hydrostatic_pressure * geom.panel_area
    else:
        hydro_resp = None
        f_x_h = f_y_h = force_h = 0.0

    # (a) local plate bending: exact SSSS plate field a x b, isotropic rigidity
    plate_cap = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    eta = case.axial_stress / geom.yield_stress
    plate_cap_eff = plate_cap * math.sqrt(max(1.0 - eta ** 2, 0.0))
    m_plate = 0.0
    if have_patch:
        pr = plate_patch_response(
            span_x=geom.frame_spacing, span_y=geom.bay_spacing,
            d1=geom.plate_rigidity, d2=geom.plate_rigidity,
            d3=geom.plate_rigidity, pressure=p_ice,
            patch=(0.5 - 0.5 * lx / geom.frame_spacing,
                   0.5 + 0.5 * lx / geom.frame_spacing,
                   0.5 - 0.5 * ly / geom.bay_spacing,
                   0.5 + 0.5 * ly / geom.bay_spacing),
            terms=terms, poisson=geom.poisson, sample=15)
        m_plate += max(pr["m_x_max"], pr["m_y_max"])
    if case.hydrostatic_pressure > 0.0:
        pr = plate_patch_response(
            span_x=geom.frame_spacing, span_y=geom.bay_spacing,
            d1=geom.plate_rigidity, d2=geom.plate_rigidity,
            d3=geom.plate_rigidity, pressure=case.hydrostatic_pressure,
            terms=terms, poisson=geom.poisson, sample=15)
        m_plate += max(pr["m_x_max"], pr["m_y_max"])
    u_plate = m_plate / plate_cap_eff if plate_cap_eff > 0.0 else float("inf")

    # (b) longitudinal stiffener beam-column
    cap = stiffener_axial_capacity(geom)
    sigma_l = cap["limiting_stress"]
    area_e = cap["area_effective"]
    section = geom.with_plate(geom.stiffener, cap["effective_width"])
    axial_force = case.axial_stress * area_e
    pe = math.pi ** 2 * geom.young * section.inertia / geom.frame_spacing ** 2
    if 0.0 < axial_force < pe:
        beta_m = beam_column_amplification_exact(axial_force, pe)
    elif axial_force >= pe:
        beta_m = 10.0
    else:
        beta_m = 1.0
    n_overlap = (min(max(2, int(round(ly / geom.bay_spacing)) + 1), geom.n_stiffeners)
                 if have_patch else geom.n_stiffeners)
    per_stiff_ice = force_ice * f_x_ice / n_overlap
    m_ice = (beam_moment_centre_patch(per_stiff_ice, geom.frame_spacing, lx)
             if have_patch else 0.0)
    per_stiff_h = force_h * f_x_h / geom.n_stiffeners
    m_h = (beam_moment_centre_patch(per_stiff_h, geom.frame_spacing, geom.frame_spacing)
           if force_h > 0.0 else 0.0)
    bending_stress = beta_m * (m_ice + m_h) / section.elastic_modulus_section
    u_stiff = (case.axial_stress + bending_stress) / sigma_l

    # (c) transverse frame rigid-plastic collapse
    frame_section = geom.frame_section
    mp = geom.yield_stress * frame_section.plastic_modulus
    # the patch is centred in a frame space: its two bounding frames share it
    per_frame_ice = 0.5 * force_ice * f_y_ice
    per_frame_h = force_h * f_y_h / 2.0
    f_u_patch = (frame_plastic_collapse_force(mp, geom.frame_span, ly)
                 if have_patch else math.inf)
    f_u_uniform = frame_plastic_collapse_force(mp, geom.frame_span, geom.frame_span)
    u_frame = ((0.0 if not have_patch else per_frame_ice / f_u_patch)
               + (0.0 if force_h == 0.0 else per_frame_h / f_u_uniform))

    return {
        "plate_local_bending": u_plate,
        "longitudinal_stiffener": u_stiff,
        "transverse_frame": u_frame,
        "m_plate": m_plate,
        "m_plate_capacity": plate_cap_eff,
        "stiffener_limiting_stress": sigma_l,
        "stiffener_bending_stress": bending_stress,
        "moment_amplification": beta_m,
        "frame_plastic_moment": mp,
        "frame_collapse_force_patch": f_u_patch,
        "frame_collapse_force_uniform": f_u_uniform,
        "load_share_x_system": f_x_ice,
        "load_share_y_system": f_y_ice,
        "load_share_torsion_raw": (patch_resp["load_share_torsion"]
                                   if patch_resp is not None else 0.0),
        "force_ice": force_ice,
        "force_hydrostatic": force_h,
        "force_stiffener_system": force_ice * f_x_ice + force_h * f_x_h,
        "force_frame_system": force_ice * f_y_ice + force_h * f_y_h,
        "per_stiffener_ice_force": per_stiff_ice,
        "per_frame_ice_force": per_frame_ice,
        "axial_force_stiffener": axial_force,
        "euler_force_stiffener": pe,
        "stiffeners_touched": float(n_overlap),
    }


def _governing(u: Mapping[str, float]) -> str:
    keys = ("plate_local_bending", "longitudinal_stiffener", "transverse_frame")
    return max(keys, key=lambda k: (u[k], k))


def ultimate_ice_pressure(geom: "GrillageGeometry", case: "CombinedLoadCase",
                          tol: float = 1e-6, max_iter: int = 200,
                          terms: int = 61) -> dict[str, object]:
    """Largest ice patch pressure keeping every member utilisation ``<= 1``.

    Bisection on ``ice_pressure`` (the peak utilisation is monotone increasing in
    it for fixed coexisting compression and hydrostatic pressure), bracketed from
    below by geometric growth.  ``converged=False`` is returned honestly when no
    bracket is found.  ``margin_vs_design = p_ult / design_pressure``.
    """
    geom.validate()
    case.validate(geom)
    if case.patch_length <= 0.0 or case.patch_width <= 0.0:
        raise ValueError("ultimate_ice_pressure needs a positive patch size")

    def peak(p: float) -> tuple[float, dict[str, float]]:
        u = grillage_utilisations(geom, case, ice_pressure=p, terms=terms)
        return (max(u["plate_local_bending"], u["longitudinal_stiffener"],
                    u["transverse_frame"]), u)

    u0, base = peak(0.0)
    if u0 >= 1.0:
        return {
            "ice_pressure_ultimate": 0.0,
            "governing_member": _governing(base),
            "utilisation_at_limit": u0,
            "utilisation_without_ice": u0,
            "base_utilisations": base,
            "utilisations_at_limit": base,
            "margin_vs_design": 0.0 if case.design_pressure else None,
            "design_provenance": case.design_provenance,
            "converged": True,
        }
    lo = 0.0
    hi = max(1.0e5, 10.0 * (case.design_pressure or 1.0e6))
    u_hi, _ = peak(hi)
    guard = 0
    while u_hi < 1.0 and guard < 60:
        lo = hi
        hi *= 2.0
        u_hi, _ = peak(hi)
        guard += 1
    if u_hi < 1.0:
        return {
            "ice_pressure_ultimate": float("inf"),
            "governing_member": "none",
            "utilisation_at_limit": u_hi,
            "utilisation_without_ice": u0,
            "base_utilisations": base,
            "margin_vs_design": None,
            "design_provenance": case.design_provenance,
            "converged": False,
        }
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        u_mid, _ = peak(mid)
        if u_mid < 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= tol * max(hi, 1.0):
            break
    p_ult = 0.5 * (lo + hi)
    _, u_lim = peak(p_ult)
    u_peak = max(u_lim["plate_local_bending"], u_lim["longitudinal_stiffener"],
                 u_lim["transverse_frame"])
    return {
        "ice_pressure_ultimate": p_ult,
        "governing_member": _governing(u_lim),
        "utilisation_at_limit": u_peak,
        "utilisation_without_ice": u0,
        "base_utilisations": base,
        "utilisations_at_limit": u_lim,
        "margin_vs_design": (p_ult / case.design_pressure) if case.design_pressure else None,
        "design_provenance": case.design_provenance,
        "converged": True,
    }


# --------------------------------------------------------------------------- #
# panel stress -> ultimate load map
# --------------------------------------------------------------------------- #
def panel_stress_ultimate_map(geom: "GrillageGeometry",
                              axial_ratios: Sequence[float] = (0.0, 0.15, 0.30, 0.45, 0.60),
                              patch_area_ratios: Sequence[float] = (0.10, 0.25, 0.50, 1.00),
                              hydrostatic_pressures: Sequence[float] = (0.0, 5.0e4, 1.0e5),
                              patch_aspect: float = 1.0,
                              design_pressure: float | None = None,
                              design_provenance: str = "caller supplied",
                              terms: int = 61) -> dict[str, object]:
    """Machine-readable map ``(sigma_x/sigma_y, A_ice/A_panel, p_hydro) -> p_ice,ult``."""
    geom.validate()
    _positive("patch_aspect", patch_aspect)
    panel = stiffener_axial_capacity(geom)
    rows: list[dict[str, object]] = []
    for eta in axial_ratios:
        ratio = float(eta)
        if not (0.0 <= ratio < 1.0):
            raise ValueError("axial_ratios must lie in [0, 1)")
        for area_ratio in patch_area_ratios:
            ar = _fraction("patch_area_ratios", area_ratio)
            side = math.sqrt(ar * geom.panel_area)
            lx = min(side * math.sqrt(patch_aspect), geom.frame_spacing)
            ly = min(side / math.sqrt(patch_aspect), geom.bay_spacing)
            for p_h in hydrostatic_pressures:
                p_h = _non_negative("hydrostatic_pressures", p_h)
                case = CombinedLoadCase(
                    axial_stress=ratio * geom.yield_stress,
                    ice_pressure=0.0,
                    patch_length=lx,
                    patch_width=ly,
                    hydrostatic_pressure=p_h,
                    design_pressure=design_pressure,
                    design_provenance=design_provenance,
                ).validate(geom)
                res = ultimate_ice_pressure(geom, case, terms=terms)
                u_lim = res.get("utilisations_at_limit") or res["base_utilisations"]
                p_ult = float(res["ice_pressure_ultimate"])
                rows.append({
                    "sigma_x_over_yield": ratio,
                    "sigma_x_MPa": ratio * geom.yield_stress / 1e6,
                    "patch_area_ratio": ar,
                    "patch_length_m": lx,
                    "patch_width_m": ly,
                    "patch_area_m2": lx * ly,
                    "hydrostatic_pressure_MPa": p_h / 1e6,
                    "ice_pressure_ultimate_MPa": (p_ult / 1e6
                                                  if math.isfinite(p_ult) else None),
                    "governing_member": res["governing_member"],
                    "utilisation_plate": float(u_lim["plate_local_bending"]),
                    "utilisation_stiffener": float(u_lim["longitudinal_stiffener"]),
                    "utilisation_frame": float(u_lim["transverse_frame"]),
                    "margin_vs_design": res.get("margin_vs_design"),
                    "converged": bool(res.get("converged", False)),
                })
    return {
        "geometry": {
            "frame_spacing_m": geom.frame_spacing,
            "bay_spacing_m": geom.bay_spacing,
            "n_bays": int(geom.n_bays),
            "n_spans": int(geom.n_spans),
            "plate_thickness_mm": geom.plate_thickness * 1e3,
            "yield_stress_MPa": geom.yield_stress / 1e6,
            "plate_slenderness_beta": geom.plate_slenderness,
            "plate_strength_ratio": geom.faulkner_plate_strength,
            "panel_ultimate_stress_MPa": panel["panel_ultimate_stress"] / 1e6,
            "stiffener_limiting_stress_MPa": panel["limiting_stress"] / 1e6,
        },
        "design_pressure_MPa": (design_pressure / 1e6) if design_pressure else None,
        "design_provenance": design_provenance,
        "rows": rows,
    }


def write_ultimate_map(map_dict: Mapping[str, object], path: str | Path) -> Path:
    """Write the mapping table as JSON (machine-readable acceptance artifact)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "tensorfem.grillage_ice_ultimate.map/v1", **dict(map_dict)}
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def ice_pressure_area_rule(p0: float, area_ref: float = 1.0, exponent: float = 0.3,
                           provenance: str = "caller supplied") -> IACSPressureArea:
    """Thin wrapper on the existing rule surface (no tabulated rule value invented)."""
    _positive("p0", p0)
    _positive("area_ref", area_ref)
    return IACSPressureArea(p0=p0, area_ref=area_ref, exponent=exponent,
                            provenance=provenance)


def case_study_geometry() -> "GrillageGeometry":
    """Representative ice-belt grillage used by the documentation case study.

    t = 20 mm plate, longitudinal Tee stiffeners 250x10 + 80x12 at b = 0.35 m,
    transverse frames 550x10 + 120x14 at a = 2.1 m spanning ``8 x 0.35 = 2.8 m``
    between girders, EH36 steel (``sigma_y = 355 MPa``).  A documentation choice,
    not a rule extraction.
    """
    return GrillageGeometry(
        frame_spacing=2.1, bay_spacing=0.35, n_bays=8, n_spans=4,
        plate_thickness=0.020,
        stiffener=MemberSection(0.25, 0.010, 0.080, 0.012),
        frame=MemberSection(0.55, 0.010, 0.120, 0.014),
        young=2.06e11, poisson=0.3, yield_stress=3.55e8,
    ).validate()