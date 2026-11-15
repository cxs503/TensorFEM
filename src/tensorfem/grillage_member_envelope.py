"""Position-aware member-demand envelope for the ice-patch grillage anchor.

Why this module exists (round 5 of ``GAP-PP-5-ICE-STRUCT-GRILLAGE-ULTIMATE``)
---------------------------------------------------------------------------
``grillage_utilisations`` derived the member demands of the stiffened grillage
from the **smeared** orthotropic plate's *global* elastic share ``f_x``/``f_y``
and from two position-blind heuristics:

* ``n_overlap = round(l_y / b) + 1`` stiffeners were told to share the patch,
* the patch was always shared by exactly **two** frames (``0.5 * F * f_y``).

Round 4 (``docs/CASE-GAP-PP-5-...-round4.md``) showed that for a patch pressed
onto a frame line the transverse frame actually carries ~the whole patch, so the
smeared frame demand was low by 2.08x.  Probing the complementary topology in
round 5 shows the heuristics fail in the other direction as well: with the
patch centred in a *bay* (``n_spans`` odd) the smeared layer under-predicts both
the stiffener moment (~2.5x) and the frame utilisation (~2.7x), because the
global energy share mixes the two load paths while the patch only loads the
members whose tributary strip it actually covers.

Both defects are the same modelling error: **the load path is a function of the
patch position relative to the member lines, and the smeared share cannot
express that.**  This module replaces the heuristics with three explicit,
equilibrium-consistent load-path idealisations, each of which carries the
*whole* patch (no smearing, no double counting):

``B1`` stiffener path
    plate strip action (short span) hands the patch to the stiffener line(s)
    whose tributary strip it covers, weighted by the overlap width; the
    stiffeners span simply supported between frames and deliver ``F_i / 2``
    end reactions to the two bounding frames.
``B2`` frame path
    the patch goes directly into the frame line(s) whose tributary strip
    covers it (again overlap-weighted); the frame spans simply supported
    between the longitudinal girders.
``C``  system path
    the continuous discrete grillage of :mod:`tensorfem.grillage_beam_grid`
    (verified against classical beam closed forms and equilibrium in round 4),
    which lets continuity over frames and frame flexibility redistribute load.

The true structure shares between ``B1``/``B2`` through plate bending -- the
plate's two-way action, exactly what the smeared model tried to capture
globally.  Without a validated two-way split the defensible engineering answer
is the **envelope** ``max(B1, B2, C)`` per member, which is conservative
relative to every individual path, and which the round-4 discrete grid (``C``)
is used to *bound from below* (it must never exceed the envelope).

Everything here is analytic except branch ``C``: influence-line statics of a
simply supported beam, overlap weights of tributary strips, and the member
capacities already used by ``grillage_ice_ultimate`` (Paik/Johnson-Ostenfeld
limiting stress, plastic modulus).  No calibration factors.
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Mapping, Sequence

# --------------------------------------------------------------------------- #
# small analytic kernels
# --------------------------------------------------------------------------- #


def tributary_weights(positions: Sequence[float], lo: float, hi: float,
                      length: float) -> list[tuple[float, float]]:
    """Overlap weights of member tributary strips against a patch extent.

    ``positions`` are the member lines (ascending, spanning ``[0, length]``).
    Member ``k`` owns the strip between the mid-points to its neighbours,
    clipped at the panel edges.  The returned weights are **normalised to sum
    to one over the strips the patch actually touches**, so a patch that sits
    inside a single tributary loads exactly that member (the round-4 finding)
    and the patch force is conserved for any position/width.

    A patch that overlaps no strip raises ``ValueError`` (silent zero weighting
    would hide a geometry error).
    """
    pos = [float(p) for p in positions]
    if len(pos) == 0:
        raise ValueError("positions must not be empty")
    if not (0.0 <= lo < hi <= length):
        raise ValueError("patch extent must satisfy 0 <= lo < hi <= length")
    raw: list[tuple[float, float]] = []
    for k, p in enumerate(pos):
        lo_edge = 0.0 if k == 0 else 0.5 * (pos[k - 1] + p)
        hi_edge = length if k == len(pos) - 1 else 0.5 * (p + pos[k + 1])
        w = max(0.0, min(hi_edge, hi) - max(lo_edge, lo))
        raw.append((p, w))
    total = sum(w for _, w in raw)
    if total <= 0.0:
        raise ValueError("the patch does not overlap any member tributary strip")
    return [(p, w / total) for p, w in raw]


def ss_beam_moment(span: float, loads: Sequence[tuple[float, float]]) -> float:
    """Peak bending moment of a simply supported beam under point loads.

    ``loads`` are ``(position, force)`` pairs.  The influence line of a simply
    supported beam is exact, so this is the textbook closed form evaluated at
    the load points and at mid-span (a piecewise-linear moment diagram peaks at
    one of those sections).
    """
    if span <= 0.0:
        raise ValueError("span must be positive")
    stations = [min(max(float(y), 0.0), span) for y, _ in loads] + [0.5 * span]
    best = 0.0
    for z in stations:
        m = 0.0
        for y, p in loads:
            y = min(max(float(y), 0.0), span)
            if z <= y:
                m += p * z * (span - y) / span
            else:
                m += p * y * (span - z) / span
        best = max(best, abs(m))
    return best


def patch_moment_centre(span: float, force: float, patch_length: float,
                        sample: int = 400) -> float:
    """Peak moment of a centrally placed patch load on a simply supported beam.

    Closed form ``F (2L - c) / 8`` for a patch symmetric about mid-span; the
    quadrature path handles an arbitrary patch length consistently and agrees
    with the closed form (checked in the module tests).
    """
    if patch_length >= span:
        return force * span / 8.0
    lo = 0.5 * (span - patch_length)
    hi = 0.5 * (span + patch_length)
    q = force / (hi - lo)
    h = (hi - lo) / sample
    m = 0.0
    for k in range(sample):
        z = lo + (k + 0.5) * h
        m += q * h * z * (span - z) / span
    return m


def patch_moment_closed_form(span: float, force: float, patch_length: float) -> float:
    """``F (2L - c) / 8`` -- the classical central-patch moment of a S-S beam."""
    return force * (2.0 * span - patch_length) / 8.0


# --------------------------------------------------------------------------- #
# branch demands
# --------------------------------------------------------------------------- #


def _patch_rect(geom, lx: float, ly: float) -> tuple[float, float, float, float]:
    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    return (0.5 - 0.5 * lx / span_x, 0.5 + 0.5 * lx / span_x,
            0.5 - 0.5 * ly / span_y, 0.5 + 0.5 * ly / span_y)


def _member_capacities(geom):
    """Stiffener section properties and frame plastic moment (shared with the parent module)."""
    from .grillage_ice_ultimate import stiffener_axial_capacity

    cap = stiffener_axial_capacity(geom)
    section = geom.with_plate(geom.stiffener, cap["effective_width"])
    frame_mp = geom.yield_stress * geom.frame_section.plastic_modulus
    return cap, section, frame_mp


def _stiffener_utilisation(geom, section, cap, axial_stress: float,
                           moment: float) -> dict[str, float]:
    from .grillage_ice_ultimate import (beam_column_amplification_exact,
                                        beam_column_amplification_approx)

    area_e = cap["area_effective"]
    axial_force = axial_stress * area_e
    pe = math.pi ** 2 * geom.young * section.inertia / geom.frame_spacing ** 2
    if 0.0 < axial_force < pe:
        beta = beam_column_amplification_exact(axial_force, pe)
    elif axial_force >= pe:
        beta = 10.0
    else:
        beta = 1.0
    bending = beta * moment / section.elastic_modulus_section
    sigma_l = cap["limiting_stress"]
    return {
        "moment_amplification": beta,
        "bending_stress": bending,
        "axial_stress": axial_stress,
        "utilisation": (axial_stress + bending) / sigma_l,
    }


def branch_demands(geom, case, ice_pressure: float | None = None) -> dict[str, object]:
    """Analytic load-path branches ``B1`` (stiffener) and ``B2`` (frame).

    Each branch carries the **whole** patch force through one path, so the two
    are independent lower/upper bracketing idealisations of the real two-way
    plate action rather than a split of it.  Hydrostatic pressure is treated the
    same way in both branches (uniform over the panel).
    """
    geom.validate()
    case.validate(geom)
    p_ice = case.ice_pressure if ice_pressure is None else float(ice_pressure)
    if p_ice < 0.0:
        raise ValueError("ice_pressure must be non-negative")
    lx, ly = case.patch_extents(geom)
    have_patch = p_ice > 0.0 and lx > 0.0 and ly > 0.0

    cap, section, frame_mp = _member_capacities(geom)
    a = geom.frame_spacing
    b = geom.bay_spacing
    span_x = geom.n_spans * a
    span_y = geom.frame_span

    force_ice = p_ice * lx * ly if have_patch else 0.0
    p_h = case.hydrostatic_pressure
    force_h = p_h * geom.panel_area

    # ---- B1: patch -> stiffener lines -> frame reactions ------------------ #
    stiff_y = [i * b for i in range(geom.n_bays + 1)]
    frame_x = [j * a for j in range(geom.n_spans + 1)]
    y0 = 0.5 * (span_y - ly)
    x0 = 0.5 * (span_x - lx)

    b1_moment_ice = 0.0
    b1_frame_point_loads: list[tuple[float, float]] = []
    if have_patch:
        weights_y = tributary_weights(stiff_y, y0, y0 + ly, span_y)
        for y_pos, w in weights_y:
            f_i = force_ice * w
            if f_i <= 0.0:
                continue
            b1_moment_ice = max(b1_moment_ice,
                                patch_moment_centre(a, f_i, lx))
            # a patch centred between frame lines sends F_i / 2 to *this* frame
            # (the other half goes to the opposite bounding frame)
            b1_frame_point_loads.append((y_pos, 0.5 * f_i))
    b1_moment_stiff_h = patch_moment_centre(a, p_h * b * a, a) if p_h > 0.0 else 0.0
    b1_frame_h = p_h * a * span_y ** 2 / 8.0 if p_h > 0.0 else 0.0
    b1_moment_frame = (ss_beam_moment(span_y, b1_frame_point_loads)
                       if b1_frame_point_loads else 0.0)

    # ---- B2: patch -> frame lines directly -------------------------------- #
    b2_moment_frame = 0.0
    b2_frame_force = 0.0
    if have_patch:
        weights_x = tributary_weights(frame_x, x0, x0 + lx, span_x)
        for _, w in weights_x:
            f_j = force_ice * w
            b2_frame_force = max(b2_frame_force, f_j)
            b2_moment_frame = max(b2_moment_frame,
                                  patch_moment_centre(span_y, f_j, ly))
    b2_moment_stiff_ice = 0.0        # B2 sends no patch load into the stiffeners
    b2_moment_stiff_h = b1_moment_stiff_h

    b1 = {
        "patch_force": force_ice,
        "stiffener_moment_ice": b1_moment_ice,
        "stiffener_moment_hydrostatic": b1_moment_stiff_h,
        "frame_moment_ice": b1_moment_frame,
        "frame_moment_hydrostatic": b1_frame_h,
        "stiffener": _stiffener_utilisation(
            geom, section, cap, case.axial_stress,
            b1_moment_ice + b1_moment_stiff_h),
        "frame_utilisation": ((b1_moment_frame + b1_frame_h) / frame_mp
                              if frame_mp > 0.0 else math.inf),
    }
    b2 = {
        "patch_force": force_ice,
        "stiffener_moment_ice": b2_moment_stiff_ice,
        "stiffener_moment_hydrostatic": b2_moment_stiff_h,
        "frame_moment_ice": b2_moment_frame,
        "frame_moment_hydrostatic": b1_frame_h,
        "frame_force_peak": b2_frame_force,
        "stiffener": _stiffener_utilisation(
            geom, section, cap, case.axial_stress,
            b2_moment_stiff_ice + b2_moment_stiff_h),
        "frame_utilisation": ((b2_moment_frame + b1_frame_h) / frame_mp
                              if frame_mp > 0.0 else math.inf),
    }
    return {
        "ice_pressure": p_ice,
        "patch_length": lx,
        "patch_width": ly,
        "patch_rect": _patch_rect(geom, lx, ly) if have_patch else (0.0, 1.0, 0.0, 1.0),
        "panel_stiffener_span": a,
        "panel_frame_span": span_y,
        "frame_plastic_moment": frame_mp,
        "hydrostatic_force": force_h,
        "B1_stiffener_path": b1,
        "B2_frame_path": b2,
    }


@lru_cache(maxsize=256)
def _unit_grid(span_x: float, span_y: float, ei_x: float, gj_x: float,
               ei_y: float, gj_y: float, n_x_lines: int, n_y_lines: int,
               sub: int, load_split: float,
               fx1: float, fx2: float, fy1: float, fy2: float
               ) -> tuple[float, float, float]:
    """Unit-pressure member moments and stiffener energy share of the discrete grid.

    Cached because the bisections in ``ultimate_ice_pressure`` call the same
    geometry/patch dozens of times and the response is linear in pressure.
    """
    from .grillage_beam_grid import beam_grid_response

    res = beam_grid_response(
        span_x=span_x, span_y=span_y, ei_x=ei_x, gj_x=gj_x, ei_y=ei_y, gj_y=gj_y,
        n_x_lines=n_x_lines, n_y_lines=n_y_lines, sub=int(sub),
        load_split=float(load_split), pressure=1.0,
        patch=(fx1, fx2, fy1, fy2))
    return (float(res["moment_x_max"]), float(res["moment_y_max"]),
            float(res["share_x_energy"]))


def system_branch(geom, case, *, sub: int = 8, load_split: float = 0.5,
                  ice_pressure: float | None = None) -> dict[str, object]:
    """Branch ``C``: the continuous discrete grillage of ``grillage_beam_grid``.

    The patch and the hydrostatic pressure are each solved at unit pressure
    (the member moments are linear in pressure) and scaled, so the branch is as
    cheap as the analytic ones and can be called inside a bisection.
    """
    from .grillage_beam_grid import beam_grid_response, _tee_torsion_constant

    geom.validate()
    case.validate(geom)
    p_ice = case.ice_pressure if ice_pressure is None else float(ice_pressure)
    lx, ly = case.patch_extents(geom)
    have_patch = p_ice > 0.0 and lx > 0.0 and ly > 0.0

    span_x = geom.n_spans * geom.frame_spacing
    span_y = geom.frame_span
    shear = geom.young / (2.0 * (1.0 + geom.poisson))
    kwargs = dict(
        span_x=span_x, span_y=span_y,
        ei_x=geom.young * geom.stiffener_section.inertia,
        gj_x=shear * _tee_torsion_constant(geom.stiffener_section),
        ei_y=geom.young * geom.frame_section.inertia,
        gj_y=shear * _tee_torsion_constant(geom.frame_section),
        n_x_lines=geom.n_spans + 1, n_y_lines=geom.n_bays + 1,
        sub=int(sub), load_split=float(load_split),
    )
    unit_ice: tuple[float, float, float] | None = None
    unit_h: tuple[float, float, float] | None = None
    if have_patch:
        fx1, fx2, fy1, fy2 = _patch_rect(geom, lx, ly)
        unit_ice = _unit_grid(span_x, span_y, kwargs["ei_x"], kwargs["gj_x"],
                              kwargs["ei_y"], kwargs["gj_y"], kwargs["n_x_lines"],
                              kwargs["n_y_lines"], int(sub), float(load_split),
                              fx1, fx2, fy1, fy2)
    if case.hydrostatic_pressure > 0.0:
        unit_h = _unit_grid(span_x, span_y, kwargs["ei_x"], kwargs["gj_x"],
                            kwargs["ei_y"], kwargs["gj_y"], kwargs["n_x_lines"],
                            kwargs["n_y_lines"], int(sub), float(load_split),
                            0.0, 1.0, 0.0, 1.0)

    cap, section, frame_mp = _member_capacities(geom)
    # the grid orients the x-members along the hull (stiffener lines) and the
    # y-members across it (frame lines); classify the moments accordingly
    m_stiff = ((p_ice * unit_ice[0] if unit_ice is not None else 0.0)
               + (case.hydrostatic_pressure * unit_h[0]
                  if unit_h is not None else 0.0))
    m_frame = ((p_ice * unit_ice[1] if unit_ice is not None else 0.0)
               + (case.hydrostatic_pressure * unit_h[1]
                  if unit_h is not None else 0.0))
    stiff = _stiffener_utilisation(geom, section, cap, case.axial_stress, m_stiff)
    return {
        "ice_pressure": p_ice,
        "patch_length": lx,
        "patch_width": ly,
        "sub": int(sub),
        "load_split": float(load_split),
        "stiffener_moment": m_stiff,
        "frame_moment": m_frame,
        "frame_plastic_moment": frame_mp,
        "stiffener": stiff,
        "frame_utilisation": m_frame / frame_mp if frame_mp > 0.0 else math.inf,
        "share_x_energy": unit_ice[2] if unit_ice is not None else None,
    }


def member_envelope(geom, case, ice_pressure: float | None = None, *,
                    sub: int = 8, load_split: float = 0.5,
                    with_system: bool = True) -> dict[str, object]:
    """Envelope ``max(B1, B2, C)`` per member, with the governing branch named."""
    analytic = branch_demands(geom, case, ice_pressure=ice_pressure)
    system = (system_branch(geom, case, sub=sub, load_split=load_split,
                            ice_pressure=ice_pressure)
              if with_system else None)
    u_stiff = {
        "B1_stiffener_path": float(analytic["B1_stiffener_path"]["stiffener"]["utilisation"]),
        "B2_frame_path": float(analytic["B2_frame_path"]["stiffener"]["utilisation"]),
    }
    u_frame = {
        "B1_stiffener_path": float(analytic["B1_stiffener_path"]["frame_utilisation"]),
        "B2_frame_path": float(analytic["B2_frame_path"]["frame_utilisation"]),
    }
    if system is not None:
        u_stiff["C_system"] = float(system["stiffener"]["utilisation"])
        u_frame["C_system"] = float(system["frame_utilisation"])
    stiff_branch = max(u_stiff, key=lambda k: (u_stiff[k], k))
    frame_branch = max(u_frame, key=lambda k: (u_frame[k], k))
    return {
        "ice_pressure": float(analytic["ice_pressure"]),
        "patch_length": float(analytic["patch_length"]),
        "patch_width": float(analytic["patch_width"]),
        "stiffener_utilisation": u_stiff[stiff_branch],
        "stiffener_branch": stiff_branch,
        "frame_utilisation": u_frame[frame_branch],
        "frame_branch": frame_branch,
        "branch_utilisations": {"stiffener": u_stiff, "frame": u_frame},
        "system_to_envelope_stiffener": (
            u_stiff["C_system"] / u_stiff[stiff_branch]
            if system is not None and u_stiff[stiff_branch] > 0.0 else None),
        "system_to_envelope_frame": (
            u_frame["C_system"] / u_frame[frame_branch]
            if system is not None and u_frame[frame_branch] > 0.0 else None),
        "analytic": analytic,
        "system": system,
    }


# --------------------------------------------------------------------------- #
# classical reference brackets (literature / textbook closed forms)
# --------------------------------------------------------------------------- #


def classical_plate_bracket(geom) -> dict[str, float]:
    """Classical closed-form lateral-pressure brackets for one plate panel.

    Short span ``b = bay_spacing``, plastic moment per unit width
    ``M_p = sigma_y t^2 / 4`` (the same capacity the plate branch of
    ``grillage_utilisations`` uses):

    * simply supported strip, first hinge at mid-span
      (``M = p b^2 / 8 = M_p``) -> ``p = 8 M_p / b^2``;
    * clamped strip, symmetric collapse mechanism
      (``M_edge + M_mid = 2 M_p = p b^2 / 8``) -> ``p = 16 M_p / b^2``.

    These are textbook rigid-plastic / first-yield results for a one-way strip,
    which is the right idealisation for the anchor's 2.1 m x 0.35 m panel
    (aspect 6:1).  They are **not** a classification-society rule: no rule text
    was obtainable in this environment (see
    ``benchmarks/anchors/iacs-csr-w22.json``, ``acquisition_status``).
    """
    geom.validate()
    mp = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    b = geom.bay_spacing
    return {
        "plate_plastic_moment_per_width": mp,
        "panel_short_span": b,
        "panel_aspect": geom.frame_spacing / b,
        "p_first_hinge_ss_strip": 8.0 * mp / b ** 2,
        "p_plastic_collapse_clamped_strip": 16.0 * mp / b ** 2,
    }


def classical_compare(geom, *, terms: int = 61) -> dict[str, object]:
    """Compare the plate-controlled ultimate pressure with the classical strip results."""
    from .grillage_ice_ultimate import CombinedLoadCase, ultimate_ice_pressure

    bracket = classical_plate_bracket(geom)
    # a patch that fills one bay in y and half a frame space in x: the closest
    # discretisation of the "strip under uniform pressure" load of the closed form
    case = CombinedLoadCase(ice_pressure=0.0, patch_length=0.5 * geom.frame_spacing,
                           patch_width=geom.bay_spacing)
    res = ultimate_ice_pressure(geom, case, terms=terms)
    u = res.get("utilisations_at_limit") or res["base_utilisations"]
    p_model = float(res["ice_pressure_ultimate"])
    ref = bracket["p_first_hinge_ss_strip"]
    return {
        "model_p_ult": p_model,
        "model_governing_member": str(res["governing_member"]),
        "model_plate_utilisation_at_limit": float(u["plate_local_bending"]),
        "reference_ss_strip_first_hinge": ref,
        "reference_clamped_strip_plastic_collapse":
            bracket["p_plastic_collapse_clamped_strip"],
        "relative_error_vs_ss_strip": (p_model - ref) / ref,
    }