"""Bounded stiffener-instability and local-pit screening benchmarks.

These are linear bifurcation/sensitivity checks with independent closed-form
oracles.  They intentionally do not model progressive panel collapse.
"""
from __future__ import annotations

import math


TOLERANCE = 0.03


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def lateral_torsional_buckling_screen(
    *, segments: int, length: float = 3.0, young: float = 210.0e9,
    shear_modulus: float = 80.8e9, weak_axis_inertia: float = 8.0e-7,
    torsion_constant: float = 1.2e-7, warping_constant: float = 2.5e-8,
) -> dict[str, float]:
    """Screen a simply supported doubly-symmetric stiffener under moment.

    The reference is the classical uniform-moment elastic LTB expression

    ``Mcr=(pi/L)*sqrt(E*Iy*GJ)*sqrt(1+pi²*E*Iw/(L²*GJ))``.

    The computed value replaces ``(pi/L)²`` by the smallest eigenvalue of a
    centred-difference Dirichlet Laplacian.  This gives an independently
    convergent spatial discretisation without embedding the closed form in the
    numerical result.
    """
    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 2:
        raise ValueError("segments must be an integer of at least two")
    l = _positive("length", length)
    e = _positive("young", young)
    g = _positive("shear_modulus", shear_modulus)
    iy = _positive("weak_axis_inertia", weak_axis_inertia)
    jt = _positive("torsion_constant", torsion_constant)
    iw = _positive("warping_constant", warping_constant)
    h = l / (segments + 1)
    wave_number_sq = 4.0 * math.sin(math.pi / (2.0*(segments+1)))**2 / h**2
    computed = math.sqrt(e*iy * (g*jt*wave_number_sq + e*iw*wave_number_sq**2))
    exact_wave_number_sq = (math.pi/l)**2
    oracle = math.sqrt(e*iy * (g*jt*exact_wave_number_sq
                               + e*iw*exact_wave_number_sq**2))
    return {
        "critical_moment": computed,
        "oracle_moment": oracle,
        "relative_error": abs(computed/oracle - 1.0),
        "discrete_wave_number_squared": wave_number_sq,
        "oracle_wave_number_squared": exact_wave_number_sq,
    }


def circular_pit_bending_screen(
    *, cells: int, length: float = 1.0, width: float = 1.0,
    pit_center_x: float = 0.47, pit_center_y: float = 0.53,
    pit_radius: float = 0.18, rigidity_loss: float = 0.65,
) -> dict[str, float | int]:
    """Integrate a sharp circular rigidity-loss patch by cell-centre tagging.

    A unit constant-curvature plate coupon has pristine rigidity ``D0=1``.
    Inside the pit it is ``(1-rigidity_loss)*D0``.  The independent oracle is
    the exact circle area ``pi*r²``.  The field corresponds to the thickness
    ratio ``(1-rigidity_loss)**(1/3)`` for unchanged isotropic material.
    """
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 2:
        raise ValueError("cells must be an integer of at least two")
    l, b, radius = (_positive("length", length), _positive("width", width),
                    _positive("pit_radius", pit_radius))
    cx, cy = float(pit_center_x), float(pit_center_y)
    loss = float(rigidity_loss)
    if not all(math.isfinite(v) for v in (cx, cy, loss)):
        raise ValueError("pit centre and loss must be finite")
    if not 0.0 < loss < 1.0:
        raise ValueError("rigidity_loss must lie in (0, 1)")
    if radius >= min(cx, l-cx, cy, b-cy):
        raise ValueError("circular pit must lie wholly inside the coupon")
    dx, dy = l/cells, b/cells
    tagged = 0
    for i in range(cells):
        x = (i+0.5)*dx
        for j in range(cells):
            y = (j+0.5)*dy
            tagged += (x-cx)**2 + (y-cy)**2 <= radius**2
    computed_area = tagged*dx*dy
    oracle_area = math.pi*radius**2
    area = l*b
    computed_rigidity = area - loss*computed_area
    oracle_rigidity = area - loss*oracle_area
    return {
        "tagged_cells": tagged,
        "computed_pit_area": computed_area,
        "oracle_pit_area": oracle_area,
        "pit_area_relative_error": abs(computed_area/oracle_area - 1.0),
        "effective_bending_rigidity": computed_rigidity,
        "oracle_effective_bending_rigidity": oracle_rigidity,
        "rigidity_relative_error": abs(computed_rigidity/oracle_rigidity - 1.0),
        "pit_thickness_ratio": (1.0-loss)**(1.0/3.0),
    }


def run_stiffener_failure_qualification() -> dict[str, object]:
    """Execute LTB and local-pit convergence evidence below three percent."""
    ltb_meshes = (4, 8, 16, 32)
    ltb = [lateral_torsional_buckling_screen(segments=n) for n in ltb_meshes]
    pit_meshes = (16, 32, 64, 128)
    pits = [circular_pit_bending_screen(cells=n) for n in pit_meshes]
    ltb_errors = [row["relative_error"] for row in ltb]
    pit_errors = [row["pit_area_relative_error"] for row in pits]
    passed = (
        ltb_errors[-1] < TOLERANCE
        and all(a > b for a, b in zip(ltb_errors, ltb_errors[1:]))
        and max(pit_errors[-2:]) < TOLERANCE
        and max(row["rigidity_relative_error"] for row in pits[-2:]) < TOLERANCE
    )
    if not passed:
        raise AssertionError("stiffener failure screening qualification failed")
    return {
        "schema": "tensorfem.marine-stiffener-failure-qualification/1.0",
        "tolerance": TOLERANCE,
        "lateral_torsional_buckling": {"segments": list(ltb_meshes), "results": ltb},
        "local_circular_pit": {"cells_per_axis": list(pit_meshes), "results": pits},
        "scope": ("linear elastic uniform-moment lateral-torsional bifurcation of an "
                  "ideal doubly-symmetric stiffener and constant-curvature sensitivity "
                  "to one sharp circular rigidity-loss patch; not plate-stiffener "
                  "interaction, weld failure, nonlinear tripping, crack initiation, "
                  "postbuckling or complete ultimate collapse"),
        "passed": True,
    }
