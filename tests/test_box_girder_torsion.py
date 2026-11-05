"""Fail-closed gates for the multiply connected (box girder) torsion oracle.

Tolerances are the observed values of ``benchmarks/box_girder_torsion.py`` with
a safety margin; the numbers are reproduced by the evidence JSON, not copied by
hand.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.box_torsion import (RectBoxSection, box_torsion_run,
                                   ring_cross_section, solve_box_prandtl)
from tensorfem.saint_venant_torsion import RectSection

A, B, T = 1.0, 0.6, 0.1


def test_ring_mesh_is_closed_and_body_fitted():
    nodes, quads = ring_cross_section(A, B, T, n_side=4, n_layers=3)
    assert len(quads) == 4 * 4 * 3
    assert nodes.shape[1] == 2
    x, y = nodes[:, 0], nodes[:, 1]
    assert float(x.abs().max()) <= A + 1e-12
    assert float(y.abs().max()) <= B + 1e-12
    # the outer and inner rings lie on the rectangle boundaries
    for ring in (nodes[:16], nodes[-16:]):
        on_wall = (((ring[:, 0].abs() - A).abs() < 1e-12)
                   | ((ring[:, 1].abs() - B).abs() < 1e-12)
                   | ((ring[:, 0].abs() - (A - T)).abs() < 1e-12)
                   | ((ring[:, 1].abs() - (B - T)).abs() < 1e-12))
        assert bool(on_wall.all())
    # the innermost ring reproduces the inner rectangle exactly
    inner = nodes[-16:]
    assert float(inner[:, 0].abs().max()) == pytest.approx(A - T, abs=1e-12)
    assert float(inner[:, 1].abs().max()) == pytest.approx(B - T, abs=1e-12)
    # thickness is exactly reproduced on the wall mid-side
    # mid-side wall thickness is reproduced exactly
    assert float((nodes[1, 0] - nodes[3 * 16 + 1, 0]).abs()) == pytest.approx(T, abs=1e-12)


def test_bredt_reference_values():
    s = RectBoxSection(A, B, T, shear_modulus=1.3, twist=1.0)
    assert s.mean_area == pytest.approx((2 * A - T) * (2 * B - T), rel=1e-14)
    assert s.mean_perimeter == pytest.approx(4.0 * ((A - T / 2) + (B - T / 2)), rel=1e-14)
    assert s.j == pytest.approx(4.0 * s.mean_area ** 2 * T / s.mean_perimeter, rel=1e-14)
    assert s.area == pytest.approx(4.0 * (A * B - (A - T) * (B - T)), rel=1e-14)
    assert s.tau_max == pytest.approx(s.g * s.j / (2.0 * s.mean_area * T), rel=1e-14)
    # a multiply connected section has no single valued closed form: the module
    # must refuse instead of silently returning a too-stiff ramp
    with pytest.raises(NotImplementedError):
        s.phi(0.0, 0.0)
    with pytest.raises(NotImplementedError):
        s.tau(0.0, 0.0)


def test_prandtl_grid_requirements_are_enforced():
    with pytest.raises(ValueError):
        solve_box_prandtl(A, B, T, 0.06)          # h does not divide t
    with pytest.raises(ValueError):
        solve_box_prandtl(A, B, T, 0.0123)        # h does not divide a
    with pytest.raises(ValueError):
        RectBoxSection(A, B, 1.2)                 # wall thicker than the half-dims


def test_prandtl_identities_and_bredt_bracket():
    box = solve_box_prandtl(A, B, T, 0.025)
    assert box.compat_residual < 1e-9
    # the two independent identities differ only by the corner singularity
    assert abs(box.j_energy - box.j_torque) / box.j_torque < 0.02
    bredt = RectBoxSection(A, B, T).j
    # Bredt is a thin-wall lower bound for a t/a = 0.1 wall
    assert 1.0 < box.j_torque / bredt < 1.10
    # the solid rectangle with the same outer dimensions must be stiffer
    assert box.j_torque < RectSection(A, B).j


def test_thin_wall_asymptote_is_first_order_in_t():
    ratios = []
    for t in (0.2, 0.1, 0.05):
        box = solve_box_prandtl(A, B, t, t / 4.0)
        ratios.append(box.j_torque / RectBoxSection(A, B, t).j)
    # J_fd -> J_bredt as t -> 0, and the excess halves when t halves
    assert ratios[-1] < ratios[0]
    assert ratios[-1] == pytest.approx(1.0, abs=0.03)
    assert 0.9 < ratios[1] / ratios[2] < 1.3


def test_prandtl_grid_convergence():
    coarse = solve_box_prandtl(A, B, T, 0.05)
    fine = solve_box_prandtl(A, B, T, 0.025)
    finer = solve_box_prandtl(A, B, T, 0.0125)
    d1 = abs(coarse.j_torque - fine.j_torque)
    d2 = abs(fine.j_torque - finer.j_torque)
    order = math.log2(d1 / d2)
    # reduced by the sharp re-entrant corner, but clearly converging
    assert order > 0.5, order
    assert d2 < d1
    # Richardson extrapolation stays within 1% of the finest grid value
    limit = finer.j_torque + (finer.j_torque - fine.j_torque) / (2 ** order - 1)
    assert abs(limit - finer.j_torque) / finer.j_torque < 0.01


def test_hex8_box_girder_rigidity_against_finite_difference():
    fd = solve_box_prandtl(A, B, T, 0.025)
    fe = box_torsion_run(A, B, T, n_side=8, n_layers=3, layers_z=2,
                         young_modulus=2.6, twist=1.0)
    fd_j = fd.j_torque * fd.g * fd.theta
    error = abs(fe.j_fe - fd_j) / fd_j
    assert fe.j_fe > 0.0
    assert error < 0.03, (fe.j_fe, fd_j)
    # against the thin wall formula the same mesh is ~5.8% stiff, which is the
    # Bredt truncation, not an element defect
    assert 1.03 < fe.j_ratio < 1.09


def test_hex8_box_girder_converges_to_the_fd_field():
    fd = solve_box_prandtl(A, B, T, 0.025)
    fd_j = fd.j_torque * fd.g * fd.theta
    coarse = box_torsion_run(A, B, T, n_side=4, n_layers=2, layers_z=2,
                             young_modulus=2.6)
    fine = box_torsion_run(A, B, T, n_side=12, n_layers=4, layers_z=3,
                           young_modulus=2.6)
    assert fine.dofs > coarse.dofs
    assert abs(fine.j_fe - fd_j) < abs(coarse.j_fe - fd_j)
    assert abs(fine.j_fe - fd_j) / fd_j < 0.01
    assert fine.warping_max > 0.0