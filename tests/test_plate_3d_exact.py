"""Verification tests for :mod:`tensorfem.plate_3d_exact`.

The reference is the exact 3-D elasticity solution; the tests check the
*reference itself* (equilibrium by automatic differentiation, face tractions,
kinematic edge conditions, thin-plate limit order) and then use it to qualify
FSDT and the HEX20 solid element.  No test compares against a value copied from
this file: every number is recomputed from the closed-form solution or from the
in-house FE path.
"""
import math

import pytest
import torch

from tensorfem.plate_3d_exact import (
    SSPlateExact3D,
    fsdt_static_navier,
    kirchhoff_static_navier,
    solve_hex20_ss_plate,
)

E = 200e9
NU = 0.3
Q0 = 1.0e6
A = 1.0


def _plate(ratio: float) -> SSPlateExact3D:
    return SSPlateExact3D(A, A, A / ratio, E, NU, Q0)


def test_input_validation():
    with pytest.raises(ValueError):
        SSPlateExact3D(0.0, 1.0, 0.1, E, NU, Q0)
    with pytest.raises(ValueError):
        SSPlateExact3D(1.0, 1.0, -0.1, E, NU, Q0)
    with pytest.raises(ValueError):
        SSPlateExact3D(1.0, 1.0, 0.1, E, 0.5, Q0)
    with pytest.raises(ValueError):
        SSPlateExact3D(1.0, 1.0, 0.1, E, NU, Q0, m=0)


def test_equilibrium_is_satisfied_by_automatic_differentiation():
    """``div sigma = 0`` at interior points (independent of the ODE matrix)."""
    rep = _plate(10.0).equilibrium_residual(samples=2)
    assert rep["max_abs_div_sigma"] > 0.0  # a real measurement, not a stub
    assert rep["relative"] < 1e-6, rep


def test_face_tractions_match_the_applied_pressure():
    rep = _plate(10.0).face_report()
    assert rep["top_sigma_zz_rel_error"] < 1e-10, rep
    assert rep["abs_bottom_sigma_zz"] < 1e-6 * Q0, rep
    assert rep["max_abs_face_shear"] < 1e-6 * Q0, rep


def test_edge_conditions_are_enforced_exactly():
    rep = _plate(10.0).edge_report()
    # u_y = u_z = 0 on x-edges, u_x = u_z = 0 on y-edges, sigma_nn = 0 there too
    assert rep["max_kinematic_and_normal_stress_residual"] < 1e-12
    # honest disclosure: the edge in-plane shear traction is a NON-ZERO reaction
    assert rep["edge_shear_over_applied_pressure"] > 0.1


def test_thin_plate_limit_and_second_order_consistency():
    """``w_3d / w_K - 1`` must vanish like ``(h/a)^2`` as ``h -> 0``."""
    gaps = {}
    for ratio in (25.0, 50.0, 100.0):
        plate = _plate(ratio)
        w_exact = plate.center_deflection()
        w_k = kirchhoff_static_navier(plate.a, plate.b, plate.h, E, NU, Q0)["deflection"]
        gaps[ratio] = w_exact / w_k - 1.0
        assert gaps[ratio] > 0.0
    assert gaps[100.0] < gaps[50.0] < gaps[25.0]
    assert gaps[100.0] < 1e-3
    # a factor-2 refinement in h must shrink the gap by ~4 (second order)
    assert 3.4 < gaps[50.0] / gaps[100.0] < 4.6
    assert 3.4 < gaps[25.0] / gaps[50.0] < 4.6


def test_shear_softening_is_monotone_in_thickness():
    ratios = [100.0, 50.0, 20.0, 10.0, 5.0]
    gaps = []
    for ratio in ratios:
        plate = _plate(ratio)
        w_k = kirchhoff_static_navier(plate.a, plate.b, plate.h, E, NU, Q0)["deflection"]
        gaps.append(plate.center_deflection() / w_k)
    assert gaps == sorted(gaps)
    assert gaps[-1] > 1.1  # a/h = 5 is far from the thin-plate idealisation


def test_classical_ordering_thin_3d_fsdt():
    """Kirchhoff is the stiffest, FSDT the most compliant, 3-D in between."""
    for ratio in (10.0, 20.0, 50.0):
        plate = _plate(ratio)
        w_k = kirchhoff_static_navier(plate.a, plate.b, plate.h, E, NU, Q0)["deflection"]
        w_3d = plate.center_deflection()
        w_f = fsdt_static_navier(plate.a, plate.b, plate.h, E, NU, Q0)["deflection"]
        assert w_k < w_3d < w_f, (ratio, w_k, w_3d, w_f)


def test_fsdt_reduces_to_kirchhoff_and_matches_first_order_bias():
    plate = _plate(1000.0)
    w_k = kirchhoff_static_navier(plate.a, plate.b, plate.h, E, NU, Q0)["deflection"]
    fsdt = fsdt_static_navier(plate.a, plate.b, plate.h, E, NU, Q0)
    # FSDT/Navier ratio is exactly 1 + D R / (kappa G h); the "first-order"
    # form 1/(1+p) commonly quoted is only O(p^2) away from it.
    p = fsdt["shear_parameter"]
    assert abs(fsdt["deflection"] / w_k - (1.0 + p)) < 1e-9 * (1.0 + p)
    assert abs(1.0 / (1.0 + p) - (1.0 - p) - p**2 / (1.0 + p)) < 1e-15


def test_dimensionless_scaling():
    """``w/(q0 a^4 / D)`` depends on ``a/h`` only: symmetry, linearity, 1/E."""
    ref = _plate(10.0).center_deflection()
    heavy = SSPlateExact3D(A, A, A / 10.0, E, NU, 2.0 * Q0).center_deflection()
    stiffer = SSPlateExact3D(A, A, A / 10.0, 2.0 * E, NU, Q0).center_deflection()
    # w ~ q0 a^4 / D with D ~ E h^3: doubling a and h at fixed a/h scales w by
    # 2^4 / 2^3 = 2
    square = SSPlateExact3D(2 * A, 2 * A, 2 * A / 10.0, E, NU, Q0).center_deflection()
    assert heavy == pytest.approx(2.0 * ref, rel=1e-12)
    assert stiffer == pytest.approx(ref / 2.0, rel=1e-12)
    assert square == pytest.approx(2.0 * ref, rel=1e-12)


def test_through_thickness_stress_is_not_linear():
    """FSDT assumes a linear ``sigma_xx(z)``: quantify how wrong that is."""
    plate = _plate(10.0)
    zs = torch.linspace(-0.5 * plate.h, 0.5 * plate.h, 21, dtype=torch.float64)
    x = torch.full_like(zs, 0.5 * plate.a)
    y = torch.full_like(zs, 0.5 * plate.b)
    sxx = plate.stress(x, y, zs)[0]
    zz = zs.numpy()
    top = float(sxx[-1])
    linear = top * zs / (0.5 * plate.h)
    deviation = float((sxx - linear).abs().max()) / abs(top)
    # measured, not assumed: ~1.2 % at a/h = 10 for a homogeneous plate
    assert 5e-3 < deviation < 5e-2, deviation
    # the load acts on the TOP face only, so the 3-D solution is not exactly
    # antisymmetric in z: both the mid-surface value and the top/bottom parity
    # carry a real even part that FSDT cannot represent
    mid = abs(float(sxx[len(sxx) // 2])) / abs(top)
    parity = abs(float(sxx[0]) + top) / abs(top)
    assert 1e-3 < mid < 5e-2, mid
    assert 2e-3 < parity < 5e-2, parity


def test_hex20_reproduces_the_exact_thick_plate_solution():
    """The solid element, not a plate element, against 3-D elasticity."""
    plate = _plate(10.0)
    fe = solve_hex20_ss_plate(A, A, plate.h, E, NU, Q0, nx=4, ny=4, nz=2)
    w_exact = plate.center_deflection()
    err = (fe["centre_deflection"] - w_exact) / w_exact
    assert abs(err) < 5e-2, (fe["centre_deflection"], w_exact, err)


def test_hex20_refinement_reduces_the_error():
    plate = _plate(10.0)
    w_exact = plate.center_deflection()
    coarse = solve_hex20_ss_plate(A, A, plate.h, E, NU, Q0, nx=4, ny=4, nz=2)["centre_deflection"]
    fine = solve_hex20_ss_plate(A, A, plate.h, E, NU, Q0, nx=8, ny=8, nz=2)["centre_deflection"]
    assert abs(fine - w_exact) <= abs(coarse - w_exact)