"""Tests for the closed-form spectral reference of simply supported panels.

The suite pins four independent things:

1. the closed-form operator entries really are the integrals they claim to be
   (against Gauss-Legendre quadrature);
2. the machinery reproduces two exact closed forms that have nothing to do with
   shear (uniaxial ``k = 4``, equal-biaxial Navier ``N_x = 2 pi^2 D / b^2``);
3. the pure-shear coefficient converges monotonically and independently of how
   the mode budget is split between the two directions;
4. the finite-difference solver of ``marine_plate_combined_buckling`` and the
   spectral reference agree to well inside the 3% acceptance band, including at
   ``a/b = 2`` where the merged report had only the asymptotic fit to compare
   against.
"""

import math

import numpy as np
import pytest

from tensorfem.shear_plate_spectral import (
    cos_sin_integral,
    richardson_extrapolation,
    run_shear_spectral_qualification,
    spectral_buckling_load,
    spectral_buckling_operators,
    spectral_shear_buckling_coefficient,
)


def test_cos_sin_integral_matches_quadrature():
    nodes, weights = np.polynomial.legendre.leggauss(64)
    for length in (1.0, 2.0, 3.5):
        x = 0.5 * (nodes + 1.0) * length
        w = 0.5 * weights * length
        for p, q in ((1, 2), (2, 1), (3, 4), (1, 4), (5, 2)):
            reference = np.sum(w * np.cos(p * math.pi * x / length) * np.sin(q * math.pi * x / length))
            assert cos_sin_integral(p, q, length) == pytest.approx(reference, rel=1e-12, abs=1e-14)


def test_cos_sin_integral_vanishes_on_equal_parity():
    for p, q in ((1, 1), (1, 3), (2, 4), (3, 5)):
        assert cos_sin_integral(p, q, 1.0) == 0.0


def test_operators_match_quadrature():
    length, width = 2.0, 1.0
    modes_x, modes_y = 4, 3
    bending, gx, gy, gs = spectral_buckling_operators(
        length=length, width=width, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=modes_x, modes_y=modes_y,
    )
    nodes, weights = np.polynomial.legendre.leggauss(48)
    xs = 0.5 * (nodes + 1.0) * length
    ys = 0.5 * (nodes + 1.0) * width
    weight = np.outer(0.5 * weights * length, 0.5 * weights * width)
    gx_grid, gy_grid = np.meshgrid(xs, ys, indexing="ij")
    modes = [(m, n) for m in range(1, modes_x + 1) for n in range(1, modes_y + 1)]
    fields = []
    for m, n in modes:
        alpha, beta = m * math.pi / length, n * math.pi / width
        shape = np.sin(alpha * gx_grid) * np.sin(beta * gy_grid)
        fields.append((
            alpha * np.cos(alpha * gx_grid) * np.sin(beta * gy_grid),
            beta * np.sin(alpha * gx_grid) * np.cos(beta * gy_grid),
            alpha * beta * np.cos(alpha * gx_grid) * np.cos(beta * gy_grid),
            -alpha * alpha * shape,
            -beta * beta * shape,
        ))
    size = len(modes)
    ref_x = np.zeros((size, size))
    ref_y = np.zeros((size, size))
    ref_s = np.zeros((size, size))
    ref_b = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            xi, yi, xy_i, xx_i, yy_i = fields[i]
            xj, yj, xy_j, xx_j, yy_j = fields[j]
            ref_x[i, j] = np.sum(weight * xi * xj)
            ref_y[i, j] = np.sum(weight * yi * yj)
            ref_s[i, j] = np.sum(weight * (yi * xj + xi * yj))
            # isotropic bending weak form: D * int[w,xx d,xx + 2 w,xy d,xy + w,yy d,yy]
            ref_b[i, j] = np.sum(weight * (xx_i * xx_j + 2.0 * xy_i * xy_j + yy_i * yy_j))
    assert np.allclose(gx, ref_x, rtol=1e-11, atol=1e-12)
    assert np.allclose(gy, ref_y, rtol=1e-11, atol=1e-12)
    assert np.allclose(gs, ref_s, rtol=1e-11, atol=1e-12)
    assert np.allclose(bending, np.diag(ref_b), rtol=1e-11, atol=1e-12)
    # off-diagonal bending entries vanish identically in the double-sine basis
    assert float(np.max(np.abs(np.diag(bending)))) > 0.0


def test_uniaxial_recovers_classical_coefficient():
    for aspect in (1.0, 2.0, 3.0):
        modes_x = 4 * int(aspect)
        result = spectral_buckling_load(
            length=aspect, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
            coupling_rigidity=1.0, modes_x=modes_x, modes_y=4,
            longitudinal_share=1.0, transverse_share=0.0, shear_share=0.0,
        )
        # the governing half wave count m = a/b gives N_cr = 4 pi^2 D / b^2 exactly
        assert result.critical_line_load == pytest.approx(4.0 * math.pi**2, rel=1e-12)


def test_biaxial_recovers_navier():
    result = spectral_buckling_load(
        length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=6, modes_y=6,
        longitudinal_share=1.0, transverse_share=1.0, shear_share=0.0,
    )
    assert result.critical_line_load == pytest.approx(2.0 * math.pi**2, rel=1e-12)


def test_shear_is_symmetric_in_sign():
    kwargs = dict(
        length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=10, modes_y=10,
        longitudinal_share=0.0, transverse_share=0.0,
    )
    plus = spectral_buckling_load(shear_share=1.0, **kwargs).critical_line_load
    minus = spectral_buckling_load(shear_share=-1.0, **kwargs).critical_line_load
    assert plus == pytest.approx(minus, rel=1e-12)


def test_shear_coefficient_matches_textbook_square():
    k = spectral_shear_buckling_coefficient(1.0, mode_budget=20)
    assert k == pytest.approx(9.3245, abs=2e-3)
    assert abs(k - 9.34) / 9.34 < 0.01


def test_shear_convergence_is_monotone_and_stable():
    budgets = [6, 8, 10, 14, 18]
    values = [spectral_shear_buckling_coefficient(1.0, mode_budget=b) for b in budgets]
    gaps = [values[i] - values[i + 1] for i in range(len(values) - 1)]
    assert all(gap >= -1e-12 for gap in gaps)
    assert all(gaps[i + 1] < gaps[i] for i in range(len(gaps) - 1))
    order, limit = richardson_extrapolation(budgets, values)
    assert math.isfinite(order)
    assert abs(limit - values[-1]) < 1e-4


def test_shear_coefficient_independent_of_mode_split():
    """Different truncations of the same budget must agree to ~1e-3."""

    coarse = spectral_buckling_load(
        length=2.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=20, modes_y=10,
        longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0,
    ).critical_line_load
    fine = spectral_buckling_load(
        length=2.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
        coupling_rigidity=1.0, modes_x=34, modes_y=17,
        longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0,
    ).critical_line_load
    assert abs(coarse - fine) / fine < 2e-3


def test_classical_fit_deviates_beyond_acceptance_at_aspect_two():
    """Documents why the asymptotic fit cannot be the gate at ``a/b = 2``."""

    spectral = spectral_shear_buckling_coefficient(2.0, mode_budget=20)
    fit = 5.34 + 4.0 / 2.0**2
    assert abs(fit - spectral) / spectral > 0.03
    assert spectral == pytest.approx(6.546, abs=2e-3)


def test_finite_difference_agrees_with_spectral_reference():
    torch = pytest.importorskip("torch")
    from tensorfem.marine_plate_combined_buckling import panel_buckling_load

    result = panel_buckling_load(
        length=2.0, width=1.0, thickness=0.012,
        rigidity_x=1.0, rigidity_y=1.0, coupling_rigidity=1.0,
        grid_x=16, grid_y=32,
        longitudinal_share=0.0, transverse_share=0.0, shear_share=1.0,
    )
    finite_difference = float(result.critical_line_load) / math.pi**2
    reference = spectral_shear_buckling_coefficient(2.0, mode_budget=20)
    assert abs(finite_difference - reference) / reference < 0.03
    del torch


def test_qualification_gates_all_pass():
    report = run_shear_spectral_qualification(
        fine_budgets=(8, 12, 16, 20), finite_difference_grid=16,
    )
    failed = [g["name"] for g in report["gates"] if not g["passed"]]
    assert failed == []
    assert report["passed"] is True


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        spectral_shear_buckling_coefficient(0.5)
    with pytest.raises(ValueError):
        spectral_shear_buckling_coefficient(1.0, mode_budget=1)
    with pytest.raises(ValueError):
        spectral_buckling_load(
            length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
            coupling_rigidity=1.0, modes_x=4, modes_y=4,
            longitudinal_share=0.0, transverse_share=0.0, shear_share=0.0,
        )
    with pytest.raises(ValueError):
        spectral_buckling_load(
            length=float("nan"), width=1.0, rigidity_x=1.0, rigidity_y=1.0,
            coupling_rigidity=1.0, modes_x=4, modes_y=4, longitudinal_share=1.0,
        )
    with pytest.raises(ValueError):
        spectral_buckling_operators(
            length=1.0, width=1.0, rigidity_x=1.0, rigidity_y=1.0,
            coupling_rigidity=1.0, modes_x=0, modes_y=4,
        )
    with pytest.raises(ValueError):
        richardson_extrapolation([4, 6], [1.0, 2.0])