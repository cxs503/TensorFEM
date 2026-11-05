"""Verification of the Boussinesq/Love half-space oracle and the truncated quarter model.

Two independent layers of evidence:

* the two closed forms (Boussinesq surface settlement, Newmark corner stress) are
  graded *against a composite Gauss quadrature of the point-load kernels* -- no
  closed form is trusted on its own, and the quadrature has no branch cut to get
  wrong;
* the merged HEX8 path is graded against both, on an unbounded problem truncated
  to a finite box, so the test also pins down that the truncation is applied with
  the weakest admissible constraints (rollers) rather than a clamped bottom.

The heavy refinement ladder lives behind ``TENSORFEM_RUN_SLOW_HALFSPACE=1``.
"""
import math
import os

import numpy as np
import pytest
import torch

from tensorfem.elastic_halfspace import (
    HalfSpaceResult,
    auto_growth,
    corner_vertical_stress,
    graded_axis,
    graded_box_mesh,
    influence,
    point_load_settlement,
    settlement,
    settlement_study,
    solve_halfspace_patch,
    vertical_stress,
)

# --------------------------------------------------------------------------- #
# reference quadrature of the point-load kernels (independent of the closed forms)
# --------------------------------------------------------------------------- #
_GAUSS_X, _GAUSS_W = np.polynomial.legendre.leggauss(10)


def stress_quadrature(pressure, lx, ly, x, y, z, panels=40):
    """``int_A (3p/2pi) z^3 / (r^2+z^2)^{5/2} dA`` -- Boussinesq point-load kernel."""
    total = 0.0
    for i in range(panels):
        x0, x1 = i * lx / panels, (i + 1) * lx / panels
        gx = 0.5 * (x1 - x0) * _GAUSS_X + 0.5 * (x1 + x0)
        wx = 0.5 * (x1 - x0) * _GAUSS_W
        for j in range(panels):
            y0, y1 = j * ly / panels, (j + 1) * ly / panels
            gy = 0.5 * (y1 - y0) * _GAUSS_X + 0.5 * (y1 + y0)
            wy = 0.5 * (y1 - y0) * _GAUSS_W
            gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
            r2 = (gxx - x) ** 2 + (gyy - y) ** 2 + z * z
            total += float((3.0 / (2.0 * math.pi) * pressure * z ** 3
                            * np.outer(wx, wy) / r2 ** 2.5).sum())
    return total


def settlement_quadrature(pressure, young, poisson, lx, ly, x, y, panels=160):
    """``(p(1-nu^2)/pi E) int_A dA/r`` -- valid for field points outside the patch."""
    total = 0.0
    for i in range(panels):
        x0, x1 = i * lx / panels, (i + 1) * lx / panels
        gx = 0.5 * (x1 - x0) * _GAUSS_X + 0.5 * (x1 + x0)
        wx = 0.5 * (x1 - x0) * _GAUSS_W
        for j in range(panels):
            y0, y1 = j * ly / panels, (j + 1) * ly / panels
            gy = 0.5 * (y1 - y0) * _GAUSS_X + 0.5 * (y1 + y0)
            wy = 0.5 * (y1 - y0) * _GAUSS_W
            gxx, gyy = np.meshgrid(gx, gy, indexing="ij")
            r = np.sqrt((gxx - x) ** 2 + (gyy - y) ** 2)
            total += float((np.outer(wx, wy) / r).sum())
    return pressure * (1.0 - poisson ** 2) / (math.pi * young) * total


# --------------------------------------------------------------------------- #
# closed-form oracles
# --------------------------------------------------------------------------- #
def _mixed_derivative(u, v, h):
    return (influence(u + h, v + h) - influence(u + h, v - h)
            - influence(u - h, v + h) + influence(u - h, v - h)) / (4.0 * h * h)


def test_influence_is_the_antiderivative_of_the_point_load_kernel():
    h = 1e-4
    for u, v in ((0.3, 0.7), (1.0, 1.0), (2.5, 0.4), (0.05, 3.0)):
        # Richardson-extrapolated central difference: kills the O(h^2) term so the
        # identity is checked to 1e-9 rather than to the stencil's own accuracy
        mixed = (4.0 * _mixed_derivative(u, v, 0.5 * h) - _mixed_derivative(u, v, h)) / 3.0
        assert mixed == pytest.approx(1.0 / math.hypot(u, v), rel=1e-7)


def test_influence_is_homogeneous_of_degree_one():
    for u, v, lam in ((0.7, 1.3, 3.7), (2.0, 0.25, 0.1)):
        assert influence(lam * u, lam * v) == pytest.approx(lam * influence(u, v), rel=1e-12)


def test_corner_and_centre_settlement_of_a_square_patch():
    p, e, nu, a = 1.7e6, 2.0e10, 0.3, 1.5
    corner = settlement(p, e, nu, a, a, a, a)
    centre = settlement(p, e, nu, 2 * a, 2 * a, a, a)
    # the centre is the closed form's own superposition identity: four quadrants
    assert centre == pytest.approx(4.0 * corner, rel=1e-12)
    # order of magnitude: the same patch cannot settle more than a few times the
    # classic circular footing 2(1-nu^2)pa/E
    circular = 2.0 * (1.0 - nu ** 2) * p * a / e
    assert 1.0 < centre / circular < 1.5


def test_settlement_matches_point_load_quadrature_outside_the_patch():
    p, e, nu, a = 1.0, 1.0, 0.3, 1.0
    for x, y in ((3.0, 3.0), (2.5, 0.0), (0.0, 4.0)):
        exact = settlement(p, e, nu, 2 * a, 2 * a, x, y)
        quad = settlement_quadrature(p, e, nu, 2 * a, 2 * a, x, y, panels=60)
        assert exact == pytest.approx(quad, rel=2e-6)


def test_settlement_scales_linearly_with_pressure_and_inversely_with_modulus():
    args = dict(lx=2.0, ly=2.0, x=1.0, y=1.0)
    base = settlement(1.0, 1.0, 0.3, **args)
    assert settlement(4.0, 1.0, 0.3, **args) == pytest.approx(4.0 * base, rel=1e-12)
    assert settlement(1.0, 4.0, 0.3, **args) == pytest.approx(base / 4.0, rel=1e-12)
    assert settlement(1.0, 1.0, 0.3, 4.0, 4.0, 2.0, 2.0) == pytest.approx(2.0 * base, rel=1e-12)


def test_square_patch_centre_matches_the_equal_area_circular_footing():
    """Love (1929) gives ``2(1-nu^2) p R / E`` for a flexible circular footing of
    radius ``R``; a square footing of equal area settles within ~1% of it."""
    p, e, nu, a = 1.0, 1.0, 0.3, 1.0
    square = settlement(p, e, nu, 2 * a, 2 * a, a, a)
    radius = 2 * a / math.sqrt(math.pi)
    circular = 2.0 * (1.0 - nu ** 2) * p * radius / e
    assert square == pytest.approx(circular, rel=0.01)


def test_point_load_settlement_decays_like_one_over_r():
    load, e, nu = 5.0, 2.0e11, 0.3
    near = point_load_settlement(load, e, nu, 1.0)
    far = point_load_settlement(load, e, nu, 4.0)
    assert far == pytest.approx(near / 4.0, rel=1e-12)


def test_vertical_stress_matches_point_load_quadrature():
    p, a = 1.0, 1.0
    cases = ((0.25, 0.25, 0.05), (0.25, 0.25, 0.5), (1.0, 1.0, 1.0),
             (3.0, 3.0, 1.0), (0.5, 0.5, 2.0), (-0.4, 1.4, 0.8))
    for x, y, z in cases:
        closed = vertical_stress(p, 2 * a, 2 * a, x, y, z)
        quad = stress_quadrature(p, 2 * a, 2 * a, x, y, z)
        assert closed == pytest.approx(quad, rel=1e-8, abs=1e-12)


def test_vertical_stress_deep_branch_uses_atan2_not_arctan():
    # m^2 n^2 > m^2 + n^2 + 1 for a wide patch seen from a shallow depth: the
    # textbook arctan form loses a pi there and returns a negative stress.
    p, lx, ly, z = 1.0, 10.0, 10.0, 1.0
    m, n = lx / z, ly / z
    assert m * m * n * n > m * m + n * n + 1.0
    closed = corner_vertical_stress(p, lx, ly, z)
    quad = stress_quadrature(p, lx, ly, lx, ly, z, panels=160)
    assert closed == pytest.approx(quad, rel=1e-7)
    assert closed > 0.0


def test_vertical_stress_is_bounded_by_the_pressure_and_decays():
    p, a = 1.0, 1.0
    previous = math.inf
    for z in (0.05, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0):
        value = vertical_stress(p, 2 * a, 2 * a, 0.3, 0.3, z)
        assert 0.0 < value <= p
        assert value < previous
        previous = value
    # far field must fall back to the point-load decay P/(pi z^2) * 3 ... with
    # P = p*(2a)^2 = 4 and the total vertical force carried by any deep plane = P
    deep = vertical_stress(p, 2 * a, 2 * a, 1.0, 1.0, 40.0)
    assert deep == pytest.approx(3.0 * 4.0 / (2.0 * math.pi * 40.0 ** 2), rel=0.05)


# --------------------------------------------------------------------------- #
# mesh helpers
# --------------------------------------------------------------------------- #
def test_graded_axis_keeps_the_first_cell_and_lands_on_the_extent():
    axis = graded_axis(0.25, 4, None, 12, 20.0)
    assert axis[0] == 0.0 and axis[-1] == pytest.approx(20.0, rel=1e-12)
    assert float(axis[1] - axis[0]) == pytest.approx(0.25, rel=1e-12)
    assert torch.all(axis[1:] > axis[:-1])
    assert auto_growth(0.25, 12, 19.0) > 1.0


def test_graded_axis_rejects_impossible_requests():
    with pytest.raises(ValueError):
        auto_growth(0.25, 4, 0.5)          # four cells of 0.25 cannot reach 0.5
    with pytest.raises(ValueError):
        graded_axis(0.0, 2, None, 4, 10.0)
    with pytest.raises(ValueError):
        graded_axis(0.25, 2, 1.0, 4, 10.0)  # ratio must exceed one
    axis = graded_axis(0.5, 2, 1.3, 6, 10.0)
    assert axis[-1] == pytest.approx(10.0, rel=1e-12)


def test_graded_box_mesh_has_positive_jacobians():
    from tensorfem.solid3d import elasticity_matrix_3d, hex8_stiffness

    xs = graded_axis(0.25, 2, None, 6, 6.0)
    nodes, cells, shape = graded_box_mesh(xs, xs.clone(), xs.clone())
    assert shape == (8, 8, 8) and len(nodes) == 9 ** 3 and len(cells) == 512
    modulus = elasticity_matrix_3d(nodes.new_tensor(1.0).expand(512),
                                   nodes.new_tensor(0.3).expand(512))
    ke, _ = hex8_stiffness(nodes[cells], modulus)   # raises on a non-positive Jacobian
    assert ke.shape == (512, 24, 24)
    assert torch.isfinite(ke).all()


# --------------------------------------------------------------------------- #
# truncated quarter model against the exact half-space
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def coarse() -> HalfSpaceResult:
    return solve_halfspace_patch(n_fine=2, n_grow=8, nz_layers=10, rtol=1e-8)


def test_quarter_model_balances_the_applied_load(coarse):
    # the reaction sum is the only check a closed-body benchmark would also pass
    assert coarse.reaction_balance < 1e-6


def test_quarter_model_settlement_is_close_and_conservative(coarse):
    # a truncated, coarse model is stiffer than the half-space: it must *under*shoot
    assert 0.0 < coarse.w_center < coarse.w_center_exact
    assert 0.0 < coarse.w_corner < coarse.w_corner_exact
    assert coarse.center_error < 0.15
    assert coarse.corner_error < 0.20
    assert coarse.n_dofs == 3 * coarse.n_nodes


def test_quarter_model_vertical_stress_decays_and_is_compressive(coarse):
    assert all(value < 0.0 for value in coarse.stress_fe)
    magnitudes = [abs(value) for value in coarse.stress_fe]
    assert all(b < a for a, b in zip(magnitudes, magnitudes[1:]))
    # deep enough that the load looks like a point load, the coarse mesh converges
    deep = [(fe, ex) for d, fe, ex in zip(coarse.depths, coarse.stress_fe, coarse.stress_exact)
            if 3.0 <= d <= 12.0]
    assert deep
    for fe, ex in deep:
        assert abs(fe - ex) / abs(ex) < 0.20


def test_domain_truncation_shifts_settlement_past_every_self_check():
    """The point of the module: both boxes balance the load, only one is right.

    Measured (not assumed): squeezing the far field to four patch widths makes the
    surface settlement *larger* than the half-space value, because the load can no
    longer spread and the stress under the patch is amplified; opening it to twenty
    patch widths brings it below.  The truncation is therefore not a conservative
    error, and nothing in the residual or the reaction sum reveals it.
    """

    tight = solve_halfspace_patch(n_fine=2, n_grow=4, nz_layers=10, growth=1.3,
                                  lateral=4.0, rtol=1e-10)
    roomy = solve_halfspace_patch(n_fine=2, n_grow=10, nz_layers=10, growth=1.3,
                                  lateral=20.0, rtol=1e-10)
    assert tight.reaction_balance < 1e-6 and roomy.reaction_balance < 1e-6
    assert abs(tight.w_center - roomy.w_center) / roomy.w_center > 0.05
    assert tight.center_error > 0.10          # 4-patch box: 16% high, not low
    assert roomy.center_error < 0.08          # 20-patch box: 5% low


@pytest.mark.skipif(os.environ.get("TENSORFEM_RUN_SLOW_HALFSPACE") != "1",
                    reason="slow refinement ladder: set TENSORFEM_RUN_SLOW_HALFSPACE=1")
def test_refinement_ladder_converges_towards_boussinesq():
    results = settlement_study()
    assert len(results) == 3
    errors = [result.center_error for result in results]
    assert errors[0] > errors[1] > errors[2], errors
    assert errors[-1] < 0.06
    # observed order from the last two levels, then a Richardson estimate
    h1, h2 = results[-2].finest_spacing, results[-1].finest_spacing
    order = math.log((results[-2].w_center - results[-1].w_center) /
                     (results[-1].w_center - results[-1].w_center_exact)) / math.log(h1 / h2) \
        if abs(results[-1].w_center - results[-1].w_center_exact) > 1e-12 else 2.0
    assert 0.5 < order < 4.0
    extrapolated = (results[-1].w_center
                    + (results[-1].w_center - results[-2].w_center) / (2.0 ** order - 1.0))
    # measured plateau: 4.9% (n_fine=4) -> 4.7% (n_fine=8), so the extrapolated
    # value is a few percent out; see docs/elastic-halfspace-boussinesq-verification.md
    assert abs(extrapolated - results[-1].w_center_exact) / results[-1].w_center_exact < 0.04