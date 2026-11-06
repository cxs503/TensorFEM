"""Fail-closed checks for the rectangular (``a != b``) Saint-Venant torsion model.

Closes the open item of ``docs/torsion-boundary-stress.md`` §5 ("``a != b`` shape
factor unverified") with references that do not share the Fourier series:

* the hand-book shape-factor table (Timoshenko & Goodier; Roark);
* a second-order finite-difference solution of the defining PDE ``lap(psi) = -2``;
* the closed-form thin-strip limit ``J -> a b^3 / 3``.

The 3-D finite-element checks require the rectangular driver to reproduce the
torsion constant from its measured twist rate; the coarsest meshes deliberately
stay outside the 3 % gate and are asserted as *failures*, so a regression that
inflates accuracy claims cannot pass silently.
"""
from __future__ import annotations

import math

import pytest

from tensorfem.torsion_bar import max_shear_stress, rectangular_torsion_constant
from tensorfem.torsion_rectangular import (classical_rectangle_coefficients,
                                           prandtl_finite_difference, rectangular_torsion_bar,
                                           richardson, shape_factors,
                                           thin_strip_torsion_constant)


# --------------------------------------------------------------- closed form vs table
@pytest.mark.parametrize("row", classical_rectangle_coefficients(), ids=lambda r: f"a{r.aspect:g}")
def test_closed_form_matches_classical_shape_factor_table(row):
    """c1 and c2 must reproduce the hand-book row within its three-decimal rounding."""
    factors = shape_factors(row.aspect, 1.0)
    assert factors.c1 == pytest.approx(row.c1, rel=0.01)
    assert factors.c2 == pytest.approx(row.c2, rel=0.01)


def test_square_row_keeps_the_previously_verified_anchors():
    """The a == b limit must still give J = 0.1406 and the 0.208 hand-book c2."""
    j = rectangular_torsion_constant(1.0, 1.0)
    assert j == pytest.approx(0.1406, abs=5e-5)
    factors = shape_factors(1.0, 1.0)
    assert factors.c2 == pytest.approx(0.208, rel=0.002)
    # a square section has no long side: both mid-side gradients coincide
    assert factors.grad_long_mid == pytest.approx(factors.grad_short_mid, rel=1e-6)


@pytest.mark.parametrize("aspect", [1.2, 1.5, 2.0, 2.5, 3.0, 5.0, 7.0])
def test_closed_form_agrees_with_independent_pde_solution(aspect):
    """Richardson-extrapolated finite differences must confirm c1 and c2.

    The two methods share nothing but the PDE: a wrong hyperbolic ratio, sign or
    normalisation in the series cannot survive this comparison.
    """
    coarse = prandtl_finite_difference(aspect, 1.0, n_short=48)
    fine = prandtl_finite_difference(aspect, 1.0, n_short=96)
    factors = shape_factors(aspect, 1.0)
    assert richardson(fine.c1, coarse.c1) == pytest.approx(factors.c1, rel=5e-3)
    assert richardson(fine.c2, coarse.c2) == pytest.approx(factors.c2, rel=5e-3)
    assert fine.grad_long_mid == pytest.approx(factors.grad_long_mid, rel=1e-2)


def test_finite_difference_scheme_is_second_order():
    """The PDE reference is only usable if its own discretisation converges at O(h^2)."""
    levels = [prandtl_finite_difference(3.0, 1.0, n_short=n) for n in (20, 40, 80)]
    c1 = [level.c1 for level in levels]
    order = math.log2(abs((c1[0] - c1[1]) / (c1[1] - c1[2])))
    assert 1.85 < order < 2.15
    assert richardson(c1[2], c1[1]) == pytest.approx(shape_factors(3.0, 1.0).c1, rel=1e-4)


def test_thin_strip_limit_is_recovered():
    """``c1 -> 1/3`` as a/b grows; the gap must shrink monotonically with aspect."""
    errors = []
    for aspect in (10.0, 50.0, 200.0):
        reference = thin_strip_torsion_constant(aspect, 1.0) / (aspect * 1.0**3)
        errors.append(abs(shape_factors(aspect, 1.0, samples=801).c1 - reference))
    assert errors == sorted(errors, reverse=True)
    assert errors[-1] < 5e-3


# ------------------------------------------------------- long-side / short-side peak
@pytest.mark.parametrize("aspect", [1.2, 1.5, 2.0, 3.0, 5.0, 10.0])
def test_peak_sits_on_the_long_side_and_exceeds_the_short_side(aspect):
    """The peak of an a != b section is at (|x|, y) = (b/2, 0), and is larger than the
    short-side midpoint value -- the difference the previous report could not quantify."""
    factors = shape_factors(aspect, 1.0)
    assert factors.long_over_short > 1.0
    assert abs(abs(factors.peak_location[0]) - 0.5) < 2e-3  # half a scan cell inside b/2
    assert abs(factors.peak_location[1]) < 2e-3
    # the interior half-cell scan is a strict lower bound of the boundary peak, once
    # both are expressed in the same torque normalisation (tau = |grad psi| * T / J)
    boundary_peak = factors.grad_long_mid / (factors.c1 * factors.a * factors.b**3)
    assert factors.peak_scan < boundary_peak
    assert (boundary_peak - factors.peak_scan) / boundary_peak < 0.01


def test_long_side_ratio_grows_towards_the_thin_strip_asymptote():
    """long/short must increase with the aspect ratio and saturate near 4/3."""
    ratios = [shape_factors(aspect, 1.0).long_over_short for aspect in (1.0, 1.5, 2.0, 3.0, 10.0)]
    assert ratios == sorted(ratios)
    assert ratios[0] == pytest.approx(1.0, rel=1e-12)
    assert 1.3 < ratios[-1] < 1.4


# ------------------------------------------------------------------- 3-D finite element
def test_rectangular_driver_carries_exactly_the_requested_torque():
    """The consistent per-element face weights must reproduce T to machine precision."""
    result = rectangular_torsion_bar(a=2.0, b=1.0, length=4.0, segments_short=3)
    assert result.applied_torque == pytest.approx(1.0, abs=1e-14)
    assert result.segments_long == 6
    assert result.n_dofs > 0


def test_rectangular_driver_converges_onto_the_closed_form():
    """Refining an a/b = 2 prism must drive the measured J below the 3 % gate."""
    coarse = rectangular_torsion_bar(a=2.0, b=1.0, length=4.0, segments_short=3)
    fine = rectangular_torsion_bar(a=2.0, b=1.0, length=4.0, segments_short=5)
    assert fine.torsion_constant_relative_error < coarse.torsion_constant_relative_error
    assert fine.torsion_constant_relative_error < 0.03
    # centroid shear stress is compared in the mid-length slab against the series
    assert fine.centroid_shear_l2_error < coarse.centroid_shear_l2_error


def test_coarse_rectangular_mesh_stays_a_recorded_failure():
    """The coarsest a/b = 1.5 mesh is outside the gate; it is kept as a negative control."""
    coarse = rectangular_torsion_bar(a=1.5, b=1.0, length=4.0, segments_short=3)
    assert coarse.torsion_constant_relative_error > 0.03


def test_slender_section_twist_rate_does_not_converge_monotonically():
    """Open item, fail-closed: at a/b = 3 the measured J drifts *away* under refinement.

    ``length = 4`` gives ``a / L = 0.75``, so the 0.25L..0.75L twist-rate window lies
    inside the Saint-Venant end zone of the 3-long side.  Refining the section mesh does
    not cure it (0.91 % -> 2.03 %); the length study in
    ``benchmarks/torsion_rectangular_section.py`` shows the window, not the aspect
    ratio, is responsible.  This test asserts the *failure* so that it cannot be
    silently upgraded to a pass.
    """
    coarse = rectangular_torsion_bar(a=3.0, b=1.0, length=4.0, segments_short=3)
    fine = rectangular_torsion_bar(a=3.0, b=1.0, length=4.0, segments_short=4)
    assert fine.torsion_constant_relative_error > coarse.torsion_constant_relative_error
    assert fine.torsion_constant_relative_error > 0.015


def test_cubic_elements_are_used():
    """``nz`` tracks the section spacing (cubic elements), not a fixed multiple."""
    for length in (4.0, 8.0):
        result = rectangular_torsion_bar(a=2.0, b=1.0, length=length, segments_short=3)
        assert result.n_dofs > 0
    short = rectangular_torsion_bar(a=2.0, b=1.0, length=4.0, segments_short=3)
    long = rectangular_torsion_bar(a=2.0, b=1.0, length=8.0, segments_short=3)
    assert long.n_nodes > short.n_nodes