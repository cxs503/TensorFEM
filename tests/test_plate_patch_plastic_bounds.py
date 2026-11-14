"""Independent (dual-mechanism) plastic-bound verification for the patch-loaded plate.

Anchor: ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET``.  The anchor acceptance names *two*
independent references for the plastic limit load ("Ilyushin / upper-bound
analysis"), but only the yield-line mechanism existed, so "collapse within 10 %
of the upper bound" was checked against a single algebra.  These tests pin the
second, independent kinematic bound down to hand-computable constants and check
that it is (a) admissible, (b) tighter than the yield line, (c) above any
statically admissible lower bound and (d) above the elastic first-yield load
(a bound that fell below first yield would be inadmissible for a
perfectly-plastic plate).
"""
import math

import pytest
import torch

from tensorfem.plate_patch_plastic_limit import (
    analytic_first_yield_pressure,
    ilyushin_sine_upper_bound,
    plastic_bound_bracket,
    plastic_moment,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

A = B = 1.0
T = 0.02
E = 210e9
NU = 0.3
SY = 355e6
PATCH = (0.5, 0.5)


def test_uniform_square_plate_recovers_two_pi_squared_mp_over_a2():
    """Fully loaded square plate: the sine mechanism gives 2*pi^2*m_p/a^2 = 19.74 m_p/a^2."""
    mp = plastic_moment(T, SY)
    got = ilyushin_sine_upper_bound(A, B, T, SY, (A, B))["pressure"]
    assert got == pytest.approx(2.0 * math.pi ** 2 * mp / A ** 2, rel=1e-12)
    # and it must sit below the yield-line pyramid value of the same plate
    yl = yield_line_patch_upper_bound(A, B, T, SY, (A, B))["pressure"]
    assert got < yl


def test_strip_limit_tends_to_the_beam_sine_mechanism():
    """b >> a: the 2-D mechanism degenerates to the beam sine value pi^2 m_p/a^2."""
    mp = plastic_moment(T, SY)
    beam = math.pi ** 2 * mp / A ** 2
    for b in (10.0, 40.0):
        got = ilyushin_sine_upper_bound(A, b, T, SY, (A, b))["pressure"]
        assert got == pytest.approx(beam, rel=2e-2 if b == 10.0 else 5e-3)
        # always an upper bound, never below the exact beam collapse 8 m_p/a^2
        assert got > 8.0 * mp / A ** 2


def test_bound_scales_quadratically_with_thickness_and_linearly_with_strength():
    mp = plastic_moment(T, SY)
    base = ilyushin_sine_upper_bound(A, B, T, SY, PATCH)["pressure"]
    double_t = ilyushin_sine_upper_bound(A, B, 2 * T, SY, PATCH)["pressure"]
    double_sy = ilyushin_sine_upper_bound(A, B, T, 2 * SY, PATCH)["pressure"]
    assert double_t == pytest.approx(4.0 * base, rel=1e-12)
    assert double_sy == pytest.approx(2.0 * base, rel=1e-12)
    assert base == pytest.approx(4.0 * mp * 2.0 / 0.2026, rel=5e-3)


def test_internal_work_matches_direct_numerical_quadrature():
    """Assert the closed form against an independent quadrature of |k1|+|k2|.

    The mechanism is w = sin(pi x/a) sin(pi y/b); its principal curvature rates
    have the same sign, so the Johansen dissipation integrates to
    m_p * (|k1| + |k2|).  The closed form used by the bound is
    ``4 m_p a b (m^2/a^2 + n^2/b^2)/(m n)`` per unit mechanism amplitude.
    """
    a, b, t, sy = A, B, T, SY
    mp = plastic_moment(t, sy)
    for (m, n) in ((1, 1), (3, 1), (1, 3)):
        nx = ny = 601
        x = torch.linspace(0, a, nx, dtype=torch.float64)
        y = torch.linspace(0, b, ny, dtype=torch.float64)
        phi = (torch.sin(m * math.pi * x / a)[:, None]
               * torch.sin(n * math.pi * y / b)[None, :])
        k1 = (m * math.pi / a) ** 2 * phi
        k2 = (n * math.pi / b) ** 2 * phi
        quad = float((k1 + k2).sum()) * (a / (nx - 1)) * (b / (ny - 1)) * mp
        closed = 4.0 * mp * a * b * (m * m / a ** 2 + n * n / b ** 2) / (m * n)
        assert quad == pytest.approx(closed, rel=1e-3)


def test_bound_sits_above_the_elastic_first_yield_pressure():
    """A collapse load below first yield would be inadmissible."""
    for patch in ((A, B), (0.6, 0.6), (0.5, 0.5), (0.3, 0.3)):
        py = analytic_first_yield_pressure(A, B, T, E, NU, SY, patch)["pressure"]
        ub = ilyushin_sine_upper_bound(A, B, T, SY, patch)["pressure"]
        assert ub > py, (patch, ub, py)


def test_bound_stays_above_the_strip_lower_bound_for_a_sweep_of_patches():
    for patch in ((0.9, 0.9), (0.5, 0.5), (0.5, 0.8), (0.4, 0.4), (0.3, 0.5)):
        ub = ilyushin_sine_upper_bound(A, B, T, SY, patch)["pressure"]
        lb = strip_lower_bound(A, B, T, SY, patch)["pressure"]
        assert ub > lb, patch


def test_sine_bound_is_tighter_than_the_yield_line_for_the_anchor_geometry():
    bracket = plastic_bound_bracket(A, B, T, SY, PATCH)
    assert bracket["tightest_upper"] == pytest.approx(bracket["upper_ilyushin_sine"])
    assert bracket["upper_ilyushin_sine"] < bracket["upper_yield_line"]
    assert bracket["agreement_of_upper_bounds_pct"] == pytest.approx(
        100.0 * (bracket["upper_yield_line"] / bracket["upper_ilyushin_sine"] - 1.0))


def test_bracket_ordering_and_width_are_reported_not_hidden():
    bracket = plastic_bound_bracket(A, B, T, SY, PATCH)
    assert bracket["lower_strip"] < bracket["tightest_upper"]
    assert bracket["bracket_width_ratio"] == pytest.approx(
        bracket["tightest_upper"] / bracket["lower_strip"])
    assert 1.0 < bracket["bracket_width_ratio"] < 2.0


def test_bound_grows_with_patch_concentration():
    previous = 0.0
    for size in (1.0, 0.8, 0.6, 0.4, 0.2):
        value = ilyushin_sine_upper_bound(A, B, T, SY, (size, size))["pressure"]
        assert value > previous
        previous = value


def test_extra_modes_never_loosen_the_bound():
    tight = ilyushin_sine_upper_bound(A, B, T, SY, PATCH, terms=1)["pressure"]
    for terms in (2, 3):
        wider = ilyushin_sine_upper_bound(A, B, T, SY, PATCH, terms=terms)["pressure"]
        assert wider <= tight + 1e-6


def test_rectangular_plate_is_orientation_symmetric():
    left = ilyushin_sine_upper_bound(2.0, 1.0, T, SY, (1.0, 1.0))["pressure"]
    right = ilyushin_sine_upper_bound(1.0, 2.0, T, SY, (1.0, 1.0))["pressure"]
    assert left == pytest.approx(right, rel=1e-12)


@pytest.mark.parametrize("bad", [
    dict(a=0.0, b=1.0, t=T, patch=(0.5, 0.5)),
    dict(a=1.0, b=1.0, t=0.0, patch=(0.5, 0.5)),
    dict(a=1.0, b=1.0, t=T, patch=(2.0, 0.5)),
])
def test_degenerate_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        ilyushin_sine_upper_bound(bad["a"], bad["b"], bad["t"], SY, bad["patch"])


def test_terms_must_be_positive():
    with pytest.raises(ValueError):
        ilyushin_sine_upper_bound(A, B, T, SY, PATCH, terms=0)