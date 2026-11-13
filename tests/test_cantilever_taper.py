"""Regression gates for the taper axis of the case-pool anchor ``CASE-CANTILEVER-TIP``.

The anchor grades a cantilever tip deflection against
``delta_EB = PL^3/3EI`` / ``delta_TI = PL^3/3EI + PL/(kappa G A)`` with a
``< 3 %`` criterion.  Every pre-existing route of the anchor holds the section
constant; here the depth tapers linearly, so the flexibility becomes the
integral ``int (L-x)^2/I(x) dx`` and the oracle is

    delta_bend  = (P L^3/(E I0)) B(c),   B(c) = [-c - c^2/2 - ln(1-c)]/c^3
    delta_shear = (P L/(kappa G A0)) S(c),  S(c) = -ln(1-c)/c

Four things are locked so the taper cannot silently regress:

* the closed form against two independent routes (power series, adaptive
  quadrature) and a sympy re-derivation, plus the ``c -> 0`` prismatic limits;
* the numerical trap that the closed form cancels catastrophically for small
  ``c`` (``B(1e-9)`` must still be ``1/3``) -- this is a regression for a bug
  found while writing the driver;
* the discretisation statements the document makes: a lumped (midpoint) ``EI``
  per element is second order, an exactly integrated element stiffness is
  fourth order, and a prismatic member is exact in both schemes;
* the 3 % case-pool gate on the 3-D HEX20 model, with the clamped-root
  ``u_x = 0`` control that must recover most of the residual.

Evidence: ``benchmarks/verified/fem_cantilever_taper/{result.json,gates_verified.json}``.
"""
import math

import numpy as np
import pytest

from tensorfem.tapered_cantilever import (
    TaperCase,
    bending_factor,
    bending_factor_series,
    bending_factor_symbolic,
    frame_tip_deflection,
    shear_factor,
    shear_factor_series,
    solve_tapered_cantilever,
    taper_limits,
    tip_face_loads,
    tapered_hex20_mesh,
    validate_mesh_geometry,
)

TAPERS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)
TOLERANCE = 0.03


# --------------------------------------------------------------------------- #
# oracle
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("c", TAPERS)
def test_oracle_routes_agree(c):
    case = TaperCase(taper=c)
    quad = case.delta_quadrature()
    assert quad["total"] == pytest.approx(case.delta_timoshenko(), rel=1e-10)
    assert bending_factor_series(c) == pytest.approx(bending_factor(c), rel=1e-12)
    assert shear_factor_series(c) == pytest.approx(shear_factor(c), rel=1e-12)


@pytest.mark.parametrize("c", (0.1, 0.5, 0.9))
def test_sympy_rederivation(c):
    report = bending_factor_symbolic(c)
    assert report["residual"] < 1e-10
    assert report["series_residual"] < 1e-10


def test_prismatic_limit_recovers_case_pool_oracles():
    limits = taper_limits()
    assert limits["bending_over_prismatic"] == pytest.approx(1.0, abs=1e-8)
    assert limits["shear_over_prismatic"] == pytest.approx(1.0, abs=1e-8)
    # And against the literal case-pool expressions.
    case = TaperCase(taper=0.0)
    i0 = case.width * case.depth_root**3 / 12.0
    a0 = case.width * case.depth_root
    assert case.delta_euler() == pytest.approx(case.load * case.length**3 / (3 * case.young * i0), rel=1e-14)
    assert case.delta_shear() == pytest.approx(
        case.load * case.length / (case.kappa * case.shear_modulus * a0), rel=1e-14)


def test_small_taper_is_not_cancellation_limited():
    """The closed form loses ~16 digits for c < 1e-4; the series branch must not.

    For c = 1e-9 the direct expression evaluates ``(-c - c^2/2 - ln(1-c))/c^3``
    as (3.33e-28)/(1e-27) with every term at the 1e-9 scale, i.e. it returns 0.
    """
    eps = 1e-9
    assert bending_factor(eps) == pytest.approx(1.0 / 3.0, rel=1e-6)
    assert shear_factor(eps) == pytest.approx(1.0, rel=1e-6)
    naive = (-eps - 0.5 * eps**2 - math.log1p(-eps)) / eps**3
    assert abs(naive - 1.0 / 3.0) > 1e-3, "the naive expression is expected to be unusable here"


def test_invalid_taper_is_rejected():
    for bad in (-0.1, 1.0, 1.5):
        with pytest.raises(ValueError):
            TaperCase(taper=bad)
        with pytest.raises(ValueError):
            bending_factor(bad)


def test_wedge_divergence_identity():
    """``B(c) c^3 = -c - c^2/2 - ln(1-c)`` hence ``B(c) ~ -ln(1-c)`` as c -> 1."""
    previous = 0.0
    for c in (0.9, 0.99, 0.999, 0.9999):
        bend = bending_factor(c)
        assert abs(bend * c**3 - (-c - 0.5 * c * c - math.log1p(-c))) / bend < 1e-12
        assert bend > previous
        previous = bend
    assert bending_factor(0.9999) > 7.0        # amplification ~23x, not a bounded 3x


# --------------------------------------------------------------------------- #
# 1-D frame
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("scheme", ("midpoint", "gauss"))
def test_frame_is_exact_on_a_prismatic_member(scheme):
    case = TaperCase(taper=0.0)
    for n in (1, 2, 4, 8):
        assert frame_tip_deflection(case, n, sampling=scheme) == pytest.approx(case.delta_euler(), rel=1e-12)


def test_midpoint_inertia_is_second_order_and_gauss_is_fourth_order():
    case = TaperCase(taper=0.5)
    ref = case.delta_euler()
    n = np.array([4.0, 8.0, 16.0, 32.0])
    mid = np.array([abs(frame_tip_deflection(case, int(k), sampling="midpoint") / ref - 1.0) for k in n])
    gau = np.array([abs(frame_tip_deflection(case, int(k), sampling="gauss") / ref - 1.0) for k in n])
    order_mid = -np.polyfit(np.log(n), np.log(mid), 1)[0]
    order_gau = -np.polyfit(np.log(n), np.log(gau), 1)[0]
    assert order_mid == pytest.approx(2.0, abs=0.2)
    assert order_gau > 3.5
    assert mid[-1] < TOLERANCE


# --------------------------------------------------------------------------- #
# graded HEX20 solid
# --------------------------------------------------------------------------- #
def test_tip_face_load_pattern_is_consistent():
    case = TaperCase(taper=0.5)
    nodes, _ = tapered_hex20_mesh(case, 4, 2, 2)
    forces = tip_face_loads(nodes, case)
    total = float(forces[2::3].sum())
    assert total == pytest.approx(case.load, rel=1e-14)
    assert float(forces[2::3].min()) < 0.0 < float(forces[2::3].max())


def test_geometry_is_exact_before_any_deflection_is_read():
    geo = validate_mesh_geometry(TaperCase(taper=0.5), 16, 2, 2)
    assert geo["surface_deviation"] < 1e-12
    assert geo["area_error"] < 1e-12
    assert geo["max_element_aspect"] < 10.0


def test_solid_meets_the_case_pool_tolerance():
    case = TaperCase(taper=0.5)
    res = solve_tapered_cantilever(case, 16, 2, 2)
    assert res["tip_deflection"] / case.delta_timoshenko() - 1.0 == pytest.approx(0.0, abs=TOLERANCE)


def test_clamped_root_explains_most_of_the_residual():
    case = TaperCase(taper=0.5)
    ref = case.delta_timoshenko()
    clamp = solve_tapered_cantilever(case, 16, 2, 2, root_bc="clamp_all")["tip_deflection"] / ref - 1.0
    free = solve_tapered_cantilever(case, 16, 2, 2, root_bc="lateral_free")["tip_deflection"] / ref - 1.0
    assert clamp < 0.0 and free < 0.0
    assert abs(free) < 0.5 * abs(clamp)


def test_unknown_root_bc_is_rejected():
    with pytest.raises(ValueError):
        solve_tapered_cantilever(TaperCase(), 4, 2, 2, root_bc="pinned")