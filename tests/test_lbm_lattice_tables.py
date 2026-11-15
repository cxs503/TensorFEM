"""Tests for the table-driven lattice self-check (``lbm_lattice_tables``).

The invariants are checked in *exact rational arithmetic*, so most assertions
here are equalities, not tolerances.  The tests cover

1. every shipped table passes every invariant with exactly zero error,
2. the spec-driven tables agree with independently typed literature tables,
3. the audit rejects corrupted tables (single-entry and rank-4-only),
4. the table-driven equilibrium closes the Navier-Stokes moments exactly,
5. the lattice-agnostic kernels recover closed-form channel / Taylor-Green
   solutions with the viscosity predicted by the table itself,
6. constructor validation and error paths.
"""
from __future__ import annotations

import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.lbm_lattice_tables import (  # noqa: E402
    LatticeTable,
    equilibrium,
    equilibrium_exact,
    fit_wall_planes,
    get_table,
    interior_curvature,
    isotropy_rank,
    macroscopic,
    mutate_speed_class,
    mutate_weight,
    run_channel_poiseuille,
    run_taylor_green,
    selfcheck,
    stream,
    stream_half_way,
    table_names,
)

TWO_D_THREE_D = ("D2Q9", "D3Q15", "D3Q19", "D3Q27")


# ---------------------------------------------------------------------------
# 1. exact invariants
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", table_names())
def test_table_passes_all_invariants_exactly(name):
    table = get_table(name)
    report = selfcheck(table)
    assert report.passed, report.failures
    for check in report.checks:
        assert check.max_abs_error == 0, (check.key, check.max_abs_error)


@pytest.mark.parametrize("name", table_names())
def test_weights_are_positive_and_sum_to_one(name):
    table = get_table(name)
    assert all(w > 0 for w in table.weights)
    assert sum(table.weights) == 1


@pytest.mark.parametrize("name", TWO_D_THREE_D)
def test_second_and_fourth_moment_isotropy(name):
    table = get_table(name)
    dim, cs2 = table.dim, table.cs2
    for a in range(dim):
        for b in range(dim):
            assert table.moment((a, b)) == cs2 * (1 if a == b else 0)
    # mixed fourth rank: sum w c_x^2 c_y^2 = cs2^2 for every distinct pair
    for a in range(dim):
        for b in range(dim):
            if a == b:
                continue
            assert table.moment((a, a, b, b)) == cs2 * cs2


def test_isotropy_rank_at_least_four():
    for name in TWO_D_THREE_D:
        assert isotropy_rank(get_table(name)) >= 4


# ---------------------------------------------------------------------------
# 2. spec-driven table vs independently typed literature table
# ---------------------------------------------------------------------------
def test_d2q9_matches_literature_values():
    table = get_table("D2Q9")
    generated = dict(zip(table.velocities, table.weights))
    reference = {
        (0, 0): Fraction(4, 9),
        (1, 0): Fraction(1, 9), (0, 1): Fraction(1, 9),
        (-1, 0): Fraction(1, 9), (0, -1): Fraction(1, 9),
        (1, 1): Fraction(1, 36), (-1, -1): Fraction(1, 36),
        (1, -1): Fraction(1, 36), (-1, 1): Fraction(1, 36),
    }
    assert generated == reference


def test_d3q19_matches_literature_values():
    table = get_table("D3Q19")
    generated = dict(zip(table.velocities, table.weights))
    assert len(generated) == 19
    assert generated[(0, 0, 0)] == Fraction(1, 3)
    assert all(generated[(1, 0, 0)] == generated[v] == Fraction(1, 18)
               for v in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                         (0, 0, 1), (0, 0, -1)))
    assert generated[(1, 1, 0)] == Fraction(1, 36)
    assert generated[(0, -1, 1)] == Fraction(1, 36)


# ---------------------------------------------------------------------------
# 3. the audit rejects corrupted tables
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", table_names())
def test_every_single_weight_corruption_is_rejected(name):
    table = get_table(name)
    delta = Fraction(1, 1000)
    detected = 0
    total = 0
    for i in range(table.q):
        for sign in (+1, -1):
            report = selfcheck(mutate_weight(table, i, sign * delta))
            total += 1
            detected += 0 if report.passed else 1
    assert total > 0
    assert detected == total


def test_rank_four_corruption_is_caught_only_by_the_fourth_moment():
    """Mass, first and second moments preserved; only rank-4 isotropy broken."""
    d = Fraction(1, 1000)
    cases = {
        "D2Q9": {0: 4 * d, 1: -2 * d, 2: d},
        "D3Q19": {0: 12 * d, 1: -4 * d, 2: d},
        "D3Q27": {1: -d, 2: d, 3: -3 * d / 4},
        "D3Q15": {0: 16 * d, 1: -4 * d, 3: d},
    }
    for name, changes in cases.items():
        bad = mutate_speed_class(get_table(name), changes)
        report = selfcheck(bad)
        assert report.failures == ["fourth_moment"], (name, report.failures)
        assert report.isotropy_rank == 3


def test_opposite_weight_asymmetry_is_rejected():
    table = get_table("D2Q9")
    weights = list(table.weights)
    weights[1] = weights[1] + Fraction(1, 1000)
    weights[3] = weights[3] - Fraction(1, 1000)
    bad = LatticeTable("D2Q9~asym", table.dim, table.velocities, tuple(weights))
    assert "opposite_symmetry" in selfcheck(bad).failures


# ---------------------------------------------------------------------------
# 4. equilibrium moment closure (exact)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ("D2Q9", "D3Q19", "D3Q27"))
def test_equilibrium_closes_navier_stokes_moments_exactly(name):
    table = get_table(name)
    rho = Fraction(3, 2)
    u = tuple(Fraction(k, 25) for k in range(1, table.dim + 1))
    f = equilibrium_exact(table, rho, u)
    vel = table.velocities
    cs2 = table.cs2
    assert sum(f) == rho
    for a in range(table.dim):
        assert sum(fi * c[a] for fi, c in zip(f, vel)) == rho * u[a]
    for a in range(table.dim):
        for b in range(table.dim):
            measured = sum(fi * c[a] * c[b] for fi, c in zip(f, vel))
            expected = cs2 * rho * (1 if a == b else 0) + rho * u[a] * u[b]
            assert measured == expected
    # third moment carries the Navier-Stokes stress divergence exactly
    for a in range(table.dim):
        for b in range(table.dim):
            for c_ in range(table.dim):
                measured = sum(fi * c[a] * c[b] * c[c_] for fi, c in zip(f, vel))
                delta = ((1 if a == b else 0) * u[c_]
                         + (1 if a == c_ else 0) * u[b]
                         + (1 if b == c_ else 0) * u[a])
                assert measured == cs2 * rho * delta


@pytest.mark.parametrize("name", table_names())
def test_equilibrium_round_trip_recovers_rho_and_velocity(name):
    table = get_table(name)
    rng = np.random.default_rng(7)
    grid = (3,) * table.dim
    rho = 1.0 + 1e-3 * rng.random(grid)
    u = 1e-3 * rng.random((table.dim,) + grid)
    f = equilibrium(table, rho, u)
    rho_out, u_out = macroscopic(table, f)
    assert np.max(np.abs(rho_out - rho)) < 1e-14
    assert np.max(np.abs(u_out - u)) < 1e-14


# ---------------------------------------------------------------------------
# 5. physical cross-checks
# ---------------------------------------------------------------------------
def test_channel_curvature_recovers_table_viscosity():
    table = get_table("D2Q9")
    run = run_channel_poiseuille(table, ny=12, tau=0.8, force=4e-5,
                                 max_steps=60_000, check_every=500)
    assert run["converged"]
    assert run["curvature_rel_err"] < 1e-9
    assert run["density_max_dev"] < 1e-10
    # the fitted no-slip planes are shallow and symmetric
    assert abs(run["wall_offset_bot"] - run["wall_offset_top"]) < 1e-6
    assert 0.0 < abs(run["wall_offset_bot"]) < 0.25


def test_channel_profile_is_the_closed_form_parabola():
    table = get_table("D2Q9")
    run = run_channel_poiseuille(table, ny=16, tau=0.8, force=4e-5,
                                 max_steps=80_000, check_every=500)
    assert run["rel_err_profile_fitted"] < 1e-9
    # the naive a-priori wall convention is only first-order accurate
    assert run["rel_err_profile_wall_center"] > run["rel_err_profile_fitted"]


def test_wall_offset_is_resolution_consistent():
    offsets = []
    for ny in (12, 20):
        run = run_channel_poiseuille(get_table("D2Q9"), ny=ny, tau=0.8,
                                     force=4e-5, max_steps=80_000, check_every=500)
        offsets.append(0.5 * (abs(run["wall_offset_bot"]) + abs(run["wall_offset_top"])))
    assert abs(offsets[0] - offsets[1]) < 0.05


def test_three_dimension_channel_uses_the_same_kernels():
    run = run_channel_poiseuille(get_table("D3Q19"), ny=11, tau=0.8, force=4e-5,
                                 max_steps=60_000, check_every=500)
    assert run["converged"]
    assert run["curvature_rel_err"] < 1e-9


def test_taylorgreen_viscosity_matches_the_table():
    run = run_taylor_green(get_table("D2Q9"), n=32, u0=1e-3, tau=0.65,
                           steps=150, probe_every=5, periods=1)
    assert run["nu_rel_err"] < 5e-3
    assert run["field_rel_err"] < 2e-2


def test_half_way_variant_conserves_mass():
    """The comparison variant must not lose mass (it is not the gated scheme)."""
    table = get_table("D2Q9")
    grid = (1, 8)
    f = equilibrium(table, np.ones(grid), np.zeros((2,) + grid))
    mass0 = float(f.sum())
    for _ in range(5):
        f = stream_half_way(table, f)
        assert abs(float(f.sum()) - mass0) < 1e-12


def test_stream_conserves_mass_and_momentum():
    table = get_table("D2Q9")
    grid = (4, 4)
    rng = np.random.default_rng(3)
    f = 0.1 + 0.01 * rng.random((table.q,) + grid)
    out = stream(table, f)
    assert np.allclose(out.sum(), f.sum(), atol=1e-12)


# ---------------------------------------------------------------------------
# 6. helpers and error paths
# ---------------------------------------------------------------------------
def test_interior_curvature_of_exact_parabola():
    y = np.arange(10, dtype=float)
    profile = 2e-4 * (y - 0.1) * (9.9 - y)
    assert abs(interior_curvature(profile) - 2e-4) < 1e-18
    yb, yt = fit_wall_planes(profile, 2e-4, y0=0.0)
    assert abs(yb - 0.1) < 1e-6 and abs(yt - 9.9) < 1e-6


def test_table_validation_rejects_bad_data():
    with pytest.raises(ValueError):
        LatticeTable("bad", 2, ((1, 0),), (Fraction(1, 9), Fraction(1, 9)))
    with pytest.raises(ValueError):
        LatticeTable("bad", 2, ((1, 0, 0),), (Fraction(1),))
    with pytest.raises(ValueError):
        LatticeTable("bad", 2, ((1, 0),), (Fraction(-1),))
    # a set that is not closed under -c cannot produce an opposite map
    table = LatticeTable("open", 2, ((0, 0), (1, 0)), (Fraction(1, 2), Fraction(1, 2)))
    with pytest.raises(ValueError):
        _ = table.opposite
    with pytest.raises(KeyError):
        get_table("D9Q99")


def test_table_is_immutable():
    table = get_table("D2Q9")
    with pytest.raises(Exception):
        table.cs2 = Fraction(1, 4)  # type: ignore[misc]