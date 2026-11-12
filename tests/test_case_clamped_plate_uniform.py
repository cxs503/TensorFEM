"""CASE-FEM-CLAMPED-PLATE-UNIFORM: clamped square plate, uniform pressure.

Anchor: alpha = w_max D/(q a^4) for a clamped square plate in the Kirchhoff
regime (a/t = 100) against the Timoshenko & Woinowsky-Krieger (1959) Sec. 30,
Table 30 entry 0.00126 (nu = 0.3).  Acceptance gate 3 %.

These tests are the fast slice of ``benchmarks/case_clamped_plate_uniform.py``;
the full mesh/sensitivity study lives there and writes
``benchmarks/artifacts/CASE-FEM-CLAMPED-PLATE-UNIFORM/``.
"""
from __future__ import annotations

import pytest

from tensorfem.plate_uniform_load import (
    centre_node_deflection,
    clamped_uniform_ritz,
    solve_uniform_square_plate,
    ss_uniform_kirchhoff,
    ss_uniform_ritz,
)

SIDE = 1.0
E = 2.1e11
NU = 0.3
Q = 1.0e6
T = 0.01          # a/t = 100
TABLE_ALPHA = 0.00126   # Timoshenko & Woinowsky-Krieger 1959, Sec. 30, Table 30
GATE = 0.03


def alpha(w: float, thickness: float = T, poisson: float = NU,
          pressure: float = Q, side: float = SIDE) -> float:
    """w * D / (q a^4) with D = E t^3 / (12 (1 - nu^2))."""
    D = E * thickness ** 3 / (12.0 * (1.0 - poisson ** 2))
    return w * D / (pressure * side ** 4)


def test_ritz_machinery_reproduces_navier_series():
    """The clamped reference comes from a code path that hits the closed form."""
    ritz_ss = ss_uniform_ritz(SIDE, E, NU, T, Q, n_modes=8)
    series = ss_uniform_kirchhoff(SIDE, E, NU, T, Q)
    got = alpha(ritz_ss.deflection(0.5, 0.5))
    ref = alpha(series["deflection"])
    assert abs(got - ref) / ref < 1e-3


def test_ritz_clamped_centre_deflection_matches_table():
    """Independent reference: does the Ritz solve land on the tabulated 0.00126?"""
    for n in (10, 12):
        got = alpha(clamped_uniform_ritz(SIDE, E, NU, T, Q, n_modes=n).deflection(0.5, 0.5))
        assert abs(got - TABLE_ALPHA) / TABLE_ALPHA < GATE, (n, got)


def test_fe_clamped_plate_uniform_centre_deflection_within_3pct():
    """PRIMARY ANCHOR: Q4 Mindlin 16x16 (17x17 nodes), a/t = 100 vs 0.00126."""
    res = solve_uniform_square_plate(16, SIDE, E, NU, T, Q, "clamped")
    got = alpha(centre_node_deflection(res))
    err = abs(got - TABLE_ALPHA) / TABLE_ALPHA
    assert err < GATE, "alpha = %.8f vs %.5f (%.3f%%)" % (got, TABLE_ALPHA, 100 * err)


def test_fe_normalisation_is_scale_invariant():
    """a -> 2a, t -> 2t, q -> 2q must leave the dimensionless alpha unchanged."""
    base = alpha(centre_node_deflection(
        solve_uniform_square_plate(8, SIDE, E, NU, T, Q, "clamped")))
    scaled = alpha(centre_node_deflection(
        solve_uniform_square_plate(8, 2 * SIDE, E, NU, 2 * T, 2 * Q, "clamped")),
        thickness=2 * T, pressure=2 * Q, side=2 * SIDE)
    assert abs(scaled - base) / base < 1e-12


def test_mesh_refinement_reduces_error_monotonically():
    """Coarse 4x4 must be worse than 8x8, and 8x8 worse than 16x16 (thin, clamped)."""
    errs = []
    for nx in (4, 8, 16):
        got = alpha(centre_node_deflection(
            solve_uniform_square_plate(nx, SIDE, E, NU, T, Q, "clamped")))
        errs.append(abs(got - TABLE_ALPHA) / TABLE_ALPHA)
    assert errs[0] > errs[1] > errs[2], errs
    assert errs[-1] < GATE


@pytest.mark.parametrize("thickness", [0.005, 0.01])
def test_thin_limit_thickness_independence(thickness):
    """For a/t >= 100 alpha is thickness independent (Kirchhoff limit)."""
    got = alpha(centre_node_deflection(
        solve_uniform_square_plate(16, SIDE, E, NU, thickness, Q, "clamped")),
        thickness=thickness)
    assert abs(got - TABLE_ALPHA) / TABLE_ALPHA < GATE


def test_thick_plate_shear_increases_deflection():
    """a/t = 10: Mindlin transverse shear must raise alpha above the thin value."""
    thin = alpha(centre_node_deflection(
        solve_uniform_square_plate(16, SIDE, E, NU, T, Q, "clamped")))
    thick = alpha(centre_node_deflection(
        solve_uniform_square_plate(16, SIDE, E, NU, 0.10, Q, "clamped")),
        thickness=0.10)
    assert thick > thin * 1.01, (thin, thick)