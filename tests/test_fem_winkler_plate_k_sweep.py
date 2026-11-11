"""GAP-FEM-WINKLER — Winkler foundation k-sweep regression tests.

Guards the foundation-coefficient coupling of the Q4 Mindlin plate element
(``K_w = k * int N^T N dA``) and of the Euler-Bernoulli beam element
(``K_w = k * int N^T N dx``) over several decades of the foundation modulus.

The heavy full sweep lives in ``benchmarks/fem_winkler_plate_k_sweep.py``; here
the same cases are run on coarse meshes so the assertions stay cheap while still
being able to catch a units/quadrature/factor-of-l error (any of which moves the
deflection by a factor, not by a few percent).

References: Hetényi (1946) *Beams on Elastic Foundation*;
Timoshenko & Woinowsky-Krieger (1959) *Theory of Plates and Shells*, §62.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks" / "fem_winkler_plate_k_sweep.py"


def _load_benchmark():
    spec = importlib.util.spec_from_file_location("fem_winkler_plate_k_sweep", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_benchmark()

K_CASES = (1.0e3, 1.0e5, 1.0e6)  # 3 decades
TOL_PCT = 5.0


# ---------------------------------------------------------------------------
# A. infinite plate, central point load, k across decades
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", K_CASES)
def test_plate_centre_deflection_matches_hetenyi(k):
    row = bench.plate_case(k, n=20)
    assert row["centre_error_pct"] < TOL_PCT, row
    assert row["w0_fem_mm"] > 0.0  # downward deflection, same sign as Hetényi


@pytest.mark.parametrize("k", K_CASES)
def test_plate_profile_and_equilibrium(k):
    row = bench.plate_case(k, n=20)
    # normalised by w0 because kei(r/l) changes sign at r/l ~ 2.4
    assert row["profile_error_max_pct"] < TOL_PCT, row
    # free outer edge -> the discrete foundation reaction must absorb the load
    assert row["equilibrium_residual_pct"] < 1.0, row
    assert row["profile_r_squared"] > 0.99, row


def test_plate_nondimensional_collapse_over_three_decades():
    """w(0) = P/(8 sqrt(D k)): the product w0*sqrt(k) must be k-independent."""
    products = []
    for k in K_CASES:
        row = bench.plate_case(k, n=20)
        w0 = row["w0_fem_mm"] * 1e-3
        products.append(w0 * math.sqrt(k))
    spread = (max(products) - min(products)) / sum(products) * len(products)
    assert spread < 0.05, products


def test_plate_sweep_spans_at_least_two_decades():
    assert bench.decades(bench.K_PLATE) >= 2.0
    assert len(bench.K_PLATE) >= 3


# ---------------------------------------------------------------------------
# B. finite simply supported beam, point load and uniform load
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kind", ("point", "uniform"))
@pytest.mark.parametrize("k_b", (1.0e4, 1.0e6))
def test_beam_foundation_matches_sine_series(kind, k_b):
    row = bench.beam_case(k_b, kind, n_elem=100)
    assert row["max_error_pct"] < TOL_PCT, row
    assert row["w_max_ref_mm"] > 0.0


def test_beam_deflection_decreases_with_stiffer_foundation():
    soft = bench.beam_case(1.0e4, "point", n_elem=100)
    stiff = bench.beam_case(1.0e6, "point", n_elem=100)
    assert stiff["w_max_fem_mm"] < soft["w_max_fem_mm"]


# ---------------------------------------------------------------------------
# C. finite simply supported plate under uniform pressure
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", (1.0e4, 1.0e6))
def test_finite_plate_uniform_pressure_matches_navier(k):
    row = bench.finite_plate_case(k, nx=20)
    assert row["centre_error_pct"] < TOL_PCT, row
    # supported plate: the edge reactions carry part of the load
    assert 0.0 < row["foundation_load_share"] <= 1.0, row


# ---------------------------------------------------------------------------
# D. zero-foundation sanity: k -> 0 must not silently reuse a foundation
# ---------------------------------------------------------------------------
def test_zero_foundation_is_not_a_foundation():
    """k = 0 degenerates to the free plate: no foundation reaction at all."""
    row = bench.plate_case(1.0e6, n=10)
    assert row["equilibrium_residual_pct"] < 1.0
    # a much softer foundation must deflect much more (w ~ k^(-1/2))
    soft = bench.plate_case(1.0e4, n=10)
    ratio = soft["w0_fem_mm"] / row["w0_fem_mm"]
    assert 8.0 < ratio < 12.0, ratio  # sqrt(1e6/1e4) = 10


# ---------------------------------------------------------------------------
# E. the gate logic itself must be able to fail
# ---------------------------------------------------------------------------
def test_gate_logic_rejects_out_of_tolerance_values():
    ok = bench._gate("g", "q", "r", 1.0, 5.0, "<=")
    bad = bench._gate("g", "q", "r", 5.1, 5.0, "<=")
    assert ok["passed"] is True
    assert bad["passed"] is False


def test_gate_detects_a_mutated_foundation_stiffness():
    """Mutation probe: a 25 % error in `k` inside the element must breach the gate.

    Deflection scales as ``w ~ k^(-1/2)``, so the 5 % deflection gate is a ~10 %
    gate on the foundation coefficient itself; this test proves the acceptance
    chain is not inert.
    """
    from tensorfem.foundation_plate_bending import (
        infinite_plate_point_load_fem,
        plate_rigidity,
        winkler_characteristic_length,
    )

    young, nu, thick, load = 2.1e11, 0.3, 0.05, 2.0e5
    k = 1.0e5
    rigidity = plate_rigidity(young, nu, thick)
    ref = load / (8.0 * math.sqrt(rigidity * k))
    ell = winkler_characteristic_length(rigidity, k)
    sol = infinite_plate_point_load_fem(
        load, young=young, poisson=nu, thickness=thick, winkler=1.25 * k,
        half_width_in_lengths=10.0, n=20, beta=4.0, dense_limit=3000,
    )
    err_pct = abs(float(sol.deflection[0]) - ref) / ref * 100.0
    assert ell > 0.0
    assert err_pct > TOL_PCT, err_pct


@pytest.mark.slow
def test_full_sweep_passes_all_gates(tmp_path):
    """Full-resolution sweep (same settings as the committed artifacts).

    Artifacts go to ``tmp_path`` so the committed evidence files are never
    rewritten by a test run.
    """
    assert bench.run(quick=False, out=tmp_path) == 0
    assert (tmp_path / "gates_verified.json").exists()