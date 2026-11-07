"""Tests for the structural verification ladder (beam / plate / modal).

Three layers:

* unit tests on the closed forms and on the solvers of
  :mod:`tensorfem.structural_verification` (fast meshes only);
* the *gates* of the campaign node (tip deflection < 1 %, plate centre
  coefficient < 2 %, modes 1--5 < 1 %) recomputed live, so a regression in the
  element, the load vector or the eigensolver fails CI without depending on the
  committed JSON;
* a consistency check of the committed artifacts in ``benchmarks/verified/``
  (verdicts, anchors, figure files) so the evidence cannot drift from the code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfem import structural_verification as sv

ROOT = Path(__file__).resolve().parents[1]
VERIFIED = ROOT / "benchmarks" / "verified"

# Beam: 1 m x 10 mm x 10 mm steel, 5 N/m.
BEAM_L, BEAM_B, BEAM_H, BEAM_E, BEAM_Q = 1.0, 0.010, 0.010, 200.0e9, 5.0
BEAM_I = BEAM_B * BEAM_H**3 / 12.0
BEAM_EI = BEAM_E * BEAM_I

# Plate: 1 m x 1 m x 10 mm, E = 210 GPa, nu = 0.3, p = 10 kPa.
PLATE_SIDE, PLATE_E, PLATE_NU, PLATE_T, PLATE_P = 1.0, 210.0e9, 0.3, 0.010, 1.0e4

# Modal beam: same 1 m bar, rho = 7850 kg/m^3.
MODAL_RHO_A = 7850.0 * BEAM_B * BEAM_H


# ---------------------------------------------------------------------------
# Closed forms
# ---------------------------------------------------------------------------
def test_beam_closed_form_satisfies_boundary_and_tip_values():
    L, EI, q = BEAM_L, BEAM_EI, BEAM_Q
    assert sv.euler_bernoulli_cantilever_deflection(0.0, L, EI, q) == 0.0
    # Clamped: w'(0) = 0 -> finite-difference check of the quartic.
    h = 1e-7
    slope0 = (
        sv.euler_bernoulli_cantilever_deflection(h, L, EI, q)
        - sv.euler_bernoulli_cantilever_deflection(0.0, L, EI, q)
    ) / h
    assert abs(slope0) < 1e-9
    tip = sv.euler_bernoulli_cantilever_tip(L, EI, q)
    assert tip["tip_deflection"] == pytest.approx(q * L**4 / (8.0 * EI), rel=1e-15)
    assert tip["tip_rotation"] == pytest.approx(q * L**3 / (6.0 * EI), rel=1e-15)
    assert sv.euler_bernoulli_cantilever_deflection(L, L, EI, q) == pytest.approx(
        tip["tip_deflection"], rel=1e-14
    )


def test_consistent_udl_load_vector_equilibrium():
    q, L = BEAM_Q, 0.25
    fe = sv.beam_consistent_udl_load(load=q, length=L)
    assert fe.sum() == pytest.approx(q * L)  # force resultant
    # Moment about the left node: end force times lever arm plus both nodal
    # moments must equal q*L^2/2 (sagging positive).
    moment = fe[2] * L + fe[1] + fe[3]
    assert moment == pytest.approx(q * L**2 / 2.0)


def test_cantilever_roots_match_tabulated_values():
    roots = sv.cantilever_roots(5)
    assert len(roots) == 5
    assert all(b > a for a, b in zip(roots, roots[1:]))
    for z, table in zip(roots, sv.CANTILEVER_BETA_L_TABLE):
        assert z == pytest.approx(table, rel=1e-5)
        assert abs(math.cos(z) * math.cosh(z) + 1.0) / math.cosh(z) < 1e-12


def test_observed_order_recovers_power_law():
    h = np.array([1.0, 0.5, 0.25, 0.125])
    assert sv.observed_order(h, 3.0 * h**2) == pytest.approx(2.0, abs=1e-12)
    assert sv.observed_order(h, 5.0 * h**4) == pytest.approx(4.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Rung 1: cantilever under uniform load
# ---------------------------------------------------------------------------
def test_beam_tip_deflection_matches_analytic_within_gate():
    exact = sv.euler_bernoulli_cantilever_tip(BEAM_L, BEAM_EI, BEAM_Q)
    for n in (2, 8, 32):
        sol = sv.solve_cantilever_udl(BEAM_L, BEAM_EI, BEAM_Q, n)
        err = abs(sol.tip_deflection - exact["tip_deflection"]) / exact["tip_deflection"]
        assert err < 1e-2, (n, err)
        # clamped shear reaction balances the total load
        assert sol.clamped_shear() == pytest.approx(-BEAM_Q * BEAM_L, rel=1e-9)


def test_beam_field_convergence_order_at_least_one_and_a_half():
    levels = (2, 4, 8, 16)
    errs = []
    for n in levels:
        sol = sv.solve_cantilever_udl(BEAM_L, BEAM_EI, BEAM_Q, n)
        assert sol.deflection_at(BEAM_L) == pytest.approx(sol.tip_deflection, rel=1e-14)
        errs.append(sv.beam_l2_relative_error(sol))
    order = sv.observed_order(np.array([BEAM_L / n for n in levels]), np.array(errs))
    assert order >= 1.5, order
    assert errs[0] > errs[-1]  # monotone convergence


# ---------------------------------------------------------------------------
# Rung 2: Kirchhoff plate
# ---------------------------------------------------------------------------
def test_navier_series_alpha_matches_tabulated_value():
    w = sv.plate_centre_deflection_series(
        side=PLATE_SIDE, young=PLATE_E, poisson=PLATE_NU, thickness=PLATE_T,
        pressure=PLATE_P, terms=201,
    )
    D = PLATE_E * PLATE_T**3 / (12.0 * (1.0 - PLATE_NU**2))
    alpha = w * D / (PLATE_P * PLATE_SIDE**4)
    assert alpha == pytest.approx(0.0040624, rel=1e-4)


def test_kirchhoff_plate_centre_coefficient_within_two_percent():
    sol = sv.kirchhoff_ss_plate_uniform_load(
        side=PLATE_SIDE, young=PLATE_E, poisson=PLATE_NU, thickness=PLATE_T,
        pressure=PLATE_P, n_x=16,
    )
    err = abs(sol.alpha - sv.PLATE_ALPHA_TIMOSHENKO) / sv.PLATE_ALPHA_TIMOSHENKO
    assert err < 2e-2, (sol.alpha, err)
    # conforming displacement element: stiffness overestimated -> w below exact
    assert sol.alpha < 0.0040624
    X, Y, W = sol.grid()
    assert W.shape == (17, 17)
    assert W[8, 8] == pytest.approx(sol.centre_deflection)


def test_kirchhoff_plate_convergence_order_and_symmetry():
    levels = (4, 8, 16)
    errs, alphas = [], []
    for n in levels:
        sol = sv.kirchhoff_ss_plate_uniform_load(
            side=PLATE_SIDE, young=PLATE_E, poisson=PLATE_NU, thickness=PLATE_T,
            pressure=PLATE_P, n_x=n,
        )
        alphas.append(sol.alpha)
        errs.append(abs(sol.alpha - 0.0040624) / 0.0040624)
        # symmetry about both centre lines
        assert np.allclose(sol.deflection, sol.deflection[::-1, :], atol=1e-18)
        assert np.allclose(sol.deflection, sol.deflection[:, ::-1], atol=1e-18)
        # edges are restrained
        assert np.allclose(sol.deflection[0, :], 0.0, atol=1e-18)
        assert np.allclose(sol.deflection[:, 0], 0.0, atol=1e-18)
    order = sv.observed_order(np.array([PLATE_SIDE / n for n in levels]), np.array(errs))
    assert order >= 1.5, order
    assert alphas[0] < alphas[1] < alphas[2] < 0.0040624  # monotone from below


def test_plate_solver_guards_and_thin_limit():
    with pytest.raises(ValueError):
        sv.kirchhoff_ss_plate_uniform_load(n_x=5)  # odd mesh: no centre node
    with pytest.raises(ValueError):
        sv.kirchhoff_ss_plate_uniform_load(n_x=8, poisson=0.5)
    # Deflection scales linearly with pressure and inversely with D.
    a = sv.kirchhoff_ss_plate_uniform_load(n_x=8, pressure=1.0e4)
    b = sv.kirchhoff_ss_plate_uniform_load(n_x=8, pressure=2.0e4)
    assert b.centre_deflection == pytest.approx(2.0 * a.centre_deflection, rel=1e-12)
    c = sv.kirchhoff_ss_plate_uniform_load(n_x=8, thickness=0.020)
    assert a.alpha == pytest.approx(c.alpha, rel=2e-2)  # alpha is thickness-free


# ---------------------------------------------------------------------------
# Rung 3: cantilever modes
# ---------------------------------------------------------------------------
def test_modal_frequencies_match_analytic_within_one_percent():
    ladder = sv.cantilever_modal_ladder(
        length=BEAM_L, EI=BEAM_EI, rho_a=MODAL_RHO_A, elements=(8, 16), modes=5
    )
    errors = ladder.relative_errors()
    assert errors.shape == (2, 5)
    assert errors[-1].max() < 1e-2
    # first frequency of a 1 m, 10 mm steel bar is ~8.15 Hz
    assert ladder.exact_angular[0] / (2.0 * math.pi) == pytest.approx(8.154, abs=1e-3)


def test_modal_convergence_order_at_least_one_and_a_half():
    levels = (8, 16, 32)
    ladder = sv.cantilever_modal_ladder(
        length=BEAM_L, EI=BEAM_EI, rho_a=MODAL_RHO_A, elements=levels, modes=5
    )
    errors = ladder.relative_errors()
    h = np.array([BEAM_L / n for n in levels])
    for m in range(5):
        assert sv.observed_order(h, errors[:, m]) >= 1.5


# ---------------------------------------------------------------------------
# Committed artifacts
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["fem_beam", "fem_plate", "fem_modal"])
def test_committed_verified_artifacts_are_consistent(case):
    result = json.loads((VERIFIED / case / "result.json").read_text())
    assert result["verdict"] == "PASS"
    assert result["generated_by"] == "benchmarks/structural_ladder.py"
    assert result["levels"], "no refinement levels recorded"
    assert (VERIFIED / case / "convergence.png").stat().st_size > 5000
    for gate in result["gates"]:
        assert gate["passed"], gate
        assert gate["relation"] in ("le", "ge")
        if gate["relation"] == "le":
            assert gate["value"] <= gate["limit"], gate
        else:
            assert gate["value"] >= gate["limit"], gate

    if case == "fem_beam":
        # tip deflection is superconvergent: last level must be at round-off
        assert result["levels"][-1]["tip_relative_error"] < 1e-9
        assert result["observed_order_field_l2"] >= 1.5
    elif case == "fem_plate":
        gate_row = next(row for row in result["levels"] if row["n_x"] == 16)
        assert gate_row["error_vs_timoshenko_00406"] < 2e-2
        assert result["observed_order"] >= 1.5
    else:
        gate_row = next(row for row in result["levels"] if row["elements"] == 16)
        assert max(gate_row["relative_errors"]) < 1e-2
        assert min(result["observed_orders"].values()) >= 1.5


def test_verified_summary_aggregates_all_rungs():
    summary = json.loads((VERIFIED / "structural_ladder_summary.json").read_text())
    assert summary["verdict"] == "PASS"
    assert set(summary["rungs"]) == {"fem_beam", "fem_plate", "fem_modal"}
    assert all(v == "PASS" for v in summary["rungs"].values())
    assert len(summary["anchors"]) == 3
    assert all(a["passed"] for a in summary["anchors"])
    for anchor in summary["anchors"]:
        assert anchor["relative_error"] <= 2e-2