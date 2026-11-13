"""CASE-CANTILEVER-TIP axial-preload rung -- oracle independence and evidence audit.

Three independent duties:

1. **Oracle independence.**  The amplification is recomputed here by a shooting
   integration of the beam-column ODE (hand-written RK4, no sympy, no library
   code) and compared with the closed forms the driver uses.  A transcription
   slip in the driver's sympy derivation therefore cannot survive this test.
2. **Evidence audit.**  ``gates_verified.json`` is re-derived from
   ``result.json`` (a gate whose value or pass flag does not match the data
   fails), and the physical content is asserted: tension must stiffen,
   compression must amplify, the clamp-free pole must sit on the Euler load, a
   single element near ``N_cr`` must be wrong by design, and refinement must
   recover the exact answer.
3. **Live library check.**  A tiny beam-column solve is run through the driver
   on the merged ``tensorfem.buckling`` matrices so this anchor fails if the
   library's geometric stiffness changes under it.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
ANCHOR_DIR = REPO / "benchmarks" / "verified" / "fem_cantilever_beamcolumn"
RESULT = ANCHOR_DIR / "result.json"
GATES = ANCHOR_DIR / "gates_verified.json"
DOC = REPO / "docs" / "CASE-CANTILEVER-TIP-beamcolumn-amplification.md"
TOLERANCE = 0.03


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses need the module registered
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def driver():
    return _load("fem_cantilever_beamcolumn_run", ANCHOR_DIR / "run.py")


@pytest.fixture(scope="module")
def result():
    return json.loads(RESULT.read_text())


@pytest.fixture(scope="module")
def gates():
    return json.loads(GATES.read_text())


# --------------------------------------------------------------------------- #
# 1. independent oracle: shooting integration of the ODE
# --------------------------------------------------------------------------- #
def shade_solve(u: float, tension: bool, steps: int = 4000) -> float:
    """delta/delta_EB by RK4 shooting on E I w'''' -+ N w'' = 0, E I = P = L = 1.

    State (w, w', w'', w'''); the ODE is homogeneous, so the solution is a
    linear combination of two unit-initial-condition trajectories and the two
    remaining boundary conditions (M(L) = 0, V_tot(L) = -P) are a 2x2 system.
    """
    s = u * u * (1.0 if tension else -1.0)
    h = 1.0 / steps

    def integrate(a: float, b: float) -> np.ndarray:
        y = np.array([0.0, 0.0, a, b])

        def f(state: np.ndarray) -> np.ndarray:
            w1, w2, w3, w4 = state
            return np.array([w2, w3, w4, s * w3])

        for _ in range(steps):
            k1 = f(y)
            k2 = f(y + 0.5 * h * k1)
            k3 = f(y + 0.5 * h * k2)
            k4 = f(y + h * k3)
            y = y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        return y

    ya = integrate(1.0, 0.0)
    yb = integrate(0.0, 1.0)
    # M(L) = w''(L) = 0 ; V_tot(L) = w'''(L) - s w'(L) = -P = -1
    A = np.array([[ya[2], yb[2]], [ya[3] - s * ya[1], yb[3] - s * yb[1]]])
    rhs = np.array([0.0, -1.0])
    a, b = np.linalg.solve(A, rhs)
    w_tip = a * ya[0] + b * yb[0]
    return 3.0 * w_tip            # delta_EB = P L^3/(3 E I) = 1/3


@pytest.mark.parametrize("u", [0.5, 1.0, 1.25, 1.5])
def test_oracle_matches_independent_shooting(driver, u):
    assert shade_solve(u, True) == pytest.approx(driver.af_tension(u), rel=1e-9, abs=1e-12)
    assert shade_solve(u, False) == pytest.approx(driver.af_compression(u), rel=1e-9, abs=1e-12)


def test_oracle_limits_and_identities(driver):
    assert driver.af_tension(0.0) == pytest.approx(1.0, abs=1e-15)
    assert driver.af_compression(0.0) == pytest.approx(1.0, abs=1e-15)
    # two-term small-u series 1 -+ 2u^2/5 + 17u^4/105 (the u^2 term alone is only good to
    # O(u^4); u = 1e-4 would additionally lose the difference u - tanh u to cancellation)
    u = 0.1
    series_t = 1 - 2 * u**2 / 5 + 17 * u**4 / 105
    series_c = 1 + 2 * u**2 / 5 + 17 * u**4 / 105
    # truncated after u^4, so the residual is the O(u^6) term (~6e-8 relative at u = 0.1)
    assert driver.af_tension(u) == pytest.approx(series_t, rel=1e-6)
    assert driver.af_compression(u) == pytest.approx(series_c, rel=1e-6)
    # analytic continuation AF_c(i u) = AF_t(u)
    for u in (0.5, 1.0, 1.5):
        assert driver.af_compression_complex(1j * u).real == pytest.approx(
            driver.af_tension(u), rel=1e-12)


def test_pole_is_the_clamp_free_euler_load(driver):
    assert driver.U_CR == pytest.approx(math.pi / 2, rel=0, abs=0)
    assert driver.PCR == pytest.approx(math.pi**2 * driver.EI / (4 * driver.L**2), rel=1e-15)


# --------------------------------------------------------------------------- #
# 2. evidence audit
# --------------------------------------------------------------------------- #
def test_manifest_shape_and_verdict(gates):
    assert gates["schema"] == "tensorfem.verified-gates.v1"
    assert gates["mission"] == "CASE-CANTILEVER-TIP"
    assert gates["anchor"] == "fem_cantilever_beamcolumn"
    assert gates["passed"] is True
    assert gates["failed_gates"] == []
    ids = [g["id"] for g in gates["gates"]]
    assert len(ids) == len(set(ids)), "duplicate gate ids"


def test_every_gate_recomputes_from_its_own_threshold(gates):
    for g in gates["gates"]:
        op = {"<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
              "<": lambda a, b: a < b, ">": lambda a, b: a > b}[g["operator"]]
        assert op(g["value"], g["threshold"]) == g["passed"], g["id"]


def test_case_pool_gates_match_the_ladder(result, gates):
    rows = result["ladder"]
    by_id = {g["id"]: g for g in gates["gates"]}
    for case in ("tension", "compression"):
        for u in result["problem"]["levels_u"]:
            sel = [r for r in rows if r["case"] == case and r["u"] == u
                   and r["elements"] in (4, 8, 16, 32)]
            worst = max(r["rel_error"] for r in sel)
            g = by_id[f"fe_vs_exact_{case}_u{u:g}"]
            assert g["value"] == pytest.approx(worst, rel=1e-12)
            assert g["passed"] and worst < TOLERANCE


def test_gate_values_match_the_reported_data(result, gates):
    by_id = {g["id"]: g for g in gates["gates"]}
    rows = result["ladder"]
    assert by_id["tension_stiffens"]["value"] == pytest.approx(
        max(r["amplification_FE"] for r in rows if r["case"] == "tension"), rel=1e-12)
    assert by_id["compression_amplifies"]["value"] == pytest.approx(
        min(r["amplification_FE"] for r in rows if r["case"] == "compression"), rel=1e-12)
    buck = result["buckling_ladder"]
    assert by_id["buckling_eigenvalue_from_same_matrices"]["value"] == pytest.approx(
        buck[-1]["rel_error"], rel=1e-12)
    sc = result["oracle_selfcheck"]
    assert by_id["oracle_ode_residual_tension"]["value"] == pytest.approx(
        sc["residual_ladder_tension"], abs=1e-15)
    assert by_id["analytic_continuation_identity"]["value"] == pytest.approx(
        sc["continuation_identity_max_rel"], abs=1e-15)


def test_physics_direction_and_monotonicity(result):
    rows = result["ladder"]
    for case in ("tension", "compression"):
        for n in (1, 8, 32):
            amps = [r["amplification_FE"] for r in rows if r["case"] == case
                    and r["elements"] == n]
            amps.sort()
            if case == "tension":
                assert max(amps) < 1.0
            else:
                assert min(amps) > 1.0
                assert amps == sorted(amps)          # amplifies with preload


def test_convergence_order_beats_second_order(result):
    for key, conv in result["convergence"].items():
        assert conv["order"] >= 3.0, (key, conv["order"])
        assert conv["errors"][-1] < conv["errors"][0]


def test_single_element_is_required_to_fail_near_critical(result):
    rows = [r for r in result["ladder"]
            if r["elements"] == 1 and r["u"] == 1.5 and r["case"] == "compression"]
    assert rows and rows[0]["rel_error"] > 0.05
    fine = [r for r in result["ladder"]
            if r["elements"] == 32 and r["u"] == 1.5 and r["case"] == "compression"][0]
    assert fine["rel_error"] < TOLERANCE


def test_buckling_eigenvalue_and_its_order(result):
    buck = result["buckling_ladder"]
    assert buck[-1]["rel_error"] < 1e-6
    errors = [b["rel_error"] for b in buck]
    assert errors[-1] < errors[0]
    assert all(errors[i] > errors[i + 1] for i in range(len(errors) - 1))


def test_limitations_are_declared(result):
    text = " ".join(result["limitations"])
    for token in ("Euler-Bernoulli", "small-deflection", "dead", "warping"):
        assert token.lower() in text.lower()


def test_doc_exists_and_carries_the_generated_numbers(result):
    assert DOC.exists()
    doc = DOC.read_text()
    worst = max(g["value"] for g in result["gates"]
                if g["id"].startswith("fe_vs_exact_") and "small-deflection" in g["note"])
    assert f"{1e2 * worst:.4f} %" in doc
    assert f"{result['problem']['pcr_fixed_free_N']:.4f}" in doc
    assert "PASS" in doc


# --------------------------------------------------------------------------- #
# 3. live check on the merged library
# --------------------------------------------------------------------------- #
def test_live_library_beam_column(driver):
    for u, tension in ((1.0, True), (1.0, False), (1.5, False)):
        fe = driver.fe_solve(u, 8, tension)["amplification"]
        exact = driver.af_tension(u) if tension else driver.af_compression(u)
        assert abs(fe - exact) / exact < 1e-4


def test_live_library_buckling(driver):
    buck = driver.fe_buckling(8)
    assert buck["rel_error"] < 1e-5
    assert driver.fe_solve(0.0, 1, True)["amplification"] == pytest.approx(1.0, abs=1e-13)