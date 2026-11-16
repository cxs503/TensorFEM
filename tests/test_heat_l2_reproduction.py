"""Reproduction gate for the CASE-FEM-HEAT-L2 certificate (m22937).

What was missing
----------------
``benchmarks/verified/fem_heat_conduction_l2/run.py --check`` re-runs the whole
4..64 ladder and compares the committed artifact set to a fresh run at
``DRIFT_RTOL = 1e-12``.  The mode was reachable but *never reached*: the suite
exercised ``numeric_drift`` only on hand-written dictionaries
(``test_reproduction_detector_flags_drift_and_ignores_volatile_fields``) and
checked the README against the JSON (``test_readme_is_rendered_from_the_certificate``).
Nothing re-executed the solver, so a certificate produced by a code path that no
longer exists on ``main`` -- or one whose committed numbers were edited to keep a
re-run from failing -- still passed.  That is the m22874 lesson one level deeper:
the README was bound to the JSON, the JSON was never bound to the code.

What this file adds
-------------------
* one **full re-run** of the ladder from the current checkout, compared field by
  field against ``result.json`` and ``gates_verified.json`` (slow lane: a single
  ``convergence_study`` call at nx = 4..64, registered in
  ``tests/slow_lane_manifest.txt``; the suite's fast lane keeps the controls);
* fail-closed controls on the *real* committed payload -- a tampered level, a
  flipped verdict, a measured quantity that has drifted into ``VOLATILE_KEYS`` --
  so the gate cannot pass vacuously.

The evidence for the closure, and the reconciliation of the re-opened
deliverable, is in ``docs/CASE-FEM-HEAT-L2-reproduction-gate-m22937.md``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANCHOR = REPO_ROOT / "benchmarks" / "verified" / "fem_heat_conduction_l2"

#: Keys that carry a measurement.  A time stamp may be volatile; an error norm
#: may not.  The reproduction gate is only meaningful while this holds.
PHYSICS_KEYS = (
    "l2_relative",
    "l2_nodal_rms_relative",
    "h1_relative",
    "max_relative",
    "heat_balance_residual",
    "quadrature_spread",
    "l2_by_quadrature",
    "h1_by_quadrature",
    "observed_orders_l2",
    "observed_orders_h1",
    "nodal_rms_orders",
    "mean_order",
    "power_law_residual",
    "independent_cross_check",
)


def _run_module():
    """Import the benchmark runner without executing ``main()``."""
    spec = importlib.util.spec_from_file_location("fem_heat_l2_run_repro", ANCHOR / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fem_heat_l2_run_repro"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _run_module()


@pytest.fixture(scope="module")
def committed():
    """The committed artifact set, verbatim."""
    return {
        "result": json.loads((ANCHOR / "result.json").read_text()),
        "manifest": json.loads((ANCHOR / "gates_verified.json").read_text()),
    }


# ---------------------------------------------------------------------------
# the gate itself: the committed numbers must come out of *this* checkout
# ---------------------------------------------------------------------------
def test_committed_certificate_reproduces_from_this_checkout(runner, committed):
    """Re-run the 4..64 ladder and compare it to the committed certificate.

    ``recompute_without_writing()`` is the read-only half of ``run.py --check``:
    same ``convergence_study``, same payload builder, no file touched.  The
    comparison is the shipped ``numeric_drift`` walk at the predeclared
    ``DRIFT_RTOL``, which ignores clocks and interpreter metadata and nothing
    else.
    """
    levels, manifest, (orders, h1_orders, nodal_orders, residual, cross) = (
        runner.recompute_without_writing()
    )
    fresh = runner.build_result_payload(
        levels=levels,
        orders=orders,
        h1_orders=h1_orders,
        nodal_orders=nodal_orders,
        residual=residual,
        cross=cross,
        elapsed=0.0,
        figures=committed["result"]["figures"],  # hashes, not re-rendered plots
    )

    worst, msgs = runner.numeric_drift(committed["result"], fresh)
    assert not msgs, (
        "the committed result.json is not what this checkout computes "
        f"(worst relative drift {worst:.3e}); regenerate with\n"
        "    PYTHONPATH=src python3 benchmarks/verified/fem_heat_conduction_l2/run.py\n"
        "and commit result.json + gates_verified.json + README.md + figures together\n"
        f"first messages: {msgs[:6]}"
    )
    assert worst <= runner.DRIFT_RTOL

    worst_g, msgs_g = runner.numeric_drift(committed["manifest"], manifest)
    assert not msgs_g, f"the committed gate verdicts drifted (worst {worst_g:.3e}): {msgs_g[:6]}"

    # "reproduced" must not be satisfiable by reproducing a failure
    assert manifest["passed"] is True, manifest["failed_gates"]
    assert not manifest["failed_gates"]
    assert len(levels) == len(committed["result"]["mesh_levels"])


# ---------------------------------------------------------------------------
# fail-closed controls (cheap: they tamper the committed payload, no re-run)
# ---------------------------------------------------------------------------
def test_volatile_keys_do_not_swallow_the_measured_physics(runner):
    """A clock may be exempt from the comparison; an error norm may not.

    ``VOLATILE_KEYS`` is subtracted before the walk, so anything that lands in
    it becomes invisible to the reproduction gate.  This pins the exemption list
    to timestamp/metadata so a future edit cannot quietly whitelist physics.
    """
    assert set(runner.VOLATILE_KEYS) == {"generated_at", "elapsed_seconds", "sha256", "environment"}
    assert not (set(runner.VOLATILE_KEYS) & set(PHYSICS_KEYS))

    # and the walk really does reach the gate metric at the finest level
    committed = json.loads((ANCHOR / "result.json").read_text())
    tampered = copy.deepcopy(committed)
    tampered["mesh_levels"][-1]["l2_relative"] *= 1.0 + 1e-6
    worst, msgs = runner.numeric_drift(committed, tampered)
    assert msgs, "a 1e-6 change of the finest L2 norm went unnoticed"
    assert worst == pytest.approx(1e-6, rel=1e-3)
    assert any("l2_relative" in m for m in msgs)


def test_tampered_norm_is_caught_on_every_level(runner, committed):
    """Every rung of the ladder is inside the comparison, not just the finest."""
    committed_result = committed["result"]
    for index in range(len(committed_result["mesh_levels"])):
        tampered = copy.deepcopy(committed_result)
        tampered["mesh_levels"][index]["l2_relative"] *= 1.0 + 1e-5
        _worst, msgs = runner.numeric_drift(committed_result, tampered)
        assert any(f"mesh_levels[{index}]" in m for m in msgs), (
            f"level {index} is not covered by the reproduction walk: {msgs}"
        )


def test_flipped_gate_verdict_is_caught(runner, committed):
    """A verdict is never tolerance-blind, even with the value left untouched."""
    committed_manifest = committed["manifest"]
    for index, gate in enumerate(committed_manifest["gates"]):
        tampered = copy.deepcopy(committed_manifest)
        tampered["gates"][index]["passed"] = not gate["passed"]
        _worst, msgs = runner.numeric_drift(committed_manifest, tampered)
        assert any(f"gates[{index}]/passed" in m for m in msgs), (
            f"flipping {gate['id']} went unnoticed: {msgs}"
        )


def test_relaxed_gate_threshold_is_caught(runner, committed):
    """Relaxing a threshold in the committed manifest must not survive a re-run."""
    committed_manifest = committed["manifest"]
    index = next(
        i for i, g in enumerate(committed_manifest["gates"]) if g["id"] == "l2_pair_order_ge_1p95"
    )
    tampered = copy.deepcopy(committed_manifest)
    tampered["gates"][index]["threshold"] = 1.80
    _worst, msgs = runner.numeric_drift(committed_manifest, tampered)
    assert any(f"gates[{index}]/threshold" in m for m in msgs), msgs


def test_gate_tolerance_is_predeclared_below_the_printed_precision(runner):
    """``DRIFT_RTOL`` must stay far below the sixth digit the README prints.

    A zero tolerance would only hold on one BLAS build; a loose one would let a
    published number change.  1e-12 sits three orders of magnitude below the
    smallest printed relative difference (the README prints six significant
    digits), so both failure modes stay closed.
    """
    assert 0.0 < runner.DRIFT_RTOL <= 1e-9
