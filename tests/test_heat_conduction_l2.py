"""Integration test: CASE-FEM-HEAT-L2 convergence study via public API.

Two layers:

* the study itself (:func:`run_heat_conduction_l2`), re-run once per session --
  the acceptance is a *rate*, so the assertions are on the observed orders of
  the domain-integral L2 norm, not on a single error value;
* the certificate on disk (``benchmarks/verified/fem_heat_conduction_l2``),
  checked statically so a stale plot or a relaxed gate cannot slip through.

The regression of m20995 lives in
``test_nodal_rms_pseudo_norm_is_biased_low`` / ``test_certificate_gates_test_the_acceptance``:
the report used to carry a nodal-RMS sampling pseudo-norm whose observed orders
(1.81 / 1.88 / 1.94) understate the kernel's true second order and whose gate
(``>= 1.8``) did not test the acceptance (``>= 2``).
"""
import math

import hashlib
import json
from pathlib import Path

import pytest

from tensorfem import run_heat_conduction_l2, HeatConductionL2Result


@pytest.fixture(scope="module")
def result() -> HeatConductionL2Result:
    """One run of the default four-level ladder, shared by all assertions."""
    return run_heat_conduction_l2()


def test_heat_conduction_l2_returns_result(result):
    """run_heat_conduction_l2 returns a HeatConductionL2Result."""
    assert isinstance(result, HeatConductionL2Result)


def test_heat_conduction_l2_four_levels(result):
    """Default run has 4 mesh levels."""
    assert len(result.levels) == 4


def test_heat_conduction_l2_convergence_order(result):
    """Every asymptotic L2 pair reaches the 1.95 band under the integral norm."""
    assert result.mean_order >= 1.95, f"Mean order {result.mean_order:.4f} < 1.95"
    assert result.observed_orders[-1] >= 1.95, result.observed_orders
    for order, level in zip(result.observed_orders, result.levels[:-1]):
        if level.nx >= 8:
            assert order >= 1.95, f"pair from nx={level.nx}: p={order:.4f}"


def test_heat_conduction_l2_finest_error(result):
    """Finest mesh integral L2 relative error < 3%."""
    assert result.levels[-1].l2_relative < 0.03


def test_heat_conduction_l2_heat_balance(result):
    """Heat balance residual at free nodes is machine precision."""
    for lv in result.levels:
        assert lv.heat_balance_residual < 1e-10, (
            f"Level {lv.nx}x{lv.ny}: heat_balance={lv.heat_balance_residual:.2e}"
        )


def test_heat_conduction_l2_all_gates_pass(result):
    """All acceptance gates pass."""
    assert result.all_gates_pass, result.gates


def test_heat_conduction_l2_orders_approach_two(result):
    """Observed orders stay in the neighbourhood of 2.0 and rise monotonically."""
    orders = result.observed_orders
    assert len(orders) == 3
    for o in orders:
        assert 1.9 <= o <= 2.1, f"Order {o:.4f} outside the two-sided band"
    assert orders[-1] > orders[0], "Orders should increase with refinement"


def test_heat_conduction_l2_h1_order(result):
    """H1 seminorm order is consistent with the theory (1.0 for bilinear Q4)."""
    assert all(0.9 <= p <= 1.1 for p in result.h1_orders), result.h1_orders


def test_quadrature_does_not_move_the_norms(result):
    """The reported norms must not depend on the post-processing rule.

    A quadrature-starved error integral can manufacture an order; the 4-point
    and 8-point values are carried next to the 6-point one so this is checkable.
    """
    for lv in result.levels:
        assert set(lv.l2_by_quadrature) == {4, 6, 8}
        # the gate band applies where the order is measured (nx >= 8); on the
        # coarsest level a smooth errexp on a 1/4-size element still needs the
        # 8-point rule (spread ~5e-6), which is why that level carries no order
        limit = 1e-6 if lv.nx >= 8 else 1e-5
        assert lv.quadrature_spread <= limit, (lv.nx, lv.quadrature_spread)
    assert result.order_shift_quadrature <= 1e-3, result.order_shift_quadrature


def test_nodal_rms_pseudo_norm_is_biased_low(result):
    """Regression (m20995): the nodal-RMS pseudo-norm understates the order.

    ``sqrt(mean(e**2))`` over all nodes is dominated by interior nodes while the
    Dirichlet nodes carry exactly zero error, so its ratio to the true norm
    drifts with h and it measures a systematically lower order (1.81 -> 1.97
    instead of 1.99 -> 2.00).  It must stay a *diagnostic*: if it ever agrees
    with the integral norm to within 0.02 on the finest pair, this study is no
    longer measuring what the predeclared gates claim and the ladder has to be
    re-examined.
    """
    assert len(result.nodal_rms_orders) == len(result.observed_orders)
    for p_pseudo, p_integral in zip(result.nodal_rms_orders, result.observed_orders):
        assert p_pseudo < p_integral - 0.02, (p_pseudo, p_integral)
    assert result.nodal_rms_orders[-1] < 1.95, result.nodal_rms_orders
    assert result.observed_orders[-1] >= 1.95, result.observed_orders


def test_heat_conduction_l2_custom_mesh():
    """Custom mesh levels work."""
    res = run_heat_conduction_l2(mesh_levels=[(2, 2), (4, 4), (8, 8)])
    assert len(res.levels) == 3
    assert res.levels[0].n_elements == 4
    assert res.levels[-1].n_elements == 64


# ---------------------------------------------------------------------------
# certificate evidence (m20372): the figures must reach GitHub and stay bound
# to the numbers they were drawn from
# ---------------------------------------------------------------------------
ANCHOR = Path(__file__).resolve().parents[1] / "benchmarks" / "verified" / "fem_heat_conduction_l2"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_certificate_figures_render_under_docs():
    """A declared figure outside ``docs/`` is stripped by the push gate.

    ``tp_pusher.py`` carries ``docs/**`` binaries and drops every other
    ``*.png``, so ``benchmarks/verified/<anchor>/fig_*.png`` would certify
    evidence that never leaves the working tree.
    """
    payload = json.loads((ANCHOR / "gates_verified.json").read_text())
    figures = payload["artifacts"]["figures"]
    assert figures, "the certificate must name the figures it ships"
    for rel in figures:
        assert rel.startswith("docs/"), f"{rel} is not carried by the push gate"
        assert (REPO_ROOT / rel).is_file(), f"{rel} is declared but absent"


def test_figures_are_bound_to_the_run_that_produced_them():
    """``result.json`` records the sha256 of the plots it produced.

    Both the numbers and the plots are committed, and nothing else links them: a
    stale plot (re-run the solver, forget the figure) is otherwise invisible --
    the same failure mode as a certificate that names a figure nobody can open.
    """
    result = json.loads((ANCHOR / "result.json").read_text())
    recorded = {entry["path"]: entry["sha256"] for entry in result["figures"]}
    declared = set(json.loads((ANCHOR / "gates_verified.json").read_text())["artifacts"]["figures"])
    assert set(recorded) == declared, "result.json and the certificate name different figures"
    for rel, digest in recorded.items():
        assert digest == _sha256(REPO_ROOT / rel), f"{rel} does not match the recorded hash"


def test_certificate_gates_test_the_acceptance():
    """Every shipped gate must pass *and* the order gates must test ``>= 2``.

    Regression (m20995): the certificate shipped ``convergence_order_ge_1p8``
    (threshold 1.8) against an acceptance of "L2 convergence rate >= 2nd
    order", so a PASS said nothing about the acceptance.  The thresholds are now
    checked here, not just the verdicts.
    """
    manifest = json.loads((ANCHOR / "gates_verified.json").read_text())
    gates = {g["id"]: g for g in manifest["gates"]}
    assert manifest["passed"] is True, manifest["failed_gates"]
    assert not manifest["failed_gates"]
    # the *L2* order gates carry the acceptance; the H1 gate is supplementary
    order_gates = {k: v for k, v in gates.items() if "l2" in k and "order" in k}
    assert order_gates, "the certificate must carry an order gate"
    for gate_id, gate in order_gates.items():
        if gate["operator"].startswith(">="):
            assert gate["threshold"] >= 1.95, (gate_id, gate["threshold"])
        assert gate["passed"] is True, gate_id
    # the gate that must never come back
    assert "convergence_order_ge_1p8" not in gates


def test_certificate_reports_the_integral_norm():
    """The gate metric is the domain integral; the pseudo-norm is a diagnostic."""
    payload = json.loads((ANCHOR / "result.json").read_text())
    assert payload["norm_definition"]["gate_metric"].lower().startswith("relative l2 domain integral")
    levels = payload["mesh_levels"]
    assert len(levels) >= 5, "the ladder must reach nx = 64 for the asymptotic band"
    assert levels[-1]["nx"] == 64
    for lv in levels:
        # the integral norm is strictly larger than the nodal sample norm here
        assert lv["l2_relative"] > lv["l2_nodal_rms_relative"], lv["nx"]
        assert lv["l2_by_quadrature"]["6"] == pytest.approx(lv["l2_relative"], rel=1e-12)
    orders = payload["observed_orders_l2"]
    nodal = payload["nodal_rms_orders"]
    assert orders[-1] >= 1.95
    assert nodal[-1] < orders[-1] - 0.02
    assert payload["independent_cross_check"]["worst_norm_relative_difference"] <= 1e-9


def test_independent_solver_agrees_on_one_level():
    """The from-scratch numpy/scipy Q4 solver reproduces the library's L2 norm."""
    import importlib.util
    import sys

    import torch

    from tensorfem.heat_conduction_l2 import integral_error_norms

    spec = importlib.util.spec_from_file_location("fem_heat_independent", ANCHOR / "independent_sparse.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fem_heat_independent"] = module
    spec.loader.exec_module(module)

    level = module.solve_level(16, 16)
    nodes, elements, _ = module.structured_mesh(16, 16)
    kernel = integral_error_norms(
        torch.tensor(nodes, dtype=torch.float64),
        torch.tensor(elements, dtype=torch.long),
        torch.tensor(level["temperature"], dtype=torch.float64),
        8,
    )
    assert abs(kernel["l2_relative"] - level["l2_relative"]) / level["l2_relative"] < 1e-9
    assert abs(kernel["h1_relative"] - level["h1_relative"]) / level["h1_relative"] < 1e-9
    # the order of magnitude the gate band expects, so a silent unit change fails
    assert level["l2_relative"] == pytest.approx(2.8069221880e-03, rel=1e-6)

# ---------------------------------------------------------------------------
# certificate derivation (m22874): README and the gate values are *functions*
# of the artifacts, not prose maintained next to them
# ---------------------------------------------------------------------------
def _run_module():
    """Import the benchmark runner without executing ``main()``."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("fem_heat_l2_run", ANCHOR / "run.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["fem_heat_l2_run"] = module
    spec.loader.exec_module(module)
    return module


def _certificate():
    result = json.loads((ANCHOR / "result.json").read_text())
    manifest = json.loads((ANCHOR / "gates_verified.json").read_text())
    return result, manifest


def test_readme_is_rendered_from_the_certificate():
    """``README.md`` must be byte-identical to its rendering from the JSON.

    The README claims "no number here is typed by hand".  Until m22874 that was
    an honour system, and it had already failed: the file's ``generated`` stamp
    named a run 1.3 days older than the ``result.json`` committed beside it.  A
    hand-edited table row, or a gate whose verdict moved without the prose
    following, is now a test failure.
    """
    result, manifest = _certificate()
    rendered = _run_module().render_readme(result, manifest)
    on_disk = (ANCHOR / "README.md").read_text()
    assert on_disk == rendered, (
        "README.md is not the rendering of result.json + gates_verified.json; "
        "re-run run.py (or run.py --check to see the diff)"
    )
    # the stamp the README shows comes from the artifact, not from a memory of a run
    assert manifest["generated_at"] in on_disk


@pytest.mark.parametrize("gate_id", [
    "finest_l2_below_3pct",
    "l2_pair_order_ge_1p95",
    "l2_finest_order_le_2p1",
    "power_law_residual_le_1pct",
    "h1_pair_order_ge_0p9",
    "quadrature_spread_le_1e-6",
    "independent_code_l2_agreement",
    "heat_balance_residual",
])
def test_gate_value_is_recomputable_from_result_json(gate_id):
    """Recompute each gate's value here, from ``result.json`` alone.

    A gate value that cannot be reproduced from the reported numbers is a claim,
    not a measurement -- the defect class this test closes is a certificate
    edited by hand instead of regenerated.
    """
    result, manifest = _certificate()
    gates = {g["id"]: g for g in manifest["gates"]}
    assert gate_id in gates, f"{gate_id} missing from the certificate"
    gate = gates[gate_id]

    levels = result["mesh_levels"]
    orders = result["observed_orders_l2"]
    h1_orders = result["observed_orders_h1"]
    asymptotic = [p for p, lv in zip(orders, levels[:-1]) if lv["nx"] >= 8]
    h1_asymptotic = [p for p, lv in zip(h1_orders, levels[:-1]) if lv["nx"] >= 8]
    e_c, e_f = levels[-2]["l2_relative"], levels[-1]["l2_relative"]
    r = levels[-2]["h"] / levels[-1]["h"]
    e0 = e_f + (e_f - e_c) / (r ** 2.0 - 1.0)

    expected = {
        "finest_l2_below_3pct": levels[-1]["l2_relative"],
        "l2_pair_order_ge_1p95": min(asymptotic),
        "l2_finest_order_le_2p1": orders[-1],
        "power_law_residual_le_1pct": abs(e0) / e_f,
        "h1_pair_order_ge_0p9": min(h1_asymptotic),
        "quadrature_spread_le_1e-6": max(lv["quadrature_spread"] for lv in levels if lv["nx"] >= 8),
        "independent_code_l2_agreement": result["independent_cross_check"]["worst_norm_relative_difference"],
        "heat_balance_residual": levels[-1]["heat_balance_residual"],
    }[gate_id]
    assert gate["value"] == expected, (gate_id, gate["value"], expected)

    # and the verdict is the arithmetic, not a hand-typed boolean
    op = gate["operator"]
    threshold = gate["threshold"]
    recomputed = {"<=": expected <= threshold, ">=": expected >= threshold}[op]
    assert gate["passed"] is recomputed, (gate_id, op, expected, threshold)


def test_band_lower_bound_survives_the_operator_split():
    """Splitting the band into two scalar gates must not drop the lower bound.

    ``l2_finest_order_le_2p1`` only carries the upper bound; the lower bound on
    the same quantity is carried by ``l2_pair_order_ge_1p95``, whose minimum runs
    over every asymptotic pair -- the finest pair among them.  If that reasoning
    ever stops holding, the band has silently become one-sided.
    """
    result, manifest = _certificate()
    levels = result["mesh_levels"]
    orders = result["observed_orders_l2"]
    finest_pair_pairs = [p for p, lv in zip(orders, levels[:-1]) if lv["nx"] >= 8]
    assert orders[-1] in finest_pair_pairs
    gates = {g["id"]: g for g in manifest["gates"]}
    assert gates["l2_pair_order_ge_1p95"]["value"] <= orders[-1]
    assert gates["l2_finest_order_le_2p1"]["value"] == orders[-1]


def test_gate_operators_are_machine_parseable():
    """Every shipped gate must be recomputable by the platform's own auditor.

    Regression (m22874): ``l2_finest_order_in_band`` shipped the prose operator
    ``"<= and >= 1.95 2.1"``, which ``tools/verified_anchor_audit.py`` cannot
    evaluate -- it recorded the gate as unverifiable and the anchor's
    ``recomputed_overall`` as null.  A verdict nobody can recompute is not
    evidence.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "verified_anchor_audit",
        REPO_ROOT / "tools" / "verified_anchor_audit.py",
    )
    auditor = importlib.util.module_from_spec(spec)
    import sys
    sys.modules["verified_anchor_audit"] = auditor   # dataclasses resolve via sys.modules
    spec.loader.exec_module(auditor)

    _result, manifest = _certificate()
    for gate in manifest["gates"]:
        assert gate["operator"] in auditor.OPERATORS, (
            f"{gate['id']}: operator {gate['operator']!r} is not machine-evaluable"
        )
        assert isinstance(gate["threshold"], (int, float)), gate["id"]
        assert math.isfinite(gate["threshold"]), gate["id"]
        assert math.isfinite(gate["value"]), gate["id"]
        recomputed = auditor.OPERATORS[gate["operator"]](gate["value"], gate["threshold"])
        assert gate["passed"] is bool(recomputed), (gate["id"], gate["passed"], recomputed)


def test_reproduction_detector_flags_drift_and_ignores_volatile_fields():
    """The ``run.py --check`` comparator: sensitive to physics, blind to clocks.

    Predeclared (m22874): relative tolerance 1e-12 -- two orders of magnitude
    below the sixth printed digit, so any drift that could change a published
    number fails, while a different BLAS reduction order does not.
    """
    module = _run_module()
    baseline = {"a": 1.0, "b": [2.0, 3.0], "generated_at": "2026-01-01", "environment": {"torch": "2.7"}}

    worst, msgs = module.numeric_drift(baseline, dict(baseline))
    assert worst == 0.0 and not msgs

    # timestamps and interpreter metadata are not measurements
    other_run = dict(baseline, generated_at="2026-02-02", environment={"torch": "9.9"})
    assert not module.numeric_drift(baseline, other_run)[1]

    # a drift big enough to change the sixth printed digit must be caught
    drifted = dict(baseline, b=[2.0, 3.0 * (1 + 1e-6)])
    worst, msgs = module.numeric_drift(baseline, drifted)
    assert msgs and worst == pytest.approx(1e-6, rel=1e-3), msgs

    # a drift below the declared tolerance is accepted, but reported
    tiny = dict(baseline, b=[2.0, 3.0 * (1 + 1e-13)])
    worst, msgs = module.numeric_drift(baseline, tiny)
    assert not msgs and worst <= module.DRIFT_RTOL

    # a changed verdict or a vanished key is never tolerance-blind
    assert module.numeric_drift({"passed": True}, {"passed": False})[1]
    assert module.numeric_drift({"k": 1.0}, {})[1]
