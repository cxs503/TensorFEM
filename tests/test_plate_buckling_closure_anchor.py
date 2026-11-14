"""Tests for the W29 plate buckling anchor register and the FEM k sweep.

The tests read the two published artifacts and check the claims the artifacts
make: the analytic closure residual, the sparse-versus-dense assembly closure,
the number and status of the anchors, the boundary-condition window, the
measured convergence order and the finest-mesh deviation against the 3% limit.
From the runner itself, only the element-order facts that would silently break
an upper/lower-bound argument are re-derived (the beam-mode roots satisfy their
characteristic equation; the BFS element reproduces a pure-twist and a
cylindrical-bending energy exactly), so a regression in either shows up without
re-running the sweep.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
REGISTER = REPO / "benchmarks/anchors/plate_buckling_closure.json"
SWEEP = REPO / "artifacts/buckling_k_sweep.json"

sys_path_added = str(REPO / "src")  # noqa: F841  (documented for readers)


def _load(path: pathlib.Path) -> dict:
    if not path.exists():
        pytest.skip(f"{path.relative_to(REPO)} not built yet (run the benchmark runner)")
    return json.loads(path.read_text())


def test_artifact_digests_are_self_consistent() -> None:
    from tensorfem.plate_buckling_closure import register_digest

    for path in (REGISTER, SWEEP):
        payload = _load(path)
        recorded = payload.pop("content_sha256")
        assert register_digest(payload) == recorded, f"{path.name} digest does not verify"
        assert len(recorded) == 64


def test_anchors_are_withheld_without_numbers() -> None:
    register = _load(REGISTER)
    anchors = register["anchors"]
    assert anchors["requested"] >= 2
    assert anchors["verbatim"] + anchors["withheld"] == anchors["requested"]
    if anchors["verbatim"] < 2:
        assert anchors["label"] == "HONEST_WITHHELD"
    for entry in anchors["entries"]:
        if entry["status"] != "VERBATIM":
            assert entry["verbatim_quote"] is None
            assert entry["source_sha256"] is None
            assert entry["numeric_value_recorded"] is False


def test_analytic_closure_residual_below_tolerance() -> None:
    sweep = _load(SWEEP)
    closure = sweep["analytic_closure"]
    assert closure["max_relative_residual"] <= closure["tolerance"] == 1e-10
    assert closure["max_round_trip_residual"] <= 1e-10
    assert all(row["pass"] for row in closure["rows"])
    assert len(closure["rows"]) == len(sweep["predeclared"]["aspect_ratios"])


def test_sparse_assembly_matches_dense_reference() -> None:
    sweep = _load(SWEEP)
    closure = sweep["assembly_closure"]
    assert closure["max_relative_residual"] <= 1e-10
    assert closure["edge_penalty_block_relative_residual"] <= 1e-10
    assert closure["pass"] is True


def test_simply_supported_matches_exact_analytic_envelope() -> None:
    sweep = _load(SWEEP)
    limit = sweep["predeclared"]["fem_deviation_limit"]
    for case in sweep["sweep"]:
        ss = case["simple_support"]
        assert ss["finest_deviation_vs_analytic"] <= limit, (
            f"a/b={case['aspect_ratio']}: {ss['finest_deviation_vs_analytic']:.4%} exceeds the limit"
        )
        assert ss["within_3_percent"] is True
        # the mesh ladder must descend monotonically towards the exact value
        values = [row["coefficient"] for row in ss["ladder"] if not row.get("skipped")]
        exact = case["analytic_reference"]["coefficient_exact"]
        errors = [abs(value - exact) for value in values]
        assert errors == sorted(errors, reverse=True), "FEM errors are not monotonically decreasing"


def test_convergence_order_is_measured_and_inside_the_window() -> None:
    sweep = _load(SWEEP)
    window = sweep["predeclared"]["expected_order_window_simple_support"]
    orders = [case["simple_support"]["convergence_order"] for case in sweep["sweep"]]
    assert all(order is not None for order in orders)
    for order in orders:
        assert window[0] <= order <= window[1], f"measured order {order:.3f} outside {window}"


def test_clamped_window_is_formed_and_closed() -> None:
    sweep = _load(SWEEP)
    limit = sweep["predeclared"]["clamped_bracket_limit"]
    for case in sweep["sweep"]:
        reference = case["clamped"]["reference"]
        assert reference["bracket"] is not None
        lower, upper = reference["bracket"]
        assert lower <= upper
        assert reference["window_relative_width"] <= limit
        # clamping raises the coefficient at every aspect ratio of the sweep; the
        # factor shrinks towards the long-plate asymptote, so only the square
        # plate carries the strong 2.5x statement
        assert case["ratio_clamped_over_simple"] > 1.5
        if abs(case["aspect_ratio"] - 1.0) < 1e-9:
            assert case["ratio_clamped_over_simple"] > 2.4


def test_boundary_condition_effect_is_ordered() -> None:
    sweep = _load(SWEEP)
    for case in sweep["sweep"]:
        k_simple = case["analytic_reference"]["coefficient_exact"]
        midpoint = case["clamped"]["reference"]["reference_midpoint"]
        assert midpoint > k_simple, "clamped coefficient must exceed the simply supported one"


def test_no_honest_failure_is_hidden() -> None:
    for payload in (_load(REGISTER), _load(SWEEP)):
        criteria = payload["criteria"]
        assert [row["id"] for row in criteria] == ["C1", "C2", "C3", "C4"]
        failed = [row for row in criteria if not row["met"]]
        assert len(failed) == len(payload["honest_failures"])
        for row in failed:
            assert row["verdict"] not in {"MEET", "HONEST_WITHHELD"}


def test_exploratory_route_is_recorded_as_open() -> None:
    sweep = _load(SWEEP)
    exploratory = sweep["exploratory_independent_route"]
    assert exploratory["verdict"] == "NOT_USED"
    assert exploratory["state"] == "OPEN"
    assert len(exploratory["competing_hypotheses"]) >= 2
    assert exploratory["discriminating_test"]


def test_bfs_element_energy_patch_is_exact() -> None:
    """Pure-twist and cylindrical-bending energies of one BFS element."""
    from tensorfem.plate_buckling_hermite import bfs_plate_element_matrices

    young, poisson, thickness = 2.1e11, 0.3, 0.012
    rigidity = young * thickness**3 / (12 * (1 - poisson**2))
    a = b = 0.25
    ke, _ = bfs_plate_element_matrices(a, b, young, poisson, thickness)
    order = [(0, 0), (1, 0), (0, 1), (1, 1)]

    def vector(fields):
        values = np.zeros(16)
        for index, (i, j) in enumerate(order):
            x, y = i * a, j * b
            for k, field in enumerate(fields):
                values[4 * index + k] = field(x, y)
        return values

    twist = vector((lambda x, y: x * y, lambda x, y: y, lambda x, y: x, lambda x, y: 1.0))
    cylindrical = vector((lambda x, y: x * x / 2, lambda x, y: x, lambda x, y: 0.0, lambda x, y: 0.0))
    assert 0.5 * twist @ ke @ twist == pytest.approx((1 - poisson) * rigidity * a * b, rel=1e-12)
    assert 0.5 * cylindrical @ ke @ cylindrical == pytest.approx(rigidity * a * b / 2, rel=1e-12)


def test_beam_mode_roots_satisfy_their_characteristic_equation() -> None:
    """The exploratory basis is only admissible at the roots of cos(b) cosh(b) = 1."""
    import importlib.util

    module_path = REPO / "benchmarks/anchors/plate_buckling_closure_anchor.py"
    if not module_path.exists():
        pytest.skip("runner not present")
    spec = importlib.util.spec_from_file_location("plate_anchor_runner", module_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    roots = runner._clamped_beam_roots(4)
    assert roots == sorted(roots)
    for beta in roots:
        assert math.cos(beta) * math.cosh(beta) - 1.0 == pytest.approx(0.0, abs=1e-8)
    xi = np.linspace(0.0, 1.0, 20001)
    value, first, _ = runner._clamped_beam_values(xi, roots[0])
    assert abs(value[0]) < 1e-12 and abs(value[-1]) < 1e-12
    assert abs(first[0]) < 1e-9 and abs(first[-1]) < 1e-9
