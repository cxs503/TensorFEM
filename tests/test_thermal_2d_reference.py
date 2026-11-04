"""Gate for the 2-D Q4 conduction verification suite.

Two kinds of tests:

* **gates** -- every anchor of ``tensorfem.thermal_2d_reference`` must pass when
  recomputed here (the evidence JSON is not trusted on its own);
* **negative controls** -- each anchor must also *fail* when the reference is
  perturbed (naive textbook surface value, doubled film coefficient, doubled
  conductivity, wrong rectangle height).  Without these a gate could be
  vacuous: a check that passes for any input proves nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorfem.thermal import ThermalModel, solve_transient_thermal

from tensorfem.thermal_2d_reference import (
    boundary_nodes,
    exact_rectangle_transient,
    exact_wall_source_convection,
    relative_l2,
    rect_mesh,
    run_all,
    run_rectangle,
    study_rectangle_transient,
    study_square_symmetry,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "thermal_2d_reference_evidence.json"
DOC = ROOT / "docs" / "thermal-2d-q4-verification.md"
ASSETS = ROOT / "docs" / "assets"

# The systems here have at most ~1200 unknowns; the full thread pool costs ~20x
# on matrices this small, so cap it while this module is running.
@pytest.fixture(scope="module", autouse=True)
def _capped_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, min(4, previous)))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def recomputed() -> dict:
    return run_all()


@pytest.fixture(scope="module")
def anchors(recomputed) -> dict:
    return {a["anchor"]: a for a in recomputed["anchors"]}


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def test_all_anchors_pass(recomputed):
    failed = [a["anchor"] for a in recomputed["anchors"] if not a["passed"]]
    assert not failed, "anchors failed: %s" % failed
    assert recomputed["all_passed"]


def test_evidence_schema_and_verdicts(evidence):
    assert evidence["schema"] == "tensorfem.thermal-2d-q4/1.0"
    assert evidence["all_passed"]
    assert evidence["anchors"], "evidence must contain anchors"
    for anchor in evidence["anchors"]:
        assert anchor["passed"], anchor["anchor"]


def test_evidence_matches_recomputation(evidence, recomputed):
    """Catches a hand-edited evidence file."""
    mine = {a["anchor"]: a for a in recomputed["anchors"]}
    for anchor in evidence["anchors"]:
        other = mine[anchor["anchor"]]
        if "relative_l2_error" in anchor:
            assert anchor["relative_l2_error"] == pytest.approx(
                other["relative_l2_error"], rel=1e-9, abs=1e-15)
        if "errors" in anchor:
            assert anchor["errors"] == pytest.approx(other["errors"], rel=1e-9)
        if "orders" in anchor:
            assert anchor["orders"] == pytest.approx(other["orders"], rel=1e-9)
    for key, values in evidence["cross_checks"].items():
        assert values, key


def test_wall_source_convection_is_machine_precise(anchors):
    wall = anchors["source_convection_wall_2d"]
    assert wall["relative_l2_error"] < 1e-10
    assert wall["max_absolute_error"] < 1e-9
    assert wall["outlet_temperature"] == pytest.approx(
        wall["exact_outlet_temperature"], rel=1e-9)


def test_wall_energy_balance(anchors):
    wall = anchors["source_convection_wall_2d"]
    assert wall["energy_balance_relative_error"] < 1e-8
    total_out = wall["robin_outflux"] + wall["heat_out_fixed_face"]
    assert total_out == pytest.approx(wall["generated_heat"], rel=1e-8)


def test_harmonic_exactness(anchors):
    assert anchors["harmonic_quadratic_nodal_exactness"]["max_absolute_error"] < 1e-10


@pytest.mark.parametrize("name", ["mms_harmonic_smooth", "mms_harmonic_smooth_distorted"])
def test_mms_convergence_order(anchors, name):
    anchor = anchors[name]
    assert anchor["observed_order"] >= anchor["gate_order"]
    assert anchor["errors"][-1] < 0.03


def test_transient_rectangle_heading(anchors):
    trans = anchors["rectangle_transient_product_series"]
    assert trans["errors"][-1] < trans["gate_error"]
    assert trans["observed_order"] >= 1.8
    # refinement must actually reduce the error (monotone)
    assert all(f < c for c, f in zip(trans["errors"][:-1], trans["errors"][1:]))


def test_transient_bounds_and_symmetry(anchors):
    bound = anchors["rectangle_transient_maximum_principle"]
    assert bound["minimum"] >= -1e-12
    assert bound["maximum"] <= 1.0 + 1e-12
    assert bound["relative_l2_error"] < bound["gate_l2"]
    sym = anchors["square_transient_reflection_symmetry"]
    assert sym["max_asymmetry"] <= sym["gate_absolute"]


# ---------------------------------------------------------------------------
# negative controls: perturb the reference, the gate must break
# ---------------------------------------------------------------------------
def _wall_fields(anchors):
    wall = anchors["source_convection_wall_2d"]
    nodes = torch.tensor(wall["field"]["nodes"], dtype=torch.float64)
    numeric = torch.tensor(wall["field"]["temperature"], dtype=torch.float64)
    return nodes, numeric, wall["values"]


def test_negative_control_naive_surface_value(anchors):
    """The `q L / h` surface value must be rejected by this anchor."""
    nodes, numeric, values = _wall_fields(anchors)
    naive = values["q"] / (2.0 * values["k"]) * (values["L"] ** 2 - nodes[:, 0] ** 2) \
        + values["q"] * values["L"] / values["h"]
    assert relative_l2(numeric, naive) > 0.15


def test_negative_control_doubled_film_coefficient(anchors):
    """A factor 2 in the Robin coefficient (or in its boundary measure)."""
    nodes, numeric, values = _wall_fields(anchors)
    wrong = exact_wall_source_convection(nodes[:, 0], values["L"], values["k"],
                                         values["q"], 2.0 * values["h"])
    # the interior is dominated by the source parabola, so the response is
    # ~12%; that is still 4x the 3% accuracy standard and 1e10 times the
    # 1e-8 gate of this anchor, hence clearly rejected.
    assert relative_l2(numeric, wrong) > 0.10


def test_negative_control_doubled_conductivity(anchors):
    nodes, numeric, values = _wall_fields(anchors)
    wrong = exact_wall_source_convection(nodes[:, 0], values["L"], 2.0 * values["k"],
                                         values["q"], values["h"])
    assert relative_l2(numeric, wrong) > 0.15


def test_negative_control_wrong_rectangle_height(anchors):
    """The transient anchor must be sensitive to the non-square geometry."""
    trans = anchors["rectangle_transient_product_series"]
    case = run_rectangle(16, 8, 1.0, 0.5, trans["values"]["t_end"], 80, 0.5)
    right = exact_rectangle_transient(case.nodes[:, 0], case.nodes[:, 1],
                                      case.t_end, 1.0, 0.5, trans["values"]["alpha"])
    wrong = exact_rectangle_transient(case.nodes[:, 0], case.nodes[:, 1],
                                      case.t_end, 1.0, 1.0, trans["values"]["alpha"])
    assert relative_l2(case.temperature, right) < 0.10
    assert relative_l2(case.temperature, wrong) > 0.15


def test_negative_control_symmetry_gate_is_not_vacuous():
    """An x-only initial field on the same square mesh must break the symmetry gate."""
    square = study_square_symmetry(n=8, t_end=0.05, n_steps=80)
    assert square["max_asymmetry"] <= square["gate_absolute"]
    nodes, conn = rect_mesh(8, 8, 1.0, 1.0)
    fixed = boundary_nodes(8, 8)
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(1.0, dtype=torch.float64),
        density=torch.tensor(1.0, dtype=torch.float64),
        specific_heat=torch.tensor(1.0, dtype=torch.float64),
        fixed_nodes=fixed,
        fixed_temperatures=torch.zeros(fixed.numel(), dtype=torch.float64),
        thickness=torch.tensor(1.0, dtype=torch.float64),
        element_type="q4",
    )
    initial = torch.sin(torch.pi * nodes[:, 0]).clone()
    time = torch.linspace(0.0, 0.05, 81, dtype=torch.float64)
    field = solve_transient_thermal(model, time, initial, theta=0.5).temperature[-1]
    grid = field.reshape(9, 9)
    assert float((grid - grid.transpose(0, 1)).abs().max()) > square["gate_absolute"]


# ---------------------------------------------------------------------------
# report / artefact gates
# ---------------------------------------------------------------------------
def test_report_exists_and_is_honest():
    assert DOC.is_file(), "docs/thermal-2d-q4-verification.md must exist"
    text = DOC.read_text(encoding="utf-8")
    for section in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6.", "## 7."):
        assert section in text
    assert "Status: **PASS**" in text
    assert "Explicit answer to the 3% question" in text
    assert "does **not** meet 3%" in text, "the coarse-mesh failure must stay documented"
    assert "%%" not in text, "unformatted percent signs in the generated report"


@pytest.mark.parametrize("name", ["thermal_2d_wall_field.png",
                                  "thermal_2d_transient_field.png",
                                  "thermal_2d_convergence.png"])
def test_figures_exist(name):
    path = ASSETS / name
    assert path.is_file(), "%s must be generated by the benchmark" % name
    assert path.stat().st_size > 5000