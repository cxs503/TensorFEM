"""Gate for the transient conduction reference study.

Two layers:

* the committed evidence file must be complete and every anchor must pass;
* a cheap subset is *recomputed* here so a stale or edited evidence file cannot
  hide a regression in `solve_transient_thermal`.

The last test is a negative control: it deliberately perturbs the diffusivity
and shows that the same comparison does reject the perturbed solve, i.e. the
gate is discriminating rather than trivially green.
"""
import json
from pathlib import Path

import pytest
import torch

from tensorfem.thermal_transient_reference import (
    exact_slab_mode,
    exact_slab_step,
    line_mesh,
    observed_order,
    run_slab,
    study_adiabatic_conservation,
    study_capacity_and_stiffness_scaling,
    study_q4_strip_consistency,
)
from tensorfem.thermal import ThermalModel, solve_transient_thermal

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "thermal_transient_reference_evidence.json"
DOC = ROOT / "docs" / "thermal-transient-reference-verification.md"
FLOAT = torch.float64


@pytest.fixture(scope="module")
def evidence():
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def _anchor(evidence, name):
    for anchor in evidence["anchors"]:
        if anchor["anchor"] == name:
            return anchor
    raise AssertionError(f"anchor {name} missing from evidence")


# ---------------------------------------------------------------------------
# evidence-file layer
# ---------------------------------------------------------------------------
def test_evidence_and_doc_exist():
    assert EVIDENCE.is_file()
    assert DOC.is_file(), "generated report must be committed next to the evidence"


def test_schema_and_all_passed(evidence):
    assert evidence["schema"] == "tensorfem.thermal-transient-reference/1.0"
    assert evidence["all_passed"] is True
    assert evidence["checks"], "verdict block must be present"
    for name, check in evidence["checks"].items():
        assert check["passed"], f"{name} recorded as failed"


def test_step_problem_inside_three_percent(evidence):
    anchor = _anchor(evidence, "dirichlet_step_fourier_series")
    assert anchor["relative_l2_error"] < 0.03
    assert anchor["max_absolute_error"] < 0.01


def test_spatial_and_temporal_orders_from_evidence(evidence):
    spatial = _anchor(evidence, "spatial_order_sine_mode")
    assert all(o >= 1.85 for o in spatial["orders"])
    stepwise = _anchor(evidence, "step_problem_h_convergence")
    assert all(o >= 1.85 for o in stepwise["orders"])
    schemes = _anchor(evidence, "temporal_order_theta_method")["schemes"]
    assert 0.85 <= schemes["theta=1.0"]["observed_order"] <= 1.15
    assert 1.8 <= schemes["theta=0.5"]["observed_order"] <= 2.2
    # monotone error decrease in every refinement family
    for family in (spatial["errors"], stepwise["errors"],
                   schemes["theta=1.0"]["errors"], schemes["theta=0.5"]["errors"]):
        assert all(b < a for a, b in zip(family[:-1], family[1:])), family


def test_cross_checks_from_evidence(evidence):
    q4 = _anchor(evidence, "q4_strip_matches_line2")
    assert q4["relative_l2_error"] < 1e-8
    cap = _anchor(evidence, "capacity_scaling_and_null_space")
    assert cap["relative_error"] < 1e-12
    assert cap["null_space_max_abs"] < 1e-10


# ---------------------------------------------------------------------------
# recomputed layer (cheap subset, guards against a stale evidence file)
# ---------------------------------------------------------------------------
def test_recomputed_step_problem_matches_closed_form():
    case = run_slab(n=16, t_end=0.1, n_steps=400, theta=1.0, initial="step")
    exact = exact_slab_step(case.nodes[:, 0], 0.1)
    error = float(torch.linalg.vector_norm(case.final_temperature - exact)
                  / torch.linalg.vector_norm(exact))
    assert error < 0.03, error


def test_recomputed_spatial_order_is_second_order():
    errors = []
    for n in (8, 16):
        case = run_slab(n=n, t_end=0.1, n_steps=1500, theta=0.5, initial="mode")
        exact = exact_slab_mode(case.nodes[:, 0], 0.1)
        errors.append(float(torch.linalg.vector_norm(case.final_temperature - exact)
                            / torch.linalg.vector_norm(exact)))
    assert observed_order(*errors) >= 1.85


def test_recomputed_q4_strip_agrees_with_line2():
    result = study_q4_strip_consistency(n=8, n_steps=200)
    assert result["relative_l2_error"] < 1e-8


def test_recomputed_adiabatic_conservation():
    result = study_adiabatic_conservation(n=10, t_end=0.3, n_steps=20)
    assert result["relative_drift"] < 1e-10


def test_recomputed_capacity_scaling():
    result = study_capacity_and_stiffness_scaling(n=5)
    assert result["relative_error"] < 1e-12
    assert result["null_space_max_abs"] < 1e-10


# ---------------------------------------------------------------------------
# reference-quality and discrimination checks
# ---------------------------------------------------------------------------
def test_fourier_series_is_term_count_converged():
    x = torch.linspace(0.0, 1.0, 33, dtype=FLOAT)
    coarse = exact_slab_step(x, 0.1, n_terms=20)
    fine = exact_slab_step(x, 0.1, n_terms=400)
    assert float((coarse - fine).abs().max()) < 1e-10


def test_fourier_series_satisfies_boundary_and_decay():
    x = torch.tensor([0.0, 0.5, 1.0], dtype=FLOAT)
    values = exact_slab_step(x, 0.2)
    assert abs(float(values[0])) < 1e-14 and abs(float(values[2])) < 1e-14
    # midpoint decays monotonically and stays inside (0, T0)
    series = [float(exact_slab_step(x, t)[1]) for t in (0.01, 0.1, 0.5)]
    assert all(0.0 < v < 1.0 for v in series)
    assert series[0] > series[1] > series[2]


def test_gate_rejects_wrong_diffusivity():
    """Negative control: a 20 % diffusivity error must break the 3 % gate."""
    nodes, conn = line_mesh(16)
    alpha_wrong = 1.2
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(alpha_wrong, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=torch.tensor([0, 16], dtype=torch.long),
        fixed_temperatures=torch.zeros(2, dtype=FLOAT),
        thickness=torch.tensor(1.0, dtype=FLOAT),
    )
    time = torch.linspace(0.0, 0.1, 401, dtype=FLOAT)
    initial = torch.ones(len(nodes), dtype=FLOAT)
    initial[torch.tensor([0, 16])] = 0.0
    solved = solve_transient_thermal(model, time, initial, theta=1.0)
    exact = exact_slab_step(nodes[:, 0], 0.1)
    error = float(torch.linalg.vector_norm(solved.temperature[-1] - exact)
                  / torch.linalg.vector_norm(exact))
    assert error > 0.03, error


def test_theta_below_half_still_fails_closed():
    nodes, conn = line_mesh(4)
    model = ThermalModel(
        nodes=nodes, elements=conn,
        conductivity=torch.tensor(1.0, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=torch.tensor([0, 4], dtype=torch.long),
        fixed_temperatures=torch.zeros(2, dtype=FLOAT),
        thickness=torch.tensor(1.0, dtype=FLOAT),
    )
    with pytest.raises(ValueError):
        solve_transient_thermal(model, torch.linspace(0.0, 0.1, 6, dtype=FLOAT),
                                torch.ones(len(nodes), dtype=FLOAT), theta=0.4)

@pytest.mark.parametrize("heterogeneous", [False, True])
def test_finite_contact_transient_mode_refines_and_conserves_heat(heterogeneous):
    from tensorfem.thermal_transient_reference import study_transient_contact_mode

    levels = [study_transient_contact_mode(n, heterogeneous=heterogeneous) for n in (4, 8, 16, 32)]
    for key in ("max_temperature_error_K", "max_jump_error_K"):
        orders = [observed_order(a[key], b[key]) for a, b in zip(levels, levels[1:])]
        assert all(1.9 < p < 2.1 for p in orders), (key, orders)
    assert max(row["max_relative_heat_drift"] for row in levels) < 1e-11
    fine = levels[-1]
    for key in ("left_flux_rel_linf", "right_flux_rel_linf", "contact_flux_rel_linf"):
        assert observed_order(levels[-2][key], fine[key]) > 1.9
        assert fine[key] < 3e-4
    assert fine["measured_decay_rate"] == pytest.approx(fine["exact_decay_rate"], rel=1e-4)


def test_equal_wall_ramp_and_hold_reaction_heat(record_property):
    """Both Dirichlet reactions supply half the unit slab's storage change."""
    ramp_end, end = .04, .09
    wave = torch.pi*torch.arange(1, 80, 2, dtype=FLOAT)

    def ramp_storage(t):
        # Spatial integral of the continuum solution for a ramp continued to t.
        return (t-1/12+(8*torch.exp(-wave.square()*t)/wave**4).sum())/ramp_end

    stored = torch.stack((wave.new_zeros(()), ramp_storage(ramp_end),
                          ramp_storage(end)-ramp_storage(end-ramp_end)))
    reference = stored.diff()/2
    errors = []
    for phase, value in zip(("ramp", "hold"), reference):
        record_property(f"{phase}_wall_reference_J_m2", value.item())
    for n, steps in ((12, 32), (24, 64), (48, 128), (48, 256)):
        nodes, conn = line_mesh(n)
        unit = nodes.new_tensor(1.)
        model = ThermalModel(nodes=nodes, elements=conn, conductivity=unit, density=unit,
                             specific_heat=unit, thickness=unit, fixed_nodes=torch.tensor([0, n]),
                             fixed_temperatures=nodes.new_zeros(2))
        time = torch.cat((torch.linspace(0., ramp_end, steps+1, dtype=FLOAT),
                          torch.linspace(ramp_end, end, steps+1, dtype=FLOAT)[1:]))
        wall = (time/ramp_end).clamp(max=1.)[:, None].expand(-1, 2)
        result = solve_transient_thermal(model, time, nodes.new_zeros(n+1), theta=.5,
                                         fixed_temperature_history=wall)
        inward = result.nodal_reaction_heat_steps()[:, [0, n]].reshape(2, steps, 2).sum(1)
        uptake = torch.trapezoid(result.temperature[::steps], nodes[:, 0]).diff()
        balance = (uptake-inward.sum(1)).abs().max().item()
        assert balance < 1e-12
        error = (inward-reference[:, None]).abs().amax(1)
        errors.append(error)
        prefix = f"n{n}_steps{steps}"
        record_property(f"{prefix}_balance_J_m2", balance)
        for phase, value in zip(("ramp", "hold"), error):
            record_property(f"{prefix}_{phase}_wall_error_J_m2", value.item())
    for coarse, fine in zip(errors[:2], errors[1:3]):
        assert torch.all(coarse > 3*fine)
    assert torch.all(errors[-1] < .01*reference)
    assert torch.all(abs(errors[-1]-errors[-2]) < .05*errors[-1])
