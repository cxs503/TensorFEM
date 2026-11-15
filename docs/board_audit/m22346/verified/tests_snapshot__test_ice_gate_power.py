"""Tests for the GAP-ICE-GATE-POWER information-content gate.

The tests are written against the *ecosystem* requirement that a REAL criterion
must supply (a) a baseline source, (b) a threshold tied to a detection rate (or
observed order) and (c) at least one rejected negative control.  They are
deliberately *non-tautological*: the control outcomes are produced from real
numbers read out of the ice chain, out of an independent report and out of an
independent in-repository BEM route, and ``test_mutation_makes_the_gate_accept``
proves the decision follows the data path rather than a constant.
"""
from __future__ import annotations

import math

import pytest

from tensorfem import ice_gate_power as gate


@pytest.fixture(scope="module")
def verdict():
    return gate.gate_verdict(per_edge=80, n_mc=20_000)


# --------------------------------------------------------------------------- #
# (a) the judged quantity is read from the chain, with the chain's geometry
# --------------------------------------------------------------------------- #
def test_declared_value_is_read_from_the_chain():
    d = gate.declared_added_mass_per_length("heave")
    p = gate.chain_particulars()
    assert d["added_mass_total_kg"] == pytest.approx(p["mass_kg"], rel=1e-12)
    assert d["value_kg_per_m"] == pytest.approx(p["mass_kg"] / p["length_m"], rel=1e-12)
    assert d["value_t_per_m"] == pytest.approx(86.1, abs=0.05)


def test_chain_geometry_is_the_audited_one():
    """Guards the m22156 failure mode: the reference used T=8 m instead of 6 m."""
    p = gate.chain_particulars()
    assert (p["length_m"], p["breadth_m"], p["draught_m"]) == (100.0, 20.0, 6.0)
    assert p["block_coefficient"] == pytest.approx(0.70)
    assert p["mass_kg"] == pytest.approx(8.610e6, rel=1e-9)


def test_judged_quantity_is_two_dimensional():
    """Guards the m22156 failure mode: a 3-D kg value compared with a 2-D kg/m one."""
    d = gate.declared_added_mass_per_length("heave")
    ref = gate.section_added_mass_2d(20.0, 6.0, per_edge=40)
    assert 0.1 * ref < d["value_kg_per_m"] < 10.0 * ref


# --------------------------------------------------------------------------- #
# the reference route: validated on an exact anchor, converged in panel count
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("condition", ["omega0", "infinity"])
def test_deep_cylinder_recovers_rho_pi_a_squared(condition):
    check = gate.deep_cylinder_check(depth=8.0, n=480)
    value = check["even_mirror"] if condition == "omega0" else check["odd_mirror"]
    assert value == pytest.approx(check["exact_kg_per_m"], rel=0.03)


def test_two_routes_bracket_the_independent_anchor(verdict):
    lo = verdict["reference"]["band"]["min_kg_per_m"]
    hi = verdict["reference"]["band"]["max_kg_per_m"]
    ref = verdict["reference"]["primary_kg_per_m"]
    assert lo < ref < hi
    assert verdict["reference"]["band_half_width_rel"] < gate.DEFAULT_U_REL


@pytest.mark.parametrize("condition", ["omega0", "infinity"])
def test_convergence_order_is_measured_and_positive(condition):
    conv = gate.convergence_study(per_edges=(160, 320, 640), free_surface=condition)
    assert conv["observed_order"] > 0.2          # sublinear: corner singularity
    assert conv["observed_order"] < 2.5          # but a genuine, reported order
    rich = conv["richardson_kg_per_m"]
    assert abs(conv["values_kg_per_m"][-1] / rich - 1.0) < 5e-3


# --------------------------------------------------------------------------- #
# (b) the threshold expressed as a detection rate
# --------------------------------------------------------------------------- #
def test_guard_band_arithmetic():
    assert gate.acceptance_limit(0.15, 0.029, 2.0) == pytest.approx(0.092)
    assert gate.rejection_limit(0.15, 0.029, 2.0) == pytest.approx(0.208)
    assert gate.decide(0.15, 0.15, 0.029, 2.0) == "indeterminate"
    assert gate.decide(0.05, 0.15, 0.029, 2.0) == "accept"
    assert gate.decide(-0.30, 0.15, 0.029, 2.0) == "reject"


def test_power_at_audited_deviation_is_near_one(verdict):
    assert verdict["power"]["at_audited_deviation"] >= 0.95
    assert verdict["power"]["mc_at_audited_deviation"] >= 0.95


def test_monte_carlo_matches_normal_approximation():
    analytic = gate.detection_power_analytic(0.30, 0.15, 0.029, 2.0)
    mc = gate.detection_power(0.30, 0.15, 0.029, 2.0, n_mc=60_000, seed=11)
    assert mc == pytest.approx(analytic, abs=0.01)


def test_guard_band_is_more_informative_than_the_naive_rule():
    """The naive rule is a coin flip on the tolerance it is supposed to police."""
    naive = gate.detection_power_analytic(0.15, 0.15, 0.029, 0.0)
    guarded_false_accept = gate.false_accept_probability(0.15, 0.15, 0.029, 2.0)
    assert naive == pytest.approx(0.5, abs=1e-12)
    assert guarded_false_accept < 0.05
    # and the guard band does not buy power by being trigger happy at delta = 0
    assert gate.false_accept_probability(0.0, 0.15, 0.029, 2.0) > 0.99


def test_critical_deviation_exceeds_the_tolerance(verdict):
    """With u=2.9% a 15% tolerance is only *decidable* above ~25% deviation."""
    cd = verdict["power"]["critical_deviation_95"]
    assert 0.15 < cd < 0.30
    assert gate.detection_power_analytic(cd, 0.15, 0.029, 2.0) == pytest.approx(0.95, abs=2e-3)


# --------------------------------------------------------------------------- #
# (c) controls: real values, predeclared outcomes, actual outcomes
# --------------------------------------------------------------------------- #
def test_all_controls_behave_as_predeclared(verdict):
    for c in verdict["controls"]:
        assert c["got"] == c["expected"], (c["name"], c["deviation"], c["got"])


def test_negative_controls_are_real_numbers_not_scaled_synthetics(verdict):
    """Every control must be traceable to a computation or a chain channel."""
    by_name = {c["name"]: c for c in verdict["controls"]}
    declared = verdict["declared"]["heave"]["value_kg_per_m"]
    assert by_name["N1_chain_heave_declared"]["value_kg_per_m"] == pytest.approx(declared)
    assert by_name["N2_chain_surge_channel"]["value_kg_per_m"] == pytest.approx(0.1 * declared)
    assert by_name["N3_sign_flipped_heave"]["value_kg_per_m"] == pytest.approx(-declared)
    assert by_name["N4_lock_dead_zero"]["value_kg_per_m"] == 0.0
    assert by_name["P3_repo_route_even_mirror"]["value_kg_per_m"] == pytest.approx(
        verdict["reference"]["repo_route_omega0_kg_per_m"])
    for c in verdict["controls"]:
        assert c["provenance"]


def test_heave_declaration_is_rejected(verdict):
    assert verdict["decision"]["heave"] == "reject"
    assert verdict["deviations"]["heave"] < -0.5
    assert verdict["pass"] is True


def test_degenerate_gates_are_rejected():
    """A gate that accepts everything, and one that accepts nothing, both fail."""
    too_wide = gate.meta_gate(2.0, 0.623)
    too_tight = gate.meta_gate(0.0, 0.623)
    assert too_wide["informative"] is False
    assert too_tight["informative"] is False
    assert gate.meta_gate(0.15, 0.623)["informative"] is True


def test_reference_disagreeing_with_the_anchor_cannot_be_decided(monkeypatch):
    """The guard band is data-driven: a route that contradicts the anchor widens it."""
    monkeypatch.setattr(gate, "section_added_mass_2d", lambda *a, **kw: 86.1e3)
    mutated = gate.gate_verdict(per_edge=20, dofs=("heave",), n_mc=2_000)
    assert mutated["rule"]["u_used"] > 1.0          # 165% anchor disagreement
    assert mutated["decision"]["heave"] == "indeterminate"
    assert mutated["pass"] is False


def test_coherent_mutation_makes_the_gate_accept(monkeypatch):
    """Non-tautology check: with a (wrong but) coherent reference the gate accepts."""
    monkeypatch.setattr(gate, "section_added_mass_2d", lambda *a, **kw: 86.1e3)
    monkeypatch.setitem(gate.REFERENCE_19234, "value_kg_per_m", 86.1e3)
    mutated = gate.gate_verdict(per_edge=20, dofs=("heave",), n_mc=2_000)
    assert mutated["rule"]["u_used"] == pytest.approx(gate.DEFAULT_U_REL)
    assert mutated["decision"]["heave"] == "accept"
    assert mutated["pass"] is False        # the negative controls now misbehave