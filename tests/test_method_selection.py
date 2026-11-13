"""Rules of ``tensorfem.method_selection`` + the beam reference-selection record.

The applied record is *re-derived* here from its own raw probe errors: the
tests never trust a stored ``selected``/``admissible``/``margin`` field, they
recompute it (same discipline as ``tools/verified_anchor_audit.py``).
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import pytest

from tensorfem.method_selection import (
    MethodCandidate,
    SelectionRecord,
    certify_selection,
    pareto_frontier,
    select_method,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks" / "beam_reference_method_selection_record.json"
sys.path.insert(0, str(ROOT / "benchmarks"))


# ---------------------------------------------------------------------------
# R1 comparability / fail-closed input handling
# ---------------------------------------------------------------------------
def test_mismatched_probe_sets_are_refused():
    a = MethodCandidate("a", 1.0, {"x": 0.01})
    b = MethodCandidate("b", 2.0, {"y": 0.01})
    with pytest.raises(ValueError, match="not comparable"):
        select_method([a, b], 0.03)


def test_extra_probe_is_also_a_mismatch():
    a = MethodCandidate("a", 1.0, {"x": 0.01})
    b = MethodCandidate("b", 2.0, {"x": 0.01, "y": 0.02})
    with pytest.raises(ValueError, match="extra probes"):
        select_method([a, b], 0.03)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1e-3])
def test_non_finite_or_negative_errors_are_refused(bad):
    with pytest.raises(ValueError):
        select_method([MethodCandidate("a", 1.0, {"x": bad})], 0.03)


def test_duplicate_names_and_empty_input_refused():
    a = MethodCandidate("a", 1.0, {"x": 0.01})
    with pytest.raises(ValueError, match="duplicate"):
        select_method([a, MethodCandidate("a", 2.0, {"x": 0.02})], 0.03)
    with pytest.raises(ValueError, match="at least one"):
        select_method([], 0.03)
    with pytest.raises(ValueError, match="no probes"):
        select_method([MethodCandidate("a", 1.0, {})], 0.03)


@pytest.mark.parametrize("tol", [0.0, -0.03, float("nan")])
def test_invalid_tolerance_refused(tol):
    with pytest.raises(ValueError, match="tolerance"):
        select_method([MethodCandidate("a", 1.0, {"x": 0.01})], tol)


# ---------------------------------------------------------------------------
# R2/R3 admissibility, cost ordering and tie-breaks
# ---------------------------------------------------------------------------
def test_boundary_error_equal_to_tolerance_is_admissible():
    rec = select_method([MethodCandidate("edge", 1.0, {"x": 0.03})], 0.03)
    assert rec.admissible == ("edge",)
    assert rec.selected == "edge"
    assert rec.margin["edge"]["x"] == pytest.approx(0.0, abs=1e-15)


def test_cheapest_admissible_wins_even_when_less_accurate():
    cheap = MethodCandidate("cheap", 1.0, {"x": 0.02})
    fine = MethodCandidate("fine", 5.0, {"x": 1e-6})
    rec = select_method([fine, cheap], 0.03)
    assert rec.selected == "cheap"
    assert rec.margin["cheap"]["x"] == pytest.approx(0.02 / 0.03 - 1.0)
    # both on the frontier: cost buys accuracy
    assert set(rec.pareto) == {"cheap", "fine"}


def test_cost_tie_breaks_on_accuracy_then_name():
    a = MethodCandidate("alpha", 4.0, {"x": 0.029})
    b = MethodCandidate("beta", 4.0, {"x": 0.001})
    rec = select_method([a, b], 0.03)
    assert rec.selected == "beta"
    # identical cost and identical accuracy -> lexicographic, deterministic
    rec2 = select_method(
        [MethodCandidate("zeta", 4.0, {"x": 0.01}), MethodCandidate("alpha", 4.0, {"x": 0.01})],
        0.03,
    )
    assert rec2.selected == "alpha"


def test_cheapest_inadmissible_is_not_selected():
    rec = select_method(
        [MethodCandidate("cheap", 1.0, {"x": 0.20}), MethodCandidate("fine", 9.0, {"x": 0.01})],
        0.03,
    )
    assert rec.selected == "fine"
    assert rec.margin["cheap"]["x"] == pytest.approx(0.20 / 0.03 - 1.0)


def test_worst_probe_is_the_binding_one():
    rec = select_method([MethodCandidate("a", 1.0, {"p": 0.001, "q": 0.02, "r": 0.005})], 0.03)
    assert rec.worst_probe["a"] == "q"
    assert rec.worst["a"] == pytest.approx(0.02)


# ---------------------------------------------------------------------------
# R4 frontier consistency, R5 no-winner, R6 margins
# ---------------------------------------------------------------------------
def test_pareto_frontier_basic():
    cands = [
        MethodCandidate("dominated", 5.0, {"x": 0.02}),
        MethodCandidate("cheap", 1.0, {"x": 0.02}),
        MethodCandidate("accurate", 9.0, {"x": 0.001}),
    ]
    assert pareto_frontier(cands) == ("cheap", "accurate")


def test_no_admissible_method_fails_closed():
    rec = select_method(
        [MethodCandidate("a", 1.0, {"x": 0.5}), MethodCandidate("b", 2.0, {"x": 0.4})], 0.03
    )
    assert rec.decision == "no_admissible_method"
    assert rec.selected is None
    assert rec.admissible == ()
    assert certify_selection(rec)["certified"] is True


def test_margins_are_signed_and_relative():
    rec = select_method(
        [MethodCandidate("in", 1.0, {"x": 0.015}), MethodCandidate("out", 2.0, {"x": 0.045})],
        0.03,
    )
    assert rec.margin["in"]["x"] == pytest.approx(-0.5)
    assert rec.margin["out"]["x"] == pytest.approx(0.5)
    assert rec.selected == "in"


# ---------------------------------------------------------------------------
# certification: recomputation must catch tampering
# ---------------------------------------------------------------------------
def _base_record() -> SelectionRecord:
    return select_method(
        [MethodCandidate("cheap", 1.0, {"x": 0.02, "y": 0.01}),
         MethodCandidate("fine", 6.0, {"x": 0.001, "y": 0.002})],
        0.03,
    )


def test_clean_record_certifies_and_round_trips():
    rec = _base_record()
    assert certify_selection(rec)["certified"] is True
    again = SelectionRecord.from_dict(json.loads(json.dumps(rec.to_dict())))
    assert certify_selection(again)["certified"] is True
    assert again.selected == rec.selected


def _tampered(rec: SelectionRecord, **overrides) -> SelectionRecord:
    data = rec.to_dict()
    data.update(overrides)
    return SelectionRecord.from_dict(data)


def test_tampered_selected_is_caught():
    rec = _base_record()
    fake = _tampered(rec, selected="fine")
    cert = certify_selection(fake)
    assert cert["certified"] is False
    assert any("selected" in m for m in cert["mismatches"])


def test_tampered_margin_is_caught():
    rec = _base_record()
    margin = {k: dict(v) for k, v in rec.margin.items()}
    margin["cheap"]["x"] = -0.99
    fake = _tampered(rec, margin=margin)
    cert = certify_selection(fake)
    assert cert["certified"] is False
    assert any("margin" in m for m in cert["mismatches"])


def test_hidden_rule_violation_is_caught_by_recomputation():
    """A record whose raw errors contradict its own winner cannot certify."""
    rec = _base_record()
    candidates = [c.to_dict() for c in rec.candidates]
    # make "fine" both cheaper and more accurate than the declared winner
    for c in candidates:
        if c["name"] == "fine":
            c["cost"] = 0.5
            c["error"] = {"x": 0.0001, "y": 0.0001}
    fake = _tampered(rec, candidates=candidates)
    cert = certify_selection(fake)
    assert cert["certified"] is False
    assert cert["recomputed"]["selected"] == "fine"


def test_admissible_without_selection_is_caught():
    rec = _base_record()
    fake = _tampered(rec, decision="no_admissible_method", selected=None,
                     admissible=[])
    cert = certify_selection(fake)
    assert cert["certified"] is False


# ---------------------------------------------------------------------------
# the applied beam record: re-derived, not trusted
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def beam_record() -> dict:
    return json.loads(RECORD.read_text())


def test_reference_closed_form_is_reproduced_by_fem(beam_record):
    ver = beam_record["reference_verification"]
    assert ver["worst_refined_rel_error"] < 1e-10
    assert all(c["verified"] for c in ver["states"])
    # the EB closed form and the EB-FEM route must also agree
    assert beam_record["eb_route_consistency"]["worst_rel_error"] < 1e-10


def test_every_selection_block_recertifies(beam_record):
    for key, block in beam_record["selections"].items():
        probes = tuple(block["probes"])
        tol = float(block["tolerance"])
        for row in block["rows"]:
            state = next(s for s in beam_record["states"]
                         if s["slenderness"] == row["slenderness"])
            table = state["probes_nodal"]
            cands = [
                MethodCandidate(name, cost, {p: table[name][p] for p in probes})
                for name, cost in (("EB-closed", 1.0), ("TIMO-closed", 3.0),
                                   ("TIMO-fem-n2", 6.0))
            ]
            rec = select_method(cands, tol)
            assert certify_selection(rec)["certified"] is True
            assert row["selected"] == rec.selected, (key, row)
            assert row["admissible"] == list(rec.admissible), (key, row)


def test_tip_only_probe_widens_the_admissible_band(beam_record):
    """Probe choice moves the admissible slenderness: tip-only is optimistic."""
    allprobe = beam_record["selections"]["nodal_all_probes|tol=0.03"]
    tiponly = beam_record["selections"]["tip_only|tol=0.03"]
    assert tiponly["first_EB_admissible_slenderness"] < \
        allprobe["first_EB_admissible_slenderness"]
    ratio = allprobe["first_EB_admissible_slenderness"] / \
        tiponly["first_EB_admissible_slenderness"]
    assert ratio > 1.3


def test_breakeven_routes_agree(beam_record):
    for key, block in beam_record["breakeven"].items():
        assert block["route_agreement_lt_3pct"], key
        assert block["rel_diff_between_routes"] < 0.03
        # mid-span (the binding probe) needs a slender beam than tip
        assert block["closed_form_w_mid"] > block["closed_form_w_tip"]


def test_breakeven_is_where_the_deficit_crosses_the_line(beam_record):
    """Interpolate the recorded sweep and compare with the declared break-even."""
    for tol_key, block in beam_record["breakeven"].items():
        tol = float(tol_key.split("=")[1])
        states = sorted(beam_record["states"], key=lambda s: s["slenderness"])
        pts = [(s["slenderness"], s["probes_nodal"]["EB-closed"]["w_mid"]) for s in states]
        below = [(x, e) for x, e in pts if e < tol]
        above = [(x, e) for x, e in pts if e > tol]
        lo_x, lo_e = above[-1]
        hi_x, hi_e = below[0]
        interp = lo_x + (tol - lo_e) * (hi_x - lo_x) / (hi_e - lo_e)
        assert block["closed_form_w_mid"] == pytest.approx(interp, rel=0.05)


def test_protocol_inflation_is_recorded_and_grows_with_slenderness(beam_record):
    rows = beam_record["protocol_sensitivity"]
    first, last = rows[0], rows[-1]
    assert first["inflation_factor"] < last["inflation_factor"]
    assert last["inflation_factor"] > 10.0
    for row in rows:
        assert row["w_mid_field_n1"] >= row["w_mid_nodal"]


def test_model_deficit_matches_the_analytic_shear_share(beam_record):
    """EB tip deficit *is* the analytic shear share -- an identity, not a fit."""
    for state in beam_record["states"]:
        assert state["probes_nodal"]["EB-closed"]["w_tip"] == pytest.approx(
            state["analytic_shear_share_tip"], rel=1e-9
        )


def test_deficit_orders_probes_w_mid_worse_than_tip(beam_record):
    for state in beam_record["states"]:
        eb = state["probes_nodal"]["EB-closed"]
        assert eb["w_mid"] > eb["w_tip"]
        assert eb["psi_tip"] < 1e-12   # shear-insensitive control probe


# ---------------------------------------------------------------------------
# library-level guards the record depends on
# ---------------------------------------------------------------------------
def test_elements_are_nodal_exact_for_uniform_load():
    """Both 2-node elements reproduce nodal deflections at any mesh -- the
    premise that makes a nodal protocol measure model error only."""
    from beam_reference_method_selection import (L, NODAL_MESH, eb_exact_w,
                                                 fem_solve, timo_exact_w)

    for ratio in (3.0, 10.0, 40.0):
        h = L / ratio
        u_eb = fem_solve("EB", h, NODAL_MESH)
        u_ti = fem_solve("TIMO", h, NODAL_MESH)
        assert abs(float(u_eb[2 * NODAL_MESH]) - eb_exact_w(L, h)) / eb_exact_w(L, h) < 1e-12
        assert abs(float(u_ti[2 * NODAL_MESH]) - timo_exact_w(L, h)) / timo_exact_w(L, h) < 1e-12


def test_timoshenko_is_softer_than_euler_bernoulli_always():
    from beam_reference_method_selection import L, NODAL_MESH, fem_solve

    for ratio in (2.0, 5.0, 20.0):
        h = L / ratio
        w_eb = float(fem_solve("EB", h, NODAL_MESH)[2 * NODAL_MESH])
        w_ti = float(fem_solve("TIMO", h, NODAL_MESH)[2 * NODAL_MESH])
        assert w_ti > w_eb