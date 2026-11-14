"""Tests for the independent discrete beam-grid reference (round 4).

The module exists to check what the smeared orthotropic layer of
``grillage_ice_ultimate`` *assumes*: the split of an ice patch between the
stiffener system and the frame system.  These tests lock down

* the element/assembly against simply supported beam closed forms (a beam grid
  that cannot reproduce ``5qL^4/384EI`` validates nothing),
* vertical equilibrium and isotropy symmetry,
* the patch/tributary intersection (a patch inside one bay must load the member
  line that owns the bay -- the defect this round found),
* mesh and load-split sensitivity,
* the two member-demand verdicts: the stiffener line agrees with the smeared
  layer within the 15 % ledger band, the frame line does not (documented defect).
"""
import math

import pytest

from tensorfem.grillage_beam_grid import (
    beam_grid_case_study,
    beam_grid_response,
    compare_with_smeared,
    run_beam_grid_verification,
    smeared_member_demands,
)

E, I = 2.06e11, 1.0e-4
SPAN, Q = 4.0, 1.0e4


def _beam(**kw):
    kwargs = dict(span_x=SPAN, span_y=1.0, ei_x=E * I, gj_x=1.0e3,
                  ei_y=E * I, gj_y=1.0e3, n_x_lines=9, n_y_lines=1,
                  sub=8, support_edges=("x0", "x1"))
    kwargs.update(kw)
    return beam_grid_response(**kwargs)


# --------------------------------------------------------------------------- #
# element / assembly verification
# --------------------------------------------------------------------------- #
def test_ss_beam_uniform_load_closed_form():
    """``w_max = 5qL^4/(384EI)`` and ``M_max = qL^2/8`` for a uniform line load."""
    res = _beam(pressure=Q, load_split=1.0)      # unit tributary => line load Q
    assert res["w_max"] == pytest.approx(5.0 * Q * SPAN ** 4 / (384.0 * E * I), rel=5e-3)
    assert res["moment_x_max"] == pytest.approx(Q * SPAN ** 2 / 8.0, rel=2e-2)


def test_ss_beam_central_point_load_closed_form():
    """``w_max = FL^3/(48EI)`` and ``M_max = FL/4`` for a central point load."""
    F = 1.0e5
    res = _beam(point_loads=[(0.5 * SPAN, 0.0, F)])
    assert res["w_max"] == pytest.approx(F * SPAN ** 3 / (48.0 * E * I), rel=5e-3)
    assert res["moment_x_max"] == pytest.approx(F * SPAN / 4.0, rel=5e-3)


def test_vertical_equilibrium_closes():
    """Sum of vertical reactions equals the applied patch force."""
    res = beam_grid_case_study(sub=6)
    assert res["reaction_total"] == pytest.approx(res["load_total"], rel=1e-9)


def test_isotropic_grid_shares_are_symmetric():
    """An isotropic square grid under a uniform load must split 50/50 by symmetry."""
    res = beam_grid_response(span_x=6.0, span_y=6.0, ei_x=E * I, gj_x=2.0e5,
                             ei_y=E * I, gj_y=2.0e5, n_x_lines=7, n_y_lines=7,
                             sub=4, pressure=1.0e4)
    assert res["share_x_reaction"] == pytest.approx(res["share_y_reaction"], abs=1e-9)
    assert res["share_x_energy"] == pytest.approx(res["share_y_energy"], abs=1e-9)
    assert res["share_x_energy"] + res["share_y_energy"] + \
        res["share_torsion_energy"] == pytest.approx(1.0, abs=1e-12)


def test_rigid_y_family_sends_load_to_girders():
    """A rigid frame family becomes a rigid support line: the load leaves via y."""
    kwargs = dict(span_x=6.0, span_y=6.0, gj_x=2.0e5, gj_y=2.0e5,
                  n_x_lines=7, n_y_lines=7, sub=4, pressure=1.0e4)
    iso = beam_grid_response(ei_x=E * I, ei_y=E * I, **kwargs)
    rigid_y = beam_grid_response(ei_x=E * I, ei_y=1.0e6 * E * I, **kwargs)
    rigid_x = beam_grid_response(ei_x=1.0e6 * E * I, ei_y=E * I, **kwargs)
    assert rigid_y["share_y_reaction"] > iso["share_y_reaction"]
    assert rigid_x["share_x_reaction"] > iso["share_x_reaction"]
    # rigid supports shorten the *other* family's spans, so that family still
    # supplies the bending energy -- recorded because it is counter-intuitive
    assert rigid_y["share_x_energy"] > rigid_y["share_y_energy"]
    assert rigid_x["share_y_energy"] > rigid_x["share_x_energy"]


# --------------------------------------------------------------------------- #
# load application
# --------------------------------------------------------------------------- #
def test_patch_inside_one_bay_loads_that_member_line_only():
    """A patch confined to one bay must not be smeared over every member line.

    Regression for the defect found in round 4: the tributary weights were not
    intersected with the patch extent, so a 0.35 m wide patch was spread over all
    nine stiffener lines and the peak stiffener moment came out ~5x too low.
    """
    narrow = beam_grid_case_study(sub=6)
    # same total force, spread over the whole width instead of one bay
    spread = beam_grid_case_study(sub=6, patch=(0.4375, 0.5625, 0.0, 1.0),
                                  pressure=6.0e6 / 8.0)
    assert narrow["load_total"] == pytest.approx(spread["load_total"], rel=1e-12)
    assert narrow["moment_x_max"] > 2.0 * spread["moment_x_max"]


def test_patch_outside_grid_is_rejected():
    with pytest.raises(ValueError):
        beam_grid_response(span_x=1.0, span_y=1.0, ei_x=E * I, gj_x=1.0,
                           ei_y=E * I, gj_y=1.0, n_x_lines=2, n_y_lines=2,
                           pressure=1.0, patch=(1.2, 0.3, 0.0, 1.0))


def test_invalid_arguments_are_rejected():
    base = dict(span_x=1.0, span_y=1.0, ei_x=E * I, gj_x=1.0, ei_y=E * I,
                gj_y=1.0, n_x_lines=2, n_y_lines=2)
    with pytest.raises(ValueError):
        beam_grid_response(load_split=1.5, **base)
    with pytest.raises(ValueError):
        beam_grid_response(**{**base, "span_x": -1.0})
    with pytest.raises(ValueError):
        beam_grid_response(**{**base, "n_x_lines": 1})


# --------------------------------------------------------------------------- #
# convergence and assumption sensitivity
# --------------------------------------------------------------------------- #
def test_mesh_convergence_of_the_share():
    coarse = beam_grid_case_study(sub=3)
    fine = beam_grid_case_study(sub=12)
    assert fine["share_x_energy"] == pytest.approx(coarse["share_x_energy"], rel=0.02)
    assert fine["share_x_reaction"] == pytest.approx(coarse["share_x_reaction"], rel=0.02)


def test_load_split_assumption_barely_moves_the_frame_demand():
    """The frame moment is a structural result, not an artefact of ``load_split``."""
    moments = [beam_grid_case_study(sub=4, load_split=s)["moment_y_max"]
               for s in (0.0, 0.5, 1.0)]
    assert max(moments) / min(moments) < 1.05


# --------------------------------------------------------------------------- #
# the verification-debt verdicts
# --------------------------------------------------------------------------- #
def test_smeared_energy_share_is_within_the_declared_band():
    """Smeared energy share vs the discrete grid energy share (band 20 %)."""
    cmp = compare_with_smeared(sub=12)
    smeared = cmp["smeared_share_x"]
    discrete = cmp["discrete_share_x_energy"]
    assert cmp["share_x_smeared_vs_discrete_energy"] == pytest.approx(
        (smeared - discrete) / discrete, rel=1e-12)
    assert abs(cmp["share_x_smeared_vs_discrete_energy"]) < 0.20
    # both models agree the stiffener system is the softer path for this grillage
    assert smeared < 0.5 and discrete < 0.5


def test_stiffener_demand_agrees_with_the_discrete_grid_within_15_percent():
    """The smeared stiffener moment is conservative and inside the ledger band."""
    demands = smeared_member_demands()
    discrete = beam_grid_case_study(sub=24)["moment_x_max"]
    err = (demands["stiffener_moment"] - discrete) / discrete
    assert abs(err) < 0.15, err
    assert err > 0.0                      # conservative, not non-conservative


def test_frame_demand_defect_is_documented_not_hidden():
    """The smeared frame demand is ~2x low; the frame capacity is ~4x optimistic.

    This is a *finding*, locked so it cannot silently change: if the criterion is
    repaired later the numbers below will move and this test must be updated.
    """
    payload = run_beam_grid_verification()
    findings = payload["findings"]
    assert findings["frame_moment_ratio_discrete_over_smeared"] > 1.5
    assert findings["frame_capacity_ratio_smeared_over_discrete"] > 2.0
    assert findings["frame_utilisation_at_ultimate_discrete"] > \
        findings["frame_utilisation_at_ultimate_smeared"]
    # the governing mode of the case study itself is the plate field, so the
    # anchor's headline ultimate is unaffected by the frame defect
    assert findings["governing_member_at_ultimate"] == "plate_local_bending"
    assert findings["frame_utilisation_at_ultimate_discrete"] < 1.0


def test_verification_suite_passes():
    payload = run_beam_grid_verification()
    failed = [g["gate"] for g in payload["gates"] if not g["pass"]]
    assert not failed, failed
    assert payload["n_gates"] >= 15