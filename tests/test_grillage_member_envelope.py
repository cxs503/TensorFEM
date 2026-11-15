"""Gates for the round-5 member load-path envelope (anchor GAP-PP-5).

The module replaces two position-blind heuristics of the smeared layer
(``n_overlap`` stiffeners share the patch, exactly two frames share it) with
overlap-weighted tributary load paths.  These tests lock down

* the tributary weights themselves (they must pick the owning member and
  conserve the patch force),
* the analytic kernels against classical simply-supported beam closed forms
  (``FL/4``, ``F(2L-c)/8``, ``p a L^2/8``),
* the envelope ordering (every branch <= envelope, system <= envelope),
* the defect the module exists to fix: the legacy smeared frame demand is
  non-conservative for a patch on a frame line and for uniform lateral pressure,
* the classical plate bracket the anchor reports as its literature comparison.
"""
import math

import pytest

from tensorfem.grillage_ice_ultimate import (
    CombinedLoadCase,
    GrillageGeometry,
    MemberSection,
    case_study_geometry,
    grillage_utilisations,
)
from tensorfem.grillage_member_envelope import (
    branch_demands,
    classical_compare,
    classical_plate_bracket,
    member_envelope,
    patch_moment_closed_form,
    ss_beam_moment,
    system_branch,
    tributary_weights,
)

BAY = 0.35
SPAN = 2.1


def _lines(spacing, n):
    return [i * spacing for i in range(n + 1)]


# --------------------------------------------------------------------------- #
# tributary weights
# --------------------------------------------------------------------------- #
def test_patch_on_a_member_line_loads_that_member_alone():
    """A patch that sits inside one tributary strip must load only that line."""
    weights = tributary_weights(_lines(BAY, 8), 1.225, 1.575, 8 * BAY)
    owner = max(weights, key=lambda pw: pw[1])
    assert owner[0] == pytest.approx(1.40)
    assert owner[1] == pytest.approx(1.0)
    assert sum(w for _, w in weights) == pytest.approx(1.0)


def test_patch_straddling_two_tributaries_splits_by_overlap():
    """A patch centred between two lines splits 50/50; force is conserved."""
    weights = tributary_weights(_lines(BAY, 8), 1.05, 1.40, 8 * BAY)
    top = sorted((w for _, w in weights), reverse=True)[:2]
    # a patch spanning one full bay, from member line to member line, sits half in
    # each of the two adjacent tributary strips
    assert top[0] == pytest.approx(0.5, abs=1e-9)
    assert top[1] == pytest.approx(0.5, abs=1e-9)
    assert sum(w for _, w in weights) == pytest.approx(1.0)


def test_patch_outside_the_panel_is_rejected():
    with pytest.raises(ValueError):
        tributary_weights(_lines(BAY, 8), 3.0, 3.2, 8 * BAY)


# --------------------------------------------------------------------------- #
# analytic kernels vs classical closed forms
# --------------------------------------------------------------------------- #
def test_ss_beam_point_load_matches_classical_fl_over_4():
    span, force = 2.8, 1.0e6
    assert ss_beam_moment(span, [(0.5 * span, force)]) == pytest.approx(
        force * span / 4.0, rel=1e-12)


def test_ss_beam_offcentre_load_matches_influence_line():
    span, force, y = 3.0, 5.0e5, 1.0
    assert ss_beam_moment(span, [(y, force)]) == pytest.approx(
        force * y * (span - y) / span, rel=1e-12)


def test_central_patch_closed_form_matches_quadrature():
    span, force, c = 2.8, 1.0e6, 0.35
    closed = patch_moment_closed_form(span, force, c)
    assert closed == pytest.approx(force * (2.0 * span - c) / 8.0, rel=1e-12)
    # the (2L - c)/8 form is a point-load limit check: c -> 0 gives FL/4
    assert patch_moment_closed_form(span, force, 0.0) == pytest.approx(
        force * span / 4.0, rel=1e-12)


# --------------------------------------------------------------------------- #
# branches and envelope ordering
# --------------------------------------------------------------------------- #
def _case(width=1.05, pressure=6.0e6, hydro=0.0):
    return CombinedLoadCase(ice_pressure=pressure, patch_length=width,
                            patch_width=BAY, hydrostatic_pressure=hydro)


@pytest.mark.parametrize("n_spans,expect", [(4, "on_frame_line"), (3, "in_bay")])
def test_branch_frame_force_follows_the_patch_position(n_spans, expect):
    """B2 must hand the whole patch to one frame on a frame line, half in a bay."""
    geom = GrillageGeometry(frame_spacing=2.1, bay_spacing=BAY, n_bays=8,
                            n_spans=n_spans, plate_thickness=0.02,
                            stiffener=MemberSection(0.25, 0.010, 0.080, 0.012),
                            frame=MemberSection(0.55, 0.010, 0.120, 0.014))
    res = branch_demands(geom, _case())
    total = res["B1_stiffener_path"]["patch_force"]
    peak = res["B2_frame_path"]["frame_force_peak"]
    if expect == "on_frame_line":
        assert peak == pytest.approx(total, rel=1e-12)
    else:
        assert peak == pytest.approx(0.5 * total, rel=1e-12)


def test_envelope_bounds_every_branch_and_the_system_model():
    geom = case_study_geometry()
    for n_spans in (4, 3):
        g = GrillageGeometry(**{**geom.__dict__, "n_spans": n_spans})
        for width in (0.35, 0.70, 1.05):
            case = _case(width=width)
            env = member_envelope(g, case, sub=6)
            branch = env["branch_utilisations"]
            for member in ("stiffener", "frame"):
                for name, value in branch[member].items():
                    assert env[f"{member}_utilisation"] >= value - 1e-12, (name, member)
            assert env["stiffener_utilisation"] >= 0.0
            assert env["frame_utilisation"] >= 0.0


def test_system_branch_is_linear_in_pressure():
    """The unit-pressure caching must not change the linear scaling."""
    geom = case_study_geometry()
    a = system_branch(geom, _case(pressure=1.0e6))
    b = system_branch(geom, _case(pressure=3.0e6))
    assert b["frame_moment"] == pytest.approx(3.0 * a["frame_moment"], rel=1e-9)
    assert b["stiffener_moment"] == pytest.approx(3.0 * a["stiffener_moment"], rel=1e-9)


def test_uniform_load_frame_moment_matches_classical_paL2_over_8():
    """Uniform pressure: the frame must carry its own tributary strip."""
    geom = case_study_geometry()
    p = 1.0e5
    res = system_branch(geom, CombinedLoadCase(hydrostatic_pressure=p))
    analytic = p * geom.frame_spacing * geom.frame_span ** 2 / 8.0
    assert res["frame_moment"] == pytest.approx(analytic, rel=0.15)


def test_legacy_smeared_frame_demand_is_non_conservative_uniform_load():
    """Documented defect: the legacy layer halves a load that fills both panels.

    The smeared layer reports ~1/25 of the tributary-strip moment for a uniform
    pressure (it uses a single panel's force and the ``two frames share it``
    heuristic), so its frame utilisation is not a safety margin.
    """
    geom = case_study_geometry()
    case = CombinedLoadCase(hydrostatic_pressure=1.0e5)
    legacy = grillage_utilisations(geom, case)
    system = system_branch(geom, case)
    assert legacy["transverse_frame"] < 0.2 * system["frame_utilisation"]


def test_legacy_smeared_frame_demand_is_non_conservative_on_a_frame_line():
    """The round-4 finding, kept as a regression guard on the legacy branch."""
    geom = case_study_geometry()
    case = _case(width=1.05)
    legacy = grillage_utilisations(geom, case)
    system = system_branch(geom, case)
    assert legacy["transverse_frame"] < 0.5 * system["frame_utilisation"]


def test_member_model_switch_keeps_the_legacy_default():
    geom = case_study_geometry()
    case = _case(width=0.70)
    default = grillage_utilisations(geom, case)
    explicit = grillage_utilisations(geom, case, member_model="smeared")
    assert default["transverse_frame"] == explicit["transverse_frame"]
    assert default["longitudinal_stiffener"] == explicit["longitudinal_stiffener"]
    # the panel branch carries no patch load into the stiffener through B2, but
    # B1 does, so the panel stiffener utilisation must dominate the smeared one
    panel = grillage_utilisations(geom, case, member_model="panel")
    assert panel["longitudinal_stiffener"] > default["longitudinal_stiffener"]
    for key in ("u_stiffener_panel", "u_frame_panel", "u_frame_system"):
        assert key in panel


def test_unknown_member_model_is_rejected():
    geom = case_study_geometry()
    with pytest.raises(ValueError):
        grillage_utilisations(geom, _case(), member_model="whatever")


# --------------------------------------------------------------------------- #
# classical plate bracket (the anchor's literature comparison)
# --------------------------------------------------------------------------- #
def test_classical_plate_bracket_formulas():
    geom = case_study_geometry()
    br = classical_plate_bracket(geom)
    mp = geom.yield_stress * geom.plate_thickness ** 2 / 4.0
    assert br["p_first_hinge_ss_strip"] == pytest.approx(8.0 * mp / geom.bay_spacing ** 2)
    assert br["p_plastic_collapse_clamped_strip"] == pytest.approx(
        16.0 * mp / geom.bay_spacing ** 2)
    assert br["p_first_hinge_ss_strip"] < br["p_plastic_collapse_clamped_strip"]


def test_plate_controlled_ultimate_is_within_15_percent_of_the_classical_strip():
    res = classical_compare(case_study_geometry())
    assert res["model_governing_member"] == "plate_local_bending"
    assert abs(res["relative_error_vs_ss_strip"]) < 0.15, res
    # and it must sit inside the [first hinge, plastic collapse] bracket
    assert res["reference_ss_strip_first_hinge"] <= res["model_p_ult"] <= \
        res["reference_clamped_strip_plastic_collapse"]