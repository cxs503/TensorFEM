"""Verification gates for the exact large-angle righting-arm oracle.

Every anchor here is an *independent* route to the same number:

* the free-surface height solved from ``A(phi, h0) = B*T`` versus the
  wall-sided closure ``h0 = T*cos(phi)``;
* the clipped-section centroid versus closed-form trapezoid integrals
  ``y_B = BM*tan(phi)`` and ``z_B = KB + BM*tan(phi)**2/2``;
* the exact arm versus the textbook wall-sided form
  ``sin(phi)*(GM + BM*tan(phi)**2/2)``;
* the Simpson integral of the exact arm versus
  ``GM*(1-cos) + BM*(sec+cos-2)/2``;
* the merged ``tensorfem.marine_hydrostatics`` linear model versus the exact
  arm, which is the verification debt this module was written to measure.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfem.marine_hydrostatics import SCHEMA as HYDROSTATICS_SCHEMA
from tensorfem.marine_hydrostatics import analyze_box_barge, load_hydrostatics_case
from tensorfem.marine_large_angle_stability import (
    SMALL_ANGLE_LIMIT_DEG,
    BoxBarge,
    WallSidedBarge,
    gz_closed_form,
    polygon_section,
)

LENGTH = 90.0
BEAM = 20.0
DEPTH = 8.0
DRAFT = 6.0
KG = 4.5


def _barge(**overrides) -> BoxBarge:
    parameters = dict(length=LENGTH, beam=BEAM, depth=DEPTH, draft=DRAFT, kg=KG)
    parameters.update(overrides)
    return BoxBarge(**parameters)


def _hydrostatics_case(barge: BoxBarge, heel_deg: float) -> dict:
    return {
        "schema": HYDROSTATICS_SCHEMA,
        "name": "large-angle audit case",
        "units": "SI",
        "geometry": {"length": barge.length, "beam": barge.beam, "depth": barge.depth},
        "fluid": {"density": barge.density, "gravity": barge.gravity},
        "condition": {
            "mode": "specified_draft",
            "draft": barge.draft,
            "kg": barge.kg,
            "heel_angle_deg": heel_deg,
        },
    }


def test_free_surface_height_matches_wall_sided_closure():
    """Solved flotation offset equals ``T*cos(phi)`` for both heel signs."""
    barge = _barge()
    for heel in (-9.0, -5.0, -1.0, 0.0, 0.5, 3.0, 7.5, 9.9):
        solved = barge.solve_free_surface_height(heel)
        closed = barge.free_surface_height_closed_form(heel)
        assert abs(solved - closed) <= 2e-13 * max(1.0, barge.depth)


def test_immersed_area_and_waterplane_follow_wall_sided_geometry():
    """Unit-length displacement is conserved and the chord grows as ``B/cos``."""
    barge = _barge()
    target = barge.beam * barge.draft
    upper = 0.995 * barge.wall_sided_envelope_deg
    for index in range(1, 25):
        heel = upper * index / 24.0
        section = barge.immersed_section(heel)
        assert section.relative_area_error(target) <= 1e-12
        expected_chord = barge.beam / math.cos(math.radians(heel))
        assert abs(section.waterline_width - expected_chord) <= 1e-9 * expected_chord
        assert section.deck_immersed is False


def test_clipped_centroid_matches_analytic_trapezoid_integrals():
    """Independent 1-D integration of the waterline plane versus 2-D clipping."""
    barge = _barge()
    upper = min(0.995 * barge.deck_edge_angle_deg, 0.995 * barge.bottom_exposure_angle_deg)
    assert upper > 1.0
    for index in range(1, 20):
        heel = upper * index / 19.0
        section = barge.immersed_section(heel)
        tangent = math.tan(math.radians(heel))
        expected_y = barge.bm * tangent
        expected_z = barge.kb + 0.5 * barge.bm * tangent**2
        assert abs(section.centroid_y - expected_y) <= 1e-9 * max(1.0, abs(expected_y))
        assert abs(section.centroid_z - expected_z) <= 1e-9 * expected_z


def test_exact_arm_reproduces_the_textbook_wall_sided_formula():
    """Headline anchor: geometry versus ``sin(phi)*(GM + BM*tan^2/2)``."""
    for barge in (_barge(), _barge(beam=30.0, draft=3.0, kg=1.2), _barge(draft=7.5, kg=6.9)):
        upper = 0.995 * barge.wall_sided_envelope_deg
        for index in range(1, 40):
            heel = upper * index / 39.0
            exact = barge.gz(heel)
            closed = barge.wall_sided_gz(heel)
            assert abs(exact - closed) <= 1e-9 * max(1.0, abs(closed))


def test_small_angle_slope_and_merged_upright_hydrostatics_agree():
    """The exact arm degenerates to ``GM*sin(phi)`` and the merged GM matches."""
    barge = _barge()
    merged = analyze_box_barge(_hydrostatics_case(barge, 0.0))
    assert merged.gm == pytest.approx(barge.gm, rel=1e-13)
    assert merged.bm == pytest.approx(barge.bm, rel=1e-13)
    assert merged.kb == pytest.approx(barge.kb, rel=1e-13)
    small = 1e-4
    slope = barge.gz(small) / math.sin(math.radians(small))
    assert slope == pytest.approx(barge.gm, rel=1e-9)


def test_merged_linear_arm_error_is_bounded_inside_the_declared_envelope():
    """Audit of ``docs/marine-hydrostatics.md``: "<3% over the declared range".

    True for the shipped proportions, false in general: the relative error of
    the linear arm is ``r/(1+r)`` with ``r = BM*tan^2(phi)/(2*GM)``, so it
    reaches 3% at ``GM = BM*tan^2(phi)/2*(1/0.03 - 1)``.
    """
    for heel in (2.0, 5.0, 8.0, 9.9):
        assert _barge().linear_arm_relative_error(heel) < 0.03
        merged = analyze_box_barge(_hydrostatics_case(_barge(), heel))
        exact = _barge().gz(heel)
        assert abs(merged.righting_arm - exact) / exact < 0.03

    tender = _barge(kg=6.0)  # GM = 0.5556 m against BM = 5.5556 m
    assert tender.gm < tender.bm
    assert tender.linear_arm_relative_error(9.9) > 0.03
    threshold = 0.5 * tender.bm * math.tan(math.radians(9.9)) ** 2 * (1.0 / 0.03 - 1.0)
    assert tender.gm < threshold


def test_dynamic_stability_matches_the_closed_form_integral():
    """``integral GZ dphi`` by Simpson versus ``GM(1-cos) + BM(sec+cos-2)/2``."""
    barge = _barge()
    for heel in (1.0, 4.0, 8.0, 0.99 * barge.wall_sided_envelope_deg):
        numerical = barge.dynamic_stability(heel)
        closed = barge.wall_sided_dynamic_stability(heel)
        assert abs(numerical - closed) <= 1e-8 * max(1.0, abs(closed))
    coarse = barge.dynamic_stability(8.0, panels=64)
    fine = barge.dynamic_stability(8.0, panels=1024)
    assert abs(coarse - fine) <= 1e-5 * abs(fine)


def test_zero_arm_angle_closed_form_matches_the_numerical_root():
    """An unstable section's ``GZ = 0`` root equals ``atan(sqrt(-2GM/BM))``."""
    reference = _barge(depth=12.0, kg=0.0)
    unstable = _barge(depth=12.0, kg=reference.km + 0.1)
    assert unstable.gm < 0.0
    closed = unstable.zero_arm_angle_deg()
    assert closed is not None and 0.0 < closed < unstable.wall_sided_envelope_deg
    assert unstable.gz(closed) == pytest.approx(0.0, abs=1e-12)
    low, high = 1e-9, 0.999 * unstable.wall_sided_envelope_deg
    for _ in range(200):
        middle = 0.5 * (low + high)
        if unstable.gz(middle) < 0.0:
            low = middle
        else:
            high = middle
    assert low == pytest.approx(closed, rel=1e-9)


def test_deck_edge_angle_is_the_validity_limit_of_the_closed_form():
    """At ``phi_deck`` the free surface passes through the starboard deck edge."""
    barge = _barge()
    phi_deck = barge.deck_edge_angle_deg
    assert phi_deck == pytest.approx(math.degrees(math.atan(2 * (DEPTH - DRAFT) / BEAM)), rel=1e-13)
    section = barge.immersed_section(phi_deck)
    level = DEPTH * math.cos(math.radians(phi_deck)) - 0.5 * BEAM * math.sin(
        math.radians(phi_deck))
    assert abs(level - section.free_surface_height) <= 1e-12
    assert barge.immersed_section(0.98 * phi_deck).deck_immersed is False
    assert barge.immersed_section(1.02 * phi_deck).deck_immersed is True
    beyond = 1.05 * phi_deck
    deviation = abs(barge.gz(beyond) - barge.wall_sided_gz(beyond)) / abs(barge.gz(beyond))
    assert deviation > 1e-5


def test_bottom_exposure_also_closes_the_closed_form_envelope():
    """A deep narrow section surfaces its rising bilge before the deck edge."""
    narrow = _barge(beam=6.0, depth=15.0, draft=5.0, kg=2.0)
    assert narrow.bottom_exposure_angle_deg < narrow.deck_edge_angle_deg
    assert narrow.wall_sided_envelope_deg == narrow.bottom_exposure_angle_deg
    inside = 0.99 * narrow.bottom_exposure_angle_deg
    assert abs(narrow.gz(inside) - narrow.wall_sided_gz(inside)) <= 1e-9 * abs(narrow.gz(inside))
    outside = 1.10 * narrow.bottom_exposure_angle_deg
    assert narrow.immersed_section(outside).deck_immersed is False
    divergence = abs(narrow.gz(outside) - narrow.wall_sided_gz(outside)) / abs(narrow.gz(outside))
    assert divergence > 0.01
    # The geometric arm stays finite and keeps its sign; only the closed form
    # is outside its assumption.
    assert narrow.gz(outside) > 0.0
    assert narrow.wall_sided_gz(outside) > 0.0


def test_negative_controls_are_detected():
    """Symmetry, the ``KG`` lever identity, and rejection of the linear arm."""
    barge = _barge()
    for heel in (1.0, 6.0, 11.0):
        assert barge.gz(-heel) == pytest.approx(-barge.gz(heel), rel=1e-12, abs=1e-14)
    raised = _barge(kg=KG + 1.0)
    for heel in (3.0, 9.0):
        shift = raised.gz(heel) - barge.gz(heel)
        assert shift == pytest.approx(-math.sin(math.radians(heel)), rel=1e-10)
    tender = _barge(kg=6.0)
    assert tender.linear_arm_relative_error(tender.deck_edge_angle_deg) > 0.03


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"draft": 0.0}, "draft must be finite and positive"),
        ({"draft": 8.0}, "between 0 and the depth"),
        ({"draft": 9.0}, "between 0 and the depth"),
        ({"beam": -1.0}, "beam must be finite and positive"),
        ({"kg": float("nan")}, "kg must be finite"),
        ({"density": 0.0}, "density must be finite and positive"),
    ],
)
def test_invalid_sections_fail_closed(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _barge(**kwargs)


def test_invalid_requests_fail_closed():
    barge = _barge()
    with pytest.raises(ValueError, match="inside"):
        barge.gz(90.0)
    with pytest.raises(ValueError, match="panels must be even"):
        barge.dynamic_stability(5.0, panels=101)
    with pytest.raises(ValueError, match="samples must be at least 2"):
        barge.gz_curve(samples=1)
    with pytest.raises(ValueError, match="non-negative"):
        barge.gz_curve(heel_max_deg=-1.0)


def test_evidence_file_matches_recomputation():
    """The stored audit evidence must still equal a fresh recomputation."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "benchmarks" / "marine_large_angle_stability_evidence.json"
    if not path.exists():  # pragma: no cover - evidence ships with the repo
        pytest.fail("audit evidence file is missing; run benchmarks/marine_large_angle_stability_audit.py")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    anchors = {row["id"]: row for row in evidence["anchors"]}
    assert anchors["exact_gz_vs_wall_sided"]["relative_error"] <= 1e-9
    assert anchors["dynamic_stability_vs_closed_form"]["relative_error"] <= 1e-8
    assert anchors["flotation_offset_vs_closed_form"]["relative_error"] <= 1e-12
    assert anchors["centroid_vs_trapezoid_integrals"]["relative_error"] <= 1e-9
    assert anchors["merged_hydrostatics_gm"]["relative_error"] <= 1e-12
    assert anchors["exact_gz_vs_wall_sided"]["verdict"] == "PASS"
    assert evidence["audit"]["gate"] == 0.03
    for row in evidence["anchors"]:
        assert row["relative_error"] <= row["gate"], row["id"]


# ============================================================================
# Branch agent/m3495 (PR #87): independent implementation gates.
# ============================================================================
# Verification anchors for the wall-sided large-angle stability module.
#
# The evidence in ``benchmarks/marine_large_angle_stability_evidence.json`` and
# the tables in ``docs/assets/marine_large_angle_stability/tables.md`` are
# regenerated by ``benchmarks/marine_large_angle_stability.py``; these tests
# re-derive the same quantities from the library so evidence and code cannot
# drift apart silently.
# """


ROOT = Path(__file__).parents[1]
EVIDENCE = ROOT / "benchmarks" / "marine_wall_sided_stability_evidence.json"
TABLES = ROOT / "docs" / "assets" / "marine_large_angle_stability" / "tables.md"
EXAMPLES = ROOT / "examples" / "marine"

CASES = {
    "design_barge": (40.0, 12.0, 4.0, 2.0, 2.10),
    "shallow_draft_barge": (40.0, 12.0, 6.0, 2.0, 2.50),
    "small_gm_barge": (80.0, 20.0, 10.0, 4.0, 9.60),
    "negative_gm_barge": (80.0, 20.0, 10.0, 4.0, 10.83),
    "deck_limited_barge": (30.0, 10.0, 4.0, 2.5, 2.00),
}


def _ws_barge(name: str) -> WallSidedBarge:
    length, beam, depth, draft, kg = CASES[name]
    return WallSidedBarge(name, length, beam, depth, draft, kg)


def _heels(limit: float, count: int = 19) -> list[float]:
    return [limit * index / (count - 1) for index in range(count)]


@pytest.mark.parametrize("name", sorted(CASES))
def test_closed_form_matches_independent_polygon_oracle(name):
    """Equation (1) against the shoelace/polygon lever decomposition, eq. (2)."""

    barge = _ws_barge(name)
    limit = barge.closed_form_limit_deg
    scale = max(abs(barge.gz_numerical(heel)) for heel in _heels(limit))
    for heel in _heels(limit):
        closed = barge.gz_closed_form(heel)
        oracle = barge.gz_numerical(heel)
        assert abs(closed - oracle) <= 1e-12 * max(scale, 1.0)


@pytest.mark.parametrize("name", sorted(CASES))
def test_equilibrium_waterplane_conserves_displacement(name):
    barge = _ws_barge(name)
    for heel in _heels(barge.closed_form_limit_deg):
        state = barge.section_state(heel)
        assert state["area_error"] == pytest.approx(0.0, abs=1e-10 * barge.beam * barge.draft)


def test_lever_decomposition_reproduces_the_analytic_centroid():
    """``y_B = BM tan(phi)``, ``KB' = T/2 + BM tan^2(phi)/2`` from the oracle."""

    barge = _ws_barge("shallow_draft_barge")
    for heel in (3.0, 9.0, 15.0, barge.closed_form_limit_deg * 0.999):
        state = barge.section_state(heel)
        tan_phi = math.tan(math.radians(heel))
        assert state["y_b"] == pytest.approx(barge.bm * tan_phi, rel=1e-10)
        assert state["z_b"] == pytest.approx(
            barge.kb + 0.5 * barge.bm * tan_phi**2, rel=1e-10
        )


def test_port_starboard_symmetry_is_odd():
    barge = _ws_barge("design_barge")
    for heel in (2.0, 7.5, 14.0):
        assert barge.gz_closed_form(-heel) == pytest.approx(-barge.gz_closed_form(heel), abs=1e-14)
        assert barge.gz_numerical(-heel) == pytest.approx(-barge.gz_numerical(heel), abs=1e-12)


def test_combined_section_helper_matches_the_barge_wrapper():
    section = polygon_section(12.0, 6.0, 2.0, 12.0)
    barge = _ws_barge("shallow_draft_barge")
    state = barge.section_state(12.0)
    assert section["area"] == pytest.approx(state["area"], rel=1e-14)
    assert section["y_b"] == pytest.approx(state["y_b"], rel=1e-14)
    assert section["z_b"] == pytest.approx(state["z_b"], rel=1e-14)
    # Unit-length convenience wrapper agrees with the barge formulation.
    bm = 12.0**2 / (12.0 * 2.0)
    gm = 0.5 * 2.0 + bm - 2.5
    phi = math.radians(12.0)
    assert gz_closed_form(12.0, 2.0, 2.5, 12.0) == pytest.approx(
        math.sin(phi) * (gm + 0.5 * bm * math.tan(phi) ** 2), rel=1e-14
    )


def test_closed_form_fails_closed_beyond_keel_emergence():
    barge = _ws_barge("shallow_draft_barge")
    assert barge.closed_form_limit_reason == "keel_emergence"
    limit = barge.closed_form_limit_deg
    with pytest.raises(ValueError, match="keel_emergence"):
        barge.gz_closed_form(limit + 0.5)
    # The last admissible heel is still fine, so the guard is not over-tight.
    assert barge.gz_closed_form(limit * (1.0 - 1e-9)) == pytest.approx(
        barge.gz_numerical(limit * (1.0 - 1e-9)), rel=1e-12
    )


def test_closed_form_also_fails_closed_where_deck_immersion_binds():
    barge = _ws_barge("deck_limited_barge")
    assert barge.closed_form_limit_reason == "deck_edge_immersion"
    with pytest.raises(ValueError, match="deck_edge_immersion"):
        barge.gz_closed_form(barge.deck_immersion_limit_deg + 1.0)


def test_numerical_oracle_stays_valid_past_keel_emergence():
    """Past keel emergence the body sinks; the closed form must not be reused."""

    barge = _ws_barge("shallow_draft_barge")
    limit = barge.closed_form_limit_deg
    state = barge.section_state(limit + 10.0)
    assert state["keel_emerged"] is True
    assert state["sinkage"] < 0.0
    assert state["area_error"] == pytest.approx(0.0, abs=1e-10 * barge.beam * barge.draft)
    extrapolated = barge.gz_closed_form(limit + 10.0, strict=False)
    oracle = barge.gz_numerical(limit + 10.0)
    assert abs(extrapolated - oracle) / oracle > 0.05


def test_deck_immersion_fails_closed():
    barge = _ws_barge("shallow_draft_barge")
    # The upright-draft bound is where the closed form stops; the equilibrium
    # (constant displacement) bound is later because the hull rises once the
    # keel emerges.
    assert (
        barge.equilibrium_deck_immersion_deg() > barge.deck_immersion_limit_deg + 1.0
    )
    equilibrium = barge.equilibrium_deck_immersion_deg()
    with pytest.raises(ValueError, match="deck edge is immersed"):
        barge.section_state(equilibrium + 1.0)
    # The strict flag keeps the raw geometry reachable for diagnostics.
    assert barge.section_state(equilibrium + 1.0, strict=False)["deck_immersed"] is True


def test_angle_of_loll_identity_matches_oracle_zero_crossing():
    barge = _ws_barge("negative_gm_barge")
    assert barge.gm < 0.0
    loll = barge.angle_of_loll_deg
    assert loll == pytest.approx(
        math.degrees(math.atan(math.sqrt(-2.0 * barge.gm / barge.bm))), rel=1e-14
    )
    assert barge.loll_inside_closed_form_range is True
    # tan^2(phi_loll) = -2 GM / BM makes equation (1) vanish there.
    assert barge.gz_closed_form(loll) == pytest.approx(0.0, abs=1e-13)
    lo, hi = 0.0, barge.closed_form_limit_deg
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if barge.gz_numerical(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    assert 0.5 * (lo + hi) == pytest.approx(loll, abs=1e-9)
    assert barge.gz_numerical(loll) == pytest.approx(0.0, abs=1e-12)


def test_positive_gm_has_no_loll_and_recovers_the_upright_arm():
    barge = _ws_barge("design_barge")
    assert barge.angle_of_loll_deg is None
    assert barge.loll_inside_closed_form_range is True
    assert barge.gz_closed_form(0.0) == 0.0
    assert barge.gz_numerical(0.0) == pytest.approx(0.0, abs=1e-14)


def test_small_angle_model_is_within_three_percent_when_bm_over_gm_is_mild():
    for name in ("design_barge", "shallow_draft_barge", "deck_limited_barge"):
        barge = _ws_barge(name)
        for heel in (2.0, 5.0, 8.0, SMALL_ANGLE_LIMIT_DEG):
            exact = barge.gz_numerical(heel)
            error = abs(barge.gz_small_angle(heel) - exact) / abs(exact)
            assert error < 0.03


def test_small_angle_model_debt_for_a_wide_shallow_low_gm_barge():
    """The merged 10-deg cap does not by itself bound the GZ error at 3%."""

    barge = _ws_barge("small_gm_barge")
    exact = barge.gz_numerical(SMALL_ANGLE_LIMIT_DEG)
    error = abs(barge.gz_small_angle(SMALL_ANGLE_LIMIT_DEG) - exact) / abs(exact)
    assert 0.03 < error < 0.20
    # The error is the discarded second-order rise half*BM*tan^2(phi) over GM.
    predicted = 0.5 * (barge.bm / barge.gm) * math.tan(math.radians(SMALL_ANGLE_LIMIT_DEG)) ** 2
    assert error == pytest.approx(predicted / (1.0 + predicted), rel=1e-10)


def test_small_angle_model_fails_closed_beyond_its_declared_range():
    barge = _ws_barge("design_barge")
    with pytest.raises(ValueError, match="small-angle"):
        barge.gz_small_angle(10.5)


def test_curve_table_flags_the_end_of_the_closed_form_range():
    barge = _ws_barge("deck_limited_barge")
    limit = barge.closed_form_limit_deg
    rows = barge.curve([0.0, limit * 0.5, limit * (1.0 - 1e-9)])
    assert all(row["closed_form_inside_range"] for row in rows)
    assert rows[-1]["gz_closed_form"] == pytest.approx(rows[-1]["gz_numerical"], rel=1e-12)
    assert rows[0]["displacement_error"] == pytest.approx(0.0, abs=1e-12)


def test_matches_the_merged_small_angle_module_on_its_example_case():
    case = load_hydrostatics_case(EXAMPLES / "box_barge_specified_draft.json")
    legacy = analyze_box_barge(case)
    barge = WallSidedBarge(
        "example",
        legacy.length,
        legacy.beam,
        legacy.depth,
        legacy.draft,
        legacy.kg,
        legacy.fluid_density,
        legacy.gravity,
    )
    assert barge.gm == pytest.approx(legacy.gm, rel=1e-14)
    assert barge.bm == pytest.approx(legacy.bm, rel=1e-14)
    heel = legacy.heel_angle_deg
    assert barge.gz_small_angle(heel) == pytest.approx(legacy.righting_arm, rel=1e-14)
    exact = barge.gz_numerical(heel)
    assert abs(legacy.righting_arm - exact) / exact < 0.03
    # The exact arm is always the larger one: the small-angle model is a lower
    # bound for a positive-GM wall-sided section.
    assert exact > legacy.righting_arm


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(length=0.0, beam=12.0, depth=4.0, draft=2.0, kg=2.0), "length"),
        (dict(length=40.0, beam=-1.0, depth=4.0, draft=2.0, kg=2.0), "beam"),
        (dict(length=40.0, beam=12.0, depth=4.0, draft=4.0, kg=2.0), "draft"),
        (dict(length=40.0, beam=12.0, depth=4.0, draft=2.0, kg=float("nan")), "kg"),
        (dict(length=40.0, beam=12.0, depth=4.0, draft=2.0, kg=2.0, fluid_density=0.0), "fluid_density"),
    ],
)
def test_invalid_barges_fail_closed(kwargs, match):
    with pytest.raises(ValueError, match=match):
        WallSidedBarge("bad", **kwargs)


def test_empty_name_and_NaN_heel_fail_closed():
    with pytest.raises(ValueError, match="name"):
        WallSidedBarge("   ", 40.0, 12.0, 4.0, 2.0, 2.0)
    barge = _ws_barge("design_barge")
    with pytest.raises(ValueError, match="heel_deg"):
        barge.gz_closed_form(float("nan"))


def test_evidence_matches_a_fresh_recomputation():
    """Anti-drift: stored evidence must equal the current library output."""

    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["reference"]["small_angle_limit_deg"] == SMALL_ANGLE_LIMIT_DEG
    assert evidence["agreement"]["max_abs_closed_form_minus_oracle"] < 1e-12
    assert evidence["agreement"]["max_abs_closed_form_minus_oracle_relative"] < 1e-11
    assert evidence["agreement"]["max_displacement_error"] < 1e-10
    names = {case["barge"]["name"] for case in evidence["cases"]}
    assert names == set(CASES)
    for case in evidence["cases"]:
        barge = _ws_barge(case["barge"]["name"])
        assert case["barge"]["gm"] == pytest.approx(barge.gm, rel=1e-13)
        assert case["barge"]["closed_form_limit_deg"] == pytest.approx(
            barge.closed_form_limit_deg, rel=1e-13
        )
        for row in case["rows"]:
            heel = row["heel_angle_deg"]
            assert row["gz_numerical"] == pytest.approx(barge.gz_numerical(heel), abs=1e-12)
            if row["gz_closed_form"] is not None:
                assert row["gz_closed_form"] == pytest.approx(
                    barge.gz_closed_form(heel), abs=1e-12
                )
            else:
                assert abs(heel) > barge.closed_form_limit_deg * (1.0 - 1e-12)
    loll = evidence["loll_identity"]
    assert loll["absolute_difference_deg"] < 1e-9
    keel = evidence["keel_emergence_regime"]
    assert keel["max_extrapolation_relative_deviation"] > 0.05
    assert "Table A" in TABLES.read_text()


# ============================================================================
# Post-merge gate: the two independent implementations shipped side by side.
# ============================================================================
def test_branch_closed_form_agrees_with_merged_centroid_route():
    """``WallSidedBarge.gz_closed_form`` must match ``BoxBarge.gz`` in its range.

    ``BoxBarge`` reaches the arm by projecting the clipped-section centroid;
    ``WallSidedBarge`` evaluates ``sin(phi)*(GM + 0.5*BM*tan^2 phi)``.  The two
    share no numerical core, so agreement to round-off is a real consistency
    proof for the merged module rather than a tautology.
    """
    geometries = [
        # (length, beam, depth, draft, kg)
        (90.0, 20.0, 8.0, 6.0, 4.5),
        (80.0, 20.0, 10.0, 4.0, 9.6),
        (40.0, 12.0, 4.0, 2.0, 2.1),
    ]
    for length, beam, depth, draft, kg in geometries:
        oracle_side = BoxBarge(length=length, beam=beam, depth=depth, draft=draft, kg=kg)
        branch_side = WallSidedBarge("cross-check", length, beam, depth, draft, kg)
        limit = min(
            oracle_side.deck_edge_angle_deg, oracle_side.bottom_exposure_angle_deg
        )
        assert branch_side.closed_form_limit_deg == pytest.approx(limit, abs=1e-12)
        scale = 0.0
        for index in range(41):
            heel = limit * index / 40.0
            reference = oracle_side.gz(heel)
            scale = max(scale, abs(reference))
            assert branch_side.gz_closed_form(heel) == pytest.approx(reference, abs=1e-11)
            assert branch_side.gz_numerical(heel) == pytest.approx(reference, abs=1e-11)
        assert scale > 0.0


def test_merged_module_exposes_both_public_surfaces():
    """Neither side's public API may be dropped by the merge."""
    import tensorfem.marine_large_angle_stability as module

    for name in (
        # m3275 audit surface
        "BoxBarge",
        "ImmersedSection",
        # branch surface
        "WallSidedBarge",
        "polygon_section",
        "gz_closed_form",
        "SCHEMA",
        "SMALL_ANGLE_LIMIT_DEG",
    ):
        assert hasattr(module, name), name
        assert name in module.__all__, name


def _pr87_cross_check():
    """Reproduce the PR #87 cross-check in-process.

    ``artifacts/`` is filtered by the delivery gate, so a published branch can
    legitimately ship without the JSON.  The sweep is deterministic and cheap,
    so the test recomputes it and only uses the file, when present, as a
    witness that the shipped numbers agree.
    """
    import importlib.util

    path = ROOT / "benchmarks" / "pr87_merge_cross_check.py"
    spec = importlib.util.spec_from_file_location("pr87_merge_cross_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    computed = module.build_payload(verbose=False)
    shipped = ROOT / "artifacts" / "pr87_merge_cross_check.json"
    if shipped.is_file():
        evidence = json.loads(shipped.read_text(encoding="utf-8"))
        assert evidence["schema"] == computed["schema"]
        assert evidence["overall"] == pytest.approx(computed["overall"], rel=0, abs=0)
        assert evidence["verdict"] == computed["verdict"]
    return computed


def test_both_evidence_schemas_survive_the_merge():
    """Each implementation keeps its own evidence file and schema."""
    audit = json.loads(
        (ROOT / "benchmarks" / "marine_large_angle_stability_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    branch = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert "anchors" in audit and "audit" in audit
    assert branch["schema"] == "tensorfem.marine-large-angle-stability-evidence.v1"
    cross = _pr87_cross_check()
    assert cross["verdict"] == "PASS"
    assert cross["overall"]["max_rel_diff_closed_form_vs_centroid"] < 1e-9
