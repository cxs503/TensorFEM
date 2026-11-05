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

import math

import pytest

from tensorfem.marine_hydrostatics import SCHEMA as HYDROSTATICS_SCHEMA
from tensorfem.marine_hydrostatics import analyze_box_barge
from tensorfem.marine_large_angle_stability import BoxBarge

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