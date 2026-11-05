"""Verification of the large-angle `GZ` curve against closed-form anchors.

Anchors used here are independent of the implementation under test:

* the **wall-sided stability formula** ``GZ = sin(phi) (GM + BM tan^2(phi)/2)``
  (Biran, *Ship Hydrostatics and Stability*, 2003, Sec. 4.9; Lewis (ed.),
  *Principles of Naval Architecture* Vol. I, SNAME 1988, Sec. 2.6), which is an
  exact consequence of the prismatic section geometry inside its validity band;
* the **exact 90 deg geometry** ``xB = B/2 - B T/(2 D)``, ``zB = D/2``;
* the **exact dynamic-stability integral** ``int sin (GM + BM tan^2/2) dphi =
  GM (1 - cos) + BM (sec + cos - 2)/2``;
* the **metacentric small-angle slope** ``dGZ/dphi|_{0} = GM``;
* an **independent second formulation** of the same section (closed-form
  antiderivative of the clamped waterline height, integrated numerically on a
  fine grid) used to confirm the polygon-clipping path in every regime
  (wall-sided, bilge emergence, deck immersion).
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfem.marine_hydrostatics import analyze_box_barge
from tensorfem.marine_large_angle_gz import (
    DEFAULT_ANGLES_DEG,
    MAX_HEEL_DEG,
    analyze_large_angle_case,
    cross_curves,
    dynamic_stability,
    gz_curve,
    section_at_heel,
    wall_sided_dynamic_area,
    wall_sided_gz,
    wall_sided_validity_deg,
    write_gz_report,
)

EXAMPLES = Path(__file__).parents[1] / "examples" / "marine"

# Reference barge: bilge emergence (18.43 deg) and deck immersion (33.69 deg)
# are separated, so all three geometric regimes can be exercised.
BARGE = dict(beam=12.0, depth=6.0, draft=2.0, kg=2.1)
# T = D/2: both wall-sided limits coincide (single-boundary case).
BARGE_COINCIDENT = dict(beam=12.0, depth=4.0, draft=2.0, kg=2.1)
# Wide/shallow barge: deck immerses long before the keel edge dries out.
BARGE_WIDE = dict(beam=20.0, depth=3.0, draft=1.0, kg=1.4)


def _section(geom, heel_deg):
    return section_at_heel(geom["beam"], geom["depth"], geom["beam"] * geom["draft"], heel_deg)


def _gz(geom, heel_deg):
    section = _section(geom, heel_deg)
    phi = math.radians(heel_deg)
    return section.xb * math.cos(phi) + (section.zb - geom["kg"]) * math.sin(phi)


def _gm_bm(geom):
    kb = 0.5 * geom["draft"]
    bm = geom["beam"] ** 2 / (12.0 * geom["draft"])
    return kb + bm - geom["kg"], bm, kb


def _oracle_section(geom, heel_deg, samples=200001):
    """Independent formulation: antiderivative solve + fine-grid trapezoid."""
    beam, depth = geom["beam"], geom["depth"]
    area = beam * geom["draft"]
    tangent = math.tan(math.radians(heel_deg))
    half = 0.5 * beam

    def primitive(u):  # int_0^u clamp(v, 0, D) dv
        if u <= 0.0:
            return 0.0
        if u <= depth:
            return 0.5 * u * u
        return depth * u - 0.5 * depth * depth

    def immersed(h):
        if tangent == 0.0:
            return beam * min(max(h, 0.0), depth)
        return (primitive(h + half * tangent) - primitive(h - half * tangent)) / tangent

    low, high = -2.0 * beam, 2.0 * beam
    for _ in range(200):
        mid = 0.5 * (low + high)
        if immersed(mid) < area:
            low = mid
        else:
            high = mid
    h = 0.5 * (low + high)

    x = np.linspace(-half, half, samples + 1)
    w = np.clip(h + x * tangent, 0.0, depth)
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    a = float(trapz(w, x))
    mx = float(trapz(x * w, x))
    mz = float(trapz(0.5 * w * w, x))
    return h, a, mx / a, mz / a


# ----------------------------------------------------------------------
# Anchor 1: wall-sided closed form, inside its validity band
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "geom", [BARGE, BARGE_COINCIDENT, BARGE_WIDE], ids=["barge", "coincident", "wide"]
)
def test_gz_matches_wall_sided_formula_to_machine_precision(geom):
    gm, bm, _ = _gm_bm(geom)
    limit = min(wall_sided_validity_deg(geom["beam"], geom["depth"], geom["draft"]))
    angles = [limit * fraction for fraction in (0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)]
    assert max(angles) < limit
    for angle in angles:
        reference = wall_sided_gz(gm, bm, angle)
        assert _gz(geom, angle) == pytest.approx(reference, rel=1e-12, abs=1e-14)
        section = _section(geom, angle)
        assert section.bilge_emerged is False
        assert section.deck_immersed is False


def test_wall_sided_predictions_of_xb_and_zb():
    """``xB = BM tan(phi)`` and ``zB = KB + BM tan^2(phi)/2`` inside the band."""
    geom = BARGE
    gm, bm, kb = _gm_bm(geom)
    for angle in (1.0, 5.0, 12.0, 18.0):
        phi = math.radians(angle)
        section = _section(geom, angle)
        assert section.xb == pytest.approx(bm * math.tan(phi), rel=1e-12)
        assert section.zb == pytest.approx(kb + 0.5 * bm * math.tan(phi) ** 2, rel=1e-12)


# ----------------------------------------------------------------------
# Anchor 2: the validity boundary of the closed form is sharp and located
# ----------------------------------------------------------------------
def test_wall_sided_formula_stops_being_valid_exactly_at_the_geometric_limit():
    geom = BARGE
    gm, bm, _ = _gm_bm(geom)
    bilge_limit, deck_limit = wall_sided_validity_deg(geom["beam"], geom["depth"], geom["draft"])
    assert bilge_limit == pytest.approx(math.degrees(math.atan2(2 * geom["draft"], geom["beam"])))
    assert deck_limit == pytest.approx(
        math.degrees(math.atan2(2 * (geom["depth"] - geom["draft"]), geom["beam"]))
    )
    assert bilge_limit < deck_limit < 90.0
    # still exact at the boundary
    assert _gz(geom, bilge_limit) == pytest.approx(wall_sided_gz(gm, bm, bilge_limit), rel=1e-9)
    assert _section(geom, bilge_limit).bilge_emerged is False

    # and demonstrably wrong past it, with the error growing as the heel increases
    def deviation(angle):
        return abs(_gz(geom, angle) / wall_sided_gz(gm, bm, angle) - 1.0)

    assert _section(geom, bilge_limit + 0.5).bilge_emerged is True
    assert deviation(bilge_limit + 0.5) > 5e-4
    assert deviation(bilge_limit + 1.0) > 3.0 * deviation(bilge_limit + 0.5)
    assert deviation(bilge_limit + 10.0) > 1e-1
    # The second limit is a wall-sided estimate only: with D > 2T the keel edge
    # dries out first, the waterline height falls below T, and the real deck
    # immersion happens later than the closed-form angle.
    assert _section(geom, deck_limit + 0.5).deck_immersed is False
    true_deck = next(
        angle / 10.0 for angle in range(1, 901) if _section(geom, angle / 10.0).deck_immersed
    )
    assert deck_limit < true_deck < 90.0


def test_deck_immersion_is_the_binding_limit_for_a_deep_draft_barge():
    """With ``D < 2 T`` the deck immerses inside the wall-sided band."""
    geom = dict(beam=12.0, depth=3.0, draft=2.0, kg=1.5)
    gm, bm, _ = _gm_bm(geom)
    bilge_limit, deck_limit = wall_sided_validity_deg(geom["beam"], geom["depth"], geom["draft"])
    assert deck_limit < bilge_limit
    assert _section(geom, deck_limit - 0.2).deck_immersed is False
    assert _section(geom, deck_limit + 0.2).deck_immersed is True
    assert _section(geom, deck_limit + 0.2).bilge_emerged is False
    # the closed form is exact up to the binding limit and wrong past it
    assert _gz(geom, deck_limit) == pytest.approx(wall_sided_gz(gm, bm, deck_limit), rel=1e-9)
    assert abs(_gz(geom, deck_limit + 2.0) / wall_sided_gz(gm, bm, deck_limit + 2.0) - 1.0) > 5e-3


# ----------------------------------------------------------------------
# Anchor 3: exact 90 deg geometry
# ----------------------------------------------------------------------
@pytest.mark.parametrize("geom", [BARGE, BARGE_WIDE], ids=["barge", "wide"])
def test_ninety_degree_geometry_and_righting_arm_are_exact(geom):
    beam, depth, draft = geom["beam"], geom["depth"], geom["draft"]
    section = _section(geom, 90.0)
    assert section.immersed_area == pytest.approx(beam * draft, rel=1e-12)
    assert section.zb == pytest.approx(0.5 * depth, rel=1e-12)
    # wetted slice of width B*T/D clamped to the starboard side
    width = beam * draft / depth
    assert width < beam
    assert section.xb == pytest.approx(0.5 * beam - 0.5 * width, rel=1e-12)
    assert _gz(geom, 90.0) == pytest.approx(0.5 * depth - geom["kg"], rel=1e-12, abs=1e-12)


# ----------------------------------------------------------------------
# Anchor 4: metacentric slope at the origin
# ----------------------------------------------------------------------
def test_initial_slope_reproduces_gm_from_small_angle_hydrostatics():
    geom = BARGE
    gm, _, _ = _gm_bm(geom)
    upright = analyze_box_barge(
        json.loads((EXAMPLES / "box_barge_specified_draft.json").read_text())
    )
    assert upright.gm == pytest.approx(gm)
    for step in (1e-2, 1e-3):
        slope = _gz(geom, step) / math.sin(math.radians(step))
        # leading correction is +BM tan^2(phi)/2, i.e. ~2e-8 relative at 0.01 deg
        assert slope == pytest.approx(gm, rel=1e-7)
    # Richardson: central difference converges to GM at second order
    h = 1e-2
    forward = (_gz(geom, 2 * h) - _gz(geom, 0.0)) / math.radians(2 * h)
    backward = (_gz(geom, h) - _gz(geom, 0.0)) / math.radians(h)
    assert abs((2 * backward - forward) - gm) / gm < 1e-6


# ----------------------------------------------------------------------
# Anchor 5: constant displacement is enforced geometrically
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "geom", [BARGE, BARGE_COINCIDENT, BARGE_WIDE], ids=["barge", "coincident", "wide"]
)
def test_immersed_area_is_conserved_at_every_heel(geom):
    target = geom["beam"] * geom["draft"]
    for angle in DEFAULT_ANGLES_DEG:
        section = _section(geom, angle)
        assert section.immersed_area == pytest.approx(target, rel=1e-12)
    assert _section(geom, 0.0).waterline_height_centre == pytest.approx(geom["draft"], rel=1e-12)
    assert _section(geom, 0.0).xb == pytest.approx(0.0, abs=1e-13)


# ----------------------------------------------------------------------
# Anchor 6: independent second formulation of the section
# ----------------------------------------------------------------------
@pytest.mark.parametrize("angle", [2.0, 10.0, 18.0, 25.0, 40.0, 55.0, 70.0, 85.0])
def test_polygon_section_agrees_with_independent_antiderivative_formulation(angle):
    geom = BARGE
    section = _section(geom, angle)
    h, area, xb, zb = _oracle_section(geom, angle)
    assert section.waterline_height_centre == pytest.approx(h, rel=1e-10, abs=1e-10)
    assert section.immersed_area == pytest.approx(area, rel=1e-9)
    assert section.xb == pytest.approx(xb, rel=1e-8, abs=1e-10)
    assert section.zb == pytest.approx(zb, rel=1e-8)


# ----------------------------------------------------------------------
# Anchor 7: KN cross-curve decomposition
# ----------------------------------------------------------------------
def test_kn_arms_decompose_the_gz_curve_for_any_kg():
    geom = BARGE
    angles = (0.0, 10.0, 25.0, 45.0, 70.0, 90.0)
    reference = gz_curve(geom["beam"], geom["depth"], geom["draft"], 3.0, angles)
    table = cross_curves(geom["beam"], geom["depth"], geom["draft"], [2.1, 3.0, 4.5], angles)
    assert table["angles_deg"] == list(angles)
    assert table["kn"] == pytest.approx(list(reference.kn), rel=1e-14, abs=1e-14)
    for kg in (2.1, 3.0, 4.5):
        recomputed = [
            kn - kg * math.sin(math.radians(angle)) for kn, angle in zip(reference.kn, angles)
        ]
        assert table["gz_by_kg"][f"{kg:g}"] == pytest.approx(recomputed, rel=1e-12, abs=1e-14)
        direct = gz_curve(geom["beam"], geom["depth"], geom["draft"], kg, angles)
        assert list(direct.gz) == pytest.approx(recomputed, rel=1e-12, abs=1e-14)
    # KN is loading-independent: zero upright, metacentric slope KB + BM at the
    # origin, and exactly D/2 at 90 deg (buoyancy centroid at mid-depth).
    assert table["kn"][0] == pytest.approx(0.0, abs=1e-13)
    gm, bm, kb = _gm_bm(geom)
    slope_angle = 1e-3
    near = gz_curve(geom["beam"], geom["depth"], geom["draft"], 0.0, [0.0, slope_angle])
    assert near.kn[1] / math.sin(math.radians(slope_angle)) == pytest.approx(kb + bm, rel=1e-7)
    assert table["kn"][-1] == pytest.approx(geom["depth"] / 2.0, rel=1e-12)


# ----------------------------------------------------------------------
# Anchor 8: dynamic-stability integral
# ----------------------------------------------------------------------
def test_dynamic_stability_matches_the_exact_wall_sided_integral_and_converges():
    geom = BARGE
    gm, bm, _ = _gm_bm(geom)
    limit = min(wall_sided_validity_deg(geom["beam"], geom["depth"], geom["draft"]))
    exact = wall_sided_dynamic_area(gm, bm, limit)
    errors = []
    for steps in (40, 400):
        angles = [limit * index / steps for index in range(steps + 1)]
        curve = gz_curve(geom["beam"], geom["depth"], geom["draft"], geom["kg"], angles)
        errors.append(abs(dynamic_stability(angles, curve.gz) / exact - 1.0))
    assert errors[0] < 1e-4
    assert errors[1] < 1e-6
    # trapezoid is second order: 10x refinement should shrink the error ~100x
    assert errors[0] / errors[1] > 50.0


# ----------------------------------------------------------------------
# Anchor 9: odd symmetry in heel
# ----------------------------------------------------------------------
def test_gz_is_an_odd_function_of_heel():
    geom = BARGE
    angles = (-80.0, -45.0, -20.0, -5.0, 0.0, 5.0, 20.0, 45.0, 80.0)
    curve = gz_curve(geom["beam"], geom["depth"], geom["draft"], geom["kg"], angles)
    lookup = dict(zip(curve.angles_deg, curve.gz))
    for angle in (5.0, 20.0, 45.0, 80.0):
        assert lookup[-angle] == pytest.approx(-lookup[angle], rel=1e-12, abs=1e-12)
    assert lookup[0.0] == pytest.approx(0.0, abs=1e-13)


# ----------------------------------------------------------------------
# Negative controls: the solver must distinguish stability from instability
# ----------------------------------------------------------------------
def test_negative_control_high_kg_produces_negative_initial_stability():
    geom = dict(beam=12.0, depth=6.0, draft=2.0, kg=8.0)  # KG > KM = 7.0
    gm, _, _ = _gm_bm(geom)
    assert gm < 0.0
    assert _gz(geom, 1.0) < 0.0
    curve = gz_curve(geom["beam"], geom["depth"], geom["draft"], geom["kg"], [0.0, 1.0, 5.0, 15.0])
    assert curve.gm == pytest.approx(gm)
    assert all(arm < 0.0 for arm in curve.gz[1:])


def test_negative_control_wrong_beam_in_bm_is_detected():
    """Mutating B -> B/2 in BM (a classic slip) must fail the wall-sided anchor."""
    geom = BARGE
    gm, bm, _ = _gm_bm(geom)
    # B -> B/2 in BM is a factor-4 BM error: the tan^2 term is still small at
    # 5 deg but becomes the dominant discriminator once the heel is large.
    for angle, threshold in ((12.0, 1e-2), (18.0, 4e-2)):
        mutated = wall_sided_gz(gm, bm / 4.0, angle)
        assert abs(_gz(geom, angle) / mutated - 1.0) > threshold


# ----------------------------------------------------------------------
# Fail-closed input validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(beam=0.0), "beam must be finite and positive"),
        (dict(beam=float("nan")), "beam must be finite and positive"),
        (dict(draft=float("inf")), "draft must be finite and positive"),
        (dict(draft=6.0), "draft must be smaller than depth"),
        (dict(kg=float("nan")), "kg must be a finite number"),
        (dict(angles_deg=(0.0,)), "at least two heel angles"),
        (dict(angles_deg=(0.0, 95.0)), "must lie in \\[-90, 90\\]"),
    ],
)
def test_invalid_curve_inputs_fail_closed(kwargs, match):
    base = dict(beam=12.0, depth=6.0, draft=2.0, kg=2.1)
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        gz_curve(**base)


@pytest.mark.parametrize(
    "args,match",
    [
        ((12.0, 6.0, 72.1, 10.0), "smaller than the full section area"),
        ((12.0, 6.0, 24.0, 120.0), "must lie in"),
        ((0.0, 6.0, 24.0, 10.0), "beam must be finite and positive"),
    ],
)
def test_invalid_section_inputs_fail_closed(args, match):
    with pytest.raises(ValueError, match=match):
        section_at_heel(*args)


def test_cross_curve_input_validation():
    with pytest.raises(ValueError, match="kg_values must be a non-empty sequence"):
        cross_curves(12.0, 6.0, 2.0, [])
    assert MAX_HEEL_DEG == 90.0


# ----------------------------------------------------------------------
# Case-file integration
# ----------------------------------------------------------------------
def test_example_case_round_trip_and_json_report(tmp_path):
    case = json.loads((EXAMPLES / "large_angle_box_barge.json").read_text())
    curve = analyze_large_angle_case(case)
    upright = analyze_box_barge(case)
    assert curve.draft == upright.draft
    assert curve.kg == upright.kg
    assert curve.gm == pytest.approx(upright.gm)
    assert curve.displacement_volume == pytest.approx(40.0 * 12.0 * 2.0)
    assert curve.displacement_mass == pytest.approx(1000.0 * 40.0 * 12.0 * 2.0)
    path = write_gz_report(curve, tmp_path / "gz.json")
    payload = json.loads(path.read_text())
    assert payload["schema"] == "tensorfem.marine-large-angle-gz.v1"
    assert payload["angles_deg"] == list(curve.angles_deg)
    assert payload["gz"] == pytest.approx(list(curve.gz))
    assert payload["max_gz"] == pytest.approx(curve.max_gz)
    assert len(payload["bilge_emerged"]) == len(curve.angles_deg)


def test_case_file_with_non_zero_heel_is_rejected():
    case = json.loads((EXAMPLES / "large_angle_box_barge.json").read_text())
    case["condition"]["heel_angle_deg"] = 8.0
    with pytest.raises(ValueError, match="requires condition.heel_angle_deg = 0"):
        analyze_large_angle_case(case)


# ----------------------------------------------------------------------
# Stored evidence must not drift away from the code
# ----------------------------------------------------------------------
def test_benchmark_evidence_matches_the_current_implementation():
    evidence = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "large_angle_gz_reference.json").read_text()
    )
    geom = evidence["geometry"]
    curve = gz_curve(
        geom["beam"],
        geom["depth"],
        geom["draft"],
        geom["kg"],
        [row["heel_deg"] for row in evidence["rows"]],
    )
    assert curve.gz == pytest.approx(tuple(row["gz"] for row in evidence["rows"]), rel=0, abs=1e-12)
    assert curve.kn == pytest.approx(tuple(row["kn"] for row in evidence["rows"]), rel=0, abs=1e-12)
    assert curve.max_gz == pytest.approx(evidence["max_gz"], rel=1e-12)
    assert curve.angle_of_max_gz_deg == pytest.approx(evidence["angle_of_max_gz_deg"])
    assert curve.dynamic_stability_area == pytest.approx(
        evidence["dynamic_stability_area"], rel=1e-12
    )
    assert evidence["max_abs_residual_in_band"] < 1e-12
    assert evidence["gz_at_90_deg"] == pytest.approx(evidence["gz_at_90_deg_analytic"], rel=1e-12)
    assert evidence["true_deck_immersion_deg"] > evidence["wall_sided_limits_deg"]["deck_estimate"]
