import json
import math
from pathlib import Path

import pytest

from tensorfem.marine_hydrostatics import analyze_box_barge, load_hydrostatics_case, write_hydrostatics_report


EXAMPLES = Path(__file__).parents[1] / "examples" / "marine"


def test_specified_draft_matches_box_barge_analytic_oracle_below_three_percent(tmp_path):
    case = load_hydrostatics_case(EXAMPLES / "box_barge_specified_draft.json")
    result = analyze_box_barge(case)
    assert result.displacement_volume == pytest.approx(960.0)
    assert result.displacement_mass == pytest.approx(960000.0)
    assert result.center_of_buoyancy == pytest.approx((20.0, 0.0, 1.0))
    assert result.waterplane_inertia_roll == pytest.approx(40.0 * 12.0**3 / 12.0)
    assert result.bm == pytest.approx(12.0**2 / (12.0 * 2.0))
    assert result.gm == pytest.approx(1.0 + 6.0 - 2.1)
    # Independent small-angle oracle uses phi instead of sin(phi).
    oracle = 960000.0 * 9.81 * result.gm * math.radians(8.0)
    assert abs(result.restoring_moment / oracle - 1.0) < 0.03
    path = write_hydrostatics_report(result, tmp_path / "report.json")
    assert json.loads(path.read_text())["gm"] == pytest.approx(4.9)


def test_load_case_derives_equilibrium_draft_and_combined_kg():
    result = analyze_box_barge(load_hydrostatics_case(EXAMPLES / "box_barge_load_case.json"))
    total_mass = 2700000.0
    expected_draft = total_mass / (1025.0 * 60.0 * 18.0)
    expected_kg = (1800000.0 * 2.8 + 750000.0 * 4.2 + 150000.0 * 1.2) / total_mass
    assert result.draft == pytest.approx(expected_draft)
    assert result.displacement_mass == pytest.approx(total_mass)
    assert result.kg == pytest.approx(expected_kg)
    assert result.gm > 0.0 and result.restoring_moment > 0.0


def _base():
    return json.loads((EXAMPLES / "box_barge_specified_draft.json").read_text())


@pytest.mark.parametrize("mutation,match", [
    (lambda x: x.update(units="knots-feet"), "SI"),
    (lambda x: x["geometry"].update(beam=0), "positive"),
    (lambda x: x["condition"].update(heel_angle_deg=11), "small-angle"),
    (lambda x: x["condition"].update(draft=4.0), "molded depth"),
    (lambda x: x.update(extra=True), "invalid case keys"),
])
def test_invalid_inputs_fail_closed(mutation, match):
    case = _base()
    mutation(case)
    with pytest.raises(ValueError, match=match):
        analyze_box_barge(case)


def test_overloaded_load_case_fails_closed():
    case = json.loads((EXAMPLES / "box_barge_load_case.json").read_text())
    case["condition"]["loads"][0]["mass"] = 10_000_000.0
    with pytest.raises(ValueError, match="molded depth"):
        analyze_box_barge(case)
