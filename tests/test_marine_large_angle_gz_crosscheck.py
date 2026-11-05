"""Cross-implementation agreement of the large-angle `GZ` capability.

The package now carries two *independent* exact solutions of the same wall-sided
problem:

* ``tensorfem.marine_large_angle_gz`` -- polygon clipping against the waterline,
  waterline offset from the corner bracket, closed interval ``[-90, 90]``;
* ``tensorfem.marine_large_angle_stability.BoxBarge`` -- bisection on the
  immersed-area residual, open interval ``(-90, 90)``.

Neither calls the other, so comparing them is a real measurement: a coding error
in one engine (wrong clip predicate, transposed lever arm, sign slip) separates
the curves far above round-off, whereas a bug shared by both would not be
detected by this test alone -- that is the job of the closed-form anchors in
``tests/test_marine_large_angle_gz.py``.  The two families of tests are
complementary and neither is a substitute for the other.

Reference barge throughout: ``B = 12 m, D = 6 m, T = 2 m, KG = 2.1 m`` unless a
case says otherwise.
"""
import json
from pathlib import Path

import pytest

from tensorfem.marine_large_angle_gz import (
    MAX_HEEL_DEG,
    gz_curve,
    section_at_heel,
    wall_sided_validity_deg,
)
from tensorfem.marine_large_angle_stability import BoxBarge

REPO = Path(__file__).parents[1]
EVIDENCE = REPO / "benchmarks" / "large_angle_gz_crosscheck.json"

# Every ordering of the two wall-sided limits, plus a slender and a wide hull.
GEOMETRIES = (
    dict(name="reference", beam=12.0, depth=6.0, draft=2.0, kg=2.1),
    dict(name="deep_draft_deck_first", beam=12.0, depth=6.0, draft=3.0, kg=3.0),
    dict(name="coincident_limits", beam=12.0, depth=4.0, draft=2.0, kg=2.1),
    dict(name="wide_shallow", beam=20.0, depth=3.0, draft=1.0, kg=1.4),
    dict(name="slender", beam=6.0, depth=6.0, draft=1.0, kg=1.2),
)
LENGTH = 40.0
ABS_TOL = 1e-10  # metres; both engines bisect to machine precision


def _oracle(geom):
    return BoxBarge(
        length=LENGTH,
        beam=geom["beam"],
        depth=geom["depth"],
        draft=geom["draft"],
        kg=geom["kg"],
    )


def _curve(geom, angles):
    return gz_curve(geom["beam"], geom["depth"], geom["draft"], geom["kg"], angles)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=[g["name"] for g in GEOMETRIES])
def test_gz_agrees_with_the_merged_exact_oracle(geom):
    """Two independent engines must return the same righting arms."""
    angles = [0.5 * index for index in range(180)]  # 0 .. 89.5 deg
    curve = _curve(geom, angles)
    oracle = _oracle(geom)
    for angle, arm in zip(curve.angles_deg, curve.gz):
        assert arm == pytest.approx(oracle.gz(angle), rel=0, abs=ABS_TOL)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=[g["name"] for g in GEOMETRIES])
def test_gz_agrees_on_the_negative_heel_branch(geom):
    """Mirror branch: ``GZ(-phi) = -GZ(+phi)`` in both engines."""
    angles = [0.5 * index for index in range(1, 179)]
    curve = _curve(geom, angles)
    oracle = _oracle(geom)
    for angle, arm in zip(curve.angles_deg, curve.gz):
        assert -arm == pytest.approx(oracle.gz(-angle), rel=0, abs=ABS_TOL)


@pytest.mark.parametrize("geom", GEOMETRIES, ids=[g["name"] for g in GEOMETRIES])
def test_wall_sided_envelope_and_gm_match_the_merged_oracle(geom):
    """Same physics, different code: the validity band must coincide exactly."""
    beam, depth, draft = geom["beam"], geom["depth"], geom["draft"]
    bilge, deck = wall_sided_validity_deg(beam, depth, draft)
    oracle = _oracle(geom)
    assert min(bilge, deck) == pytest.approx(oracle.wall_sided_envelope_deg, rel=0, abs=1e-12)
    curve = _curve(geom, [0.0, 10.0])
    assert curve.gm == pytest.approx(oracle.gm, rel=0, abs=1e-12)
    # KN(0) = 0, so the initial GZ slope is -KG; both engines share it.
    assert curve.kn[0] == pytest.approx(0.0, rel=0, abs=1e-12)


def test_ninety_degree_endpoint_is_exact_where_the_merged_oracle_is_closed():
    """The closed endpoint is this module's added coverage, not a disagreement."""
    geom = GEOMETRIES[0]
    oracle = _oracle(geom)
    with pytest.raises(ValueError, match="inside \\(-90, 90\\)"):
        oracle.gz(MAX_HEEL_DEG)
    with pytest.raises(ValueError, match="inside \\(-90, 90\\)"):
        oracle.gz(-MAX_HEEL_DEG)

    curve = _curve(geom, [89.0, 90.0])
    assert curve.gz[-1] == pytest.approx(oracle.depth / 2.0 - oracle.kg, rel=0, abs=1e-12)
    section = section_at_heel(geom["beam"], geom["depth"], geom["beam"] * geom["draft"], 90.0)
    assert section.xb == pytest.approx(
        geom["beam"] / 2.0 - geom["beam"] * geom["draft"] / (2.0 * geom["depth"]), rel=0, abs=1e-12
    )
    assert section.zb == pytest.approx(geom["depth"] / 2.0, rel=0, abs=1e-12)


def test_agreement_test_is_sensitive_to_a_small_kg_error():
    """Negative control: the comparison above is not vacuous at this tolerance."""
    geom = GEOMETRIES[0]
    angle = 30.0
    honest = _curve(geom, [angle, angle + 1.0]).gz[0]
    oracle = _oracle(geom).gz(angle)
    perturbed = _curve(dict(geom, kg=geom["kg"] + 1e-3), [angle, angle + 1.0]).gz[0]
    assert honest == pytest.approx(oracle, rel=0, abs=ABS_TOL)
    # dGZ/dKG = -sin(phi) = -0.5, so a 1 mm KG slip moves the arm by 0.5 mm,
    # i.e. 5e6 times the tolerance.
    assert abs(perturbed - oracle) == pytest.approx(0.5e-3, rel=1e-6)
    assert abs(perturbed - oracle) > 1e3 * ABS_TOL


def test_crosscheck_evidence_matches_the_current_implementation():
    """Stored audit evidence must not drift away from the code."""
    evidence = json.loads(EVIDENCE.read_text())
    assert evidence["schema"] == "tensorfem.marine-large-angle-gz.crosscheck.v1"
    assert evidence["max_abs_gz_difference_m"] <= evidence["abs_tolerance_m"]
    assert evidence["max_envelope_difference_deg"] < 1e-12
    assert evidence["max_gm_difference_m"] < 1e-12
    assert evidence["merged_oracle_refuses_90_deg"] is True
    for row in evidence["geometries"]:
        primary = next(g for g in GEOMETRIES if g["name"] == row["name"])
        assert row["geometry"]["beam"] == primary["beam"]
        curve = _curve(primary, [row["angle_of_max_difference_deg"], row["angle_of_max_difference_deg"] + 0.5])
        oracle = _oracle(primary)
        replayed = abs(curve.gz[0] - oracle.gz(row["angle_of_max_difference_deg"]))
        assert replayed == pytest.approx(row["max_abs_gz_difference_m"], rel=1e-9, abs=1e-18)
    for row in evidence["ninety_degree_endpoint"]:
        assert row["merged_oracle_refuses_90_deg"] is True
        assert row["abs_error_m"] < 1e-12
        assert row["xb_at_90_deg_m"] == pytest.approx(row["xb_at_90_deg_analytic_m"], rel=0, abs=1e-12)
        assert row["zb_at_90_deg_m"] == pytest.approx(row["zb_at_90_deg_analytic_m"], rel=0, abs=1e-12)


def test_crosscheck_gate_detects_an_injected_disagreement():
    """The audit's own gate must fail closed on a poisoned record."""
    from scripts.audit_large_angle_gz_crosscheck import ABS_TOL_M, check

    evidence = json.loads(EVIDENCE.read_text())
    assert check(evidence) == []

    poisoned = json.loads(json.dumps(evidence))
    poisoned["max_abs_gz_difference_m"] = 10.0 * ABS_TOL_M
    assert check(poisoned) != []

    poisoned = json.loads(json.dumps(evidence))
    poisoned["max_abs_error_at_90_deg_m"] = 1e-3
    assert check(poisoned) != []

    poisoned = json.loads(json.dumps(evidence))
    poisoned["merged_oracle_refuses_90_deg"] = False
    assert check(poisoned) != []