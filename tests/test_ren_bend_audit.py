"""Observation, control and split regressions; physical holdouts still fail."""

import json
from copy import deepcopy

import numpy as np
import pytest

from benchmarks.score_ren_bending import CONTROLS, DATA, beam_check, endpoint, read_table, score
from tensorfem.cohesive_bending import BilinearCohesiveLaw, build_three_point_bend
from tensorfem.ice_bend_transient import rupture


def coupon(*, integrated, fraction=0.25):
    return build_three_point_bend(
        span=0.18,
        depth=0.04,
        thickness=0.08,
        young_modulus=2e8,
        poisson_ratio=0.3,
        law=BilinearCohesiveLaw.from_material(2e8, 1e6, 1.5),
        nx=12,
        ny=8,
        integrate_load_patch=integrated,
        load_patch_fraction=fraction,
    )


def test_partial_edge_load_and_characteristic_length():
    model = coupon(integrated=True)
    x = model.nodes[model.load_dofs // 2, 0]
    half_width = 0.5 * 0.25 * model.depth
    dx = model.span / 12
    weights = -model.load_weights
    assert sum(weights) == pytest.approx(1)
    assert weights @ x == pytest.approx(model.span / 2)
    # Exact mean of a V-shaped top-edge virtual displacement over the patch.
    assert weights @ abs(x - model.span / 2) == pytest.approx(half_width / 2)
    assert weights[0] == pytest.approx(half_width / (4 * dx))
    assert len(set(x)) == 3
    for other in (coupon(integrated=False), coupon(integrated=True, fraction=0)):
        assert len(set(other.nodes[other.load_dofs // 2, 0])) == 1
    for ratio in [25, 100]:
        law = BilinearCohesiveLaw.from_material(2e8, 1e6, 1.5, penalty_ratio=ratio)
        assert law.cohesive_length == pytest.approx(0.0003)


def test_peak_frame_matches_every_step_observation_and_contact_law():
    best = None
    calls = 0

    def record(*frame):
        nonlocal best, calls
        calls += 1
        if best is None or frame[1] > best[1]:
            best = frame

    stiffness = 5e7
    result = rupture(coupon(integrated=True), punch_stiffness_N_m=stiffness, observe_step=record)
    assert calls == result["steps"] + 1
    assert best[:3] == tuple(max(result["history"], key=lambda f: f[1]))
    assert best[1] == result["peak_N"]
    assert best[2] == result["peak_deflection_m"]
    assert best[3] == result["peak_midspan_top_deflection_m"]
    assert abs(best[3] - best[2]) > 1e-7
    assert best[1] == pytest.approx(stiffness * (best[0] - best[2]))
    assert np.all(np.diff(np.array(result["history"])[:, 0]) > 0)


def test_exact_gate_and_analytical_shared_strength_bound():
    assert not endpoint(110, 100)["pass"]
    assert not endpoint(90, 100)["pass"]
    assert endpoint(109.99, 100)["pass"]
    with pytest.raises(ValueError):
        endpoint(float("nan"), 100)
    bound = beam_check(read_table(DATA))
    assert bound["minimum_worst_peak_relative_error"] == pytest.approx(0.12051179)
    assert bound["strength_gap_Pa"] > 0
    assert bound["rounding_scenario"]["common_limits_Pa"] == pytest.approx([864528, 885160], abs=1)


def test_score_ignores_producer_verdict_and_requires_both_holdouts():
    specimens = read_table(DATA)
    data = json.loads(CONTROLS.read_text())
    data["rows"] = [r for r in data["rows"] if r["case"] == "baseline"]
    for row in data["rows"]:
        row.update(joint_pass=True, peak_N=1.0)
    result = score(data, specimens)
    assert result["baseline_pass_count"] == 0
    assert result["rows"][1]["peak_N"] == pytest.approx(443, abs=0.1)
    invalid = deepcopy(data)
    invalid["rows"][0]["inputs"]["strength_Pa"] = 8e5
    with pytest.raises(ValueError, match="baseline protocol"):
        score(invalid, specimens)
    data["rows"].pop()
    with pytest.raises(ValueError, match="both holdouts"):
        score(data, specimens)
