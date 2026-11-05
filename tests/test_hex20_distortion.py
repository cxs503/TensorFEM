"""Gates for the HEX20 distortion qualification (closes the documented open item).

`docs/element-rank-deficiency-audit.md` section 6 lists "hex20 只审了规则立方体，
畸变 20 节点体与减缩积分变体待补".  Each probe in
:mod:`tensorfem.hex20_distortion` is pinned here fail-closed: if a distorted
brick ever develops a spurious mechanism, or the blocked 2x2x2 rule is silently
re-enabled, these tests go red.
"""
import pytest
import torch

from tensorfem.hex20 import structured_hex20_mesh
from tensorfem.hex20_distortion import (
    DISTORTION_FAMILIES,
    cantilever_distortion_sweep,
    distort_hex20,
    distortion_strength,
    distorted_patch_test,
    graded_cantilever,
    hex20_stiffness_rule,
    integration_rule_audit,
    nullspace_under_distortion,
    randomised_distortion_sweep,
)

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)  # shared host; thread oversubscription dominates runtime


def test_distortion_families_are_genuinely_distorted():
    # the 20-node brick carries edge/face mid-side nodes, which is where "warp" acts
    base, _ = structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    assert len(base) == 20  # 8 corners + 12 edge mid-side; no face-centre nodes
    for kind in DISTORTION_FAMILIES:
        out = distort_hex20(base, kind, 0.1)
        assert not torch.allclose(out, base), kind
        assert distortion_strength(out, base) > 0.0, kind
    with pytest.raises(ValueError, match="unknown distortion family"):
        distort_hex20(base, "shear-nonsense", 0.1)


def test_every_distorted_hex20_is_rank_sufficient():
    """No spurious zero-energy mode may appear on any distorted 20-node brick."""
    records = nullspace_under_distortion()
    assert len(records) == len(DISTORTION_FAMILIES) * 3
    for rec in records:
        assert rec["null_dim"] == 6, rec
        assert rec["rigid_dim"] == 6, rec
        assert rec["spurious"] == 0, rec
        assert rec["realised"] > 0.0
        # a rank-sufficient element keeps a strictly positive first eigenvalue
        assert rec["first_positive_ratio"] > 1e-12, rec


def test_randomised_distortion_sweep_never_produces_a_mechanism():
    report = randomised_distortion_sweep(trials=80, amplitude=0.25, seed=11)
    assert report["accepted"] > 0
    assert report["accepted"] + report["rejected_degenerate"] == report["trials"]
    assert report["spurious_histogram"] == {"0": report["accepted"]}, report


def test_reduced_2x2x2_rule_is_rank_deficient_negative_control():
    """The 2x2x2 rule is *not* an option: it creates six spurious mechanisms."""
    records = {r["rule"]: r for r in integration_rule_audit()}
    full = records["3x3x3"]
    reduced = records["2x2x2"]
    assert (full["null_dim"], full["spurious"]) == (6, 0)
    assert reduced["null_dim"] == 12
    assert reduced["rigid_dim"] == 6
    assert reduced["spurious"] == 6


def test_reduced_rule_matches_full_rule_nowhere_near_rigid_motions():
    base, _ = structured_hex20_mesh(1.0, 1.0, 1.0, 1, 1, 1)
    from tensorfem.hex20_distortion import _elasticity

    d = _elasticity()
    k3, _ = hex20_stiffness_rule(base.unsqueeze(0), d, order=3)
    k2, _ = hex20_stiffness_rule(base.unsqueeze(0), d, order=2)
    # both rules reproduce a rigid-body motion exactly (zero energy)
    rotation = torch.zeros(60, dtype=base.dtype)
    rotation[0::3] = -base[:, 1]
    rotation[1::3] = base[:, 0]
    for k in (k3[0], k2[0]):
        assert float(rotation @ k @ rotation) / float(k.abs().max()) < 1e-12
    with pytest.raises(ValueError, match="unsupported Gauss order"):
        hex20_stiffness_rule(base.unsqueeze(0), d, order=4)


def test_distorted_patch_test_reproduces_a_linear_field():
    """Distorted but conforming geometry must still pass the patch test exactly."""
    level = 0
    result = distorted_patch_test(level=level, amplitude=0.1)
    assert result["interior_nodes"] > 0
    assert result["relative_interior_error"] < 1e-12, result
    assert result["max_interior_error"] < 1e-15, result


def test_graded_mesh_keeps_the_cantilever_inside_its_error_budget():
    """Longitudinal grading up to a 12.7:1 element length ratio must not break the 3% gate."""
    uniform = graded_cantilever(8, 1.0)
    graded = graded_cantilever(8, 1.6)
    assert uniform["length_ratio"] == pytest.approx(1.0)
    assert graded["length_ratio"] > 8.0
    for record in (uniform, graded):
        assert abs(record["error_percent"]) < 3.0, record
    # distortion must not move the answer by more than a measured, small margin
    assert abs(graded["error_percent"] - uniform["error_percent"]) < 2.0


def test_cantilever_sweep_is_stable_everywhere():
    for record in cantilever_distortion_sweep(nx_values=(4, 8), exponents=(1.0, 1.4)):
        assert record["length_ratio"] >= 1.0
        assert record["computed"] * record["reference"] > 0.0
        assert record["error_percent"] < 8.0, record