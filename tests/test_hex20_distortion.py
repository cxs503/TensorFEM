"""Distorted / curved-geometry HEX20 qualification (rank + patch + cantilever A/B).

Closes the open item recorded in ``docs/element-rank-deficiency-audit.md`` §6:
the 20-node brick had been audited only on the regular unit cube.  Every
assertion here is an integer or round-off-level statement except the Timoshenko
comparison, which carries an explicit 3% gate.
"""
import torch
import pytest

from tensorfem.hex20 import cantilever_hex20_benchmark
from tensorfem.hex20_distortion import (
    affine_patch_error,
    brick_jacobian_extremes,
    curved_brick,
    distorted_brick,
    rank_probe,
    warped_cantilever,
    warped_hex20_mesh,
    consistent_end_shear,
)

D = torch.float64


def test_distorted_and_curved_bricks_keep_the_six_rigid_body_modes():
    """No hourglass mode may appear when the brick geometry is degraded."""
    cases = [
        distorted_brick(0.02),
        distorted_brick(0.10),
        distorted_brick(0.20),
        curved_brick(0.01),
        curved_brick(0.04),
        distorted_brick(0.10, curvature=0.04),
    ]
    for nodes in cases:
        lo, hi = brick_jacobian_extremes(nodes)
        assert lo > 0.0, "geometry must stay regular before the rank probe"
        assert hi > lo
        record = rank_probe(nodes)
        assert record.dofs == 60
        assert record.null_dim == 6, "a free brick has exactly six rigid-body modes"
        assert record.spurious == 0


def test_regular_cube_is_recovered_at_zero_distortion():
    nodes = distorted_brick(0.0)
    cube_corners = torch.tensor(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        dtype=D,
    )
    assert torch.allclose(nodes[:8], cube_corners, atol=0.0)
    mid = torch.stack([(cube_corners[a] + cube_corners[b]) / 2 for a, b in (
        (0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )])
    assert torch.allclose(nodes[8:], mid, atol=0.0)


def test_shared_audit_reports_the_distorted_hex20():
    """The library-wide rank audit must cover the degraded geometries."""
    from tensorfem.element_rank_audit import audit_library

    records = {r.element: r for r in audit_library()}
    for name in ("hex20 (distorted corners)", "hex20 (curved faces, 4% midside bulge)"):
        assert name in records
        assert records[name].spurious == 0
        assert records[name].null_dim == records[name].expected_rigid_dim == 6


def test_affine_patch_is_exact_on_a_warped_conforming_mesh():
    """A quadratic isoparametric mapping represents affine fields exactly."""
    for warp in (0.0, 0.01, 0.05):
        error = affine_patch_error(warp=warp)
        assert error < 1e-10, f"warp={warp} affine patch error {error:.3e}"


def test_warp_actually_distorts_the_mesh():
    nodes0, _ = warped_hex20_mesh(2.0, 1.0, 0.8, 2, 2, 2, warp=0.0)
    nodes1, _ = warped_hex20_mesh(2.0, 1.0, 0.8, 2, 2, 2, warp=0.05)
    moved = (nodes1 - nodes0).norm(dim=1) > 0
    assert moved.any(), "the warp field must move at least one interior node"
    boundary = (
        (nodes0[:, 0] <= 1e-12) | (nodes0[:, 0] >= 2.0 - 1e-12)
        | (nodes0[:, 1] <= 1e-12) | (nodes0[:, 1] >= 1.0 - 1e-12)
        | (nodes0[:, 2] <= 1e-12) | (nodes0[:, 2] >= 0.8 - 1e-12)
    )
    assert torch.allclose(nodes1[boundary], nodes0[boundary]), "boundary must be untouched"


def test_jacobian_gate_rejects_a_collapsed_distorted_brick():
    """Fail-closed behaviour must survive the distortion path."""
    nodes = distorted_brick(1.6)
    lo, _ = brick_jacobian_extremes(nodes)
    assert lo <= 0.0
    with pytest.raises(ValueError, match="Jacobian"):
        rank_probe(nodes)


def test_consistent_end_shear_reproduces_the_documented_benchmark():
    """Self-check of the load assembly against the merged HEX20 benchmark.

    At ``ny = nz = 1`` the shape-function-integrated end shear must reproduce the
    documented corner/midside consistent pattern, i.e. the same tip deflection.
    """
    legacy, _ = cantilever_hex20_benchmark(8)
    record = warped_cantilever(8, warp=0.0, element="hex20", ny=1, nz=1)
    assert record["applied_load"] == pytest.approx(1.0e6, rel=1e-12)
    assert record["computed"] == pytest.approx(float(legacy), rel=1e-10)


def test_end_shear_total_load_is_mesh_independent():
    from tensorfem.hex20 import structured_hex20_mesh
    from tensorfem.solid3d import structured_hex_mesh

    for builder, element, dims in (
        (structured_hex20_mesh, "hex20", (8, 1, 1)),
        (structured_hex20_mesh, "hex20", (8, 2, 2)),
        (structured_hex_mesh, "hex8", (8, 2, 2)),
    ):
        nodes, cells = builder(10.0, 1.0, 1.0, *dims)
        forces = consistent_end_shear(
            nodes, cells, element=element, end_axis=0, total_load=-1e6, area=1.0, load_axis=2
        )
        assert float(-forces.sum()) == pytest.approx(1.0e6, rel=1e-12)
        # the traction must act only on the loaded end plane
        loaded = torch.where(forces != 0)[0]
        assert torch.allclose(nodes[loaded // 3, 0], torch.full_like(nodes[loaded // 3, 0], 10.0))
        assert torch.all(loaded % 3 == 2)


def test_warped_cantilever_hex20_stays_inside_the_three_percent_gate():
    """Distortion must not destroy the qualified HEX20 bending result."""
    baseline = warped_cantilever(8, warp=0.0, element="hex20")
    assert baseline["error_percent"] < 3.0
    for warp in (0.01, 0.02, 0.04):
        record = warped_cantilever(8, warp=warp, element="hex20")
        assert record["error_percent"] < 3.0
        change = abs(record["computed"] - baseline["computed"]) / abs(baseline["computed"])
        assert change < 0.01, f"warp={warp} changed the tip deflection by {change:.3%}"


def test_warped_hex8_baseline_remains_locking_dominated():
    """Guards the A/B: the HEX8 control must stay far outside the gate."""
    record = warped_cantilever(8, warp=0.0, element="hex8")
    assert record["error_percent"] > 10.0