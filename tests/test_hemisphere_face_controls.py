import math

import numpy as np
import pytest
import torch

from scripts.hemisphere_hex20_oracle_study import _compare
from tensorfem.hemisphere_benchmark_fields import surface_stress


def test_faces_have_opposite_bending_stress_and_zero_rigid_motion_stress():
    nodes = torch.tensor([[-1, -1, 10], [1, -1, 10], [1, 1, 10], [-1, 1, 10]], dtype=torch.float64)
    elements = torch.tensor([[0, 1, 2, 3]])
    zero = torch.zeros_like(nodes)
    rotation = zero.clone()
    rotation[:, 1] = nodes[:, 0]
    sigma = surface_stress(nodes, elements, zero, rotation)[0]
    assert sigma[0, 0] == pytest.approx(0.02 * 6.825e7 / (1 - 0.3**2))
    assert torch.equal(sigma[0], -sigma[1])
    spin = torch.tensor([0.25, -0.125, 0.375], dtype=nodes.dtype).expand_as(nodes)
    for radial in (True, False):
        rigid = surface_stress(
            nodes, elements, torch.linalg.cross(spin, nodes), spin, radial_directors=radial
        )
        assert rigid.abs().max() < 1.0e-7


def test_comparator_pairs_physical_faces_and_omits_stations_outside_oracle():
    phi, theta = np.deg2rad([55.0, 45.0])
    mid = [math.sin(phi) * math.cos(theta), math.sin(phi) * math.sin(theta), math.cos(phi)]
    shell = dict(
        nodes=[mid] * 4 + [[0, 0, 1]] * 4,
        elements=[[0, 1, 2, 3], [4, 5, 6, 7]],
        von_mises_top_bottom=[[2, 8], [2, 8]],
    )
    table = dict(
        grid_shape=(2, 2),
        polar_angle=torch.deg2rad(torch.tensor([30.0, 70.0, 30.0, 70.0])),
        azimuth=torch.deg2rad(torch.tensor([20.0, 20.0, 70.0, 70.0])),
    )
    outer, inner = [table | {"von_mises": torch.full((4,), v)} for v in [2.0, 8.0]]
    result = _compare(shell, outer, inner)
    assert len(result["stations"]["phi"]) == 1
    for face in ("outer", "inner"):
        assert result["face_regions"][face]["interior"]["max_relative_error"] == pytest.approx(0)


def test_default_recovery_preserves_archived_signed_face_components():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data = json.loads(
        (root / "docs/assets/benchmark-clouds/hemisphere-hole/results.json").read_bytes()
    )
    case = next(case for case in data["cases"] if case["mesh"] == 8)

    def array(key):
        return torch.tensor(case[key], dtype=torch.float64)

    actual = surface_stress(
        array("nodes"), torch.tensor(case["elements"]), array("displacement"), array("rotation")
    )
    np.testing.assert_allclose(actual.numpy(), case["stress_top_bottom"], rtol=1e-9, atol=1e-8)
