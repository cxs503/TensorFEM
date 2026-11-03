"""Quarter-ring curved-cantilever verification gates.

Reference: ``delta_tangent = (3 pi/4 - 2) P R^3 /(E I)``, ``delta_radial =
P R^3 /(2 E I)`` for a clamped quarter ring loaded along the free-end tangent.
All gates compare against that closed form; the coarse mesh is required to
*fail* so a regression that silently re-tunes the elements cannot pass.
"""
import math

import torch

from tensorfem.curved_beam_benchmark import (
    RADIAL_COEFFICIENT,
    TANGENT_COEFFICIENT,
    quarter_ring_reference,
    quarter_ring_shell,
)


def test_reference_coefficients_are_closed_form():
    reference = quarter_ring_reference(10.0, 0.2, 0.1, 2.1e5, 0.3, 1.0)
    bending = 1.0 * 10.0**3 / (2.1e5 * 0.2 * 0.1**3 / 12.0)
    assert abs(TANGENT_COEFFICIENT - (3.0 * math.pi / 4.0 - 2.0)) < 1e-15
    assert abs(RADIAL_COEFFICIENT - 0.5) < 1e-15
    assert abs(reference["delta_tangent"] / (TANGENT_COEFFICIENT * bending) - 1.0) < 1e-3
    assert abs(reference["delta_radial"] / (RADIAL_COEFFICIENT * bending) - 1.0) < 1e-12
    # axial and shear corrections are small but retained, not dropped
    assert 0.0 < reference["axial_term"] < 1e-4 * bending
    assert 0.0 < reference["shear_term"] < 1e-4 * bending


def test_load_and_moment_balance():
    for element in ("flat", "cylindrical"):
        result = quarter_ring_shell(16, 2, element=element)
        assert abs(result.reaction_tangent / result.load - 1.0) < 1e-6
        assert abs(result.reaction_moment / (result.load * result.radius) - 1.0) < 1e-6


def test_flat_shell_matches_closed_form():
    result = quarter_ring_shell(32, 2, element="flat")
    assert result.tangent_error < 0.01
    assert result.radial_error < 0.01


def test_cylindrical_shell_matches_closed_form():
    result = quarter_ring_shell(32, 2, element="cylindrical")
    assert result.tangent_error < 0.01
    assert result.radial_error < 0.01


def test_coarse_mesh_fails_closed():
    result = quarter_ring_shell(4, 2, element="flat")
    assert result.tangent_error > 0.03


def test_convergence_is_monotone_in_tangent_response():
    errors = [
        quarter_ring_shell(n, 2, element="flat").tangent_error
        for n in (8, 16, 32)
    ]
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 0.01


def test_thickness_invariance_rejects_membrane_locking():
    """The normalized response must stay thickness independent."""
    bending = None
    for thickness in (0.04, 0.1, 0.25):
        result = quarter_ring_shell(16, 2, thickness=thickness, element="flat")
        inertia = 0.2 * thickness**3 / 12.0
        scale = 1.0 * 10.0**3 / (2.1e5 * inertia)
        coefficient = result.tip_tangent / scale
        if bending is None:
            bending = coefficient
        assert abs(coefficient / TANGENT_COEFFICIENT - 1.0) < 0.02
        assert abs(coefficient - bending) < 1e-3


def test_width_mesh_does_not_change_the_response():
    coarse = quarter_ring_shell(16, 1, element="flat")
    fine = quarter_ring_shell(16, 4, element="flat")
    assert abs(fine.tip_tangent / coarse.tip_tangent - 1.0) < 0.02


def test_reciprocity_of_the_radial_response():
    """A radial end load must produce ``pi P R^3 /(4 E I)`` directly."""
    from tensorfem.curved_beam_benchmark import _flat_ring_mesh
    from tensorfem.shell4 import shell4_stiffness

    n_arc, n_width = 32, 2
    nodes, elements = _flat_ring_mesh(n_arc, n_width, 10.0, 0.2, torch.float64, torch.device("cpu"))
    ndof = 6 * nodes.shape[0]
    k = torch.zeros((ndof, ndof), dtype=torch.float64)
    for conn in elements:
        ke = shell4_stiffness(nodes[conn], 2.1e5, 0.3, 0.1)
        dofs = torch.stack(tuple(6 * conn + i for i in range(6)), 1).reshape(-1)
        k[dofs[:, None], dofs] += ke
    force = torch.zeros(ndof, dtype=torch.float64)
    tip = torch.arange(0, n_width + 1)
    for node in tip:
        force[6 * int(node)] -= 1.0 / (n_width + 1)  # radial (+x) end load
    root = n_arc * (n_width + 1) + torch.arange(0, n_width + 1)
    fixed = torch.tensor(sorted({6 * int(n) + i for n in root for i in range(6)}))
    mask = torch.ones(ndof, dtype=torch.bool)
    mask[fixed] = False
    free = torch.nonzero(mask).flatten()
    u = torch.zeros(ndof, dtype=torch.float64)
    u[free] = torch.linalg.solve(k[free[:, None], free], force[free])
    radial = u.reshape(-1, 6)[tip].mean(dim=0)[0]
    reference = quarter_ring_reference(10.0, 0.2, 0.1, 2.1e5, 0.3, 1.0)
    assert abs(float(radial.abs()) / reference["delta_radial_load"] - 1.0) < 0.01