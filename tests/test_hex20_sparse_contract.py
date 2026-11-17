"""Real dense/sparse equivalence for the newly inventoried CPU HEX20 exports."""

import torch
import pytest

from tensorfem import assemble_hex20_stiffness_sparse, solve_hex20_sparse
from tensorfem.hex20 import Hex20Model, solve_hex20, structured_hex20_mesh


@pytest.mark.parametrize("prescribed", [False, True])
def test_sparse_hex20_preserves_displacements_reactions_stress_and_energy(prescribed):
    nodes, elements = structured_hex20_mesh(2.0, 1.0, 0.8, 2, 1, 1)
    left = torch.where(nodes[:, 0] == 0)[0]
    fixed = (3 * left[:, None] + torch.arange(3)).reshape(-1)
    force = nodes.new_zeros(3 * len(nodes))
    tip = torch.where(nodes[:, 0] == 2)[0]
    force[3 * tip + 2] = -100.0 / len(tip)
    values = nodes.new_full((len(fixed),), 0.001) if prescribed else None
    model = Hex20Model(
        nodes, elements, nodes.new_tensor(70e9), nodes.new_tensor(0.27), force, fixed, values
    )
    dense = solve_hex20(model)
    sparse = solve_hex20_sparse(model)
    stiffness, _, _ = assemble_hex20_stiffness_sparse(model)
    assert stiffness.format == "csc"
    assert torch.allclose(
        torch.from_numpy(stiffness.toarray()), dense.stiffness, rtol=1e-12, atol=1e-5
    )
    for field in ("displacement", "reaction", "strain", "stress", "strain_energy"):
        expected, actual = getattr(dense, field), getattr(sparse, field)
        scale = max(float(expected.abs().max()), 1e-15)
        assert torch.allclose(actual, expected, rtol=1e-5, atol=scale * 1e-7), field
    assert sparse.stiffness is None  # sparse solve must not materialize a dense operator
    total_reaction = sparse.reaction.reshape(-1, 3).sum(0)
    assert torch.allclose(total_reaction, -force.reshape(-1, 3).sum(0), atol=1e-5)
