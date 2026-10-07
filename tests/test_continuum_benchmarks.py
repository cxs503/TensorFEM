import torch

from tensorfem.continuum import ContinuumModel, solve_continuum
from tensorfem.continuum_benchmarks import mapped_q4_mesh, run_cook_membrane


DTYPE = torch.float64


def test_distorted_q4_patch_recovers_affine_field():
    # Deliberately non-parallelogram macro-domain; every element is distorted.
    corners = torch.tensor(((0., 0.), (3., .35), (2.55, 2.2), (-.3, 1.6)), dtype=DTYPE)
    nodes, elements = mapped_q4_mesh(corners, 4, 3)
    ux = .017*nodes[:, 0] - .012*nodes[:, 1] + .003
    uy = .009*nodes[:, 0] + .021*nodes[:, 1] - .007
    exact = torch.stack((ux, uy), 1).reshape(-1)
    # Dirichlet boundary drives the affine patch test; all interior DOFs are solved.
    ids = torch.arange(len(nodes)).reshape(4, 5)
    boundary_nodes = torch.unique(torch.cat((ids[0], ids[-1], ids[:, 0], ids[:, -1])))
    fixed = torch.stack((2*boundary_nodes, 2*boundary_nodes+1), 1).reshape(-1)
    model = ContinuumModel(nodes, elements, torch.tensor(17.), torch.tensor(.27),
                           torch.tensor(.4), torch.zeros(2*len(nodes), dtype=DTYPE),
                           fixed, prescribed_values=exact[fixed])
    result = solve_continuum(model)
    error = float(torch.linalg.vector_norm(result.displacement-exact) /
                  torch.linalg.vector_norm(exact))
    assert error < 1e-11, {"computed_norm": float(torch.linalg.vector_norm(result.displacement)),
                           "reference_norm": float(torch.linalg.vector_norm(exact)),
                           "relative_error": error}


def test_cook_membrane_is_below_three_percent():
    result = run_cook_membrane(16)
    assert result.relative_error < .03, result


def test_cook_membrane_monotonic_mesh_convergence():
    results = [run_cook_membrane(n) for n in (4, 8, 16)]
    errors = [r.relative_error for r in results]
    assert all(a > b for a, b in zip(errors, errors[1:])), results
    assert results[-1].relative_error < .03, results
