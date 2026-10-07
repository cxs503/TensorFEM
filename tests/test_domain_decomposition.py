import pytest
import torch

from tensorfem.domain_decomposition import (
    AdditiveSchwarz, build_subdomains, contiguous_owners, distributed_sum,
    graph_owners, solve_schwarz,
)

D = torch.float64


def chain_matrix(n: int) -> torch.Tensor:
    diag = torch.full((n,), 2.0, dtype=D); diag[0] = diag[-1] = 1.0
    # Ground both ends weakly enough to retain a nontrivial coupled benchmark.
    diag[0] += 1.0; diag[-1] += 1.0
    A = torch.diag(diag) + torch.diag(torch.full((n-1,), -1., dtype=D), 1)
    A += torch.diag(torch.full((n-1,), -1., dtype=D), -1)
    return A.to_sparse_coo().coalesce()


@pytest.mark.parametrize("partition", ["contiguous", "graph"])
@pytest.mark.parametrize("nparts", [1, 2, 3, 5])
def test_schwarz_matches_dense_and_is_partition_independent(partition, nparts):
    A = chain_matrix(31); b = torch.sin(torch.arange(31, dtype=D) + .3)
    result = solve_schwarz(A, b, nparts=nparts, overlap=1,
                           partition=partition, rtol=1e-11, maxiter=100)
    exact = torch.linalg.solve(A.to_dense(), b)
    error = torch.linalg.vector_norm(result.x-exact)/torch.linalg.vector_norm(exact)
    assert result.converged
    assert float(error) < 1e-9  # much stricter than the project 3% gate


def test_subdomain_maps_cover_and_reconstruct_continuous_interface():
    A = chain_matrix(12); owners = contiguous_owners(12, 3)
    domains = build_subdomains(A, owners, overlap=1)
    assert torch.equal(torch.sort(torch.cat([s.owned for s in domains])).values,
                       torch.arange(12))
    assert [s.interface.tolist() for s in domains] == [[3], [4, 7], [8]]
    u = solve_schwarz(A, torch.arange(12, dtype=D), nparts=3).x
    # Every duplicate halo reads exactly the same global interface unknown.
    for left, right in zip(domains[:-1], domains[1:]):
        shared = torch.tensor(sorted(set(left.local.tolist()) & set(right.local.tolist())))
        assert shared.numel() > 0
        left_values = u[left.local][left.global_to_local(12)[shared]]
        right_values = u[right.local][right.global_to_local(12)[shared]]
        assert torch.equal(left_values, right_values)


def test_additive_schwarz_positive_and_fail_closed_validation():
    A = chain_matrix(8); domains = build_subdomains(A, contiguous_owners(8, 2))
    z = AdditiveSchwarz(A, domains)(torch.arange(1, 9, dtype=D))
    assert torch.dot(z, torch.arange(1, 9, dtype=D)) > 0
    with pytest.raises(ValueError, match="labels"):
        build_subdomains(A, torch.tensor([0,0,0,0,2,2,2,2]))
    with pytest.raises(ValueError, match="overlap"):
        build_subdomains(A, contiguous_owners(8, 2), overlap=-1)
    with pytest.raises(ValueError):
        contiguous_owners(4, 5)
    with pytest.raises(RuntimeError, match="not initialized"):
        distributed_sum(torch.ones(1))


def test_disconnected_or_singular_local_problem_fails_closed():
    bad = torch.tensor([[1., -1.], [-1., 1.]], dtype=D).to_sparse_coo()
    domains = build_subdomains(bad, contiguous_owners(2, 2), overlap=1)
    with pytest.raises(ValueError, match="positive definite"):
        AdditiveSchwarz(bad, domains)
