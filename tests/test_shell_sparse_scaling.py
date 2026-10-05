import pytest
import torch

from tensorfem.finite_rotation_layered_shell4 import solve_finite_rotation_arc_path
from tensorfem.shell_sparse_scaling import (
    _drilling_stabilized, benchmark_shell_sparse_scaling,
)
from tensorfem.sparse_shell4_plastic_qualification import yielding_shell4_case


def test_drilling_stabilization_preserves_sparse_layout_and_other_diagonal():
    matrix = torch.eye(4, dtype=torch.float64).to_sparse_coo()
    free = torch.tensor([0, 5, 6, 11])
    result = _drilling_stabilized(matrix, free, 1e-3).to_dense()
    assert result[0, 0] == 1 and result[2, 2] == 1
    assert result[1, 1] == pytest.approx(1.001)
    assert result[3, 3] == pytest.approx(1.001)


def test_tangent_cache_identity_mismatch_fails_closed(tmp_path):
    cache = tmp_path / "wrong.pt"
    torch.save({"mesh": 4, "free": torch.tensor([1]),
                "tangent": torch.eye(1).to_sparse_coo(),
                "assembly_seconds": 0.}, cache)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        benchmark_shell_sparse_scaling(2, tangent_cache=cache)


def test_ilu_gmres_complete_arc_path_matches_dense_when_scipy_available():
    pytest.importorskip("numpy"); pytest.importorskip("scipy")
    model, load, fixed = yielding_shell4_case()
    options = dict(steps=3, step_size=.01, maximum_step=.01, load_scale=.1,
                   augmented_scaling="normalized", tolerance=2e-7)
    dense = solve_finite_rotation_arc_path(model, load, fixed, **options)
    diagnostics = []
    iterative = solve_finite_rotation_arc_path(
        model, load, fixed, linear_solver="ilu_gmres",
        diagnostics=diagnostics, **options,
    )
    assert dense.converged and iterative.converged
    assert iterative.load_factor == pytest.approx(dense.load_factor, rel=1e-6)
    assert torch.allclose(iterative.displacement, dense.displacement,
                          rtol=1e-6, atol=1e-9)
    krylov = [row for row in diagnostics if row.get("krylov_iterations") is not None]
    assert krylov and max(row["krylov_relative_residual"] for row in krylov) < 1e-8
