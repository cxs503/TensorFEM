import pytest
import torch
import hashlib

from tensorfem.finite_rotation_layered_shell4 import solve_finite_rotation_arc_path
from tensorfem.shell_sparse_scaling import (
    ShellSparseScalingPoint, _drilling_stabilized,
    benchmark_shell_sparse_scaling, sparse_scaling_qualification,
    verify_legacy_sparse_tangent_cache, verify_sparse_scaling_report,
)
from tensorfem.marine_panel_ultimate_fe import build_panel_case
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
    with pytest.raises(RuntimeError, match="integrity sidecar"):
        benchmark_shell_sparse_scaling(2, tangent_cache=cache)


def test_hashed_cache_tampering_fails_before_loading(tmp_path):
    cache = tmp_path / "tampered.pt"
    cache.write_bytes(b"not a tensor cache")
    cache.with_suffix(".pt.sha256").write_text(
        hashlib.sha256(cache.read_bytes()).hexdigest()+"\n")
    cache.write_bytes(cache.read_bytes()+b"changed")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        benchmark_shell_sparse_scaling(2, tangent_cache=cache)


def test_legacy_cache_requires_reassembly_not_hash_endorsement(tmp_path):
    case = build_panel_case(2)
    mask = torch.ones(case.model.n_dofs, dtype=torch.bool)
    mask[case.fixed_dofs] = False
    free = torch.nonzero(mask).flatten()
    cache = tmp_path / "legacy.pt"
    wrong = torch.eye(len(free), dtype=case.model.nodes.dtype).to_sparse_coo()
    torch.save({"mesh": 2, "free": free, "tangent": wrong,
                "assembly_seconds": 0.}, cache)
    with pytest.raises(RuntimeError, match="failed reassembly verification"):
        verify_legacy_sparse_tangent_cache(2, cache)
    assert not cache.with_suffix(".pt.sha256").exists()


def _scaling_point(*, dofs=10882, error=1e-6, residual=1e-10,
                   ilu_factor_bytes=80_000_000):
    return ShellSparseScalingPoint(
        42, dofs, 566460, 120., 1.9, .28, 7, residual, error,
        6_800_000, ilu_factor_bytes, 1.4, .03, 1e-15,
        6_800_000, 99_000_000, 947_000_000,
    )


def test_release_qualification_is_fail_closed_and_hash_stable():
    passed = sparse_scaling_qualification([_scaling_point()])
    assert passed["passed"] and passed["performance_claim"] == "memory_only"
    assert passed["qualified_metrics"]["active_dofs"] == 10882
    assert len(passed["evidence_sha256"]) == 64
    assert passed == sparse_scaling_qualification([_scaling_point()])
    assert not sparse_scaling_qualification([_scaling_point(dofs=9999)])["passed"]
    assert not sparse_scaling_qualification([_scaling_point(error=.02)])["passed"]
    assert not sparse_scaling_qualification([_scaling_point(residual=2e-8)])["passed"]
    assert not sparse_scaling_qualification([
        _scaling_point(ilu_factor_bytes=120_000_000)])["passed"]


def test_report_verifier_rejects_hash_and_gate_tampering(tmp_path):
    report = sparse_scaling_qualification([_scaling_point()])
    assert verify_sparse_scaling_report(report)["passed"]
    target = tmp_path / "report.json"
    import json
    target.write_text(json.dumps(report))
    assert verify_sparse_scaling_report(target)["evidence_sha256"] == report["evidence_sha256"]
    tampered = dict(report)
    tampered["performance_claim"] = "time_and_memory"
    with pytest.raises(RuntimeError, match="hash mismatch"):
        verify_sparse_scaling_report(tampered)
    inconsistent = dict(report)
    inconsistent["passed"] = False
    # Rehashing cannot bypass semantic consistency checks.
    import tensorfem.shell_sparse_scaling as scaling
    unsigned = dict(inconsistent); unsigned.pop("evidence_sha256")
    inconsistent["evidence_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(RuntimeError, match="inconsistent"):
        scaling.verify_sparse_scaling_report(inconsistent)


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
