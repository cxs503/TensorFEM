#!/usr/bin/env python3
"""Benchmark the optional sparse direct backend on real panel tangents."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch

from tensorfem.finite_rotation_layered_shell4 import (
    assemble_finite_rotation_layered_shell4,
    assemble_finite_rotation_layered_shell4_sparse,
)
from tensorfem.layered_shell4_plasticity import LayeredShell4State
from tensorfem.marine_panel_ultimate_fe import build_panel_case
from tensorfem.sparse_direct import SparseLinearSolver, sparse_backend_statuses


def _bytes(tensor: torch.Tensor) -> int:
    if tensor.layout == torch.sparse_coo:
        tensor = tensor.coalesce()
        return (tensor.values().numel()*tensor.values().element_size()
                + tensor.indices().numel()*tensor.indices().element_size())
    return tensor.numel()*tensor.element_size()


def _timed_solve(action, repetitions: int):
    values = []
    result = None
    for _ in range(repetitions):
        started = perf_counter(); result = action(); values.append(perf_counter()-started)
    return result, min(values)


def benchmark(mesh: int, repetitions: int) -> dict:
    case = build_panel_case(mesh)
    state = LayeredShell4State.virgin(case.model)
    displacement = torch.zeros(case.model.n_dofs, dtype=case.model.nodes.dtype)
    mask = torch.ones(case.model.n_dofs, dtype=torch.bool); mask[case.fixed_dofs] = False
    free = torch.nonzero(mask).flatten()
    dense_response = assemble_finite_rotation_layered_shell4(case.model, displacement, state)
    sparse_response = assemble_finite_rotation_layered_shell4_sparse(
        case.model, displacement, state, active_dofs=free,
    )
    dense = dense_response.tangent[free[:, None], free]
    sparse = sparse_response.tangent.coalesce()
    load = case.reference_load[free]
    rhs = torch.stack((load, torch.linspace(-1., 1., len(free), dtype=dense.dtype)), dim=1)

    dense_answer, dense_seconds = _timed_solve(
        lambda: torch.linalg.solve(dense, rhs), repetitions,
    )
    solver = SparseLinearSolver("auto")
    started = perf_counter(); factor = solver.factorize(sparse); factor_seconds = perf_counter()-started
    sparse_answer, sparse_seconds = _timed_solve(lambda: factor.solve(rhs), repetitions)
    relative = float(torch.linalg.vector_norm(sparse_answer-dense_answer)
                     / torch.linalg.vector_norm(dense_answer))

    # Real tangent in the bordered Newton topology.  The last row is a
    # dimensionless normalized arc constraint direction and the last column
    # is the actual panel reference load.
    arc_row = rhs[:, 1] / torch.linalg.vector_norm(rhs[:, 1])
    bordered = torch.zeros((len(free)+1, len(free)+1), dtype=dense.dtype)
    bordered[:-1, :-1] = dense
    bordered[:-1, -1] = -load
    bordered[-1, :-1] = arc_row
    bordered[-1, -1] = 1.
    bordered_rhs = torch.cat((load, load.new_tensor([.1])))
    bordered_dense_answer, bordered_dense_seconds = _timed_solve(
        lambda: torch.linalg.solve(bordered, bordered_rhs), repetitions,
    )
    bordered_sparse = bordered.to_sparse_coo().coalesce()
    started = perf_counter(); bordered_factor = solver.factorize(bordered_sparse)
    bordered_factor_seconds = perf_counter()-started
    bordered_sparse_answer, bordered_sparse_seconds = _timed_solve(
        lambda: bordered_factor.solve(bordered_rhs), repetitions,
    )
    bordered_relative = float(torch.linalg.vector_norm(
        bordered_sparse_answer-bordered_dense_answer
    ) / torch.linalg.vector_norm(bordered_dense_answer))
    return {
        "mesh": f"{mesh}x{mesh}", "free_dofs": len(free),
        "tangent": {
            "relative_solution_difference": relative,
            "dense_solve_seconds": dense_seconds,
            "sparse_factor_seconds": factor_seconds,
            "sparse_solve_seconds": sparse_seconds,
            "dense_bytes": _bytes(dense), "sparse_matrix_bytes": _bytes(sparse),
            "factor_estimated_bytes": factor.diagnostics.estimated_factor_bytes,
            "factor_nnz": factor.diagnostics.factor_nnz,
            "solve_calls": factor.diagnostics.solve_calls,
            "right_hand_sides": factor.diagnostics.right_hand_sides,
            "last_relative_residual": factor.diagnostics.last_relative_residual,
        },
        "bordered_corrector": {
            "relative_solution_difference": bordered_relative,
            "dense_solve_seconds": bordered_dense_seconds,
            "sparse_factor_seconds": bordered_factor_seconds,
            "sparse_solve_seconds": bordered_sparse_seconds,
            "dense_bytes": _bytes(bordered),
            "sparse_matrix_bytes": _bytes(bordered_sparse),
            "factor_estimated_bytes": bordered_factor.diagnostics.estimated_factor_bytes,
            "last_relative_residual": bordered_factor.diagnostics.last_relative_residual,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meshes", nargs="+", type=int, default=(4, 8))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {
        "schema": "tensorfem.sparse-panel-benchmark.v1",
        "backend_inventory": [row.__dict__ for row in sparse_backend_statuses()],
        "cases": [benchmark(mesh, args.repetitions) for mesh in args.meshes],
    }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text+"\n")
    print(text)


if __name__ == "__main__":
    main()
