"""Real-discretization arc-length evidence for the corotational Shell4.

This module intentionally qualifies only a clamped, one-element cantilever
path.  It exercises the assembled Shell4 internal force and its autograd
Hessian at every Crisfield corrector iteration; it is not a postbuckling or
general curved-shell qualification.
"""
from __future__ import annotations

import torch

from .arc_length import general_shell_arc_problem, solve_arc_length
from .general_shell_nonlinear import GeneralShellMesh


def cantilever_shell_problem(*, dtype: torch.dtype = torch.float64):
    """Return a small finite-rotation Shell4 cantilever and its reduced problem."""
    nodes = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
         [1.0, 0.2, 0.0], [0.0, 0.2, 0.0]], dtype=dtype)
    mesh = GeneralShellMesh(nodes, torch.tensor([[0, 1, 2, 3]]),
                            young=2.1e6, poisson=0.3, thickness=0.03)
    load = torch.zeros(24, dtype=dtype)
    load[8] = load[14] = -10.0
    fixed = tuple(range(6)) + tuple(range(18, 24))
    problem, free = general_shell_arc_problem(mesh, load, fixed)
    return mesh, load, problem, free


def trace_cantilever_shell(*, step_size: float = 0.05, steps: int = 8):
    """Trace the real Shell4 cantilever equilibrium path.

    ``step_size * steps = 0.4`` is used by the convergence evidence.  Callers
    choosing another total arc length should not compare endpoints directly.
    """
    _mesh, _load, problem, free = cantilever_shell_problem()
    result = solve_arc_length(
        problem, torch.zeros(len(free), dtype=torch.float64), steps=steps,
        step_size=step_size, maximum_step=step_size, load_scale=0.1,
        tolerance=1e-6, max_iterations=15, minimum_step=1e-6)
    return result, free


def run_shell4_arc_path_evidence() -> dict[str, object]:
    """Run step-size and discrete-equilibrium gates for the limited path."""
    rows = []
    for ds, steps in ((0.1, 4), (0.05, 8)):
        result, free = trace_cantilever_shell(step_size=ds, steps=steps)
        if not result.converged:
            raise AssertionError("Shell4 arc-length path did not converge")
        point = result.points[-1]
        probe = int(torch.nonzero(free == 14).flatten()[0])
        rows.append({
            "step_size": ds,
            "steps": steps,
            "load_factor": point.load_factor,
            "tip_displacement": float(point.displacement[probe]),
            "iterations": sum(p.iterations for p in result.points),
            "maximum_residual_norm": max(p.residual_norm for p in result.points),
        })
    load_change = abs(rows[0]["load_factor"] / rows[1]["load_factor"] - 1.0)
    displacement_change = abs(rows[0]["tip_displacement"] /
                              rows[1]["tip_displacement"] - 1.0)
    passed = load_change < 0.03 and displacement_change < 0.03
    if not passed:
        raise AssertionError("Shell4 arc-length step convergence failed")
    return {
        "schema": "tensorfem.shell4-arc-path/1.0",
        "model": "one flat-facet corotational Shell4 cantilever",
        "rows": rows,
        "relative_step_change": {
            "load_factor": load_change,
            "tip_displacement": displacement_change,
        },
        "passed": True,
        "boundary": ("real assembled residual and consistent autograd tangent; "
                     "no shell snap-through, postbuckling, plasticity, or mesh-convergence claim"),
    }
