"""Why did the load-controlled elastica rung fail under mesh refinement?

The merged validation study (`docs/elastica-cantilever-validation.md`) reported
that ``n = 8`` and ``n = 16`` "do not converge" while ``n = 2, 4`` do, and
attributed it to a worsening condition number of the central-difference tangent.
This script tests that attribution.

It measures, for each mesh, on the *same* energy, tangent and Newton loop:

* the residual level a converged state actually reaches (the round-off floor of
  the internal-force assembly, which is a cancellation of terms of magnitude
  ``EA / L`` and therefore grows with the mesh);
* what the pre-fix convergence predicate ``||r|| <= 1e-10`` (absolute) makes of
  that state;
* a **replay of the pre-fix loop** (the load-step halving driven by that
  predicate, restored verbatim from the merged code) so the reported load
  factors of the old study can be reproduced and the failure attributed.

Outputs ``artifacts/elastica_cantilever/floor_study.json`` (every number used by
the report is read back from there).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from tensorfem.elastica_cantilever import (
    _beam_energy,
    corotational_beam_cantilever,
    solve_elastica,
)

LENGTH = 10.0
YOUNG = 1.2e6
THICKNESS = 0.1
WIDTH = 1.0
FLEXURAL = YOUNG * WIDTH * THICKNESS ** 3 / 12.0   # E I = 10
AXIAL = YOUNG * WIDTH * THICKNESS                  # E A


def _gradient(nodes0, dofs, lengths):
    q = dofs.detach().clone().requires_grad_(True)
    return torch.autograd.grad(
        _beam_energy(nodes0, q.reshape(-1, 3), AXIAL, FLEXURAL, lengths), q)[0]


def _tangent(nodes0, dofs, lengths, h: float = 1e-6):
    """The pre-fix central-difference Hessian, restored verbatim."""
    base = dofs.detach().clone()
    cols = []
    for j in range(base.numel()):
        step = torch.zeros_like(base)
        step[j] = h
        cols.append((_gradient(nodes0, base + step, lengths)
                     - _gradient(nodes0, base - step, lengths)) / (2 * h))
    K = torch.stack(cols, dim=1)
    return 0.5 * (K + K.T)


def replay_prefix_loop(alpha: float, elements: int, *, tolerance: float = 1e-10,
                       increment: float = 0.1, minimum_increment: float = 1e-3,
                       max_iterations: int = 15) -> dict:
    """Pre-fix load-controlled loop: absolute residual test drives step halving.

    Identical to :func:`tensorfem.elastica_cantilever.corotational_beam_cantilever`
    as merged in commit cffdb7ea except that the convergence predicate is the
    single absolute test ``||r|| <= tolerance`` -- i.e. the round-off branch is
    removed.  Everything else (energy, tangent, load stepping) is the same code.
    """
    load = alpha * FLEXURAL / LENGTH ** 2
    xs = np.linspace(0.0, LENGTH, elements + 1)
    nodes0 = torch.tensor(np.stack([xs, np.zeros_like(xs)], axis=1), dtype=torch.float64)
    lengths = torch.full((elements,), LENGTH / elements, dtype=torch.float64)
    dofs = torch.zeros(3 * (elements + 1), dtype=torch.float64)
    free = torch.arange(3, dofs.numel())
    force = torch.zeros_like(dofs)
    force[3 * elements + 1] = -load

    factor, step, iterations, rejected = 0.0, min(increment, 1.0), 0, 0
    q = dofs.clone()
    residual = float("inf")
    while factor < 1.0 - 1e-14:
        target = min(1.0, factor + step)
        trial = q.clone()
        converged = False
        for _ in range(max_iterations):
            iterations += 1
            res = _gradient(nodes0, trial, lengths) - target * force
            sol = res[free].detach()
            residual = float(torch.linalg.vector_norm(sol))
            if residual <= tolerance:
                converged = True
                break
            try:
                delta = torch.linalg.solve(
                    _tangent(nodes0, trial, lengths)[free[:, None], free], -sol)
            except torch.linalg.LinAlgError:
                break
            trial = trial.detach().clone()
            trial[free] += delta
        if converged:
            q = trial.detach().clone()
            factor = target
            step = min(step * 1.5, 0.5, 1.0 - factor + 1e-15) if factor < 1.0 else step
        else:
            rejected += 1
            step *= 0.5
            if step < minimum_increment:
                break
    tip = q.detach().reshape(-1, 3)[-1]
    exact = solve_elastica(alpha)
    deflection = -float(tip[1])
    return {
        "alpha": alpha,
        "elements": elements,
        "converged": bool(factor >= 1.0 - 1e-9),
        "load_factor": factor,
        "iterations": iterations,
        "rejected_increments": rejected,
        "residual_at_last_trial": residual,
        "tip_deflection": deflection,
        "deflection_error": abs(deflection / exact.tip_deflection(LENGTH) - 1.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1,10")
    ap.add_argument("--meshes", default="2,4,8,16,24,32")
    ap.add_argument("--out", default="artifacts/elastica_cantilever/floor_study.json")
    args = ap.parse_args()
    torch.set_num_threads(1)
    alphas = [float(a) for a in args.alphas.split(",")]
    meshes = [int(m) for m in args.meshes.split(",")]

    fixed, replay = [], []
    for alpha in alphas:
        for n in meshes:
            case = corotational_beam_cantilever(alpha, n)
            fixed.append({
                "alpha": alpha, "elements": n, "converged": case.converged,
                "load_factor": case.load_factor, "iterations": case.iterations,
                "residual": case.residual,
                "deflection_error": case.deflection_error,
                # what the pre-fix absolute predicate makes of the same state
                "prefix_predicate_accepts": case.residual <= 1e-10,
            })
            if n <= 16:                      # replay is expensive, keep it small
                replay.append(replay_prefix_loop(alpha, n))

    payload = {
        "schema": "tensorfem.elastica-convergence-floor.v1",
        "model": {"length": LENGTH, "flexural_rigidity": FLEXURAL,
                  "axial_rigidity": AXIAL},
        "question": "does the n=8/16 failure of the merged study come from the "
                    "tangent condition number or from the convergence predicate?",
        "fixed_convergence_test": fixed,
        "prefix_replay": replay,
    }
    with open(args.out, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    for row in fixed:
        print(f"fixed  a={row['alpha']:5} n={row['elements']:3d} "
              f"conv={row['converged']} dErr={row['deflection_error']*100:7.4f}% "
              f"res={row['residual']:.2e} old_pred={row['prefix_predicate_accepts']}")
    for row in replay:
        print(f"replay a={row['alpha']:5} n={row['elements']:3d} "
              f"factor={row['load_factor']:.3f} rejected={row['rejected_increments']} "
              f"dErr={row['deflection_error']*100:7.4f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())