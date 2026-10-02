"""Opt-in sparse benchmark: 100,000 independent spring DOFs (not in pytest)."""
import argparse, json, time, torch
from tensorfem.sparse_core import assemble_coo, solve_sparse_static

def main():
    p = argparse.ArgumentParser(); p.add_argument("--dofs", type=int, default=100_000)
    p.add_argument("--output"); a = p.parse_args(); n = a.dofs
    dtype = torch.float64; t0 = time.perf_counter()
    # One scalar ground spring per DOF: u_i=f_i/k_i=1 is analytical, while
    # still exercising 100k-element vector assembly and sparse matvec/CG.
    stiffness = torch.linspace(1., 2., n, dtype=dtype)
    # Keep COO as the portable default. PyTorch 2.12 still labels its CSR API
    # beta; callers can explicitly request CSR when their backend is qualified.
    K = assemble_coo(stiffness[:, None, None], torch.arange(n)[:, None], n)
    assembled = time.perf_counter() - t0
    f = stiffness.clone(); t1 = time.perf_counter()
    r = solve_sparse_static(K, f, rtol=1e-10, maxiter=30)
    solved = time.perf_counter() - t1
    error = float(torch.linalg.vector_norm(r.displacement - 1.) / n ** .5)
    evidence = {"dofs": n, "nnz": K.values().numel(), "assembly_seconds": assembled,
                "solve_seconds": solved, "iterations": r.iterative.iterations,
                "relative_residual": r.iterative.relative_residual,
                "displacement_relative_error": error, "passed": error < .03}
    print(json.dumps(evidence, indent=2))
    if a.output:
        with open(a.output, "w") as h: json.dump(evidence, h, indent=2)
    if not evidence["passed"]: raise SystemExit(1)

if __name__ == "__main__": main()
