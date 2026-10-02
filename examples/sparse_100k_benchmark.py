"""Opt-in sparse benchmark: 100,000 independent spring DOFs (not in pytest)."""
import argparse, json, time, torch
from tensorfem.sparse_core import assemble_coo
from tensorfem.sparse_advanced import jacobi_inverse, solve_multiple_rhs

def main():
    p = argparse.ArgumentParser(); p.add_argument("--dofs", type=int, default=100_000)
    p.add_argument("--rhs", type=int, default=1)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--output"); a = p.parse_args(); n = a.dofs
    if a.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(a.device); dtype = torch.float64; t0 = time.perf_counter()
    # One scalar ground spring per DOF: u_i=f_i/k_i=1 is analytical, while
    # still exercising 100k-element vector assembly and sparse matvec/CG.
    stiffness = torch.logspace(0., 6., n, dtype=dtype, device=device)
    # Keep COO as the portable default. PyTorch 2.12 still labels its CSR API
    # beta; callers can explicitly request CSR when their backend is qualified.
    K = assemble_coo(stiffness[:, None, None], torch.arange(n, device=device)[:, None], n)
    assembled = time.perf_counter() - t0
    f = stiffness[:, None].repeat(1, a.rhs); t1 = time.perf_counter()
    inv = jacobi_inverse(K)
    r = solve_multiple_rhs(K, f, method="cg", jacobi=inv, rtol=1e-10, maxiter=30)
    solved = time.perf_counter() - t1
    error = float(torch.linalg.vector_norm(r.x - 1.) / (n*a.rhs) ** .5)
    evidence = {"dofs": n, "right_hand_sides": a.rhs, "device": a.device,
                "condition_number": 1e6, "nnz": K.values().numel(),
                "assembly_seconds": assembled, "solve_seconds": solved,
                "iterations": list(r.iterations),
                "relative_residual": list(r.relative_residuals),
                "displacement_relative_error": error, "passed": error < .03}
    print(json.dumps(evidence, indent=2))
    if a.output:
        with open(a.output, "w") as h: json.dump(evidence, h, indent=2)
    if not evidence["passed"]: raise SystemExit(1)

if __name__ == "__main__": main()
