"""Run the tip-loaded cantilever (elastica) validation sweep and dump JSON evidence.

Usage::

    PYTHONPATH=src OMP_NUM_THREADS=1 python scripts/run_elastica_cantilever_benchmark.py \
        --alphas 1,2,5,10 --meshes 2,4 --out artifacts/elastica_cantilever

Each case is solved with load control against the independently derived elastica
oracle.  Non-converged cases are recorded, never dropped.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from tensorfem.elastica_cantilever import (
    cantilever_benchmark,
    elastica_from_ode,
    oracle_cross_check,
    solve_elastica,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="1,2,5,10")
    ap.add_argument("--meshes", default="2,4")
    ap.add_argument("--out", default="artifacts/elastica_cantilever")
    ap.add_argument("--increment", type=float, default=0.1)
    ap.add_argument("--tolerance", type=float, default=1e-10)
    args = ap.parse_args()
    torch.set_num_threads(1)
    # the case geometry -- the reported rigidities are derived from these, so a
    # change here can never leave a stale constant in the evidence
    length, width, thickness, young = 10.0, 1.0, 0.1, 1.2e6
    flexural = young * width * thickness ** 3 / 12.0
    alpha_unit = 1.0
    alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
    meshes = [int(m) for m in args.meshes.split(",") if m.strip()]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    oracle = []
    for alpha in alphas:
        q = solve_elastica(alpha); b = elastica_from_ode(alpha)
        oracle.append({"alpha": alpha, "tip_rotation_rad": q.tip_rotation,
                       "deflection_ratio": q.deflection_ratio,
                       "shortening_ratio": q.shortening_ratio,
                       "bvp_deflection_ratio": b.deflection_ratio,
                       "bvp_shortening_ratio": b.shortening_ratio,
                       "route_difference": oracle_cross_check(alpha)})

    start = time.time()
    bench = cantilever_benchmark(alphas=alphas, meshes=meshes,
                                 increment=args.increment, tolerance=args.tolerance)
    payload = {
        "schema": "tensorfem.elastica-cantilever.v1",
        "model": {"length": length, "width": width, "thickness": thickness,
                  "young": young, "poisson": 0.0,
                  # derived, not typed: E I = E b t^3 / 12 and P = alpha E I / L^2
                  "flexural_rigidity": flexural,
                  "axial_rigidity": young * width * thickness,
                  "tip_load": alpha_unit * flexural / length ** 2,
                  "alpha_definition": "P L^2 / (E I)",
                  "load": f"P = {alpha_unit * flexural / length ** 2:g} * alpha"},
        "solver": {"element": "corotational 2-node Euler-Bernoulli beam, 3 DOF/node",
                   "increment": args.increment, "tolerance": args.tolerance,
                   "threads": 1},
        "tolerance": 0.03,
        "environment": {"python": platform.python_version(), "torch": torch.__version__},
        "oracle": oracle,
        "benchmark": bench.as_dict(),
        "wall_seconds": time.time() - start,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(bench.as_dict()["qualified"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
