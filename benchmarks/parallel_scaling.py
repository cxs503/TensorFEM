"""Strong/weak MPI scaling study for the TensorFEM distributed Poisson path.

    python benchmarks/parallel_scaling.py [--mesh 640] [--iters 20] [--repeats 2]

Writes ``artifacts/parallel_scaling/{strong.json,weak.json,scaling.json,*.png}``.
The predeclared protocol lives in ``tensorfem/parallel/scaling.py`` and
``docs/parallel-domain-decomposition.md``; nothing is dropped from the grid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfem.parallel.scaling import (DEFAULT_STRONG, DEFAULT_WEAK,  # noqa: E402
                                        strong_scaling, weak_scaling)

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "parallel_scaling")


def _print_table(study: dict) -> None:
    print(f"\n== {study['study']} scaling (config={study['config']}) ==")
    print(f"{'ranks':>5} {'grid':>11} {'dofs':>9} {'T[s]':>9} {'s/iter':>9}"
          f" {'S(p)':>6} {'E(p)':>6} {'L2':>11} {'nbr':>4}")
    for p in study["points"]:
        g = f"{p['mesh']}x{p.get('mesh_y', p['mesh'])}"
        print(f"{p['ranks']:>5} {g:>11} {p['global_dofs']:>9d} "
              f"{p['cg_seconds']:>9.4f} {p['seconds_per_iter']:>9.5f} "
              f"{p['speedup']:>6.2f} {p['efficiency']:>6.3f} "
              f"{p['l2_error']:>11.4e} {p['max_neighbors']:>4d}")


def _figures(strong: dict, weak: dict) -> list[str]:
    os.makedirs(OUT, exist_ok=True)
    made = []

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    r = [p["ranks"] for p in strong["points"]]
    ax[0].plot(r, [p["speedup"] for p in strong["points"]], "o-", label="measured")
    ax[0].plot(r, r, "k--", label="ideal")
    ax[0].set_xlabel("ranks"); ax[0].set_ylabel("speedup S(p)")
    ax[0].set_title(f"strong scaling, {strong['points'][0]['mesh']}x"
                    f"{strong['points'][0].get('mesh_y', strong['points'][0]['mesh'])}")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(r, [p["efficiency"] for p in strong["points"]], "s-", color="tab:red")
    ax[1].axhline(0.60, color="k", ls="--", label="gate 0.60")
    ax[1].set_xlabel("ranks"); ax[1].set_ylabel("efficiency E(p)")
    ax[1].set_title("parallel efficiency"); ax[1].legend(); ax[1].grid(alpha=.3)
    f = os.path.join(OUT, "strong_scaling.png")
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig); made.append(f)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    rw = [p["ranks"] for p in weak["points"]]
    ax[0].loglog(rw, [p["cg_seconds"] for p in weak["points"]], "o-")
    ax[0].set_xlabel("ranks"); ax[0].set_ylabel("wall time [s] per solve")
    ax[0].set_title("weak scaling: time vs ranks (ideal: flat)")
    ax[0].grid(alpha=.3, which="both")
    ax[1].plot(rw, [p["efficiency"] for p in weak["points"]], "s-", color="tab:green")
    ax[1].axhline(0.60, color="k", ls="--", label="gate 0.60")
    ax[1].set_xlabel("ranks"); ax[1].set_ylabel("weak efficiency")
    ax[1].set_title("weak scaling efficiency"); ax[1].legend(); ax[1].grid(alpha=.3)
    f = os.path.join(OUT, "weak_scaling.png")
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig); made.append(f)

    fig, ax = plt.subplots(figsize=(6, 4))
    r = [p["ranks"] for p in strong["points"]]
    ax.plot(r, [p["cg_seconds"] / p["iters"] * 1e3 for p in strong["points"]],
            "o-", label="total per iteration")
    ax.plot(r, [(p.get("t_matvec_s") or 0) * 1e3 for p in strong["points"]],
            "s-", label="matvec (incl. halo)")
    ax.plot(r, [(p.get("per_iter_comm_s") or 0) * 1e3 for p in strong["points"]],
            "^-", label="halo + 2 Allreduce")
    ax.set_xlabel("ranks"); ax.set_ylabel("ms per CG iteration")
    ax.set_title("iteration cost breakdown"); ax.legend(); ax.grid(alpha=.3)
    f = os.path.join(OUT, "cost_breakdown.png")
    fig.tight_layout(); fig.savefig(f, dpi=130); plt.close(fig); made.append(f)
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=int, default=DEFAULT_STRONG["mesh"])
    ap.add_argument("--iters", type=int, default=DEFAULT_STRONG["iters"])
    ap.add_argument("--repeats", type=int, default=DEFAULT_STRONG["repeats"])
    ap.add_argument("--weak-base", type=int, default=DEFAULT_WEAK["mesh_per_rank"])
    ap.add_argument("--ranks", type=str, default="1,2,4,8")
    args = ap.parse_args()
    ranks = tuple(int(x) for x in args.ranks.split(","))
    os.makedirs(OUT, exist_ok=True)

    t0 = time.perf_counter()
    strong = strong_scaling(ranks, {"mesh": args.mesh, "iters": args.iters,
                                    "repeats": args.repeats})
    _print_table(strong)
    weak = weak_scaling(ranks, {"mesh_per_rank": args.weak_base,
                                "iters": args.iters, "repeats": args.repeats})
    _print_table(weak)

    figs = _figures(strong, weak)
    bundle = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": os.uname().nodename,
        "cpu_count": os.cpu_count(),
        "wall_seconds_total": time.perf_counter() - t0,
        "strong": strong,
        "weak": weak,
        "figures": [os.path.relpath(f, os.path.join(OUT, "..", "..")) for f in figs],
        "gates": {
            "efficiency_at_4_ranks_strong": strong["gate"],
            "weak_efficiency_at_4_ranks": weak["gate"],
            "all_passed": strong["gate"]["passed"] and weak["gate"]["passed"],
        },
    }
    for name, obj in (("strong.json", strong), ("weak.json", weak),
                      ("scaling.json", bundle)):
        with open(os.path.join(OUT, name), "w") as fh:
            json.dump(obj, fh, indent=2)
    print(f"\ngates: strong E(4)={strong['gate']['value']:.3f} "
          f"passed={strong['gate']['passed']}; "
          f"weak E(4)={weak['gate']['value']:.3f} passed={weak['gate']['passed']}")
    print("wrote", os.path.relpath(OUT, os.getcwd()))
    return 0 if bundle["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())