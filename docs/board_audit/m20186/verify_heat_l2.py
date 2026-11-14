#!/usr/bin/env python3
"""Independent re-check of m20143 / CASE-FEM-HEAT-L2 (board audit m20186).

Does NOT use the mission's own run.py driver.  It imports the delivered module
from the m20143 work tree and
  (a) reproduces the four declared mesh levels and compares them with
      benchmarks/verified/fem_heat_conduction_l2/result.json,
  (b) extends the ladder to 64x64 / 128x128 to test whether the observed order
      really approaches the theoretical Q4 value 2.0 (the anchor claim is ">=2nd order"),
  (c) recomputes the L2 error restricted to free dofs (the mission normalises over
      all nodes including the exact Dirichlet boundary nodes).

Usage: PYTHONPATH=<m20143>/src python verify_heat_l2.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import torch

from tensorfem.heat_conduction_l2 import _exact_temperature, _solve_level, _structured_quad_mesh, _edge_nodes
from tensorfem.thermal import ThermalModel, solve_steady_thermal

FLOAT = torch.float64
M20143 = Path(os.environ.get("M20143", "/root/tp-agent/work/m20143"))
RESULT = M20143 / "benchmarks/verified/fem_heat_conduction_l2/result.json"
LADDER = [tuple(int(v) for v in p.split("x")) for p in os.environ.get(
    "LADDER", "4x4,8x8,16x16,32x32,64x64,128x128").split(",")]


def free_dof_l2(nx: int, ny: int) -> float:
    """Relative L2 error over interior (free) dofs only."""
    nodes, elems = _structured_quad_mesh(nx, ny)
    bdry = _edge_nodes(nx, ny)
    fixed_nodes = torch.tensor(bdry, dtype=torch.long)
    model = ThermalModel(
        nodes=nodes, elements=elems,
        conductivity=torch.tensor(1.0, dtype=FLOAT),
        density=torch.tensor(1.0, dtype=FLOAT),
        specific_heat=torch.tensor(1.0, dtype=FLOAT),
        thickness=torch.tensor(1.0, dtype=FLOAT),
        fixed_nodes=fixed_nodes,
        fixed_temperatures=_exact_temperature(nodes[fixed_nodes]),
        element_type="q4",
    )
    T = solve_steady_thermal(model)
    e = _exact_temperature(nodes)
    mask = torch.ones(len(nodes), dtype=torch.bool)
    mask[fixed_nodes] = False
    err = T[mask] - e[mask]
    return float(torch.sqrt(torch.mean(err ** 2)) / torch.sqrt(torch.mean(e[mask] ** 2)))


def order(a: float, b: float, r: float = 2.0) -> float:
    return math.log(a / b) / math.log(r)


def main() -> int:
    ref = json.loads(RESULT.read_text()) if RESULT.exists() else {}
    ref_levels = {(lv["nx"], lv["ny"]): lv for lv in ref.get("mesh_levels", [])}

    out = {"reproduced_levels": [], "declared_vs_reproduced": [], "extended_ladder": [],
           "free_dof_l2": [], "orders": {}}

    ladder = LADDER
    l2_all, l2_free, hs = [], [], []
    for nx, ny in ladder:
        lv = _solve_level(nx, ny)
        l2_all.append(lv.l2_relative)
        hs.append(lv.h)
        l2_free.append(free_dof_l2(nx, ny))
        rec = {
            "mesh": f"{nx}x{ny}", "n_dof": lv.n_dof, "h": lv.h,
            "l2_relative_all_nodes": lv.l2_relative,
            "l2_relative_free_dofs": l2_free[-1],
            "max_relative": lv.max_relative,
            "solve_residual": lv.heat_balance_residual,
        }
        out["reproduced_levels"].append(rec)
        if (nx, ny) in ref_levels:
            r = ref_levels[(nx, ny)]
            out["declared_vs_reproduced"].append({
                "mesh": f"{nx}x{ny}",
                "declared_l2_relative": r["l2_relative"],
                "reproduced_l2_relative": lv.l2_relative,
                "abs_diff": abs(r["l2_relative"] - lv.l2_relative),
                "rel_diff": abs(r["l2_relative"] - lv.l2_relative) / r["l2_relative"],
                "declared_n_dof": r["n_dof"], "reproduced_n_dof": lv.n_dof,
            })

    o_all = [order(l2_all[i], l2_all[i + 1]) for i in range(len(l2_all) - 1)]
    o_free = [order(l2_free[i], l2_free[i + 1]) for i in range(len(l2_free) - 1)]
    out["orders"] = {
        "ladder": [f"{a}x{b}" for a, b in ladder],
        "observed_order_all_nodes": o_all,
        "observed_order_free_dofs": o_free,
        "declared_orders_all_nodes": ref.get("observed_orders_l2"),
        "declared_mean_order": ref.get("mean_order"),
        "reproduced_mean_order_first_three": sum(o_all[:3]) / 3.0,
        "asymptotic_order_finest_pair_64_128": o_all[-1],
        "count_pairs_below_2": sum(1 for o in o_all if o < 2.0),
        "count_pairs_free_dofs_below_2": sum(1 for o in o_free if o < 2.0),
    }
    out["claim_check"] = {
        "anchor_claim": "L2 convergence rate >= 2nd order",
        "declared_gate_threshold": 1.8,
        "mean_order_1p876_lt_2": (ref.get("mean_order") or 0) < 2.0,
        "all_pairs_below_2_in_declared_ladder": all(o < 2.0 for o in ref.get("observed_orders_l2", [])),
        "asymptotic_recovery_to_2_on_finer_ladder": o_all[-1] >= 1.95,
    }
    print(json.dumps(out, indent=2))
    Path(os.environ.get("OUT", str(Path(__file__).resolve().parent / "verify_heat_l2.json"))).write_text(
        json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())