#!/usr/bin/env python3
"""Kerr (three-parameter) foundation plate: exact Navier-type benchmark.

Verification debt for the recently merged ``tensorfem.kerr_foundation`` module
(m14902).  The module shipped with unit-level tests only (element stiffness,
equivalent parameters, symmetry, PSD) and no full-plate analytical benchmark.

Benchmark
---------
A simply-supported square plate (side ``L``) on a Kerr foundation
``q_f = k w - G_p nabla^2 w`` under a single sinusoidal load term
``q = q0 sin(pi x/L) sin(pi y/L)``.  The governing equation

    D nabla^4 w + k w - G_p nabla^2 w = q0 sin(pi x/L) sin(pi y/L)

admits the exact closed-form centre deflection (Navier double-series, single
surviving term)::

    W = q0 / [ 4 D (pi/L)^4 + k + 2 G_p (pi/L)^2 ],   D = E t^3 / (12 (1-nu^2)).

The FEM side uses the production Q4 Mindlin element
(``tensorfem.plate.q4_mindlin_stiffness``) plus the production Kerr foundation
stiffness (``tensorfem.kerr_foundation.q4_kerr_foundation_stiffness``) with the
hard simply-supported Mindlin boundary (w=0, tangential rotation=0).  The two
sides share no code: the analytic side is a scalar formula, the FEM side is the
assembled element system.

Reproduce
---------
    PYTHONPATH=src:benchmarks/verified python benchmarks/verified/fem_kerr_plate/run.py
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))

from tensorfem.plate import (  # noqa: E402
    _shape,
    q4_mindlin_stiffness,
    structured_square_mesh,
)
from tensorfem.kerr_foundation import q4_kerr_foundation_stiffness  # noqa: E402

# --- fixed case parameters -------------------------------------------------
L = 1.0            # plate side (m)
E = 2.1e11         # Young (Pa)
NU = 0.3           # Poisson
T = 0.01           # thickness (m)
Q0 = 1.0e4         # load amplitude (Pa)
K_WINKLER = 1.0e8  # Winkler spring coefficient (N/m^3)
G_SHEAR = 5.0e4    # Kerr shear parameter (N/m)
D = E * T**3 / (12.0 * (1.0 - NU**2))

MESHES = (4, 8, 16, 32)


def exact_center_deflection(k: float, gp: float) -> float:
    """Closed-form centre deflection for the single sinusoidal load term."""
    denom = 4.0 * D * (math.pi / L) ** 4 + k + 2.0 * gp * (math.pi / L) ** 2
    return Q0 / denom


def solve_plate(n: int, k: float, gp: float) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Assemble + solve the simply-supported square on a Kerr foundation."""
    nodes, elems = structured_square_mesh(n, L, dtype=torch.float64)
    ndof = 3 * nodes.shape[0]
    K = torch.zeros((ndof, ndof), dtype=torch.float64)
    f = torch.zeros(ndof, dtype=torch.float64)
    gpq = 1.0 / 3.0**0.5
    for conn in elems:
        xy = nodes[conn]
        ke = q4_mindlin_stiffness(xy, E, NU, T)
        ke += q4_kerr_foundation_stiffness(xy, k, gp)
        fe = torch.zeros(12, dtype=torch.float64)
        for xif in (-gpq, gpq):
            for etaf in (-gpq, gpq):
                xi, eta = nodes.new_tensor(xif), nodes.new_tensor(etaf)
                N, dN_nat = _shape(xi, eta)
                J = dN_nat.T @ xy
                p = N @ xy
                q = Q0 * math.sin(math.pi * p[0].item() / L) * math.sin(math.pi * p[1].item() / L)
                fe[0::3] += N * q * torch.linalg.det(J)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dofs[:, None], dofs] += ke
        f[dofs] += fe
    tol = L * 1e-12
    xedge = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - L).abs() < tol)
    yedge = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - L).abs() < tol)
    edge = xedge | yedge
    fixed = torch.cat((
        3 * torch.nonzero(edge, as_tuple=False).flatten(),
        3 * torch.nonzero(yedge, as_tuple=False).flatten() + 1,
        3 * torch.nonzero(xedge, as_tuple=False).flatten() + 2,
    )).unique()
    all_dofs = torch.arange(ndof)
    free = all_dofs[~torch.isin(all_dofs, fixed)]
    u = torch.zeros(ndof, dtype=torch.float64)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    # centre node = index (n//2)*(n+1) + n//2
    c = (n // 2) * (n + 1) + n // 2
    return float(u[3 * c]), u, nodes


def main() -> None:
    t0 = time.time()
    results = {}
    for (k, gp, tag) in [
        (K_WINKLER, G_SHEAR, "kerr_full"),
        (K_WINKLER, 0.0, "winkler_limit"),
        (0.0, 0.0, "free_plate"),
    ]:
        W_exact = exact_center_deflection(k, gp)
        rows = []
        for n in MESHES:
            w_fem, u, nodes = solve_plate(n, k, gp)
            err = abs(w_fem / W_exact - 1.0)
            rows.append({"n": n, "w_fem": w_fem, "w_exact": W_exact, "rel_err": err})
        # convergence order between finest two meshes
        e1, e2 = rows[-2]["rel_err"], rows[-1]["rel_err"]
        order = math.log(e1 / e2) / math.log(2.0) if e2 > 0 else float("nan")
        results[tag] = {
            "k": k, "gp": gp, "w_exact": exact_center_deflection(k, gp),
            "w_fem_fine": rows[-1]["w_fem"],
            "rel_err_fine": rows[-1]["rel_err"],
            "convergence_order": order,
            "meshes": rows,
        }

    payload = {
        "benchmark": "fem_kerr_plate",
        "anchor": "GAP-FEM-KERR-PLATE",
        "case": {
            "side_m": L, "young_Pa": E, "poisson": NU, "thickness_m": T,
            "q0_Pa": Q0, "winkler_k_N_m3": K_WINKLER, "shear_gp_N_m": G_SHEAR,
            "rigidity_D_Nm": D,
        },
        "results": results,
        "wall_s": time.time() - t0,
    }
    out = HERE / "result.json"
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps({t: {kk: results[t][kk] for kk in
                          ("w_exact", "w_fem_fine", "rel_err_fine", "convergence_order")}
                     for t in results}, indent=2))

    # ---- figures ----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        # left: deflection profile along x at y=L/2 for finest Kerr mesh
        n = MESHES[-1]
        w_fem, u, nodes = solve_plate(n, K_WINKLER, G_SHEAR)
        W_ex = exact_center_deflection(K_WINKLER, G_SHEAR)
        jmid = n // 2
        xs = []
        wf, we = [], []
        for i in range(n + 1):
            c = jmid * (n + 1) + i
            x = float(nodes[c, 0])
            xs.append(x)
            wf.append(float(u[3 * c]))
            we.append(W_ex * math.sin(math.pi * x / L) * math.sin(math.pi / 2))
        ax[0].plot(xs, wf, "o-", ms=3, label="FEM (Q4 Mindlin + Kerr)")
        ax[0].plot(xs, we, "-", lw=2, label="exact Kerr (closed form)")
        ax[0].set_xlabel("x (m)")
        ax[0].set_ylabel("w (m)")
        ax[0].set_title("Centre-line deflection, Kerr foundation")
        ax[0].legend()
        # right: convergence
        ns = [r["n"] for r in results["kerr_full"]["meshes"]]
        errs = [r["rel_err"] for r in results["kerr_full"]["meshes"]]
        ax[1].loglog(ns, errs, "o-")
        ax[1].loglog(ns, [errs[0] * (ns[0] / x) ** 2 for x in ns], "--", label="ref O(h^2)")
        ax[1].set_xlabel("elements per side")
        ax[1].set_ylabel("centre-deflection rel. error")
        ax[1].set_title("Convergence (Kerr)")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(HERE / "kerr_plate_convergence.png", dpi=130)
        plt.close(fig)
    except Exception as exc:  # figures are non-fatal
        print("figure skipped:", exc)

    print("wrote", out)


if __name__ == "__main__":
    import time
    main()
