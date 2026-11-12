#!/usr/bin/env python3
"""Independent check of the Morley 30 deg rhombic-plate anchor (audit-only tool).

Method: *not* finite elements and *not* the delivered code.  The Kirchhoff
biharmonic problem on a parallelogram is split into two second-order problems

    laplacian(w) = phi,   laplacian(phi) = q / D
    w = 0 and phi = 0 on all four straight edges

(the second condition is exactly the simply-supported ``M_n = 0`` condition,
because on a straight edge ``w = 0`` implies ``w,ss = 0`` hence
``M_n = -D w,nn = 0`` and ``laplacian(w) = w,nn + w,ss = 0``).

The Laplacian is expressed in the oblique coordinates (xi, eta) of the
parallelogram and discretised with second-order central differences on a
uniform grid, so the only discretisation error is h^2 plus the corner
singularity.  This gives a second, method-independent estimate of the
normalised centre deflection that the FE ladder can be judged against.

    python scripts/audit/independent_biharmonic.py --out runs/m16196_audit/independent_biharmonic.json
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.sparse as spa
import scipy.sparse.linalg as spla


def laplacian(n: int, skew_deg: float, side: float):
    """Sparse 9-point Laplacian of a grid function on the (n-1)^2 interior nodes."""
    a = math.radians(skew_deg)
    p = np.array([1.0 / side, 0.0])                        # grad(xi)
    q = np.array([-math.cos(a) / (side * math.sin(a)), 1.0 / (side * math.sin(a))])  # grad(eta)
    A, B, C = p @ p, p @ q, q @ q
    h = 1.0 / n
    idx = np.arange((n - 1) ** 2).reshape(n - 1, n - 1)

    rows, cols, vals = [], [], []
    for i in range(1, n):
        for j in range(1, n):
            r = idx[i - 1, j - 1]
            # u_xixi
            for di, dj, v in ((1, 0, A / h ** 2), (-1, 0, A / h ** 2), (0, 0, -2 * A / h ** 2)):
                ci, cj = i + di, j + dj
                rows.append(r)
                cols.append(idx[ci - 1, cj - 1] if 0 < ci < n and 0 < cj < n else None)
                vals.append(v)
            # u_etaeta
            for di, dj, v in ((0, 1, C / h ** 2), (0, -1, C / h ** 2), (0, 0, -2 * C / h ** 2)):
                ci, cj = i + di, j + dj
                rows.append(r)
                cols.append(idx[ci - 1, cj - 1] if 0 < ci < n and 0 < cj < n else None)
                vals.append(v)
            # 2 B u_xieta
            for di, dj in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                sgn = 1.0 if di * dj > 0 else -1.0
                ci, cj = i + di, j + dj
                rows.append(r)
                cols.append(idx[ci - 1, cj - 1] if 0 < ci < n and 0 < cj < n else None)
                vals.append(2.0 * B * sgn / (4 * h ** 2))
    keep = [k for k, c in enumerate(cols) if c is not None]
    rows = np.asarray(rows)[keep]
    cols = np.asarray(cols)[keep]
    vals = np.asarray(vals)[keep]
    return spa.csr_matrix((vals, (rows, cols)), shape=((n - 1) ** 2, (n - 1) ** 2)), A, B, C


def solve(n, skew_deg, side, rigidity, pressure):
    L, A, B, C = laplacian(n, skew_deg, side)
    phi = spla.spsolve(L, np.full((n - 1) ** 2, pressure / rigidity))
    w = spla.spsolve(L, phi)
    centre = w.reshape(n - 1, n - 1)[n // 2 - 1, n // 2 - 1]
    return float(centre), (A, B, C)


def main() -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/m16196_audit/independent_biharmonic.json")
    ap.add_argument("--sides", default="100")
    ap.add_argument("--grids", default="16,32,64,96,128")
    a = ap.parse_args()
    q, E, nu = 1.0, 10.92, 0.3
    rows = []
    for side in [float(s) for s in a.sides.split(",")]:
        for lh in (1000.0, 100.0):
            h = side / lh
            D = E * h ** 3 / (12.0 * (1 - nu ** 2))
            for n in [int(g) for g in a.grids.split(",")]:
                t0 = time.time()
                wc, coeffs = solve(n, 30.0, side, D, q)
                rows.append({
                    "side": side, "L_over_h": lh, "D": D, "n": n,
                    "centre_deflection": wc,
                    "wc_N": wc / (1e-3 * q * side ** 4 / D),
                    "seconds": time.time() - t0,
                })
                print(json.dumps(rows[-1]), flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "method": "oblique-coordinate 9-point FD split biharmonic (w=0, laplacian(w)=0 on all edges); "
                  "second order; independent of the FE ladder under audit",
        "skew_deg": 30.0, "poisson": nu, "young": E, "pressure": q,
        "rows": rows,
    }
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print("wrote", out)
    return doc


if __name__ == "__main__":
    main()