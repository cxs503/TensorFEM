"""Independent FV (cut-cell, axis-aligned) check of the 2-D heave added mass (omega0 route).

Third discretisation family next to the audited BEM.  Cell-centred finite volumes
for laplace(phi)=0 on the doubled section [-c,c]x[-h,h] removed from a box
[-R,R]^2, with

    dphi/dn_fluid = V * n_z   on the body faces (V = 1 heave velocity)
    phi = 0                   on the outer box (truncation at R)

Body faces coincide with cell faces, so the cut cells are exact (no staircase
error).  Two functionals of the same discrete solution are reported (force
integral and kinetic energy) as an internal error indicator, plus sweeps in h and
R so both the discretisation and the truncation error are visible.
"""
from __future__ import annotations

import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

RHO = 1025.0
C, H = 10.0, 6.0


def solve_fv(h: float, box: float) -> dict:
    n = int(round(2 * box / h))
    xc = -box + (np.arange(n) + 0.5) * h
    X, Z = np.meshgrid(xc, xc, indexing="ij")
    inside_body = (np.abs(X) < C) & (np.abs(Z) < H)
    fluid = ~inside_body

    idx, loc, k = {}, np.full((n, n), -1, dtype=int), 0
    for i in range(n):
        for j in range(n):
            if fluid[i, j]:
                idx[(i, j)] = k
                loc[i, j] = k
                k += 1
    ncell = len(idx)
    rows, cols, vals = [], [], []
    rhs = np.zeros(ncell)

    # face flux helper: returns (coefficient_on_phi_cell, constant_flux)
    for (i, j), kk in idx.items():
        diag = 0.0
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if ii < 0 or ii >= n or jj < 0 or jj >= n:
                diag += -2.0 / h                     # Dirichlet phi=0 at h/2
            elif inside_body[ii, jj]:
                rhs[kk] -= float(dj)                 # flux = V*n_z(fluid), n_z = dj
            else:
                rows.append(kk); cols.append(loc[ii, jj]); vals.append(1.0 / h)
                diag += -1.0 / h
        rows.append(kk); cols.append(kk); vals.append(diag)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(ncell, ncell))
    phi = spla.spsolve(A, rhs)

    def flux(i, j, di, dj):
        """Flux through the face of cell (i,j) in direction (di,dj)."""
        ii, jj = i + di, j + dj
        kk = loc[i, j]
        if ii < 0 or ii >= n or jj < 0 or jj >= n:
            return -2.0 * phi[kk] / h
        if inside_body[ii, jj]:
            return float(dj)                          # V=1 on z-faces, 0 on x-faces
        return (phi[loc[ii, jj]] - phi[kk]) / h

    force = 0.0
    energy = 0.0
    for (i, j), kk in idx.items():
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < n and 0 <= jj < n and inside_body[ii, jj] and dj != 0:
                force += phi[kk] * (-dj) * h         # body outward n_z = -dj
        gx = 0.5 * (flux(i, j, 1, 0) - flux(i, j, -1, 0))
        gz = 0.5 * (flux(i, j, 0, 1) - flux(i, j, 0, -1))
        energy += (gx ** 2 + gz ** 2) * h * h

    return {
        "h": h, "box": box, "cells_fluid": ncell,
        "A_half_force_t_per_m": abs(RHO * force) / 2.0 / 1e3,
        "A_half_energy_t_per_m": abs(RHO * energy) / 2.0 / 1e3,
    }


def richardson(v1: float, v2: float, v3: float, r: float = 2.0) -> tuple:
    """Observed order and limit from three successive refinements (ratio r)."""
    e1, e2 = v2 - v1, v3 - v2
    if e2 == 0.0 or e1 / e2 <= 0.0:
        return float("nan"), float("nan")
    p = math.log(e1 / e2) / math.log(r)
    return p, v3 + e2 / (r ** p - 1.0)


if __name__ == "__main__":
    res = {}
    for box in (60.0, 120.0, 240.0):
        for h in ((1.0,) if box == 240.0 else (1.0, 0.5, 0.25)):
            r = solve_fv(h, box)
            res[(box, h)] = r
            print(f"box={box:5.1f} h={h:4.2f} force={r['A_half_force_t_per_m']:8.3f} t/m "
                  f"energy={r['A_half_energy_t_per_m']:8.3f} t/m", flush=True)
    for box in (60.0, 120.0):
        v = [res[(box, h)]["A_half_force_t_per_m"] for h in (1.0, 0.5, 0.25)]
        p, lim = richardson(*v)
        print(f"box={box:5.1f} h->0: order={p:.3f} limit={lim:.3f} t/m", flush=True)
    print("R-series at h=1.0:", [(b, round(res[(b, 1.0)]["A_half_force_t_per_m"], 3))
                                for b in (60.0, 120.0, 240.0)], flush=True)
    v60 = res[(60.0, 0.25)]["A_half_force_t_per_m"]
    v120 = res[(120.0, 0.25)]["A_half_force_t_per_m"]
    print(f"box 60 -> 120 at h=0.25: {v60:.3f} -> {v120:.3f} t/m "
          f"({v120 / v60 - 1.0:+.3%})", flush=True)