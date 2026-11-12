"""Element-level rank census for ``cylindrical_shell4`` and its integration variants.

Why this file exists
--------------------
``benchmarks/verified/fem_shell_modal`` records two honest FAILs: the assembled
cylinder spectrum contains rotation-only modes and its lowest physical mode is
50 % below the Donnell (DMV) oracle.  Both symptoms are properties of the
*element*, so they are decidable without re-running the eigenvalue ladder: a
four-node shell element owns 24 DOFs and therefore exactly **6** zero-energy
modes (three rigid translations, three rigid rotations).  Every extra null
vector is a mechanism produced by reduced integration: a direction the element
does not resist, which makes the assembled spectrum mesh-dependent and too soft.

This script prints and stores that census for the production rule and for four
alternative integration rules, so the attribution claim in
``docs/CASE-FEM-SHELL-MODAL-mechanism-attribution.md`` is reproducible in
seconds instead of a full modal ladder.

It deliberately does **not** decide which variant is *correct*; it only decides
rank sufficiency (a necessary condition).  Accuracy is judged in the doc from
the modal ladder numbers.

Usage
-----
    python benchmarks/verified/fem_shell_modal/rank_census.py            # print
    python benchmarks/verified/fem_shell_modal/rank_census.py --write    # + rank_census.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.tensorfem.cylindrical_shell4 import cylindrical_shell4_stiffness

RIGID_MODES = 6
SHEAR_FACTOR = 5.0 / 6.0
RANK_TOL = 1e-9          # relative to the largest eigenvalue
DRILLING_FACTOR = 1e-3


def _element(x0: float = 75.0, dx: float = 25.0, dtheta: float = math.pi / 48) -> torch.Tensor:
    return torch.tensor([[x0, 0.0], [x0 + dx, 0.0],
                         [x0 + dx, dtheta], [x0, dtheta]], dtype=torch.float64)


def _variant_stiffness(x_theta, radius, young, poisson, thickness,
                       *, membrane: str, shear: str) -> torch.Tensor:
    """Same kinematics as the production element, different integration rules.

    ``membrane``/``shear`` in {"1pt", "2x2", "mitc"}; bending is always 2x2.
    """
    dtype = x_theta.dtype
    R = torch.as_tensor(float(radius), dtype=dtype)
    E = torch.as_tensor(float(young), dtype=dtype)
    nu = torch.as_tensor(float(poisson), dtype=dtype)
    t = torch.as_tensor(float(thickness), dtype=dtype)
    xy = torch.stack((x_theta[:, 0], R * x_theta[:, 1]), dim=1)
    D0 = E / (1 - nu ** 2) * torch.stack((
        torch.stack((torch.ones_like(nu), nu, torch.zeros_like(nu))),
        torch.stack((nu, torch.ones_like(nu), torch.zeros_like(nu))),
        torch.stack((torch.zeros_like(nu), torch.zeros_like(nu), (1 - nu) / 2))))
    Db, Dm = D0 * t ** 3 / 12, D0 * t
    Ds = SHEAR_FACTOR * E / (2 * (1 + nu)) * t * torch.eye(2, dtype=dtype)

    def fields(xi, eta):
        N = xy.new_tensor([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta),
                           (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)]) / 4
        nat = xy.new_tensor([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)],
                             [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]]) / 4
        J = xy.T @ nat
        det = torch.linalg.det(J)
        d = nat @ torch.linalg.inv(J)
        Bm = torch.zeros((3, 24), dtype=dtype)
        Bb = torch.zeros_like(Bm)
        Bs = torch.zeros((2, 24), dtype=dtype)
        for a in range(4):
            j = 6 * a
            dxa, dsa = d[a]
            Bm[0, j] = dxa
            Bm[1, j + 1] = dsa
            Bm[1, j + 2] = N[a] / R
            Bm[2, j] = dsa
            Bm[2, j + 1] = dxa
            Bb[0, j + 4] = dxa
            Bb[1, j + 3] = -dsa
            Bb[2, j + 4] = dsa
            Bb[2, j + 3] = -dxa
            Bs[0, j + 2] = dxa
            Bs[0, j + 4] = N[a]
            Bs[1, j + 2] = dsa
            Bs[1, j + 1] = -N[a] / R
            Bs[1, j + 3] = -N[a]
        return det, Bm, Bb, Bs

    g = 3.0 ** -0.5
    pts = ((-g, -g), (g, -g), (g, g), (-g, g))
    K = torch.zeros((24, 24), dtype=dtype)
    for xi, eta in pts:
        det, Bm, Bb, _ = fields(xi, eta)
        K += Bb.T @ Db @ Bb * det
        if membrane == "2x2":
            K += Bm.T @ Dm @ Bm * det
    if membrane == "1pt":
        det0, Bm0, _, _ = fields(0.0, 0.0)
        K += Bm0.T @ Dm @ Bm0 * det0 * 4.0

    if shear == "1pt":
        det0, _, _, Bs0 = fields(0.0, 0.0)
        K += Bs0.T @ Ds @ Bs0 * det0 * 4.0
    elif shear == "2x2":
        for xi, eta in pts:
            det, _, _, Bs = fields(xi, eta)
            K += Bs.T @ Ds @ Bs * det
    elif shear == "mitc":
        BA, BC = fields(0.0, -1.0)[3], fields(0.0, 1.0)[3]
        BB, BD = fields(1.0, 0.0)[3], fields(-1.0, 0.0)[3]
        for xi, eta in pts:
            det = fields(xi, eta)[0]
            Bans = torch.zeros((2, 24), dtype=dtype)
            Bans[0] = 0.5 * (1 - eta) * BA[0] + 0.5 * (1 + eta) * BC[0]
            Bans[1] = 0.5 * (1 + xi) * BB[1] + 0.5 * (1 - xi) * BD[1]
            K += Bans.T @ Ds @ Bans * det

    det0 = fields(0.0, 0.0)[0]
    ids = torch.arange(4, dtype=torch.long)
    rn = 6 * ids + 5
    lap = xy.new_tensor([[2.0, -1.0, 0.0, -1.0], [-1.0, 2.0, -1.0, 0.0],
                         [0.0, -1.0, 2.0, -1.0], [-1.0, 0.0, -1.0, 2.0]])
    K[rn[:, None], rn] += DRILLING_FACTOR * E * t * abs(det0) * 4.0 * lap

    T = torch.zeros((24, 24), dtype=dtype)
    for a, theta in enumerate(x_theta[:, 1]):
        one, zero = torch.ones_like(theta), torch.zeros_like(theta)
        basis = torch.stack((torch.stack((one, zero, zero)),
                             torch.stack((zero, torch.cos(theta), -torch.sin(theta))),
                             torch.stack((zero, torch.sin(theta), torch.cos(theta)))))
        T[6 * a:6 * a + 3, 6 * a:6 * a + 3] = basis
        T[6 * a + 3:6 * a + 6, 6 * a + 3:6 * a + 6] = basis
    return T.T @ K @ T


def census(K: np.ndarray, tol: float = RANK_TOL) -> dict:
    w = np.linalg.eigvalsh(K)
    null = np.abs(w) <= tol * np.abs(w).max()
    return {
        "nullspace_dimension": int(null.sum()),
        "spurious_mechanisms": int(null.sum()) - RIGID_MODES,
        "smallest_positive_eigenvalue": float(w[null.sum()]) if null.sum() < w.size else None,
        "largest_eigenvalue": float(w.max()),
    }


def run() -> dict:
    x_theta = _element()
    radius, young, poisson, thickness = 300.0, 3e6, 0.3, 3.0
    variants = {
        "production_membrane1pt_shear1pt": None,
        "membrane1pt_shear2x2": ("1pt", "2x2"),
        "membrane1pt_shear_mitc": ("1pt", "mitc"),
        "membrane2x2_shear1pt": ("2x2", "1pt"),
        "membrane2x2_shear2x2": ("2x2", "2x2"),
        "membrane2x2_shear_mitc": ("2x2", "mitc"),
    }
    out = {}
    for name, rules in variants.items():
        if rules is None:
            K = cylindrical_shell4_stiffness(x_theta, radius, young, poisson, thickness)
        else:
            K = _variant_stiffness(x_theta, radius, young, poisson, thickness,
                                   membrane=rules[0], shear=rules[1])
        entry = census(K.numpy())
        entry["rule"] = rules or ("1pt", "1pt")
        out[name] = entry
    return {
        "schema": "tensorfem.shell4-rank-census/1",
        "element": "cylindrical_shell4",
        "element_params": {"radius": radius, "young": young, "poisson": poisson,
                           "thickness": thickness, "chord_m": 25.0,
                           "dtheta": math.pi / 48},
        "rank_tolerance_relative": RANK_TOL,
        "rigid_modes_expected": RIGID_MODES,
        "variants": out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write rank_census.json next to this script")
    args = ap.parse_args()
    result = run()
    print(f"element={result['element']} rigid modes expected={RIGID_MODES}")
    for name, e in result["variants"].items():
        flag = "RANK-SUFFICIENT" if e["spurious_mechanisms"] <= 0 else "DEFICIENT"
        print(f"  {name:34s} null={e['nullspace_dimension']:2d} "
              f"spurious={e['spurious_mechanisms']:+2d}  {flag}")
    if args.write:
        path = Path(__file__).parent / "rank_census.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()