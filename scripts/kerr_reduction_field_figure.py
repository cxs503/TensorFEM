"""Deflection field figure for the exact two-field Kerr plate (report artifact)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorfem.foundation_plate_bending import kerr_equivalent_pasternak  # noqa: E402
from tensorfem.kerr_foundation import q4_kerr_foundation_stiffness  # noqa: E402
from tensorfem.plate import _shape, q4_mindlin_stiffness, structured_square_mesh  # noqa: E402

DTYPE = torch.float64
L, E, NU, T, Q0 = 1.0, 2.1e11, 0.3, 0.01, 1.0e4
D = E * T**3 / (12.0 * (1.0 - NU**2))
LAM = (math.pi / L) ** 2
K_U, K_L, G_P = 4.0e7, 2.0e7, 1.0e6
WL = [0, 3, 6, 9]
N = 24


def _load(nodes, elements):
    f = torch.zeros(3 * nodes.shape[0], dtype=DTYPE)
    gq = 1.0 / 3.0**0.5
    for conn in elements:
        xy = nodes[conn]
        fe = torch.zeros(12, dtype=DTYPE)
        for xi_f in (-gq, gq):
            for eta_f in (-gq, gq):
                Nn, dN = _shape(nodes.new_tensor(xi_f), nodes.new_tensor(eta_f))
                J = dN.T @ xy
                p = Nn @ xy
                q = Q0 * math.sin(math.pi * p[0].item() / L) * math.sin(math.pi * p[1].item() / L)
                fe[0::3] += Nn * q * torch.linalg.det(J)
        dofs = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        f[dofs] += fe
    return f


def _support(nodes):
    tol = L * 1e-12
    xe = (nodes[:, 0].abs() < tol) | ((nodes[:, 0] - L).abs() < tol)
    ye = (nodes[:, 1].abs() < tol) | ((nodes[:, 1] - L).abs() < tol)
    ix = torch.nonzero(xe, as_tuple=False).flatten()
    iy = torch.nonzero(ye, as_tuple=False).flatten()
    ie = torch.nonzero(xe | ye, as_tuple=False).flatten()
    return torch.cat((3 * ie, 3 * iy + 1, 3 * ix + 2)).unique(), ie


def solve(nodes, elements, two_field: bool, k_eq: float = 0.0, g_eq: float = 0.0):
    nnode = nodes.shape[0]
    ndof = 4 * nnode if two_field else 3 * nnode
    K = torch.zeros((ndof, ndof), dtype=DTYPE)
    f = torch.zeros(ndof, dtype=DTYPE)
    f[: 3 * nnode] = _load(nodes, elements)
    for conn in elements:
        xy = nodes[conn]
        dw = torch.stack((3 * conn, 3 * conn + 1, 3 * conn + 2), dim=1).reshape(-1)
        K[dw[:, None], dw] += q4_mindlin_stiffness(xy, E, NU, T)
        if two_field:
            kuw = q4_kerr_foundation_stiffness(xy, K_U, 0.0)
            kw1 = q4_kerr_foundation_stiffness(xy, K_U + K_L, G_P)
            K[dw[:, None], dw] += kuw
            d1 = 3 * nnode + conn
            K[d1[:, None], d1] += kw1[WL][:, WL]
            K[dw[WL][:, None], d1] += -kuw[WL][:, WL]
            K[d1[:, None], dw[WL]] += -kuw[WL][:, WL]
        else:
            K[dw[:, None], dw] += q4_kerr_foundation_stiffness(xy, k_eq, g_eq)
    fixed, iedge = _support(nodes)
    if two_field:
        fixed = torch.cat((fixed, 3 * nnode + iedge)).unique()
    free = torch.arange(ndof)
    free = free[~torch.isin(free, fixed)]
    u = torch.zeros(ndof, dtype=DTYPE)
    u[free] = torch.linalg.solve(K[free[:, None], free], f[free])
    return u[: 3 * nnode]


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nodes, elements = structured_square_mesh(N, L, dtype=DTYPE)
    u_exact = solve(nodes, elements, True)
    k_eq, g_eq = kerr_equivalent_pasternak(K_U, K_L, G_P)
    u_red = solve(nodes, elements, False, k_eq, g_eq)
    w_ex = u_exact[0::3].reshape(N + 1, N + 1).numpy()
    w_red = u_red[0::3].reshape(N + 1, N + 1).numpy()
    xs = nodes[:, 0].reshape(N + 1, N + 1)[:, 0].numpy()
    ys = nodes[:, 1].reshape(N + 1, N + 1)[0, :].numpy()

    fig = plt.figure(figsize=(13, 4.3))
    ax0 = fig.add_subplot(1, 3, 1, projection="3d")
    X, Y = torch.meshgrid(torch.tensor(xs), torch.tensor(ys), indexing="xy")
    ax0.plot_surface(X.numpy(), Y.numpy(), w_ex * 1e3, cmap="viridis", linewidth=0)
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("y (m)")
    ax0.set_zlabel("w (mm)")
    ax0.set_title("exact two-field Kerr, w field (G$_p$=1e6 N/m)")

    cf = fig.add_subplot(1, 3, 2)
    cs = cf.contourf(xs, ys, w_ex * 1e3, levels=14, cmap="viridis")
    cf.contour(xs, ys, w_ex * 1e3, levels=8, colors="k", linewidths=0.4)
    cf.set_aspect("equal")
    cf.set_xlabel("x (m)")
    cf.set_ylabel("y (m)")
    cf.set_title("contour, exact w (mm)")
    fig.colorbar(cs, ax=cf, shrink=0.85)

    cd = fig.add_subplot(1, 3, 3)
    import numpy as np

    with np.errstate(invalid="ignore", divide="ignore"):
        rel = (w_red / w_ex - 1.0) * 100.0
    rel[~np.isfinite(rel)] = np.nan  # supported edge: w -> 0, ratio undefined
    cs2 = cd.contourf(xs, ys, rel, levels=14, cmap="RdBu_r")
    cd.set_aspect("equal")
    cd.set_xlabel("x (m)")
    cd.set_ylabel("y (m)")
    cd.set_title("reduction error (w$_{red}$/w$_{exact}$-1), %")
    fig.colorbar(cs2, ax=cd, shrink=0.85)

    fig.tight_layout()
    out = Path(__file__).resolve().parent.parent / "docs" / "assets" / "kerr-foundation-field.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)
    print("centre exact", float(w_ex[N // 2, N // 2]), "centre reduced", float(w_red[N // 2, N // 2]))


if __name__ == "__main__":
    main()