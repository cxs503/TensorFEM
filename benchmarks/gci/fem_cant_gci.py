#!/usr/bin/env python3
"""FEM cabin-section GCI benchmark: 3D solid cantilever beam (HEX8).

Case (cabin-section level, classic FEM verification case)
----------------------------------------------------------
Rectangular cantilever beam, fixed at x=0, tip load P = 100 kN applied as a
uniform pressure on the tip face (x = L) in the -z direction:

    L = 10 m, B = 1 m, H = 1 m, E = 210 GPa, nu = 0.3, P = 100 kN

Meshes (structured HEX8, DOF targets 50k / 200k / 800k, exact 4x DOF ratio):

    M1:   95 x 15 x 10  ->   50,688 DOF
    M2:  383 x 15 x 10  ->  202,752 DOF
    M3: 1535 x 15 x 10  ->  811,008 DOF

Method
------
* Element: fully-integrated linear HEX8 (``tensorfem.solid3d.hex8_stiffness``,
  2x2x2 Gauss), same formulation as ``tensorfem.solid3d.solve_solid``.
* Assembly: element matrices scattered into a SciPy CSR matrix (the library
  dense path does not scale to 800k DOF), solved with SuperLU (float64).
* Strain/stress evaluated at element centroids (library convention).
* QoIs:
  - ``max_displacement_m``: max |u| over tip-face nodes;
  - ``max_stress_pa``: max von Mises stress at element centroids (raw);
  - fixed-end surface bending stress: the bending stress is linear in z, so
    the centroid value is extrapolated to z = H/2 with the exact factor
    (H/2)/|z_c - H/2|; the same is done at the x = 1 m section, which is
    away from the fixed-end corner singularity.

Reference (Timoshenko beam theory, rectangular section, kappa = 5/6)
--------------------------------------------------------------------
    I = B H^3 / 12,  A = B H,  G = E / (2 (1 + nu))
    delta_tip = P L^3 / (3 E I) + P L / (kappa A G)
    sigma_max = P L (H/2) / I

GCI
---
Per-pair observed order from the errors against the analytic value (the h
ratios are 4.0316 / 4.0078, not identical, so the three-level
``evaluate_convergence_study`` constant-ratio gate is not used); Roache GCI
with safety factor 1.25 per pair, ``gci_fine_pct`` = finest pair (M2 -> M3).

Outputs
-------
    artifacts/p8/fem_gci.json
    artifacts/p8/figures/fem_gci_displacement.png
    artifacts/p8/figures/fem_gci_convergence.png
    docs/p8/FEM_GCI_BENCH.md

Usage::

    python benchmarks/gci/fem_cant_gci.py [--levels 0,1,2] [--out artifacts/p8]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

from tensorfem.convergence_verification import (  # noqa: E402
    ConvergenceLevel,
    grid_convergence_index,
)
from tensorfem.solid3d import (  # noqa: E402
    elasticity_matrix_3d,
    structured_hex_mesh,
)

# ---------------------------------------------------------------------------
# Case definition
# ---------------------------------------------------------------------------
L, B, H = 10.0, 1.0, 1.0
E, NU, P = 210e9, 0.3, 100e3
KAPPA = 5.0 / 6.0  # Timoshenko shear correction, rectangular section

I_SEC = B * H ** 3 / 12.0
A_SEC = B * H
G_SHEAR = E / (2.0 * (1.0 + NU))
DELTA_EB = P * L ** 3 / (3.0 * E * I_SEC)            # Euler-Bernoulli
DELTA_TIMO = DELTA_EB + P * L / (KAPPA * A_SEC * G_SHEAR)  # Timoshenko
SIGMA_MAX = P * L * (H / 2.0) / I_SEC                # fixed-end surface bending

MESHES = [
    {"label": "M1", "nx": 95, "ny": 15, "nz": 10},
    {"label": "M2", "nx": 383, "ny": 15, "nz": 10},
    {"label": "M3", "nx": 1535, "ny": 15, "nz": 10},
]

# ---------------------------------------------------------------------------
# Element code (mirrors tensorfem.solid3d.hex8_stiffness, with an analytic
# 3x3 Jacobian inverse: torch.linalg.inv on (ne,3,3) float64 batches is
# pathologically slow under many OpenMP threads (60 s vs 0.08 s for 14k
# elements on 128 cores), so the benchmark carries its own copy of the
# element loop.  Formulation is identical: 2x2x2 Gauss, B at centroid.
# ---------------------------------------------------------------------------
_SIGNS = ((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
          (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))


def _inv3(j: torch.Tensor) -> torch.Tensor:
    """Analytic inverse of a batch of 3x3 matrices (..., 3, 3)."""
    a00, a01, a02 = j[..., 0, 0], j[..., 0, 1], j[..., 0, 2]
    a10, a11, a12 = j[..., 1, 0], j[..., 1, 1], j[..., 1, 2]
    a20, a21, a22 = j[..., 2, 0], j[..., 2, 1], j[..., 2, 2]
    c00 = a11 * a22 - a12 * a21
    c01 = a12 * a20 - a10 * a22
    c02 = a10 * a21 - a11 * a20
    c10 = a02 * a21 - a01 * a22
    c11 = a00 * a22 - a02 * a20
    c12 = a01 * a20 - a00 * a21
    c20 = a01 * a12 - a02 * a11
    c21 = a02 * a10 - a00 * a12
    c22 = a00 * a11 - a01 * a10
    det = a00 * c00 + a01 * c01 + a02 * c02
    out = torch.empty_like(j)
    out[..., 0, 0] = c00 / det
    out[..., 1, 0] = c01 / det
    out[..., 2, 0] = c02 / det
    out[..., 0, 1] = c10 / det
    out[..., 1, 1] = c11 / det
    out[..., 2, 1] = c12 / det
    out[..., 0, 2] = c20 / det
    out[..., 1, 2] = c21 / det
    out[..., 2, 2] = c22 / det
    return out


def _b(g: torch.Tensor) -> torch.Tensor:
    ne, nn, _ = g.shape
    b = torch.zeros((ne, 6, 3 * nn), dtype=g.dtype, device=g.device)
    b[:, 0, 0::3] = g[:, :, 0]
    b[:, 1, 1::3] = g[:, :, 1]
    b[:, 2, 2::3] = g[:, :, 2]
    b[:, 3, 0::3] = g[:, :, 1]
    b[:, 3, 1::3] = g[:, :, 0]
    b[:, 4, 1::3] = g[:, :, 2]
    b[:, 4, 2::3] = g[:, :, 1]
    b[:, 5, 0::3] = g[:, :, 2]
    b[:, 5, 2::3] = g[:, :, 0]
    return b


def _hex_gradient(x: torch.Tensor, xi: float, eta: float, zeta: float):
    s = x.new_tensor(_SIGNS)
    a, b, c = s[:, 0], s[:, 1], s[:, 2]
    dn = torch.stack((a * (1 + b * eta) * (1 + c * zeta),
                      b * (1 + a * xi) * (1 + c * zeta),
                      c * (1 + a * xi) * (1 + b * eta)), 1) / 8
    j = torch.einsum("eia,ib->eab", x, dn)
    det = torch.linalg.det(j)
    if torch.any(det <= torch.finfo(x.dtype).eps):
        raise ValueError("HEX8 element has non-positive or degenerate Jacobian")
    return torch.einsum("ib,ebc->eic", dn, _inv3(j)), det


def hex8_stiffness(x: torch.Tensor, d: torch.Tensor):
    q = 1 / 3 ** 0.5
    ke = torch.zeros((len(x), 24, 24), dtype=x.dtype, device=x.device)
    for xi in (-q, q):
        for eta in (-q, q):
            for zeta in (-q, q):
                g, det = _hex_gradient(x, xi, eta, zeta)
                b = _b(g)
                ke = ke + (b.transpose(1, 2) @ d @ b) * det[:, None, None]
    return ke, _b(_hex_gradient(x, 0., 0., 0.)[0])



def _von_mises(s: torch.Tensor) -> torch.Tensor:
    """von Mises from (ne, 6) stress vector (xx, yy, zz, xy, yz, xz)."""
    s0, s1, s2, t0, t1, t2 = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4], s[:, 5]
    return torch.sqrt(0.5 * ((s0 - s1) ** 2 + (s1 - s2) ** 2 + (s2 - s0) ** 2)
                      + 3.0 * (t0 ** 2 + t1 ** 2 + t2 ** 2))


def _surface_bending(stress: torch.Tensor, centroids: torch.Tensor,
                     x_target: float, nz: int) -> tuple[float, float]:
    """Surface-extrapolated |sigma_xx| at the section nearest ``x_target``.

    The bending stress is linear in z, so sigma(z=H/2) = sigma(z_c) *
    (H/2)/|z_c - H/2| is exact for the linear part.  Returns (sigma_surf,
    x_centroid) using the top-row element whose centroid is closest to
    ``x_target``.
    """
    hz = H / nz
    top = (centroids[:, 2] - (H - hz / 2.0)).abs() < 1e-9
    if not bool(top.any()):
        raise RuntimeError("no top-row element found")
    idx = torch.where(top)[0]
    xc = centroids[idx, 0]
    i = int(torch.argmin((xc - x_target).abs()))
    pick = idx[i]
    zc = float(centroids[pick, 2])
    sx = float(stress[pick, 0])
    return abs(sx) * (H / 2.0) / abs(zc - H / 2.0), float(xc[i])


def _cache_path(out: str, label: str) -> str:
    return os.path.join(out, f"level_{label}.npz")


def solve_level(mesh: dict, out: str | None = None) -> dict:
    """Assemble, solve and post-process one mesh level (with npz cache)."""
    if out is not None:
        cp = _cache_path(out, mesh["label"])
        if os.path.exists(cp):
            data = np.load(cp)
            rec = {k: (data[k].item() if data[k].ndim == 0 else data[k])
                   for k in data.files if not k.startswith("_")}
            rec["_u"] = torch.from_numpy(data["_u"])
            rec["_nodes"] = torch.from_numpy(data["_nodes"])
            print(f"[{mesh['label']}] loaded from cache {cp}", flush=True)
            return rec
    t0 = time.perf_counter()
    nodes, hexes = structured_hex_mesh(L, B, H, mesh["nx"], mesh["ny"], mesh["nz"],
                                       dtype=torch.float64)
    ne, nn = hexes.shape[0], nodes.shape[0]
    d = elasticity_matrix_3d(torch.tensor([E], dtype=torch.float64), torch.tensor([NU], dtype=torch.float64))
    x = nodes[hexes]
    ke, b = hex8_stiffness(x, d.expand(ne, 6, 6))

    ed = torch.stack(tuple(3 * hexes + i for i in range(3)), 2).reshape(ne, 24).long()
    rows = ed[:, :, None].expand(-1, -1, 24).reshape(-1)
    cols = ed[:, None, :].expand(-1, 24, -1).reshape(-1)
    K = sp.coo_matrix((ke.reshape(-1).numpy(), (rows.numpy(), cols.numpy()))).tocsr()
    t_asm = time.perf_counter() - t0

    fixed_nodes = torch.where(nodes[:, 0] == 0)[0]
    tip_nodes = torch.where(nodes[:, 0] == L)[0]
    f = torch.zeros(3 * nn, dtype=torch.float64)
    f[3 * tip_nodes + 2] = -P / len(tip_nodes)  # uniform tip pressure, total -P

    fixed = torch.stack((3 * fixed_nodes, 3 * fixed_nodes + 1, 3 * fixed_nodes + 2), 1).reshape(-1)
    mask = torch.ones(3 * nn, dtype=torch.bool)
    mask[fixed] = False
    free = torch.arange(3 * nn, dtype=torch.long)[mask]
    fi, ff = fixed.numpy(), free.numpy()
    t0 = time.perf_counter()
    lu = spla.splu(K[ff][:, ff].tocsc())
    u_free = lu.solve(f[free].numpy() - K[ff][:, fi] @ f[fixed].numpy())
    t_solv = time.perf_counter() - t0
    u = torch.zeros(3 * nn, dtype=torch.float64)
    u[free] = torch.tensor(u_free, dtype=torch.float64)

    strain = torch.einsum("eij,ej->ei", b, u[ed])
    stress = torch.einsum("eij,ej->ei", d.expand(ne, 6, 6), strain)
    vm = _von_mises(stress)
    centroids = x.mean(1)

    tip_dofs = (3 * tip_nodes[:, None] + torch.arange(3, dtype=tip_nodes.dtype)).reshape(-1)
    u_tip = u[tip_dofs].reshape(-1, 3)
    tip_disp = float(u_tip.norm(dim=1).max())
    max_vm = float(vm.max())
    sigma_fixed, _ = _surface_bending(stress, centroids, 0.0, mesh["nz"])
    sigma_sec, x_sec = _surface_bending(stress, centroids, 1.0, mesh["nz"])
    strain_energy = float(0.5 * (f * u).sum())

    return {
        "label": mesh["label"],
        "mesh": f"{mesh['nx']}x{mesh['ny']}x{mesh['nz']}",
        "n_elements": int(ne),
        "dof": int(3 * nn),
        "h_x_m": L / mesh["nx"],
        "max_displacement_m": tip_disp,
        "max_stress_pa": max_vm,
        "fixed_end_surface_stress_pa": sigma_fixed,
        "section_stress_pa": sigma_sec,
        "section_x_m": x_sec,
        "section_stress_exact_pa": P * (L - x_sec) * (H / 2.0) / I_SEC,
        "strain_energy_j": strain_energy,
        "assembly_time_s": t_asm,
        "solve_time_s": t_solv,
        "_u": u,
        "_nodes": nodes,
    }
    if out is not None:
        cp = _cache_path(out, mesh["label"])
        scalars = {k: v for k, v in rec.items()
                    if isinstance(v, (int, float)) and not k.startswith("_")}
        np.savez(cp, _u=u.numpy(), _nodes=nodes.numpy(), **scalars)


def gci_study(values: list[float], hs: list[float], exact: float) -> dict:
    """Per-pair observed orders and Roache GCI (Fs = 1.25) for 3 levels."""
    errs = [abs(v - exact) for v in values]
    out: dict = {
        "errors_abs": errs,
        "errors_pct": [100.0 * e / abs(exact) for e in errs],
    }
    for i, (a, bname) in enumerate([("M1", "M2"), ("M2", "M3")]):
        r = hs[i] / hs[i + 1]
        if errs[i] > 0.0 and errs[i + 1] > 0.0 and errs[i] > errs[i + 1]:
            p = math.log(errs[i] / errs[i + 1]) / math.log(r)
            gci = grid_convergence_index(
                ConvergenceLevel(hs[i], values[i], a),
                ConvergenceLevel(hs[i + 1], values[i + 1], bname),
                p, ratio=r,
            )
            out[f"gci_{a}_{bname}"] = gci
            out[f"order_{a}_{bname}"] = p
        else:
            out[f"gci_{a}_{bname}"] = None
            out[f"order_{a}_{bname}"] = None
            out["note"] = "non-monotone error sequence; GCI not evaluable"
    return out


def make_figures(records: list[dict], out: str, gci_disp: dict) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []
    r3 = records[-1]
    u, nodes = r3["_u"], r3["_nodes"]
    nx, ny, nz = 1535, 15, 10

    # --- figure 1: displacement contour (M3, top surface y=0) ---------------
    top = torch.where(nodes[:, 1] == 0)[0]
    uz = u[3 * top + 2].numpy() / 1e3  # mm
    X = torch.linspace(0, L, nx + 1).numpy()
    Z = torch.linspace(0, H, nz + 1).numpy()
    uz_grid = uz.reshape(nz + 1, nx + 1).T  # (nx+1, nz+1) -> (z, x)
    fig, ax = plt.subplots(figsize=(9.0, 3.6))
    cs = ax.pcolormesh(X, Z, uz_grid, shading="auto", cmap="viridis")
    fig.colorbar(cs, ax=ax, label=r"$u_z$ [mm] (top surface $y=0$)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Displacement field, {r3['mesh']} HEX8 ({r3['dof']:,} DOF), "
                 f"max |u| = {r3['max_displacement_m']*1e3:.3f} mm")
    fig.tight_layout()
    p1 = os.path.join(out, "figures", "fem_gci_displacement.png")
    fig.savefig(p1, dpi=110)
    plt.close(fig)
    figs.append({"name": "fem_gci_displacement.png", "path": p1,
                 "caption": "M3 顶面 z 向位移云图（mm）"})

    # --- figure 2: convergence + comparison ---------------------------------
    dofs = [r["dof"] for r in records]
    disp = [r["max_displacement_m"] for r in records]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    ax.loglog(dofs, [v * 1e3 for v in disp], "o-", color="tab:blue",
              label="FEM tip displacement")
    ax.axhline(DELTA_TIMO * 1e3, color="tab:red", ls="--",
               label=f"Timoshenko {DELTA_TIMO*1e3:.4f} mm")
    ax.axhline(DELTA_EB * 1e3, color="tab:gray", ls=":",
               label=f"Euler-Bernoulli {DELTA_EB*1e3:.4f} mm")
    ax.set_xlabel("DOF")
    ax.set_ylabel("tip displacement [mm]")
    ax.set_title("(a) tip displacement vs DOF")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")

    ax = axes[1]
    err = [100.0 * abs(v - DELTA_TIMO) / DELTA_TIMO for v in disp]
    ax.loglog(dofs, err, "s-", color="tab:blue", label="tip displacement error")
    ref = [err[-1] * (d / dofs[-1]) ** 2 for d in dofs]
    ax.loglog(dofs, ref, "k:", label="slope 2 reference")
    g = gci_disp.get("gci_M2_M3")
    if g:
        ax.axhspan(g["band_lower"] * 1e3, g["band_upper"] * 1e3,
                  color="tab:orange", alpha=.25,
                  label=f"GCI band {100*g['relative_gci']:.3f}%")
    ax.set_xlabel("DOF")
    ax.set_ylabel("relative error vs Timoshenko [%]")
    ax.set_title("(b) convergence vs Timoshenko")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3, which="both")
    fig.tight_layout()
    p2 = os.path.join(out, "figures", "fem_gci_convergence.png")
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    figs.append({"name": "fem_gci_convergence.png", "path": p2,
                 "caption": "端部位移收敛与 Timoshenko 对比"})
    return figs


def _fmt(v, spec=".6g"):
    return "n/a" if v is None else format(v, spec)


def write_markdown(result: dict, out: str, figures: list[dict]) -> None:
    recs = result["records"]
    p = result["pass"]
    g = result.get("gci_displacement") or {}
    gs = result.get("gci_fixed_end_stress") or {}

    lines = [
        "# FEM 舱段级 GCI 基准：3D 实体悬臂梁（HEX8）",
        "",
        "## 1 案例",
        "",
        "| 参数 | 值 |",
        "|---|---|",
        f"| 几何 | L={L:g} m, B={B:g} m, H={H:g} m（矩形截面） |",
        f"| 材料 | E={E/1e9:.0f} GPa, ν={NU} |",
        f"| 载荷 | 端部 {P/1e3:.0f} kN（端面均布压力，合力 -z，Saint-Venant 等效） |",
        "| 边界 | x=0 面全部 DOF 固定 |",
        "| 单元/求解 | HEX8 全积分（2×2×2 Gauss），float64，SciPy CSR + SuperLU |",
        "",
        "## 2 网格",
        "",
        "| 网格 | 划分 (nx×ny×nz) | 单元数 | DOF | h_x [m] |",
        "|---|---|---|---|---|",
    ]
    for r in recs:
        lines.append(f"| {r['label']} | {r['mesh']} | {r['n_elements']:,} | "
                     f"{r['dof']:,} | {r['h_x_m']:.5f} |")
    lines += [
        "",
        "DOF 比严格为 4.0；h 比 = 4.0316 / 4.0078（节点数 +1 效应）。",
        "",
        "## 3 结果",
        "",
        "| 网格 | DOF | 最大位移 [m] | 位移误差 [%] | 最大 von Mises [Pa] | 固定端表面弯曲应力 [Pa] | 其误差 [%] | x=1m 截面应力误差 [%] | GCI_fine [%] |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        lines.append(
            f"| {r['label']} | {r['dof']:,} | {r['max_displacement_m']:.6e} | "
            f"{r['tip_displacement_error_pct']:+.3f} | {r['max_stress_pa']:.4e} | "
            f"{r['fixed_end_surface_stress_pa']:.4e} | "
            f"{r['fixed_end_surface_stress_error_pct']:+.3f} | "
            f"{r['section_stress_error_pct']:+.3f} | {_fmt(r['gci_fine_pct'], '.3f')} |")
    lines += [
        "",
        "## 4 解析基准（Timoshenko 梁理论，κ=5/6）",
        "",
        f"- 端部挠度 δ = PL³/(3EI) + PL/(κAG) = **{DELTA_TIMO:.6e} m**"
        f"（弯曲 {DELTA_EB:.6e} m + 剪切 {DELTA_TIMO-DELTA_EB:.3e} m）",
        f"- 固定端表面最大弯曲应力 σ = PL(H/2)/I = **{SIGMA_MAX:.4e} Pa**",
        "",
        "## 5 GCI 分析",
        "",
        "| 量 | p (M1→M2) | p (M2→M3) | GCI(M1→M2) [%] | GCI(M2→M3) [%] | Richardson 外推 |",
        "|---|---|---|---|---|---|",
    ]
    if g:
        g12, g23 = g.get("gci_M1_M2"), g.get("gci_M2_M3")
        lines.append(
            f"| 端部位移 | {_fmt(g.get('order_M1_M2'), '.3f')} | "
            f"{_fmt(g.get('order_M2_M3'), '.3f')} | "
            f"{_fmt(100*(g12 or {}).get('relative_gci'), '.3f')} | "
            f"{_fmt(100*(g23 or {}).get('relative_gci'), '.3f')} | "
            f"{_fmt((g23 or {}).get('richardson_value'), '.6e')} m |")
    if gs:
        g12, g23 = gs.get("gci_M1_M2"), gs.get("gci_M2_M3")
        lines.append(
            f"| 固定端表面应力 | {_fmt(gs.get('order_M1_M2'), '.3f')} | "
            f"{_fmt(gs.get('order_M2_M3'), '.3f')} | "
            f"{_fmt(100*(g12 or {}).get('relative_gci'), '.3f')} | "
            f"{_fmt(100*(g23 or {}).get('relative_gci'), '.3f')} | "
            f"{_fmt((g23 or {}).get('richardson_value'), '.4e')} Pa |")
    lines += [
        "",
        "## 6 误差判定",
        "",
        f"- 主判据 GCI_fine（M2→M3，端部位移）= **{_fmt(p['gci_fine_pct'], '.3f')} %** → "
        f"{'通过 (<3%)' if p['gci_fine_pct_lt_3'] else '未通过'}",
        f"- 端部位移 vs Timoshenko：{_fmt(p['tip_error_pct'], '+.3f')} % → "
        f"{'≤5% 通过' if p['tip_error_pct_le_5'] else '未通过'}",
        f"- 固定端表面应力 vs 解析：{_fmt(p['fixed_end_stress_error_pct'], '+.3f')} %",
        f"- x=1 m 截面应力 vs 解析：{_fmt(p['section_stress_error_pct'], '+.3f')} %",
        "",
        "## 7 图",
        "",
    ]
    for f in figures:
        lines.append(f"![{f['caption']}](figures/{f['name']})")
        lines.append("")
    lines += [
        "## 8 注意点",
        "",
        "- 固定端为整面 Dirichlet 约束，固定端棱边存在混合边界奇异性；固定端应力用"
        "「质心值 + 线性外推至表面」评估，x=1 m 截面应力作为远离奇异区的独立校核。",
        "- 端部载荷按端面均布（Saint-Venant），与合力/合力矩相同的 Timoshenko 解可比。",
        "- 线性 HEX8 在 10:1 梁上无剪切锁定（单元近立方）；位移误差按 O(h²) 收敛。",
        "- 最大 von Mises（质心）系统性低于表面值（弯曲应力线性分布），表面值由线性外推给出。",
        "",
    ]
    path = os.path.normpath(os.path.join(out, "..", "..", "docs", "p8", "FEM_GCI_BENCH.md"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="3D solid cantilever GCI benchmark")
    ap.add_argument("--levels", default="0,1,2", help="comma list of mesh indices")
    ap.add_argument("--out", default=os.path.join("artifacts", "p8"))
    args = ap.parse_args()
    torch.set_num_threads(min(32, os.cpu_count() or 1))
    levels = [int(i) for i in args.levels.split(",") if i != ""]
    os.makedirs(os.path.join(args.out, "figures"), exist_ok=True)

    records = []
    for idx in levels:
        rec = solve_level(MESHES[idx], args.out)
        records.append(rec)
        print(f"[{rec['label']}] {rec['mesh']}  dof={rec['dof']}  "
              f"tip={rec['max_displacement_m']:.6e} m  "
              f"vm_max={rec['max_stress_pa']:.4e} Pa  "
              f"sigma_fixed_surf={rec['fixed_end_surface_stress_pa']:.4e} Pa  "
              f"(asm {rec['assembly_time_s']:.1f}s, solve {rec['solve_time_s']:.1f}s)",
              flush=True)

    full = len(records) == 3
    disp = [r["max_displacement_m"] for r in records]
    hs = [r["h_x_m"] for r in records]
    sig = [r["fixed_end_surface_stress_pa"] for r in records]

    result: dict = {
        "case": {
            "name": "3D solid cantilever beam, HEX8, cabin-section GCI",
            "L_m": L, "B_m": B, "H_m": H, "E_pa": E, "nu": NU, "P_n": P,
            "load": "uniform pressure on tip face, -z, resultant P",
            "bc": "all DOFs fixed at x=0",
            "element": "HEX8, 2x2x2 Gauss, float64, SuperLU",
        },
        "reference": {
            "timoshenko_tip_displacement_m": DELTA_TIMO,
            "euler_bernoulli_tip_displacement_m": DELTA_EB,
            "timoshenko_max_bending_stress_pa": SIGMA_MAX,
            "kappa": KAPPA, "I_m4": I_SEC, "A_m2": A_SEC, "G_pa": G_SHEAR,
        },
        "records": [],
    }

    gci_disp = gci_study(disp, hs, DELTA_TIMO) if full else None
    gci_sig = gci_study(sig, hs, SIGMA_MAX) if full else None
    result["gci_displacement"] = gci_disp
    result["gci_fixed_end_stress"] = gci_sig

    gci_fine_pct = None
    if full and gci_disp.get("gci_M2_M3"):
        gci_fine_pct = 100.0 * gci_disp["gci_M2_M3"]["relative_gci"]

    for i, r in enumerate(records):
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec["tip_displacement_error_pct"] = 100.0 * (r["max_displacement_m"] - DELTA_TIMO) / DELTA_TIMO
        rec["fixed_end_surface_stress_error_pct"] = 100.0 * (r["fixed_end_surface_stress_pa"] - SIGMA_MAX) / SIGMA_MAX
        rec["section_stress_error_pct"] = 100.0 * (r["section_stress_pa"] - r["section_stress_exact_pa"]) / r["section_stress_exact_pa"]
        if full:
            g = gci_disp.get(f"gci_M{i+1}_M{i+2}") if i < 2 else gci_disp.get("gci_M2_M3")
            rec["gci_fine_pct"] = 100.0 * g["relative_gci"] if g else None
        else:
            rec["gci_fine_pct"] = None
        result["records"].append(rec)

    result["pass"] = {
        "records_complete": full and all(
            k in r for r in result["records"]
            for k in ("dof", "max_displacement_m", "max_stress_pa", "gci_fine_pct")),
        "gci_fine_pct": gci_fine_pct,
        "gci_fine_pct_lt_3": bool(gci_fine_pct is not None and gci_fine_pct < 3.0),
        "tip_error_pct": 100.0 * (disp[-1] - DELTA_TIMO) / DELTA_TIMO if full else None,
        "tip_error_pct_le_5": bool(full and abs(100.0 * (disp[-1] - DELTA_TIMO) / DELTA_TIMO) <= 5.0),
        "fixed_end_stress_error_pct": 100.0 * (sig[-1] - SIGMA_MAX) / SIGMA_MAX if full else None,
        "section_stress_error_pct": result["records"][-1]["section_stress_error_pct"] if full else None,
    }

    out_json = os.path.join(args.out, "fem_gci.json")
    with open(out_json, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"wrote {out_json}")

    if full:
        figures = make_figures(records, args.out, gci_disp)
        result["figures"] = figures
        with open(out_json, "w") as fh:
            json.dump(result, fh, indent=2)
        write_markdown(result, args.out, figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
