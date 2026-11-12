"""HEX20 stress-recovery qualification against the exact 3-D plate solution.

Closes the open item of ``docs/plate-3d-exact-anchor.md`` §8.1 bullet 3: the
``tau_xz`` through-thickness profile had an analytic reference but **no
independent measure of the finite-element recovery error**.  This script measures
it, separating the three things that are usually conflated:

* discretisation error  -> stress at the 27 Gauss points (superconvergent sites),
  compared with the exact solution at the *same* physical points;
* recovery error        -> the same quantity sampled at the element boundary
  ``x = 0`` where the exact shear peaks, by (a) the pointwise element derivative
  and (b) least-squares nodal recovery + shape-function interpolation;
* Barlow extrapolation  -> Lagrange extrapolation from the three Gauss planes to
  the free edge, i.e. what a well-built post-processor should do.

Every number in ``docs/plate-3d-stress-recovery.md`` is read back from
``docs/assets/plate_3d_stress/evidence.json`` (written here) or from the
generated ``tables.md``; nothing is typed in by hand.

Run: ``python benchmarks/plate_3d_stress_recovery.py`` (~4 min, 4 meshes).
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

from tensorfem import hex20_stress_recovery as hsr
from tensorfem.plate_3d_exact import SSPlateExact3D, solve_hex20_ss_plate

A = B = 1.0
H = 0.1
YOUNG = 1.0e9
POISSON = 0.3
Q0 = 1.0e6
MESHES = ((4, 2), (6, 4), (8, 4))
ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "assets", "plate_3d_stress")
Z_SAMPLES = 41
_VOIGT = {0: "sxx", 1: "syy", 2: "szz", 3: "txy", 4: "tyz", 5: "txz"}


def _exact_components(exact: SSPlateExact3D, x, y, z):
    """Exact stresses in the repository's Voigt order ``[xx, yy, zz, xy, yz, xz]``."""
    sxx, syy, szz, txy, txz, tyz = exact.stress(x, y, z)
    return {"sxx": sxx, "syy": syy, "szz": szz, "txy": txy, "tyz": tyz, "txz": txz}


def _rel(err: torch.Tensor, scale: float) -> dict:
    return {
        "max_abs_err": float(err.abs().max()),
        "rms_err": float(torch.sqrt((err ** 2).mean())),
        "max_rel": float(err.abs().max()) / scale,
        "rms_rel": float(torch.sqrt((err ** 2).mean())) / scale,
    }


def _gauss_metrics(nodes, cells, u, exact) -> dict:
    g = hsr.hex20_gauss_stresses(nodes, cells, u, YOUNG, POISSON)
    coords, sig = g["coords"], g["stress"]
    x = coords[..., 0].reshape(-1)
    y = coords[..., 1].reshape(-1)
    z = coords[..., 2].reshape(-1) - 0.5 * H  # FE mesh spans z in [0, H]
    ex = _exact_components(exact, x, y, z)
    # distance from the nearest plate edge, used to separate a boundary layer
    edge_dist = torch.minimum(
        torch.minimum(x, A - x), torch.minimum(y, B - y)
    ) / H
    out = {}
    for comp, idx in (("sxx", 0), ("szz", 2), ("txz", 5), ("tyz", 4)):
        fe = sig[..., idx].reshape(-1)
        scale = float(ex[comp].abs().max())
        err = fe - ex[comp]
        out[comp] = _rel(err, scale)
        near_edge = edge_dist < 0.5
        if comp in ("txz", "tyz") and bool(near_edge.any()):
            out[comp]["near_edge_le_half"] = _rel(err[near_edge], scale)
            out[comp]["interior_gt_half"] = _rel(err[~near_edge], scale)
    out["n_gauss_points"] = int(x.numel())
    out["edge_distance_over_h_min"] = float(edge_dist.min())
    return out


def _profile_pointwise(nodes, cells, u, x, ys, zs) -> torch.Tensor:
    pts = [[x, ys, 0.5 * H + z] for z in zs.tolist()]
    return hsr.line_profile(nodes, cells, u, YOUNG, POISSON, pts)


def _profile_nodal(nodes, cells, u, x, ys, zs) -> torch.Tensor:
    nodal = hsr.hex20_nodal_stresses(nodes, cells, u, YOUNG, POISSON)["nodal"]
    pts = [[x, ys, 0.5 * H + z] for z in zs.tolist()]
    return torch.stack([hsr.interpolate_nodal(nodes, cells, nodal, p) for p in pts])


def _profile_metrics(fe: torch.Tensor, ex: dict, component: int) -> dict:
    fe_v = fe[..., component]
    ex_v = ex[_VOIGT[component]].reshape(-1)
    peak_ex = float(ex_v.abs().max())
    ipk_ex = int(ex_v.abs().argmax())
    ipk_fe = int(fe_v.abs().argmax())
    # value of the recovered field at the location of the *analytic* extremum:
    # this is the metric that answers "can a post-processor read the peak here?"
    fe_at_exact_peak = float(fe_v[ipk_ex])
    signed_peak_ex = float(ex_v[ipk_ex])
    # trapezoid integral over the sampled thickness
    integ_fe = float(torch.trapezoid(fe_v, dx=H / (len(fe_v) - 1)))
    integ_ex = float(torch.trapezoid(ex_v, dx=H / (len(fe_v) - 1)))
    mask = ex_v.abs() > 0.05 * peak_ex
    rel_max = float(((fe_v - ex_v).abs()[mask] / ex_v.abs()[mask]).max())
    return {
        "peak_fe": float(fe_v[ipk_fe]),
        "peak_exact": peak_ex,
        "peak_rel_error": fe_at_exact_peak / signed_peak_ex - 1.0,
        "signed_peak_exact": signed_peak_ex,
        "peak_value_at_exact_location": fe_at_exact_peak,
        "profile_max_over_exact_peak": float(fe_v[ipk_fe]) / peak_ex,
        "z_peak_over_h": float(ipk_fe) / (len(fe_v) - 1) - 0.5,
        "z_peak_over_h_exact": float(ipk_ex) / (len(fe_v) - 1) - 0.5,
        "integral_fe": integ_fe,
        "integral_exact": integ_ex,
        "integral_rel_error": integ_fe / integ_ex - 1.0,
        "max_pointwise_rel_error_above_5pct": rel_max,
        "l2_rel": float(torch.linalg.norm(fe_v - ex_v) / torch.linalg.norm(ex_v)),
    }


def _edge_spread(nodes, cells, u, point) -> dict:
    """Values recovered from every element that contains ``point`` (jump measure)."""
    vals = []
    for idx in range(len(cells)):
        sub = cells[idx:idx + 1]
        try:
            hsr.locate_hex20_element(nodes, sub, point)
        except ValueError:
            continue
        vals.append(float(hsr.tau_xz(
            hsr.pointwise_stress(nodes, sub, u, YOUNG, POISSON, point)["stress"])))
    return {"n_elements": len(vals), "min": min(vals), "max": max(vals),
            "relative_spread": (max(vals) - min(vals)) / abs(sum(vals) / len(vals))}


def _barlow_edge(nodes, cells, u, exact, y_natural: float, t_natural: float) -> dict:
    """Extrapolate the 3 Gauss values in ``r`` to the free edge ``r = -1``."""
    r_nodes = hsr.GAUSS_1D
    values = []
    for r in r_nodes:
        sig = hsr.hex20_stresses_at_natural(nodes, cells[:1], u, YOUNG, POISSON,
                                            [(r, y_natural, t_natural)])[0, 0]
        values.append(sig)
    values = torch.stack(values)
    target = torch.tensor(-1.0, dtype=values.dtype)
    weights = []
    for i, xi in enumerate(r_nodes):
        w = torch.tensor(1.0, dtype=values.dtype)
        for j, xj in enumerate(r_nodes):
            if i != j:
                w = w * (target - xj) / (xi - xj)
        weights.append(w)
    extrap = torch.einsum("i,ic->c", torch.stack(weights), values)
    coord = hsr.hex20_point_coordinates(nodes, cells[:1], (0.0, y_natural, t_natural))[0, 0]
    z_ex = float(coord[2]) - 0.5 * H
    ex = _exact_components(exact, coord[0].reshape(1), coord[1].reshape(1),
                           torch.tensor(z_ex).reshape(1))
    ex_txz = float(ex["txz"][0])
    near = hsr.hex20_stresses_at_natural(nodes, cells[:1], u, YOUNG, POISSON,
                                        [(r_nodes[0], y_natural, t_natural)])[0, 0]
    return {
        "point": [float(coord[0]), float(coord[1]), z_ex],
        "tau_xz_gauss_nearest": float(near[5]),
        "tau_xz_barlow_edge": float(extrap[5]),
        "tau_xz_exact": ex_txz,
        "gauss_rel_error": float(near[5]) / ex_txz - 1.0,
        "barlow_rel_error": float(extrap[5]) / ex_txz - 1.0,
    }


def run_case(nx: int, nz: int, exact: SSPlateExact3D) -> dict:
    t0 = time.time()
    sol = solve_hex20_ss_plate(A, B, H, YOUNG, POISSON, q0=Q0, nx=nx, ny=nx, nz=nz,
                               sparse=True)
    nodes, cells, u = sol["nodes"], sol["cells"], sol["displacement"]
    w_exact = exact.center_deflection()
    zs = torch.linspace(-0.5 * H, 0.5 * H, Z_SAMPLES, dtype=nodes.dtype)
    edge_pt = [0.0, 0.5 * B, 0.5 * H]
    centre_pt = [0.5 * A, 0.5 * B, 0.5 * H]

    ex_edge = _exact_components(exact, torch.zeros_like(zs), torch.full_like(zs, 0.5 * B), zs)
    fe_edge = _profile_pointwise(nodes, cells, u, 0.0, 0.5 * B, zs)
    fe_edge_nodal = _profile_nodal(nodes, cells, u, 0.0, 0.5 * B, zs)

    xs = torch.full_like(zs, 0.5 * A)
    ys = torch.full_like(zs, 0.5 * B)
    ex_centre = _exact_components(exact, xs, ys, zs)
    fe_centre = _profile_pointwise(nodes, cells, u, 0.5 * A, 0.5 * B, zs)

    # nearest Gauss plane in y and z to (y = b/2, z = mid-surface)
    elem_x0 = None
    for idx in range(len(cells)):
        sub = cells[idx:idx + 1]
        try:
            hsr.locate_hex20_element(nodes, sub, edge_pt)
        except ValueError:
            continue
        elem_x0 = sub
        break
    if elem_x0 is None:  # pragma: no cover
        raise RuntimeError("no element holds the edge probe point")
    xs_coord = hsr.hex20_point_coordinates(nodes, elem_x0, hsr.GAUSS_3D)[0]
    dist = ((xs_coord[:, 1] - 0.5 * B) ** 2 + (xs_coord[:, 2] - 0.5 * H) ** 2)
    nearest = hsr.GAUSS_3D[int(dist.argmin())]
    barlow = _barlow_edge(nodes, elem_x0, u, exact, nearest[1], nearest[2])
    barlow["gauss_natural_s_t"] = [nearest[1], nearest[2]]

    v_x = Q0 * exact.alpha / (exact.alpha ** 2 + exact.beta ** 2)
    edge_metrics = _profile_metrics(fe_edge, ex_edge, 5)
    nodal_metrics = _profile_metrics(fe_edge_nodal, ex_edge, 5)
    for m in (edge_metrics, nodal_metrics):
        m["peak_over_meanV_h"] = m["peak_value_at_exact_location"] / (v_x / H)
        m["mean_to_peak_shortfall"] = 1.0 - 1.0 / m["peak_over_meanV_h"]
    return {
        "nx": nx, "nz": nz, "n_dofs": sol["n_dofs"], "n_elements": sol["n_elements"],
        "h_over_h_element": 1.0 / nz,
        "centre_deflection": sol["centre_deflection"],
        "centre_deflection_exact": w_exact,
        "centre_deflection_rel_error": sol["centre_deflection"] / w_exact - 1.0,
        "shear_force_v_x": v_x,
        "tau_edge_pointwise": edge_metrics,
        "tau_edge_nodal_interp": nodal_metrics,
        "sigma_xx_centre": _profile_metrics(fe_centre, ex_centre, 0),
        "edge_spread": _edge_spread(nodes, cells, u, edge_pt),
        "barlow": barlow,
        "gauss": _gauss_metrics(nodes, cells, u, exact),
        "runtime_s": time.time() - t0,
        "_curves": {"zs": zs, "fe_edge": fe_edge, "fe_edge_nodal": fe_edge_nodal,
                    "ex_edge": ex_edge, "fe_centre": fe_centre, "ex_centre": ex_centre},
    }


def _figures(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q0 = Q0
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    r0 = results[0]
    zs = r0["_curves"]["zs"].tolist()
    ax.plot(zs, (r0["_curves"]["ex_edge"]["txz"] / q0).tolist(), "k-", lw=2.2,
            label="exact 3-D")
    for r in results:
        ax.plot(r["_curves"]["zs"].tolist(),
                (r["_curves"]["fe_edge"][..., 5] / q0).tolist(), "o--", ms=3,
                label=f"FE pointwise {r['nx']}x{r['nx']}x{r['nz']}")
    ax.set_xlabel("z / m (mid-surface = 0)")
    ax.set_ylabel(r"$\tau_{xz}/q_0$ at (x=0, y=b/2)")
    ax.axhline(0.0, color="0.7", lw=0.6)
    ax.legend(fontsize=7)
    ax.set_title("through-thickness shear at the plate edge")
    fig.tight_layout()
    fig.savefig(os.path.join(path, "fig1_tau_profile_edge.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    dofs = [r["n_dofs"] for r in results]
    ax.loglog(dofs, [abs(r["tau_edge_pointwise"]["peak_rel_error"]) for r in results],
              "s-", label="pointwise @ edge node")
    ax.loglog(dofs, [abs(r["tau_edge_nodal_interp"]["peak_rel_error"]) for r in results],
              "^-", label="nodal LSQ + interpolation")
    ax.loglog(dofs, [abs(r["barlow"]["barlow_rel_error"]) for r in results], "o-",
              label="Barlow extrapolation to edge")
    ax.loglog(dofs, [abs(r["gauss"]["txz"]["max_rel"]) for r in results], "d-",
              label="Gauss-point max")
    ax.axhline(0.03, color="r", ls=":", lw=1.2, label="3 % gate")
    ax.set_xlabel("degrees of freedom")
    ax.set_ylabel("relative error of the shear peak")
    ax.legend(fontsize=7)
    ax.set_title("recovery mode decides the error, not only the mesh")
    fig.tight_layout()
    fig.savefig(os.path.join(path, "fig2_recovery_convergence.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for r in results:
        ax.plot(r["_curves"]["zs"].tolist(),
                (r["_curves"]["ex_centre"]["sxx"] / q0).tolist(), "k-", lw=1.6)
        ax.plot(r["_curves"]["zs"].tolist(),
                (r["_curves"]["fe_centre"][..., 0] / q0).tolist(), "--", lw=1.1,
                label=f"FE {r['nx']}x{r['nx']}x{r['nz']}")
    ax.set_xlabel("z / m (mid-surface = 0)")
    ax.set_ylabel(r"$\sigma_{xx}/q_0$ at the plate centre")
    ax.legend(fontsize=7)
    ax.set_title("bending stress through the thickness (black = exact)")
    fig.tight_layout()
    fig.savefig(os.path.join(path, "fig3_sigma_xx_centre.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    labels, values = [], []
    for r in results:
        labels += [f"{r['nx']}x{r['nx']}x{r['nz']}\npoint", f"{r['nx']}x{r['nx']}x{r['nz']}\nnodal",
                   f"{r['nx']}x{r['nx']}x{r['nz']}\nbarlow"]
        values += [r["tau_edge_pointwise"]["peak_rel_error"] * 100,
                   r["tau_edge_nodal_interp"]["peak_rel_error"] * 100,
                   r["barlow"]["barlow_rel_error"] * 100]
    colors = ["C0", "C1", "C2"] * len(results)
    ax.bar(range(len(values)), values, color=colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, fontsize=5.5, rotation=90)
    ax.axhline(3.0, color="r", ls=":")
    ax.set_ylabel("shear-peak error / %")
    ax.set_title("signed shear-peak error by recovery mode")
    fig.tight_layout()
    fig.savefig(os.path.join(path, "fig4_recovery_bars.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    r = results[-1]
    keys = ["sxx", "szz", "txz", "tyz"]
    width = 0.2
    for i, k in enumerate(keys):
        axes[0].bar([i], [r["gauss"][k]["max_rel"] * 100], width=width, label=f"{k} max")
        axes[1].bar([i + 0.2], [r["gauss"][k]["rms_rel"] * 100], width=width, label=f"{k} rms")
    for ax in axes:
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.set_xlabel("stress component")
        ax.set_ylabel("relative error / %")
    axes[0].set_title("max |err| / max |exact|")
    axes[1].set_title("rms(err) / max |exact|")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"superconvergent (Gauss-point) stress accuracy, mesh {r['nx']}x{r['nx']}x{r['nz']}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(path, "fig5_gauss_error.png"), dpi=150)
    plt.close(fig)


def _tables(results) -> str:
    lines = ["### 表 1 — 剪切峰值 `tau_xz/q0`：恢复方式决定误差\n",
             "| 网格 | 自由度 | pointwise@边节点 | 误差 | nodal 插值 | 误差 | Barlow 外推 | 误差 | 精确值 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        lines.append(
            f"| {r['nx']}×{r['nx']}×{r['nz']} | {r['n_dofs']} | "
            f"{r['tau_edge_pointwise']['peak_fe']/Q0:.4f} | "
            f"{r['tau_edge_pointwise']['peak_rel_error']*100:+.2f} % | "
            f"{r['tau_edge_nodal_interp']['peak_fe']/Q0:.4f} | "
            f"{r['tau_edge_nodal_interp']['peak_rel_error']*100:+.2f} % | "
            f"{r['barlow']['tau_xz_barlow_edge']/Q0:.4f} | "
            f"{r['barlow']['barlow_rel_error']*100:+.2f} % | "
            f"{r['tau_edge_pointwise']['peak_exact']/Q0:.4f} |")
    lines.append("\n### 表 2 — 剖面形状与静力学一致性（x=0, y=b/2）\n")
    lines.append("| 网格 | 峰值位置 z/h | 峰值/(V_x/h) | 积分 ∫tau_xz dz | V_x | 积分误差 | L2 相对 | 逐点最大相对(>5% 峰值) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        m = r["tau_edge_pointwise"]
        lines.append(
            f"| {r['nx']}×{r['nx']}×{r['nz']} | {m['z_peak_over_h']:+.4f} | "
            f"{m['peak_over_meanV_h']:.4f} | {m['integral_fe']:.6e} | "
            f"{m['integral_exact']:.6e} | {m['integral_rel_error']*100:+.3f} % | "
            f"{m['l2_rel']*100:.2f} % | {m['max_pointwise_rel_error_above_5pct']*100:.2f} % |")
    lines.append("\n### 表 3 — 超收敛（Gauss 点）应力误差，最细网格\n")
    lines.append("| 分量 | max 相对误差 | rms 相对误差 | 近边界(<0.5h) max | 内部 max |")
    lines.append("|---|---:|---:|---:|---:|")
    r = results[-1]
    for k in ("sxx", "szz", "txz", "tyz"):
        g = r["gauss"][k]
        near = g.get("near_edge_le_half", {}).get("max_rel")
        inner = g.get("interior_gt_half", {}).get("max_rel")
        lines.append(
            f"| {k} | {g['max_rel']*100:.2f} % | {g['rms_rel']*100:.2f} % | "
            f"{(f'{near*100:.2f} %' if near is not None else '—')} | "
            f"{(f'{inner*100:.2f} %' if inner is not None else '—')} |")
    lines.append("\n### 表 4 — 位移与剪切静力学总览\n")
    lines.append("| 网格 | 单元 | w_c FE | w_c 精确 | w_c 误差 | ∫tau_xz dz 误差 | 运行 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['nx']}×{r['nx']}×{r['nz']} | {r['n_elements']} | "
            f"{r['centre_deflection']:.6e} | {r['centre_deflection_exact']:.6e} | "
            f"{r['centre_deflection_rel_error']*100:+.3f} % | "
            f"{r['tau_edge_pointwise']['integral_rel_error']*100:+.3f} % | "
            f"{r['runtime_s']:.0f} s |")
    return "\n".join(lines) + "\n"


def main() -> dict:
    os.makedirs(ASSET_DIR, exist_ok=True)
    exact = SSPlateExact3D(A, B, H, YOUNG, POISSON, q0=Q0)
    results = []
    for nx, nz in MESHES:
        print(f"[plate_3d_stress] mesh {nx}x{nx}x{nz} ...", flush=True)
        results.append(run_case(nx, nz, exact))
        print(f"   peak rel {results[-1]['tau_edge_pointwise']['peak_rel_error']*100:+.2f} %"
              f"  barlow {results[-1]['barlow']['barlow_rel_error']*100:+.2f} %"
              f"  ({results[-1]['runtime_s']:.0f} s)", flush=True)
    eigencheck = {
        "exact_peak_tau_over_q0": float(exact.stress(
            torch.tensor(0.0), torch.tensor(0.5 * B), torch.tensor(0.0))[4]) / Q0,
        "exact_peak_location_z_over_h": 0.0,
        "shear_force_v_x": Q0 * exact.alpha / (exact.alpha ** 2 + exact.beta ** 2),
        "centre_deflection_exact": exact.center_deflection(),
        "face_report": exact.face_report(),
        "equilibrium": exact.equilibrium_residual(),
    }
    payload = {
        "case": {"a": A, "b": B, "h": H, "young": YOUNG, "poisson": POISSON, "q0": Q0,
                 "a_over_h": A / H, "z_samples": Z_SAMPLES, "meshes": [list(m) for m in MESHES]},
        "exact_reference": eigencheck,
        "results": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
    }
    with open(os.path.join(ASSET_DIR, "evidence.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    _figures(results, ASSET_DIR)
    with open(os.path.join(ASSET_DIR, "tables.md"), "w") as fh:
        fh.write(_tables(results))
    print(f"[plate_3d_stress] wrote {ASSET_DIR}/evidence.json, tables.md, figures")
    return payload


if __name__ == "__main__":
    main()