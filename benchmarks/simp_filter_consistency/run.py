"""P218-N04 -- SIMP density-filter and sensitivity-consistency qualification.

Runs the four acceptance studies of campaign node P218-N04 on the
``tensorfem.simp_topology`` kernels and writes machine-readable evidence
(JSON), figures (PNG) and the numbers-only tables of the report.

Phases (each writes ``runs/P218-N04/phase_<name>.json``):

``c1``       checkerboard negative control + filtered designs (24x12) and the
             mesh sweep (20x10 / 40x20 / 60x30, cone radius = 4 h).
``c2``       analytic vs central-difference sensitivities, 100 random
             elements x delta = 1e-4, 1e-5, 1e-6, for the cone filter, the
             Helmholtz/PDE filter and the (wrong) sensitivity-filter heuristic.
``c3``       mesh independence with a fixed physical filter radius.
``c4``       topology complexity vs filter radius.
``kernels``  cone vs Helmholtz filter operator equivalence / conservation.
``figures``  render the PNG figures from the phase JSON files.
``report``   assemble ``result.json`` + the markdown number tables.

Usage
-----
    python run.py --phase all --out docs/assets/simp-filter-consistency
    python run.py --phase c1            # single phase (used for chunked runs)

The script is deterministic (fixed seeds) and needs no cluster; the full
sequence runs in a few minutes on one core pair.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import tensorfem.simp_topology as st  # noqa: E402

torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

RUNS = REPO / "runs" / "P218-N04"
DELTAS = (1e-4, 1e-5, 1e-6)


def dkey(delta: float) -> str:
    """JSON key of a finite-difference step (1e-4 -> '0.0001')."""
    return f"{delta:g}"
L, H = 2.0, 1.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def panel(nx: int, ny: int, filter_type: str, radius: float, *,
          max_iter: int = 250, tol: float = 1e-4, vf: float = 0.5) -> dict:
    cfg = st.SIMPConfig(E0=1.0, E_min=1e-9, nu=0.3, p=3.0, thickness=1.0,
                        volume_frac=vf, r_min=radius, max_iter=max_iter, tol=tol,
                        filter_type=filter_type, solver="sparse")
    t0 = time.time()
    out = st.run_simp_design(nx, ny, config=cfg)
    dt = time.time() - t0
    rho = out["result"].rho
    rho_t = out["result"].rho_tilde
    design = st.topology_complexity(rho, nx, ny)
    physical = st.topology_complexity(rho_t, nx, ny)
    return {
        "nx": nx, "ny": ny, "h": L / nx, "filter_type": filter_type,
        "radius": radius, "iters": out["result"].n_iterations,
        "converged": bool(out["result"].converged),
        "compliance": float(out["compliance"]),
        "volume_fraction": float(out["volume_fraction"]),
        "design_metrics": design, "physical_metrics": physical,
        "hf_design": hf_fraction(rho, nx, ny), "hf_physical": hf_fraction(rho_t, nx, ny),
        "wall_time_s": round(dt, 2),
        "_rho": rho, "_rho_tilde": rho_t, "_nodes": out["nodes"],
        "_elements": out["elements"], "_forces": out["forces"],
        "_fixed": out["fixed_dofs"],
    }


def compliance_of(p: dict, rho_field: torch.Tensor, filter_type: str = "none",
                  radius: float = 0.0) -> float:
    """Compliance of an arbitrary density field on the mesh of ``p``."""
    nodes, elements = p["_nodes"], p["_elements"]
    n_dofs = 2 * nodes.shape[0]
    cfg = st.SIMPConfig(E0=1.0, E_min=1e-9, nu=0.3, p=3.0, thickness=1.0,
                        filter_type=filter_type, r_min=radius, solver="sparse")
    ke0 = st.compute_element_stiffness_base(nodes, elements, cfg.E0, cfg.nu,
                                           cfg.thickness, cfg.plane, torch.float64)
    filt = st.build_density_filter(nodes, elements, radius, filter_type, torch.float64)
    rho_t = filt.apply(rho_field)
    ke = st.simp_interpolate(rho_t, ke0, cfg.E_min, cfg.E0, cfg.p)
    K = st.assemble_global_stiffness_sparse(elements, ke, n_dofs)
    u = st.solve_system_sparse(K, p["_forces"], p["_fixed"])
    return float(p["_forces"] @ u)


def hf_fraction(rho: torch.Tensor, nx: int, ny: int, cut: float = 0.5) -> float:
    """Share of the (mean-removed) spectral energy above ``cut * Nyquist``."""
    g = rho.reshape(ny, nx).double()
    g = g - g.mean()
    p = torch.fft.fft2(g).abs() ** 2
    ky = torch.fft.fftfreq(ny).abs() * 2.0
    kx = torch.fft.fftfreq(nx).abs() * 2.0
    band = torch.maximum(ky[:, None], kx[None, :])
    tot = p[band > 0].sum()
    return float(p[band >= cut].sum() / tot) if float(tot) > 0 else 0.0


def strip_internal(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def dump(name: str, payload: dict) -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    path = RUNS / f"phase_{name}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[{name}] wrote {path.relative_to(REPO)}")


def load(name: str) -> dict:
    return json.loads((RUNS / f"phase_{name}.json").read_text())


# ---------------------------------------------------------------------------
# C1 -- checkerboard suppression
# ---------------------------------------------------------------------------

def phase_c1() -> None:
    out = {"criterion": "C1", "reference_mesh": [24, 12], "radius": 0.25,
           "cases": {}, "sweep": []}
    for ft, r in (("none", 0.0), ("linear", 0.25), ("helmholtz", 0.25)):
        p = panel(24, 12, ft, r)
        out["cases"][ft] = strip_internal(p)
        RUNS.mkdir(parents=True, exist_ok=True)
        torch.save({"rho": p["_rho"], "rho_tilde": p["_rho_tilde"]},
                   RUNS / f"field_24x12_{ft}.pt")
        print(f"  {ft:9s} C={p['compliance']:8.3f} "
              f"CI_phys={p['physical_metrics']['checkerboard_index']:.4f} "
              f"hf_phys={p['hf_physical']:.4f} t={p['wall_time_s']}s", flush=True)

    dump("c1", out)


def phase_c1_sweep() -> None:
    out = {"criterion": "C1-mesh-sweep", "sweep": []}
    # mesh sweep with the mesh-scaled radius R = 4 h (as in the campaign)
    for nx, ny in ((20, 10), (40, 20), (60, 30)):
        h = L / nx
        radius = 4 * h
        row = {"nx": nx, "ny": ny, "h": h, "radius": radius}
        none_panel = None
        for ft in ("none", "linear", "helmholtz"):
            p = panel(nx, ny, ft, 0.0 if ft == "none" else radius,
                      max_iter=150)  # sweep only: same designs, half the cost
            if ft == "none":
                none_panel = p
            row[ft] = {
                "compliance": p["compliance"],
                "ci_physical": p["physical_metrics"]["checkerboard_index"],
                "hf_physical": p["hf_physical"],
                "perimeter_physical": p["physical_metrics"]["perimeter"],
                "volume_fraction": p["volume_fraction"],
                "iters": p["iters"],
            }
            print(f"  {nx}x{ny} {ft:9s} C={p['compliance']:8.3f} "
                  f"CI={row[ft]['ci_physical']:.4f} hf={row[ft]['hf_physical']:.4f}", flush=True)
        # stiffness inflation of the pathological design: the no-filter optimum
        # re-interpreted through the very filter it was optimised without
        c_reg = compliance_of(none_panel, none_panel["_rho"], "linear", radius)
        row["regularisation_penalty"] = c_reg / row["none"]["compliance"]
        row["reinterpreted_compliance"] = c_reg
        print(f"  {nx}x{ny} regularisation penalty = {row['regularisation_penalty']:.2f}x",
              flush=True)
        out["sweep"].append(row)
    dump("c1_sweep", out)


def phase_c1_summary() -> None:
    out = load("c1")
    out["sweep"] = load("c1_sweep")["sweep"]
    none_c = out["cases"]["none"]["compliance"]
    cone = out["cases"]["linear"]
    pde = out["cases"]["helmholtz"]
    out["summary"] = {
        "ci_none": out["cases"]["none"]["physical_metrics"]["checkerboard_index"],
        "ci_cone": cone["physical_metrics"]["checkerboard_index"],
        "ci_pde": pde["physical_metrics"]["checkerboard_index"],
        "hfreac_none": out["cases"]["none"]["hf_physical"],
        "hfreac_cone": cone["hf_physical"],
        "hfreac_pde": pde["hf_physical"],
        "suppression_ratio_cone": out["cases"]["none"]["physical_metrics"]["checkerboard_index"]
        / cone["physical_metrics"]["checkerboard_index"],
        "suppression_ratio_pde": out["cases"]["none"]["physical_metrics"]["checkerboard_index"]
        / pde["physical_metrics"]["checkerboard_index"],
        "compliance_none": none_c,
        "compliance_cone": cone["compliance"],
        "compliance_pde": pde["compliance"],
        "compliance_ratio_cone": cone["compliance"] / none_c,
        "compliance_ratio_pde": pde["compliance"] / none_c,
        "perimeter_none": out["cases"]["none"]["physical_metrics"]["perimeter"],
        "perimeter_cone": cone["physical_metrics"]["perimeter"],
        "perimeter_pde": pde["physical_metrics"]["perimeter"],
    }
    dump("c1", out)


# ---------------------------------------------------------------------------
# C2 -- sensitivity consistency
# ---------------------------------------------------------------------------

def phase_c2() -> None:
    proto = panel(24, 12, "none", 0.0, max_iter=1)
    nodes, elements = proto["_nodes"], proto["_elements"]
    centroids = nodes[elements.long()].mean(dim=1)
    rho = 0.5 + 0.25 * torch.sin(2 * math.pi * centroids[:, 0] / L) \
        * torch.cos(3 * math.pi * centroids[:, 1] / H)
    rho = rho.clamp(0.2, 0.8)

    out = {"criterion": "C2", "mesh": [24, 12], "radius": 0.25,
           "n_elements_sampled": 100, "seed": 42, "cases": {}}
    for ft in ("linear", "helmholtz", "sensitivity"):
        cfg = st.SIMPConfig(r_min=0.25, filter_type=ft, solver="sparse",
                            max_iter=1, tol=1e-4, volume_frac=0.5)
        t0 = time.time()
        res = st.validate_sensitivity_fd(nodes, elements, proto["_forces"],
                                        proto["_fixed"], rho, cfg,
                                        n_samples=100, delta=DELTAS, seed=42)
        dt = time.time() - t0
        per_delta = {}
        for d in DELTAS:
            sel = [r for r in res if r["delta"] == d]
            per_delta[dkey(d)] = {
                "n": len(sel),
                "max_abs_error": max(r["abs_error"] for r in sel),
                "median_abs_error": statistics.median(r["abs_error"] for r in sel),
                "max_rel_error": max(r["relative_error"] for r in sel),
                "median_rel_error": statistics.median(r["relative_error"] for r in sel),
                "clamped": sum(r["clamped"] for r in sel),
                "max_abs_gradient": max(abs(r["analytical_sensitivity"]) for r in sel),
                "roundoff_product": max(r["abs_error"] for r in sel) * d,
            }
        out["cases"][ft] = {"per_delta": per_delta, "wall_time_s": round(dt, 2),
                            "samples": res}
        print(f"  {ft}: " + " ".join(
            f"d={d:g}:abs={per_delta[dkey(d)]['max_abs_error']:.2e},"
            f"rel={per_delta[dkey(d)]['max_rel_error']:.2e}" for d in DELTAS), flush=True)
    # Scale test: the absolute tolerance of C2 is not scale invariant.  With
    # the load scaled by 1/10 the compliance is O(0.5) and the finite-
    # difference error must drop by 10^2 (err ~ eps_C/delta, eps_C ~ s^2).
    proto2 = panel(24, 12, "none", 0.0, max_iter=1)
    scaled = {}
    for ft in ("linear", "helmholtz"):
        cfg = st.SIMPConfig(r_min=0.25, filter_type=ft, solver="sparse",
                            max_iter=1, tol=1e-4, volume_frac=0.5)
        res = st.validate_sensitivity_fd(nodes, elements, proto2["_forces"] * 0.1,
                                        proto2["_fixed"], rho, cfg,
                                        n_samples=100, delta=DELTAS, seed=42)
        scaled[ft] = {dkey(d): {
            "max_abs_error": max(r["abs_error"] for r in res if r["delta"] == d),
            "max_rel_error": max(r["relative_error"] for r in res if r["delta"] == d),
            "max_abs_gradient": max(abs(r["analytical_sensitivity"])
                                    for r in res if r["delta"] == d),
        } for d in DELTAS}
        print(f"  scaled(0.1) {ft}: " + " ".join(
            f"d={d:g}:abs={scaled[ft][dkey(d)]['max_abs_error']:.2e}" for d in DELTAS),
            flush=True)
    out["scaled_load"] = {"load_scale": 0.1, "cases": scaled,
                          # unit-load error / scaled-load error: must be ~1/s^2
                          "scale_law_ratio": {
                              ft: {dkey(d): out["cases"][ft]["per_delta"][dkey(d)]["max_abs_error"]
                                   / scaled[ft][dkey(d)]["max_abs_error"]
                                   for d in DELTAS} for ft in scaled}}
    dump("c2", out)


# ---------------------------------------------------------------------------
# C3 -- mesh independence (fixed physical radius)
# ---------------------------------------------------------------------------

def phase_c3() -> None:
    out = {"criterion": "C3", "radius": 0.2, "meshes": [], "unfiltered": []}
    for nx, ny in ((24, 12), (36, 18), (48, 24), (60, 30)):
        p = panel(nx, ny, "linear", 0.2, max_iter=250)
        out["meshes"].append(strip_internal(p))
        print(f"  {nx}x{ny} C={p['compliance']:8.3f} vf={p['volume_fraction']:.5f} "
              f"t={p['wall_time_s']}s", flush=True)
    for nx, ny in ((24, 12), (36, 18), (48, 24)):
        p = panel(nx, ny, "none", 0.0, max_iter=250)
        out["unfiltered"].append(strip_internal(p))
        print(f"  no-filter {nx}x{ny} C={p['compliance']:8.3f} "
              f"CI={p['physical_metrics']['checkerboard_index']:.4f}", flush=True)
    cs = [m["compliance"] for m in out["meshes"]]
    vfs = [m["volume_fraction"] for m in out["meshes"]]
    out["summary"] = {
        "compliances": cs,
        "volume_fractions": vfs,
        "max_volume_deviation": max(abs(v - 0.5) for v in vfs),
        "max_compliance_deviation_vs_finest": max(abs(c - cs[-1]) for c in cs) / cs[-1],
        "max_compliance_step_change": max(abs(b - a) / a for a, b in zip(cs, cs[1:])),
        "unfiltered_compliances": [m["compliance"] for m in out["unfiltered"]],
        "unfiltered_max_step_change": max(
            abs(b["compliance"] - a["compliance"]) / a["compliance"]
            for a, b in zip(out["unfiltered"], out["unfiltered"][1:])),
    }
    dump("c3", out)


# ---------------------------------------------------------------------------
# C4 -- complexity vs filter radius
# ---------------------------------------------------------------------------

def phase_c4() -> None:
    radii = (0.10, 0.15, 0.20, 0.30)
    out = {"criterion": "C4", "mesh": [36, 18], "radii": list(radii), "cases": {}}
    for ft in ("linear", "helmholtz"):
        rows = []
        for r in radii:
            p = panel(36, 18, ft, r, max_iter=200)
            rows.append({
                "radius": r,
                "compliance": p["compliance"],
                "perimeter": p["physical_metrics"]["perimeter"],
                "perimeter_design": p["design_metrics"]["perimeter"],
                "ci": p["physical_metrics"]["checkerboard_index"],
                "gray_fraction": p["physical_metrics"]["gray_fraction"],
                "n_components": p["physical_metrics"]["n_components"],
                "hf": p["hf_physical"],
            })
            print(f"  {ft:9s} R={r:.2f} C={p['compliance']:8.2f} "
                  f"peri={rows[-1]['perimeter']:.0f} CI={rows[-1]['ci']:.4f}", flush=True)
        peri = [r["perimeter"] for r in rows]
        ci = [r["ci"] for r in rows]
        comp = [r["compliance"] for r in rows]
        out["cases"][ft] = {
            "rows": rows,
            "perimeter_monotone_decreasing": all(b < a for a, b in zip(peri, peri[1:])),
            "perimeter_non_increasing": all(b <= a for a, b in zip(peri, peri[1:])),
            "perimeter_relative_drop": (max(peri) - min(peri)) / max(peri),
            "ci_monotone_decreasing": all(b < a for a, b in zip(ci, ci[1:])),
            "compliance_monotone_increasing": all(b > a for a, b in zip(comp, comp[1:])),
        }
    dump("c4", out)


# ---------------------------------------------------------------------------
# filter operator study (cone vs Helmholtz PDE)
# ---------------------------------------------------------------------------

def phase_kernels() -> None:
    proto = panel(20, 10, "none", 0.0, max_iter=1)
    nodes, elements = proto["_nodes"], proto["_elements"]
    ne = elements.shape[0]
    out = {"radius": 0.2, "cases": {}}
    for r in (0.1, 0.2, 0.3):
        cone = st.build_density_filter(nodes, elements, r, "linear")
        pde = st.build_density_filter(nodes, elements, r, "helmholtz")
        centroids = nodes[elements.long()].mean(dim=1)
        smooth = 0.5 + 0.3 * torch.sin(6 * centroids[:, 0]) * torch.cos(4 * centroids[:, 1])
        imp = torch.zeros(ne, dtype=torch.float64)
        imp[ne // 2 + 5] = 1.0
        a, b = pde.apply(smooth), cone.apply(smooth)
        ai, bi = pde.apply(imp), cone.apply(imp)
        out["cases"][f"{r:g}"] = {
            "smooth_relL2": float((a - b).norm() / b.norm()),
            "smooth_max_abs": float((a - b).abs().max()),
            "smooth_corr": float(torch.corrcoef(torch.stack([a, b]))[0, 1]),
            "impulse_peak_pde": float(ai.max()),
            "impulse_peak_cone": float(bi.max()),
            "impulse_relL2": float((ai - bi).norm() / bi.norm()),
            "volume_error_pde": abs(float(a.sum() - smooth.sum())) / float(smooth.sum()),
            "volume_error_cone": abs(float(b.sum() - smooth.sum())) / float(smooth.sum()),
            "uniform_error_pde": float((pde.apply(torch.full((ne,), 0.42, dtype=torch.float64))
                                        - 0.42).abs().max()),
            "pde_length_scale": pde.r,
        }
        print(f"  R={r}: smooth relL2={out['cases'][f'{r:g}']['smooth_relL2']:.4f} "
              f"corr={out['cases'][f'{r:g}']['smooth_corr']:.5f}", flush=True)
    # adjoint (transpose) check of both operators
    gen = torch.Generator().manual_seed(5)
    rho = 0.5 + 0.2 * torch.rand(ne, dtype=torch.float64, generator=gen)
    dc = torch.randn(ne, dtype=torch.float64, generator=gen)
    adj = {}
    for ft in ("linear", "helmholtz"):
        f = st.build_density_filter(nodes, elements, 0.2, ft)
        an = f.sensitivity(dc, rho)
        eps = 1e-7
        worst = 0.0
        for idx in range(0, ne, 7):
            rp, rm = rho.clone(), rho.clone()
            rp[idx] += eps
            rm[idx] -= eps
            fd = float(((f.apply(rp) - f.apply(rm)) / (2 * eps)) @ dc)
            worst = max(worst, abs(fd - float(an[idx])) / max(abs(float(an[idx])), 1e-14))
        adj[ft] = worst
        print(f"  adjoint {ft}: worst rel {worst:.2e}", flush=True)
    out["adjoint_worst_rel"] = adj
    dump("kernels", out)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def phase_figures(outdir: Path, dpi: int = 150) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    outdir.mkdir(parents=True, exist_ok=True)
    c1, c2, c3, c4, kern = load("c1"), load("c2"), load("c3"), load("c4"), load("kernels")

    # --- Fig 1: density fields + mid-row cut
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.4))
    names = {"none": "no filter", "linear": "cone filter (R=0.25)",
             "helmholtz": "Helmholtz PDE filter (R=0.25)"}
    nx, ny = c1["reference_mesh"]
    for col, ft in enumerate(("none", "linear", "helmholtz")):
        field = torch.load(RUNS / f"field_24x12_{ft}.pt")
        rho_t = field["rho_tilde"].reshape(ny, nx).numpy()
        ci = c1["cases"][ft]["physical_metrics"]["checkerboard_index"]
        im = axes[0, col].imshow(rho_t, origin="lower", cmap="viridis", vmin=0, vmax=1,
                                 aspect="auto")
        axes[0, col].set_title(f"{names[ft]}\nCI = {ci:.4f}")
        axes[0, col].set_xlabel("element i")
        axes[0, col].set_ylabel("element j")
        plt.colorbar(im, ax=axes[0, col], fraction=0.046)
        mid = rho_t[ny // 2]
        axes[1, col].plot(np.arange(nx), mid, "o-", ms=3, lw=1)
        axes[1, col].set_ylim(-0.05, 1.05)
        axes[1, col].set_title("mid-row cut $\\tilde\\rho$(i, ny/2)")
        axes[1, col].set_xlabel("element i")
        axes[1, col].set_ylabel("density")
        axes[1, col].grid(alpha=0.3)
    fig.suptitle("C1: checkerboard suppression on the 24x12 cantilever "
                 "(physical field $\\tilde\\rho$)" if False else
                 "C1 checkerboard control, 24x12 cantilever, physical density field",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_density_fields.png", dpi=dpi)
    plt.close(fig)

    # --- Fig 2: FD vs analytic
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ft, title in ((axes[0], "linear", "cone filter"),
                          (axes[1], "helmholtz", "Helmholtz PDE filter")):
        for d, marker in zip(DELTAS, ("o", "s", "^")):
            sel = [r for r in c2["cases"][ft]["samples"] if r["delta"] == d]
            x = [abs(r["analytical_sensitivity"]) for r in sel]
            y = [max(r["abs_error"], 1e-18) for r in sel]
            ax.loglog(x, y, marker, ms=3.5, alpha=0.6, label=f"$\\delta$={d:g}")
        ax.axhline(1e-4, color="k", ls="--", lw=1, label="C2 tolerance 1e-4")
        ax.set_xlabel("|analytic sensitivity|")
        ax.set_ylabel("|FD - analytic|")
        ax.set_title(f"C2 sensitivity consistency, {title} (100 elements)")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_sensitivity_fd.png", dpi=dpi)
    plt.close(fig)

    # --- Fig 3: mesh independence
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    h = [m["h"] for m in c3["meshes"]]
    c = [m["compliance"] for m in c3["meshes"]]
    ax[0].semilogx(h, c, "o-", label="filtered (R=0.2)")
    hu = [m["h"] for m in c3["unfiltered"]]
    cu = [m["compliance"] for m in c3["unfiltered"]]
    ax[0].semilogx(hu, cu, "s--", label="no filter")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("element size h")
    ax[0].set_ylabel("final compliance C")
    ax[0].set_title("C3 compliance vs mesh")
    ax[0].grid(alpha=0.3, which="both")
    ax[0].legend()
    vf = [m["volume_fraction"] for m in c3["meshes"]]
    ax[1].plot(range(len(vf)), vf, "o-")
    ax[1].axhline(0.5, color="k", ls="--", lw=1)
    ax[1].set_xticks(range(len(vf)))
    ax[1].set_xticklabels([f"{m['nx']}x{m['ny']}" for m in c3["meshes"]])
    ax[1].set_ylim(0.49, 0.51)
    ax[1].set_ylabel("volume fraction")
    ax[1].set_title("C3 volume constraint (target 0.5)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_mesh_independence.png", dpi=dpi)
    plt.close(fig)

    # --- Fig 4: complexity vs radius
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax2 = ax.twinx()
    for ft, marker in (("linear", "o"), ("helmholtz", "s")):
        rows = c4["cases"][ft]["rows"]
        ax.plot([r["radius"] for r in rows], [r["perimeter"] for r in rows],
                marker + "-", label=f"perimeter ({ft})")
        ax2.plot([r["radius"] for r in rows], [r["ci"] for r in rows],
                 marker + "--", alpha=0.6, label=f"CI ({ft})")
    ax.set_xlabel("filter radius R")
    ax.set_ylabel("physical perimeter [element edges]")
    ax2.set_ylabel("checkerboard index")
    ax.set_title("C4 topology complexity vs filter radius (36x18)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_radius_sensitivity.png", dpi=dpi)
    plt.close(fig)

    # --- Fig 5: filter kernels
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    radii = sorted(kern["cases"], key=float)
    ax[0].plot([float(r) for r in radii],
               [kern["cases"][r]["smooth_relL2"] for r in radii], "o-")
    ax[0].set_xlabel("filter radius R")
    ax[0].set_ylabel("relative $L_2$ difference")
    ax[0].set_title("Helmholtz vs cone filter, smooth field")
    ax[0].grid(alpha=0.3)
    ax[1].plot([float(r) for r in radii],
               [kern["cases"][r]["volume_error_pde"] for r in radii], "o-", label="PDE")
    ax[1].plot([float(r) for r in radii],
               [kern["cases"][r]["volume_error_cone"] for r in radii], "s--", label="cone")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("filter radius R")
    ax[1].set_ylabel("relative volume error")
    ax[1].set_title("volume conservation of both filters")
    ax[1].grid(alpha=0.3, which="both")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(outdir / "fig5_filter_kernels.png", dpi=dpi)
    plt.close(fig)
    print(f"[figures] wrote 5 PNGs into {outdir}")


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------

def phase_report(outdir: Path) -> None:
    c1, c2, c3, c4, kern = load("c1"), load("c2"), load("c3"), load("c4"), load("kernels")
    s1, s3 = c1["summary"], c3["summary"]
    verdict = {
        "C1_checkerboard_index_unfiltered_gt_0.3": s1["ci_none"] > 0.3,
        "C1_checkerboard_index_filtered_lt_0.05":
            s1["ci_cone"] < 0.05 and s1["ci_pde"] < 0.05,
        "C1_suppression_ratio_ge_5":
            s1["suppression_ratio_cone"] >= 5.0 and s1["suppression_ratio_pde"] >= 5.0,
        "C1_compliance_ratio_ge_5": False,  # measured 1.0 < r < 5 -> falsified
        "C2_rel_error_lt_1pct_all_deltas": all(
            c2["cases"][ft]["per_delta"][dkey(d)]["max_rel_error"] < 1e-2
            for ft in ("linear", "helmholtz") for d in DELTAS),
        "C2_abs_error_le_1e-4_at_1e-4_and_1e-5": all(
            c2["cases"][ft]["per_delta"][dkey(d)]["max_abs_error"] <= 1e-4
            for ft in ("linear", "helmholtz") for d in (1e-4, 1e-5)),
        "C2_abs_error_le_1e-4_at_1e-6_cone":
            c2["cases"]["linear"]["per_delta"][dkey(1e-6)]["max_abs_error"] <= 1e-4,
        "C2_abs_error_le_1e-4_at_1e-6_pde":
            c2["cases"]["helmholtz"]["per_delta"][dkey(1e-6)]["max_abs_error"] <= 1e-4,
        "C2_abs_error_le_1e-4_all_deltas_scaled_load": all(
            c2["scaled_load"]["cases"][ft][dkey(d)]["max_abs_error"] <= 1e-4
            for ft in ("linear", "helmholtz") for d in DELTAS),
        "C2_roundoff_scale_law": all(
            50.0 < c2["scaled_load"]["scale_law_ratio"][ft][dkey(d)]
            for ft in ("linear", "helmholtz") for d in (1e-5, 1e-6)),
        "C2_sensitivity_filter_heuristic_rejected":
            c2["cases"]["sensitivity"]["per_delta"][dkey(1e-4)]["max_rel_error"] > 0.05,
        "C3_volume_deviation_lt_2pct": s3["max_volume_deviation"] < 0.02,
        "C3_compliance_deviation_lt_3pct": s3["max_compliance_deviation_vs_finest"] < 0.03,
        # cone filter: strictly decreasing perimeter; Helmholtz/PDE: one
        # plateau (158 -> 158), i.e. only non-increasing -- reported as such.
        "C4_perimeter_strictly_monotone_cone":
            c4["cases"]["linear"]["perimeter_monotone_decreasing"],
        "C4_perimeter_non_increasing_both":
            c4["cases"]["linear"]["perimeter_monotone_decreasing"]
            and c4["cases"]["helmholtz"]["perimeter_non_increasing"],
        "C4_ci_monotone": c4["cases"]["linear"]["ci_monotone_decreasing"]
            and c4["cases"]["helmholtz"]["ci_monotone_decreasing"],
        "kernel_adjoint_exact": all(v < 1e-6 for v in kern["adjoint_worst_rel"].values()),
        "kernel_pde_matches_cone_lt_5pct": all(
            v["smooth_relL2"] < 0.05 for v in kern["cases"].values()),
    }
    # round-off floor of the audit: eps_C = |err| * delta (flat across deltas)
    floors = {ft: [c2["cases"][ft]["per_delta"][dkey(d)]["roundoff_product"] for d in DELTAS]
              for ft in ("linear", "helmholtz", "sensitivity")}
    result = {
        "node": "P218-N04",
        "fd_roundoff_floor": {
            "per_filter": floors,
            "eps_C_median": statistics.median(
                floors["linear"] + floors["helmholtz"]),
            "delta_at_which_1e-4_is_reached":
                statistics.median(floors["linear"] + floors["helmholtz"]) / 1e-4,
        },
        "repository": "TensorFEM",
        "module": "src/tensorfem/simp_topology.py",
        "tests": "tests/test_simp_filter_consistency.py",
        "study": "benchmarks/simp_filter_consistency/run.py",
        "verdict": verdict,
        "criteria": {
            "C1": s1, "C2": {ft: c2["cases"][ft]["per_delta"]
                             for ft in ("linear", "helmholtz", "sensitivity")},
            "C3": s3, "C4": {ft: {k: v for k, v in c4["cases"][ft].items() if k != "rows"}
                             for ft in ("linear", "helmholtz")},
        },
        "c4_rows": {ft: c4["cases"][ft]["rows"] for ft in c4["cases"]},
        "mesh_sweep": c1["sweep"],
        "kernels": kern,
    }
    result["mesh_family"] = c3["meshes"]
    result["_c2_scale_ratio"] = c2["scaled_load"]["scale_law_ratio"]
    result["mesh_family_unfiltered"] = c3["unfiltered"]
    (REPO / "benchmarks" / "simp_filter_consistency").mkdir(parents=True, exist_ok=True)
    (REPO / "benchmarks" / "simp_filter_consistency" / "result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print("[report] wrote benchmarks/simp_filter_consistency/result.json")
    # paper-level markdown report, numbers injected from result.json
    import report_md  # local module: benchmarks/simp_filter_consistency/report_md.py
    doc = REPO / "docs" / "P218-N04-simp-filter-consistency.md"
    doc.write_text(report_md.build(result, "assets/simp-filter-consistency"))
    print(f"[report] wrote {doc.relative_to(REPO)}")
    for k, v in verdict.items():
        print(f"   {'PASS' if v else 'FAIL'}  {k}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "c1", "c1_sweep", "c1_summary", "c2", "c3", "c4",
                             "kernels", "figures", "report"])
    ap.add_argument("--out", default=str(REPO / "docs" / "assets" / "simp-filter-consistency"))
    ap.add_argument("--dpi", type=int, default=150, help="figure raster resolution")
    args = ap.parse_args()
    outdir = Path(args.out)
    if args.phase in ("all", "c1"):
        phase_c1()
        phase_c1_summary()
    if args.phase == "c1_sweep":
        phase_c1_sweep()
    if args.phase == "c1_summary":
        phase_c1_summary()
    if args.phase in ("all", "c2"):
        phase_c2()
    if args.phase in ("all", "c3"):
        phase_c3()
    if args.phase in ("all", "c4"):
        phase_c4()
    if args.phase in ("all", "kernels"):
        phase_kernels()
    if args.phase in ("all", "figures"):
        phase_figures(outdir, dpi=args.dpi)
    if args.phase in ("all", "report"):
        phase_report(outdir)


if __name__ == "__main__":
    main()