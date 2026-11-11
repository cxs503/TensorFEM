"""Figures for the CASE-NAFEMS-LE11 spheroidal-shell benchmark report.

Every plotted number comes from the JSON written by
``benchmarks/nafems_le11_ellipsoid.py``; nothing is transcribed by hand.
"""
from __future__ import annotations

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

GATE = 5.0      # percent


def _save(fig, figdir: str, name: str) -> str:
    os.makedirs(figdir, exist_ok=True)
    path = os.path.join(figdir, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_convergence(payload: dict, figdir: str) -> str:
    sweep = payload["sweep"]
    n = [s["facets"] for s in sweep]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(n, [100 * s["err_phi_max_scaled"] for s in sweep], "o-",
              label=r"max $|N_\varphi - N_\varphi^{ref}|/N_{scale}$")
    ax.loglog(n, [100 * s["err_theta_max_scaled"] for s in sweep], "s-",
              label=r"max $|N_\theta - N_\theta^{ref}|/N_{scale}$")
    ax.loglog(n, [100 * s["w_equator_err"] for s in sweep], "^-",
              label=r"$|w_{eq} - w_{eq}^{ref}|/w_{eq}^{ref}$")
    ax.axhline(GATE, color="crimson", ls="--", lw=1.2, label="5% gate")
    ref = [100 * sweep[1]["err_phi_max_scaled"] * (n[1] / x) for x in n]
    ax.loglog(n, ref, "k:", lw=1.0, label=r"$O(h)$ slope")
    ax.set_xlabel("meridional elements (pole $\\to$ equator)")
    ax.set_ylabel("error [%]")
    ax.set_title("Spheroid $a/b=2$: convergence, gate 5%")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    return _save(fig, figdir, "convergence.png")


def fig_meridian_fields(payload: dict, figdir: str) -> str:
    prof = payload["profile_128"]
    th = [math.degrees(d["theta"]) for d in prof]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.plot(th, [d["N_phi_ref"] for d in prof], "k-", lw=2.2, label="reference (statics)")
    ax.plot(th, [d["N_s_fe"] for d in prof], "o", ms=3.0, mfc="none",
            label="TensorFEM ring element (128 el.)")
    ax.set_xlabel(r"eccentric angle $\theta$ [deg]")
    ax.set_ylabel(r"$N_\varphi$  [p a]")
    ax.set_title(r"meridional membrane force: pole $\to$ equator")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(th, [d["N_theta_ref"] for d in prof], "k-", lw=2.2, label="reference")
    ax.plot(th, [d["N_theta_fe"] for d in prof], "o", ms=3.0, mfc="none", label="FE, 128 el.")
    ax.axhline(0.0, color="0.5", lw=0.8)
    ax.set_xlabel(r"eccentric angle $\theta$ [deg]")
    ax.set_ylabel(r"$N_\theta$  [p a]")
    ax.set_title("hoop force: tension at the pole, compression at the equator")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, figdir, "meridian_fields.png")


def fig_hoop_cloud(payload: dict, figdir: str) -> str:
    """Field plot of N_theta over the closed spheroid (half-meridian mirrored)."""
    gc = payload["fine_mesh_gauss"]
    a, b = payload["parameters"]["a_equatorial"], payload["parameters"]["b_polar"]
    xs, zs, vals = [], [], []
    for d in gc:
        r, phi, val = d["r"], d["phi"], d["N_theta_fe"]
        xs.append(r)
        zs.append(b * math.cos(d["theta"]))
        vals.append(val)
    # mirror: upper/lower half and both sides of the axis
    X, Z, V = [], [], []
    for x, z, v in zip(xs, zs, vals):
        for sx in (1.0, -1.0):
            for sz in (1.0, -1.0):
                X.append(sx * x)
                Z.append(sz * z)
                V.append(v)
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    cf = ax.tricontourf(X, Z, V, levels=24, cmap="coolwarm")
    cb = fig.colorbar(cf, ax=ax)
    cb.set_label(r"$N_\theta$  $[p\,a]$  (FE, 256 elements)")
    tt = [i * math.pi / 200 for i in range(201)]
    ax.plot([a * math.sin(t) for t in tt], [b * math.cos(t) for t in tt], "k-", lw=0.8)
    ax.plot([-a * math.sin(t) for t in tt], [b * math.cos(t) for t in tt], "k-", lw=0.8)
    ax.plot([a * math.sin(t) for t in tt], [-b * math.cos(t) for t in tt], "k-", lw=0.8)
    ax.plot([-a * math.sin(t) for t in tt], [-b * math.cos(t) for t in tt], "k-", lw=0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("r")
    ax.set_ylabel("z")
    ax.set_title("Hoop membrane force of the pressurised spheroid "
                 r"($b/a=0.5$): blue = compression at the equator")
    return _save(fig, figdir, "hoop_field.png")


def fig_equator_layer(payload: dict, figdir: str) -> str:
    """Zoom on the equatorial elements: mesh dependence of the discrete layer."""
    prof = payload["profile_128"]
    th = [math.degrees(d["theta"]) for d in prof]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0))
    ax = axes[0]
    ax.plot(th, [d["N_theta_ref"] for d in prof], "k-", lw=2.0, label="reference")
    ax.plot(th, [d["N_theta_fe"] for d in prof], "o", ms=2.5, mfc="none",
            label="FE, 128 el.")
    ax.set_xlim(60.0, 90.0)
    ax.set_xlabel(r"$\theta$ [deg]")
    ax.set_ylabel(r"$N_\theta$  [p a]")
    ax.set_title("equatorial zoom: 128 elements")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]
    sweep = payload["sweep"]
    n = [s["facets"] for s in sweep]
    ax.semilogx(n, [100 * s["N_theta_equator_err"] for s in sweep], "s-",
                label=r"$N_\theta$ at the equator")
    ax.semilogx(n, [100 * s["N_phi_equator_err"] for s in sweep], "o-",
                label=r"$N_\varphi$ at the equator")
    ax.semilogx(n, [100 * s["N_pole_err"] for s in sweep], "^-",
                label=r"$N_\varphi$ at the pole")
    ax.axhline(GATE, color="crimson", ls="--", lw=1.2, label="5% gate")
    ax.set_xlabel("meridional elements")
    ax.set_ylabel("relative error [%]")
    ax.set_title("point errors vs mesh: the equatorial hoop error plateaus")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, figdir, "equator_layer.png")


def fig_aspect(payload: dict, figdir: str) -> str:
    pts = payload["sensitivity"]["aspect_ratio"]
    ratio = [p["b_over_a"] for p in pts]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy(ratio, [100 * p["err_phi_max_scaled"] for p in pts], "o-",
                label=r"max $N_\varphi$ error (scaled)")
    ax.semilogy(ratio, [100 * p["err_theta_max_scaled"] for p in pts], "s-",
                label=r"max $N_\theta$ error (scaled)")
    ax.semilogy(ratio, [100 * p["w_equator_err"] for p in pts], "^-",
                label=r"$w_{eq}$ error")
    ax.axhline(GATE, color="crimson", ls="--", lw=1.2, label="5% gate")
    ax.axvline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"aspect ratio $b/a$   (oblate $<1$, sphere $=1$, prolate $>1$)")
    ax.set_ylabel("error [%]")
    ax.set_title("sensitivity to the spheroid aspect ratio (128 elements)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    return _save(fig, figdir, "aspect_sensitivity.png")


def make_figures(payload: dict, figdir: str) -> list[str]:
    return [
        fig_convergence(payload, figdir),
        fig_meridian_fields(payload, figdir),
        fig_hoop_cloud(payload, figdir),
        fig_equator_layer(payload, figdir),
        fig_aspect(payload, figdir),
    ]