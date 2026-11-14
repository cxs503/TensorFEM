"""Figures for the v5 plate added-mass cross-validation.

Reads ``xval_plate_added_mass_v5_artifacts/`` (one JSON per shard, plus the
aggregated ``evidence.json``) and writes the PNGs referenced by
``docs/xval_implicit_ibm_aitken_v5.md``.

Figure 0 recomputes a short coupled run locally to show the flow field
(vorticity of the D2Q9 box with the plate markers overlaid) -- the field is not
part of the shard records, so it is regenerated deterministically here.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from tensorfem.ibm_aitken_added_mass import (  # noqa: E402
    IBMImplicitFSI,
    LBMFluid2D,
    SpringMountedBody,
    analytic_added_mass,
    estimate_period,
    estimate_period_damped,
    estimate_period_fft,
    plate_markers,
)

OUT = os.path.join(HERE, "xval_plate_added_mass_v5_artifacts")
CONFIGS = ["c07_H288", "c10_H288", "c14_H288", "c20_H288",
           "c10_H144", "c10_H288_ratio1p0", "c10_H288_ratio1p0_fixed"]


def _load():
    recs = {}
    for k, name in enumerate(CONFIGS):
        p = os.path.join(OUT, f"config_{k:02d}_{name}.json")
        if os.path.exists(p):
            recs[name] = json.load(open(p))
    ev = None
    p = os.path.join(OUT, "evidence.json")
    if os.path.exists(p):
        ev = json.load(open(p))
    return recs, ev


def fig_vorticity(recs):
    """Flow field: vorticity of the D2Q9 box with the plate markers overlaid."""
    box, c = 144, 10.0
    m_a = analytic_added_mass(c, 1.0)
    m_s = m_a / 0.6
    k = (m_s + m_a) * (2.0 * math.pi / 600.0) ** 2
    body = SpringMountedBody(plate_markers((box / 2.0, box / 2.0), c), mass=m_s,
                             stiffness=k, y0=0.2)
    fluid = LBMFluid2D((box, box), omega=1.9)
    fsi = IBMImplicitFSI(body, fluid, dt=1.0, dx=1.0, ref_length=2.0 * c,
                         sub_tol=1e-6, max_sub_iter=30)
    for _ in range(150):            # a quarter period, peak velocity
        fsi.step()
    u = fluid.macroscopic(include_force=True)[1]
    ux, uy = u[..., 0], u[..., 1]
    dudy, dudx = np.gradient(ux)
    dvdy, dvdx = np.gradient(uy)
    curl = dvdx - dudy
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 5.0))
    im = ax[0].imshow(ux.T, origin="lower", cmap="RdBu_r",
                      vmin=-np.abs(ux).max(), vmax=np.abs(ux).max())
    ax[0].plot(body.markers()[:, 0], body.markers()[:, 1], "k-", lw=2.5)
    ax[0].set_title("u_x with plate markers (t = 150)")
    fig.colorbar(im, ax=ax[0], fraction=0.046)
    vmax = np.percentile(np.abs(curl), 99.5)
    im = ax[1].imshow(curl.T, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax[1].plot(body.markers()[:, 0], body.markers()[:, 1], "k-", lw=2.5)
    ax[1].set_title("vorticity $\\partial_x u_y-\\partial_y u_x$")
    fig.colorbar(im, ax=ax[1], fraction=0.046)
    for a in ax:
        a.set_xlabel("x [lattice]")
        a.set_ylabel("y [lattice]")
    fig.suptitle(f"implicit IBM + Aitken, plate c={c:g} dx, "
                 f"m_a/m_s=0.6, body y={body.y:+.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v5_field_vorticity.png"), dpi=130)
    plt.close(fig)


def fig_history(recs):
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for name in ("c07_H288", "c10_H288", "c14_H288", "c20_H288"):
        r = recs.get(name)
        if not r:
            continue
        y = np.asarray(r["y_history"])
        ax.plot(np.arange(y.size) * r["dt"], y, lw=1.0,
                label=f"c={r['half_width']:g} (H={r['box']})")
    ax.set_xlabel("t [lattice steps]")
    ax.set_ylabel("plate displacement y")
    ax.set_title("free vibration of the spring-mounted plate "
                 "(4 analytic periods)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v5_history.png"), dpi=130)
    plt.close(fig)


def fig_added_mass(recs, ev):
    sweep = [recs[n] for n in ("c07_H288", "c10_H288", "c14_H288", "c20_H288")
             if n in recs]
    if not sweep:
        return
    c = np.array([r["half_width"] for r in sweep])
    B = np.array([r["blockage"] for r in sweep])
    m_eff = np.array([r["added_mass_measured"] for r in sweep])
    m_lamb = np.array([r["m_a_analytic"] for r in sweep])
    m_smear = np.array([r["m_a_smeared"] for r in sweep])
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax[0].plot(c, m_lamb, "k--", label="$m_a=\\rho\\pi c^2$ (Lamb)")
    ax[0].plot(c, m_smear, "b:", label="$\\rho\\pi (c+1.5\\Delta)^2$")
    ax[0].plot(c, m_eff, "o-", label="measured (period fit)")
    ax[0].set_xlabel("plate half-width c [lattice]")
    ax[0].set_ylabel("added mass per unit length")
    ax[0].set_title(f"added mass, H=288, m_a/m_s=0.6")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(B, m_eff / m_lamb, "o-")
    ax[1].axhline(1.0, color="k", ls="--", lw=1.0, label="Lamb (unconfined)")
    model = (ev or {}).get("verdict", {}).get("model_form")
    if model:
        bb = np.linspace(0, B.max() * 1.05, 50)
        for r in sweep:
            pass
        pred = np.pi * (c + model["delta"]) ** 2 * (1.0 + model["beta"] * B)
        ax[1].plot(B, pred / m_lamb, "s--", alpha=0.7,
                   label=f"fit $\\delta$={model['delta']:.2f}, "
                         f"$\\beta$={model['beta']:.2f}")
    ax[1].set_xlabel("blockage B = (2c+1)/H")
    ax[1].set_ylabel("$m_{a,eff}/m_{a,Lamb}$")
    ax[1].set_title("smearing + finite-domain bias")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v5_added_mass.png"), dpi=130)
    plt.close(fig)


def fig_frequency(recs, ev):
    if not ev:
        return
    rows = ev["verdict"]["rows"]
    names = [r["config"] for r in rows]
    got = [r["f_ratio"] for r in rows]
    ref = [r["f_ratio_lamb"] for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(x - 0.2, ref, 0.4, label="analytic $\\sqrt{m_s/(m_s+m_a)}$")
    ax.bar(x + 0.2, got, 0.4, label="measured (implicit IBM + Aitken)")
    for xi, r in zip(x, rows):
        ax.text(xi + 0.2, got[list(x).index(xi) if False else list(x).index(xi)],
                f"{r['freq_err_pct']:+.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("c10_H288_", "c10_H288\n") for n in names],
                       fontsize=7)
    ax.set_ylabel("$f_{fsi}/f_{vac}$")
    ax.set_title("coupled frequency ratio vs analytic added mass "
                 "(labels: error, acceptance 15%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v5_frequency_ratio.png"), dpi=130)
    plt.close(fig)


def fig_window_bias(recs):
    """The v4 failure mode: legacy vs window-safe period as the record grows."""
    r = recs.get("c10_H288")
    if not r:
        return
    y = np.asarray(r["y_history"])
    dt = r["dt"]
    T_ref = r["period"]
    frac = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    lens = [int(min(len(y), round(f * T_ref / dt + 1))) for f in frac]
    legacy, safe = [], []
    for n in lens:
        seg = y[:n]
        p_old = estimate_period(seg, dt)
        if not np.isfinite(p_old):
            p_old = estimate_period_fft(seg, dt)
        legacy.append(p_old)
        fit = estimate_period_damped(seg, dt)
        safe.append(fit["period"] if fit["valid"] else np.nan)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(np.array(lens) * dt, np.array(legacy) / T_ref, "o-",
            label="legacy (zero-crossing / FFT)")
    ax.plot(np.array(lens) * dt, np.array(safe) / T_ref, "s-",
            label="window-safe fit (invalid records dropped)")
    ax.axhline(1.0, color="k", ls="--", lw=1.0, label="4-period reference")
    ax.axvline(300, color="r", ls=":", lw=1.2, label="v4 record length (300)")
    ax.set_xlabel("record length [time units]")
    ax.set_ylabel("$T_{est}/T_{ref}$")
    ax.set_title("period estimate vs record length (c=10, H=288)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v5_window_bias.png"), dpi=130)
    plt.close(fig)


def fig_subiter(recs):
    a = recs.get("c10_H288_ratio1p0")
    f = recs.get("c10_H288_ratio1p0_fixed")
    if not a or not f:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for r, label, style in ((a, "Aitken", "-"), (f, "fixed $\\omega=1$", "-")):
        its = np.asarray(r["sub_iter_history"])
        ax.plot(np.arange(its.size), its, style, lw=0.9, alpha=0.8,
                label=f"{label}: mean {its.mean():.2f}, max {its.max()}")
    ax.set_xlabel("time step")
    ax.set_ylabel("sub-iterations")
    ax.set_title("interface sub-iterations at $m_a/m_s=1.0$ (H=288, c=10)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "v5_subiterations.png"), dpi=130)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    recs, ev = _load()
    if "--field" in sys.argv:
        fig_vorticity(recs)
    if recs:
        fig_history(recs)
        fig_added_mass(recs, ev)
        fig_window_bias(recs)
        fig_subiter(recs)
    if ev:
        fig_frequency(recs, ev)
    print("figures ->", sorted(f for f in os.listdir(OUT) if f.endswith(".png")))


if __name__ == "__main__":
    main()