"""Build the figures and the acceptance JSON for the FSI strong-coupling node.

Reads the machine-produced JSON files in ``artifacts/fsi_strong_coupling`` and
writes

* ``artifacts/fsi_strong_coupling/fig_*.png``  (matplotlib)
* ``docs/reports/P153W2-N03_fsi_lbm.json``     (criteria + numbers + paths)
* ``docs/reports/P153W2-N03_fsi_lbm.md``       (paper-style report)

No number is typed by hand: everything is read from the JSON artefacts.
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts", "fsi_strong_coupling")
DOCS = os.path.join(ROOT, "docs", "reports")
os.makedirs(DOCS, exist_ok=True)


def load(name):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


cyl12 = load("cylinder_D12.json")
cyl20 = load("cylinder_D20.json")
cyl12f = load("cylinder_D12_ulat005.json")
fsi = load("fsi2_D20_u0.05_local.json")
structure = load("structure.json")
interface = load("interface.json")


# ---------------------------------------------------------------------------
# Figure 1: cylinder 2D-2 validation
# ---------------------------------------------------------------------------
def fig_cylinder():
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                           gridspec_kw={"height_ratios": [1, 1]})
    for data, label in ((cyl12, "D=12 cells"), (cyl20, "D=20 cells")):
        if data is None:
            continue
        t = np.asarray(data["force_history"]["t"])
        cd = np.asarray(data["force_history"]["CD"])
        cl = np.asarray(data["force_history"]["CL"])
        ax[0].plot(t, cd, lw=0.8, label=f"{label}: <CD>={data['CD_mean']:.3f}")
        ax[1].plot(t, cl, lw=0.8, label=f"{label}: St={data['St']:.3f}")
    ref = cyl20["reference"] if cyl20 else {"CD": 3.23, "St": 0.30}
    ax[0].axhline(ref["CD"], color="k", ls="--", lw=1,
                  label=f"Turek-Hron 2D-2 CD={ref['CD']}")
    ax[0].set_ylabel("$C_D$")
    ax[0].set_title("Turek-Hron 2D-2 cylinder in channel, Re=100 (resolved LBM)")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("$C_L$")
    ax[1].set_xlabel("t [s]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(ART, "fig_cylinder_validation.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 2: coupled FSI-2 history
# ---------------------------------------------------------------------------
def fig_fsi():
    if fsi is None:
        return None
    h = fsi["history"]
    t = np.asarray(h["t"])
    fig, ax = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    ax[0].plot(t, np.asarray(h["tip"]) * 1e3, lw=0.9)
    ax[0].set_ylabel("tip $w$ [mm]")
    ax[0].set_title(f"Turek-Hron FSI-2 (coupled LBM-FE), cells/D={fsi['cells_per_D']}, "
                    f"u_lat={fsi['u_lat']}")
    ax[1].plot(t, np.asarray(h["drag"]), lw=0.9, label="cylinder drag")
    ax[1].plot(t, np.asarray(h["lift"]), lw=0.9, label="cylinder lift")
    ax[1].set_ylabel("force [N/m]")
    ax[1].legend(fontsize=8)
    ax[2].plot(t, np.asarray(h["sub_iterations"]), ".", ms=2, label="sub-iterations")
    ax2 = ax[2].twinx()
    ax2.semilogy(t, np.maximum(np.asarray(h["force_residual"]), 1e-16), "r-", lw=0.6,
                 label="force residual")
    ax[2].set_ylabel("sub-iterations")
    ax2.set_ylabel("interface force residual")
    ax[2].set_xlabel("t [s]")
    fig.tight_layout()
    path = os.path.join(ART, "fig_fsi2_history.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 3: energy ledger
# ---------------------------------------------------------------------------
def fig_energy():
    if fsi is None:
        return None
    h = fsi["history"]
    t = np.asarray(h["t"])
    dt = float(np.mean(np.diff(t)))
    fluid = np.asarray(h["fluid_ke"]) - np.asarray(h["fluid_ke"])[0]
    struct = np.asarray(h["struct_ke"]) + np.asarray(h["struct_pe"])
    p_in = np.asarray(h["inlet_power"])
    p_out = np.asarray(h["outlet_power"])
    visc = np.asarray(h["viscous"])
    work_in = np.cumsum(p_in - p_out) * dt
    visc_cum = np.cumsum(visc) * dt
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t, fluid, label="fluid KE - initial")
    ax[0].plot(t, struct, label="structure KE + strain")
    ax[0].set_ylabel("energy [J/m]")
    ax[0].set_yscale("symlog", linthresh=1e-3)
    ax[0].legend(fontsize=8)
    ax[0].set_title("Energy ledger of the coupled run")
    ax[1].plot(t, work_in, label="net inflow work")
    ax[1].plot(t, visc_cum, label="viscous dissipation")
    ax[1].plot(t, work_in - visc_cum, "--", label="inflow - viscous")
    ax[1].plot(t, struct + fluid, label="stored energy")
    ax[1].set_ylabel("cumulative [J/m]")
    ax[1].set_xlabel("t [s]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(ART, "fig_energy_ledger.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 4: verification summary
# ---------------------------------------------------------------------------
def fig_summary():
    items = []
    if cyl20:
        items += [("2D-2 $C_D$", cyl20["error_CD"] * 100, 10),
                  ("2D-2 $C_L$ amp", cyl20["error_CL"] * 100, 10),
                  ("2D-2 $St$", cyl20["error_St"] * 100, 10)]
    if structure:
        items += [("beam $f_1$", structure["frequency_error"] * 100, 3),
                  ("beam energy drift", structure["energy_drift"] * 100, 1)]
    if interface:
        items += [("interface force", interface["force_conservation_error"] * 100, 1)]
    if fsi:
        items += [("FSI-2 $f$", fsi["error_frequency"] * 100, 5),
                  ("FSI-2 $A$", fsi["error_amplitude"] * 100, 10)]
    if not items:
        return None
    labels = [i[0] for i in items]
    errs = [i[1] for i in items]
    tol = [i[2] for i in items]
    x = np.arange(len(items))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - 0.2, errs, 0.4, label="relative error [%]")
    ax.bar(x + 0.2, tol, 0.4, label="acceptance tolerance [%]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("percent")
    ax.legend(fontsize=8)
    ax.set_title("Verification summary (bar = error, line = criterion)")
    fig.tight_layout()
    path = os.path.join(ART, "fig_summary.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    figs = {}
    for name, fn in (("cylinder", fig_cylinder), ("fsi2", fig_fsi),
                     ("energy", fig_energy), ("summary", fig_summary)):
        path = fn()
        if path:
            figs[name] = os.path.relpath(path, ROOT)
    payload = {
        "figures": figs,
        "cylinder": {k: (cyl20 or {}).get(k) for k in
                     ("CD_mean", "CL_amp", "St", "error_CD", "error_CL", "error_St")},
        "structure": structure,
        "interface": interface,
        "fsi2": {k: (fsi or {}).get(k) for k in
                 ("frequency_Hz", "amplitude_m", "mean_CD", "amp_CL",
                  "mean_sub_iterations", "converged_fraction", "ledger",
                  "error_frequency", "error_amplitude", "window")},
    }
    with open(os.path.join(DOCS, "P153W2-N03_fsi_lbm_figures.json"), "w",
              encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in payload.items() if k != "figures"}, indent=1)[:2500])
    print("figures:", figs)


if __name__ == "__main__":
    main()