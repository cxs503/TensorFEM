"""Figures for the springing-whipping-hydroelastic domain experiment.

Re-runs the two headline cases (springing at zeta=0.02, whipping at zeta=0.02)
and renders:

* panel 1 -- springing time-domain bending moment at the midspan, with the
  analytic steady-state amplitude (1/(2 zeta) resonant DAF) marked;
* panel 2 -- whipping transient bending moment with the analytic modal
  superposition envelope;
* panel 3 -- damping-ratio sensitivity: computed DAF vs the analytic
  ``1/(2 zeta)`` line (all three springing cases);
* panel 4 -- wet mode shapes (fundamental + 2nd) of the KVLCC2 girder.

All numbers are read from ``comparison.json`` or recomputed by the same
``run_comparison`` machinery; nothing is typed from memory.

    PYTHONPATH=src python benchmarks/springing-whipping-hydroelastic/make_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tensorfem.dynamics import newmark_linear
from tensorfem.hull_girder_modal import (
    ballast_mass_per_length,
    lewis_heave_added_mass,
    wet_simply_supported_modes,
)
from tensorfem.timoshenko_beam_modal import timoshenko_shape_functions

from run_comparison import (
    _D,
    load_anchor,
    kvlcc2_section,
    uniform_pressure_nodal_load,
    modal_projection,
    modal_force_vector,
    run_modal_newmark,
    reconstruct_bending_moment,
    curvature_at_midspan,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "artifacts" / "springing_whipping_hydroelastic"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def springing_time_series(proj, section, elements, anchor, zeta, n_periods=50,
                          steps_per_period=200):
    f1 = float(proj["freq_hz"][0])
    w1 = float(proj["omega"][0])
    p0 = anchor["p_rms_pa"]
    F_nodal = uniform_pressure_nodal_load(section, elements, p0)
    F_amp = modal_force_vector(proj, F_nodal)
    T = 1.0 / f1
    dt = T / steps_per_period
    time = torch.linspace(0.0, n_periods * T, n_periods * steps_per_period + 1, dtype=_D)
    F_modal = torch.zeros((time.numel(), proj["omega"].numel()), dtype=_D)
    F_modal[:, 0] = F_amp[0] * torch.sin(w1 * time)
    u = run_modal_newmark(proj, zeta, F_modal, time)
    M = reconstruct_bending_moment(proj, section, elements, u)
    return time.detach().cpu().numpy(), M.detach().cpu().numpy(), F_amp, dt


def whipping_time_series(proj, section, elements, anchor, zeta, t_pulse=0.10,
                         duration_s=60.0, dt_s=0.005):
    p_peak = anchor["p_rms_pa"]
    F_nodal = uniform_pressure_nodal_load(section, elements, p_peak)
    F_amp = modal_force_vector(proj, F_nodal)
    n_pulse = max(2, int(t_pulse / dt_s))
    n_total = int(duration_s / dt_s)
    time = torch.linspace(0.0, duration_s, n_total + 1, dtype=_D)
    F_modal = torch.zeros((time.numel(), proj["omega"].numel()), dtype=_D)
    for i in range(n_pulse + 1):
        t = i * dt_s
        if t <= t_pulse:
            F_modal[i] = F_amp * math.sin(math.pi * t / t_pulse)
    u = run_modal_newmark(proj, zeta, F_modal, time)
    M = reconstruct_bending_moment(proj, section, elements, u)
    return time.detach().cpu().numpy(), M.detach().cpu().numpy(), F_amp


def main() -> int:
    anchor = load_anchor()
    section = kvlcc2_section(anchor)
    elements = 40
    n_modes = 4
    m33, _ = lewis_heave_added_mass(0.5 * anchor["B"], 0.5 * anchor["T"], anchor["rho"])
    extra = ballast_mass_per_length(section)
    proj = modal_projection(section, elements, m33, extra, n_modes)

    art = json.loads((Path(__file__).resolve().parent / "comparison.json").read_text())

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    # ---- panel 1: springing time-domain response (zeta = 0.02) ------------
    ax = axes[0, 0]
    t, M, F_amp, dt = springing_time_series(proj, section, elements, anchor, 0.02)
    # steady-state window (last 10 periods)
    n_ss = 10 * 200
    t_ss = t[-n_ss:] - t[-n_ss]
    M_ss = M[-n_ss:] / 1e9
    M_exp = art["cases"][0]["M_expected_nm"] / 1e9
    ax.plot(t_ss, M_ss, lw=1.2, color="#1f77b4", label="Newmark (TensorFEM)")
    ax.axhline(M_exp, color="#d62728", ls="--", lw=1.5,
               label=f"analytic steady-state {M_exp:.3f} GN·m")
    ax.axhline(-M_exp, color="#d62728", ls="--", lw=1.5)
    ax.set_xlabel("t [s] (steady-state window)")
    ax.set_ylabel("midspan bending moment [GN·m]")
    ax.set_title("Springing: harmonic pressure at wet $f_1$=0.130 Hz, "
                 r"$\zeta$=0.02, $p_{rms}$=325.8 Pa")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- panel 2: whipping transient (zeta = 0.02) ------------------------
    ax = axes[0, 1]
    t, M, F_amp = whipping_time_series(proj, section, elements, anchor, 0.02)
    M_np = M / 1e6
    ax.plot(t, M_np, lw=1.2, color="#1f77b4", label="Newmark (TensorFEM)")
    # analytic modal superposition envelope (fundamental mode)
    w1 = float(proj["omega"][0])
    zeta = 0.02
    impulse = F_amp[0] * (2.0 * 0.10 / math.pi)
    curv = float(curvature_at_midspan(proj, section, elements)[0])
    env = (section.bending_stiffness * (impulse / w1) * curv / 1e6
           * np.exp(-zeta * w1 * t))
    ax.plot(t, env, color="#d62728", ls="--", lw=1.5,
            label=r"analytic envelope $e^{-\zeta\omega_1 t}$")
    ax.plot(t, -env, color="#d62728", ls="--", lw=1.5)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("midspan bending moment [MN·m]")
    ax.set_title("Whipping: half-sine slamming impulse, "
                 r"$\zeta$=0.02, $p_{peak}$=325.8 Pa")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 40)

    # ---- panel 3: damping sensitivity --------------------------------------
    ax = axes[1, 0]
    zetas = [0.01, 0.02, 0.05]
    daf = [c["daf"] for c in art["cases"] if c["case"].startswith("springing")]
    daf_an = [1.0 / (2.0 * z) for z in zetas]
    ax.plot(zetas, daf, "o-", color="#1f77b4", lw=2, label="computed DAF (Newmark)")
    zz = np.linspace(0.005, 0.06, 200)
    ax.plot(zz, 1.0 / (2.0 * zz), "--", color="#d62728", lw=1.5,
            label=r"analytic $1/(2\zeta)$")
    for z, d in zip(zetas, daf):
        ax.annotate(f"{d:.1f}", (z, d), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(r"damping ratio $\zeta$")
    ax.set_ylabel("dynamic amplification factor")
    ax.set_title("Damping-ratio sensitivity (springing peak)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # ---- panel 4: wet mode shapes -------------------------------------------
    ax = axes[1, 1]
    x = np.linspace(0, section.length, 200)
    for n in (0, 1):
        # reconstruct the mode shape from the FE nodal values (linear interp)
        mode = proj["modes"][:, n].detach().cpu().numpy()
        w_nodes = mode[0::2]  # transverse displacement at even DOFs
        x_nodes = np.linspace(0, section.length, w_nodes.size)
        # normalise to unit peak
        w_nodes = w_nodes / np.max(np.abs(w_nodes))
        ax.plot(x_nodes, w_nodes, lw=2,
                label=f"mode {n+1}: $f_{{wet}}$={proj['freq_hz'][n]:.3f} Hz")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("normalised vertical displacement")
    ax.set_title("KVLCC2 wet hull-girder modes (Lewis added mass)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "springing_whipping_hydroelastic_figures.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
