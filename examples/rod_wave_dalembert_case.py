"""Case driver: CASE-DG-DALEMBERT-ELASTIC-1D (1D rod wave vs d'Alembert).

Runs the convergence study, the reflection/dispersion/energy cross-checks, writes
``artifacts/CASE-DG-DALEMBERT-ELASTIC-1D/case_result.json`` and four figures.

    PYTHONPATH=src python examples/rod_wave_dalembert_case.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.rod_wave_dalembert import (
    STEEL,
    compare_stress_profile,
    exact_rod_stress,
    run_case,
    solve_bar_wave,
)

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "CASE-DG-DALEMBERT-ELASTIC-1D"
OUT.mkdir(parents=True, exist_ok=True)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return value


def main() -> dict:
    result = run_case()
    parameters = result["parameters"]
    end_time = parameters["end_time_s"]

    # profiles used by figure 1 (coarse / medium / fine) and the exact curve
    profiles = {}
    for n_elem in (200, 800, 1600):
        profiles[n_elem] = compare_stress_profile(
            n_elem=n_elem, length=parameters["length_m"], young=parameters["young_Pa"],
            density=parameters["density_kg_m3"], pulse=__import__(
                "tensorfem.rod_wave_dalembert", fromlist=["GaussianPulse"]
            ).GaussianPulse(
                amplitude=parameters["pulse_amplitude_m_s"],
                center=parameters["pulse_center_m"],
                width=parameters["pulse_width_m"],
            ),
            end_time=end_time,
        )
    x_fine = np.linspace(0.0, parameters["length_m"], 4001)
    from tensorfem.rod_wave_dalembert import GaussianPulse

    pulse = GaussianPulse(
        amplitude=parameters["pulse_amplitude_m_s"],
        center=parameters["pulse_center_m"],
        width=parameters["pulse_width_m"],
    )
    sigma_exact = exact_rod_stress(
        x_fine, end_time, pulse=pulse, young=parameters["young_Pa"],
        density=parameters["density_kg_m3"], length=parameters["length_m"], boundary="free",
    ) / 1e6

    # --- figure 1: stress profile, FE vs d'Alembert -------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax = axes[0]
    ax.plot(x_fine, sigma_exact, "k-", lw=2.0, label="d'Alembert exact (method of images)")
    for n_elem, style in ((200, ":"), (800, "--"), (1600, "-")):
        row = profiles[n_elem]
        ax.plot(row["x"], row["stress_fem"] / 1e6, style, lw=1.2,
                label=f"FEM n={n_elem} (rel L2 {row['rel_l2']*100:.2f} %)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"axial stress $\sigma$ [MPa]")
    ax.set_title(f"t = {end_time*1e3:.2f} ms, free ends")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    row = profiles[1600]
    ax.plot(row["x"], row["stress_fem"] / 1e6, "-", lw=1.0, label="FEM n=1600")
    ax.plot(row["x"], row["stress_exact"] / 1e6, "k--", lw=1.0, label="exact")
    ax.plot(row["x"], (row["stress_fem"] - row["stress_exact"]) / 1e6, "r-", lw=0.8,
            label="FEM - exact")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("[MPa]")
    ax.set_title("error field (fine mesh)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_stress_profile.png", dpi=150)
    plt.close(fig)

    # --- figure 2: h-convergence -------------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for tag, rows, marker in (("free ends", result["rows_free"], "o"),
                              ("clamped ends", result["rows_fixed"], "s")):
        h = [r["h"] for r in rows]
        err = [r["rel_l2"] for r in rows]
        ax.loglog(h, err, marker + "-", label=tag)
    h = np.array([r["h"] for r in result["rows_free"]])
    e0 = result["rows_free"][0]["rel_l2"]
    ax.loglog(h, e0 * (h / h[0]) ** 2, "k:", label=r"$\propto h^2$")
    ax.axhline(0.02, color="r", ls="--", lw=1.0, label="2 % acceptance")
    ax.set_xlabel("element size h [m]")
    ax.set_ylabel("relative L2 error of $\\sigma(x,t^\\ast)$")
    ax.set_title("second-order convergence")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_convergence.png", dpi=150)
    plt.close(fig)

    # --- figure 3: dispersion law + reflected lobe sign ---------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8))
    ax = axes[0]
    theta = np.linspace(1e-3, 2.0, 400)
    from tensorfem.rod_wave_dalembert import numerical_phase_speed_ratio

    ax.plot(theta, [numerical_phase_speed_ratio(float(t)) for t in theta], "-",
            label="consistent mass")
    ax.plot(theta, [numerical_phase_speed_ratio(float(t), mass_form="lumped") for t in theta],
            "--", label="lumped mass")
    ax.plot(theta, 1.0 + theta ** 2 / 24.0, "k:", label=r"$1+\theta^2/24$")
    meas = result["dispersion"]
    ax.plot([m["theta"] for m in meas], [m["c_num_over_c_formula"] for m in meas], "ro",
            label="mode frequencies (time domain)")
    ax.set_xlabel(r"$\theta = k h$")
    ax.set_ylabel(r"$c_{num}/c$")
    ax.set_title("numerical dispersion of the 1D bar")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    ax = axes[1]
    run = result["run"]
    nodes = np.asarray(run["nodes"])
    snap_t = end_time
    probe = None
    t = np.asarray(run["probe"]["t"])
    sigma_probe = np.asarray(run["probe"]["stress"]) / 1e6
    ax.plot(t * 1e3, sigma_probe, lw=1.0)
    ax.axvline(result["probe"]["arrival_theory_s"] * 1e3, color="k", ls="--", lw=1.0,
               label="theory arrival (x - x0)/c")
    ax.set_xlabel("t [ms]")
    ax.set_ylabel(r"$\sigma$ at probe [MPa]")
    ax.set_title(f"probe x = {result['probe']['x_m']:.2f} m")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_dispersion_probe.png", dpi=150)
    plt.close(fig)

    # --- figure 4: reflection parity ---------------------------------------
    fixed = compare_stress_profile(
        n_elem=1600, length=parameters["length_m"], young=parameters["young_Pa"],
        density=parameters["density_kg_m3"], pulse=pulse, end_time=end_time,
        boundary="fixed",
    )
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.plot(profiles[1600]["x"], profiles[1600]["stress_exact"] / 1e6, "k-", lw=1.6,
            label="d'Alembert, free ends")
    ax.plot(fixed["x"], fixed["stress_exact"] / 1e6, "b--", lw=1.4,
            label="d'Alembert, clamped ends")
    ax.plot(profiles[1600]["x"], profiles[1600]["stress_fem"] / 1e6, "r:", lw=1.0,
            label="FEM, free ends (n=1600)")
    ax.axvspan(fixed["reflected"]["x"] - 1.5, fixed["reflected"]["x"] + 1.5, color="0.9",
               label="reflected lobe")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(r"$\sigma$ [MPa]")
    ax.set_title("reflection parity: free keeps the sign, clamped inverts it")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_reflection_parity.png", dpi=150)
    plt.close(fig)

    payload = _jsonable({k: v for k, v in result.items() if k != "run"})
    payload["figure_files"] = sorted(p.name for p in OUT.glob("*.png"))
    payload["steel"] = dict(STEEL)
    (OUT / "case_result.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: payload[k] for k in ("verdict", "tolerance", "finest",
                                            "translation_error", "decomposition_error",
                                            "probe", "energy_drift", "figure_files")}, indent=1))
    return payload


if __name__ == "__main__":
    main()