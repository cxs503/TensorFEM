"""Case runner: Cummins radiation memory kernel + Kramers-Kronig causality gate.

Produces ``docs/radiation_memory_results.json``, the markdown tables in
``docs/CASE-FEM-RADIATION-MEMORY-KK.md`` and four figures.  Reference data are
closed-form (single-pole and sign-changing two-pole kernels); nothing here is
compared against this module's own output, so each number is an independent
verification rather than a regression snapshot.
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem import radiation_memory as rm

C, LAM, A_INF = 2.0, 1.5, 3.0
OUT_DOCS = os.path.join(os.path.dirname(__file__), "..", "docs")
BAND = np.linspace(0.01, 50.0, 2001)
TIMES = np.arange(0.0, 15.0 + 1e-12, 0.01)


def reference(times=TIMES, band=BAND):
    return rm.single_pole_pair(times, band, C, LAM, A_INF)


def two_pole(band, c=2.0, a=1.0, b=6.0):
    damping = c * (a * a / (a * a + band ** 2) - b * b / (b * b + band ** 2))
    added = A_INF + c * (a / (a * a + band ** 2) - b / (b * b + band ** 2))
    return damping, added


def main():
    ref = reference()
    kernel = rm.cummins_kernel(BAND, ref["damping"], TIMES)
    results = {"parameters": {"single_pole_c": C, "single_pole_lambda": LAM,
                              "added_mass_inf": A_INF,
                              "band_rad_s": [float(BAND[0]), float(BAND[-1])],
                              "band_points": int(BAND.size),
                              "kernel_grid_s": [0.0, float(TIMES[-1])]}}

    # --- V1: transform pair -------------------------------------------------
    v1 = {
        "kernel_rel_error": float(np.max(np.abs(kernel - ref["kernel"])) / C),
        "damping_rel_error": float(
            np.max(np.abs(rm.damping_from_kernel(TIMES, kernel, BAND) - ref["damping"]))
            / ref["damping"].max()
        ),
        "added_mass_rel_error": float(
            np.max(np.abs(rm.added_mass_from_kernel(TIMES, kernel, BAND[1:], A_INF)
                          - ref["added_mass"][1:]))
            / np.max(np.abs(ref["added_mass"] - A_INF))
        ),
    }
    v1["truncation"] = {
        k: float(val) for k, val in
        rm.kernel_truncation_estimate(BAND, ref["damping"], TIMES).items()
    }

    # --- V2/V3: causality gate ---------------------------------------------
    accepted = rm.kramers_kronig_residual(BAND, ref["damping"], ref["added_mass"], A_INF)
    other = rm.single_pole_pair(TIMES, BAND, C, 3.0, A_INF)["added_mass"]
    v3 = {
        "accepted": accepted["residual"],
        "damping_scaled_5pct": rm.kramers_kronig_residual(
            BAND, 1.05 * ref["damping"], ref["added_mass"], A_INF)["residual"],
        "cross_pair": rm.kramers_kronig_residual(
            BAND, ref["damping"], other, A_INF)["residual"],
        "level_shift_ainf_20pct": rm.kramers_kronig_residual(
            BAND, ref["damping"], ref["added_mass"], A_INF + 0.2)["residual"],
    }

    # resolution study: how fine must a table be for a 3% gate?
    resolution = []
    for n in (21, 41, 81, 161, 401, 1001, 2001):
        band = np.linspace(0.2, 12.0, n)
        damping, added = two_pole(band)
        gate = rm.kramers_kronig_residual(band, damping, added, A_INF)
        resolution.append({"n_points": n, "residual": gate["residual"],
                           "passed_at_3pct": gate["passed"]})
    coarse_residual = rm.kramers_kronig_residual(
        np.linspace(0.2, 12.0, 41),
        *two_pole(np.linspace(0.2, 12.0, 41)), A_INF)["residual"]

    # --- V4: Cummins force vs frequency domain ------------------------------
    speed = 0.3
    dt = 0.01
    times = np.arange(0.0, 40.0 + 1e-12, dt)
    kernel_t = rm.cummins_kernel(BAND, ref["damping"], times)
    sweep, time_domain, freq_domain = [], [], []
    for omega_0 in np.linspace(0.5, 6.0, 12):
        velocity = speed * np.cos(omega_0 * times)
        acceleration = -omega_0 * speed * np.sin(omega_0 * times)
        memory = rm.memory_force(times, kernel_t, velocity)
        rad_time = -(A_INF * acceleration + memory)
        a_at = float(np.interp(omega_0, BAND, ref["added_mass"]))
        b_at = float(np.interp(omega_0, BAND, ref["damping"]))
        rad_freq = -(a_at * acceleration + b_at * velocity)
        tail = slice(times.size // 2, None)
        sweep.append(float(omega_0))
        time_domain.append(float(np.max(np.abs(rad_time[tail]))))
        freq_domain.append(float(np.max(np.abs(rad_freq[tail]))))
    v4_error = float(np.max(np.abs(np.array(time_domain) - np.array(freq_domain)))
                     / np.max(np.abs(np.array(freq_domain))))

    # --- V5: dissipation energy balance ------------------------------------
    omega_0 = 2.0
    periods = 8
    t_e = np.arange(0.0, periods * 2 * math.pi / omega_0 + 1e-12, 0.005)
    v_e = speed * np.cos(omega_0 * t_e)
    k_e = rm.cummins_kernel(BAND, ref["damping"], t_e)
    mem_e = rm.memory_force(t_e, k_e, v_e)
    dissipated = float(np.trapezoid(-mem_e * v_e, t_e))
    expected = float(np.interp(omega_0, BAND, ref["damping"])) * float(
        np.trapezoid(v_e ** 2, t_e))

    # --- V6: state space ----------------------------------------------------
    fit = rm.fit_exponential_kernel(TIMES, kernel, n_modes=10)
    model = rm.StateSpaceMemory(fit["coefficients"], -fit["rates"])
    velocity = speed * np.cos(1.7 * TIMES)
    direct = rm.memory_force(TIMES, kernel, velocity)
    recursive = model.force(velocity, 0.01)
    tail = slice(TIMES.size // 2, None)
    v6 = {
        "n_modes": 10,
        "kernel_fit_rel_error": fit["relative_error"],
        "force_rel_error": float(np.max(np.abs(recursive[tail] - direct[tail]))
                                 / np.max(np.abs(direct[tail]))),
        "poles_max_real_part": float(np.max(fit["rates"] * -1.0)),
    }

    # --- V7: free decay -----------------------------------------------------
    c_d, lam_d = 0.5, 1.0
    band_d = np.linspace(1e-3, 60.0, 4001)
    damp_d = c_d * lam_d / (lam_d * lam_d + band_d ** 2)
    added_d = A_INF + c_d / (lam_d * lam_d + band_d ** 2)
    decay_rows = []
    lags = np.arange(900) * 0.01
    kernel_d = rm.cummins_kernel(band_d, damp_d, lags)
    for omega_n in (0.9, 1.2, 1.6):
        stiffness = omega_n ** 2 * (1.0 + float(np.interp(omega_n, band_d, added_d)))
        zeta_fd = rm.decay_damping_ratio(band_d, added_d, damp_d, 1.0, omega_n)
        sim = rm.simulate_cummins(1.0, A_INF, stiffness, lags, kernel_d, 0.01, 3000,
                                  x0=1.0)
        v_s, x_s = sim["v"], sim["x"]
        peaks = np.where((v_s[:-1] > 0.0) & (v_s[1:] <= 0.0))[0]
        keep = np.abs(x_s[peaks]) > 1e-3
        slope = np.polyfit(sim["t"][peaks][keep], np.log(np.abs(x_s[peaks][keep])), 1)[0]
        zeta_decay = float(-slope / omega_n)
        decay_rows.append({
            "omega_n_pred_rad_s": float(omega_n),
            "zeta_frequency_domain": float(zeta_fd),
            "zeta_free_decay": zeta_decay,
            "rel_error": float(abs(zeta_decay - zeta_fd) / zeta_fd),
        })

    # --- figures ------------------------------------------------------------
    os.makedirs(OUT_DOCS, exist_ok=True)
    fig1, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(TIMES, ref["kernel"], "k-", lw=2, label="exact $c e^{-\\lambda t}$")
    axes[0].plot(TIMES, kernel, "r--", lw=1.4, label="B→K transform")
    axes[0].plot(TIMES, fit["fit"], "b:", lw=1.4, label="10-mode state space")
    axes[0].set_xlabel("lag t [s]"); axes[0].set_ylabel("K(t) [kg/s$^2$]")
    axes[0].set_title("memory kernel: transform and state-space model")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].semilogy(BAND, np.abs(rm.damping_from_kernel(TIMES, kernel, BAND)
                                  - ref["damping"]) / ref["damping"].max(), label="B round trip")
    axes[1].semilogy(BAND[1:], np.abs(rm.added_mass_from_kernel(TIMES, kernel, BAND[1:], A_INF)
                                      - ref["added_mass"][1:]) / np.max(np.abs(ref["added_mass"] - A_INF)),
                     label="A from K")
    axes[1].set_xlabel("$\\omega$ [rad/s]"); axes[1].set_ylabel("relative error")
    axes[1].set_title("transform pair residuals (V1)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig1.tight_layout(); fig1.savefig(os.path.join(OUT_DOCS, "fig_radiation_memory_kernel.png"), dpi=130)

    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(BAND, ref["added_mass"], "k-", lw=2, label="exact A($\\omega$)")
    axes[0].plot(accepted["reconstructed"],
                 "r--", lw=1.4, label="Kramers-Kronig from B")
    axes[0].plot(BAND, ref["damping"] + A_INF, "b:", lw=1.2, label="B($\\omega$)+$A_\\infty$ (offset)")
    axes[0].set_xlabel("$\\omega$ [rad/s]"); axes[0].set_ylabel("[kg]")
    axes[0].set_title("causality gate on the exact pair"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].errorbar([r["omega_n_pred_rad_s"] for r in decay_rows],
                     [r["zeta_free_decay"] for r in decay_rows],
                     yerr=[abs(r["zeta_free_decay"] - r["zeta_frequency_domain"]) for r in decay_rows],
                     fmt="o", capsize=4, label="Cummins free decay")
    axes[1].plot([r["omega_n_pred_rad_s"] for r in decay_rows],
                 [r["zeta_frequency_domain"] for r in decay_rows], "k--", label="B/(2$\\omega_n$(M+A))")
    axes[1].set_xlabel("wet frequency $\\omega_n$ [rad/s]"); axes[1].set_ylabel("damping ratio $\\zeta$")
    axes[1].set_title("radiation damping ratio (V7)"); axes[1].legend(); axes[1].grid(alpha=0.3)
    fig2.tight_layout(); fig2.savefig(os.path.join(OUT_DOCS, "fig_radiation_memory_gate.png"), dpi=130)

    fig3, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(sweep, freq_domain, "k-", lw=2, label="frequency domain $-A\\ddot x-B\\dot x$")
    axes[0].plot(sweep, time_domain, "ro", ms=4, label="Cummins memory integral")
    axes[0].set_xlabel("$\\omega$ [rad/s]"); axes[0].set_ylabel("radiation force amplitude [N]")
    axes[0].set_title("time vs frequency domain (V4)"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].loglog([r["n_points"] for r in resolution], [r["residual"] for r in resolution],
                   "o-", label="two-pole pair, band 0.2-12 rad/s")
    axes[1].axhline(0.03, color="r", ls="--", label="3% gate")
    axes[1].set_xlabel("tabulated frequencies"); axes[1].set_ylabel("KK residual")
    axes[1].set_title("gate resolution requirement"); axes[1].legend(); axes[1].grid(alpha=0.3, which="both")
    fig3.tight_layout(); fig3.savefig(os.path.join(OUT_DOCS, "fig_radiation_memory_convergence.png"), dpi=130)

    results.update({
        "V1_transform_pair": v1,
        "V2_gate_accepted": {"residual": accepted["residual"],
                             "pointwise": accepted["residual_pointwise"],
                             "band": accepted["band"]},
        "V3_gate_negative_controls": v3,
        "gate_resolution_study": resolution,
        "gate_on_41_point_band": coarse_residual,
        "V4_time_vs_frequency_rel_error": v4_error,
        "V5_dissipation": {"dissipated": dissipated, "expected": expected,
                           "rel_error": float(abs(dissipated - expected) / expected)},
        "V6_state_space": v6,
        "V7_free_decay": decay_rows,
        "figures": ["docs/fig_radiation_memory_kernel.png",
                    "docs/fig_radiation_memory_gate.png",
                    "docs/fig_radiation_memory_convergence.png"],
    })
    path = os.path.join(OUT_DOCS, "radiation_memory_results.json")
    with open(path, "w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
