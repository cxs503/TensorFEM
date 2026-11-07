"""Generate evidence table + figures for the low-frequency acoustic asymptotics.

Usage: python3 docs/gen_acoustic_low_frequency_evidence.py
Writes docs/acoustic-low-frequency-asymptotic-evidence.json and PNGs under
docs/assets/acoustic-low-freq/.  All numbers are computed here (zero hand-copy).
"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.acoustic_radiation import (
    dipole_radiation_efficiency,
    leppington_radiation_efficiency,
    low_frequency_radiation_efficiency,
    low_frequency_scaling_exponent,
    maidanik_radiation_efficiency,
    monopole_radiation_efficiency,
    radiation_efficiency_rayleigh,
)

AIR_RHO, AIR_C, PI = 1.225, 343.0, math.pi
OUT = os.path.join(os.path.dirname(__file__), "assets", "acoustic-low-freq")
os.makedirs(OUT, exist_ok=True)


def freq_of(ka, length, width):
    return (ka / max(length, width)) * AIR_C / (2 * PI)


def legacy_sigma(length, width, frequency, m, n):
    """Superseded m6164 calibrated model (negative control)."""
    k = 2 * PI * frequency / AIR_C
    k_mn = math.sqrt((m * PI / length) ** 2 + (n * PI / width) ** 2)
    geom = 1.03 if (m % 2 == 1 and n % 2 == 1) else 0.1
    return (k / k_mn) ** 2 * geom


CASES = [  # length, width, m, n, ka
    (1.0, 1.0, 1, 1, 0.05), (1.0, 1.0, 1, 1, 0.10), (1.0, 1.0, 1, 1, 0.20),
    (1.0, 1.0, 1, 3, 0.10), (1.0, 1.0, 1, 3, 0.20),
    (2.0, 1.0, 1, 1, 0.10), (2.0, 1.0, 3, 1, 0.20),
    (1.0, 2.0, 1, 1, 0.10),
    (1.0, 1.0, 2, 1, 0.10), (1.0, 1.0, 2, 1, 0.20), (1.0, 1.0, 2, 1, 0.40),
    (2.0, 1.0, 2, 1, 0.20), (1.0, 1.0, 1, 2, 0.20),
]

rows = []
for a, b, m, n, ka in CASES:
    f = freq_of(ka, a, b)
    sig_num = radiation_efficiency_rayleigh(
        length=a, width=b, frequency=f, rho=AIR_RHO, c=AIR_C,
        m=m, n=n, n_points=48)
    sig_ana = low_frequency_radiation_efficiency(
        length=a, width=b, frequency=f, c=AIR_C, m=m, n=n)
    sig_leg = legacy_sigma(a, b, f, m, n)
    sig_lep = leppington_radiation_efficiency(
        length=a, width=b, frequency=f, c=AIR_C, m=m, n=n)
    rows.append({
        "length_m": a, "width_m": b, "mode": [m, n], "ka": ka,
        "frequency_hz": round(f, 6),
        "sigma_rayleigh_routeA": sig_num,
        "sigma_analytic_asymptote": sig_ana,
        "error_analytic_pct": (abs(sig_num - sig_ana) / sig_ana * 100)
        if sig_ana > 0 else None,
        "sigma_superseded_calibrated": sig_leg,
        "error_superseded_pct": (abs(sig_leg - sig_ana) / sig_ana * 100)
        if sig_ana > 0 else None,
        "sigma_leppington": sig_lep,
        "scaling_exponent": low_frequency_scaling_exponent(m, n),
    })

# even-even slope study (prefactor not derived -> slope only)
even_even = []
for ka in (0.4, 0.2, 0.1, 0.05):
    sig = radiation_efficiency_rayleigh(
        length=1.0, width=1.0, frequency=freq_of(ka, 1, 1), rho=AIR_RHO, c=AIR_C,
        m=2, n=2, n_points=48)
    even_even.append({"ka": ka, "sigma_rayleigh_routeA": sig})
slopes = [math.log(even_even[i]["sigma_rayleigh_routeA"] /
                   even_even[i + 1]["sigma_rayleigh_routeA"]) /
          math.log(even_even[i]["ka"] / even_even[i + 1]["ka"])
          for i in range(len(even_even) - 1)]

# pre-coincidence prefactor ratio: old 1.03 factor vs analytic
k_probe = 0.1
k_mn = math.sqrt(2.0) * PI
prefactor_ratio = monopole_radiation_efficiency(
    length=1.0, width=1.0, frequency=freq_of(k_probe, 1, 1), c=AIR_C, m=1, n=1
) / (k_probe / k_mn) ** 2

summary = {
    "air": {"rho": AIR_RHO, "c": AIR_C},
    "cases": rows,
    "max_error_analytic_pct": max(r["error_analytic_pct"] for r in rows),
    "max_error_analytic_monopole_pct": max(
        r["error_analytic_pct"] for r in rows if r["scaling_exponent"] == 2),
    "max_error_analytic_dipole_pct": max(
        r["error_analytic_pct"] for r in rows if r["scaling_exponent"] == 4),
    "superseded_model_errors_pct": {
        f"mode{r['mode']}_a{r['length_m']}_b{r['width_m']}_ka{r['ka']}":
            round(r["error_superseded_pct"], 2) for r in rows
        if r["error_superseded_pct"] is not None},
    "superseded_max_error_pct": max(
        r["error_superseded_pct"] for r in rows
        if r["error_superseded_pct"] is not None),
    "superseded_violations_over_15pct": sum(
        1 for r in rows
        if r["error_superseded_pct"] is not None and r["error_superseded_pct"] > 15.0),
    "even_even_slope_study": {"points": even_even, "loglog_slopes": slopes},
    "prefactor_ratio_1p1_square": prefactor_ratio,
}
with open(os.path.join(os.path.dirname(__file__),
                       "acoustic-low-frequency-asymptotic-evidence.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

# --------------------------------------------------------------------------
# Figure 1: sigma vs ka, Rayleigh (markers) against analytic asymptotes
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 5.0))
kas = np.logspace(-2, 0, 17)
series = [
    (1.0, 1.0, 1, 1, "tab:blue", "(1,1) a=b"),
    (1.0, 1.0, 1, 3, "tab:green", "(1,3) a=b"),
    (2.0, 1.0, 3, 1, "tab:red", "(3,1) a=2b"),
    (1.0, 1.0, 2, 1, "tab:purple", "(2,1) a=b"),
    (1.0, 1.0, 2, 2, "tab:orange", "(2,2) a=b"),
]
for a, b, m, n, color, label in series:
    xs, ys, ya = [], [], []
    for ka in kas:
        f = freq_of(ka, a, b)
        xs.append(ka)
        ys.append(radiation_efficiency_rayleigh(
            length=a, width=b, frequency=f, rho=AIR_RHO, c=AIR_C,
            m=m, n=n, n_points=40))
        ya.append(low_frequency_radiation_efficiency(
            length=a, width=b, frequency=f, c=AIR_C, m=m, n=n))
    ax.loglog(xs, ys, "o", color=color, ms=4, label=f"Rayleigh {label}")
    if max(ya) > 0:
        ax.loglog(xs, ya, "-", color=color, lw=1.2,
                  label=f"analytic {label} ($k^{{{low_frequency_scaling_exponent(m, n)}}}$)")

ref_ka = np.array([0.02, 0.2])
ax.loglog(ref_ka, 5.2e-2 * ref_ka**2, "k--", lw=0.8, alpha=0.6)
ax.text(0.021, 2.5e-5, r"$\propto k^2$", fontsize=9)
ax.loglog(ref_ka, 3.0e-2 * ref_ka**4, "k:", lw=0.8, alpha=0.6)
ax.text(0.09, 6e-10, r"$\propto k^4$", fontsize=9)
ax.set_xlabel(r"$ka$  (acoustic wavenumber $\times$ plate length)")
ax.set_ylabel(r"radiation efficiency $\sigma$")
ax.set_title("Baffled simply-supported plate: Rayleigh integral vs\nexact compact-source asymptotes")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=7, ncol=2, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sigma_vs_ka_asymptotes.png"), dpi=140)
plt.close(fig)

# --------------------------------------------------------------------------
# Figure 2: error of superseded calibrated model vs fixed analytic law
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 4.6))
error_series = [
    (1.0, 1.0, 1, 1, "tab:blue", "(1,1) a=b"),
    (2.0, 1.0, 1, 1, "tab:cyan", "(1,1) a=2b"),
    (1.0, 1.0, 1, 3, "tab:green", "(1,3) a=b"),
    (2.0, 1.0, 3, 1, "tab:red", "(3,1) a=2b"),
    (1.0, 1.0, 2, 1, "tab:purple", "(2,1) a=b"),
]
for a, b, m, n, color, label in error_series:
    xs, old_err, new_err = [], [], []
    for ka in np.logspace(-2, -0.3, 14):
        f = freq_of(ka, a, b)
        ana = low_frequency_radiation_efficiency(
            length=a, width=b, frequency=f, c=AIR_C, m=m, n=n)
        num = radiation_efficiency_rayleigh(
            length=a, width=b, frequency=f, rho=AIR_RHO, c=AIR_C,
            m=m, n=n, n_points=40)
        xs.append(ka)
        old_err.append(abs(legacy_sigma(a, b, f, m, n) - ana) / ana * 100)
        new_err.append(abs(num - ana) / ana * 100)
    ax.loglog(xs, old_err, "-", color=color, lw=1.4,
              label=f"superseded fit {label}")
    ax.loglog(xs, new_err, "--", color=color, lw=0.9, alpha=0.8)
ax.axhline(15.0, color="k", ls="-.", lw=1.0)
ax.text(0.0105, 17, "15% acceptance gate", fontsize=8)
ax.axhline(3.0, color="gray", ls=":", lw=1.0)
ax.text(0.0105, 1.1, "3% target (dashed = new analytic law error)", fontsize=8)
ax.set_xlabel(r"$ka$")
ax.set_ylabel("error vs exact asymptote  [%]")
ax.set_title("Verification debt quantified: superseded calibrated low-frequency model\n(solid) vs exact analytic law (dashed)")
ax.grid(True, which="both", alpha=0.25)
ax.legend(fontsize=7, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "legacy_model_error.png"), dpi=140)
plt.close(fig)

print(json.dumps({
    "max_err_monopole_pct": round(summary["max_error_analytic_monopole_pct"], 3),
    "max_err_dipole_pct": round(summary["max_error_analytic_dipole_pct"], 3),
    "superseded_max_err_pct": round(summary["superseded_max_error_pct"], 2),
    "superseded_violations": summary["superseded_violations_over_15pct"],
    "even_even_slopes": [round(s, 4) for s in slopes],
    "prefactor_ratio": round(prefactor_ratio, 5),
    "figures": sorted(os.listdir(OUT)),
}, indent=2))