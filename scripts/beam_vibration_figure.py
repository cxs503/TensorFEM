"""Generate the beam-vibration verification figure and report data."""
import math
import json
import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfem.modal import (
    cantilever_beam_modes, cantilever_exact_angular_frequency,
    simply_supported_beam_modes, simply_supported_exact_angular_frequency,
    clamped_clamped_beam_modes, clamped_clamped_exact_angular_frequency,
    clamped_pinned_beam_modes, clamped_pinned_exact_angular_frequency,
)

L, EI, rho_a = 1.0, 1.0, 1.0

# --- Convergence study for all four BCs ---
bc_configs = [
    ("Cantilever (CF)", cantilever_beam_modes, cantilever_exact_angular_frequency, 4),
    ("Simply supported (SS)", simply_supported_beam_modes, simply_supported_exact_angular_frequency, None),
    ("Clamped–clamped (CC)", clamped_clamped_beam_modes, clamped_clamped_exact_angular_frequency, 4),
    ("Clamped–pinned (CP)", clamped_pinned_beam_modes, clamped_pinned_exact_angular_frequency, 4),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, (name, solver, exact_fn, max_mode) in enumerate(bc_configs):
    ax = axes[idx]
    exact = exact_fn(L, EI, rho_a)
    n_vals = [2, 4, 8, 16, 32, 64]
    errors = []
    for n in n_vals:
        result = solver(L, elements=n, EI=EI, rho_a=rho_a, modes=1)
        fe_val = result.angular_frequencies[0].item()
        errors.append(abs(fe_val - exact) / exact)

    ax.loglog(n_vals, errors, "o-", label="FE vs analytical", linewidth=2, markersize=6)
    # Reference slope h^2 (order 2)
    ref_n = np.array(n_vals, dtype=float)
    ref_err = errors[0] * (ref_n[0] / ref_n) ** 2
    ax.loglog(ref_n, ref_err, "k--", alpha=0.4, label=r"$O(h^2)$ reference")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Relative error")
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Print error table
    print(f"\n=== {name} ===")
    print(f"  Analytical omega_1 = {exact:.10f}")
    for n, err in zip(n_vals, errors):
        print(f"  n={n:3d}: error = {err:.2e}")

plt.suptitle("Euler–Bernoulli Beam Vibration: Convergence of Fundamental Frequency\n"
             "All boundary conditions verified against closed-form analytical roots",
             fontsize=12, fontweight="bold")
plt.tight_layout()

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
buf.seek(0)
png_base64 = base64.b64encode(buf.read()).decode("ascii")
plt.close()

# --- Error summary table ---
summary = []
for name, solver, exact_fn, max_mode in bc_configs:
    exact = exact_fn(L, EI, rho_a)
    for n in [8, 16, 32]:
        result = solver(L, elements=n, EI=EI, rho_a=rho_a, modes=1)
        fe_val = result.angular_frequencies[0].item()
        err = abs(fe_val - exact) / exact
        summary.append({
            "bc": name,
            "elements": n,
            "fe_omega": fe_val,
            "exact_omega": exact,
            "relative_error": err,
            "within_3pct": err < 0.03,
        })

# --- Characteristic roots table ---
roots_table = [
    {"bc": "Cantilever (CF)", "beta_1_L": 1.875104068711961, "source": "Blevins 1979, Table 8-1"},
    {"bc": "Simply supported (SS)", "beta_1_L": math.pi, "source": "Exact: n*pi"},
    {"bc": "Clamped–clamped (CC)", "beta_1_L": 4.730040744862704, "source": "Blevins 1979, Table 8-1"},
    {"bc": "Clamped–pinned (CP)", "beta_1_L": 3.926602312047917, "source": "Blevins 1979, Table 8-1"},
]

report_data = {
    "summary": summary,
    "roots": roots_table,
    "figure_base64": png_base64,
}

with open("/root/tp-agent/work/m2653/docs/benchmarks/beam_vibration_verification.json", "w") as f:
    json.dump(report_data, f, indent=2)

print(f"\nFigure size: {len(png_base64)} bytes base64")
print("Report data written to docs/benchmarks/beam_vibration_verification.json")
