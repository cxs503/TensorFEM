"""Generate figures for the XVAL report."""
from __future__ import annotations

import json
import os
import base64
import io

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load results
output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts", "xval_aitken_ibm")
with open(os.path.join(output_dir, "xval_results.json")) as f:
    results = json.load(f)

figures = []

# Figure 1: Aitken relaxation convergence
aitken_data = results["aitken"]["convergence_test"]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(len(aitken_data["omegas"])), aitken_data["omegas"], "b-o", markersize=4)
ax.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="omega0=1.0")
ax.set_xlabel("Iteration")
ax.set_ylabel("Relaxation parameter ω")
ax.set_title("Aitken Delta-squared relaxation parameter evolution")
ax.legend()
ax.grid(True, alpha=0.3)

buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
buf.seek(0)
png_base64 = base64.b64encode(buf.read()).decode()
figures.append({
    "name": "aitken_relaxation_convergence",
    "caption": "Aitken relaxation parameter ω evolution over 30 iterations with synthetic residuals. The parameter adapts dynamically based on the residual history.",
    "png_base64": png_base64,
})
plt.close(fig)

# Figure 2: Structural solver verification
struct_data = results["structural"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Frequency comparison
ax1.bar(["Analytic", "Numerical"], 
        [struct_data["frequency"]["analytic_Hz"], struct_data["frequency"]["numerical_Hz"]],
        color=["#2196F3", "#4CAF50"])
ax1.set_ylabel("Frequency [Hz]")
ax1.set_title("First natural frequency")
ax1.grid(True, alpha=0.3)

# Static deflection comparison
ax2.bar(["Analytic", "Numerical"],
        [struct_data["static_deflection"]["analytic_m"], struct_data["static_deflection"]["numerical_m"]],
        color=["#2196F3", "#4CAF50"])
ax2.set_ylabel("Deflection [m]")
ax2.set_title("Static tip deflection (unit load)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
buf.seek(0)
png_base64 = base64.b64encode(buf.read()).decode()
figures.append({
    "name": "structural_verification",
    "caption": "Structural solver verification: first natural frequency and static tip deflection match analytic solutions to machine precision.",
    "png_base64": png_base64,
})
plt.close(fig)

# Figure 3: Algorithm comparison summary
algo_data = results["algorithmic_comparison"]
categories = list(algo_data.keys())
differences = [algo_data[k]["difference"][:50] + "..." for k in categories]

fig, ax = plt.subplots(figsize=(10, 5))
y_pos = np.arange(len(categories))
ax.barh(y_pos, [1]*len(categories), color="#2196F3", alpha=0.7)
for i, (cat, diff) in enumerate(zip(categories, differences)):
    ax.text(0.05, i, f"{cat}: {diff}", va="center", fontsize=9, transform=ax.get_yaxis_transform())
ax.set_xlim(0, 1.1)
ax.set_yticks(y_pos)
ax.set_yticklabels([c.replace("_", " ").title() for c in categories])
ax.set_xlabel("Algorithmic difference present")
ax.set_title("TensorFEM vs LIFE: Algorithmic Comparison")
ax.set_xticks([])

plt.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
buf.seek(0)
png_base64 = base64.b64encode(buf.read()).decode()
figures.append({
    "name": "algorithmic_comparison",
    "caption": "Summary of algorithmic differences between TensorFEM and LIFE implementations. Key differences are in fluid solver (hybrid vs pure IBM) and structural model (1D beam vs 2D/3D FE).",
    "png_base64": png_base64,
})
plt.close(fig)

# Save figures JSON
with open(os.path.join(output_dir, "figures.json"), "w") as f:
    json.dump({"figures": figures}, f, indent=2)

print(f"Generated {len(figures)} figures")
print("Figures saved to:", os.path.join(output_dir, "figures.json"))
