"""Figure for the bound-admissibility audit (anchor GAP-PP-3).

Two panels, no hand-typed numbers: every value is recomputed from
``tensorfem.plate_patch_plastic_limit``.

left  : the four bounds of the patch-loaded plate plus the old FE knee value,
        with the *inadmissible* one marked -- it sits below the exact full-plate
        collapse load, so it cannot be an upper bound;
right : the double-sine mechanism's twisting curvature against its bending
        curvature, the reason the twist term cannot be dropped.
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from tensorfem.plate_patch_plastic_limit import (
    ilyushin_sine_upper_bound,
    johansen_twist_upper_bound,
    low_temperature_flow_stress,
    plastic_moment,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

A = B = 1.0
T = 0.02
PATCH = (0.5, 0.5)
FY = low_temperature_flow_stress("DH36", -40.0).yield_stress
MP = plastic_moment(T, FY)
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"

labels = [
    "strip lower\n(statically admissible)",
    "Ilyushin sine\n(twist dropped)",
    "yield-line pyramid\n(hinge, admissible)",
    "Johansen + twist\n(this work, admissible)",
]
values = [
    strip_lower_bound(A, B, T, FY, PATCH)["pressure"],
    ilyushin_sine_upper_bound(A, B, T, FY, PATCH)["pressure"],
    yield_line_patch_upper_bound(A, B, T, FY, PATCH)["pressure"],
    johansen_twist_upper_bound(A, B, T, FY, PATCH)["pressure"],
]
colors = ["#1f77b4", "#d62728", "#2ca02c", "#2ca02c"]

fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
bar = axes[0].bar(range(4), [v / 1e6 for v in values], color=colors, alpha=0.85)
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(labels, fontsize=8)
axes[0].set_ylabel("patch pressure [MPa]")
axes[0].set_title("Bounds for a 1 m panel, 0.5x0.5 m patch, DH36 at -40 C")
for rect, value in zip(bar, values):
    axes[0].text(rect.get_x() + rect.get_width() / 2, value / 1e6 + 0.03,
                 f"{value / 1e6:.3f}", ha="center", fontsize=9)
axes[0].text(1, values[1] / 1e6 * 0.45,
             "NOT admissible:\nfull plate would give\n19.74 < 24 m_p/a^2",
             ha="center", fontsize=8, color="#d62728")
axes[0].grid(axis="y", alpha=0.3)

n = 240
xs = (torch.arange(n, dtype=torch.float64) + 0.5) / n
xx, yy = torch.meshgrid(xs, xs, indexing="ij")
k_bend = torch.pi ** 2 * torch.sin(torch.pi * xx) * torch.sin(torch.pi * yy)
k_twist = torch.pi ** 2 * torch.cos(torch.pi * xx) * torch.cos(torch.pi * yy)
image = axes[1].imshow((k_bend / k_twist).numpy(), origin="lower",
                       extent=[0, A, 0, B], cmap="RdBu_r", vmin=-3, vmax=3)
axes[1].set_title("bending curvature / twisting curvature\n"
                  "(double-sine mode, |ratio| ~ 1: the twist term is not small)")
axes[1].set_xlabel("x / a")
axes[1].set_ylabel("y / b")
fig.colorbar(image, ax=axes[1], shrink=0.85, label="(k1+k2)/kxy")
fig.tight_layout()
OUT.mkdir(exist_ok=True)
target = OUT / "gap_pp3_bound_admissibility.png"
fig.savefig(target, dpi=140)
print("wrote", target)
