"""Report figures for the linear-stress-gradient buckling reference (mission 4510).

Numbers are read from the committed evidence JSONs -- nothing is hand copied.
Outputs three compact PNGs under ``/tmp`` (report payload budget), the same
data as the repository figures but downsampled so the report request stays well
under the 250 KB limit that silently fails on the platform.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EV = json.loads((ROOT / "benchmarks" / "plate_stress_gradient_buckling_evidence.json").read_text("utf-8"))
CC = json.loads((ROOT / "benchmarks" / "plate_stress_gradient_crosscheck_evidence.json").read_text("utf-8"))
OUT = Path("/tmp/pr88r3")
OUT.mkdir(exist_ok=True)

# ---- fig 1: gradient sweep and aspect sweep ------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=110)
psi = [row["stress_gradient"] for row in EV["gradient_sweep"]]
k = [row["coefficient_k"] for row in EV["gradient_sweep"]]
axes[0].plot(psi, k, "o-", color="#1f4e79", lw=1.6, ms=4, label="Ritz / Galerkin (this work)")
lit = [(a["stress_gradient"], a["literature_k"]) for a in EV["literature_anchors"] if a["aspect_ratio"] == 1.0]
if lit:
    axes[0].plot(*zip(*lit), "s", mfc="none", mec="#c00000", ms=9, mew=1.4, label="tabulated k (4.00 / 7.81)")
axes[0].set_xlabel(r"edge stress ratio $\psi=N_2/N_1$")
axes[0].set_ylabel(r"$k=N_1 b^2/(\pi^2 D)$")
axes[0].set_title("square panel, a/b = 1")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=7)

rows = sorted(EV["aspect_sweep"], key=lambda r: r["aspect_ratio"])
for key, style, name in (
    ("psi_plus_1", "o-", r"$\psi=+1$ uniform"),
    ("psi_zero", "s--", r"$\psi=0$ triangular"),
    ("psi_minus_1", "^-", r"$\psi=-1$ bending"),
):
    axes[1].plot([r["aspect_ratio"] for r in rows], [r[key] for r in rows], style, lw=1.5, ms=4, label=name)
axes[1].axhline(23.9, color="#c00000", lw=0.9, ls=":", label="tabulated 23.9 (min over a/b)")
axes[1].set_xlabel("aspect ratio a/b")
axes[1].set_ylabel("k")
axes[1].set_title(r"pure bending minimum vs a/b  ($k_{min}$=%.3f)" % EV["pure_bending_minimum_k"])
axes[1].grid(alpha=0.3)
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig1_coefficient_curves.png", pil_kwargs={"optimize": True})
plt.close(fig)

# ---- fig 2: cross-implementation parity ---------------------------------
rows = CC["rows"]
a1 = [r for r in rows if abs(r["aspect_ratio"] - 1.0) < 1e-12]
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), dpi=110)
g = np.array([r["galerkin_k"] for r in a1])
f = np.array([r["finite_difference_k"] for r in a1])
lim = [min(g.min(), f.min()) * 0.9, max(g.max(), f.max()) * 1.05]
axes[0].plot(lim, lim, "-", color="0.6", lw=1, label="y = x")
axes[0].plot(g, f, "o", color="#1f4e79", ms=5, label="a/b = 1, $\\psi$ = +1 ... -1")
axes[0].set_xlabel("variational (closed-form ramp integral) k")
axes[0].set_ylabel("strong-form finite-difference k")
axes[0].set_title("two independent discretisations")
axes[0].grid(alpha=0.3)
axes[0].legend(fontsize=7)

psi_all = [r["stress_gradient"] for r in rows]
err = [100.0 * r["relative_error"] for r in rows]
for aspect, marker in ((1.0, "o"), (2.0, "s")):
    sel = [(p, e) for p, e, r in zip(psi_all, err, rows) if abs(r["aspect_ratio"] - aspect) < 1e-12]
    if sel:
        axes[1].plot(*zip(*sel), marker + "-", lw=1.4, ms=4, label=f"a/b = {aspect:g}")
axes[1].axhline(1.0, color="#c00000", ls="--", lw=1, label="1 % gate")
axes[1].axhline(3.0, color="#7f7f7f", ls=":", lw=1, label="3 % acceptance")
axes[1].set_yscale("log")
axes[1].set_xlabel(r"$\psi$")
axes[1].set_ylabel("relative disagreement [%]")
axes[1].set_title("worst = %.3f %%" % CC["worst_relative_error_percent"])
axes[1].grid(alpha=0.3, which="both")
axes[1].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "fig2_crosscheck_parity.png", pil_kwargs={"optimize": True})
plt.close(fig)

# ---- fig 3: critical mode of the pure-bending panel (repo figure, rescaled)
try:
    from PIL import Image

    src = ROOT / "docs" / "assets" / "plate_stress_gradient" / "pure_bending_mode.png"
    img = Image.open(src).convert("RGB")
    img.thumbnail((900, 900))
    img.save(OUT / "fig3_pure_bending_mode.png", optimize=True)
except Exception as exc:  # pragma: no cover - figure is optional
    print("pil downscale failed:", exc)

for path in sorted(OUT.iterdir()):
    print(path.name, path.stat().st_size)