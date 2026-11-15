"""XVAL entry point: Shan-Chen Laplace pressure calibration.

Anchor: ``XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION``
External reference: Young-Laplace law ΔP = σ/R (classical capillarity).
Acceptance: Laplace-law fit R² > 0.95 and per-point error < 15 %.

The Young-Laplace law is the fundamental relation for curved interfaces:
the pressure jump across a curved interface is proportional to the surface
tension and inversely proportional to the radius of curvature.

For the Shan-Chen pseudo-potential LBM, the thermodynamic equation of state
(Yuan & Schaefer 2006) gives:

    P = cs² ρ_tot + cs²/2 Σ_{s,s'} G_{ss'} ψ_s ψ_{s'}

For a two-component system with G_aa = G_bb = 0, G_ab = G_ba = G:

    P = cs² (ρ_a + ρ_b) + cs² G ψ_a ψ_b

A circular droplet of the dense phase in the dilute phase should exhibit
a pressure jump ΔP = P_inside - P_outside = σ / R, where σ is the
surface tension determined by the interaction strength G and the
pseudo-potential form.

Run:
    PYTHONPATH=src python examples/lbm_shan_chen_laplace_pressure_xval.py

Writes:
    docs/assets/lbm_shan_chen_laplace/results.json
    docs/assets/lbm_shan_chen_laplace/fig*.png
    docs/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION.md

References
----------
[1] Yuan, P., Schaefer, L. (2006). Phys. Fluids 18, 042101.
    "Equations of state in a lattice Boltzmann model"
[2] Shan, X., Chen, H. (1993). Phys. Rev. E 47, 1815.
[3] Krueger et al. (2017). The Lattice Boltzmann Method, sec. 9.3.
[4] Huang, H., Sukop, M.C., Lu, X.-Y. (2015). Lattice Boltzmann Method,
    sec. 5.4 (Laplace law test).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tensorfem.lbm_shan_chen_mcmp import (
    MCMPConfig, ShanChenMCMP, laplace_pressure_calibration,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "lbm_shan_chen_laplace"
DOC = ROOT / "docs" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION.md"

ASSETS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- study config
RADII = [10.0, 15.0, 20.0, 25.0, 30.0]
N = 128
G_AB = 5.0
RHO_IN = 1.4
RHO_OUT = 0.2
STEPS = 8000
TAU = 1.0
INTERFACE_WIDTH = 3.0

THRESHOLD_LAPLACE_ERROR = 0.15  # 15% owner standard
THRESHOLD_R_SQUARED = 0.95


# ------------------------------------------------------------------- figures
def make_figures(results: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []

    # Fig 1: ΔP vs 1/R (Laplace law)
    fig, ax = plt.subplots(figsize=(7, 5))
    inv_R = [1.0 / r["radius"] for r in results]
    dP = [r["delta_P"] for r in results]
    sigma = results[0]["sigma_estimated"]
    ax.plot(inv_R, dP, "o", ms=8, label="measured ΔP")
    inv_R_fit = np.linspace(min(inv_R) * 0.9, max(inv_R) * 1.1, 100)
    ax.plot(inv_R_fit, sigma * inv_R_fit, "r--", lw=1.5,
            label=f"fit: σ = {sigma:.4f} (R² = {results[0]['r_squared']:.4f})")
    ax.set_xlabel("1/R [lattice$^{-1}$]")
    ax.set_ylabel("ΔP [lattice units]")
    ax.set_title("Young-Laplace law: ΔP = σ/R")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    p = ASSETS / "fig1_laplace_law.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(str(p.relative_to(ROOT)))

    # Fig 2: Pressure profile for one radius
    fig, ax = plt.subplots(figsize=(7, 4))
    sample = results[2]  # R=20
    r_bins = np.array(sample["r_bins"])
    P_avg = np.array(sample["P_profile"])
    rho_avg = np.array(sample["rho_a_profile"])
    ax2 = ax.twinx()
    ax.plot(r_bins, P_avg, "b-", lw=1.5, label="P(r)")
    ax2.plot(r_bins, rho_avg, "r--", lw=1, alpha=0.7, label="ρ_a(r)")
    ax.axvline(sample["interface_radius"], color="gray", ls=":", alpha=0.5,
               label=f"interface R={sample['interface_radius']:.1f}")
    ax.set_xlabel("r [lattice]")
    ax.set_ylabel("P [lattice units]")
    ax2.set_ylabel("ρ_a")
    ax.set_title(f"Radial profiles (R={sample['radius']:.0f}, {sample['steps']} steps)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax.grid(True, alpha=0.3)
    p = ASSETS / "fig2_radial_profiles.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(str(p.relative_to(ROOT)))

    # Fig 3: Spurious current convergence (if we had time series, show final values)
    fig, ax = plt.subplots(figsize=(6, 4))
    radii = [r["radius"] for r in results]
    sc = [r["spurious_current"] for r in results]
    ax.bar(range(len(radii)), sc, color="tab:blue")
    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels([f"R={r:.0f}" for r in radii], fontsize=8)
    ax.set_ylabel("max |u| (spurious current)")
    ax.set_title("Spurious currents at equilibrium")
    ax.grid(True, alpha=0.3, axis="y")
    p = ASSETS / "fig3_spurious_currents.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(str(p.relative_to(ROOT)))

    # Fig 4: Laplace error per radius
    fig, ax = plt.subplots(figsize=(6, 4))
    errors = [r["laplace_error_pct"] for r in results]
    ax.bar(range(len(radii)), errors, color="tab:orange")
    ax.axhline(THRESHOLD_LAPLACE_ERROR * 100, color="r", ls="--",
               label=f"{THRESHOLD_LAPLACE_ERROR*100:.0f}% threshold")
    ax.set_xticks(range(len(radii)))
    ax.set_xticklabels([f"R={r:.0f}" for r in radii], fontsize=8)
    ax.set_ylabel("Laplace error [%]")
    ax.set_title("Per-radius error vs Young-Laplace prediction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    p = ASSETS / "fig4_laplace_errors.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    figs.append(str(p.relative_to(ROOT)))

    return figs


# ---------------------------------------------------------------- doc render
def render_doc(results: list[dict], figs: list[str]) -> str:
    sigma = results[0]["sigma_estimated"]
    r_squared = results[0]["r_squared"]
    all_pass = all(
        r["laplace_error_pct"] < THRESHOLD_LAPLACE_ERROR * 100
        for r in results
    ) and r_squared >= THRESHOLD_R_SQUARED
    verdict = "PASS" if all_pass else "FAIL"

    table_rows = "\n".join(
        f"| {r['radius']:.0f} | {r['interface_radius']:.2f} | "
        f"{r['P_inside']:.6f} | {r['P_outside']:.6f} | "
        f"{r['delta_P']:.6f} | {r['laplace_predicted']:.6f} | "
        f"{r['laplace_error_pct']:.2f}% | "
        f"{r['spurious_current']:.2e} | "
        f"{'✓' if r['laplace_error_pct'] < THRESHOLD_LAPLACE_ERROR * 100 else '✗'} |"
        for r in results
    )

    fig_refs = "\n".join(
        f"![{Path(f).name}]({f})" for f in figs
    )

    return f"""# XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION

Anchor: `XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION`  
External reference: Young-Laplace law ΔP = σ/R (classical capillarity; Yuan & Schaefer 2006)  
Acceptance: Laplace-law fit R² ≥ {THRESHOLD_R_SQUARED} and per-point error < {THRESHOLD_LAPLACE_ERROR*100:.0f}%  
**Verdict: {verdict}** (R² = {r_squared:.4f}, σ = {sigma:.4f})

Every number in this document is injected from
`docs/assets/lbm_shan_chen_laplace/results.json` by
`examples/lbm_shan_chen_laplace_pressure_xval.py` (no hand-typed value).

## 1. Problem setting and reference

The Young-Laplace law is the fundamental relation for curved fluid interfaces:

$$\\Delta P = \\frac{{\\sigma}}{{R}}$$

where ΔP is the pressure jump across the interface, σ is the surface tension,
and R is the radius of curvature.

For the Shan-Chen pseudo-potential LBM, the thermodynamic equation of state
(Yuan & Schaefer 2006, Phys. Fluids 18, 042101) is:

$$P = c_s^2 \\rho_{{\\text{{tot}}}} + \\frac{{c_s^2}}{{2}} \\sum_{{s,s'}} G_{{ss'}} \\psi_s \\psi_{{s'}}$$

For D2Q9, $c_s^2 = 1/3$. With $G_{{aa}} = G_{{bb}} = 0$ and $G_{{ab}} = G_{{ba}} = G$:

$$P = c_s^2 (\\rho_a + \\rho_b) + c_s^2 G \\psi_a \\psi_b$$

A circular droplet of the dense phase (component a, $\\rho_{{\\text{{in}}}}$) in the
dilute phase (component b, $\\rho_{{\\text{{out}}}}$) should exhibit a pressure jump
consistent with the Young-Laplace law.

## 2. Method

The solver `src/tensorfem/lbm_shan_chen_mcmp.py` implements the two-component
Shan-Chen MCMP model with:
- D2Q9 lattice, SRT collision, Guo (2002) forcing
- Barycentric velocity (validated against lbmpy)
- Exponential pseudo-potential $\\psi = \\rho_0(1 - e^{{-\\rho/\\rho_0}})$

The `pressure_field()` method computes the thermodynamic pressure from the EOS.
For each droplet radius R, a circular drop is initialized with a tanh profile
and equilibrated for {STEPS} steps. The radial pressure profile is averaged,
and ΔP is measured as the difference between the plateau values inside and
outside the interface.

## 3. Parameters

| parameter | value |
|---|---|
| domain | {N} × {N}, periodic |
| relaxation | $\\tau_a = \\tau_b = {TAU}$ ($\\omega = 1$) |
| interaction | $g_{{ab}} = g_{{ba}} = {G_AB}$ |
| pseudo potential | exponential, $\\rho_0 = 1.0$ |
| droplet radii | {', '.join(f'{r:.0f}' for r in RADII)} |
| $\\rho_{{\\text{{in}}}}$ | {RHO_IN} |
| $\\rho_{{\\text{{out}}}}$ | {RHO_OUT} |
| equilibration | {STEPS} steps per radius |
| interface width | {INTERFACE_WIDTH} |

## 4. Results

| R | R_interface | P_inside | P_outside | ΔP measured | ΔP predicted (σ/R) | error | spurious current | pass |
|---|---|---|---|---|---|---|---|---|
{table_rows}

Surface tension estimate: **σ = {sigma:.4f}** (from linear fit of ΔP vs 1/R)  
Fit quality: **R² = {r_squared:.4f}**

## 5. Figures

{fig_refs}

## 6. Error assessment

- Laplace law fit R² = {r_squared:.4f} {'≥' if r_squared >= THRESHOLD_R_SQUARED else '<'} {THRESHOLD_R_SQUARED} → {'PASS' if r_squared >= THRESHOLD_R_SQUARED else 'FAIL'}
- Maximum per-point Laplace error = {max(r['laplace_error_pct'] for r in results):.2f}% {'<' if max(r['laplace_error_pct'] for r in results) < THRESHOLD_LAPLACE_ERROR * 100 else '≥'} {THRESHOLD_LAPLACE_ERROR*100:.0f}% → {'PASS' if max(r['laplace_error_pct'] for r in results) < THRESHOLD_LAPLACE_ERROR * 100 else 'FAIL'}
- Maximum spurious current = {max(r['spurious_current'] for r in results):.2e} (should be ≪ 1 for equilibrium)
- Maximum mass drift = {max(r['mass_drift'] for r in results):.2e} (should be ≪ 1)

**Overall verdict: {verdict}**

## 7. Reproduction

```bash
cd /srv/tensor/agent/work/m21532
PYTHONPATH=src python examples/lbm_shan_chen_laplace_pressure_xval.py
```

Artifacts: `docs/assets/lbm_shan_chen_laplace/results.json`, `fig*.png`
"""


# ------------------------------------------------------------------- main
def main():
    t0 = time.time()
    print(f"Running Laplace pressure calibration (n={N}, g_ab={G_AB}, steps={STEPS})...")
    print(f"Radii: {RADII}")

    results = laplace_pressure_calibration(
        radii=RADII,
        n=N,
        g_ab=G_AB,
        rho_in=RHO_IN,
        rho_out=RHO_OUT,
        steps=STEPS,
        interface_width=INTERFACE_WIDTH,
        tau=TAU,
    )

    wall = time.time() - t0
    print(f"Completed in {wall:.1f} s")

    # Summary
    sigma = results[0]["sigma_estimated"]
    r_squared = results[0]["r_squared"]
    print(f"\nSurface tension estimate: σ = {sigma:.4f}")
    print(f"R² = {r_squared:.4f}")
    for r in results:
        print(f"  R={r['radius']:.0f}: ΔP={r['delta_P']:.6f}, "
              f"predicted={r['laplace_predicted']:.6f}, "
              f"error={r['laplace_error_pct']:.2f}%, "
              f"spurious={r['spurious_current']:.2e}")

    # Save results
    res_out = {
        "anchor": "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION",
        "sigma_estimated": sigma,
        "r_squared": r_squared,
        "threshold_laplace_error": THRESHOLD_LAPLACE_ERROR,
        "threshold_r_squared": THRESHOLD_R_SQUARED,
        "wall_time_s": round(wall, 1),
        "config": {
            "n": N, "g_ab": G_AB, "rho_in": RHO_IN, "rho_out": RHO_OUT,
            "steps": STEPS, "tau": TAU, "interface_width": INTERFACE_WIDTH,
            "radii": RADII,
        },
        "results": results,
    }
    results_path = ASSETS / "results.json"
    results_path.write_text(json.dumps(res_out, indent=2))
    print(f"\nResults saved to {results_path}")

    # Figures
    figs = make_figures(results)
    print(f"Figures: {figs}")

    # Document
    doc = render_doc(results, figs)
    DOC.write_text(doc)
    print(f"Document: {DOC}")

    # Verdict
    all_pass = all(
        r["laplace_error_pct"] < THRESHOLD_LAPLACE_ERROR * 100
        for r in results
    ) and r_squared >= THRESHOLD_R_SQUARED
    verdict = "PASS" if all_pass else "FAIL"
    print(f"\nVerdict: {verdict}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
