"""Qualification run for :mod:`tensorfem.axisym_shell`.

Writes ``benchmarks/verified/fem_axisym_shell/result.json`` plus figures used by
the report, and prints a table.  Every number in the documentation is copied
from this JSON, so the doc cannot drift from the code.
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.axisym_shell_benchmarks import (clamped_cylinder,  # noqa: E402
                                               membrane_cone, membrane_cylinder,
                                               sphere_pressure, torus_pressure)

OUT = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "verified",
                   "fem_axisym_shell")

CYL = dict(radius=1.0, length=2.0, thickness=0.01, young=2.0e11, poisson=0.3,
           pressure=1.0e6)
SPH = dict(radius=10.0, thickness=0.04, young=6.825e7, poisson=0.3, pressure=1.0)


def main() -> dict:
    os.makedirs(OUT, exist_ok=True)
    results = {}

    results["cylinder_open"] = membrane_cylinder(closed_ends=False, facets=8, **CYL)
    results["cylinder_closed"] = membrane_cylinder(closed_ends=True, facets=8, **CYL)

    sphere = {}
    for facets in (4, 8, 16, 32):
        sphere[facets] = sphere_pressure(facets=facets, exact_geometry=True, **SPH)
    results["sphere"] = {str(k): v for k, v in sphere.items()}

    clamped = {}
    for facets in (50, 100, 200):
        clamped[facets] = clamped_cylinder(radius=1.0, length=0.5, thickness=0.01,
                                          young=2.0e11, poisson=0.3, pressure=1.0e6,
                                          facets=facets)
    results["clamped_cylinder"] = {str(k): v for k, v in clamped.items()}
    results["clamped_cylinder_full_integration"] = clamped_cylinder(
        radius=1.0, length=0.5, thickness=0.01, young=2.0e11, poisson=0.3,
        pressure=1.0e6, facets=200, shear_integration="full")

    # documented failure region: general curved meridians
    results["cone"] = membrane_cone(r_small=0.5, r_large=1.5,
                                    rise=0.5773502691896257, thickness=0.01,
                                    young=2.0e11, poisson=0.3, pressure=1.0e6,
                                    facets=12)
    torus = {}
    for facets in (4, 8, 16, 32, 64):
        torus[facets] = torus_pressure(ring_radius=1.0, tube_radius=0.25,
                                       thickness=0.01, young=2.0e11, poisson=0.3,
                                       pressure=1.0e6, facets=facets,
                                       exact_geometry=True)
        torus[facets]["faceted"] = torus_pressure(
            ring_radius=1.0, tube_radius=0.25, thickness=0.01, young=2.0e11,
            poisson=0.3, pressure=1.0e6, facets=facets, exact_geometry=False)
    results["torus"] = {str(k): v for k, v in torus.items()}

    with open(os.path.join(OUT, "result.json"), "w") as fh:
        json.dump(results, fh, indent=1)

    # ---------------- figures ---------------- #
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.2))

    ax = axes[0]
    analytic = SPH["pressure"] * SPH["radius"] / 2.0
    for facets, style in ((4, "o--"), (8, "s-"), (16, "^-")):
        rec = results["sphere"][str(facets)]
        phi = [math_phi for math_phi in
               [0.5 * 3.141592653589793 * i / facets for i in range(facets + 1)
                for _ in range(2)]][:len(rec["N_membrane"])]
        ax.plot(range(len(rec["N_membrane"])), rec["N_membrane"], style,
                label=f"FE, {facets} elements")
    ax.axhline(analytic, color="k", lw=1.4, label="exact $pR/2$ = 5 N/m")
    ax.set_xlabel("Gauss point index along the meridian (pole $\\to$ equator)")
    ax.set_ylabel("$N_\\theta$  [N/m]")
    ax.set_title("Closed sphere under pressure:\ncircular-arc meridians")
    ax.legend(fontsize=8)

    ax = axes[1]
    facets = sorted(int(k) for k in results["sphere"])
    err = [results["sphere"][str(f)]["N_theta_max_rel_error"] for f in facets]
    ax.loglog(facets, err, "o-", label="arc meridians")
    ax.loglog(facets, [err[0] * (facets[0] / f) ** 2 for f in facets], "k:",
              label="$O(N^{-2})$ reference")
    ax.set_xlabel("meridional elements per 90 deg")
    ax.set_ylabel("max rel. error in $N_\\theta$")
    ax.set_title("Sphere membrane convergence")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    rec = results["clamped_cylinder"]["200"]
    x = rec["x"]
    w_mem = rec["w_membrane"]
    ax.plot(x, [v / w_mem for v in rec["w_fe"]], "o", ms=3,
            label="ring element (200 elements)")
    ax.plot(x, [v / w_mem for v in rec["w_exact"]], "-", lw=1.2,
            label="closed form (Shell4 qualification oracle)")
    ax.set_xlabel("axial coordinate $x$  [m]")
    ax.set_ylabel("$w / w_{mem}$")
    ax.set_title("Clamped-ends cylinder:\nbending boundary layers")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "axisym_shell_qualification.png"), dpi=130)

    # limitation figure
    fig2, ax2 = plt.subplots(figsize=(6.2, 4.0))
    tf = sorted(int(k) for k in results["torus"])
    ax2.loglog(tf, [results["torus"][str(f)]["N_theta_max_rel_error"] for f in tf],
               "o-", label="torus, arc meridians")
    ax2.loglog(tf, [results["torus"][str(f)]["N_phi_max_rel_error"] for f in tf],
               "s--", label="torus, $N_\\phi$")
    ax2.loglog(tf, [results["torus"][str(f)]["faceted"]["N_theta_max_rel_error"]
                    for f in tf], "^-", label="torus, faceted chords")
    ax2.axhline(0.03, color="r", ls=":", label="3% accuracy gate")
    ax2.set_xlabel("meridional elements per 90 deg")
    ax2.set_ylabel("max rel. error")
    ax2.set_title("Documented limitation: general curved meridians\n"
                  "(error is mesh independent $\\Rightarrow$ formulation, not resolution)")
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT, "axisym_shell_limitation.png"), dpi=130)

    # ---------------- console table ---------------- #
    def row(name, rec, key, fmt="%.3e"):
        print(f"  {name:<38s} {key:<26s} {fmt % rec[key]}")

    print("== C1 cylinder (membrane exact) ==")
    for name in ("cylinder_open", "cylinder_closed"):
        row(name, results[name], "N_theta_max_rel_error")
        row(name, results[name], "w_max_rel_error")
    print("== C2 sphere, circular-arc meridians (exact pR/2 = 5 N/m) ==")
    for f in sorted(int(k) for k in results["sphere"]):
        rec = results["sphere"][str(f)]
        print(f"  facets={f:<3d} N_theta {rec['N_theta_max_rel_error']:.3e}  "
              f"N_phi {rec['N_phi_max_rel_error']:.3e}  w {rec['w_rel_error']:.3e}")
    print("== C3 clamped cylinder vs independent closed form ==")
    for f in sorted(int(k) for k in results["clamped_cylinder"]):
        rec = results["clamped_cylinder"][str(f)]
        print(f"  facets={f:<4d} L2 {rec['w_rel_l2_error']:.3e} "
              f"max/w_mem {rec['w_max_scaled_error']:.3e}")
    print("== C5 documented limitation (general curved meridians) ==")
    print(f"  cone N_theta error {results['cone']['N_theta_max_rel_error']:.3e}")
    for f in sorted(int(k) for k in results["torus"]):
        rec = results["torus"][str(f)]
        print(f"  torus facets={f:<3d} N_theta {rec['N_theta_max_rel_error']:.3e} "
              f"N_phi {rec['N_phi_max_rel_error']:.3e}")
    return results


if __name__ == "__main__":
    main()