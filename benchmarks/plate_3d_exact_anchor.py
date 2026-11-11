"""Programmatic evidence for ``docs/plate-3d-exact-anchor.md``.

Runs the exact 3-D elasticity anchor, the FSDT and Kirchhoff Navier values, and
the HEX20 solid-element bridge; writes
``docs/assets/plate_3d_exact/*.json`` and the report figures.  Every number in
the document is read back from this JSON, nothing is typed by hand.

Usage::

    python benchmarks/plate_3d_exact_anchor.py            # full evidence
    python benchmarks/plate_3d_exact_anchor.py --quick    # skip the finest FE mesh
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfem.plate_3d_exact import (  # noqa: E402
    SSPlateExact3D,
    fsdt_static_navier,
    kirchhoff_static_navier,
    solve_hex20_ss_plate,
)

E = 200.0e9
NU = 0.3
Q0 = 1.0e6
A = 1.0
OUT = ROOT / "docs" / "assets" / "plate_3d_exact"


def fsdt_sigma_xx_profile(plate: SSPlateExact3D, zs, x=None, y=None):
    """FSDT in-plane stress ``sigma_xx(z)``: strictly linear through the thickness."""
    fsdt = fsdt_static_navier(plate.a, plate.b, plate.h, E, NU, Q0, plate.m, plate.n)
    al, be = plate.alpha, plate.beta
    x = 0.5 * plate.a if x is None else x
    y = 0.5 * plate.b if y is None else y
    kx = al * fsdt["psi_x"] * math.cos(al * x) * math.sin(be * y)
    ky = be * fsdt["psi_y"] * math.sin(al * x) * math.cos(be * y)
    common = math.sin(al * x) * math.sin(be * y)
    fac = -E / (1.0 - NU**2) * (kx + NU * ky) * common
    return fac * zs, fsdt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---------------------------------------------------------------- a/h sweep
    ratios = [100.0, 50.0, 20.0, 10.0, 8.0, 5.0]
    sweep = []
    for ratio in ratios:
        plate = SSPlateExact3D(A, A, A / ratio, E, NU, Q0)
        w_k = kirchhoff_static_navier(A, A, plate.h, E, NU, Q0)["deflection"]
        fsdt = fsdt_static_navier(A, A, plate.h, E, NU, Q0)
        row = {
            "a_over_h": ratio,
            "w_exact_3d": plate.center_deflection(),
            "w_kirchhoff": w_k,
            "w_fsdt": fsdt["deflection"],
            "exact_over_kirchhoff": plate.center_deflection() / w_k,
            "fsdt_over_kirchhoff": fsdt["deflection"] / w_k,
            "fsdt_error_vs_3d_pct": 100.0 * (fsdt["deflection"] / plate.center_deflection() - 1.0),
            "shear_parameter": fsdt["shear_parameter"],
        }
        sweep.append(row)
        print(f"a/h={ratio:6.1f}  w3d/wK={row['exact_over_kirchhoff']:.6f}  "
              f"FSDT-vs-3D={row['fsdt_error_vs_3d_pct']:+.2f}%", flush=True)

    # ------------------------------------------------- self-consistency of the anchor
    ref = SSPlateExact3D(A, A, A / 10.0, E, NU, Q0)
    checks = {
        "equilibrium": ref.equilibrium_residual(samples=2),
        "faces": ref.face_report(),
        "edges": ref.edge_report(),
    }

    # -------------------------------------------------------- thin-limit convergence
    limit = []
    for ratio in (12.5, 25.0, 100.0):
        plate = SSPlateExact3D(A, A, A / ratio, E, NU, Q0)
        w_k = kirchhoff_static_navier(A, A, plate.h, E, NU, Q0)["deflection"]
        limit.append({"a_over_h": ratio, "relative_gap": plate.center_deflection() / w_k - 1.0})
    for prev, cur in zip(limit, limit[1:]):
        prev["gap_ratio_to_next"] = prev["relative_gap"] / cur["relative_gap"]

    # ------------------------------------------------------------------ FE bridges
    meshes = [(4, 4, 2), (6, 6, 2)] if args.quick else [(4, 4, 2), (6, 6, 2), (8, 8, 3)]
    fe_rows = []
    fe_fields = {}
    for ratio, (nx, ny, nz) in [(10.0, m) for m in meshes] + [(5.0, (6, 6, 2)), (20.0, (6, 6, 2))]:
        plate = SSPlateExact3D(A, A, A / ratio, E, NU, Q0)
        t1 = time.time()
        fe = solve_hex20_ss_plate(A, A, plate.h, E, NU, Q0, nx=nx, ny=ny, nz=nz)
        w_exact = plate.center_deflection()
        row = {
            "a_over_h": ratio,
            "mesh": [nx, ny, nz],
            "n_dofs": fe["n_dofs"],
            "n_elements": fe["n_elements"],
            "w_fe": fe["centre_deflection"],
            "w_exact_3d": w_exact,
            "error_pct": 100.0 * (fe["centre_deflection"] / w_exact - 1.0),
            "seconds": round(time.time() - t1, 1),
        }
        fe_rows.append(row)
        if ratio == 10.0:
            nodes = fe["nodes"]
            disp = fe["displacement"]
            top = (nodes[:, 2] - plate.h).abs() < 1e-9
            yc = (nodes[:, 1] - 0.5 * A).abs() < 1e-9
            line = top & yc
            order = torch.argsort(nodes[line, 0])
            fe_fields[f"nx{nx}_nz{nz}"] = {
                "x": nodes[line, 0][order].tolist(),
                "w": (-disp[2::3][line][order]).tolist(),
            }
        print(f"a/h={ratio:5.1f} mesh={nx}x{ny}x{nz} ndof={fe['n_dofs']:6d} "
              f"err={row['error_pct']:+.3f}% ({row['seconds']}s)", flush=True)

    # ------------------------------------------------------ through-thickness fields
    zs = torch.linspace(-0.5 * ref.h, 0.5 * ref.h, 41, dtype=torch.float64)
    xt = torch.full_like(zs, 0.5 * ref.a)
    yt = torch.full_like(zs, 0.5 * ref.b)
    sxx = ref.stress(xt, yt, zs)[0]
    txz = ref.stress(xt, yt, zs)[4]
    fsdt_sxx, fsdt_meta = fsdt_sigma_xx_profile(ref, zs.numpy())
    linear = float(sxx[-1]) * zs.numpy() / (0.5 * ref.h)
    profile = {
        "z": zs.tolist(),
        "sigma_xx_3d": sxx.tolist(),
        "sigma_xx_fsdt": fsdt_sxx.tolist(),
        "sigma_xx_linear_reference": linear.tolist(),
        "tau_xz_3d": txz.tolist(),
        "tau_xz_fsdt_constant": float(
            fsdt_meta["shear_stiffness"] * (ref.alpha * fsdt_meta["deflection"] - fsdt_meta["psi_x"])
        ),
        "sigma_xx_max_deviation_from_linear": float((sxx - torch.as_tensor(linear)).abs().max()) / abs(float(sxx[-1])),
        "tau_xz_peak_3d": float(txz.abs().max()),
        "tau_xz_peak_over_q0": float(txz.abs().max()) / Q0,
    }

    evidence = {
        "meta": {
            "script": "benchmarks/plate_3d_exact_anchor.py",
            "E": E, "nu": NU, "q0": Q0, "a": A, "b": A,
            "load": "p = q0 sin(pi x/a) sin(pi y/b) on the top face",
            "wall_seconds": round(time.time() - t0, 1),
            "quick": args.quick,
        },
        "thickness_sweep": sweep,
        "thin_limit": limit,
        "anchor_self_check": checks,
        "hex20_bridge": fe_rows,
        "through_thickness": profile,
    }
    (OUT / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True))

    # ------------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ratios_plot = [r["a_over_h"] for r in sweep][::-1]
    ax.plot(ratios_plot, [r["exact_over_kirchhoff"] for r in sweep][::-1], "o-", label="3-D elasticity (exact)")
    ax.plot(ratios_plot, [r["fsdt_over_kirchhoff"] for r in sweep][::-1], "s--", label="FSDT Navier (exact)")
    for row in fe_rows:
        exact = row["w_exact_3d"]
        fe = row["w_fe"]
        wk = kirchhoff_static_navier(A, A, A / row["a_over_h"], E, NU, Q0)["deflection"]
        ax.plot([row["a_over_h"]], [fe / wk], "x", color="crimson", ms=9,
                label="HEX20 FE" if row is fe_rows[0] else None)
    ax.axhline(1.0, color="k", lw=0.8, label="Kirchhoff (thin plate)")
    ax.set_xscale("log")
    ax.set_xlabel("a / h")
    ax.set_ylabel(r"$w_c / w_c^{Kirchhoff}$")
    ax.set_title("Thick-plate deflection: 3-D exact vs FSDT vs HEX20 FE")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_deflection_vs_thickness.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    conv = [r for r in fe_rows if r["a_over_h"] == 10.0]
    labels = [f"{r['mesh'][0]}x{r['mesh'][1]}x{r['mesh'][2]}" for r in conv]
    ax.bar(labels, [r["error_pct"] for r in conv], color="steelblue")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhline(3.0, color="crimson", ls="--", lw=1.0, label="3 % acceptance gate")
    ax.set_xlabel("HEX20 mesh (nx x ny x nz), a/h = 10")
    ax.set_ylabel(r"$(w_{FE}-w_{3D})/w_{3D}$  [%]")
    ax.set_title("HEX20 convergence against the 3-D exact solution")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_hex20_convergence.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    zz = torch.as_tensor(profile["z"]).numpy()
    axes[0].plot([v / 1e6 for v in profile["sigma_xx_3d"]], zz, "-", label="3-D exact")
    axes[0].plot([v / 1e6 for v in profile["sigma_xx_fsdt"]], zz, "--", label="FSDT (linear)")
    axes[0].set_xlabel(r"$\sigma_{xx}$ [MPa]")
    axes[0].set_ylabel("z [m]")
    axes[0].set_title("In-plane stress, a/h = 10, plate centre")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot([v / 1e6 for v in profile["tau_xz_3d"]], zz, "-", label="3-D exact")
    axes[1].axvline(profile["tau_xz_fsdt_constant"] / 1e6, color="tab:orange", ls="--", label="FSDT (constant)")
    axes[1].set_xlabel(r"$\tau_{xz}$ [MPa]")
    axes[1].set_ylabel("z [m]")
    axes[1].set_title(r"Transverse shear, $z$-profile")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_through_thickness_stress.png", dpi=150)
    plt.close(fig)

    if fe_fields:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        xs = torch.linspace(0.0, A, 201, dtype=torch.float64)
        exact_line = [float(ref.deflection(xs[i], 0.5 * A, 0.5 * ref.h)) for i in range(len(xs))]
        ax.plot(xs.numpy(), exact_line, "k-", lw=1.5, label="3-D exact, y = b/2")
        for key, data in fe_fields.items():
            ax.plot(data["x"], data["w"], "o--", ms=3, label=f"HEX20 {key}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("w [m] (downward)")
        ax.set_title("Deflection along y = b/2, a/h = 10")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "fig4_deflection_profile.png", dpi=150)
        plt.close(fig)

    print(json.dumps(sweep, indent=2))
    print("evidence written to", OUT / "evidence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())