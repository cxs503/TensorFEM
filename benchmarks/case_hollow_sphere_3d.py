"""CASE-HOLLOW-SPHERE-3D driver: 3-D thick-walled sphere under internal pressure.

Runs the HEX8 baseline ladder and the HEX20 (quadratic, curved-boundary) ladder
against the closed-form Lame solution, scores the case probes against the <5%
acceptance line, writes a machine-readable artifact and two figures.

Usage::

    PYTHONPATH=$PWD/src python3 benchmarks/case_hollow_sphere_3d.py --quick
    PYTHONPATH=$PWD/src python3 benchmarks/case_hollow_sphere_3d.py \
        --out artifacts/case_hollow_sphere_3d

Exit status is 0 only when the finest mesh of every ladder passes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from tensorfem.hollow_sphere_3d import (CASE, TOLERANCE, convergence,
                                        lame_sphere, shell_volume,
                                        solve_hollow_sphere, strain_energy)

# Baseline ladder: the trilinear family used by the existing sphere benchmark.
HEX8_LADDER = [(4, 2), (6, 2)]
# Selected method: 20-node serendipity bricks, curved-boundary projection.
HEX20_LADDER = [(4, 2), (6, 2)]
QUICK_HEX8 = [(6, 2), (8, 2)]
QUICK_HEX20 = [(2, 2), (4, 2)]
# The case verdict is carried by the selected method; HEX8 is the baseline.
SELECTED_LADDER = "hex20"


def run_ladder(order, levels, case=CASE, keep_arrays=False):
    rows = []
    for n, radial_layers in levels:
        started = time.time()
        result = solve_hollow_sphere(n, radial_layers, order=order, case=case,
                                     keep_arrays=keep_arrays)
        result["wall_seconds"] = time.time() - started
        rows.append(result)
        mesh = result["mesh"]
        print(f"  order={order} n={n} layers={radial_layers} "
              f"nodes={mesh['nodes']} dofs={mesh['degrees_of_freedom']} "
              f"worst={result['metrics']['worst_relative_error']:.4%} "
              f"passed={result['passed']} [{result['wall_seconds']:.1f}s]",
              flush=True)
    return rows


def summarise(rows):
    """Ref-vs-got table for the finest mesh plus the convergence verdict."""
    finest = rows[-1]
    metrics = finest["metrics"]
    table = []
    for name, row in metrics["probes"].items():
        table.append({"quantity": name, "probe": row["probe"], "unit": row["unit"],
                      "ref": row["ref"], "got": row["got"], "error": row["error"],
                      "tolerance": TOLERANCE, "pass": row["error"] < TOLERANCE})
    for name, row in metrics["secondary_probes"].items():
        table.append({"quantity": name, "probe": row["probe"], "unit": row["unit"],
                      "ref": row["ref"], "got": row["got"], "error": row["error"],
                      "normalisation": row["normalisation"], "tolerance": TOLERANCE,
                      "pass": row["error"] < TOLERANCE})
    return {"finest_mesh": finest["mesh"], "table": table,
            "worst_relative_error": metrics["worst_relative_error"],
            "worst_quantity": metrics["worst_quantity"],
            "worst_secondary_error": metrics["worst_secondary_error"],
            "tolerance": TOLERANCE,
            "table_passed": bool(all(row["pass"] for row in table)),
            "convergence": convergence(rows),
            "equilibrium": finest["equilibrium"],
            "loads": finest["loads"],
            "solver": finest["solver"],
            "energy": finest["energy"],
            "lame_strain_energy_j": strain_energy(CASE),
            "shell_volume_m3": shell_volume(CASE),
            "passed": bool(finest["passed"] and all(row["pass"] for row in table))}


def layer_profile(rows):
    """Element-level FE profile vs Lame on the finest mesh (for the figures)."""
    finest = rows[-1]
    fields = finest["metrics"]["fields"]
    radius = torch.tensor(fields["radius"], dtype=torch.float64)
    order = torch.argsort(radius)
    radius = radius[order]
    reference = lame_sphere(radius)
    return {"radius": radius.tolist(),
            "radial_stress_fe": torch.tensor(fields["radial_stress"])[order].tolist(),
            "radial_stress_ref": reference["radial_stress"].tolist(),
            "hoop_stress_fe": torch.tensor(fields["hoop_stress"])[order].tolist(),
            "hoop_stress_ref": reference["hoop_stress"].tolist(),
            "von_mises_fe": torch.tensor(fields["von_mises"])[order].tolist(),
            "von_mises_ref": reference["von_mises"].tolist()}


def figures(ladders, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    produced = []
    # 1) radial profiles on the finest mesh of each element family
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    for order, rows, style in ((1, ladders["hex8"], "o-"), (2, ladders["hex20"], "s--")):
        profile = layer_profile(rows)
        r = np.array(profile["radius"])
        label = f"FE {'HEX8' if order == 1 else 'HEX20'} n={rows[-1]['mesh']['face_divisions']}"
        axes[0].plot(r, np.array(profile["radial_stress_fe"]) / 1e6, style, ms=3, label=label)
        axes[1].plot(r, np.array(profile["hoop_stress_fe"]) / 1e6, style, ms=3, label=label)
        axes[2].plot(r, np.array(profile["von_mises_fe"]) / 1e6, style, ms=3, label=label)
    rr = np.linspace(CASE["inner_radius_m"], CASE["outer_radius_m"], 200)
    ref = lame_sphere(torch.tensor(rr))
    for ax, key in zip(axes, ("radial_stress", "hoop_stress", "von_mises")):
        ax.plot(rr, ref[key].numpy() / 1e6, "k-", lw=2, label="Lame (exact)")
        ax.set_xlabel("radius r [m]")
        ax.set_ylabel(f"{key.replace('_', ' ')} [MPa]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_title("CASE-HOLLOW-SPHERE-3D: stress vs radius")
    fig.tight_layout()
    path = out_dir / "profiles.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    produced.append(path)

    # 2) convergence of the worst probe error
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for order, rows, style in ((1, ladders["hex8"], "o-"), (2, ladders["hex20"], "s--")):
        h = [1.0 / row["mesh"]["face_divisions"] for row in rows]
        err = [row["metrics"]["worst_relative_error"] for row in rows]
        ax.loglog(h, err, style, label=f"{'HEX8' if order == 1 else 'HEX20'} worst probe")
    ax.axhline(TOLERANCE, color="r", ls=":", label="acceptance 5%")
    ax.set_xlabel("mesh size h ~ 1/n (cube-face divisions)")
    ax.set_ylabel("worst probe relative error")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    ax.set_title("convergence vs analytic solution")
    fig.tight_layout()
    path = out_dir / "convergence.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    produced.append(path)

    # 3) meridional field: FE von Mises vs analytic on a cut plane
    finest = ladders["hex20"][-1]
    arrays = finest["metrics"]["fields"].get("cut_plane")
    if arrays:
        x = np.array(arrays["x"])[:, 1]
        z = np.array(arrays["x"])[:, 2]
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
        for ax, key, title in ((axes[0], "von_mises_fe", "FE HEX20 von Mises"),
                               (axes[1], "von_mises_ref", "Lame von Mises")):
            cs = ax.tricontourf(z, x, np.array(arrays[key]) / 1e6, levels=18, cmap="viridis")
            fig.colorbar(cs, ax=ax, label="von Mises [MPa]")
            ax.set_aspect("equal")
            ax.set_xlabel("z [m]")
            ax.set_ylabel("y [m]")
            ax.set_title(title)
        fig.suptitle("meridional cut (x = 0), inner pressure 1 MPa")
        fig.tight_layout()
        path = out_dir / "field_cut.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        produced.append(path)
    return [str(p) for p in produced]


def meridional_cut(result):
    """Centroid (y, z) samples on the x ~ 0 plane for the field figure."""
    arrays = result.get("arrays")
    if arrays is None:
        return None
    centres = arrays["centres"]
    mask = centres[:, 0].abs() < 0.12 * CASE["outer_radius_m"]
    if int(mask.sum()) < 12:
        return None
    ref = lame_sphere(torch.linalg.vector_norm(centres[mask], dim=1))["von_mises"]
    return {"x": centres[mask].tolist(),
            "von_mises_fe": arrays["von_mises"][mask].tolist(),
            "von_mises_ref": ref.tolist()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="reduced ladders")
    parser.add_argument("--out", default="artifacts/case_hollow_sphere_3d",
                        help="artifact directory")
    parser.add_argument("--skip-hex20", action="store_true")
    args = parser.parse_args(argv)
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("CASE-HOLLOW-SPHERE-3D", flush=True)
    print(f"  analytic strain energy {strain_energy(CASE):.6g} J, "
          f"shell volume {shell_volume(CASE):.6g} m^3", flush=True)
    ladders = {}
    print("HEX8 ladder", flush=True)
    ladders["hex8"] = run_ladder(1, QUICK_HEX8 if args.quick else HEX8_LADDER)
    print("HEX20 ladder", flush=True)
    ladders["hex20"] = run_ladder(2, QUICK_HEX20 if args.quick else HEX20_LADDER,
                                  keep_arrays=True)

    cut = meridional_cut(ladders["hex20"][-1])
    if cut is not None:
        ladders["hex20"][-1]["metrics"]["fields"]["cut_plane"] = cut

    artifact = {
        "case_code": "HOLLOW-SPHERE-3D",
        "repo": "TensorFEM",
        "problem": CASE,
        "reference": {
            "source": "Timoshenko & Goodier, Theory of Elasticity, 3rd ed. (1970) Sec. 5.6; Lame (1852)",
            "formula": "sigma_r = A - B/r^3, sigma_t = A + B/(2 r^3), u_r = [(1-2nu) A r + (1+nu) B/(2 r^2)]/E, A = p a^3/(b^3-a^3), B = A b^3",
        },
        "tolerance": TOLERANCE,
        "ladders": {name: [{"mesh": r["mesh"], "metrics": r["metrics"],
                            "energy": r["energy"], "equilibrium": r["equilibrium"],
                            "loads": r["loads"], "solver": r["solver"],
                            "wall_seconds": r["wall_seconds"], "passed": r["passed"]}
                           for r in rows]
                    for name, rows in ladders.items()},
        "summary": {name: summarise(rows) for name, rows in ladders.items()},
        "selection": ("HEX20 with curved-boundary projection: same cube-sphere topology as HEX8, "
                      "quadratic radial/hoop resolution, identical pressure load generator"),
    }
    metadata = {"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "torch": torch.__version__,
                "platform": sys.platform}
    artifact["environment"] = metadata
    (out_dir / "case_hollow_sphere_3d.json").write_text(json.dumps(artifact, indent=1))
    figures(ladders, out_dir)

    artifact["verdict"] = {
        "selected_ladder": SELECTED_LADDER,
        "selected_passed": artifact["summary"][SELECTED_LADDER]["passed"],
        "baseline_reached_tolerance": artifact["summary"]["hex8"]["worst_relative_error"] < TOLERANCE,
        "tolerance": TOLERANCE,
    }
    passed = bool(artifact["verdict"]["selected_passed"])
    print(json.dumps({name: {"worst": row["worst_relative_error"],
                             "worst_quantity": row["worst_quantity"],
                             "passed": row["passed"]}
                      for name, row in artifact["summary"].items()}, indent=1))
    print("CASE VERDICT:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())