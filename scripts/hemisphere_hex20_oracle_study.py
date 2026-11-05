#!/usr/bin/env python3
"""Independent stress oracle for the MacNeal--Harder 18-degree-hole hemisphere.

The production path recovers shell surface stresses from a flat-facet shell4
mesh and, until now, had no independent reference for them
(``stress_accuracy_qualified=False``, "recovered shell stresses have no
independent oracle").  This script builds that reference from a completely
different mechanical model -- a three-dimensional HEX20 solid of revolution --
and reports:

1. the oracle's own accuracy budget against two closed-form baselines
   (plane-strain Lamé ring, thin-strip bending at the benchmark aspect ratio);
2. the loaded-point displacement of the benchmark versus the published
   ``0.0924`` and versus the shell4 answer, with mesh convergence;
3. the outer/inner surface von Mises stress at matched stations, oracle against
   shell4, region by region -- including the region next to the point loads
   where both solutions are expected to diverge;
4. a JSON evidence file and two figures.

Run::

    OMP_NUM_THREADS=4 PYTHONPATH=src python scripts/hemisphere_hex20_oracle_study.py
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import torch

from tensorfem.hemisphere_benchmark_fields import hemisphere_fields
from tensorfem.hemisphere_hex20_oracle import (MH_REFERENCE_DISPLACEMENT, shell_surface_stress_table,
                                               von_mises)
from tensorfem.hemisphere_hex20_reference import (hemisphere_hex20_model, lame_ring_model,
                                                  thin_strip_bending_validation)

STATIONS = {
    # (label, polar-angle range in degrees, azimuth range in degrees)
    "interior": (30.0, 80.0, 15.0, 75.0),
    "near_hole_edge": (18.0, 30.0, 15.0, 75.0),
    "near_equator_load": (80.0, 90.0, 0.0, 90.0),
}


def _phi_theta(nodes: torch.Tensor, elements: torch.Tensor):
    """Polar angle and azimuth of every element centre of the shell4 mesh."""
    nodes = torch.as_tensor(nodes, dtype=torch.float64)
    elements = torch.as_tensor(elements, dtype=torch.long)
    centre = nodes[elements].mean(1)
    radius = torch.linalg.vector_norm(centre, dim=1)
    phi = torch.arccos(torch.clamp(centre[:, 2] / radius, -1.0, 1.0))
    theta = torch.arctan2(centre[:, 1], centre[:, 0])
    return torch.rad2deg(phi), torch.rad2deg(theta), radius


def _oracle_studies(meshes):
    """Loaded-point displacement and surface stresses for each oracle mesh."""
    studies = []
    for nphi, ntheta, nz in meshes:
        started = time.time()
        result, probe = hemisphere_hex20_model(nphi, ntheta, nz)
        displacement = float(result.displacement[3 * probe])
        table = shell_surface_stress_table(result, nphi, ntheta, nz, outer=True)
        inner = shell_surface_stress_table(result, nphi, ntheta, nz, outer=False)
        studies.append({
            "mesh": {"nphi": nphi, "ntheta": ntheta, "nz": nz, "nodes": len(result.nodes),
                     "elements": len(result.elements), "dofs": 3 * len(result.nodes)},
            "probe_displacement": displacement,
            "probe_relative_error": abs(displacement / MH_REFERENCE_DISPLACEMENT - 1.0),
            "equilibrium": result.equilibrium(),
            "wall_seconds": time.time() - started,
            "_outer": table,
            "_inner": inner,
        })
    return studies


def _match(shell_phi, shell_theta, eval_phi, eval_theta) -> torch.Tensor:
    """Nearest oracle station (in angle) for every shell4 element centre."""
    dphi = (shell_phi[:, None] - eval_phi[None, :]).abs()
    dtheta = (shell_theta[:, None] - eval_theta[None, :]).abs()
    distance = dphi ** 2 + dtheta ** 2
    return distance.argmin(1)


def _compare(shell, table_outer, table_inner):
    """Region-wise von Mises comparison of shell4 against the oracle."""
    phi_s, theta_s, _ = _phi_theta(shell["nodes"], shell["elements"])
    outer_index = _match(phi_s, theta_s, torch.rad2deg(table_outer["polar_angle"]),
                         torch.rad2deg(table_outer["azimuth"]))
    inner_index = _match(phi_s, theta_s, torch.rad2deg(table_inner["polar_angle"]),
                         torch.rad2deg(table_inner["azimuth"]))
    # shell4 records both surfaces per element; take the larger as the outer one
    shell_vm = torch.as_tensor(shell["von_mises_top_bottom"], dtype=torch.float64).max(1).values
    oracle_vm = table_outer["von_mises"][outer_index]
    oracle_signed_max = torch.stack((table_outer["von_mises"][outer_index],
                                     table_inner["von_mises"][inner_index]), 1).max(1).values
    relative = (shell_vm - oracle_vm).abs() / oracle_vm
    regions = {}
    for name, (p0, p1, t0, t1) in STATIONS.items():
        mask = ((phi_s >= p0) & (phi_s <= p1) & (theta_s >= t0) & (theta_s <= t1))
        if not bool(mask.any()):
            continue
        regions[name] = {
            "elements": int(mask.sum()),
            "oracle_von_mises_mean": float(oracle_vm[mask].mean()),
            "shell4_von_mises_mean": float(shell_vm[mask].mean()),
            "mean_relative_error": float(relative[mask].mean()),
            "max_relative_error": float(relative[mask].max()),
            "max_absolute_relative_error_against_two_sided_oracle":
                float(((shell_vm - oracle_signed_max).abs() / oracle_signed_max)[mask].max()),
        }
    return {
        "regions": regions,
        "global_mean_relative_error": float(relative.mean()),
        "global_max_relative_error": float(relative.max()),
        "median_relative_error": float(relative.median()),
        "stations": {"phi": phi_s.tolist(), "theta": theta_s.tolist(),
                     "shell4_von_mises": shell_vm.tolist(),
                     "oracle_von_mises": oracle_vm.tolist(),
                     "oracle_von_mises_inner": table_inner["von_mises"][inner_index].tolist(),
                     "relative_error": relative.tolist()},
    }


def _figures(out_dir: Path, studies, shell_runs, comparisons, meridian_theta=45.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []

    # 1. displacement convergence
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    sizes = [max(s["mesh"]["nphi"], s["mesh"]["ntheta"]) for s in studies]
    errors = [100 * s["probe_relative_error"] for s in studies]
    ax.plot(sizes, errors, "o-", label="HEX20 solid oracle (nz=1)")
    for n, record in shell_runs.items():
        ax.plot([n], [100 * record["probe"]["value"] / 0.0924 - 100], "s",
                label="shell4 flat facet (n=%d)" % n)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.axhspan(-3, 3, color="0.85", label="+/-3% gate")
    ax.set_xlabel("in-plane mesh count")
    ax.set_ylabel("loaded-point displacement error vs 0.0924 [%]")
    ax.set_title("MacNeal-Harder 18 deg hole: displacement convergence")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out_dir / "hemisphere-oracle-displacement-convergence.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # 2. surface von Mises along a meridian away from both point loads
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    finest = comparisons[max(comparisons)]
    phi = torch.tensor(finest["stations"]["phi"])
    theta = torch.tensor(finest["stations"]["theta"])
    pick = (theta - meridian_theta).abs() < 2.5
    order = phi[pick].argsort()
    ax.plot(phi[pick][order], torch.tensor(finest["stations"]["oracle_von_mises"])[pick][order],
            "o-", ms=3, label="HEX20 solid oracle (independent)")
    ax.plot(phi[pick][order], torch.tensor(finest["stations"]["shell4_von_mises"])[pick][order],
            "s--", ms=3, label="shell4 flat facet")
    ax.set_xlabel("polar angle phi [deg]")
    ax.set_ylabel("outer-surface von Mises stress")
    ax.set_title("Surface stress along the theta=45 deg meridian")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out_dir / "hemisphere-oracle-surface-stress.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)

    # 3. region-wise error map
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    rel = torch.tensor(finest["stations"]["relative_error"]) * 100
    sc = ax.scatter(theta, phi, c=rel.clamp(max=50), s=14, cmap="magma")
    fig.colorbar(sc, ax=ax, label="|shell4 - oracle| / oracle [%] (clipped 50)")
    ax.set_xlabel("azimuth theta [deg]")
    ax.set_ylabel("polar angle phi [deg]")
    ax.set_title("Surface von Mises error map at matched stations")
    path = out_dir / "hemisphere-oracle-error-map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    made.append(path)
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--meshes", default="20,24,1;28,36,1;34,42,1")
    parser.add_argument("--shell-meshes", default="32,40")
    parser.add_argument("--out", default="benchmarks/tutorials/hemisphere-hex20-oracle.json")
    parser.add_argument("--figures", default="docs/assets/hemisphere-hex20-oracle")
    args = parser.parse_args(argv)

    meshes = [tuple(int(v) for v in block.split(",")) for block in args.meshes.split(";")]
    shell_meshes = [int(v) for v in args.shell_meshes.split(",")]

    report = {
        "schema": "tensorfem.hemisphere-hex20-oracle.v1",
        "case": "MacNeal-Harder hemisphere with an 18-degree hole",
        "oracle": "three-dimensional HEX20 solid of revolution (no shell theory)",
        "reference_displacement": MH_REFERENCE_DISPLACEMENT,
        "oracle_validation": {
            "lame_ring": lame_ring_model(4, 48, 2),
            "thin_strip_bending": [thin_strip_bending_validation(nx=nx, ny=1, nz=nz)
                                   for nx, nz in ((16, 2), (24, 2), (32, 3))],
        },
    }

    studies = _oracle_studies(meshes)
    report["oracle_meshes"] = [{k: v for k, v in s.items() if not k.startswith("_")}
                               for s in studies]

    shell_runs, comparisons = {}, {}
    for n in shell_meshes:
        record = hemisphere_fields(n)
        shell_runs[n] = record
        table_outer = studies[-1]["_outer"]
        table_inner = studies[-1]["_inner"]
        comparisons[n] = _compare(record, table_outer, table_inner)
    report["shell4"] = {
        str(n): {"mesh": n, "probe": record["probe"], "validation": record["validation"],
                 "comparison_against_finest_oracle": comparisons[n],
                 "stress_accuracy_qualified": False}
        for n, record in shell_runs.items()
    }
    report["stress_accuracy_qualified"] = False
    report["scope"] = (
        "independent von Mises comparison at matched surface stations; the shell4 "
        "stress flag stays False because the agreement is reported, not silently "
        "turned into a pass, and because the ratio is not < 3% in every region")

    figures = _figures(Path(args.figures), studies, shell_runs, comparisons)
    report["figures"] = [str(p) for p in figures]
    report["generated_by"] = "scripts/hemisphere_hex20_oracle_study.py"
    try:
        report["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        report["git_commit"] = "unknown"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({n: report["shell4"][str(n)]["comparison_against_finest_oracle"]["regions"]
                      for n in shell_meshes}, indent=1))
    print("wrote", out, "and", len(figures), "figures")
    return 0


if __name__ == "__main__":
    sys.exit(main())