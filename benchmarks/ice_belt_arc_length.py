"""Arc-length (Riks) limit-load benchmark for the ice-belt collapse surface.

Produces ``artifacts/ice_belt_collapse/arc_length.json`` (all numbers,
programmatic, no transcribed values) and the figures

* ``arc_plate_field.png``  -- load-shortening path of the imperfect plate field
  with the Riks peak, the load-control corroboration and the Smith/Faulkner
  ultimate (criterion: ``|peak/Smith-1| <= 10 %``);
* ``arc_ice_limit.png``    -- pressure-deflection path of the ice-belt bay with
  the Riks plastic limit, the twice-elastic-slope design limit and the code
  strip yield line;
* ``arc_mesh.png``         -- limit load vs mesh (h-refinement of the
  discretisation-restricted upper bound).

The mesh sets are chosen so the same script can run one mesh per process
(``--shard i/n``) for a cluster sweep::

    PYTHONPATH=src python3 benchmarks/ice_belt_arc_length.py            # default set
    PYTHONPATH=src python3 benchmarks/ice_belt_arc_length.py --shard 0/2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfem.ice_belt_collapse import (  # noqa: E402
    arc_length_status, ice_belt_case, ice_belt_limit_pressure_arc_case,
    plate_field_collapse_arc_case,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ART = os.path.join(ROOT, "artifacts", "ice_belt_collapse")
FIG = os.path.join(ROOT, "docs", "assets", "ice_belt_collapse")

PLATE_MESHES = [(16, 8), (24, 12), (32, 16)]
ICE_MESHES = [(24, 8), (32, 10), (48, 16)]


def _plate(mesh, dlam0=0.02, n_steps=140):
    nx, ny = mesh
    t0 = time.time()
    out = plate_field_collapse_arc_case(nx=nx, ny=ny, dlam0=dlam0, n_steps=n_steps)
    out["seconds"] = time.time() - t0
    out["mesh"] = [nx, ny]
    return out


def _ice(mesh, dlam0=0.03, n_steps=170):
    nx, ny = mesh
    t0 = time.time()
    out = ice_belt_limit_pressure_arc_case(nx=nx, ny=ny, dlam0=dlam0, n_steps=n_steps)
    out["seconds"] = time.time() - t0
    out["mesh"] = [nx, ny]
    return out


def run_benchmark(meshes=None, which=("plate", "ice"), shard=None, nshard=1) -> dict:
    meshes = meshes or {}
    plate_meshes = meshes.get("plate", PLATE_MESHES)
    ice_meshes = meshes.get("ice", ICE_MESHES)
    jobs = ([("plate", m) for m in plate_meshes if "plate" in which]
            + [("ice", m) for m in ice_meshes if "ice" in which])
    if shard is not None:
        jobs = [job for k, job in enumerate(jobs) if k % nshard == shard]
    out = {"plate": [], "ice": []}
    for kind, mesh in jobs:
        row = _plate(mesh) if kind == "plate" else _ice(mesh)
        out[kind].append(row)
        key = "sigma_fem_peak" if kind == "plate" else "limit_pressure"
        print(f"{kind} {mesh}: {key} = {row[key]:.6g} "
              f"rel_error {row['rel_error']:+.4f} "
              f"energy {row['energy_residual']:+.4%} "
              f"({row['seconds']:.1f} s)", flush=True)
    return out


def _load_existing():
    path = os.path.join(ART, "arc_length.json")
    if not os.path.exists(path):
        return {"plate": [], "ice": []}
    with open(path) as fh:
        return json.load(fh)


def merge_and_write(new: dict) -> dict:
    os.makedirs(ART, exist_ok=True)
    merged = _load_existing()
    for kind in ("plate", "ice"):
        have = {tuple(r["mesh"]): r for r in merged.get(kind, [])}
        for row in new.get(kind, []):
            have[tuple(row["mesh"])] = row
        rows = [have[k] for k in sorted(have)]
        for row in rows:                       # backfill from the stored path
            if "limit_status" not in row and "limit" in row:
                row["limit_status"] = arc_length_status(row["limit"])
        merged[kind] = rows
    with open(os.path.join(ART, "arc_length.json"), "w") as fh:
        json.dump(merged, fh, indent=1)
    print("wrote", os.path.join(ART, "arc_length.json"))
    return merged


def figures(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(FIG, exist_ok=True)
    plate = sorted(data.get("plate", []), key=lambda r: r["mesh"][0])
    ice = sorted(data.get("ice", []), key=lambda r: r["mesh"][0])
    if plate:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        row = plate[0]
        lam = np.asarray(row["path"]["lam"])
        w = -np.asarray(row["path"]["w"]) * 1e3
        ax.plot(w, lam * 355e6 / 1e6, "o-", ms=3, lw=1.1,
                label=f"Riks path {row['mesh'][0]}x{row['mesh'][1]}")
        ax.axhline(row["sigma_smith"] / 1e6, color="tab:red", ls="--",
                   label="Smith/Faulkner ultimate")
        ax.axhline(row["sigma_fem_peak"] / 1e6, color="tab:green", ls=":",
                   label="Riks peak (limit load)")
        ax.set_xlabel("plate-field centre deflection [mm]")
        ax.set_ylabel("mean compressive stress [MPa]")
        ax.set_title("Plate field under end compression")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(FIG, "arc_plate_field.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print("wrote", path)
    if ice:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        row = ice[0]
        lam = np.asarray(row["path"]["lam"]) * row["pressure_at_area"] / 1e3
        w = np.asarray(row["path"]["w"]) * 1e3
        ax.plot(w, lam, "o-", ms=3, lw=1.1,
                label=f"Riks path {row['mesh'][0]}x{row['mesh'][1]}")
        ax.axhline(row["code_pressure"] / 1e3, color="tab:red", ls="--",
                   label="code strip yield line 8Mp/b$^2$")
        ax.axhline(row["limit_pressure"] / 1e3, color="tab:green", ls=":",
                   label="Riks plastic limit")
        design = ice_belt_case(nx=row["mesh"][0], ny=row["mesh"][1], steps=30)
        ax.axhline(design["collapse_pressure"] / 1e3, color="tab:orange", ls="-.",
                   label="twice-elastic-slope design limit")
        ax.set_xlabel("bay centre deflection [mm]")
        ax.set_ylabel("ice pressure [kPa]")
        ax.set_title("Ice-belt bay under the IACS pressure-area load")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        path = os.path.join(FIG, "arc_ice_limit.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print("wrote", path)
    if len(plate) > 1 or len(ice) > 1:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        if plate:
            cells = [r["mesh"][0] * r["mesh"][1] for r in plate]
            ax.plot(cells, [r["sigma_fem_peak"] / r["sigma_smith"] - 1.0
                            for r in plate], "s-", label="plate field vs Smith")
        if ice:
            cells = [r["mesh"][0] * r["mesh"][1] for r in ice]
            ax.plot(cells, [r["rel_error"] for r in ice], "o-",
                    label="ice belt vs code yield line")
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_xscale("log")
        ax.set_xlabel("elements")
        ax.set_ylabel("relative deviation from the reference")
        ax.set_title("h-refinement of the limit load")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        path = os.path.join(FIG, "arc_mesh.png")
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print("wrote", path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None,
                    help="i/n: run every n-th mesh in this process")
    ap.add_argument("--kind", default="both", choices=("plate", "ice", "both"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    shard = None
    nshard = 1
    if args.shard:
        shard, nshard = (int(x) for x in args.shard.split("/"))
    which = ("plate",) if args.kind == "plate" else (
        ("ice",) if args.kind == "ice" else ("plate", "ice"))
    data = run_benchmark(which=which, shard=shard, nshard=nshard)
    merged = merge_and_write(data)
    if not args.no_figures:
        figures(merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())