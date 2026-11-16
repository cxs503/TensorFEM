#!/usr/bin/env python
"""Displacement-controlled ladder for GAP-PP-3-ICE-STRUCT-PLASTIC-SET.

The anchor's published limit-load evidence was measured on a *load-controlled*
ramp, which can only stop at a limit point and never walks the post-limit
branch; the gate documents
(``docs/CASE-FEM-PLATE-PATCH-PLASTIC-PATH-ADMISSIBILITY.md`` §5,
``docs/CASE-GAP-PP-3-3GRID-MATRIX-VERDICT.md`` §6) therefore ask for
displacement-controlled loading before the "limit load vs upper bound < 10 %"
criterion can be judged at all.

This driver runs the same case as the archived ladder with the centre
deflection as the control parameter
(:func:`tensorfem.plate_patch_plastic_displacement.run_patch_plastic_displacement_path`),
on a mesh ladder and a through-thickness layer ladder, and writes one JSON per
queue shard; ``--analyse`` merges them, computes the criteria block and the
figures.  Nothing is transcribed by hand.

Usage::

    PYTHONPATH=src python scripts/gap_pp3_displacement_ladder.py --plan-index 0
    PYTHONPATH=src python scripts/gap_pp3_displacement_ladder.py --analyse
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tensorfem.plate_patch_plastic_displacement import (  # noqa: E402
    run_patch_plastic_displacement_path,
)
from tensorfem.plate_patch_plastic_limit import (  # noqa: E402
    analytic_first_yield_pressure,
    ilyushin_sine_upper_bound,
    low_temperature_flow_stress,
    plastic_bound_bracket,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "outputs", "gap_pp3_displacement_ladder")

# --- case: identical to the archived 3-grid matrix -------------------------
A, B, T = 2.0, 1.0, 0.020
PATCH = (0.5, 0.5)
E_MOD, NU = 206e9, 0.3
GRADE, TEMPERATURE_C = "DH36", -20.0
HARDENING = 1000e6

RATIOS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)

#: One entry per queue shard.  ``tag`` names the ladder the run belongs to.
PLANS = [
    dict(tag="mesh", nx=8, ny=4, layers=6, hardening=HARDENING),
    dict(tag="mesh", nx=16, ny=8, layers=6, hardening=HARDENING),
    dict(tag="mesh", nx=24, ny=12, layers=6, hardening=HARDENING),
    dict(tag="layers", nx=16, ny=8, layers=4, hardening=HARDENING),
    dict(tag="layers", nx=16, ny=8, layers=12, hardening=HARDENING),
    dict(tag="ideal", nx=16, ny=8, layers=6, hardening=1e6),
]


def _key(plan: dict) -> str:
    return f"{plan['tag']}_nx{plan['nx']}x{plan['ny']}_L{plan['layers']}_H{plan['hardening']:.0e}"


def check_alignment(nx: int, ny: int) -> None:
    """Reject a mesh on which the centred patch is not element aligned."""
    u, v = PATCH
    for offset, size, length, n in ((0.5 * (A - u), u, A, nx),
                                    (0.5 * (B - v), v, B, ny)):
        step = length / n
        for point in (offset, offset + size):
            ratio = point / step
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    f"mesh {nx}x{ny} cannot carry the {u}x{v} m patch: edge at "
                    f"{point:.4f} m is off the element grid (step {step:.6f} m)")


def references() -> dict:
    flow = low_temperature_flow_stress(GRADE, TEMPERATURE_C)
    sy = flow.yield_stress
    first = analytic_first_yield_pressure(A, B, T, E_MOD, NU, sy, PATCH)
    bracket = plastic_bound_bracket(A, B, T, sy, PATCH)
    return {
        "yield_stress": sy,
        "first_yield_pressure": first["pressure"],
        "first_yield_sigma_eq_per_pa": first.get("sigma_eq_per_pa"),
        "yield_line_upper": yield_line_patch_upper_bound(A, B, T, sy, PATCH)["pressure"],
        "ilyushin_sine_upper": ilyushin_sine_upper_bound(A, B, T, sy, PATCH)["pressure"],
        "strip_lower": strip_lower_bound(A, B, T, sy, PATCH)["pressure"],
        "bracket": {k: v for k, v in bracket.items() if isinstance(v, (int, float))},
        "flow_stress": flow.as_dict(),
    }


def run_plan(index: int) -> dict:
    plan = PLANS[index]
    check_alignment(plan["nx"], plan["ny"])
    refs = references()
    path = run_patch_plastic_displacement_path(
        A, B, T, young=E_MOD, poisson=NU,
        yield_stress=refs["yield_stress"], hardening=plan["hardening"],
        patch=PATCH, nx=plan["nx"], ny=plan["ny"], layers=plan["layers"],
        deflection_ratios=RATIOS, unload_steps=14, geometric=False,
        inplane_restrained=True, max_iterations=25)
    record = dict(plan=dict(plan), key=_key(plan), case=dict(
        a=A, b=B, t=T, patch=list(PATCH), young=E_MOD, poisson=NU,
        grade=GRADE, temperature_c=TEMPERATURE_C), result=path.as_dict())
    path_out = os.path.join(OUT_DIR, _key(plan) + ".json")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(path_out, "w") as handle:
        json.dump(record, handle, indent=1)
    return record


def _successive(values: list[float]) -> dict:
    """Successive-grid change of a measured quantity (self-convergence)."""
    change = [abs(b - a) / abs(b) for a, b in zip(values, values[1:]) if b]
    out = {"values": values, "successive_change": change}
    if change:
        out["finest_change"] = change[-1]
    if len(change) >= 2 and change[0] > 0 and change[1] > 0:
        out["ratio"] = change[0] / change[1]
    return out


def analyse() -> dict:
    records, missing = [], []
    for index in range(len(PLANS)):
        path = os.path.join(OUT_DIR, _key(PLANS[index]) + ".json")
        if not os.path.exists(path):
            missing.append({"plan_index": index, "key": _key(PLANS[index]),
                            "status": "not_run"})
            continue
        with open(path) as handle:
            records.append(json.load(handle))
    if not records:
        raise SystemExit("no shard available: nothing to analyse")
    refs = references()
    ub = refs["yield_line_upper"]
    lb = refs["strip_lower"]
    sine = refs["ilyushin_sine_upper"]

    def block(record: dict) -> dict:
        res = record["result"]
        plateau = res["plateau_factor"]
        return {
            "key": record["key"],
            "mesh": [record["plan"]["nx"], record["plan"]["ny"]],
            "layers": record["plan"]["layers"],
            "hardening": record["plan"]["hardening"],
            "capacity_ratio": res["capacity_ratio"],
            "peak_factor": res["peak_factor"],
            "plateau_factor": plateau,
            "turning_point": res["limit_is_turning_point"],
            "tangent_ratio": res["tangent_ratio"],
            "first_yield_factor": res["first_yield_factor"],
            "first_yield_error": (
                None if res["first_yield_factor"] is None else
                (res["first_yield_factor"] - refs["first_yield_pressure"])
                / refs["first_yield_pressure"]),
            "plateau_vs_upper": (plateau - ub) / ub if plateau else None,
            "plateau_vs_ilyushin": (plateau - sine) / sine if plateau else None,
            "inside_bracket": bool(plateau and lb <= plateau <= ub),
            "permanent_deflection": res["permanent_deflection"],
            "permanent_set_valid": res["permanent_set_valid"],
            "peak_deflection": res["peak_deflection"],
            "peak_over_thickness": (res["peak_deflection"] / T
                                    if res["peak_deflection"] else None),
            "load_points": len(res["path"]),
            "steps_bisected": res["steps_bisected"],
            "notes": res["notes"],
        }

    blocks = [block(r) for r in records]

    def r_of(b: dict) -> str:
        for plan, rec in zip(PLANS, blocks):
            if rec["key"] == b["key"]:
                return plan["tag"]
        return "?"

    mesh = [b for b in blocks if r_of(b) == "mesh"]
    layers = [b for b in blocks if r_of(b) == "layers"]
    ideal = [b for b in blocks if r_of(b) == "ideal"]

    mesh_sorted = sorted(mesh, key=lambda b: b["mesh"][0] * b["mesh"][1])
    mesh_conv = _successive([b["plateau_factor"] for b in mesh_sorted])
    yield_conv = _successive([b["first_yield_factor"] for b in mesh_sorted])

    payload = {
        "missing_shards": missing,
        "case": dict(a=A, b=B, t=T, patch=list(PATCH), young=E_MOD, poisson=NU,
                     grade=GRADE, temperature_c=TEMPERATURE_C,
                     hardening=HARDENING, deflection_ratios=list(RATIOS),
                     geometric=False, inplane_restrained=True),
        "references": refs,
        "runs": blocks,
        "mesh_convergence": mesh_conv,
        "first_yield_convergence": yield_conv,
        "criteria": {},
    }

    finest = mesh_sorted[-1] if mesh_sorted else None
    criteria = payload["criteria"]
    if finest is None:
        payload["criteria"] = {"verdict": "not_run",
                               "reason": "no mesh-ladder shard available"}
        out = os.path.join(HERE, "..", "outputs", "gap_pp3_displacement_ladder.json")
        with open(out, "w") as handle:
            json.dump(payload, handle, indent=1)
        return payload
    criteria["transition_vs_analytic"] = {
        "requirement": "< 5 %",
        "raw_error_finest_mesh": finest["first_yield_error"],
        "successive_change": yield_conv.get("finest_change"),
        "note": ("raw FE first-yield pressure vs the Navier/Mises analytic value; "
                 "the element's own elastic bias is *not* removed here"),
    }
    criteria["limit_vs_upper_bound"] = {
        "requirement": "< 10 % vs Ilyushin / upper bound",
        "plateau_factor": finest["plateau_factor"],
        "vs_yield_line_upper": finest["plateau_vs_upper"],
        "vs_ilyushin_sine": finest["plateau_vs_ilyushin"],
        "turning_point": finest["turning_point"],
        "note": ("the reported value is the pressure at the largest *controlled* "
                 "deflection; without a turning point it is a lower bound on the "
                 "plateau, and it is never tuned to the references"),
    }
    criteria["bracket"] = {
        "requirement": "strip lower <= plateau <= yield-line upper",
        "strip_lower": lb,
        "yield_line_upper": ub,
        "inside": all(b["inside_bracket"] for b in mesh_sorted),
    }
    criteria["mesh_stability"] = {
        "requirement": "finest two meshes agree within 10 %",
        "finest_change": mesh_conv.get("finest_change"),
        "met": (mesh_conv.get("finest_change") is not None
                and mesh_conv["finest_change"] < 0.10),
    }
    criteria["layer_sensitivity"] = {
        "values": [{"layers": b["layers"], "plateau_factor": b["plateau_factor"],
                    "capacity_ratio": b["capacity_ratio"]} for b in layers],
        "spread": (None if len(layers) < 2 else
                   (max(b["plateau_factor"] for b in layers)
                    - min(b["plateau_factor"] for b in layers))
                   / max(b["plateau_factor"] for b in layers)),
    }
    criteria["permanent_set"] = {
        "values": [{"mesh": b["mesh"], "layers": b["layers"],
                    "permanent_deflection": b["permanent_deflection"],
                    "valid": b["permanent_set_valid"]} for b in mesh_sorted],
        "all_valid": all(b["permanent_set_valid"] for b in mesh_sorted),
    }
    criteria["near_ideal_hardening"] = [
        {"key": b["key"], "plateau_factor": b["plateau_factor"],
         "turning_point": b["turning_point"]} for b in ideal]

    out = os.path.join(HERE, "..", "outputs", "gap_pp3_displacement_ladder.json")
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1)
    return payload


def figures(payload: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder = os.path.join(HERE, "..", "docs", "figures")
    os.makedirs(folder, exist_ok=True)
    import glob

    # 1. load-deflection paths of the mesh ladder
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in sorted(glob.glob(os.path.join(OUT_DIR, "mesh_*.json"))):
        with open(name) as handle:
            rec = json.load(handle)
        path = rec["result"]["path"]
        ax.plot([p[0] * 1e3 for p in path], [p[1] / 1e6 for p in path], "o-",
                ms=3, label=f"{rec['plan']['nx']}x{rec['plan']['ny']}")
    refs = payload["references"]
    for value, label, style in (
            (refs["first_yield_pressure"], "analytic first yield $p_y$", ":"),
            (refs["ilyushin_sine_upper"], "Ilyushin sine bound", "--"),
            (refs["yield_line_upper"], "yield-line upper bound", "-."),
            (refs["strip_lower"], "strip lower bound", ":")):
        ax.axhline(value / 1e6, color="0.4", ls=style, lw=1.0)
        ax.annotate(label, (ax.get_xlim()[1], value / 1e6), fontsize=6,
                    ha="right", va="bottom", color="0.35")
    ax.set_xlabel("centre deflection $w_c$ [mm]")
    ax.set_ylabel("patch pressure $p$ [MPa]")
    ax.set_title("displacement-controlled paths vs rigid-plastic references")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "gap_pp3_displacement_paths.png"), dpi=150)
    plt.close(fig)

    # 2. plastic zone figure (max |M|/m_p along the path) + convergence
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for name in sorted(glob.glob(os.path.join(OUT_DIR, "mesh_*.json"))):
        with open(name) as handle:
            rec = json.load(handle)
        path = rec["result"]["path"]
        left.plot([p[0] * 1e3 for p in path], [p[3] for p in path], "o-", ms=3,
                  label=f"{rec['plan']['nx']}x{rec['plan']['ny']}")
    left.axhline(1.0, color="0.4", ls="--", lw=1.0)
    left.set_xlabel("centre deflection $w_c$ [mm]")
    left.set_ylabel(r"peak $|M|/m_p$")
    left.set_title("section plastification")
    left.grid(alpha=0.3)
    left.legend(fontsize=7)
    mesh = sorted([b for b in payload["runs"] if b["layers"] == 6
                   and b["hardening"] == HARDENING],
                  key=lambda b: b["mesh"][0] * b["mesh"][1])
    right.plot([b["mesh"][1] for b in mesh], [b["plateau_factor"] / 1e6 for b in mesh],
               "s-", label="plateau $p$")
    right.plot([b["mesh"][1] for b in mesh],
               [b["first_yield_factor"] / 1e6 for b in mesh], "^--",
               label="FE first yield")
    right.axhline(payload["references"]["yield_line_upper"] / 1e6, color="0.4",
                  ls="-.", lw=1.0)
    right.axhline(payload["references"]["first_yield_pressure"] / 1e6, color="0.4",
                  ls=":", lw=1.0)
    right.set_xscale("log")
    right.set_xlabel("elements along the short side $n_y$")
    right.set_ylabel("pressure [MPa]")
    right.set_title("mesh convergence")
    right.grid(alpha=0.3)
    right.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, "gap_pp3_displacement_convergence.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-index", type=int, default=None)
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    if args.analyse:
        payload = analyse()
        if args.figures:
            figures(payload)
        print(json.dumps(payload["criteria"], indent=1))
        return
    if args.plan_index is None:
        parser.error("either --plan-index or --analyse is required")
    record = run_plan(args.plan_index)
    res = record["result"]
    print(json.dumps({k: res[k] for k in ("peak_factor", "plateau_factor",
                                          "limit_is_turning_point",
                                          "first_yield_factor",
                                          "permanent_deflection",
                                          "permanent_set_valid")}, indent=1))


if __name__ == "__main__":
    main()