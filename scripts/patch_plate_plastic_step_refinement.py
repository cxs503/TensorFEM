"""Load-step (ramp-density) refinement of the plate-panel plastic limit load.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET``.

Open defect this script closes
------------------------------
``docs/CASE-FEM-PLATE-PATCH-PLASTIC-PATH-ADMISSIBILITY.md`` §5 states it in one
sentence:

    The FE limit load moves by 39 % between nx = 8 (1.1997 MPa) and nx = 12
    (1.6695 MPa): it is not a converged number, and the two ramps also differ in
    density (5 vs 21 load points), so the difference mixes mesh and ramp effects.

The archived ladder was therefore *confounded*: the only two admissible meshes
were run with different load steps, so nobody can say whether the 39 % is a mesh
effect or a stepping artefact of a criterion-based limit extractor.  This script
separates the two by a two-way study:

* **ramp axis** -- at fixed mesh, the ramp is refined by halving the load step
  (``delta p / p_y = 0.10, 0.05, 0.025``) with the top of the ramp fixed;
* **mesh axis** -- the same step sizes are repeated on ``nx = 8, 12, 16``, so
  every mesh has a *matched-step* companion and the mesh effect can be read at
  constant ramp density.

What is measured per run (nothing transcribed by hand)
------------------------------------------------------
``limit_tight``      criterion-based limit load (:func:`limit_load_from_path`
                     with the tight criteria set), ``None`` when the path is
                     inadmissible -- fail-closed, never guessed;
``peak_factor``      last *committed* (admissible) pressure of the ramp;
``ramp_truncated``   whether the admissibility gate stopped the ramp, with the
                     reason string;
``first_yield``      pressure at which the centre surface Mises stress reaches
                     the flow stress (field test, mesh-dependent);
``permanent_set``    residual centre deflection after unloading, withheld when
                     the unload leg did not converge.

Honesty rules encoded in the analysis
-------------------------------------
* A step-refinement claim needs **two admissible** step sizes on the *same*
  mesh; a missing or inadmissible member is reported as ``not_run`` and the
  claim is ``withheld`` instead of being interpolated.
* The extrapolated (``delta p -> 0``) value is only quoted when the observed
  differences are monotone; otherwise the answer is ``no_convergence_claim``.
* The limit load is compared with the rigid-plastic bracket
  ``[strip_lower, johansen_twist]``; being *outside* it is reported as such
  rather than being explained away.

Usage::

    PYTHONPATH=src python scripts/patch_plate_plastic_step_refinement.py \
        --plan-index 0                 # one (nx, step) case per queue shard
    PYTHONPATH=src python scripts/patch_plate_plastic_step_refinement.py --analyse
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from patch_plate_plastic_mesh_ladder import (  # noqa: E402  (sibling module)
    ART,
    richardson,
    run_one,
)

#: ramp top, as a multiple of the analytic first-yield pressure
TOP_P_Y = 3.0
#: load steps, as multiples of p_y (halving sequence)
STEP_SEQUENCE = (0.10, 0.05, 0.025)
#: meshes of the (matched-step) ladder
MESH_SEQUENCE = (8, 12, 16)
#: unload points after the top of the ramp (permanent set leg)
UNLOAD_STEPS = 6
LAYERS = 4

#: the (mesh, step) cases actually submitted to the queue, one per shard -- the
#: full 3 x 3 grid, so every mesh has a matched-step partner and every mesh also
#: gets its own step-refinement series
PLAN = tuple((nx, dp) for dp in STEP_SEQUENCE for nx in MESH_SEQUENCE)


def case_tag(nx: int, dp: float) -> str:
    return f"nx{nx}_dp{dp:g}"


def run_case(nx: int, dp: float, *, top: float = TOP_P_Y) -> dict:
    """One ramp: fixed mesh, fixed step, dense top, unload leg at the end."""
    start = time.time()
    record = run_one(nx, layers=LAYERS, geometry="free", unload_steps=UNLOAD_STEPS,
                     top=top, step=dp)
    tight = record["limit_estimates"]["tight"]
    loose = record["limit_estimates"]["loose"]
    out = {
        "tag": case_tag(nx, dp),
        "nx": nx,
        "step_p_y": dp,
        "top_p_y": top,
        "layers": LAYERS,
        "geometry": "free",
        "n_load_steps_requested": int(round(top / dp)),
        "n_committed_points": len(record["path"]),
        "references": record["references"],
        "section_capacity": record["section_capacity"],
        "limit_tight": tight.get("limit_load"),
        "limit_loose": loose.get("limit_load"),
        "limit_tight_reason": tight.get("reason"),
        "criteria_spread_tight": tight.get("spread"),
        "measurements_tight": tight.get("measurements"),
        "admissible": bool((tight.get("admissibility") or {}).get("admissible", False)),
        "admissibility_violations": (tight.get("admissibility") or {}).get("violations", []),
        "peak_factor": record["peak_factor"],
        "ramp_truncated": record["ramp_truncated"],
        "ramp_completed": record["ramp_completed"],
        "collapse_is_limit_point": record["limit_point_searched"],
        "first_yield": record["first_yield"],
        "permanent_deflection": record["permanent_deflection"],
        "permanent_set_valid": record["permanent_set_valid"],
        "unload_converged_steps": record["unload_converged_steps"],
        "unload_failure": record["unload_failure"],
        "elapsed_s": time.time() - start,
        # a compact copy of the path keeps the artifact self-contained but small
        "path": record["path"][::2] + record["path"][-1:],
    }
    return out


def _interp_reference(record: dict) -> dict:
    """p_y (analytic first yield) recovered from the case record's references."""
    ref = record.get("references") or {}
    for key in ("p_y", "first_yield_pressure", "elastic_first_yield"):
        if key in ref:
            value = ref[key]
            return {"p_y": value if isinstance(value, (int, float)) else value.get("pressure")}
    return {}


def _rel(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0.0:
        return None
    return a / b - 1.0


def analyse_series(series: list[dict]) -> dict:
    """Step-size convergence of one mesh from its (step -> limit) series.

    ``series`` must be sorted by decreasing step.  Returns a verdict dictionary;
    every claim carries the evidence it was derived from, and a claim that the
    data cannot support is ``withheld`` with the reason.
    """
    usable = [r for r in series if r.get("limit_tight") is not None]
    out: dict = {
        "nx": series[0]["nx"] if series else None,
        "steps": [r["step_p_y"] for r in series],
        "limits": [r.get("limit_tight") for r in series],
        "n_usable": len(usable),
        "series": series,
    }
    if len(usable) < 2:
        out["verdict"] = "withheld"
        out["reason"] = ("fewer than two admissible step sizes: a step-refinement "
                         "claim cannot be made")
        return out
    a, b = usable[-2], usable[-1]          # two finest admissible steps
    change = _rel(b["limit_tight"], a["limit_tight"])
    out["finest_two"] = {"steps": [a["step_p_y"], b["step_p_y"]],
                         "limits": [a["limit_tight"], b["limit_tight"]],
                         "relative_change": change}
    # monotone in step? a converging convention shrinks the change as dp -> 0
    monotone = True
    changes = []
    for prev, nxt in zip(usable[:-1], usable[1:]):
        c = _rel(nxt["limit_tight"], prev["limit_tight"])
        changes.append(c)
    for i in range(1, len(changes)):
        if changes[i] is None or changes[i - 1] is None:
            continue
        if abs(changes[i]) > abs(changes[i - 1]) * (1.0 + 1e-9):
            monotone = False
    out["successive_changes"] = changes
    out["change_decreasing"] = monotone
    if change is not None and abs(change) < 0.02:
        out["verdict"] = "step_independent"
    elif monotone:
        out["verdict"] = "converging"
    else:
        out["verdict"] = "not_converged"
    # observed order in the step, only when three admissible steps exist *and*
    # the differences shrink monotonically with the step (an extrapolation from
    # a non-monotone sequence is not an extrapolation -- fail closed)
    if len(usable) >= 3 and monotone:
        hs = [r["step_p_y"] for r in usable[-3:]]
        ps = [r["limit_tight"] for r in usable[-3:]]
        q, p_inf = richardson(hs, ps, bracket=(0.05, 12.0))
        out["step_richardson"] = {"order": q, "limit_dp_to_zero": p_inf}
    return out


def matched_mesh_effect(records: list[dict]) -> list[dict]:
    """Mesh effect at *matched* load step: the confound this script removes."""
    by_step: dict[float, dict[int, dict]] = {}
    for r in records:
        by_step.setdefault(r["step_p_y"], {})[r["nx"]] = r
    out = []
    for dp, meshes in sorted(by_step.items(), reverse=True):
        nx_sorted = sorted(meshes)
        entry: dict = {"step_p_y": dp, "meshes": nx_sorted,
                       "limits": [meshes[n].get("limit_tight") for n in nx_sorted]}
        pairs = []
        for lo, hi in zip(nx_sorted[:-1], nx_sorted[1:]):
            pairs.append({
                "pair": [lo, hi],
                "relative_change": _rel(meshes[hi].get("limit_tight"),
                                        meshes[lo].get("limit_tight")),
                "same_ramp_density": True,
            })
        entry["pairs"] = pairs
        out.append(entry)
    return out


def bracket_check(limit: float | None, record: dict) -> dict:
    """Is the measured limit inside the rigid-plastic static/kinematic bracket?"""
    if limit is None:
        return {"verdict": "withheld", "reason": "no admissible limit load"}
    bounds = (record.get("references") or {}).get("bounds") or {}
    lower = bounds.get("strip_lower")
    upper_kinematic = bounds.get("johansen_twist") or bounds.get("yield_line")
    out = {"limit": limit, "lower": lower, "upper": upper_kinematic,
           "p_yield_analytic": (record.get("references") or {}).get("p_yield_analytic")}
    if lower is None or upper_kinematic is None:
        out["verdict"] = "withheld"
        out["reason"] = "bracket references missing"
        return out
    out["errors"] = {k: limit / v - 1.0 for k, v in bounds.items()}
    out["inside_bracket"] = bool(lower <= limit <= upper_kinematic)
    out["verdict"] = "inside" if out["inside_bracket"] else "outside"
    return out


def analyse() -> int:
    files = sorted(ART.glob("patch_plate_plastic_stepref_*.json"))
    if not files:
        print("no step-refinement artifacts yet")
        return 1
    records = [json.loads(f.read_text()) for f in files]
    summary: dict = {"files": [f.name for f in files], "records": []}
    for r in sorted(records, key=lambda q: (q["nx"], -q["step_p_y"])):
        summary["records"].append({
            "tag": r["tag"], "nx": r["nx"], "step_p_y": r["step_p_y"],
            "n_committed_points": r["n_committed_points"],
            "limit_tight": r["limit_tight"], "limit_loose": r["limit_loose"],
            "criteria_spread": r["criteria_spread_tight"],
            "admissible": r["admissible"],
            "peak_factor": r["peak_factor"],
            "ramp_truncated": r["ramp_truncated"],
            "first_yield": r["first_yield"],
            "permanent_deflection": r["permanent_deflection"],
            "permanent_set_valid": r["permanent_set_valid"],
            "elapsed_s": round(r["elapsed_s"], 1),
            "bracket": bracket_check(r["limit_tight"], r),
        })
    series = []
    for nx in sorted({r["nx"] for r in records}):
        rows = sorted([r for r in records if r["nx"] == nx],
                      key=lambda q: -q["step_p_y"])
        series.append(analyse_series(rows))
    summary["step_convergence"] = series
    summary["mesh_effect_matched_step"] = matched_mesh_effect(records)
    (ART / "patch_plate_plastic_stepref_summary.json").write_text(
        json.dumps(summary, indent=2))

    for rec in summary["records"]:
        limit = rec["limit_tight"]
        limit_text = f"{limit / 1e6:.4f} MPa" if limit else "withheld"
        print(f"[{rec['tag']}] limit {limit_text} "
              f"spread {rec['criteria_spread'] if rec['criteria_spread'] is not None else float('nan'):.3f} "
              f"peak {rec['peak_factor'] / 1e6:.4f} MPa "
              f"{'truncated' if rec['ramp_truncated'] else 'completed'} "
              f"bracket {rec['bracket']['verdict']} "
              f"({rec['elapsed_s']:.0f} s)")
    for s in series:
        print(f"[step-convergence nx={s['nx']}] {s['verdict']} "
              f"changes {['%.4f' % c if c is not None else 'n/a' for c in s.get('successive_changes', [])]} "
              f"{s.get('step_richardson', '')}")
    for m in summary["mesh_effect_matched_step"]:
        print(f"[mesh matched dp/p_y={m['step_p_y']:g}] "
              f"{list(zip(m['meshes'], [None if v is None else round(v / 1e6, 4) for v in m['limits']]))} "
              f"pairs {[('%.4f' % p['relative_change']) if p['relative_change'] is not None else 'n/a' for p in m['pairs']]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--step", type=float, default=None,
                        help="load step as a multiple of p_y")
    parser.add_argument("--plan-index", type=int, default=None,
                        help=f"run PLAN[i]: {PLAN}")
    parser.add_argument("--analyse", action="store_true")
    args = parser.parse_args(argv)
    if args.analyse:
        return analyse()
    if args.plan_index is not None:
        args.nx, args.step = PLAN[args.plan_index]
    if args.nx is None or args.step is None:
        parser.error("--nx and --step are required (or --plan-index / --analyse)")
    record = run_case(args.nx, args.step)
    out = ART / f"patch_plate_plastic_stepref_{record['tag']}.json"
    out.write_text(json.dumps(record, indent=2))
    limit = record["limit_tight"]
    print(f"[{record['tag']}] limit "
          f"{'withheld' if limit is None else '%.4f MPa' % (limit / 1e6)} "
          f"peak {record['peak_factor'] / 1e6:.4f} MPa "
          f"points {record['n_committed_points']}/{record['n_load_steps_requested']} "
          f"{'truncated' if record['ramp_truncated'] else 'completed'} "
          f"({record['elapsed_s']:.0f} s) -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())