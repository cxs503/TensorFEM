"""Mesh-convergence ladder for the plate-panel elastoplastic patch capacity.

Anchor ``GAP-PP-3-ICE-STRUCT-PLASTIC-SET``.  The previous certificate
(``docs/CASE-FEM-PLATE-PATCH-PLASTIC-LIMIT-LOAD.md``) left two acceptance items
open: the three-level mesh convergence was never run, and the FE limit load at a
single mesh was reported as "at least +14.6 % above the admissible bound" without
a converged value to compare.  This script closes both by marching *one* dense
load ramp per mesh (``nx = 8 / 12 / 16``), extracting the limit load with the
three independent definitions of
:func:`tensorfem.plate_patch_plastic_limit.limit_load_from_path`, and unloading
at the end of each ramp so that the permanent deflection of the same run is
recorded on the same mesh.

Two support idealisations are offered, because they answer different questions
and must not be mixed:

``free``      (default) simply supported, in-plane displacements free.  A linear
              flat-shell has no membrane coupling, so this is the *bending only*
              model that the yield-line / Ilyushin limit analysis refers to --
              the only support for which comparing the FE limit load with those
              bounds is admissible.
``membrane``  ``geometric=True`` + in-plane restrained: a panel welded into a
              grillage.  Membrane tension develops after the bending moments
              saturate, so the path turns over and a *real* peak exists; that
              peak is a large-deflection collapse load and is **not** comparable
              with a rigid-plastic bending bound (it is legitimately higher).

Everything is written to ``artifacts/patch_plate_plastic_ladder_<tag>.json``;
no number is transcribed by hand.

Usage::

    PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --nx 8
    PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --nx 16 --geometry membrane
    PYTHONPATH=src python scripts/patch_plate_plastic_mesh_ladder.py --analyse
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

from tensorfem.plate_patch_plastic_limit import (
    criterion_discrimination,
    analytic_first_yield_pressure,
    ilyushin_sine_upper_bound,
    johansen_twist_upper_bound,
    limit_load_from_path,
    low_temperature_flow_stress,
    plastic_bound_bracket,
    plastic_moment,
    run_patch_plastic_path,
    section_moment_capacity,
    strip_lower_bound,
    yield_line_patch_upper_bound,
)

A = B = 1.0
T = 0.02
E, NU = 210e9, 0.3
GRADE, TEMP_C = "DH36", -40.0
PATCH = (0.5 * A, 0.5 * B)
HARDENING = 1e6            # near-ideal plasticity: a *limit* load exists
UNLOAD_STEPS = 3
ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

#: criteria sets evaluated on every path (fraction of the elastic slope)
CRITERIA_SETS = {"loose": (0.10, 0.20, 0.30), "tight": (0.02, 0.05, 0.10)}


def ramp_factors(p_y: float, top: float = 3.0, step: float = 0.1) -> list[float]:
    """Dense load factors in ``p_y`` units: 0.2, 0.3, ... ``top``.

    Dense enough that the incremental-stiffness criteria resolve inside one
    interval, and that the two-line rule uses a *local* plastic tangent instead
    of a lucky last interval.
    """
    count = int(round((top - 0.2) / step)) + 1
    return [round((0.2 + step * k) * p_y, 9) for k in range(count)]


def references(flow_stress: float) -> dict:
    bracket = plastic_bound_bracket(A, B, T, flow_stress, PATCH, terms=1)
    return {
        "yield_stress": flow_stress,
        "grade": GRADE, "temperature_c": TEMP_C, "hardening": HARDENING,
        "m_p": plastic_moment(T, flow_stress),
        "patch": list(PATCH), "a": A, "b": B, "t": T,
        "p_yield_analytic": analytic_first_yield_pressure(
            A, B, T, E, NU, flow_stress, PATCH)["pressure"],
        "bounds": {
            "ilyushin_sine": ilyushin_sine_upper_bound(
                A, B, T, flow_stress, PATCH, terms=1)["pressure"],
            "yield_line": yield_line_patch_upper_bound(
                A, B, T, flow_stress, PATCH)["pressure"],
            "strip_lower": strip_lower_bound(
                A, B, T, flow_stress, PATCH)["pressure"],
            # admissible upper bound (twist retained inside the dissipation):
            # the twist-neglecting ``ilyushin_sine`` value is *below* the exact
            # full-plate collapse and is therefore reported, but not used as a
            # pass/fail yardstick.
            "johansen_twist": johansen_twist_upper_bound(
                A, B, T, flow_stress, PATCH, terms=1)["pressure"],
        },
        "bracket": bracket,
    }


def run_one(nx: int, *, layers: int, geometry: str, unload_steps: int,
            top: float, step: float) -> dict:
    flow = low_temperature_flow_stress(GRADE, TEMP_C)
    ref = references(flow.yield_stress)
    p_y = ref["p_yield_analytic"]
    membrane = geometry == "membrane"
    start = time.time()
    factors = ramp_factors(p_y, top, step)
    path = run_patch_plastic_path(
        A, B, T, young=E, poisson=NU, yield_stress=flow.yield_stress,
        hardening=HARDENING, patch=PATCH, load_factors=factors,
        nx=nx, ny=nx, layers=layers, unload_steps=unload_steps,
        exact_thickness=True, shear_correction=5.0 / 6.0,
        max_deflection_factor=6.0, bisection_tolerance=1e-2,
        geometric=membrane, inplane_restrained=membrane)
    elapsed = time.time() - start

    estimates = {name: limit_load_from_path(path, criteria=c)
                 for name, c in CRITERIA_SETS.items()}
    primary = estimates["tight"]
    record = {
        "nx": nx, "layers": layers, "geometry": geometry,
        # prescribed ramp: the comparability key of a mesh ladder (see
        # ``ramp_signature``/``comparable_ladders``)
        "ramp": {"top_over_py": top, "step_over_py": step,
                 "points": len(factors),
                 "factors_over_py": [round(f / p_y, 6) for f in factors]},
        "section_capacity": section_moment_capacity(T, flow.yield_stress, layers),
        "references": ref,
        "limit_estimates": estimates,
        "first_yield": path.first_yield_factor,
        "ramp_completed": path.ramp_completed,
        "ramp_truncated": path.ramp_truncated,
        "measured_limit_load": path.measured_limit_load,
        "limit_point_searched": path.limit_point_searched,
        "peak_factor": path.loading[-1].load_factor,
        "permanent_deflection": path.permanent_deflection,
        "permanent_set_valid": path.permanent_set_valid,
        "unload_converged_steps": path.unload_converged_steps,
        "unload_failure": path.unload_failure,
        "elapsed_s": elapsed,
        "path": [{"p": s.load_factor, "p_over_py": s.load_factor / p_y,
                  "w": s.centre_deflection, "alpha": s.max_alpha,
                  "moment_ratio": s.max_moment_ratio,
                  "sigma_surface_over_fy": s.max_surface_mises / flow.yield_stress}
                 for s in path.loading],
    }
    if path.unloading:
        record["unloading"] = [{"p": s.load_factor, "w": s.centre_deflection,
                                "moment_ratio": s.max_moment_ratio}
                               for s in path.unloading]
    limit = primary["limit_load"] if primary else None
    if limit:
        record["errors"] = {
            key: limit / value - 1.0 for key, value in ref["bounds"].items()}
    # first-yield acceptance is a *field* test, so record it here rather than
    # re-deriving from the coarse path later.
    if path.first_yield_factor:
        record["first_yield_error_vs_analytic"] = (
            path.first_yield_factor / p_y - 1.0)
    return record


def tag(record_or_args) -> str:
    if isinstance(record_or_args, dict):
        nx, geometry = record_or_args["nx"], record_or_args["geometry"]
    else:
        nx, geometry = record_or_args.nx, record_or_args.geometry
    return f"nx{nx}_{geometry}"


def richardson(hs: list[float], ps: list[float],
               bracket: tuple[float, float] = (0.1, 8.0)) -> tuple[float, float]:
    """Observed order ``q`` and the mesh-independent limit for ``p = p_inf + C h^q``.

    The ladder meshes are ``nx = 8 / 12 / 16`` whose element sizes are *not* in a
    uniform ratio, so the closed-form Richardson expression for ``r = 2`` cannot
    be used.  The order is therefore obtained by bisecting the monotone ratio

    ``R(q) = (h3^q - h2^q) / (h2^q - h1^q)``, which must equal ``(p3-p2)/(p2-p1)``.

    Returns ``(q, p_inf)``; ``(nan, None)`` when the data do not increase
    monotonically (a ladder that does not converge cannot be extrapolated).
    """
    h1, h2, h3 = hs[0], hs[1], hs[2]
    p1, p2, p3 = ps[0], ps[1], ps[2]
    d21, d32 = p2 - p1, p3 - p2
    sign = 1.0 if d21 > 0.0 else -1.0
    d21, d32 = sign * d21, sign * d32
    if not (d21 > 0.0 and d32 > 0.0):
        # non-monotone ladder (neither rising nor falling): not extrapolable
        return float("nan"), None
    target = d32 / d21
    lo, hi = bracket

    def ratio(q: float) -> float:
        return (h3 ** q - h2 ** q) / (h2 ** q - h1 ** q)

    if not (ratio(lo) > target > ratio(hi)):
        # the observed ratio is outside what any order in the bracket explains
        return float("nan"), None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if ratio(mid) > target:
            lo = mid
        else:
            hi = mid
    q = 0.5 * (lo + hi)
    slope = d32 / (h3 ** q - h2 ** q)
    return q, p3 - sign * slope * h3 ** q


def ramp_signature(record: dict) -> dict:
    """Everything that must agree before two ladder records may be differenced.

    Mesh convergence is a statement about *one* discretisation family: same
    prescribed load ramp, same layer count, same hardening, same patch, same
    material.  Differencing records that disagree in any of these measures the
    ramp (or the layering), not the mesh, so the signature is the gate of every
    Richardson estimate in :func:`analyse`.
    """
    ramp = record.get("ramp") or {}
    refs = record.get("references") or {}
    return {
        "factors_over_py": ramp.get("factors_over_py"),
        "layers": record.get("layers"),
        "hardening": refs.get("hardening"),
        "yield_stress": refs.get("yield_stress"),
        "patch": list(refs.get("patch") or []),
    }


def comparable_ladders(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split ladder entries into the one comparable family and the rest.

    Per geometry the records are grouped by :func:`ramp_signature`.  A single
    group is admitted as is.  When several groups exist, the largest one is
    admitted and every other record is *excluded with a reason* -- it is still
    reported (never silently dropped), but it does not contribute to the
    convergence or extrapolation claim.
    """
    admitted: list[dict] = []
    excluded: list[dict] = []
    for geometry in sorted({e["geometry"] for e in entries}):
        group = [e for e in entries if e["geometry"] == geometry]
        families: dict[str, list[dict]] = {}
        for entry in group:
            families.setdefault(json.dumps(entry["ramp_signature"],
                                           sort_keys=True), []).append(entry)
        if len(families) == 1:
            admitted.extend(group)
            continue
        # largest family wins; ties broken by the finest mesh in the family
        ranked = sorted(families.values(),
                        key=lambda fam: (len(fam), max(e["nx"] for e in fam)),
                        reverse=True)
        admitted.extend(ranked[0])
        reference = ranked[0][0]["ramp_signature"]
        for family in ranked[1:]:
            for entry in family:
                out = dict(entry)
                out["ladder_excluded"] = (
                    "ramp/discretisation signature differs from the admitted "
                    f"ladder ({_ramp_label(entry['ramp_signature'])} vs "
                    f"{_ramp_label(reference)}): differencing would measure the "
                    "ramp, not the mesh")
                excluded.append(out)
    return admitted, excluded


def _ramp_label(signature: dict) -> str:
    """Short description of a ramp signature for the exclusion reason."""
    factors = signature.get("factors_over_py") or []
    if not factors:
        return "ramp not recorded (legacy artifact)"
    steps = sorted({round(b - a, 6) for a, b in zip(factors, factors[1:])})
    step = steps[0] if len(steps) == 1 else "mixed"
    return (f"{len(factors)} points, {factors[0]:.2f}..{factors[-1]:.2f} p_y, "
            f"step {step}, {signature.get('layers')} layers")


def analyse() -> int:
    """Aggregate the ladder: Richardson extrapolation and acceptance verdicts."""
    files = sorted(ART.glob("patch_plate_plastic_ladder_nx*_*.json"))
    if not files:
        print("no ladder artifacts yet")
        return 1
    records = [json.loads(f.read_text()) for f in files]
    summary: dict = {"records": [], "records_excluded": [],
                     "files": [f.name for f in files]}
    by_tag: dict[str, dict] = {}
    entries: list[dict] = []
    for rec in sorted(records, key=lambda r: (r["geometry"], r["nx"])):
        tight = rec["limit_estimates"]["tight"]
        loose = rec["limit_estimates"]["loose"]
        entry = {
            "tag": tag(rec), "nx": rec["nx"], "geometry": rec["geometry"],
            "limit_tight": tight["limit_load"],
            "limit_loose": loose["limit_load"],
            "criteria_spread_tight": tight["spread"],
            "measurements_tight": tight["measurements"],
            "measured_limit_load": rec["measured_limit_load"],
            "first_yield": rec["first_yield"],
            "first_yield_error": rec.get("first_yield_error_vs_analytic"),
            "permanent_deflection": rec["permanent_deflection"],
            "permanent_set_valid": rec["permanent_set_valid"],
            "peak_factor": rec["peak_factor"],
            "elapsed_s": rec["elapsed_s"],
            "moment_plateau": rec["path"][-1]["moment_ratio"],
            "section_capacity_ratio": rec["section_capacity"]["ratio"],
        }
        bounds = rec["references"]["bounds"]
        if entry["limit_tight"]:
            entry["errors"] = {k: entry["limit_tight"] / v - 1.0
                               for k, v in bounds.items()}
        entry["ramp_signature"] = ramp_signature(rec)
        by_tag[tag(rec)] = rec
        entries.append(entry)

    # Comparability gate: a mesh ladder is a convergence statement only when all
    # of its records share one prescribed ramp (and one layering / patch).
    admitted, excluded = comparable_ladders(entries)
    summary["records"] = admitted
    summary["records_excluded"] = excluded
    for out in excluded:
        print(f"[{out['tag']}] excluded from the ladder: {out['ladder_excluded']}",
              flush=True)

    # Richardson: p_c(h) = p_inf + C h^q for the bending-only ladder.  Two runs
    # are enough for one estimate; three give the observed order q.
    for geometry in sorted({r["geometry"] for r in records}):
        ladder = [e for e in summary["records"]
                  if e["geometry"] == geometry and e["limit_tight"]]
        if len(ladder) < 2:
            continue
        hs = [1.0 / e["nx"] for e in ladder]
        ps = [e["limit_tight"] for e in ladder]
        if len(ladder) >= 3:
            order, p_inf = richardson(hs, ps)
            summary[f"extrapolation_{geometry}"] = {
                "order_estimate": order, "limit_infinity": p_inf,
                "last_three": ps, "h": hs}
        else:
            summary[f"extrapolation_{geometry}"] = {
                "order_estimate": None, "limit_infinity": None,
                "note": "three meshes required for a Richardson estimate",
                "two_mesh_values": ps, "h": hs}
        if geometry == "free":
            bounds = by_tag[ladder[0]["tag"]]["references"]["bounds"]
            for entry in ladder:
                entry["acceptance"] = {
                    "within_10pct_of_ilyushin": bool(
                        abs(entry["errors"]["ilyushin_sine"]) < 0.10),
                    # the sine bound is inadmissible for a smooth mechanism (it
                    # is below the exact full-plate collapse load), so the
                    # acceptance is also judged against the admissible bound
                    "within_10pct_of_admissible_upper": bool(
                        entry["errors"].get("johansen_twist") is not None
                        and abs(entry["errors"]["johansen_twist"]) < 0.10),
                    "below_yield_line_upper_bound": bool(
                        entry["errors"]["yield_line"] < 0.0),
                    "above_strip_lower_bound": bool(
                        entry["errors"]["strip_lower"] > 0.0),
                    "first_yield_within_5pct": bool(
                        entry["first_yield_error"] is not None
                        and abs(entry["first_yield_error"]) < 0.05),
                }
    bounds_free = next((by_tag[e["tag"]]["references"]["bounds"]
                        for e in summary["records"]
                        if e["geometry"] == "free"), None)
    extrapolation = summary.get("extrapolation_free") or {}
    if bounds_free and extrapolation.get("limit_infinity"):
        summary["extrapolated_limit_error_vs_ilyushin"] = (
            extrapolation["limit_infinity"] / bounds_free["ilyushin_sine"] - 1.0)
        summary["extrapolated_limit_error_vs_yield_line"] = (
            extrapolation["limit_infinity"] / bounds_free["yield_line"] - 1.0)
    if bounds_free:
        # The ledger acceptance is "limit load within 10 % of the Ilyushin/upper
        # bound analysis".  No solver can satisfy that while the admissible
        # bracket is wide, so the criterion itself is judged here.
        # ``johansen_twist`` was added to the reference set after the first
        # artifacts were produced, so the analysis must not assume it: the
        # admissible upper bound is the tighter of the two when both exist and
        # the yield-line bound alone otherwise (a legacy artifact must not crash
        # the whole analysis step).
        admissible_upper = min(bounds_free["yield_line"],
                               bounds_free.get("johansen_twist",
                                              bounds_free["yield_line"]))
        gate = criterion_discrimination(
            admissible_upper,
            bounds_free["strip_lower"], 0.10, label="limit load vs upper bound")
        summary["acceptance_discrimination"] = gate
        # the criterion gate belongs to the bending-only (in-plane free) ladder:
        # a membrane/grillage record is never judged against the bending bounds
        for entry in [e for e in summary["records"] if e["geometry"] == "free"]:
            entry.setdefault("acceptance", {})["criterion_discriminative"] = \
                gate["discriminative"]
        print(f"criterion gate: {gate['verdict']} "
              f"(bracket width {gate['bracket_width_ratio']:.2f}, needs "
              f"<= {gate['required_width_ratio']:.3f}; best possible error "
              f"{100 * gate['best_possible_relative_error']:.1f} %)")
    (ART / "patch_plate_plastic_ladder_summary.json").write_text(
        json.dumps(summary, indent=2))
    for entry in summary["records"]:
        limit = entry["limit_tight"]
        if limit is None:
            # no limit-load definition fired: report it, never format ``None``
            print(f"[{entry['tag']}] limit n/a (no criterion fired) "
                  f"| first yield {entry['first_yield'] / 1e6:.4f} MPa")
            continue
        print(f"[{entry['tag']}] limit {limit / 1e6:.4f} MPa "
              f"({100 * (entry.get('errors') or {}).get('ilyushin_sine', float('nan')):+.2f} % vs Ilyushin, "
              f"{100 * (entry.get('errors') or {}).get('yield_line', float('nan')):+.2f} % vs yield line) "
              f"spread {entry['criteria_spread_tight'] if entry['criteria_spread_tight'] is not None else float('nan'):.3f} "
              f"first yield {entry['first_yield'] / 1e6:.4f} MPa "
              f"perm set {(entry['permanent_deflection'] or 0.0) * 1e3:.2f} mm "
              f"lowest interp {limit / 1e6:.4f}")
    return 0


#: the campaign actually submitted to the queue: ``(nx, geometry)``
def describe(record: dict) -> str:
    """One machine-generated report line for a ladder record.

    ``limit_load_from_path`` legitimately returns ``None`` when no limit-load
    definition fires on a ramp (a ramp that never softened, or too short a
    ramp): that is a *result*, not an error, so the report says ``n/a`` instead
    of formatting ``None`` as a number.
    """
    tight = record["limit_estimates"]["tight"]
    limit = tight.get("limit_load")
    errors = record.get("errors") or {}
    spread = tight.get("spread")
    if limit is None:
        limit_text = "limit n/a (no criterion fired)"
        error_text = "vs bounds n/a"
    else:
        limit_text = f"limit {limit / 1e6:.4f} MPa"
        error_text = (f"({100 * errors.get('ilyushin_sine', float('nan')):+.2f} % "
                      f"vs Ilyushin, "
                      f"{100 * errors.get('yield_line', float('nan')):+.2f} % "
                      "vs yield line)")
    first = record.get("first_yield")
    first_text = f"{first / 1e6:.4f} MPa" if first else "n/a"
    permanent = record.get("permanent_deflection")
    if permanent is not None and record.get("permanent_set_valid"):
        permanent_text = f"{permanent * 1e3:.2f} mm"
    else:
        permanent_text = "withheld"
    spread_text = f"{spread:.3f}" if spread is not None else "n/a"
    return (f"[{tag(record)}] {limit_text} {error_text} spread {spread_text} | "
            f"first yield {first_text} | perm. set {permanent_text} | "
            f"{record.get('elapsed_s', float('nan')):.0f} s")


PLAN = ((8, "free"), (12, "free"), (16, "free"), (8, "membrane"), (12, "membrane"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--geometry", choices=("free", "membrane"), default="free")
    parser.add_argument("--unload-steps", type=int, default=UNLOAD_STEPS)
    parser.add_argument("--top", type=float, default=3.0, help="ramp top in p_y")
    parser.add_argument("--step", type=float, default=0.1, help="ramp step in p_y")
    parser.add_argument("--analyse", action="store_true")
    parser.add_argument("--plan-index", type=int, default=None,
                        help=f"run PLAN[i] (one shard per queue job): {PLAN}")
    args = parser.parse_args(argv)
    if args.analyse:
        return analyse()
    if args.plan_index is not None:
        args.nx, args.geometry = PLAN[args.plan_index]
    if args.nx is None:
        parser.error("--nx is required (or --analyse)")
    record = run_one(args.nx, layers=args.layers, geometry=args.geometry,
                     unload_steps=args.unload_steps, top=args.top, step=args.step)
    out = ART / f"patch_plate_plastic_ladder_{tag(record)}.json"
    out.write_text(json.dumps(record, indent=2))
    print(describe(record) + f" -> {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())