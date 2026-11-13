#!/usr/bin/env python3
"""Independent recompute of the CASE-FEM-HERTZ-CONTACT anchor certificate.

Why this exists
---------------
``benchmarks/verified/fem_hertz_contact_loadcontrol/run.py`` writes
``result.json`` and the certificate ``gates_verified.json``.  ``--from-dump``
rebuilds the certificate from the dump, but it does so with the *same* code that
produced it, so a defect in the gate formulas is invisible to that route, and a
gate whose ``value`` cannot be re-derived from the dump is taken on trust.

This tool re-derives, with arithmetic that shares no code with the driver:

1. the Hertz closed form from the declared parameters,
2. every per-run ``error_pct`` from ``numeric`` / ``analytic``,
3. the value of every certificate gate whose quantity is a function of the dump
   (a gate with no independent definition is reported ``not_recomputed`` rather
   than assumed correct),
4. estimator hygiene: a primary contact-radius estimate that coincides with a
   node radius is a *degenerate* estimate (the zero crossing landed on the
   node), and the three radii estimators must agree within the declared band.

It also flags the two classes of evidence defect that prose cannot catch:
``CONTRADICTED`` (a gate whose declared verdict disagrees with value<op>threshold
or whose value cannot be re-derived) and ``DEGENERATE`` (a graded quantity that
is mesh-tied rather than measured).

Usage::

    python3 tools/hertz_anchor_recompute.py                 # human summary
    python3 tools/hertz_anchor_recompute.py --json out.json # machine result
    python3 tools/hertz_anchor_recompute.py --strict        # exit 1 on findings

Exit codes: 0 no findings, 1 findings or contradiction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ANCHOR = os.path.join(HERE, "..", "benchmarks", "verified",
                      "fem_hertz_contact_loadcontrol")

# tolerance for "recomputed value equals the stored value"
REL = 1e-9
# a primary radius estimate this close to the outer estimator is node-tied
DEGENERATE_REL = 1e-6
# the three estimators are allowed to disagree by this much (pct points)
ESTIMATOR_BAND_PP = 2.0
# gates whose value comes from an extrapolation on a different refinement
# ladder than the one in this dump: band agreement is the honest check
BAND_TOL = {"truncation_extrapolated_indentation": 0.10,
            "truncation_extrapolated_indentation_aR010": 0.10}


def hertz(load, radius, young, poisson):
    """Hertz 1882 / Johnson 1985 4.2, rigid indenter on an elastic half-space."""
    e_star = young / (1.0 - poisson ** 2)
    a = (3.0 * load * radius / (4.0 * e_star)) ** (1.0 / 3.0)
    return {"contact_radius": a,
            "indentation": a * a / radius,
            "peak_pressure": 3.0 * load / (2.0 * math.pi * a * a)}


def richardson(seq):
    """Three-point Richardson limit + observed order, from raw error values.

    ``seq`` is a list of ``(parameter, error)`` in the order the parameter is
    refined.  Returns ``(limit, order)`` or ``(nan, nan)`` when the sequence is
    not monotone (then no extrapolation is defined -- the caller must say so).
    """
    if len(seq) < 3:
        return float("nan"), float("nan")
    (h0, e0), (h1, e1), (h2, e2) = seq[-3], seq[-2], seq[-1]
    r = (h0 - h1) / (h1 - h2)
    num = (e1 - e2)
    den = (e0 - e1)
    if den == 0.0 or num / den <= 0.0:
        return float("nan"), float("nan")
    p = math.log(abs(num / den)) / math.log(r)
    if not (0.0 < p < 10.0):
        return float("nan"), float("nan")
    limit = e2 - (r ** p) * (e2 - e1) / (r ** p - 1.0)
    return limit, p


def run_key(ratio, h, depth=96.0):
    return f"ratio{ratio:.2f}_h{h:.4f}_d{depth:.0f}"


def recompute(anchor_dir=ANCHOR):
    with open(os.path.join(anchor_dir, "result.json")) as fh:
        res = json.load(fh)
    with open(os.path.join(anchor_dir, "gates_verified.json")) as fh:
        cert = json.load(fh)
    tr_path = os.path.join(anchor_dir, "gates_truncation.json")
    tr = json.load(open(tr_path)) if os.path.exists(tr_path) else {"gates": []}

    findings = []
    recomputed = {}

    # --- 1. the analytic oracle itself, from the declared parameters ---------
    R = res["params"]["R"]
    E = res["params"]["E"]
    nu = res["params"]["nu"]
    for key, row in res["runs"].items():
        ref = hertz(row["load"], R, E, nu)
        for field, value in ref.items():
            stored = row["analytic"][field]
            if abs(value - stored) > REL * max(abs(value), 1e-30):
                findings.append({
                    "code": "ORACLE_MISMATCH", "run": key, "field": field,
                    "stored": stored, "recomputed": value})
        # --- 2. every per-run error from numeric/analytic -------------------
        # error_pct keys are *numeric* estimator names; the oracle field they
        # grade is looked up (several estimators grade the same oracle value).
        oracle_field = {"contact_radius_linear": "contact_radius",
                        "contact_radius_outer": "contact_radius",
                        "contact_radius_fit": "contact_radius",
                        "contact_radius_fit_consistent": "contact_radius",
                        "indentation": "indentation",
                        "peak_pressure_lumped": "peak_pressure",
                        "peak_pressure_consistent": "peak_pressure"}
        for field, err in row["error_pct"].items():
            value = 100.0 * (row["numeric"][field]
                             / row["analytic"][oracle_field[field]] - 1.0)
            if abs(value - err) > 1e-6:
                findings.append({
                    "code": "ERROR_PCT_MISMATCH", "run": key, "field": field,
                    "stored": err, "recomputed": value})
        # --- 4. estimator hygiene -------------------------------------------
        a_lin = row["numeric"]["contact_radius_linear"]
        a_out = row["numeric"]["contact_radius_outer"]
        if abs(a_lin - a_out) <= DEGENERATE_REL * abs(a_out):
            findings.append({
                "code": "DEGENERATE", "run": key,
                "detail": ("primary contact radius == outermost member node "
                           f"({a_lin:.10f}); the zero crossing landed on a node, "
                           "so the graded value is mesh-tied"),
                "error_pct": row["error_pct"]["contact_radius_linear"]})
        # Scatter is only diagnostic on the *graded* (finest) level: the
        # form-assisted estimator assumes the Hertz shape and is legitimately
        # poor on a coarse mesh (+17 pp), so judging it there would be noise.
        # It is also excluded from the primary metric by the driver.
        band = abs(row["error_pct"]["contact_radius_fit"]
                   - row["error_pct"]["contact_radius_linear"])
        if row["h_over_a"] <= 0.0075 and band > ESTIMATOR_BAND_PP:
            findings.append({
                "code": "ESTIMATOR_SCATTER", "run": key,
                "detail": f"estimators disagree by {band:.3f} pp",
                "band_pp": band})

    # --- 3. gate values re-derived from the dump ----------------------------
    def run(ratio, h, depth=96.0):
        return res["runs"][run_key(ratio, h, depth)]

    def a_err(ratio, h):
        return abs(run(ratio, h)["error_pct"]["contact_radius_linear"])

    def d_err(ratio, h):
        return abs(run(ratio, h)["error_pct"]["indentation"])

    definitions = {}
    for ratio in (0.05, 0.10):
        tag = f"aR{int(round(ratio * 100)):03d}"
        definitions[f"contact_radius_{tag}"] = lambda r=ratio: a_err(r, 0.0075)
        definitions[f"indentation_{tag}"] = lambda r=ratio: d_err(r, 0.0075)
        definitions[f"refinement_stability_indentation_{tag}"] = (
            lambda r=ratio: abs(d_err(r, 0.0075) - d_err(r, 0.0300)))
    definitions["peak_pressure_aR005"] = lambda: abs(
        run(0.05, 0.0075)["error_pct"]["peak_pressure_consistent"])
    definitions["penetration_aR005"] = lambda: run(0.05, 0.0075)["penetration_over_delta"]
    definitions["load_residual_aR005"] = lambda: run(0.05, 0.0075)["load_residual"]
    definitions["equilibrium_residual_aR005"] = (
        lambda: run(0.05, 0.0075)["equilibrium_residual_rel"])

    def spread(field):
        vals = [r["numeric"][field] for r in res["rho"]]
        return 100.0 * (max(vals) - min(vals)) / (sum(vals) / len(vals))

    definitions["rho_spread_contact_radius"] = lambda: spread("contact_radius_linear")
    definitions["rho_spread_indentation"] = lambda: spread("indentation")

    ladder = sorted(res["truncation"], key=lambda r: r["domain_depth_over_a"])
    by_depth = {r["domain_depth_over_a"]: r for r in ladder}
    if 48.0 in by_depth and 96.0 in by_depth:
        definitions["truncation_indentation_d48_to_96"] = lambda: abs(
            abs(by_depth[48.0]["error_pct"]["indentation"])
            - abs(by_depth[96.0]["error_pct"]["indentation"]))
        definitions["truncation_contact_radius_d48_to_96"] = lambda: abs(
            abs(by_depth[48.0]["error_pct"]["contact_radius_linear"])
            - abs(by_depth[96.0]["error_pct"]["contact_radius_linear"]))
    if len(ladder) >= 3:
        def extrap():
            seq = [(r["domain_depth_over_a"],
                    abs(r["error_pct"]["indentation"])) for r in ladder]
            limit, _ = richardson(seq)
            return abs(limit)
        definitions["truncation_extrapolated_indentation"] = extrap

    for gate in cert.get("gates", []) + tr.get("gates", []):
        gid = gate["id"]
        declared_pass = bool(gate["passed"])
        op, value, threshold = gate.get("operator", "<="), gate["value"], gate["threshold"]
        ok = {"<=": value <= threshold, ">=": value >= threshold,
              "==": value == threshold}.get(op)
        if ok is not None and ok != declared_pass:
            findings.append({"code": "VERDICT_CONTRADICTION", "gate": gid,
                             "value": value, "operator": op, "threshold": threshold,
                             "declared_passed": declared_pass})
        if gid in definitions:
            mine = float(definitions[gid]())
            recomputed[gid] = {"stored": value, "recomputed": mine}
            # tolerances: exact for identities, banded for a quantity that is
            # legitimately ladder-dependent (Richardson limit)
            tol = BAND_TOL.get(gid, max(1e-6, 1e-6 * abs(value)))
            if abs(mine - value) > tol:
                findings.append({"code": "GATE_VALUE_MISMATCH", "gate": gid,
                                 "stored": value, "recomputed": mine})
        else:
            recomputed[gid] = {"stored": value, "recomputed": None}

    seen, uniq = set(), []
    for f in findings:
        key = (f["code"], f.get("gate") or f.get("run"), f.get("field"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(f)
    findings = uniq

    return {
        "anchor": "benchmarks/verified/fem_hertz_contact_loadcontrol",
        "generated_by": "tools/hertz_anchor_recompute.py",
        "runs_checked": sorted(res["runs"]),
        "gates_total": len(cert.get("gates", [])),
        "gates_value_recomputed": sum(1 for v in recomputed.values()
                                      if v["recomputed"] is not None),
        "gates_verdict_checked": len(cert.get("gates", [])) + len(tr.get("gates", [])),
        "gates": recomputed,
        "findings": findings,
        "note": ("Gates whose quantity has no independent definition here are "
                 "reported with recomputed=null: their value is checked for "
                 "verdict consistency only, not re-derived."),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default=ANCHOR)
    ap.add_argument("--json", default=None)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args(argv)
    out = recompute(args.anchor)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
    print(f"runs: {len(out['runs_checked'])}  gates: {out['gates_total']} "
          f"(values re-derived: {out['gates_value_recomputed']}, "
          f"verdicts checked: {out['gates_verdict_checked']})")
    for gid, v in out["gates"].items():
        mine = "not recomputed" if v["recomputed"] is None else f"{v['recomputed']:.6g}"
        print(f"  {gid:<42} stored {v['stored']:<12.6g} recomputed {mine}")
    if out["findings"]:
        print(f"\nFINDINGS: {len(out['findings'])}")
        for f in out["findings"]:
            print("  -", json.dumps(f, ensure_ascii=False))
    else:
        print("\nFINDINGS: none")
    return 1 if (out["findings"] and args.strict) else (1 if out["findings"] else 0)


if __name__ == "__main__":
    sys.exit(main())