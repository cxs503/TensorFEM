#!/usr/bin/env python
"""Independent consistency auditor for a GAP-PP-3 3-grid matrix evidence bundle.

Why this exists
---------------
``scripts/gap_pp3_3grid_study.py`` writes one JSON per mesh, ``gap_pp3_3grid_merge.py``
merges them and ``publish_gap_pp3_3grid_matrix.py`` renders a verdict document out of
the merged JSON.  Nothing in that chain checks that the numbers *inside* the bundle
are related to each other the way the code that produced them says they are, and
nothing checks that the merged file is the one the document quotes.  A bundle whose
columns were typed by hand therefore renders a perfectly readable verdict document
(that is exactly the defect that was found in ``docs/CASE-GAP-PP-3-3GRID-MATRIX-VERDICT.md``:
its "FE first yield" column and its "raw yield error" column cannot both be true).

This auditor re-derives every *derived* column from the bundle's own primitives with
an independent implementation (it does not import the study driver), so a bundle that
was not produced by a real solve fails here:

* ``first_yield_error`` must equal ``(fyf - p_y)/p_y`` -> ``p_y`` is recoverable from
  the pair and must reproduce the stored analytic pressure;
* ``analytic_first_yield_same_mesh`` must equal ``fyf / (1 + normalised error)``;
* the bias-normalised error must be consistent with the elastic bias chain;
* ``limit_load_error_vs_ub`` must equal ``(limit - UB)/UB``;
* the convergence block (h, observed order, Richardson limit) must be reproducible
  from the per-mesh errors and the *actual* mesh ratios of the family;
* the ``criteria`` block must be reproducible from the per-mesh entries.

Usage
-----
    python scripts/audit_gap_pp3_3grid_bundle.py docs/assets/gap_pp3_3grid_results.json
    python scripts/audit_gap_pp3_3grid_bundle.py BUNDLE --markdown out.md

Exit code 0 iff every blocking / major check passes (``minor`` checks never fail the
audit, they only print).
"""
from __future__ import annotations

import argparse
import json
import math
import sys

BLOCKING = "blocking"
MAJOR = "major"
MINOR = "minor"

REL_TOL = 1e-9          # algebraic identities inside one bundle
ABS_TOL_PRESSURE = 1e-6  # Pa


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-300)


def _close(a, b, tol=REL_TOL, abs_tol=0.0) -> bool:
    if a is None or b is None:
        return a is b
    return _rel(float(a), float(b)) <= tol or abs(float(a) - float(b)) <= abs_tol


def _finding(out: list, check: str, severity: str, ok: bool, detail: str) -> None:
    out.append({"check": check, "severity": severity, "ok": bool(ok), "detail": detail})


# ---------------------------------------------------------------------------
# independent re-implementation of the convergence rules
# ---------------------------------------------------------------------------
def observed_order(errors: list[float], h: list[float]) -> float | None:
    """``p`` from the last two meshes: ``e = e_lim + C h^p``."""
    if len(errors) < 2:
        return None
    e_c, e_f = errors[-2], errors[-1]
    h_c, h_f = h[-2], h[-1]
    if e_c == 0.0 or e_f == 0.0 or h_c <= h_f or e_c * e_f <= 0.0:
        return None
    return math.log(abs(e_f) / abs(e_c)) / math.log(h_f / h_c)


def richardson_limit(errors: list[float], h: list[float]) -> float | None:
    p = observed_order(errors, h)
    if p is None or len(errors) < 2:
        return None
    e_c, e_f = errors[-2], errors[-1]
    r = h[-2] / h[-1]
    return e_f + (e_f - e_c) / (r ** p - 1.0)


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def audit_bundle(bundle: dict) -> dict:
    findings: list[dict] = []
    case = bundle.get("case")
    analytic = bundle.get("analytic")
    meshes = bundle.get("meshes") or {}
    bounds = bundle.get("bounds") or {}

    # --- structure -------------------------------------------------------
    missing = [k for k in ("case", "material", "analytic", "meshes", "criteria")
               if not bundle.get(k)]
    _finding(findings, "S1-structure", BLOCKING, not missing,
             "all top level blocks present" if not missing else f"missing blocks: {missing}")
    _finding(findings, "S2-mesh-count", BLOCKING, len(meshes) >= 3,
             f"{len(meshes)} mesh entries (3 required for a 3-grid matrix)")

    if not case or not analytic or len(meshes) < 1:
        return _summarise(findings, bundle)

    b = float(case["b"])
    p_y = analytic.get("first_yield_pressure")
    ub = analytic.get("yield_line_upper_pressure")

    # --- per mesh algebraic identities -----------------------------------
    for name, m in meshes.items():
        err = m.get("first_yield_error")
        fyf = m.get("first_yield_factor")
        if err is not None and fyf is not None and p_y:
            recovered = float(fyf) / (1.0 + float(err))
            _finding(
                findings, f"M1-analytic-recovered[{name}]", BLOCKING,
                _close(recovered, float(p_y), abs_tol=ABS_TOL_PRESSURE),
                f"p_y implied by (first_yield_factor, first_yield_error) = "
                f"{recovered/1e6:.6f} MPa vs stored analytic "
                f"{float(p_y)/1e6:.6f} MPa ({100*_rel(recovered, float(p_y)):+.3f} %)",
            )
        norm = m.get("first_yield_error_bias_normalised")
        corrected = m.get("analytic_first_yield_same_mesh")
        if norm is not None and fyf is not None and corrected:
            recovered = float(fyf) / (1.0 + float(norm))
            _finding(
                findings, f"M2-normalised-recovered[{name}]", BLOCKING,
                _close(recovered, float(corrected), abs_tol=ABS_TOL_PRESSURE),
                f"same-mesh analytic implied by (first_yield_factor, normalised "
                f"error) = {recovered/1e6:.6f} MPa vs stored "
                f"{float(corrected)/1e6:.6f} MPa",
            )
        nx, ny = m.get("nx"), m.get("ny")
        if nx and ny:
            _finding(findings, f"M3-size-arithmetic[{name}]", BLOCKING,
                     m.get("elements") in (None, nx * ny)
                     and m.get("dofs") in (None, 6 * (nx + 1) * (ny + 1)),
                     f"elements {m.get('elements')} / dofs {m.get('dofs')} vs "
                     f"{nx}x{ny} -> {nx*ny} / {6*(nx+1)*(ny+1)}")
        limit = m.get("limit_load_measured")
        limit_err = m.get("limit_load_error_vs_ub")
        if limit is not None and limit_err is not None and ub:
            recomputed = (float(limit) - float(ub)) / float(ub)
            _finding(findings, f"M4-limit-error[{name}]", BLOCKING,
                     _close(recomputed, float(limit_err), tol=1e-9),
                     f"(limit-UB)/UB = {recomputed:+.6f} vs stored {float(limit_err):+.6f}")
        pd = m.get("permanent_deflection")
        path = m.get("path") or []
        if pd is not None and m.get("permanent_set_valid"):
            peak = max((p[1] for p in path), default=None)
            ok = float(pd) > 0.0 and (peak is None or float(pd) < peak)
            _finding(findings, f"M5-permanent-set[{name}]", BLOCKING, ok,
                     f"w_res = {float(pd):.4e} m, peak = "
                     f"{'n/a' if peak is None else f'{peak:.4e} m'}")
        # a measured limit must sit inside the rigid-plastic bracket
        lower = bounds.get("strip_lower_pressure")
        if limit is not None and lower and ub:
            _finding(findings, f"M6-limit-in-bracket[{name}]", MAJOR,
                     float(lower) * 0.9 <= float(limit) <= float(ub) * 1.02,
                     f"limit {float(limit)/1e6:.4f} MPa in "
                     f"[{float(lower)/1e6:.4f}, {float(ub)/1e6:.4f}] MPa")

    # --- convergence block ----------------------------------------------
    order = sorted(meshes.items(), key=lambda kv: kv[1].get("elements", 0))
    errors = [m.get("first_yield_error") for _, m in order]
    h = [b / float(m["ny"]) for _, m in order]
    conv = (bundle.get("criteria") or {}).get("raw_error_convergence") or {}
    if len(errors) >= 2 and all(e is not None for e in errors):
        stored_h = conv.get("h")
        _finding(findings, "C1-h-values", BLOCKING,
                 stored_h is None or all(
                     _close(a, c, tol=1e-9) for a, c in zip(stored_h, h)),
                 f"h from b/ny = {[round(v, 6) for v in h]} vs stored "
                 f"{[round(v, 6) for v in stored_h] if stored_h else None}")
        p = observed_order(errors, h)
        stored_p = conv.get("observed_order")
        if p is None:
            _finding(findings, "C2-observed-order", MAJOR, stored_p is None,
                     "order undefined on this error sequence (no usable sign / "
                     "refinement) and none was claimed"
                     if stored_p is None else
                     f"stored observed_order {stored_p} is not reproducible")
        else:
            _finding(findings, "C2-observed-order", BLOCKING,
                     stored_p is not None and _close(p, stored_p, tol=1e-6, abs_tol=1e-9),
                     f"p = {p:.6f} vs stored {stored_p}")
        lim = richardson_limit(errors, h)
        stored_lim = conv.get("extrapolated_error")
        if lim is not None:
            _finding(findings, "C3-richardson-limit", BLOCKING,
                     stored_lim is not None
                     and _close(lim, stored_lim, tol=1e-6, abs_tol=1e-9),
                     f"Richardson limit {lim:+.6f} vs stored {stored_lim}")
    else:
        _finding(findings, "C1-h-values", MAJOR, False,
                 "cannot recompute convergence: an error column is missing")

    # --- criteria block --------------------------------------------------
    criteria = bundle.get("criteria") or {}
    if criteria:
        finest_name, finest = order[-1]
        finest_raw = finest.get("first_yield_error")
        block = criteria.get("transition_lt_5pct") or {}
        if finest_raw is not None and "met_at_finest_mesh" in block:
            _finding(findings, "K1-transition-flag", BLOCKING,
                     bool(block["met_at_finest_mesh"]) == (abs(finest_raw) < 0.05),
                     f"finest raw error {finest_raw:+.4f} -> met_at_finest_mesh "
                     f"{block['met_at_finest_mesh']}")
        if "met_in_mesh_limit" in block:
            # the flag is checked against the bundle's *own* reported limit (C3 checks
            # that reported number against the data, so the two cannot drift together)
            lim = conv.get("extrapolated_error")
            expected = None if lim is None else abs(lim) < 0.05
            _finding(findings, "K2-mesh-limit-flag", BLOCKING,
                     expected is not None and bool(block["met_in_mesh_limit"]) == expected,
                     f"Richardson limit {lim} -> met_in_mesh_limit "
                     f"{block['met_in_mesh_limit']}")
        lb = criteria.get("limit_lt_10pct") or {}
        if lb and finest.get("limit_load_error_vs_ub") is not None:
            _finding(findings, "K3-limit-flag", BLOCKING,
                     bool(lb.get("met_at_finest_mesh"))
                     == (abs(finest["limit_load_error_vs_ub"]) < 0.10),
                     f"finest limit error {finest['limit_load_error_vs_ub']:+.4f} -> "
                     f"met_at_finest_mesh {lb.get('met_at_finest_mesh')}")

    # --- provenance / plausibility (never blocking) ----------------------
    shards = bundle.get("shards") or {}
    _finding(findings, "P1-shard-provenance", MAJOR,
             bool(shards.get("used")),
             f"shards used: {shards.get('used')}, missing: {shards.get('missing')}")
    seconds = [m.get("seconds") for _, m in order]
    if seconds and all(s is not None for s in seconds) and len(set(seconds)) == 1:
        _finding(findings, "P2-timing-plausibility", MINOR, False,
                 f"every mesh reports the same wall time {seconds[0]} s, which is "
                 "not what a refinement study produces")
    else:
        _finding(findings, "P2-timing-plausibility", MINOR, True,
                 f"wall times per mesh: {seconds}")

    return _summarise(findings, bundle)


def _summarise(findings: list[dict], bundle: dict) -> dict:
    failed = [f for f in findings if not f["ok"] and f["severity"] in (BLOCKING, MAJOR)]
    warned = [f for f in findings if not f["ok"] and f["severity"] == MINOR]
    return {
        "ok": not failed,
        "n_checks": len(findings),
        "n_failed": len(failed),
        "n_warnings": len(warned),
        "findings": findings,
        "analysed_meshes": sorted((bundle.get("meshes") or {}).keys()),
    }


def render_markdown(report: dict) -> str:
    lines = ["| check | severity | ok | detail |", "|---|---|---|---|"]
    for f in report["findings"]:
        lines.append(f"| `{f['check']}` | {f['severity']} | "
                     f"{'yes' if f['ok'] else 'NO'} | {f['detail'].replace('|', '/')} |")
    head = (f"**audit: {'PASS' if report['ok'] else 'FAIL'}** — "
            f"{report['n_checks']} checks, {report['n_failed']} failed, "
            f"{report['n_warnings']} warnings\n")
    return head + "\n" + "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle")
    ap.add_argument("--markdown", default=None, help="write the check table here")
    args = ap.parse_args(argv)
    with open(args.bundle) as handle:
        bundle = json.load(handle)
    report = audit_bundle(bundle)
    text = render_markdown(report)
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as handle:
            handle.write(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))