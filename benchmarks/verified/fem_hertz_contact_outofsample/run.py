#!/usr/bin/env python3
"""CASE FEM-HERTZ-CONTACT (case pool id 104) -- *out-of-sample* verification.

Why this harness exists
-----------------------
The anchor certificate (``benchmarks/verified/fem_hertz_contact_loadcontrol/``)
grades the load-controlled Hertz face at exactly two working points,
``a/R = 0.05`` and ``a/R = 0.10``, and records a *failure probe* at
``a/R = 0.02`` ("low-load failure mode", documented in
``docs/CASE-FEM-HERTZ-CONTACT.md`` §5.5, deliberately **not** gated).

A two-point certificate cannot distinguish "the FE face reproduces Hertz"
from "the FE face was tuned/validated where it was looked at".  This harness
asks the falsifiable question that the certificate leaves open:

    Between the certified band (0.05 .. 0.10) and the documented failure
    probe (0.02), and outside the certified band (0.15), does the *shipped*
    load-controlled solver still land within the pool tolerance?

Criteria below are **declared before the run** (file written first; the run
only appends results).  Nothing is re-tuned after seeing an error: mesh
recipe, estimator (``contact_radius_linear``, the certificate's primary
estimator) and tolerance are inherited unchanged from the certificate.

Working points (all new: none appears in any existing artifact)
---------------------------------------------------------------
* ``a/R = 0.03``   -- below the certified band, above the 0.02 failure probe.
                      If it passes, the failure boundary is bracketed to
                      (0.02, 0.03]; if it fails, the certified band starts
                      at 0.05 and the "low-load" caveat is wider than stated.
* ``a/R = 0.075``  -- exact midpoint of the two certified points (interpolation
                      check: the error curve must not have an interior spike).
* ``a/R = 0.15``   -- above the certified band (large-contact / small-strain
                      edge of the Hertz face; the paraboloid indenter stays
                      the exact Hertz profile, so the closed form is still the
                      oracle).

Control (not gated): ``a/R = 0.02`` -- reproduction of the documented failure
probe.  A negative control must still fail; if it silently passes, the
documented failure mode is stale and the doc must be corrected.

Replication (gated): ``a/R = 0.05`` and ``0.10`` at ``h/a = 0.0075`` are
recomputed through this harness and compared with the certificate's frozen
gate values.  This is a *pipeline reproducibility* check, **not** an
independent-physics check: the same public solver entry point is used, so a
green replication only proves the certificate numbers are reproducible from a
different driver.

Reference: Hertz (1882) J. reine angew. Math. 92:156; Johnson, *Contact
Mechanics* (1985) §4.2, DOI:10.1017/cbo9781139171731.

Run:
    PYTHONPATH=src python3 benchmarks/verified/fem_hertz_contact_outofsample/run.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))

from tensorfem.hertz_halfspace_loadcontrol import (  # noqa: E402
    graded_depth,
    hertz_load_reference,
    solve_axisymmetric_contact_load_control,
    two_zone_axis,
)

CERT_PATH = os.path.join(
    HERE, "..", "fem_hertz_contact_loadcontrol", "gates_verified.json"
)

# --- dimensionless working set, identical to the certificate -----------------
R = 10.0
E = 1.0e5
NU = 0.3

# --- criteria, fixed BEFORE the run ------------------------------------------
CRITERIA = {
    "contact_radius_pct": 3.0,          # case-pool mandate
    "indentation_pct": 3.0,             # case-pool mandate
    "refinement_stability_pct": 0.5,    # |err(h/a=0.0075) - err(h/a=0.03)| in pp
    "replication_tol_pp": 0.02,         # |this harness - certificate| in pp
    "penetration_over_delta_max": 1e-5,
    "load_residual_max": 1e-9,
    "hertz_kinematic_consistency_pct": 1.0,  # |delta_FE*R/a_FE^2 - 1|
    "reference": "Hertz 1882 J. reine angew. Math. 92:156; Johnson 1985 "
                 "Contact Mechanics 4.2 DOI:10.1017/cbo9781139171731",
    "case_pool": "FEM-HERTZ-CONTACT (id 104)",
}

# Why an auxiliary gate may fail *without* failing the anchor: these gate
# families were declared here (they are not case-pool metrics) to test whether
# the certificate's robustness claims transfer outside its two points.  A
# failure is a scoped finding about the *auxiliary* claim, and is recorded in
# ``auxiliary_failures`` instead of being silently dropped or silently passed.
AUX_EXPLANATIONS = {
    "refinement_stability": "mesh-refinement stability of the contact-radius "
        "estimator (certificate threshold 0.5 pp) does not transfer to this "
        "working point; the pool tolerance itself still holds",
    "penetration": "ALM constraint quality (certificate threshold 1e-5 delta) "
        "degrades at this working point; the pool metrics are unaffected "
        "because delta is a load-control unknown, not a penetration-derived quantity",
    "kinematic_consistency": "Hertz kinematic identity delta=a^2/R is not "
        "satisfied by the recovered FE radius here, i.e. the radius estimator "
        "(not the displacement solution) is the weak link at this point",
}

PRIMARY_H = 0.0075
COARSE_H = 0.0300
DEPTH_FACTOR = 96.0
N_OUTER_BAND = 26

REPLICATION = (0.05, 0.10)
OUT_OF_SAMPLE = (0.03, 0.075, 0.15)
CONTROL = (0.02,)


def load_for_radius_ratio(ratio):
    """Hertz load producing ``a/R = ratio`` (R = E = 1 dimensionless)."""
    e_star = E / (1.0 - NU**2)
    a = ratio * R
    return 4.0 * e_star * a**3 / (3.0 * R)


def solve_point(ratio, h_rel):
    """One load-controlled solve on the certificate's mesh recipe.

    Mesh recipe copied *as a specification* from the certificate driver
    (``two_zone_axis(3a, 96a, 3a/h, 26)`` + ``graded_depth(96a, 4a, 24, 16)``)
    so that the only new thing here is the working point.
    """
    ref = hertz_load_reference(load_for_radius_ratio(ratio), R, E, NU)
    a = ref.contact_radius
    h = h_rel * a
    n_inner = int(round(3.0 * a / h))
    r_axis = two_zone_axis(3.0 * a, DEPTH_FACTOR * a, n_inner, N_OUTER_BAND)
    z_axis = graded_depth(DEPTH_FACTOR * a, 4.0 * a, 24, 16)
    t0 = time.time()
    ref, state, diag = solve_axisymmetric_contact_load_control(
        load=ref.load,
        radius=R,
        young=E,
        poisson=NU,
        r_axis=r_axis,
        z_axis=z_axis,
        rho_factor=1.0,
        load_tol=1e-10,
    )
    wall = time.time() - t0
    a_fe = state.contact_radius_linear
    delta_fe = state.prescribed_indentation
    row = {
        "a_over_R": ratio,
        "h_over_a": h_rel,
        "domain_depth_over_a": DEPTH_FACTOR,
        "load": ref.load,
        "n_elements": diag["n_elements"],
        "ndof": diag["ndof"],
        "wall_s": round(wall, 2),
        "alm_iterations": diag["alm_iterations"],
        "outer_iterations": diag["outer_iterations"],
        "load_residual": diag["load_residual"],
        "penetration_over_delta": diag["penetration_over_indentation"],
        "analytic": {
            "contact_radius": ref.contact_radius,
            "indentation": ref.indentation,
            "peak_pressure": ref.peak_pressure,
        },
        "numeric": {
            "contact_radius_linear": a_fe,
            "contact_radius_fit": state.contact_radius_fit,
            "indentation": delta_fe,
            "peak_pressure_consistent": float(
                state.metadata["peak_pressure_consistent"]
            ),
            "peak_pressure_lumped": state.peak_pressure,
            "reaction": state.reaction,
        },
    }
    row["error_pct"] = {
        "contact_radius_linear": 100.0 * (a_fe / ref.contact_radius - 1.0),
        "contact_radius_fit": 100.0
        * (state.contact_radius_fit / ref.contact_radius - 1.0),
        "indentation": 100.0 * (delta_fe / ref.indentation - 1.0),
        "peak_pressure_consistent": 100.0
        * (state.metadata["peak_pressure_consistent"] / ref.peak_pressure - 1.0),
        "peak_pressure_lumped": 100.0
        * (state.peak_pressure / ref.peak_pressure - 1.0),
    }
    # Hertz kinematic identity delta = a^2/R evaluated with the *FE* radius:
    # non-circular because delta_FE is the load-control unknown and a_FE is a
    # post-processing recovery of the pressure footprint.
    row["hertz_kinematic_consistency_pct"] = 100.0 * (
        delta_fe * R / (a_fe * a_fe) - 1.0
    )
    return row


def gate(gid, quantity, value, threshold, reference, passed_note=""):
    return {
        "id": gid,
        "quantity": quantity,
        "reference": reference,
        "value": None if value is None else float(value),
        "threshold": float(threshold),
        "operator": "<=",
        "passed": bool(value is not None and abs(value) <= threshold),
        "note": passed_note,
    }


def find(rows, ratio, h_rel):
    for r in rows:
        if abs(r["a_over_R"] - ratio) < 1e-12 and abs(r["h_over_a"] - h_rel) < 1e-12:
            return r
    raise KeyError((ratio, h_rel))


POINT_DIR = os.path.join(HERE, "probes", "points")


def _point_path(ratio, h_rel):
    return os.path.join(
        POINT_DIR,
        f"point_aR{round(ratio * 1000):03d}_h{round(h_rel * 10000):04d}.json",
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ratios", default="",
                    help="comma-separated a/R subset to solve (default: all)")
    ap.add_argument("--force", action="store_true", help="ignore the point cache")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="grade from the cached points, solve nothing")
    args = ap.parse_args(argv)
    selected = ([float(x) for x in args.ratios.split(",") if x.strip()]
                if args.ratios else list(REPLICATION + OUT_OF_SAMPLE + CONTROL))
    os.makedirs(POINT_DIR, exist_ok=True)
    rows = []
    for ratio in selected:
        for h_rel in (PRIMARY_H, COARSE_H):
            path = _point_path(ratio, h_rel)
            if os.path.exists(path) and not args.force:
                with open(path) as fh:
                    row = json.load(fh)
            else:
                if args.aggregate_only:
                    raise SystemExit(f"missing cached point {path}")
                row = solve_point(ratio, h_rel)
                with open(path, "w") as fh:
                    json.dump(row, fh, indent=1)
                    fh.write("\n")
            rows.append(row)
            print(
                f"a/R={ratio:<6g} h/a={h_rel:<7g} nel={row['n_elements']:<6d} "
                f"a_err={row['error_pct']['contact_radius_linear']:+.3f}% "
                f"d_err={row['error_pct']['indentation']:+.3f}% "
                f"kin={row['hertz_kinematic_consistency_pct']:+.3f}% "
                f"{row['wall_s']:.0f}s",
                flush=True,
            )
    if args.ratios:
        return None  # partial run: only the cache is updated

    # ---- certificate values (frozen, read from the shipped artifact) --------
    with open(CERT_PATH) as fh:
        cert = json.load(fh)
    cert_gates = {g["id"]: g for g in cert["gates"]}

    gates = []
    # 1. replication of the certificate's primary cells
    for ratio, tag in ((0.05, "aR005"), (0.10, "aR010")):
        row = find(rows, ratio, PRIMARY_H)
        for qty, gid_key in (
            ("contact_radius_linear", f"contact_radius_{tag}"),
            ("indentation", f"indentation_{tag}"),
        ):
            cert_val = cert_gates[gid_key]["value"]
            mine = abs(row["error_pct"][qty])
            gates.append(gate(
                f"replication_{qty}_{tag}",
                f"|this harness - certificate| for |err| of {qty}, a/R={ratio:g}, h/a={PRIMARY_H:g} (pp)",
                abs(mine - cert_val),
                CRITERIA["replication_tol_pp"],
                f"certificate gate {gid_key} = {cert_val:.6f}",
                "pipeline reproducibility only; same public solver entry point",
            ))

    # 2. out-of-sample acceptance (<3% at the primary mesh)
    for ratio in OUT_OF_SAMPLE:
        row = find(rows, ratio, PRIMARY_H)
        gates.append(gate(
            f"oos_contact_radius_aR{round(ratio*1000):03d}",
            f"|a_FE/a_Hertz - 1|, a/R={ratio:g}, h/a={PRIMARY_H:g} (pre-declared point)",
            abs(row["error_pct"]["contact_radius_linear"]),
            CRITERIA["contact_radius_pct"],
            "a = (3 P R/(4 E*))^(1/3)",
        ))
        gates.append(gate(
            f"oos_indentation_aR{round(ratio*1000):03d}",
            f"|delta_FE/delta_Hertz - 1|, a/R={ratio:g}, h/a={PRIMARY_H:g} (pre-declared point)",
            abs(row["error_pct"]["indentation"]),
            CRITERIA["indentation_pct"],
            "delta = a^2/R",
        ))

    # 3. refinement stability at every out-of-sample point
    for ratio in OUT_OF_SAMPLE:
        fine = find(rows, ratio, PRIMARY_H)
        coarse = find(rows, ratio, COARSE_H)
        for qty, label in (("contact_radius_linear", "contact_radius"),
                           ("indentation", "indentation")):
            gates.append(gate(
                f"oos_refinement_stability_{label}_aR{round(ratio*1000):03d}",
                f"|err(h/a={PRIMARY_H:g}) - err(h/a={COARSE_H:g})| for {label}, a/R={ratio:g} (pp)",
                abs(abs(fine["error_pct"][qty]) - abs(coarse["error_pct"][qty])),
                CRITERIA["refinement_stability_pct"],
                "refinement must not move the error by more than 0.5 pp",
            ))

    # 4. constraint quality + Hertz kinematic identity at every point
    for row in rows:
        tag = f"aR{round(row['a_over_R']*1000):03d}_h{round(row['h_over_a']*10000):04d}"
        gates.append(gate(
            f"load_residual_{tag}", "|reaction/P - 1|",
            abs(row["numeric"]["reaction"] / row["load"] - 1.0),
            CRITERIA["load_residual_max"], "load-control closure",
        ))
        gates.append(gate(
            f"penetration_{tag}", "max penetration / delta",
            row["penetration_over_delta"],
            CRITERIA["penetration_over_delta_max"], "unilateral constraint quality",
        ))
        gates.append(gate(
            f"kinematic_consistency_{tag}", "|delta_FE*R/a_FE^2 - 1| (%)",
            abs(row["hertz_kinematic_consistency_pct"]),
            CRITERIA["hertz_kinematic_consistency_pct"],
            "Hertz kinematic identity evaluated with the FE radius; non-circular "
            "because delta_FE is the load-control unknown",
        ))

    # ---- control: documented low-load failure probe (NOT gated) -------------
    control_rows = [r for r in rows if any(abs(r["a_over_R"] - c) < 1e-12 for c in CONTROL)]
    control = []
    for row in control_rows:
        control.append({
            "a_over_R": row["a_over_R"],
            "h_over_a": row["h_over_a"],
            "a_err_pct": row["error_pct"]["contact_radius_linear"],
            "delta_err_pct": row["error_pct"]["indentation"],
            "inside_3pct": bool(
                abs(row["error_pct"]["contact_radius_linear"]) <= 3.0
                and abs(row["error_pct"]["indentation"]) <= 3.0
            ),
            "note": "reproduction of the documented low-load failure probe; "
                    "not gated (a negative control is *expected* to miss)",
        })

    # ---- envelope statement --------------------------------------------------
    oos_pass = {}
    for ratio in OUT_OF_SAMPLE:
        row = find(rows, ratio, PRIMARY_H)
        oos_pass[ratio] = (
            abs(row["error_pct"]["contact_radius_linear"]) <= 3.0
            and abs(row["error_pct"]["indentation"]) <= 3.0
        )
    control_inside = {c["a_over_R"]: c["inside_3pct"] for c in control}
    envelope = {
        "certified_band": [0.05, 0.10],
        "out_of_sample_pass": {f"{k:g}": v for k, v in oos_pass.items()},
        "control_inside_3pct": {f"{k:g}": v for k, v in control_inside.items()},
        "statement": _envelope_statement(oos_pass, control_inside),
    }

    passed = [g for g in gates if g["passed"]]
    failed = [g for g in gates if not g["passed"]]
    pool_ids = [g["id"] for g in gates
                if g["id"].startswith("oos_contact_radius_")
                or g["id"].startswith("oos_indentation_")]
    pool_ok = all(g["passed"] for g in gates if g["id"] in pool_ids)
    auxiliary_failures = {}
    for g in failed:
        family = g["id"][4:] if g["id"].startswith("oos_") else g["id"]
        for prefix, why in AUX_EXPLANATIONS.items():
            if family.startswith(prefix):
                auxiliary_failures[g["id"]] = {
                    "value": g["value"],
                    "threshold": g["threshold"],
                    "scope": why,
                }
                break
    doc = {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_hertz_contact_outofsample",
        "title": "Out-of-sample verification of the load-controlled Hertz face "
                 "(case pool FEM-HERTZ-CONTACT id 104)",
        "criteria": CRITERIA,
        "provenance": {
            "replication_source": os.path.relpath(CERT_PATH, HERE),
            "solver": "tensorfem.hertz_halfspace_loadcontrol."
                      "solve_axisymmetric_contact_load_control",
            "estimator": "ContactState.contact_radius_linear (certificate primary)",
            "mesh_recipe": f"two_zone_axis(3a, {DEPTH_FACTOR:g}a, 3a/h, {N_OUTER_BAND})"
                           " + graded_depth(96a, 4a, 24, 16)",
        },
        "gates": gates,
        "controls": control,
        "envelope": envelope,
        "passed": len(passed),
        "failed_gates": [g["id"] for g in failed],
        "pool_gates_all_passed": bool(pool_ok),
        "auxiliary_failures": auxiliary_failures,
        "limitations": [
            "Pipeline-replication only: the out-of-sample solves reuse the "
            "shipped solver (same FE kernel and same contact-radius estimator), "
            "so this harness tests generality of the *working point*, not "
            "independence of the discretisation.",
            "a/R=0.02 control is a negative control and is deliberately not "
            "gated: a passing control would invalidate the documented failure "
            "mode, not the anchor.",
            "The 96a truncation depth is inherited from the certificate; the "
            "residual ~-1% indentation offset is the documented model bias and "
            "is not addressed here.",
            "Only the axisymmetric paraboloid-indenter face is covered; a "
            "spherical (non-paraboloid) indenter at large a/R is a different "
            "model whose Hertz error is O(a/R) by construction.",
        ],
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "rows": rows,
    }
    out = os.path.join(HERE, "gates_outofsample.json")
    with open(out, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"\nwrote {out}")
    print(f"gates: {len(passed)}/{len(gates)} passed; failed={doc['failed_gates']}")
    print(f"envelope: {envelope['statement']}")
    return doc


def _envelope_statement(oos_pass, control_inside):
    passed = sorted(k for k, v in oos_pass.items() if v)
    failed = sorted(k for k, v in oos_pass.items() if not v)
    parts = []
    if passed:
        parts.append("passes at a/R " + ", ".join(f"{k:g}" for k in passed))
    if failed:
        parts.append("fails at a/R " + ", ".join(f"{k:g}" for k in failed))
    ctrl = ", ".join(
        f"a/R={k:g}: {'inside' if v else 'outside'} 3%" for k, v in control_inside.items()
    )
    return (
        "beyond the certified band 0.05..0.10 the load-controlled Hertz face "
        + ("; ".join(parts) if parts else "was not exercised")
        + f"; documented failure control ({ctrl})"
    )


if __name__ == "__main__":
    main()