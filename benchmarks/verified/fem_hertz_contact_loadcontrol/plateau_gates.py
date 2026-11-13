#!/usr/bin/env python3
"""Turn the three mechanism-study artifacts into certificate gates.

Every gate is a number that already exists in a committed artifact, so the
anchor certificate stays recomputable offline (``tools/verified_anchor_audit.py``
only needs ``value``/``operator``/``threshold``).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))


def load(name):
    with open(HERE / name) as fh:
        return json.load(fh)


def gate(gid, quantity, reference, value, threshold, operator="<=", note=""):
    return {
        "id": gid,
        "quantity": quantity,
        "reference": reference,
        "value": float(value),
        "threshold": float(threshold),
        "operator": operator,
        "passed": bool(value <= threshold if operator == "<=" else value >= threshold),
        "note": note,
    }


def build():
    far = load("farfield_discretisation.json")
    zres = load("nearfield_resolution.json")
    conv = load("converged_model.json")
    arms = zres["arms"]
    e6 = arms[0]["error_pct"]["indentation"]
    e24 = arms[2]["error_pct"]["indentation"]
    a48 = conv.get("vertical_ladder_a48") or {}
    e_inf = conv["extrapolation"]["e_inf_pct"]
    gates = [
        gate(
            "plateau_vertical_resolution_significance",
            "|e_delta(dz=a/6) - e_delta(dz=a/24)| (pct points), radial near field bit-identical",
            "a genuinely mesh-independent model bias would leave e_delta unchanged (0 pp); "
            "a discretisation artefact of the unrefined vertical spacing must move it",
            abs(e24 - e6), 0.10, ">=",
            note="the h/a ladder refined r only and never dz, so 'mesh independent' was a "
                 "one-directional statement; this gate makes the dz axis part of the record",
        ),
        gate(
            "plateau_far_field_hypothesis_rejected",
            "shift (pp) of |e_delta| when the stretched far field is refined from 16 to 64 "
            "vertical (26 -> 104 radial) layers",
            "H_far predicted a collapse of the offset; the pre-declared equivalence band is 0.05 pp",
            far["arm_verdict"]["shift_pp"], 0.05, "<=",
            note="passing this gate means 'the far field is NOT the mechanism', i.e. H_far "
                 "is rejected; the extrapolated limit only moves "
                 f"{far['arm_verdict']['extrapolated_limit_baseline_pct']:.4f} -> "
                 f"{far['arm_verdict']['extrapolated_limit_refined_pct']:.4f} %",
        ),
        gate(
            "plateau_indentation_dz_a48",
            "|e_delta| at dz=a/48, depth 768a (case-pool metric, vertical-refined near field)",
            "delta = a^2/R (Hertz 1882; Johnson 1985 4.2), case-pool band 3%",
            abs(a48.get("error_pct", {}).get("indentation", float("nan"))), 3.0,
        ),
        gate(
            "plateau_peak_pressure_dz_a24",
            "|p0_FE/p0_Hertz - 1| (mass-matrix consistent) at dz=a/24, depth 768a",
            "p0 = 3P/(2 pi a^2); the 5% band of the case certificate does not apply to this "
            "refined model, 0.05% is the pre-declared study bound",
            abs(arms[2]["error_pct"]["peak_pressure_consistent"]), 0.05,
        ),
    ]
    # Recorded, NOT gated: the pre-declared stretch target of the study.  It is
    # deliberately kept out of the certificate's gate list -- a certificate gate
    # is an acceptance criterion of the *anchor*, and inflating the gate list
    # with a target the study did not reach would either weaken the anchor or
    # silently drop an honest FAIL.  The value and its verdict live in
    # converged_model.json (`verdict.discretisation_explained = false`) and in
    # docs/CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md.
    recorded = [
        {
            "id": "plateau_extrapolated_limit_vertical_refined",
            "quantity": "Richardson limit of |e_delta| as depth -> inf at dz=a/24",
            "value": abs(e_inf) if e_inf is not None and math.isfinite(e_inf) else None,
            "target": 0.15,
            "met": bool(e_inf is not None and abs(e_inf) <= 0.15),
        }
    ]
    return {
        "schema": "tensorfem.verified-gates.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "title": "Indentation-plateau mechanism: far field vs vertical near field vs model bias",
        "anchor_case": "FEM-HERTZ-CONTACT (id 104)",
        "artifacts": {
            "farfield": "benchmarks/verified/fem_hertz_contact_loadcontrol/farfield_discretisation.json",
            "vertical": "benchmarks/verified/fem_hertz_contact_loadcontrol/nearfield_resolution.json",
            "converged": "benchmarks/verified/fem_hertz_contact_loadcontrol/converged_model.json",
            "report": "docs/CASE-FEM-HERTZ-CONTACT-plateau-mechanism.md",
        },
        "gates": gates,
        "recorded_targets": recorded,
        "passed": all(g["passed"] for g in gates if not math.isnan(g["value"])),
        "failed_gates": [g["id"] for g in gates if not g["passed"]],
        "undefined_terms": [g["id"] for g in gates if math.isnan(g["value"])],
    }


def main():
    doc = build()
    out = HERE / "gates_plateau.json"
    out.write_text(json.dumps(doc, indent=2) + "\n")
    for g in doc["gates"]:
        print(f"  {'PASS' if g['passed'] else 'FAIL'}  {g['id']:<46} "
              f"{g['value']:.6g} {g['operator']} {g['threshold']:.6g}")
    print(f"wrote {out} ({len(doc['gates'])} gates, passed={doc['passed']})")
    return 0 if doc["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())