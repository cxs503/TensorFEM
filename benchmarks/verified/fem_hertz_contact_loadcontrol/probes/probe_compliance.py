#!/usr/bin/env python3
"""Probe: compliance (indentation vs load) of the load-controlled Hertz contact.

Reads the result.json of the parent anchor and checks that the FE compliance
(delta/P) matches the Hertz closed form within the declared tolerance.  This
is a *post-hoc* probe: it does not re-run the solver, it only grades the
already-recorded numbers.

Usage
-----
    python benchmarks/verified/fem_hertz_contact_loadcontrol/probes/probe_compliance.py
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT = os.path.join(HERE, "..", "result.json")
OUT = os.path.join(HERE, "compliance_probe.json")

R, E, NU = 10.0, 1e5, 0.3
E_STAR = E / (1.0 - NU**2)

CRITERIA = {
    "compliance_pct": 3.0,
    "reference": "Hertz 1882; delta = a^2/R, a = (3PR/(4E*))^(1/3) => delta/P = (9/(16 R E*^2))^(1/3) P^(-1/3)",
}


def hertz_compliance(load):
    """Closed-form compliance delta/P for a rigid sphere on an elastic half-space."""
    a = (3.0 * load * R / (4.0 * E_STAR)) ** (1.0 / 3.0)
    delta = a**2 / R
    return delta / load


def probe():
    with open(RESULT) as fh:
        result = json.load(fh)

    rows = result.get("runs", {})
    probes = []
    for key, row in sorted(rows.items()):
        if "numeric" not in row or "analytic" not in row:
            continue  # skip summary-only entries
        load = row["load"]
        delta_fe = row["numeric"]["indentation"]
        delta_ref = row["analytic"]["indentation"]
        compliance_fe = delta_fe / load
        compliance_ref = delta_ref / load
        compliance_hertz = hertz_compliance(load)
        err_pct = 100.0 * (compliance_fe / compliance_hertz - 1.0)
        probes.append({
            "key": key,
            "load": load,
            "delta_fe": delta_fe,
            "delta_ref": delta_ref,
            "compliance_fe": compliance_fe,
            "compliance_hertz": compliance_hertz,
            "error_pct": err_pct,
            "passed": abs(err_pct) <= CRITERIA["compliance_pct"],
        })

    all_passed = all(p["passed"] for p in probes)
    payload = {
        "schema": "tensorfem.probe.compliance.v1",
        "anchor": "fem_hertz_contact_loadcontrol",
        "criteria": CRITERIA,
        "probes": probes,
        "passed": all_passed,
        "failed": [p["key"] for p in probes if not p["passed"]],
    }
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(f"wrote {OUT}  ({len(probes)} probes, {'PASS' if all_passed else 'FAIL'})")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(probe())
