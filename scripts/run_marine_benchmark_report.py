#!/usr/bin/env python3
"""Build the auditable ship/offshore benchmark report.

The report is deliberately evidence-first: analytical benchmarks may qualify
when their executable oracle is below 3%, while cases requiring a physical FE
field remain ``reference-only`` until a solver field is supplied.  No field is
invented by this script.
"""
from __future__ import annotations

import argparse, hashlib, json, tempfile
from pathlib import Path

from tensorfem.marine_benchmark_suite import run_marine_benchmarks

SCHEMA = "tensorfem.marine-benchmark-report/1.0"
_PURPOSE = {
    "buckling": "local plate stability under in-plane compression",
    "fatigue": "weld hot-spot and cumulative fatigue screening",
    "fracture": "hot-spot, LEFM and crack-tip J-integral post-processing",
    "hull": "longitudinal hull-girder strength and section yielding",
    "panel": "stiffened-panel elastic response",
    "hydro": "box-barge hydrostatics and small-angle stability",
    "morison": "slender-member wave-current loading",
    "postbuckling": "imperfect plate post-buckling path",
    "strip": "imperfection/residual-stress strip response",
    "shell": "initial-stress Shell4 buckling",
}

def _purpose(capability: str) -> str:
    c = capability.lower()
    for key, value in _PURPOSE.items():
        if key in c:
            return value
    return "executable ship/offshore mechanics verification case"

def build_report() -> dict:
    evidence = run_marine_benchmarks()
    cases = []
    for item in evidence:
        d = item.to_dict()
        d.update({
            "problem": _purpose(item.capability),
            "calculation_conditions": {
                "units": item.unit,
                "reference_source": item.source,
                "acceptance": "relative error < 3%; strict inequality",
            },
            "calculation_process": [
                "construct the documented marine benchmark input",
                "evaluate the TensorFEM qualification kernel",
                "compare the computed quantity with the independent reference",
                "record error and fail closed when the 3% gate is exceeded",
            ],
            "results": {
                "computed": item.computed,
                "reference": item.reference,
                "relative_error": item.error,
                "status": "reference-only" if item.passed else "blocked",
                "scalar_status": "qualified" if item.passed else "blocked",
            },
            # A cloud is intentionally absent for scalar qualification kernels.
            # FE stress/displacement fields are accepted later by the renderer.
            "field_results": {"stress_cloud": None, "displacement_cloud": None,
                               "note": "scalar oracle case; FE field required for cloud certification"},
        })
        cases.append(d)
    report = {
        "schema": SCHEMA, "report_version": "1.0",
        "title": "TensorFEM Ship and Offshore Engineering Benchmark Qualification",
        "purpose": "Executable public verification matrix; not class-rule certification.",
        "error_policy": {"target_relative_error": 0.03,
                          "qualified": "error < 0.03",
                          "over_limit": "blocked",
                          "missing_field": "reference-only"},
        "calculation_conditions": {"unit_system": "SI", "solver": "TensorFEM",
                                    "reference_data": "independent closed-form or refined oracle"},
        "cases": cases,
        "summary": {"total": len(cases), "qualified": 0,
                     "scalar_qualified": sum(x["passed"] for x in cases),
                     "reference_only": sum(x["passed"] for x in cases),
                     "blocked": sum(not x["passed"] for x in cases)},
    }
    clean = json.loads(json.dumps(report, sort_keys=True))
    clean["evidence_sha256"] = hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return clean

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = build_report(); a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=a.output.parent, delete=False, suffix=".tmp") as f:
        json.dump(report, f, indent=2, sort_keys=True); f.write("\n"); tmp = Path(f.name)
    tmp.replace(a.output)
    print(json.dumps({"output": str(a.output), "cases": report["summary"], "sha256": report["evidence_sha256"]}))

if __name__ == "__main__":
    main()
