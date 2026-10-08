#!/usr/bin/env python3
"""Fail-closed validation for standard benchmark reports."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path

SCHEMA="tensorfem.standard-benchmark-report/1.0"
def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)
def validate(path: Path):
    report=json.loads(path.read_text()); digest=report.get("evidence_sha256")
    clean={k:v for k,v in report.items() if k!="evidence_sha256"}
    if report.get("schema")!=SCHEMA: raise ValueError("unsupported benchmark report schema")
    if digest!=hashlib.sha256(canonical(clean).encode()).hexdigest(): raise ValueError("benchmark report hash mismatch")
    cases=report.get("cases",{})
    required={"cantilever_beam_tip_load","simply_supported_plate_uniform_pressure","hertz_spherical_contact"}
    if not required.issubset(cases): raise ValueError("benchmark case missing")
    if report.get("error_policy",{}).get("target_relative_error")!=.03: raise ValueError("invalid error gate")
    if set(report.get("rendering",{}).get("formats",())) < {"json","html","pdf","docx"}:
        raise ValueError("report rendering formats incomplete")
    for name,case in cases.items():
        spec=case.get("specification")
        if not isinstance(spec,dict) or not spec.get("problem") or not spec.get("conditions"):
            raise ValueError(f"benchmark specification incomplete: {name}")
        if "finite_element" in case:
            error=case.get("relative_error"); status=case.get("status")
            if not isinstance(error,(int,float)) or status not in {"qualified","blocked","reference-only"}:
                raise ValueError(f"incomplete FE result: {name}")
            if status == "qualified":
                raise ValueError(f"scalar snapshot cannot certify a field report: {name}")
            reference = (case.get("reference", {}).get("tip_displacement")
                         or case.get("reference", {}).get("center_deflection")
                         or case.get("reference_force"))
            value = case["finite_element"].get("value")
            if reference is None or value is None or not all(math.isfinite(float(v)) for v in (reference, value)):
                raise ValueError(f"invalid scalar/reference: {name}")
            actual_error = abs(float(value)-float(reference))/abs(float(reference))
            if not math.isclose(error, actual_error, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"stored scalar error does not match values: {name}")
            scalar = case.get("scalar_status")
            if scalar not in {"qualified", "blocked"}:
                raise ValueError(f"missing scalar status: {name}")
            if (0 <= error < .03) != (scalar == "qualified"):
                raise ValueError(f"FE scalar status/error mismatch: {name}")
            if status != ("reference-only" if scalar == "qualified" else "blocked"):
                raise ValueError(f"FE field status mismatch: {name}")
    return report
def main():
    p=argparse.ArgumentParser(); p.add_argument("report",type=Path); a=p.parse_args()
    report=validate(a.report); print(json.dumps({"schema":report["schema"],"valid":True,"sha256":report["evidence_sha256"]}))
if __name__=="__main__": main()
