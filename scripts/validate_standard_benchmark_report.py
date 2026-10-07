#!/usr/bin/env python3
"""Fail-closed validation for standard benchmark reports."""
from __future__ import annotations
import argparse, hashlib, json
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
            if not isinstance(error,(int,float)) or status not in {"qualified","blocked"}:
                raise ValueError(f"incomplete FE result: {name}")
            if (error<=.03) != (status=="qualified"):
                raise ValueError(f"FE status/error mismatch: {name}")
            if status == "qualified":
                required=report["error_policy"].get("required_for_qualified",())
                missing=[key for key in required if key not in case and key not in case["finite_element"]]
                if missing:
                    raise ValueError(f"qualified result missing evidence {name}: {','.join(missing)}")
    return report
def main():
    p=argparse.ArgumentParser(); p.add_argument("report",type=Path); a=p.parse_args()
    report=validate(a.report); print(json.dumps({"schema":report["schema"],"valid":True,"sha256":report["evidence_sha256"]}))
if __name__=="__main__": main()
