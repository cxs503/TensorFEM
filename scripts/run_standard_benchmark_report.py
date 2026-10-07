#!/usr/bin/env python3
"""Generate a deterministic, machine-readable report for public closed-form cases."""
from __future__ import annotations
import argparse, json, hashlib, tempfile
from pathlib import Path
from tensorfem.standard_benchmarks import (cantilever_beam, cantilever_beam_field,
    simply_supported_plate_center, hertz_contact_force)

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--hertz-displacement", type=float, default=1e-4)
    p.add_argument("--fe-results", type=Path,
                   help="optional JSON mapping case names to FE values/field summaries")
    a=p.parse_args()
    beam=cantilever_beam(); plate=simply_supported_plate_center()
    report={"schema":"tensorfem.standard-benchmark-report/1.0",
      "cases":{
        "cantilever_beam_tip_load":{"reference":beam,
          "field_samples":[cantilever_beam_field(0.,0.),cantilever_beam_field(.5,.1),cantilever_beam_field(1.,.1)]},
        "simply_supported_plate_uniform_pressure":{"reference":plate},
        "hertz_spherical_contact":{"input":{"displacement":a.hertz_displacement},
          "reference_force":hertz_contact_force(a.hertz_displacement)},
      },
      "error_policy":{"target_relative_error":.03,"unqualified_status":"blocked",
                        "note":"closed-form references do not replace FE mesh convergence"}}
    # Dataclasses are encoded explicitly to keep the report portable.
    for case in report["cases"].values():
        if "field_samples" in case:
            case["field_samples"]=[{"x":s.x,"y":s.y,"displacement":s.displacement,"stress":s.stress}
                                    for s in case["field_samples"]]
    if a.fe_results:
        fe=json.loads(a.fe_results.read_text())
        for name,payload in fe.items():
            if name not in report["cases"]: raise ValueError(f"unknown FE case: {name}")
            target=report["cases"][name]; target["finite_element"] = payload
            ref=(target.get("reference",{}).get("tip_displacement")
                 or target.get("reference",{}).get("center_deflection")
                 or target.get("reference_force"))
            value=payload.get("value") if isinstance(payload,dict) else None
            if ref is not None and value is not None:
                error=abs(float(value)-float(ref))/abs(float(ref))
                target["relative_error"]=error
                target["status"]="qualified" if error<=.03 else "blocked"
    clean=json.loads(json.dumps(report,sort_keys=True,default=list))
    clean["evidence_sha256"]=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=a.output.parent,delete=False,suffix=".tmp") as f:
        json.dump(clean,f,indent=2,sort_keys=True); f.write("\n"); tmp=Path(f.name)
    tmp.replace(a.output); print(json.dumps({"output":str(a.output),"sha256":clean["evidence_sha256"]}))
if __name__ == "__main__": main()
