"""Run and atomically record true-3D Hertz FE negative evidence."""
import hashlib,json,tempfile,time
from dataclasses import asdict
from pathlib import Path
from tensorfem.hertz_3d_fe import solve_hertz_cap_block

def canonical(v): return json.dumps(v,sort_keys=True,separators=(",",":"),allow_nan=False)

def main():
    started=time.time();cases=[]
    specifications=(
        ("baseline_captured_contact",dict(cells=2)),
        ("large_domain_quality_mesh",dict(cells=4,lateral_size=3.2,block_depth=1.6,
            center_grading=1.5,vertical_cells=4,maximum_edge_ratio=6.)),
    )
    for name,arguments in specifications:
        result=solve_hertz_cap_block(**arguments)
        cases.append({"case":name,"arguments":arguments,"result":asdict(result)})
    clean={"schema":"tensorfem.hertz-3d-fe-negative-evidence/1.0",
        "passed":False,"qualification_status":"blocked",
        "required_maximum_relative_error":.03,
        "cases":cases,
        "blockers":["contact patch requires at least 6-8 integration cells across radius",
            "local-to-far-field conforming transition is not implemented",
            "two-level monotone Hertz convergence is absent"],
        "wall_time_seconds":time.time()-started}
    report={**clean,"evidence_sha256":hashlib.sha256(canonical(clean).encode()).hexdigest()}
    output=Path(".qualification/hertz-3d-fe/negative-evidence.json")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,suffix=".tmp") as stream:
        json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");tmp=Path(stream.name)
    tmp.replace(output);print(json.dumps({"output":str(output),"passed":False,
        "sha256":report["evidence_sha256"]}))
if __name__=="__main__": main()
