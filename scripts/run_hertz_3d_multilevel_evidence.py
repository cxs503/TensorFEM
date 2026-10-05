"""Run a real finite-strain TET4/Mortar Hertz sequence and fail closed."""
from __future__ import annotations
import hashlib, json, tempfile, time
from dataclasses import asdict
from pathlib import Path
from tensorfem.hertz_3d_fe import solve_hertz_cap_block

def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def main(output: str = ".qualification/hertz-3d-fe/multilevel-evidence.json"):
    started=time.time(); cases=[]
    for cells in (2, 3, 4):
        args={"cells": cells}
        try:
            result=solve_hertz_cap_block(**args)
            cases.append({"level":cells,"arguments":args,"status":"converged","result":asdict(result)})
        except Exception as exc:
            cases.append({"level":cells,"arguments":args,"status":"failed",
                          "error_type":type(exc).__name__,"error":str(exc)})
            break
    converged=[c for c in cases if c["status"]=="converged"]
    errors=[float(c["result"]["relative_error"]) for c in converged]
    monotone=len(errors)>=2 and all(b<=a for a,b in zip(errors,errors[1:]))
    finest_ok=bool(errors) and errors[-1]<=.03
    passed=bool(len(converged)==3 and monotone and finest_ok)
    clean={"schema":"tensorfem.hertz-3d-fe-multilevel-evidence/1.0",
           "qualification_status":"qualified" if passed else "blocked","passed":passed,
           "required_maximum_relative_error":.03,"required_levels":3,
           "monotone_relative_error":monotone,"cases":cases,
           "blockers":[] if passed else [
               "real TET4/Mortar Hertz multilevel sequence does not meet the <=3% gate",
               "symmetric quarter-domain boundary conditions are not yet part of this evidence"],
           "wall_time_seconds":time.time()-started}
    clean["evidence_sha256"]=hashlib.sha256(_canonical(clean).encode()).hexdigest()
    target=Path(output); target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=target.parent,delete=False,suffix=".tmp") as stream:
        json.dump(clean,stream,indent=2,sort_keys=True); stream.write("\n"); temporary=Path(stream.name)
    temporary.replace(target)
    print(json.dumps({"output":str(target),"passed":passed,"sha256":clean["evidence_sha256"]}))

if __name__ == "__main__": main()
