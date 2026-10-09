"""Atomically run the opt-in curved finite-strain friction qualification."""
import json,os,tempfile,time
from pathlib import Path
from tensorfem.finite_strain_contact3d import run_curved_finite_strain_friction_qualification

if __name__=="__main__":
    if os.environ.get("TENSORFEM_RUN_SLOW_FINITE_STRAIN_FRICTION")!="1":
        raise SystemExit("set TENSORFEM_RUN_SLOW_FINITE_STRAIN_FRICTION=1")
    output=Path(os.environ.get("TENSORFEM_FINITE_STRAIN_FRICTION_OUTPUT",
        ".qualification/curved-finite-strain-friction/report.json"))
    output.parent.mkdir(parents=True,exist_ok=True);started=time.time()
    report=run_curved_finite_strain_friction_qualification();report["wall_time_seconds"]=time.time()-started
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,suffix=".tmp") as stream:
        json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");tmp=Path(stream.name)
    tmp.replace(output);print(json.dumps({"passed":True,"output":str(output),
        "sha256":report["evidence_sha256"],"wall_time_seconds":report["wall_time_seconds"]}))
