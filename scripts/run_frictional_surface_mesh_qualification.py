"""Atomically persist the opt-in curved frictional mesh qualification."""
from __future__ import annotations
import hashlib,json,os,tempfile,time
from pathlib import Path
from tensorfem.frictional_surface_path import run_frictional_surface_mesh_qualification

if __name__=="__main__":
    if os.environ.get("TENSORFEM_RUN_SLOW_FRICTIONAL_SURFACE")!="1":
        raise SystemExit("set TENSORFEM_RUN_SLOW_FRICTIONAL_SURFACE=1")
    output=Path(os.environ.get("TENSORFEM_FRICTIONAL_SURFACE_OUTPUT",
        ".qualification/frictional-surface-mesh/report.json"))
    output.parent.mkdir(parents=True,exist_ok=True);started=time.time()
    report=run_frictional_surface_mesh_qualification();report["wall_time_seconds"]=time.time()-started
    report["evidence_sha256"]=hashlib.sha256(json.dumps(report,sort_keys=True,
        separators=(",",":"),allow_nan=False).encode()).hexdigest()
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,suffix=".tmp") as stream:
        json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");tmp=Path(stream.name)
    tmp.replace(output);print(json.dumps({"passed":True,"output":str(output),
        "wall_time_seconds":report["wall_time_seconds"]}))
