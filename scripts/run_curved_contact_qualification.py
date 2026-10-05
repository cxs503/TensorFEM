"""Run and atomically persist the opt-in curved-contact qualification."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import time
import hashlib

from tensorfem.surface_surface_contact3d import run_curved_surface_contact_qualification


def main() -> None:
    if os.environ.get("TENSORFEM_RUN_SLOW_CURVED_CONTACT") != "1":
        raise SystemExit("set TENSORFEM_RUN_SLOW_CURVED_CONTACT=1")
    output=Path(os.environ.get("TENSORFEM_CURVED_CONTACT_OUTPUT",
        ".qualification/curved-surface-contact3d/report.json"))
    output.parent.mkdir(parents=True,exist_ok=True)
    started=time.time()
    report=run_curved_surface_contact_qualification()
    report["wall_time_seconds"]=time.time()-started
    report["runner"]="scripts/run_curved_contact_qualification.py"
    report["evidence_sha256"]=hashlib.sha256(json.dumps(
        report,sort_keys=True,separators=(",",":"),allow_nan=False
    ).encode()).hexdigest()
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,
                                     prefix=output.name+".",suffix=".tmp") as stream:
        json.dump(report,stream,indent=2,sort_keys=True)
        stream.write("\n"); temporary=Path(stream.name)
    temporary.replace(output)
    print(json.dumps({"output":str(output),"passed":report["passed"],
                      "wall_time_seconds":report["wall_time_seconds"]}))


if __name__=="__main__": main()
