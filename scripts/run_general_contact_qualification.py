"""Atomically compose verified general 3-D contact evidence."""
import argparse,json,tempfile
from pathlib import Path
from tensorfem.general_contact_qualification import qualify_general_contact3d

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--curved-frictionless",required=True);p.add_argument("--curved-frictional",required=True)
    p.add_argument("--curved-finite-strain-friction",required=True);p.add_argument("--hertz-3d",required=True)
    p.add_argument("--output",required=True);a=p.parse_args()
    report=qualify_general_contact3d(curved_frictionless=a.curved_frictionless,
        curved_frictional=a.curved_frictional,
        curved_finite_strain_friction=a.curved_finite_strain_friction,hertz_3d=a.hertz_3d)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile("w",dir=output.parent,delete=False,suffix=".tmp") as stream:
        json.dump(report,stream,indent=2,sort_keys=True);stream.write("\n");tmp=Path(stream.name)
    tmp.replace(output)
    if not report["passed"]: raise SystemExit("general contact composite remains blocked")
    print(json.dumps({"passed":True,"scope":report["scope"],"output":str(output)}))
if __name__=="__main__": main()
