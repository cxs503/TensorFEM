#!/usr/bin/env python3
"""Generate the checked public-API manifest from the current top-level exports."""
from __future__ import annotations
import argparse,json,re,subprocess
from pathlib import Path
import tensorfem

ROOT=Path(__file__).resolve().parents[1]
INTERNAL_MARKERS=("benchmark","_demo","postprocess","qualification","benchmark_cases","shell_benchmark_suite")
BETA_MODULES=("industrial_workflow","project_workflow","result_db","result_db_v2","mesh_project","step_executor")
STABLE_PREFIXES=("model","solvers","frame2d","continuum","modal","buckling","sparse_core","modeldb",
                 "nonlinear_step","thermal","explicit_dynamics","solid_plasticity","quadratic_solid","hex20")


def introduced(module):
    path="src/"+module.replace(".","/")+".py"
    try:
        commit=subprocess.check_output(["git","log","--diff-filter=A","--format=%H","--",path],cwd=ROOT,text=True).splitlines()[-1]
        text=subprocess.check_output(["git","show",f"{commit}:pyproject.toml"],cwd=ROOT,text=True)
        return re.search(r'^version\s*=\s*"([^"]+)"',text,re.M).group(1)
    except Exception:return tensorfem.__version__


def stability(module):
    short=module.rsplit(".",1)[-1]
    if any(x in short for x in INTERNAL_MARKERS):return "internal_candidate"
    if short in BETA_MODULES:return "beta"
    if short.startswith(STABLE_PREFIXES):return "stable"
    return "experimental"


def build():
    entries=[]
    for name in tensorfem.__all__:
        obj=getattr(tensorfem,name);module=getattr(obj,"__module__","tensorfem")
        entries.append({"name":name,"module":module,"object":getattr(obj,"__name__",type(obj).__name__),
                        "stability":stability(module),"introduced":introduced(module)})
    return {"schema":"tensorfem.public-api-manifest/1.0","package_version":tensorfem.__version__,
            "policy":{"stable":"semver compatibility","beta":"deprecate before removal",
                      "experimental":"may change with release notes","internal_candidate":"kept for compatibility; move to submodules after deprecation"},
            "entries":entries}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--write",type=Path);args=parser.parse_args()
    payload=json.dumps(build(),indent=2,sort_keys=True)+"\n"
    if args.write:args.write.write_text(payload,encoding="utf-8")
    else:print(payload,end="")
