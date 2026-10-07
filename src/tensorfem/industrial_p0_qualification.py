"""Executable qualification view for the first industrial P0 vertical slices."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import torch

from .analysis_workflow import AnalysisPlan, PlanStep, plan_from_dict, plan_to_dict, run_analysis
from .general_shell_nonlinear import GeneralShellMesh, assemble_general_shell
from .global_contact2d import (GlobalContactModel, NodePolylinePair,
                               solve_global_contact_path)
from .modeldb import (BoundaryCondition, ConcentratedLoad, ElementBlock,
                      Material, ModelDB, Section)
from .shell4_state_path import shell_with_initial_imperfection, solve_shell4_state_path

SCHEMA="tensorfem.industrial-p0-qualification/1.0"


def _shell_evidence():
    phi=torch.tensor([.45,.75],dtype=torch.float64);theta=torch.tensor([0.,.35,.7],dtype=torch.float64)
    gp,gt=torch.meshgrid(phi,theta,indexing="ij")
    nodes=5*torch.stack((torch.sin(gp)*torch.cos(gt),torch.sin(gp)*torch.sin(gt),torch.cos(gp)),-1).reshape(-1,3)
    base=GeneralShellMesh(nodes,torch.tensor([[0,3,4,1],[1,4,5,2]]),2e7,.25,.03)
    imperfection=torch.zeros_like(nodes);imperfection[4]=2e-4*nodes[4]/torch.linalg.vector_norm(nodes[4])
    mesh=shell_with_initial_imperfection(base,imperfection);load=torch.zeros(36,dtype=torch.float64)
    load[24:27]=-.05*mesh.nodes[4]/torch.linalg.vector_norm(mesh.nodes[4])
    fixed=list(range(18))+[23,29,35]
    out=solve_shell4_state_path(mesh,load,fixed,steps=2,step_size=.02,load_scale=.1,tolerance=2e-7)
    _,internal,_=assemble_general_shell(mesh,out.committed.dofs,tangent=False)
    residual=float(torch.linalg.vector_norm((internal-out.committed.load_factor*load)[out.free_dofs]))
    return {"facets":2,"accepted_steps":len(out.continuation.points),"residual_norm":residual,
            "nonnegative_section_energy":bool(torch.all(out.committed.section.energy>=0)),
            "passed":out.continuation.converged and residual<3e-7}


def _contact_evidence():
    x=torch.tensor([[0.,0.],[.5,0.],[1.5,0.],[.2,.05]],dtype=torch.float64);k=torch.zeros((8,8),dtype=torch.float64)
    k[6,6]=k[7,7]=1000.;model=GlobalContactModel(x,k,(NodePolylinePair(3,(0,1,2)),),tuple(range(6)),1e4,1e3,.2)
    loads=[]
    for fx in (0.,200.,500.,900.,1200.):
        f=torch.zeros(8,dtype=torch.float64);f[-2:]=torch.tensor([fx,-100.],dtype=torch.float64);loads.append(f)
    steps=solve_global_contact_path(model,loads,tolerance=1e-11);last=steps[-1]
    segments=[s.state.histories[0].segment for s in steps]
    return {"increments":len(steps),"segments":segments,"maximum_residual":max(s.residual_norm for s in steps),
            "dissipated_energy":float(last.state.histories[0].dissipated_energy),
            "passed":segments[-1]==1 and last.state.histories[0].dissipated_energy>0 and max(s.residual_norm for s in steps)<1e-9}


def _workflow_evidence():
    db=ModelDB(nodes={10:(0.,0.),20:(1.,0.),30:(2.,0.)},
        elements=[ElementBlock("T2D2",[101,102],[[10,20],[20,30]],"bars")],
        node_sets={"ALL":{10,20,30},"LEFT":{10},"RIGHT":{30}},element_sets={"BARS":{101,102}},
        materials={"STEEL":Material("steel",(200.,.3))},sections=[Section("BARS","STEEL","truss",(2.,))],
        boundaries=[BoundaryCondition("LEFT",1,1),BoundaryCondition("ALL",2,2)],
        loads=[ConcentratedLoad("RIGHT",1,40.)],metadata={"thermal":{"conductivity":5.,"density":1.,
        "specific_heat":1.,"thickness":1.,"fixed_temperature":{"10":100.,"30":20.},"nodal_heat":{}}})
    plan=AnalysisPlan("p0 vertical slice",db,(PlanStep("structure","modeldb.linear_static"),
        PlanStep("heat","modeldb.thermal_steady")),{"length":"m","force":"N","temperature":"K"})
    roundtrip=plan_to_dict(plan_from_dict(json.loads(json.dumps(plan_to_dict(plan)))))==plan_to_dict(plan)
    with tempfile.TemporaryDirectory(prefix="tensorfem-p0-") as raw:
        execution,directory=run_analysis(plan,raw)
        result_schema=json.loads((Path(directory)/"results.json").read_text())["schema"]
    return {"steps":len(execution.steps),"lossless_roundtrip":roundtrip,"result_schema":result_schema,
            "passed":execution.completed and roundtrip and result_schema=="tensorfem.result-db.v2"}


def run_industrial_p0_qualification() -> dict[str,object]:
    shell=_shell_evidence();contact=_contact_evidence();workflow=_workflow_evidence()
    categories={
        "unified_analysis_workflow":{"status":"qualified_initial_subset","evidence":3},
        "multifacet_shell_transaction":{"status":"qualified_elastic_prototype","evidence":3},
        "global_finite_sliding_contact":{"status":"qualified_2d_prototype","evidence":4},
        "shell_integration_point_plasticity":{"status":"not_qualified","evidence":0},
        "general_3d_mortar_self_contact":{"status":"not_qualified","evidence":0},
        "distributed_production_solver":{"status":"not_qualified","evidence":0},
    }
    clean={"schema":SCHEMA,"passed":shell["passed"] and contact["passed"] and workflow["passed"],
           "shell":shell,"contact":contact,"workflow":workflow,"categories":categories}
    digest=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {**clean,"report_hash":digest}


def validate_industrial_p0_qualification(report: dict[str,object]):
    if report.get("schema")!=SCHEMA or report.get("passed") is not True:raise ValueError("invalid or failed P0 report")
    clean={k:v for k,v in report.items() if k!="report_hash"}
    digest=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest!=report.get("report_hash"):raise ValueError("P0 qualification hash mismatch")
    return report
