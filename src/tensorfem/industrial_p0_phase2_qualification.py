"""Executable qualification for layered shell plasticity and 3-D contact P0 work."""
from __future__ import annotations
import hashlib,json
import torch

from .external_benchmark_contracts import external_p0_contracts
from .global_contact3d import (GlobalSurfaceContactModel,NodeTrianglePair,
                               assemble_global_surface_contact,initial_global_surface_state,
                               solve_global_surface_contact_path)
from .layered_shell4_plasticity import (LayeredShell4Model,LayeredShell4State,
                                        assemble_layered_shell4,plane_stress_j2_update,
                                        solve_layered_shell4_path)

SCHEMA="tensorfem.industrial-p0-phase2/1.0"


def _shell_evidence():
    dtype=torch.float64
    nodes=torch.tensor([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.],
                        [0.,1.,0.],[1.,1.,0.],[2.,1.,0.]],dtype=dtype)
    model=LayeredShell4Model(nodes,torch.tensor([[0,1,4,3],[1,2,5,4]]),
                             200000.,.3,.1,250.,1500.,3)
    point=LayeredShell4State.virgin(model).points[0][0][0]
    target=310.;alpha=(target-model.yield_stress)/model.hardening
    strain=torch.tensor([target/model.young+alpha,-model.poisson*target/model.young-alpha/2,0.],dtype=dtype)
    stress,_,trial=plane_stress_j2_update(strain,model,point)
    material_error=abs(float(stress[0])/target-1.)
    fixed=[]
    for node,xyz in enumerate(nodes):
        fixed.extend((6*node+2,6*node+3,6*node+4,6*node+5))
        if float(xyz[0])==0.:fixed.extend((6*node,6*node+1))
    fixed=torch.tensor(sorted(set(fixed)));force=torch.zeros(model.n_dofs,dtype=dtype)
    force[12]=force[30]=18.
    path=solve_layered_shell4_path(model,force,fixed,[.5,1.,.25,0.],tolerance=2e-8)
    free=torch.tensor([i for i in range(model.n_dofs) if i not in set(fixed.tolist())])
    residual=max(float(torch.linalg.vector_norm(step.reaction[free])) for step in path)
    permanent=float(path[-1].displacement[12])
    passed=material_error<.03 and float(trial.alpha)>0 and residual<2e-6 and permanent>0
    return {"elements":2,"layers":3,"material_relative_error":material_error,
            "maximum_free_residual":residual,"permanent_extension":permanent,"passed":passed}


def _contact_evidence():
    dtype=torch.float64
    nodes=torch.tensor([[-2.,-2.,0.],[3.,-2.,0.],[3.,3.,0.],[-2.,3.,0.],[.2,.4,.05]],dtype=dtype)
    k=torch.zeros((15,15),dtype=dtype);k[-3:,-3:]=1000*torch.eye(3,dtype=dtype)
    model=GlobalSurfaceContactModel(nodes,k,(NodeTrianglePair(4,((0,1,2),(0,2,3))),),tuple(range(12)),1e4,1e3,.2)
    loads=[]
    for fx in (0.,300.,800.,1400.):
        f=torch.zeros(15,dtype=dtype);f[-3:]=torch.tensor([fx,0.,-100.],dtype=dtype);loads.append(f)
    steps=solve_global_surface_contact_path(model,loads,tolerance=1e-10)
    final=assemble_global_surface_contact(model,steps[-1].displacement,steps[-1].external,
                                           steps[-2].state,tangent=False)
    force_error=float(torch.linalg.vector_norm(final.force_imbalance))
    residual=max(step.residual_norm for step in steps)
    passed=force_error<1e-10 and residual<2e-8 and float(steps[-1].state.histories[0].dissipated_energy)>0
    return {"increments":len(steps),"final_face":steps[-1].state.histories[0].face,
            "force_imbalance":force_error,"maximum_residual":residual,
            "dissipated_energy":float(steps[-1].state.histories[0].dissipated_energy),"passed":passed}


def run_industrial_p0_phase2_qualification():
    shell=_shell_evidence();contact=_contact_evidence();contracts=external_p0_contracts()
    statuses={case.case_id:case.status for case in contracts}
    categories={
        "layered_plane_stress_shell_j2":{"status":"qualified_small_strain_prototype","evidence":4},
        "global_node_tri3_contact":{"status":"qualified_3d_prototype","evidence":5},
        "finite_rotation_shell_plasticity_postbuckling":{"status":"not_qualified","evidence":0},
        "general_mortar_self_contact":{"status":"not_qualified","evidence":0},
        "deformable_hertz_fe":{"status":"blocked_external_contract","evidence":0},
    }
    clean={"schema":SCHEMA,"passed":shell["passed"] and contact["passed"],"shell":shell,
           "contact":contact,"external_contract_status":statuses,"categories":categories}
    digest=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return {**clean,"report_hash":digest}


def validate_industrial_p0_phase2_qualification(report):
    if report.get("schema")!=SCHEMA or report.get("passed") is not True:raise ValueError("invalid or failed phase-2 report")
    clean={k:v for k,v in report.items() if k!="report_hash"}
    digest=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    if digest!=report.get("report_hash"):raise ValueError("phase-2 qualification hash mismatch")
    return report
