"""Auditable, dependency-free post-processing for the 18-degree-hole hemisphere."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import torch

from .modeldb import ElementBlock,ModelDB
from .result_io import write_vtk


@dataclass(frozen=True)
class HemispherePost:
    nodes: torch.Tensor
    elements: torch.Tensor
    displacement: torch.Tensor
    rotation: torch.Tensor
    deformed: torch.Tensor
    reaction_force: torch.Tensor
    reaction_moment: torch.Tensor
    applied_force: torch.Tensor
    probe: dict
    path: tuple[dict,...]
    extrema: dict
    metadata: dict
    validation: dict


def _fixed_dofs(nphi,ntheta):
    fixed=[]
    for i in range(nphi+1):
        a=i*(ntheta+1);fixed += [6*a+1,6*a+3,6*a+5]
        b=a+ntheta;fixed += [6*b,6*b+4,6*b+5]
    return sorted(set(fixed+[2]))


def build_hemisphere_post(result,*,length_unit="benchmark_length",
                          force_unit="benchmark_force") -> HemispherePost:
    """Map the solver's six-DOF layout to nodal fields and validation metrics."""
    n=len(result.nodes);expected=6*n
    if result.solution.shape!=(expected,) or result.reaction.shape!=(expected,):
        raise ValueError("hemisphere result has inconsistent DOF layout")
    q=result.solution.reshape(n,6);raw_reaction=result.reaction.reshape(n,6)
    fixed=_fixed_dofs(result.meridional_elements,result.circumferential_elements)
    constrained=torch.zeros_like(result.reaction);constrained[fixed]=result.reaction[fixed]
    constrained=constrained.reshape(n,6)
    applied=torch.zeros((n,3),dtype=q.dtype,device=q.device)
    a=result.meridional_elements*(result.circumferential_elements+1);b=a+result.circumferential_elements
    applied[a,0]=1.;applied[b,1]=-1.
    balance_force=(constrained[:,:3]+applied).sum(0)
    balance_moment=(torch.linalg.cross(result.nodes,constrained[:,:3]+applied,dim=1)+
                    constrained[:,3:]).sum(0)
    free=torch.ones(expected,dtype=torch.bool,device=q.device);free[torch.tensor(fixed)]=False
    free_residual=float(torch.linalg.vector_norm(result.reaction[free]))
    force_scale=max(float(torch.linalg.vector_norm(applied.sum(0))),
                    float(torch.sum(torch.linalg.vector_norm(applied,dim=1))),1.)
    force_balance=float(torch.linalg.vector_norm(balance_force))/force_scale
    # Normalize moment with the benchmark radius (10) and absolute applied-force scale.
    moment_balance=float(torch.linalg.vector_norm(balance_moment))/(10.*force_scale)
    translation=q[:,:3];magnitude=torch.linalg.vector_norm(translation,dim=1)
    ids=torch.arange(result.meridional_elements+1)*(result.circumferential_elements+1)
    radial=torch.sum(translation[ids]*result.nodes[ids],dim=1)/10.
    path=tuple({"node_index":int(i),"polar_order":j,"radial_displacement":float(radial[j]),
                "magnitude":float(magnitude[i])} for j,i in enumerate(ids.tolist()))
    probe={"name":"loaded_equator_x","node_index":a,"dof":0,
           "value":float(translation[a,0]),"reference":float(result.reference)}
    extrema={"displacement_magnitude":{"min":float(magnitude.min()),"max":float(magnitude.max()),
        "max_node_index":int(torch.argmax(magnitude))},
        "component_min":translation.min(0).values.tolist(),
        "component_max":translation.max(0).values.tolist()}
    error=float(result.relative_error);tolerance=.03
    validation={"reference_displacement":float(result.reference),"computed_displacement":float(result.displacement),
        "relative_error":error,"error_tolerance":tolerance,"free_residual_norm":free_residual,
        "normalized_force_balance_error":force_balance,"force_balance_tolerance":1e-7,
        "normalized_moment_balance_error":moment_balance,"moment_balance_tolerance":.03,
        "balance_force":balance_force.tolist(),"balance_moment":balance_moment.tolist(),
        "passed":error<tolerance and free_residual<1e-7 and force_balance<1e-7 and moment_balance<.03}
    metadata={"case":"MacNeal-Harder hemisphere with 18-degree hole","schema":"tensorfem.hemisphere-post.v1",
        "dof_order":["ux","uy","uz","rx","ry","rz"],"coordinate_system":"global Cartesian",
        "units":{"length":length_unit,"force":force_unit,"rotation":"radian",
                 "young_modulus":f"{force_unit}/{length_unit}^2"},
        "mesh":{"nodes":n,"elements":len(result.elements),"meridional_elements":result.meridional_elements,
                "circumferential_elements":result.circumferential_elements}}
    return HemispherePost(result.nodes,result.elements,translation,q[:,3:],result.nodes+translation,
        constrained[:,:3],constrained[:,3:],applied,probe,path,extrema,metadata,validation)


def _plain(post):
    return {"metadata":post.metadata,"validation":post.validation,"probe":post.probe,
        "extrema":post.extrema,"path":post.path,"nodes":post.nodes.tolist(),
        "elements":post.elements.tolist(),"displacement":post.displacement.tolist(),
        "rotation":post.rotation.tolist(),"deformed":post.deformed.tolist(),
        "reaction_force":post.reaction_force.tolist(),"reaction_moment":post.reaction_moment.tolist(),
        "applied_force":post.applied_force.tolist()}


def write_hemisphere_json(path,post):
    Path(path).write_text(json.dumps(_plain(post),indent=2,sort_keys=True)+"\n",encoding="utf-8")


def read_hemisphere_json(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("metadata",{}).get("schema")!="tensorfem.hemisphere-post.v1":
        raise ValueError("invalid hemisphere result schema")
    return payload


def write_hemisphere_vtk(path,post):
    db=ModelDB(nodes={i+1:tuple(x) for i,x in enumerate(post.nodes.tolist())},
        elements=[ElementBlock("CPS4",list(range(1,len(post.elements)+1)),
            [[int(i)+1 for i in e] for e in post.elements.tolist()],"SHELL")])
    write_vtk(path,db,point_data={"displacement":post.displacement,"deformed_coordinates":post.deformed,
        "reaction_force":post.reaction_force,"reaction_moment":post.reaction_moment,
        "displacement_magnitude":torch.linalg.vector_norm(post.displacement,dim=1)})


def write_hemisphere_markdown(path,post):
    v=post.validation;m=post.metadata["mesh"]
    text=("# Hemisphere post-processing report\n\n"
          f"- Mesh: {m['meridional_elements']} x {m['circumferential_elements']} "
          f"({m['nodes']} nodes, {m['elements']} elements)\n"
          f"- Probe displacement: {v['computed_displacement']:.9g}\n"
          f"- Published reference: {v['reference_displacement']:.9g}\n"
          f"- Relative error: {100*v['relative_error']:.4f}% (limit {100*v['error_tolerance']:.1f}%)\n"
          f"- Free residual norm: {v['free_residual_norm']:.3e}\n"
          f"- Normalized force balance error: {v['normalized_force_balance_error']:.3e}\n"
          f"- Normalized moment balance error: {v['normalized_moment_balance_error']:.3e}\n"
          f"- Result: **{'PASS' if v['passed'] else 'FAIL'}**\n")
    Path(path).write_text(text,encoding="utf-8")
