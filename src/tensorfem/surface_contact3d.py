"""Node-quadrature surface-to-surface contact and Hertz diagnostics.

Slave-facet tributary areas turn the local traction law in :mod:`contact3d`
into consistent surface resultants.  This is a practical low-order contact
discretisation, not mortar contact.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from .contact3d import Contact3DState, initial_contact_state, update_node_facet_contact


@dataclass(frozen=True)
class SurfaceContactState:
    slave_nodes: torch.Tensor
    local_states: tuple[Contact3DState, ...]


@dataclass(frozen=True)
class SurfaceContactUpdate:
    slave_forces: torch.Tensor
    master_forces: torch.Tensor
    state: SurfaceContactState
    active_nodes: torch.Tensor
    stored_energy: torch.Tensor
    dissipation_increment: torch.Tensor
    maximum_penetration: torch.Tensor


def tributary_areas(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Return vertex-lumped surface areas for TRI3/QUAD4 facets."""
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices must have shape (n,3)")
    if faces.ndim != 2 or faces.shape[1] not in (3,4) or faces.dtype != torch.long:
        raise ValueError("faces must be long TRI3 or QUAD4 connectivity")
    out=vertices.new_zeros(vertices.shape[0])
    splits=((0,1,2),) if faces.shape[1]==3 else ((0,1,2),(0,2,3))
    for face in faces:
        for ids in splits:
            tri=vertices[face[list(ids)]]
            area=.5*torch.linalg.vector_norm(torch.linalg.cross(tri[1]-tri[0],tri[2]-tri[0]))
            if bool(area <= torch.finfo(vertices.dtype).eps):
                raise ValueError("slave surface contains a degenerate triangle")
            out[face[list(ids)]] += area/3
    return out


def initial_surface_contact_state(slave_vertices: torch.Tensor, slave_faces: torch.Tensor,
                                  master_vertices: torch.Tensor,
                                  master_faces: torch.Tensor) -> SurfaceContactState:
    area=tributary_areas(slave_vertices,slave_faces)
    nodes=torch.nonzero(area>0,as_tuple=False).flatten()
    states=tuple(initial_contact_state(slave_vertices[i],master_vertices,master_faces) for i in nodes)
    return SurfaceContactState(nodes,states)


def update_surface_contact(
    slave_vertices: torch.Tensor, slave_faces: torch.Tensor,
    master_vertices: torch.Tensor, master_faces: torch.Tensor,
    state: SurfaceContactState, *, normal_penalty: float,
    tangential_penalty: float, friction: float,
    normal_method: str="penalty", relative_increments: torch.Tensor | None=None,
) -> SurfaceContactUpdate:
    """Integrate contact tractions using slave-node tributary areas."""
    areas=tributary_areas(slave_vertices,slave_faces)
    expected=torch.nonzero(areas>0,as_tuple=False).flatten()
    if not torch.equal(expected,state.slave_nodes) or len(state.local_states)!=len(expected):
        raise ValueError("state topology does not match slave surface")
    if relative_increments is not None and relative_increments.shape != slave_vertices.shape:
        raise ValueError("relative_increments must match slave_vertices")
    sf=torch.zeros_like(slave_vertices); mf=torch.zeros_like(master_vertices)
    new=[]; active=[]; stored=slave_vertices.new_zeros(()); diss=stored.clone(); maxpen=stored.clone()
    for node,old in zip(state.slave_nodes.tolist(),state.local_states):
        inc=None if relative_increments is None else relative_increments[node]
        local=update_node_facet_contact(slave_vertices[node],master_vertices,master_faces,old,
            normal_penalty=normal_penalty,tangential_penalty=tangential_penalty,
            friction=friction,normal_method=normal_method,relative_increment=inc)
        weight=areas[node]
        sf[node]=local.slave_force*weight; mf += local.master_forces*weight
        stored += local.stored_energy*weight; diss += local.dissipation_increment*weight
        maxpen=torch.maximum(maxpen,torch.clamp(-local.projection.gap,min=0.))
        active.append(local.state.active); new.append(local.state)
    return SurfaceContactUpdate(sf,mf,SurfaceContactState(expected,tuple(new)),
        torch.tensor(active,dtype=torch.bool,device=slave_vertices.device),stored,diss,maxpen)


@dataclass(frozen=True)
class HertzReference:
    contact_radius: float
    peak_pressure: float
    effective_modulus: float


def hertz_sphere_halfspace_reference(load: float, radius: float,
                                     young1: float, poisson1: float,
                                     young2: float, poisson2: float) -> HertzReference:
    """Classical frictionless elastic sphere/half-space Hertz reference."""
    if min(load,radius,young1,young2)<=0 or not (-1<poisson1<.5 and -1<poisson2<.5):
        raise ValueError("invalid Hertz parameters")
    effective=1/((1-poisson1**2)/young1+(1-poisson2**2)/young2)
    a=(3*load*radius/(4*effective))**(1/3)
    return HertzReference(a,3*load/(2*math.pi*a*a),effective)


def integrate_hertz_pressure(reference: HertzReference, *, radial_cells: int=64) -> float:
    """Midpoint-ring quadrature of ``p0 sqrt(1-(r/a)^2)``."""
    if radial_cells < 1: raise ValueError("radial_cells must be positive")
    dr=reference.contact_radius/radial_cells
    result=0.
    for i in range(radial_cells):
        r=(i+.5)*dr
        p=reference.peak_pressure*math.sqrt(max(0.,1-(r/reference.contact_radius)**2))
        result += p*2*math.pi*r*dr
    return result


def winkler_sphere_load(indentation: float, radius: float, penalty: float) -> float:
    """Rigid sphere on independent penalty springs; an explicit non-Hertz model."""
    if indentation<0 or radius<=0 or penalty<=0: raise ValueError("invalid Winkler parameters")
    return math.pi*penalty*radius*indentation**2
