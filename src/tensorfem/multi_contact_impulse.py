"""Deterministic small-scale PGS solver for simultaneous contact impulses."""
from __future__ import annotations
from dataclasses import dataclass
import torch

from .contact3d import _closest_triangle


@dataclass(frozen=True)
class ContactConstraint:
    key: tuple
    side_a: torch.Tensor
    weights_a: torch.Tensor
    side_b: torch.Tensor
    weights_b: torch.Tensor
    normal: torch.Tensor


@dataclass(frozen=True)
class ContactImpulseHistory:
    key: tuple
    normal_impulse: torch.Tensor
    friction_impulse: torch.Tensor


@dataclass(frozen=True)
class MultiContactState:
    histories: tuple[ContactImpulseHistory,...]


@dataclass(frozen=True)
class MultiContactResult:
    velocities: torch.Tensor
    impulses: torch.Tensor
    state: MultiContactState
    iterations: int
    complementarity_residual: torch.Tensor
    cone_residual: torch.Tensor
    kinetic_before: torch.Tensor
    kinetic_after: torch.Tensor
    converged: bool


def _effective(c,masses):
    value=torch.sum(c.weights_a**2/masses[c.side_a])
    if len(c.side_b): value+=torch.sum(c.weights_b**2/masses[c.side_b])
    return value


def _relative(c,v):
    value=c.weights_a@v[c.side_a]
    if len(c.side_b): value=value-c.weights_b@v[c.side_b]
    return value


def _apply(c,v,masses,impulse):
    v[c.side_a]+=c.weights_a[:,None]*impulse/masses[c.side_a,None]
    if len(c.side_b): v[c.side_b]-=c.weights_b[:,None]*impulse/masses[c.side_b,None]


def solve_multi_contact_impulses(velocities,masses,constraints,state,*,friction: float,
                                 restitution=0.,tolerance=1e-10,max_iterations=200):
    """Solve simultaneous normal complementarity and Coulomb disk constraints."""
    if friction<0 or not 0<=restitution<=1 or tolerance<=0 or max_iterations<1:
        raise ValueError("invalid solver parameter")
    if masses.shape!=(len(velocities),) or bool(torch.any(masses<=0)): raise ValueError("invalid nodal masses")
    ordered=tuple(sorted(constraints,key=lambda c:c.key))
    if len({c.key for c in ordered})!=len(ordered): raise ValueError("duplicate contact key")
    old={h.key:h for h in state.histories}; v=velocities.clone(); impulses=torch.zeros_like(velocities)
    initial_normal={c.key:torch.dot(_relative(c,velocities),c.normal) for c in ordered}
    current={}
    # Warm-start impulses are trial data; input state remains untouched.
    for c in ordered:
        h=old.get(c.key); z=velocities.new_zeros(())
        jn=z if h is None else torch.clamp(h.normal_impulse,min=0.)
        jt=torch.zeros(3,dtype=v.dtype,device=v.device) if h is None else h.friction_impulse.clone()
        norm=torch.linalg.vector_norm(jt); limit=friction*jn
        if bool(norm>limit) and bool(norm>0): jt*=limit/norm
        _apply(c,v,masses,jn*c.normal+jt); current[c.key]=[jn,jt]
    converged=False
    for iteration in range(1,max_iterations+1):
        maximum=0.
        for c in ordered:
            eff=_effective(c,masses); jn,jt=current[c.key]
            rel=_relative(c,v); vn=torch.dot(rel,c.normal)
            target=-restitution*min(float(initial_normal[c.key]),0.)
            new_jn=torch.clamp(jn-(vn-target)/eff,min=0.); djn=new_jn-jn
            _apply(c,v,masses,djn*c.normal)
            rel=_relative(c,v); vt=rel-torch.dot(rel,c.normal)*c.normal
            trial=jt-vt/eff; mag=torch.linalg.vector_norm(trial); limit=friction*new_jn
            new_jt=trial if bool(mag<=limit) or bool(mag==0) else trial*(limit/mag)
            _apply(c,v,masses,new_jt-jt); current[c.key]=[new_jn,new_jt]
            maximum=max(maximum,float(torch.abs(djn)),float(torch.linalg.vector_norm(new_jt-jt)))
        if maximum<=tolerance:
            converged=True; break
    histories=[]; comp=v.new_zeros(()); cone=comp.clone()
    for c in ordered:
        jn,jt=current[c.key]; target=-restitution*min(float(initial_normal[c.key]),0.)
        w=torch.dot(_relative(c,v),c.normal)-target
        violation=torch.maximum(torch.clamp(-w,min=0.),torch.abs(jn*torch.clamp(w,min=0.)))
        comp=torch.maximum(comp,violation)
        cone=torch.maximum(cone,torch.clamp(torch.linalg.vector_norm(jt)-friction*jn,min=0.))
        histories.append(ContactImpulseHistory(c.key,jn,jt))
        _apply(c,impulses,masses,jn*c.normal+jt) # masses cancel below
    impulses*=masses[:,None]
    kb=.5*torch.sum(masses[:,None]*velocities**2); ka=.5*torch.sum(masses[:,None]*v**2)
    if bool(ka>kb*(1+1e-10)+1e-12): raise RuntimeError("multi-contact impact increased kinetic energy")
    return MultiContactResult(v,impulses,MultiContactState(tuple(histories)),iteration,
                              comp,cone,kb,ka,converged)


def build_vertex_face_manifold(vertices,slave_nodes,master_vertices,master_face,*,tolerance: float):
    """Build persistent vertex/edge-endpoint constraints near one TRI3 face."""
    if tolerance<0 or len(master_face)!=3: raise ValueError("invalid manifold input")
    tri=master_vertices[master_face]
    cross=torch.linalg.cross(tri[1]-tri[0],tri[2]-tri[0]); norm=torch.linalg.vector_norm(cross)
    if bool(norm<=torch.finfo(vertices.dtype).eps): raise ValueError("degenerate master face")
    normal=cross/norm; out=[]
    for node in sorted(set(slave_nodes.tolist())):
        q,bary=_closest_triangle(vertices[node],tri)
        if bool(torch.linalg.vector_norm(vertices[node]-q)<=tolerance):
            key=("vf",int(node),*master_face.tolist())
            out.append(ContactConstraint(key,torch.tensor([node],dtype=torch.long,device=vertices.device),
                vertices.new_ones(1),master_face,bary,normal))
    return tuple(out)


def constraints_from_unified_events(events):
    """Convert one earliest-TOI event set into deterministic PGS constraints."""
    if not events:return ()
    first=min(e.toi for e in events); out=[]
    for event in events:
        if abs(event.toi-first)>1e-8: continue
        key=(event.kind,*event.side_a.tolist(),-1,*event.side_b.tolist())
        out.append(ContactConstraint(key,event.side_a,event.weights_a,event.side_b,
                                     event.weights_b,event.normal))
    ordered=tuple(sorted(out,key=lambda c:c.key))
    if len({c.key for c in ordered})!=len(ordered): raise ValueError("duplicate unified event")
    return ordered
