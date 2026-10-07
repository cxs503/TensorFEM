"""Low-order frictionless surface quadrature coupling and self-contact search.

Unlike node-tributary contact, contact tractions are integrated at slave facet
quadrature points and consistently distributed to both surfaces.  This is a
small-sliding penalty mortar foundation, not a dual-Lagrange multiplier or
frictional mortar implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch

from .contact3d import project_point_to_facets


@dataclass(frozen=True)
class MortarContactResult:
    slave_forces: torch.Tensor
    master_forces: torch.Tensor
    active_quadrature_points: int
    integrated_area: torch.Tensor
    maximum_penetration: torch.Tensor
    penalty_energy: torch.Tensor


@dataclass(frozen=True)
class MortarContactAssembly:
    """Energy-consistent two-sided contact contribution.

    ``residual`` is ordered as all slave degrees of freedom followed by all
    master degrees of freedom.  It is the gradient of the discrete penalty
    potential; ``tangent`` is its exact active-set linearisation.  Physical
    nodal contact forces are therefore ``-residual``.
    """
    residual: torch.Tensor
    tangent: torch.Tensor
    slave_forces: torch.Tensor
    master_forces: torch.Tensor
    active_quadrature_points: int
    integrated_area: torch.Tensor
    maximum_penetration: torch.Tensor
    penalty_energy: torch.Tensor


def _quadrature(x: torch.Tensor):
    """Yield `(point, shape, differential_area)` for TRI3 or QUAD4."""
    if len(x)==3:
        cross=torch.linalg.cross(x[1]-x[0],x[2]-x[0]); area=.5*torch.linalg.vector_norm(cross)
        if bool(area<=torch.finfo(x.dtype).eps): raise ValueError("degenerate slave triangle")
        for N in ((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3)):
            shape=x.new_tensor(N); yield shape@x,shape,area/3
    elif len(x)==4:
        g=1/math.sqrt(3)
        for xi in (-g,g):
            for eta in (-g,g):
                N=x.new_tensor([(1-xi)*(1-eta),(1+xi)*(1-eta),
                                (1+xi)*(1+eta),(1-xi)*(1+eta)])/4
                dx_dxi=x.new_tensor([-(1-eta),(1-eta),(1+eta),-(1+eta)])/4@x
                dx_deta=x.new_tensor([-(1-xi),-(1+xi),(1+xi),(1-xi)])/4@x
                da=torch.linalg.vector_norm(torch.linalg.cross(dx_dxi,dx_deta))
                if bool(da<=torch.finfo(x.dtype).eps): raise ValueError("degenerate slave quadrilateral")
                yield N@x,N,da
    else: raise ValueError("slave facets must be TRI3 or QUAD4")


def integrate_mortar_contact(slave_vertices: torch.Tensor,slave_faces: torch.Tensor,
                             master_vertices: torch.Tensor,master_faces: torch.Tensor,
                             *,normal_penalty: float) -> MortarContactResult:
    """Integrate frictionless penalty traction over the slave surface."""
    if normal_penalty<=0: raise ValueError("normal_penalty must be positive")
    if slave_faces.ndim!=2 or slave_faces.shape[1] not in (3,4):
        raise ValueError("invalid slave connectivity")
    sf=torch.zeros_like(slave_vertices); mf=torch.zeros_like(master_vertices)
    area=slave_vertices.new_zeros(()); maxpen=area.clone(); energy=area.clone(); active=0
    for face in slave_faces:
        x=slave_vertices[face]
        for point,N,da in _quadrature(x):
            projection=project_point_to_facets(point,master_vertices,master_faces)
            penetration=torch.clamp(-projection.gap,min=0.)
            area += da; maxpen=torch.maximum(maxpen,penetration)
            if bool(penetration<=torch.finfo(point.dtype).eps): continue
            active += 1; traction=normal_penalty*penetration*projection.normal
            sf[face] += N[:,None]*traction*da
            mf[master_faces[projection.face]] -= projection.weights[:,None]*traction*da
            energy += .5*normal_penalty*penetration**2*da
    return MortarContactResult(sf,mf,active,area,maxpen,energy)


def _mortar_potential(slave_reference: torch.Tensor, slave_vertices: torch.Tensor, slave_faces: torch.Tensor,
                      master_vertices: torch.Tensor, master_faces: torch.Tensor,
                      normal_penalty: float) -> torch.Tensor:
    """Discrete current-configuration penalty potential.

    Candidate selection and the open/closed decision form the active set and
    are intentionally held fixed during one linearisation.  Within that set,
    projection coordinates and normals remain in the autograd graph, yielding
    the geometric as well as material contact terms.  Integration uses the
    slave reference-area measure (a total-Lagrangian surface convention).
    """
    energy=slave_vertices.new_zeros(())
    for face in slave_faces:
        # A reference-area measure avoids spurious membrane traction from the
        # variation of an otherwise purely normal penalty potential.
        for _reference_point,shape,da in _quadrature(slave_reference[face]):
            point=shape@slave_vertices[face]
            projection=project_point_to_facets(point,master_vertices,master_faces)
            penetration=torch.clamp(-projection.gap,min=0.)
            energy=energy+.5*normal_penalty*penetration**2*da
    return energy


def assemble_mortar_contact(
    slave_reference: torch.Tensor, slave_faces: torch.Tensor,
    master_reference: torch.Tensor, master_faces: torch.Tensor,
    slave_displacement: torch.Tensor, master_displacement: torch.Tensor,
    *, normal_penalty: float, tangent: bool = True,
) -> MortarContactAssembly:
    """Assemble a frictionless facet-quadrature contact residual and tangent.

    Both surfaces are deformable: current coordinates, projection, normal and
    area are functions of both displacement fields.  The implementation is a
    primal penalty/active-set formulation, not a dual mortar multiplier method.
    It is intended as a compact, consistent kernel for solid surface coupling.
    """
    if normal_penalty<=0: raise ValueError("normal_penalty must be positive")
    if slave_reference.shape!=slave_displacement.shape or master_reference.shape!=master_displacement.shape:
        raise ValueError("reference coordinates and displacements must match")
    if slave_reference.ndim!=2 or slave_reference.shape[1]!=3 or master_reference.ndim!=2 or master_reference.shape[1]!=3:
        raise ValueError("surface coordinates must have shape (n,3)")
    if slave_faces.ndim!=2 or slave_faces.shape[1] not in (3,4) or slave_faces.dtype!=torch.long:
        raise ValueError("invalid slave connectivity")
    if master_faces.ndim!=2 or master_faces.shape[1] not in (3,4) or master_faces.dtype!=torch.long:
        raise ValueError("invalid master connectivity")
    ns=slave_reference.numel()
    q=torch.cat((slave_displacement.reshape(-1),master_displacement.reshape(-1))).detach().requires_grad_(True)

    def potential(dofs: torch.Tensor) -> torch.Tensor:
        slave=slave_reference+dofs[:ns].reshape_as(slave_reference)
        master=master_reference+dofs[ns:].reshape_as(master_reference)
        return _mortar_potential(slave_reference,slave,slave_faces,master,master_faces,normal_penalty)

    energy=potential(q)
    residual=torch.autograd.grad(energy,q,create_graph=tangent)[0]
    if tangent:
        matrix=torch.autograd.functional.hessian(potential,q)
    else:
        matrix=q.new_zeros((q.numel(),q.numel()))
    current_slave=slave_reference+slave_displacement
    current_master=master_reference+master_displacement
    diagnostic=integrate_mortar_contact(current_slave,slave_faces,current_master,master_faces,
                                         normal_penalty=normal_penalty)
    physical=-residual.detach()
    return MortarContactAssembly(
        residual.detach(),matrix.detach(),physical[:ns].reshape_as(slave_reference),
        physical[ns:].reshape_as(master_reference),diagnostic.active_quadrature_points,
        diagnostic.integrated_area,diagnostic.maximum_penetration,energy.detach())


def self_contact_candidates(vertices: torch.Tensor,faces: torch.Tensor,*,search_distance: float):
    """Return nonadjacent facet pairs whose axis-aligned boxes are nearby.

    Facets sharing any node are excluded, which removes same-face, edge and
    vertex neighbours from a self-contact search. Degenerate facets fail closed.
    """
    if search_distance<0 or faces.ndim!=2 or faces.shape[1] not in (3,4):
        raise ValueError("invalid search input")
    boxes=[]; node_sets=[]
    for face in faces:
        x=vertices[face]
        # The first three nodes must span area; QUAD4 is additionally checked
        # by its second triangle to reject collapsed folds.
        triangles=((0,1,2),) if len(face)==3 else ((0,1,2),(0,2,3))
        for ids in triangles:
            if bool(torch.linalg.vector_norm(torch.linalg.cross(x[ids[1]]-x[ids[0]],x[ids[2]]-x[ids[0]]))
                    <=torch.finfo(vertices.dtype).eps):
                raise ValueError("degenerate self-contact facet")
        boxes.append((x.min(0).values,x.max(0).values)); node_sets.append(set(face.tolist()))
    pairs=[]
    for i in range(len(faces)):
        for j in range(i+1,len(faces)):
            if node_sets[i]&node_sets[j]: continue
            lo=torch.maximum(boxes[i][0],boxes[j][0]); hi=torch.minimum(boxes[i][1],boxes[j][1])
            separation=torch.linalg.vector_norm(torch.clamp(lo-hi,min=0.))
            if bool(separation<=search_distance): pairs.append((i,j))
    return tuple(pairs)
