"""Quadratic ten-node tetrahedral (TET10) linear-elastic solid.

Node order is ``(0,1,2,3, 01,12,20,03,13,23)``.  Stiffness uses the
five-point, degree-three Keast tetrahedron rule, so curved isoparametric
geometry is integrated without silently falling back to a linear-tet rule.
"""
from dataclasses import dataclass
import torch

from .solid3d import SolidResult, elasticity_matrix_3d, _b


@dataclass(frozen=True)
class Tet10Model:
    nodes: torch.Tensor
    elements: torch.Tensor
    young_modulus: torch.Tensor
    poisson_ratio: torch.Tensor
    forces: torch.Tensor
    fixed_dofs: torch.Tensor
    prescribed_values: torch.Tensor | None = None

    def __post_init__(self):
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError("nodes must have shape [n,3]")
        if self.elements.ndim != 2 or self.elements.shape[1] != 10:
            raise ValueError("TET10 connectivity must have shape [e,10]")
        if self.forces.shape != (3 * len(self.nodes),):
            raise ValueError("forces must have 3*n entries")
        if self.prescribed_values is not None and self.prescribed_values.numel() != self.fixed_dofs.numel():
            raise ValueError("prescribed_values must match fixed_dofs")

    @property
    def n_dofs(self):
        return 3 * len(self.nodes)


def tet10_shape(rst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return shape values and natural derivatives at points ``[...,3]``."""
    r, s, t = rst.unbind(-1)
    l = torch.stack((1-r-s-t, r, s, t), -1)
    n = torch.stack((
        l[...,0]*(2*l[...,0]-1), l[...,1]*(2*l[...,1]-1),
        l[...,2]*(2*l[...,2]-1), l[...,3]*(2*l[...,3]-1),
        4*l[...,0]*l[...,1], 4*l[...,1]*l[...,2], 4*l[...,2]*l[...,0],
        4*l[...,0]*l[...,3], 4*l[...,1]*l[...,3], 4*l[...,2]*l[...,3]), -1)
    dl = rst.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    dn = torch.stack(tuple((4*l[...,i]-1)[...,None]*dl[i] for i in range(4)), -2)
    edges=((0,1),(1,2),(2,0),(0,3),(1,3),(2,3))
    de = torch.stack(tuple(4*(l[...,i,None]*dl[j]+l[...,j,None]*dl[i]) for i,j in edges), -2)
    return n, torch.cat((dn,de), -2)


def tet10_quadrature(*, dtype=torch.float64, device=None):
    """Degree-three five-point Keast rule; weights include reference volume."""
    bary = ((.25,.25,.25,.25),(.5,1/6,1/6,1/6),(1/6,.5,1/6,1/6),
            (1/6,1/6,.5,1/6),(1/6,1/6,1/6,.5))
    points=torch.tensor([(q[1],q[2],q[3]) for q in bary],dtype=dtype,device=device)
    weights=torch.tensor((-2/15,3/40,3/40,3/40,3/40),dtype=dtype,device=device)
    return points,weights


def tet10_stiffness(x: torch.Tensor, d: torch.Tensor):
    """Element stiffness and centroid strain matrix for batched coordinates."""
    ne=len(x); ke=x.new_zeros((ne,30,30)); points,weights=tet10_quadrature(dtype=x.dtype,device=x.device)
    bc=None
    for p,w in zip(points,weights):
        _,dn=tet10_shape(p)
        j=torch.einsum("eia,ib->eab",x,dn)
        det=torch.linalg.det(j)
        if torch.any(det <= 100*torch.finfo(x.dtype).eps):
            raise ValueError("TET10 has non-positive or degenerate Jacobian at an integration point")
        g=torch.einsum("ib,ebc->eic",dn,torch.linalg.inv(j)); b=_b(g)
        ke=ke+(b.transpose(1,2)@d@b)*(w*det)[:,None,None]
    _,dn=tet10_shape(x.new_tensor([.25,.25,.25])); j=torch.einsum("eia,ib->eab",x,dn)
    if torch.any(torch.linalg.det(j)<=0): raise ValueError("TET10 has non-positive centroid Jacobian")
    bc=_b(torch.einsum("ib,ebc->eic",dn,torch.linalg.inv(j)))
    return ke,bc


def solve_tet10(m: Tet10Model) -> SolidResult:
    ne=len(m.elements); expand=lambda a:a.to(m.nodes).reshape(-1).expand(ne)
    d=elasticity_matrix_3d(expand(m.young_modulus),expand(m.poisson_ratio))
    ke,b=tet10_stiffness(m.nodes[m.elements.long()],d)
    ed=torch.stack(tuple(3*m.elements+i for i in range(3)),2).reshape(ne,30).long()
    rows=ed[:,:,None].expand(-1,-1,30).reshape(-1); cols=ed[:,None,:].expand(-1,30,-1).reshape(-1)
    k=m.nodes.new_zeros((m.n_dofs,m.n_dofs)).index_put((rows,cols),ke.reshape(-1),accumulate=True)
    fixed=m.fixed_dofs.to(device=m.nodes.device,dtype=torch.long)
    vals=m.nodes.new_zeros(len(fixed)) if m.prescribed_values is None else m.prescribed_values.to(m.nodes)
    mask=torch.ones(m.n_dofs,dtype=torch.bool,device=m.nodes.device); mask[fixed]=False
    free=torch.arange(m.n_dofs,device=m.nodes.device)[mask]; f=m.forces.to(m.nodes)
    u=m.nodes.new_zeros(m.n_dofs).index_put((fixed,),vals)
    if free.numel(): u=u.index_put((free,),torch.linalg.solve(k[free][:,free],f[free]-k[free][:,fixed]@vals))
    strain=torch.einsum("eij,ej->ei",b,u[ed]); stress=torch.einsum("eij,ej->ei",d,strain)
    return SolidResult(u,k@u-f,strain,stress,.5*u@k@u,k)


def tet4_to_tet10(nodes: torch.Tensor, tet4: torch.Tensor):
    """Upgrade a conforming TET4 mesh, sharing each global midside node."""
    edge_order=((0,1),(1,2),(2,0),(0,3),(1,3),(2,3)); new=[p for p in nodes]; edge_nodes={}; cells=[]
    for cell in tet4.tolist():
        mids=[]
        for a,b in edge_order:
            key=tuple(sorted((cell[a],cell[b])))
            if key not in edge_nodes:
                edge_nodes[key]=len(new); new.append((nodes[key[0]]+nodes[key[1]])/2)
            mids.append(edge_nodes[key])
        cells.append(cell+mids)
    return torch.stack(new),torch.tensor(cells,dtype=torch.long,device=tet4.device)


def cantilever_tet10_benchmark(nx: int = 10):
    """Slender 3-D cantilever under uniform end shear, with Timoshenko reference.

    The one-element cross-section is intentionally severe: convergence is driven
    by the quadratic interpolation rather than transverse mesh refinement.
    """
    from .solid3d import structured_hex_mesh, hex_to_tet_mesh
    L=w=h=1.; L=10.; E=1e7; nu=.3; load=-1.
    nodes,hexes=structured_hex_mesh(L,w,h,nx,1,1,dtype=torch.float64)
    _,tet4=hex_to_tet_mesh(nodes,hexes); nodes,cells=tet4_to_tet10(nodes,tet4)
    forces=nodes.new_zeros(3*len(nodes)); boundary={}
    for c in cells[:,:4].tolist():
        for loc in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
            face=tuple(sorted(c[i] for i in loc)); boundary[face]=boundary.get(face,0)+1
    for face,count in boundary.items():
        if count==1 and all(abs(float(nodes[i,0])-L)<1e-12 for i in face):
            p=nodes[list(face)]; area=torch.linalg.norm(torch.linalg.cross(p[1]-p[0],p[2]-p[0]))/2
            for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):
                mid=(nodes[a]+nodes[b])/2; mid_id=int(torch.argmin(torch.linalg.norm(nodes-mid,dim=1)))
                forces[3*mid_id+1] += load*area/3
    fixed=torch.nonzero((nodes[:,0].abs()<1e-12)[:,None].expand(-1,3).reshape(-1)).flatten()
    result=solve_tet10(Tet10Model(nodes,cells,torch.tensor(E),torch.tensor(nu),forces,fixed))
    target=nodes.new_tensor([L,w/2,h/2]); tip_id=int(torch.argmin(torch.linalg.norm(nodes-target,dim=1)))
    computed=-result.displacement.reshape(-1,3)[tip_id,1]
    I=w*h**3/12; G=E/(2*(1+nu)); reference=abs(load)*L**3/(3*E*I)+abs(load)*L/((5/6)*G*w*h)
    return computed,computed.new_tensor(reference)


def cantilever_tet4_benchmark(nx: int = 10):
    """Matching linear-tetrahedron A/B baseline for the TET10 benchmark."""
    from .solid3d import SolidModel, structured_hex_mesh, hex_to_tet_mesh, solve_solid
    L=10.; E=1e7; nu=.3
    nodes,hexes=structured_hex_mesh(L,1.,1.,nx,1,1,dtype=torch.float64); _,cells=hex_to_tet_mesh(nodes,hexes)
    forces=nodes.new_zeros(3*len(nodes)); boundary={}
    for c in cells.tolist():
        for loc in ((0,1,2),(0,1,3),(0,2,3),(1,2,3)):
            face=tuple(sorted(c[i] for i in loc)); boundary[face]=boundary.get(face,0)+1
    for face,count in boundary.items():
        if count==1 and all(abs(float(nodes[i,0])-L)<1e-12 for i in face):
            p=nodes[list(face)]; area=torch.linalg.norm(torch.linalg.cross(p[1]-p[0],p[2]-p[0]))/2
            for i in face: forces[3*i+1] -= area/3
    fixed=torch.nonzero((nodes[:,0].abs()<1e-12)[:,None].expand(-1,3).reshape(-1)).flatten()
    result=solve_solid(SolidModel(nodes,cells,torch.tensor(E),torch.tensor(nu),forces,fixed,"tet4"))
    tip_nodes=torch.nonzero(abs(nodes[:,0]-L)<1e-12).flatten()
    computed=-result.displacement.reshape(-1,3)[tip_nodes,1].mean()
    reference=10.**3/(3*E*(1/12))+10./((5/6)*(E/(2*(1+nu))))
    return computed,computed.new_tensor(reference)
