"""Twenty-node serendipity brick (HEX20) with 3x3x3 integration.

Node order is the eight conventional HEX8 corners followed by edge midpoints
01, 12, 23, 30, 45, 56, 67, 74, 04, 15, 26 and 37.
"""
from dataclasses import dataclass
import math
import torch

from .solid3d import SolidResult, elasticity_matrix_3d, _b

_CORNERS=((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
          (-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
_EDGES=((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7))

@dataclass(frozen=True)
class Hex20Model:
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
        if self.elements.ndim != 2 or self.elements.shape[1] != 20:
            raise ValueError("HEX20 connectivity must have shape [ne,20]")
        if self.forces.shape != (3*len(self.nodes),):
            raise ValueError("forces must have 3*n entries")
        if self.prescribed_values is not None and self.prescribed_values.numel()!=self.fixed_dofs.numel():
            raise ValueError("prescribed_values must match fixed_dofs")
    @property
    def n_dofs(self): return 3*len(self.nodes)

def hex20_shape_gradients(r, s, t, *, like: torch.Tensor):
    """Return shape values and natural derivatives [20,3]."""
    r=torch.as_tensor(r,dtype=like.dtype,device=like.device); s=torch.as_tensor(s,dtype=like.dtype,device=like.device); t=torch.as_tensor(t,dtype=like.dtype,device=like.device)
    ns=[]; ds=[]
    for a,b,c in _CORNERS:
        A=1+a*r; B=1+b*s; C=1+c*t; L=a*r+b*s+c*t-2
        ns.append(A*B*C*L/8)
        ds.append(torch.stack((a*B*C*(L+A)/8,
                               b*A*C*(L+B)/8,
                               c*A*B*(L+C)/8)))
    for i,j in _EDGES:
        a,b,c=_CORNERS[i]; aa,bb,cc=_CORNERS[j]
        if a != aa:
            ns.append((1-r*r)*(1+b*s)*(1+c*t)/4)
            ds.append(torch.stack((-r*(1+b*s)*(1+c*t)/2,
                b*(1-r*r)*(1+c*t)/4,c*(1-r*r)*(1+b*s)/4)))
        elif b != bb:
            ns.append((1-s*s)*(1+a*r)*(1+c*t)/4)
            ds.append(torch.stack((a*(1-s*s)*(1+c*t)/4,
                -s*(1+a*r)*(1+c*t)/2,c*(1-s*s)*(1+a*r)/4)))
        else:
            ns.append((1-t*t)*(1+a*r)*(1+b*s)/4)
            ds.append(torch.stack((a*(1-t*t)*(1+b*s)/4,
                b*(1-t*t)*(1+a*r)/4,-t*(1+a*r)*(1+b*s)/2)))
    return torch.stack(ns),torch.stack(ds)

def _gradient(x, dn):
    j=torch.einsum("eia,ib->eab",x,dn)
    det=torch.linalg.det(j)
    scale=torch.linalg.matrix_norm(j,ord=2,dim=(-2,-1))
    threshold=64*torch.finfo(x.dtype).eps*torch.clamp(scale,min=1)**3
    if torch.any(det <= threshold):
        raise ValueError("HEX20 element has non-positive or degenerate Jacobian")
    return torch.einsum("ib,ebc->eic",dn,torch.linalg.inv(j)),det

def hex20_stiffness(x, d):
    """Element stiffness using exact-order 3x3x3 Gauss integration."""
    q=math.sqrt(3/5); points=(-q,0.,q); weights=(5/9,8/9,5/9)
    ke=x.new_zeros((len(x),60,60))
    for i,r in enumerate(points):
      for j,s in enumerate(points):
       for k,t in enumerate(points):
        _,dn=hex20_shape_gradients(r,s,t,like=x)
        g,det=_gradient(x,dn); b=_b(g)
        ke += (b.transpose(1,2)@d@b)*(det*weights[i]*weights[j]*weights[k])[:,None,None]
    _,dn=hex20_shape_gradients(0.,0.,0.,like=x)
    return ke,_b(_gradient(x,dn)[0])

def solve_hex20(m: Hex20Model):
    ne=len(m.elements)
    expand=lambda a:a.to(m.nodes).reshape(-1).expand(ne)
    d=elasticity_matrix_3d(expand(m.young_modulus),expand(m.poisson_ratio))
    ke,b=hex20_stiffness(m.nodes[m.elements.long()],d)
    ed=torch.stack(tuple(3*m.elements+i for i in range(3)),2).reshape(ne,60).long()
    rows=ed[:,:,None].expand(-1,-1,60).reshape(-1); cols=ed[:,None,:].expand(-1,60,-1).reshape(-1)
    K=m.nodes.new_zeros((m.n_dofs,m.n_dofs)).index_put((rows,cols),ke.reshape(-1),accumulate=True)
    fixed=m.fixed_dofs.to(device=m.nodes.device,dtype=torch.long)
    vals=m.nodes.new_zeros(len(fixed)) if m.prescribed_values is None else m.prescribed_values.to(m.nodes)
    mask=torch.ones(m.n_dofs,dtype=torch.bool,device=m.nodes.device); mask[fixed]=False
    free=torch.arange(m.n_dofs,device=m.nodes.device)[mask]; f=m.forces.to(m.nodes)
    u=m.nodes.new_zeros(m.n_dofs).index_put((fixed,),vals)
    if len(free): u=u.index_put((free,),torch.linalg.solve(K[free][:,free],f[free]-K[free][:,fixed]@vals))
    strain=torch.einsum("eij,ej->ei",b,u[ed]); stress=torch.einsum("eij,ej->ei",d,strain)
    return SolidResult(u,K@u-f,strain,stress,.5*u@K@u,K)

def structured_hex20_mesh(l,w,h,nx,ny,nz,*,dtype=torch.float64):
    """Conforming structured HEX20 mesh, sharing all edge nodes."""
    coords={}; nodes=[]
    def node(key):
        if key not in coords:
            coords[key]=len(nodes)
            nodes.append((key[0]*l/(2*nx),key[1]*w/(2*ny),key[2]*h/(2*nz)))
        return coords[key]
    cells=[]
    for k in range(nz):
      for j in range(ny):
       for i in range(nx):
        corners=((2*i,2*j,2*k),(2*i+2,2*j,2*k),(2*i+2,2*j+2,2*k),(2*i,2*j+2,2*k),
                 (2*i,2*j,2*k+2),(2*i+2,2*j,2*k+2),(2*i+2,2*j+2,2*k+2),(2*i,2*j+2,2*k+2))
        ids=[node(p) for p in corners]
        ids += [node(tuple((corners[a][d]+corners[b][d])//2 for d in range(3))) for a,b in _EDGES]
        cells.append(ids)
    return torch.tensor(nodes,dtype=dtype),torch.tensor(cells,dtype=torch.long)

def upgrade_hex8_to_hex20(nodes, elements):
    """Upgrade arbitrary conforming HEX8 topology, deduplicating shared edges."""
    out=nodes.tolist(); mids={}; cells=[]
    for cell in elements.tolist():
        row=list(cell)
        for a,b in _EDGES:
            edge=tuple(sorted((cell[a],cell[b])))
            if edge not in mids:
                mids[edge]=len(out); out.append(((nodes[edge[0]]+nodes[edge[1]])/2).tolist())
            row.append(mids[edge])
        cells.append(row)
    return nodes.new_tensor(out),torch.tensor(cells,dtype=torch.long,device=elements.device)

def cantilever_hex20_benchmark(nx: int = 8):
    """Slender 3-D cantilever and independent Timoshenko beam reference."""
    E,nu,L,width,height,load=210e9,.3,10.,1.,1.,-1e6
    nodes,cells=structured_hex20_mesh(L,width,height,nx,1,1)
    forces=nodes.new_zeros(3*len(nodes)); right=torch.where(nodes[:,0]==L)[0]
    corner=(((nodes[right,1]==0)|(nodes[right,1]==width)) &
            ((nodes[right,2]==0)|(nodes[right,2]==height)))
    forces[3*right[corner]+2]=load*(-1/12)
    forces[3*right[~corner]+2]=load*(1/3)
    left=torch.where(nodes[:,0]==0)[0]
    fixed=torch.stack((3*left,3*left+1,3*left+2),1).reshape(-1)
    result=solve_hex20(Hex20Model(nodes,cells,nodes.new_tensor(E),nodes.new_tensor(nu),forces,fixed))
    computed=result.displacement[3*right+2].mean()
    inertia=width*height**3/12; shear=E/(2*(1+nu))
    reference=load*L**3/(3*E*inertia)+load*L/((5/6)*shear*width*height)
    return computed,computed.new_tensor(reference)

def cantilever_hex8_benchmark(nx: int = 8):
    """Matching fully-integrated HEX8 locking baseline for A/B evidence."""
    from .solid3d import SolidModel,solve_solid,structured_hex_mesh
    E,nu,L,width,height,load=210e9,.3,10.,1.,1.,-1e6
    nodes,cells=structured_hex_mesh(L,width,height,nx,1,1)
    forces=nodes.new_zeros(3*len(nodes)); right=torch.where(nodes[:,0]==L)[0]
    forces[3*right+2]=load/len(right)
    left=torch.where(nodes[:,0]==0)[0]
    fixed=torch.stack((3*left,3*left+1,3*left+2),1).reshape(-1)
    result=solve_solid(SolidModel(nodes,cells,nodes.new_tensor(E),nodes.new_tensor(nu),forces,fixed,"hex8"))
    computed=result.displacement[3*right+2].mean()
    inertia=width*height**3/12; shear=E/(2*(1+nu))
    reference=load*L**3/(3*E*inertia)+load*L/((5/6)*shear*width*height)
    return computed,computed.new_tensor(reference)
