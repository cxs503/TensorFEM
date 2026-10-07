"""Differentiable TET4 and fully-integrated HEX8 linear elastic solids."""
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class SolidModel:
    nodes: torch.Tensor
    elements: torch.Tensor
    young_modulus: torch.Tensor
    poisson_ratio: torch.Tensor
    forces: torch.Tensor
    fixed_dofs: torch.Tensor
    element_type: str = "hex8"
    prescribed_values: torch.Tensor | None = None
    def __post_init__(self):
        nen={"tet4":4,"hex8":8}.get(self.element_type.lower())
        if nen is None: raise ValueError("element_type must be 'tet4' or 'hex8'")
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3: raise ValueError("nodes must have shape [n,3]")
        if self.elements.ndim != 2 or self.elements.shape[1] != nen: raise ValueError("invalid connectivity")
        if self.forces.shape != (3*len(self.nodes),): raise ValueError("forces must have 3*n entries")
        if self.prescribed_values is not None and self.prescribed_values.numel()!=self.fixed_dofs.numel():
            raise ValueError("prescribed_values must match fixed_dofs")
    @property
    def n_dofs(self): return 3*len(self.nodes)

@dataclass(frozen=True)
class SolidResult:
    displacement: torch.Tensor
    reaction: torch.Tensor
    strain: torch.Tensor
    stress: torch.Tensor
    strain_energy: torch.Tensor
    stiffness: torch.Tensor

def elasticity_matrix_3d(e, nu):
    if torch.any((nu<=-1)|(nu>=.5)): raise ValueError("Poisson ratio must lie in (-1, 0.5)")
    lam=e*nu/((1+nu)*(1-2*nu)); mu=e/(2*(1+nu)); ne=e.numel()
    d=torch.zeros((ne,6,6),dtype=e.dtype,device=e.device)
    d[:,:3,:3]=lam[:,None,None].expand(-1,3,3)+torch.diag_embed((2*mu)[:,None].expand(-1,3))
    d[:,3,3]=mu; d[:,4,4]=mu; d[:,5,5]=mu
    return d

def _b(g):
    ne,nn,_=g.shape; b=torch.zeros((ne,6,3*nn),dtype=g.dtype,device=g.device)
    b[:,0,0::3]=g[:,:,0]; b[:,1,1::3]=g[:,:,1]; b[:,2,2::3]=g[:,:,2]
    b[:,3,0::3]=g[:,:,1]; b[:,3,1::3]=g[:,:,0]
    b[:,4,1::3]=g[:,:,2]; b[:,4,2::3]=g[:,:,1]
    b[:,5,0::3]=g[:,:,2]; b[:,5,2::3]=g[:,:,0]
    return b

def _gradient(x,dn,name):
    j=torch.einsum("eia,ib->eab",x,dn); det=torch.linalg.det(j)
    if torch.any(det<=torch.finfo(x.dtype).eps): raise ValueError(f"{name} has non-positive or degenerate Jacobian")
    return torch.einsum("ib,ebc->eic",dn,torch.linalg.inv(j)),det

def tet4_stiffness(x,d):
    dn=x.new_tensor([[-1.,-1.,-1.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    g,det=_gradient(x,dn,"TET4 element"); b=_b(g)
    return (b.transpose(1,2)@d@b)*(det/6)[:,None,None],b

_SIGNS=((-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),(-1,-1,1),(1,-1,1),(1,1,1),(-1,1,1))
def _hex_gradient(x,xi,eta,zeta):
    s=x.new_tensor(_SIGNS); a,b,c=s[:,0],s[:,1],s[:,2]
    dn=torch.stack((a*(1+b*eta)*(1+c*zeta),b*(1+a*xi)*(1+c*zeta),c*(1+a*xi)*(1+b*eta)),1)/8
    return _gradient(x,dn,"HEX8 element")

def hex8_stiffness(x,d):
    q=1/3**.5; ke=torch.zeros((len(x),24,24),dtype=x.dtype,device=x.device)
    for xi in (-q,q):
      for eta in (-q,q):
       for zeta in (-q,q):
        g,det=_hex_gradient(x,xi,eta,zeta); b=_b(g); ke=ke+(b.transpose(1,2)@d@b)*det[:,None,None]
    return ke,_b(_hex_gradient(x,0.,0.,0.)[0])

def solve_solid(m):
    ne,nn=m.elements.shape
    expand=lambda a:a.to(dtype=m.nodes.dtype,device=m.nodes.device).reshape(-1).expand(ne)
    d=elasticity_matrix_3d(expand(m.young_modulus),expand(m.poisson_ratio)); x=m.nodes[m.elements.long()]
    ke,b=(tet4_stiffness if m.element_type.lower()=="tet4" else hex8_stiffness)(x,d)
    ed=torch.stack(tuple(3*m.elements+i for i in range(3)),2).reshape(ne,3*nn).long()
    rows=ed[:,:,None].expand(-1,-1,3*nn).reshape(-1); cols=ed[:,None,:].expand(-1,3*nn,-1).reshape(-1)
    k=torch.zeros((m.n_dofs,m.n_dofs),dtype=m.nodes.dtype,device=m.nodes.device)
    k=k.index_put((rows,cols),ke.reshape(-1),accumulate=True)
    fixed=m.fixed_dofs.to(device=m.nodes.device,dtype=torch.long)
    vals=torch.zeros_like(fixed,dtype=m.nodes.dtype) if m.prescribed_values is None else m.prescribed_values.to(m.nodes)
    mask=torch.ones(m.n_dofs,dtype=torch.bool,device=m.nodes.device); mask[fixed]=False
    free=torch.arange(m.n_dofs,device=m.nodes.device)[mask]; f=m.forces.to(m.nodes)
    u=torch.zeros(m.n_dofs,dtype=m.nodes.dtype,device=m.nodes.device).index_put((fixed,),vals)
    if free.numel(): u=u.index_put((free,),torch.linalg.solve(k[free][:,free],f[free]-k[free][:,fixed]@vals))
    strain=torch.einsum("eij,ej->ei",b,u[ed]); stress=torch.einsum("eij,ej->ei",d,strain)
    return SolidResult(u,k@u-f,strain,stress,.5*u@k@u,k)

def structured_hex_mesh(l,w,h,nx,ny,nz,*,dtype=torch.float64):
    x=torch.linspace(0,l,nx+1,dtype=dtype); y=torch.linspace(0,w,ny+1,dtype=dtype); z=torch.linspace(0,h,nz+1,dtype=dtype)
    zz,yy,xx=torch.meshgrid(z,y,x,indexing="ij"); nodes=torch.stack((xx.ravel(),yy.ravel(),zz.ravel()),1)
    n=lambda i,j,k:k*(ny+1)*(nx+1)+j*(nx+1)+i; cells=[]
    for k in range(nz):
     for j in range(ny):
      for i in range(nx): cells.append((n(i,j,k),n(i+1,j,k),n(i+1,j+1,k),n(i,j+1,k),n(i,j,k+1),n(i+1,j,k+1),n(i+1,j+1,k+1),n(i,j+1,k+1)))
    return nodes,torch.tensor(cells,dtype=torch.long)

def hex_to_tet_mesh(nodes,hexes):
    p=torch.tensor(((0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6)),device=hexes.device)
    return nodes,hexes[:,p].reshape(-1,4)
