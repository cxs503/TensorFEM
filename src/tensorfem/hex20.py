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

# --------------------------------------------------------------------------
# Inertia operator (consistent / lumped mass).  Until this block existed HEX20
# had stiffness only: ``grep -n mass src/tensorfem/hex20.py`` was empty, so the
# workhorse quadratic brick could not be used for vibration, transient or
# explicit-dynamics work.  The four public entry points below mirror the TET10
# mass API (:mod:`tensorfem.quadratic_solid`) so the two quadratic solids share
# one calling convention.
# --------------------------------------------------------------------------

# One-dimensional Gauss-Legendre nodes/weights (weights carry the [-1, 1]
# measure of one direction; the tensor product below therefore integrates over
# the reference volume 8).  Closed-form values keep the library core torch-only.
_GAUSS_1D = {
    1: ((0.0,), (2.0,)),
    2: ((-math.sqrt(1.0 / 3.0), math.sqrt(1.0 / 3.0)), (1.0, 1.0)),
    3: (
        (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0)),
        (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0),
    ),
    4: (
        (
            -math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
            -math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
            math.sqrt((3.0 - 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
            math.sqrt((3.0 + 2.0 * math.sqrt(6.0 / 5.0)) / 7.0),
        ),
        (
            (18.0 - math.sqrt(30.0)) / 36.0,
            (18.0 + math.sqrt(30.0)) / 36.0,
            (18.0 + math.sqrt(30.0)) / 36.0,
            (18.0 - math.sqrt(30.0)) / 36.0,
        ),
    ),
    5: (
        (
            -math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
            -math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
            0.0,
            math.sqrt(5.0 - 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
            math.sqrt(5.0 + 2.0 * math.sqrt(10.0 / 7.0)) / 3.0,
        ),
        (
            (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
            (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
            128.0 / 225.0,
            (322.0 + 13.0 * math.sqrt(70.0)) / 900.0,
            (322.0 - 13.0 * math.sqrt(70.0)) / 900.0,
        ),
    ),
}


def hex_gauss_rule(order: int = 3, *, dtype=torch.float64, device=None):
    """Tensor-product Gauss-Legendre rule on the reference cube ``[-1,1]^3``.

    Returns ``(points[q,3], weights[q])``.  Order ``n`` integrates exactly any
    polynomial of degree ``2n-1`` *per natural coordinate*; the weights carry the
    reference volume only (their sum is 8), which is the convention
    :func:`hex20_mass` expects from a user-supplied ``rule=``.
    """
    try:
        nodes, weights = _GAUSS_1D[int(order)]
    except KeyError:
        raise ValueError("quadrature order must be one of 1, 2, 3, 4, 5") from None
    points, wts = [], []
    for r, wr in zip(nodes, weights):
        for s, ws in zip(nodes, weights):
            for t, wt in zip(nodes, weights):
                points.append((r, s, t))
                wts.append(wr * ws * wt)
    return (
        torch.tensor(points, dtype=dtype, device=device),
        torch.tensor(wts, dtype=dtype, device=device),
    )


def _hex20_rule(rule, x: torch.Tensor):
    """Normalise a ``rule`` argument into ``(points, weights)`` on ``x``'s dtype."""
    if rule is None:
        return hex_gauss_rule(3, dtype=x.dtype, device=x.device)
    points, weights = rule
    points = torch.as_tensor(points).to(dtype=x.dtype, device=x.device)
    weights = torch.as_tensor(weights).to(dtype=x.dtype, device=x.device)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("rule points must have shape [q,3]")
    if weights.shape != (points.shape[0],):
        raise ValueError("rule weights must have shape [q]")
    return points, weights


def _hex20_density(density, x: torch.Tensor):
    ne = len(x)
    if isinstance(density, torch.Tensor):
        return density.expand(ne).to(x)
    return x.new_full((ne,), float(density))


def _hex20_scalar_mass(x: torch.Tensor, density, *, rule=None):
    """Nodal scalar mass ``rho * int N_i N_j detJ`` of shape ``[e,20,20]``."""
    rho = _hex20_density(density, x)
    points, weights = _hex20_rule(rule, x)
    values, dets = [], []
    for r, s, t in points.tolist():
        n, dn = hex20_shape_gradients(r, s, t, like=x)
        values.append(n)
        dets.append(_gradient(x, dn)[1])
    n = torch.stack(values)
    det = torch.stack(dets, 1)
    return torch.einsum("qi,eq,q,qj->eij", n, det * rho[:, None], weights, n)


def hex20_mass(x: torch.Tensor, density, *, rule=None):
    """Consistent mass matrix of batched HEX20 elements, layout ``[e,60,60]``.

    ``x`` is ``[e,20,3]`` and ``density`` a scalar or one value per element.  The
    DOF ordering equals :func:`solve_hex20` (``3*node + component``), so the
    assembly pattern carries over verbatim.

    The default rule (order-3 Gauss, 27 points) is **exact** whenever
    ``N_i N_j detJ`` is at most degree 5 per natural coordinate, which covers the
    affine (parallelepiped) and straight-planar-face HEX20 shapes; the residual
    for curved geometry is quantified in
    ``docs/hex20-consistent-mass.md``.  ``rule=`` accepts any
    ``(points, weights)`` pair on ``[-1,1]^3`` with weights carrying the
    reference volume, so a deliberately under-integrated rule (order 2, degree 3)
    can be measured instead of assumed.
    """
    m = _hex20_scalar_mass(x, density, rule=rule)
    eye = torch.eye(3, dtype=x.dtype, device=x.device)
    return torch.einsum("eij,ab->eiajb", m, eye).reshape(len(x), 60, 60)


def hex20_lumped_mass(x: torch.Tensor, density, *, scheme: str = "hrz", rule=None):
    """Diagonal (lumped) mass per DOF, shape ``[e,60]``, for explicit dynamics.

    ``scheme="hrz"`` (default) scales the consistent diagonal so the element
    total mass is preserved (Hinton-Rock-Zienkiewicz); every entry stays positive.

    ``scheme="row"`` is the naive row sum ``sum_j M_ij``.  It preserves the total
    mass but is **unusable** for the serendipity brick: the vertex shape
    functions have ``int N_vertex dV = -rho V / 8`` (while ``int N_edge dV =
    +rho V / 6``), so the eight corner DOFs receive *negative* masses and the
    lumped operator is indefinite.  The scheme is kept so the trap stays
    measurable; explicit time integration must use ``"hrz"``.
    """
    m = _hex20_scalar_mass(x, density, rule=rule)
    if scheme == "row":
        nodal = m.sum(-1)
    elif scheme == "hrz":
        diag = torch.diagonal(m, dim1=-2, dim2=-1)
        if torch.any(diag <= 0):
            raise ValueError("HRZ lumping requires a positive consistent diagonal")
        nodal = diag * (m.sum((-2, -1)) / diag.sum(-1))[:, None]
    else:
        raise ValueError("scheme must be 'hrz' or 'row'")
    return nodal.repeat_interleave(3, dim=-1)


def hex20_mass_totals(x: torch.Tensor, density, *, rule=None):
    """Element mass ``rho * int detJ`` per element (the isoparametric volume)."""
    rho = _hex20_density(density, x)
    points, weights = _hex20_rule(rule, x)
    dets = []
    for r, s, t in points.tolist():
        _, dn = hex20_shape_gradients(r, s, t, like=x)
        dets.append(_gradient(x, dn)[1])
    return rho * (torch.stack(dets, 1) * weights).sum(-1)


def assemble_hex20_mass(nodes: torch.Tensor, elements: torch.Tensor, density,
                        *, rule=None, lumping: str | None = None):
    """Assemble the global HEX20 mass matrix ``[n_dofs, n_dofs]``.

    ``lumping=None`` assembles the consistent matrix; ``lumping="hrz"``/``"row"``
    assembles the diagonal matrix built from :func:`hex20_lumped_mass`.
    """
    elements = elements.long()
    x = nodes[elements]
    ne = len(elements)
    ed = torch.stack(tuple(3 * elements + i for i in range(3)), 2).reshape(ne, 60).long()
    n_dofs = 3 * len(nodes)
    if lumping is None:
        me = hex20_mass(x, density, rule=rule)
        rows = ed[:, :, None].expand(-1, -1, 60).reshape(-1)
        cols = ed[:, None, :].expand(-1, 60, -1).reshape(-1)
        m = nodes.new_zeros((n_dofs, n_dofs))
        return m.index_put((rows, cols), me.reshape(-1), accumulate=True)
    values = hex20_lumped_mass(x, density, scheme=lumping, rule=rule).reshape(-1)
    flat = nodes.new_zeros(n_dofs * n_dofs)
    return flat.index_add(0, ed.reshape(-1) * (n_dofs + 1), values).reshape(n_dofs, n_dofs)


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

def assemble_hex20_stiffness_sparse(m: Hex20Model):
    """Assemble the global HEX20 stiffness as a ``scipy`` CSC matrix.

    The dense path (:func:`solve_hex20`) allocates an ``n_dofs x n_dofs`` array, which
    caps a practical model near ~5e3 dof.  Cross-section refinement studies need more,
    so the element stiffnesses are scattered into COO triplets and the duplicate
    entries are summed on conversion instead of into a dense buffer.
    """
    import numpy as np
    import scipy.sparse as sp

    ne = len(m.elements)
    expand = lambda a: a.to(m.nodes).reshape(-1).expand(ne)
    d = elasticity_matrix_3d(expand(m.young_modulus), expand(m.poisson_ratio))
    elements = m.elements.long()
    ke, b = hex20_stiffness(m.nodes[elements], d)
    ed = torch.stack(tuple(3 * elements + i for i in range(3)), 2).reshape(ne, 60).long()
    rows = ed[:, :, None].expand(-1, -1, 60).reshape(-1)
    cols = ed[:, None, :].expand(-1, 60, -1).reshape(-1)
    k = sp.coo_matrix(
        (ke.reshape(-1).detach().cpu().numpy().astype(np.float64),
         (rows.detach().cpu().numpy(), cols.detach().cpu().numpy())),
        shape=(m.n_dofs, m.n_dofs),
    ).tocsc()
    return k, b, ed


def solve_hex20_sparse(m: Hex20Model):
    """Solve the same boundary-value problem as :func:`solve_hex20`, sparsely.

    Identical discretisation, load vector and Dirichlet elimination; only the assembly
    and the linear solver differ (SuperLU on CSC instead of a dense LAPACK factorisation).
    Returns the same :class:`SolidResult`, so results are directly comparable with the
    dense path.
    """
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    k, b, ed = assemble_hex20_stiffness_sparse(m)
    fixed = m.fixed_dofs.to(dtype=torch.long).cpu().numpy()
    if m.prescribed_values is None:
        vals = np.zeros(len(fixed), dtype=np.float64)
    else:
        vals = m.prescribed_values.detach().cpu().numpy().astype(np.float64)
    n = m.n_dofs
    mask = np.ones(n, dtype=bool)
    mask[fixed] = False
    free = np.arange(n)[mask]
    f = m.forces.detach().cpu().numpy().astype(np.float64)
    u = np.zeros(n, dtype=np.float64)
    u[fixed] = vals
    if free.size:
        kff = k[free][:, free].tocsc()
        rhs = f[free] - k[free][:, fixed] @ vals
        u[free] = spla.splu(kff).solve(rhs)
    u_t = m.nodes.new_tensor(u)
    strain = torch.einsum("eij,ej->ei", b, u_t[ed])
    expand = lambda a: a.to(m.nodes).reshape(-1).expand(len(m.elements))
    d = elasticity_matrix_3d(expand(m.young_modulus), expand(m.poisson_ratio))
    stress = torch.einsum("eij,ei->ej", d, strain)
    reaction = m.nodes.new_tensor(k @ u) - m.forces.to(m.nodes)
    energy = m.nodes.new_tensor(0.5 * float(u @ (k @ u)))
    # ``stiffness`` is deliberately left as ``None``: materialising the dense operator
    # is exactly what this path exists to avoid.  Use
    # :func:`assemble_hex20_stiffness_sparse` when the operator itself is needed.
    return SolidResult(u_t, reaction, strain, stress, energy, None)


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
