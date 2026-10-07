"""Quarter-domain local-patch topology for the true 3-D Hertz P0 path."""
from __future__ import annotations
from dataclasses import dataclass
import torch

@dataclass(frozen=True)
class HertzQuarterPatch:
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    contact_radius: float
    contact_cells: int
    domain_radii: float
    maximum_growth_ratio: float
    minimum_surface_spacing: float

@dataclass(frozen=True)
class HertzQuarterVolume:
    """Structured TET4 volume corresponding to a quarter patch.

    This is deliberately a small, dependency-free volume generator.  It is
    useful for wiring the local patch into a solid/Mortar driver; it does not
    claim that symmetry conditions or a contact solve have been applied.
    """
    nodes: torch.Tensor
    elements: torch.Tensor
    surface_nodes: torch.Tensor
    surface_faces: torch.Tensor

def _axis(core:float,domain:float,core_cells:int,outer_cells:int,growth:float,dtype):
    inner=torch.linspace(0.,core,core_cells+1,dtype=dtype)
    if outer_cells==0: return inner
    weights=torch.tensor([growth**i for i in range(outer_cells)],dtype=dtype)
    widths=(domain-core)*weights/weights.sum()
    return torch.cat((inner,core+torch.cumsum(widths,0)))

def build_hertz_quarter_patch(*,contact_radius:float,contact_cells:int=8,
    domain_radii:float=8.,outer_cells:int=8,depth_cells:int=16,
    growth:float=1.35,dtype=torch.float64):
    """Build auditable local/outer coordinate axes for one symmetric quarter.

    The central square ``[0,a]^2`` is uniform.  Outer and depth layers grow
    geometrically but are rescaled to end exactly at the requested far field.
    """
    if contact_radius<=0 or contact_cells<6 or domain_radii<6:
        raise ValueError("positive radius, >=6 contact cells and >=6a domain required")
    if outer_cells<2 or depth_cells<4 or not 1<growth<=1.5:
        raise ValueError("invalid transition topology")
    a=contact_radius;far=domain_radii*a
    x=_axis(a,far,contact_cells,outer_cells,growth,dtype);y=x.clone()
    # Resolve one contact-zone radius through depth before transitioning.
    z=_axis(a,far,contact_cells,depth_cells-contact_cells,growth,dtype)
    maxima=[]
    for widths in (torch.diff(x),torch.diff(z)):
        ratios=widths[1:]/widths[:-1]
        maxima.append(torch.maximum(ratios,1/ratios).max())
    maximum=float(torch.stack(maxima).max())
    return HertzQuarterPatch(x,y,z,a,contact_cells,domain_radii,maximum,
        float(torch.diff(x).min()))

def quarter_patch_quality(patch:HertzQuarterPatch):
    """Return fail-closed topology metrics before volume meshing."""
    spacing=patch.contact_radius/patch.contact_cells
    return {"contact_cells_per_radius":patch.contact_cells,
        "domain_radii":patch.domain_radii,
        "surface_spacing_relative_to_radius":patch.minimum_surface_spacing/patch.contact_radius,
        "maximum_adjacent_growth_ratio":patch.maximum_growth_ratio,
        "contact_spacing_exact":abs(patch.minimum_surface_spacing/spacing-1)<1e-12,
        "passed":patch.contact_cells>=6 and patch.domain_radii>=6
                 and patch.maximum_growth_ratio<=3.0}

def build_hertz_quarter_volume(*, contact_radius: float, contact_cells: int = 8,
    domain_radii: float = 8., outer_cells: int = 8, depth_cells: int = 16,
    growth: float = 1.35, curvature_radius: float | None = None,
    dtype=torch.float64) -> HertzQuarterVolume:
    """Generate a positive-orientation TET4 quarter volume.

    The top surface is a spherical-paraboloid cap ``z=(x²+y²)/(2R)`` when
    ``curvature_radius`` is supplied; otherwise it is planar.  The returned
    top quads are retained as QUAD4 faces for a Mortar coupling.  Any invalid
    geometry fails closed rather than silently producing a bad contact mesh.
    """
    p = build_hertz_quarter_patch(contact_radius=contact_radius,
        contact_cells=contact_cells, domain_radii=domain_radii,
        outer_cells=outer_cells, depth_cells=depth_cells, growth=growth,
        dtype=dtype)
    nx, ny, nz = len(p.x), len(p.y), len(p.z)
    nodes=[]
    for k,z in enumerate(p.z):
        for i,x in enumerate(p.x):
            for j,y in enumerate(p.y):
                zz=float(z)
                if curvature_radius is not None and k == 0:
                    zz += (float(x)**2 + float(y)**2)/(2*curvature_radius)
                nodes.append((float(x),float(y),zz))
    def nid(k,i,j): return k*nx*ny+i*ny+j
    # Consistent six-way subdivision of each structured brick.
    pattern=((0,1,ny,nx*ny),(1,ny,nx*ny,nx*ny+1),
             (1,ny,nx*ny+1,nx*ny+ny+1),(1,nx*ny,nx*ny+1,nx*ny+ny+1),
             (ny,nx*ny,nx*ny+ny,nx*ny+ny+1),(1,ny,nx*ny+ny+1,nx*ny+1))
    elems=[]
    for k in range(nz-1):
      for i in range(nx-1):
       for j in range(ny-1):
        q=nid(k,i,j); base=(q,q+1,q+ny,q+ny+1,q+nx*ny,q+nx*ny+1,q+nx*ny+ny,q+nx*ny+ny+1)
        # use explicit parity-independent subdivision; orient by determinant below
        for a,b,c,d in pattern:
            vals=(base[0],base[1],base[2],base[4]) if (a,b,c,d)==pattern[0] else None
            # standard six tetrahedra around the body diagonal 0-7
        elems.extend(((base[0],base[1],base[3],base[7]),(base[0],base[3],base[2],base[7]),
                      (base[0],base[2],base[6],base[7]),(base[0],base[6],base[4],base[7]),
                      (base[0],base[4],base[5],base[7]),(base[0],base[5],base[1],base[7])))
    X=torch.tensor(nodes,dtype=dtype); E=torch.tensor(elems,dtype=torch.long)
    vol=torch.linalg.det(X[E[:,1:]]-X[E[:,0:1]])
    bad=vol<0
    if bool(torch.any(vol==0)): raise ValueError("quarter volume contains degenerate TET4")
    if bool(torch.any(bad)):
        # The structured subdivision is valid for either axis convention;
        # normalize orientation explicitly so downstream assembly is safe.
        E[bad,2], E[bad,3] = E[bad,3].clone(), E[bad,2].clone()
    surf=[]
    k=nz-1
    for i in range(nx-1):
      for j in range(ny-1): surf.append((nid(k,i,j),nid(k,i+1,j),nid(k,i+1,j+1),nid(k,i,j+1)))
    return HertzQuarterVolume(X,E,torch.unique(torch.tensor(surf)),torch.tensor(surf,dtype=torch.long))
