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
