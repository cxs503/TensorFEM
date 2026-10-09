"""Discontinuous ice shells, prescribed cracks, buoyancy and fragment diagnostics.

This module describes artificial small-strain ice plates. It provides explicit
initial geometry/material fields and conservative forces rather than a
calibrated compressive/rate-dependent sea-ice constitutive model.
"""
from __future__ import annotations

from dataclasses import dataclass,fields
import math
import torch
from .suboff_ice_collision import IceImpactConfig

D=torch.float64


@dataclass(frozen=True)
class IceSimulationConfig(IceImpactConfig):
    appendage_thickness: float = .003
    ice_boundary: str = 'clamped'
    ice_model: str = 'cohesive'
    ice_thickness_map: tuple = ()
    open_water_cells: tuple = ()
    precracked_pairs: tuple = ()
    hydrostatic_buoyancy: bool = True
    water_density: float = 1025.
    gravity: float = 9.81
    water_added_mass_per_area: float = 0.
    water_drag_per_area: float = 0.
    contact_expectation: str = 'required'

    def validate(self):
        IceImpactConfig(**{f.name:getattr(self,f.name) for f in fields(IceImpactConfig)}).validate()
        if self.ice_boundary not in ('clamped','simply_supported','free'):
            raise ValueError('ice_boundary must be clamped, simply_supported or free')
        if self.ice_model not in ('elastic','cohesive'):
            raise ValueError('ice_model must be elastic or cohesive')
        if self.contact_expectation not in ('required','absent','any'):
            raise ValueError('invalid contact_expectation')
        if not isinstance(self.hydrostatic_buoyancy,bool):
            raise ValueError('hydrostatic_buoyancy must be a boolean')
        for name in ('appendage_thickness','water_density','gravity','water_added_mass_per_area','water_drag_per_area'):
            value=getattr(self,name)
            if not math.isfinite(value) or value<0 or (name not in ('water_added_mass_per_area','water_drag_per_area') and value==0):
                raise ValueError(f'invalid {name}')
        if self.hydrostatic_buoyancy and self.water_density<=self.ice_density:
            raise ValueError('floating-ice equilibrium requires water density greater than ice density')
        n=self.ice_nx*self.ice_ny
        if self.ice_thickness_map:
            h=torch.tensor(self.ice_thickness_map,dtype=D).reshape(-1)
            if h.numel()!=n or not bool(torch.isfinite(h).all()) or bool((h<0).any()):
                raise ValueError('ice_thickness_map requires nx*ny finite nonnegative SI values')
        for cell in self.open_water_cells:
            if isinstance(cell,bool) or not isinstance(cell,int) or not 0<=cell<n:
                raise ValueError('invalid open-water grid cell ID')
        for pair in self.precracked_pairs:
            if len(pair)!=2 or any(isinstance(k,bool) or not isinstance(k,int) or not 0<=k<n for k in pair):
                raise ValueError('invalid precrack grid-cell pair')
            a,b=pair
            if abs(a//self.ice_nx-b//self.ice_nx)+abs(a%self.ice_nx-b%self.ice_nx)!=1:
                raise ValueError('precrack must join neighbouring grid cells')
        if self.ice_model=='elastic' and self.precracked_pairs:
            raise ValueError('precracks require cohesive ice')


def build_ice_sheet(length,top_elevation,c):
    c.validate()
    nx,ny=c.ice_nx,c.ice_ny
    h=(torch.tensor(c.ice_thickness_map,dtype=D).reshape(-1) if c.ice_thickness_map else
       torch.full((nx*ny,),c.ice_thickness,dtype=D))
    h=h.clone()
    if c.open_water_cells:h[list(c.open_water_cells)]=0
    # A single waterline gives each thickness its Archimedean freeboard.
    # top_elevation is the upper surface of the thickest active tile.
    hmax=float(h.max()) if bool((h>0).any()) else c.ice_thickness
    waterline=top_elevation-(1-c.ice_density/c.water_density)*hmax
    x0=-c.ice_margin;y0=-c.ice_width/2
    dx=(length+2*c.ice_margin)/nx;dy=c.ice_width/ny
    nodes=[];cells=[];active=[];thickness=[];cell_lookup={}
    for grid_id in range(nx*ny):
        if h[grid_id]<=0:continue
        j,i=divmod(grid_id,nx);x=x0+i*dx;y=y0+j*dy
        z=(waterline+(.5-c.ice_density/c.water_density)*float(h[grid_id])
           if c.hydrostatic_buoyancy else top_elevation-float(h[grid_id])/2)
        k=len(nodes);cell_lookup[grid_id]=len(cells)
        nodes.extend(((x,y,z),(x+dx,y,z),(x+dx,y+dy,z),(x,y+dy,z)))
        cells.append((k,k+1,k+2,k+3));active.append(grid_id);thickness.append(float(h[grid_id]))
    active=torch.tensor(active,dtype=torch.long)
    ice_nodes=torch.tensor(nodes,dtype=D).reshape(-1,3)
    ice_cells=torch.tensor(cells,dtype=torch.long).reshape(-1,4)
    node_area=torch.full((len(nodes),),dx*dy/4,dtype=D)
    node_h=torch.tensor(thickness,dtype=D).repeat_interleave(4)
    edge=((torch.isclose(ice_nodes[:,0],ice_nodes.new_tensor(x0)))|
          torch.isclose(ice_nodes[:,0],ice_nodes.new_tensor(x0+nx*dx))|
          torch.isclose(ice_nodes[:,1],ice_nodes.new_tensor(y0))|
          torch.isclose(ice_nodes[:,1],ice_nodes.new_tensor(y0+ny*dy)))
    if c.ice_boundary=='clamped':
        fixed=(6*torch.nonzero(edge).flatten()[:,None]+torch.arange(6)).reshape(-1)
    elif c.ice_boundary=='simply_supported':
        # w=0 only: rotations and all in-plane rigid modes remain free.
        fixed=6*torch.nonzero(edge).flatten()+2
    else:fixed=torch.empty(0,dtype=torch.long)
    precracks={tuple(sorted(p)) for p in c.precracked_pairs}
    for pair in precracks:
        if any(k not in cell_lookup for k in pair):
            raise ValueError('precrack touches an open-water cell')
    ids=[];Bs=[];normal=[];area=[];seam_ids=[];seam_pairs=[];pre=[]
    for grid_id in active.tolist():
        j,i=divmod(grid_id,nx);a=cell_lookup[grid_id];el=ice_cells[a]
        neighbours=[]
        if i+1<nx and grid_id+1 in cell_lookup:
            b=cell_lookup[grid_id+1];other=ice_cells[b]
            neighbours.append((grid_id+1,b,(el[1],el[2]),(other[0],other[3]),(1.,0.,0.),dy))
        if j+1<ny and grid_id+nx in cell_lookup:
            b=cell_lookup[grid_id+nx];other=ice_cells[b]
            neighbours.append((grid_id+nx,b,(el[3],el[2]),(other[0],other[1]),(0.,1.,0.),dx))
        for neighbour,b,aa,bb,n,edge_length in neighbours:
            sid=len(seam_pairs);seam_pairs.append((a,b))
            overlap=min(thickness[a],thickness[b]);pair=tuple(sorted((grid_id,neighbour)))
            top=min(float(ice_nodes[int(el[0]),2])+thickness[a]/2,
                    float(ice_nodes[int(ice_cells[b,0]),2])+thickness[b]/2)
            for na,nb in zip(aa,bb):
                for zeta in (-1/math.sqrt(12),1/math.sqrt(12)):
                    z=top-overlap/2+overlap*zeta
                    T=[]
                    for node in (int(na),int(nb)):
                        offset=z-float(ice_nodes[node,2])
                        M=torch.zeros((3,6),dtype=D);M[:,:3]=torch.eye(3,dtype=D)
                        M[0,4]=offset;M[1,3]=-offset;T.append(M)
                    ids.append([6*int(na)+k for k in range(6)]+[6*int(nb)+k for k in range(6)])
                    Bs.append(torch.cat((-T[0],T[1]),dim=1))
                    normal.append(n);area.append(overlap*edge_length/4);seam_ids.append(sid);pre.append(pair in precracks)
    return dict(nodes=ice_nodes,elements=ice_cells,active_grid_ids=active,
                cell_lookup=cell_lookup,thickness=torch.tensor(thickness,dtype=D),
                node_area=node_area,node_thickness=node_h,grid_thickness=h,
                grid=(x0,y0,dx,dy),waterline_m=waterline if c.hydrostatic_buoyancy else None,fixed_dofs=fixed,
                bond_ids=torch.tensor(ids,dtype=torch.long).reshape(-1,12),
                bond_B=torch.stack(Bs) if Bs else torch.empty((0,3,12),dtype=D),
                bond_normals=torch.tensor(normal,dtype=D).reshape(-1,3),
                bond_areas=torch.tensor(area,dtype=D),seam_ids=torch.tensor(seam_ids,dtype=torch.long),
                seam_pairs=torch.tensor(seam_pairs,dtype=torch.long).reshape(-1,2),
                precracked_points=torch.tensor(pre,dtype=torch.bool))


def buoyancy_response(displacement,areas,thickness,c):
    """Gravity minus hydrostatic buoyancy relative to floating equilibrium.

    Immersion is clipped to [0,h]. A fully emerged fragment experiences its
    weight, and a submerged fragment experiences buoyancy minus weight. The
    returned potential is continuous, conservative, zero at equilibrium.
    """
    if not c.hydrostatic_buoyancy:
        return torch.zeros_like(displacement),displacement.new_zeros(())
    k=c.water_density*c.gravity*areas
    high=c.ice_density/c.water_density*thickness
    low=high-thickness
    clipped=torch.maximum(low,torch.minimum(high,displacement))
    force=k*clipped
    potential=(.5*k*clipped**2+k*clipped*(displacement-clipped)).sum()
    return force,potential


def ice_fragments(sheet,damage,nodal_mass,nodal_velocity,nodal_position):
    """Connectivity uses all thickness points, retaining mass after fracture."""
    count=len(sheet['elements']);adj=[set() for _ in range(count)];severed=[]
    for sid,(a,b) in enumerate(sheet['seam_pairs'].tolist()):
        broken=bool((damage[sheet['seam_ids']==sid]>=.999).all())
        if broken:severed.append(sid)
        else:adj[a].add(b);adj[b].add(a)
    unseen=set(range(count));pieces=[]
    while unseen:
        first=min(unseen);unseen.remove(first);stack=[first];cells=[first]
        while stack:
            for b in adj[stack.pop()]:
                if b in unseen:unseen.remove(b);stack.append(b);cells.append(b)
        nodes=sheet['elements'][cells].reshape(-1)
        m=nodal_mass[nodes];total=m.sum()
        v=nodal_velocity[nodes];x=nodal_position[nodes]
        pieces.append({'grid_cells':sheet['active_grid_ids'][cells].tolist(),
                       'mass_kg':float(total),'center_m':((m[:,None]*x).sum(0)/total).tolist(),
                       'mean_velocity_m_s':((m[:,None]*v).sum(0)/total).tolist(),
                       'translational_kinetic_energy_J':float(.5*(m[:,None]*v*v).sum())})
    return {'component_count':len(pieces),'fully_severed_seams':severed,
            'total_retained_mass_kg':sum(p['mass_kg'] for p in pieces),'fragments':pieces}
