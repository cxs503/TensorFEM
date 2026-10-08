"""Experimental small-strain SUBOFF shell impact against cohesive ice tiles.

This is a dry, initial upward-impact model. Ice is an artificial elastic solid
with a prescribed grid of breakable interfaces, not a calibrated sea-ice law.
All fragments keep their mass and contact after fracture. No element erosion,
mass scaling, imposed hull displacement or graphical crack generation is used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import torch

from .cohesive import BilinearCohesiveLaw, CohesiveState
from .plate import _shape
from .spherical_shell import projected_shell4_stiffness
from .suboff_geometry import audit_closed_quads, inward_midsurface

D = torch.float64


@dataclass(frozen=True)
class IceImpactConfig:
    hull_thickness: float = .003
    hull_young: float = 210e9
    hull_poisson: float = .3
    hull_density: float = 7850.
    hull_total_mass: float = 700.
    hull_yield_reference: float = 355e6
    ice_thickness: float = .01
    ice_young: float = 5e9
    ice_poisson: float = .3
    ice_density: float = 900.
    ice_strength: float = .5e6
    ice_fracture_energy: float = 5.
    ice_nx: int = 12
    ice_ny: int = 4
    ice_margin: float = .3
    ice_width: float = 1.2
    cohesive_factor: float = 5.
    contact_factor: float = .2
    upward_speed: float = .3
    initial_gap: float = .00005
    duration: float = .006
    time_safety: float = .55
    sample_count: int = 240

    def validate(self):
        values=asdict(self)
        for key,value in values.items():
            if not math.isfinite(value):
                raise ValueError(f'{key} must be finite')
            if key not in ('hull_poisson','ice_poisson','initial_gap') and value <= 0:
                raise ValueError(f'{key} must be positive')
        if min(self.hull_poisson,self.ice_poisson) <= -1 or max(self.hull_poisson,self.ice_poisson) >= .5:
            raise ValueError('invalid Poisson ratio')
        if self.initial_gap < 0 or not 0 < self.time_safety <= .9:
            raise ValueError('invalid gap/time safety')
        if self.ice_nx < 2 or self.ice_ny < 2 or self.sample_count < 2:
            raise ValueError('mesh/sample counts must be at least two')
        if any(not isinstance(v,int) for v in (self.ice_nx,self.ice_ny,self.sample_count)):
            raise ValueError('mesh/sample counts must be integers')


def _ids(elements):
    return (6*elements[:,:,None]+torch.arange(6)).reshape(len(elements),24)


def _basis(x):
    normal=torch.linalg.cross(x[2]-x[0],x[3]-x[1],dim=0)
    normal/=torch.linalg.vector_norm(normal)
    e1=x[1]-x[0];e1-=torch.dot(e1,normal)*normal
    e1/=torch.linalg.vector_norm(e1)
    e2=torch.linalg.cross(normal,e1,dim=0)
    basis=torch.stack((e1,e2,normal))
    xy=(x-x.mean(0))@basis[:2].T
    return xy,basis


def _area(x):
    xy,_=_basis(x);a=0.
    for xi,eta in ((-3**-.5,-3**-.5),(3**-.5,-3**-.5),
                   (3**-.5,3**-.5),(-3**-.5,3**-.5)):
        _,dn=_shape(xy.new_tensor(xi),xy.new_tensor(eta))
        det=torch.linalg.det(xy.T@dn)
        if det <= 0:
            raise ValueError('invalid projected shell Jacobian')
        a+=float(det)
    return a


def _sparse(block_ids,blocks,n):
    ids=torch.cat(block_ids);ke=torch.cat(blocks)
    rows=ids[:,:,None].expand(-1,-1,ids.shape[1]).reshape(-1)
    cols=ids[:,None,:].expand(-1,ids.shape[1],-1).reshape(-1)
    return torch.sparse_coo_tensor(torch.stack((rows,cols)),ke.reshape(-1),(n,n),
                                  check_invariants=True).coalesce()


def conservative_step_bound(stiffness,mass,free):
    """Gershgorin bound for M^-1/2 K M^-1/2 on unconstrained DOFs.

    Absolute entries of the undamaged shell/cohesive/contact upper operator
    bound its spectral radius. Softening tangent may be negative, but secant
    stiffness never exceeds this initial operator. No power-iteration estimate.
    """
    if mass.ndim != 1 or not bool(torch.isfinite(mass).all()) or bool((mass<=0).any()):
        raise ValueError('positive finite diagonal mass required')
    k=stiffness.coalesce();row,col=k.indices();value=k.values()
    mask=free[row]&free[col]
    scaled=value[mask].abs()/torch.sqrt(mass[row[mask]]*mass[col[mask]])
    sums=torch.zeros_like(mass);sums.index_add_(0,row[mask],scaled)
    bound=float(sums.max())
    if not math.isfinite(bound) or bound<=0:
        raise ValueError('invalid stiffness spectral bound')
    return 2/math.sqrt(bound)


def _ice_grid(length,c):
    x0=-c.ice_margin;x1=length+c.ice_margin
    y0=-c.ice_width/2;y1=c.ice_width/2
    dx=(x1-x0)/c.ice_nx;dy=(y1-y0)/c.ice_ny
    nodes=[];cells=[]
    # Independent corners are essential: adjacent tiles can separate physically.
    for j in range(c.ice_ny):
        for i in range(c.ice_nx):
            x=x0+i*dx;y=y0+j*dy;k=len(nodes)
            nodes.extend(((x,y,0.),(x+dx,y,0.),(x+dx,y+dy,0.),(x,y+dy,0.)))
            cells.append((k,k+1,k+2,k+3))
    return torch.tensor(nodes,dtype=D),torch.tensor(cells), (x0,y0,dx,dy)


def _bonds(ice_cells,offset,dx,dy,c):
    pairs=[];normal=[];weights=[];seams=[]
    sid=0
    for j in range(c.ice_ny):
        for i in range(c.ice_nx):
            a=ice_cells[j*c.ice_nx+i]
            neighbours=[]
            if i+1<c.ice_nx:
                b=ice_cells[j*c.ice_nx+i+1]
                neighbours.append(((a[1],a[2]),(b[0],b[3]),(1.,0.,0.),dy))
            if j+1<c.ice_ny:
                b=ice_cells[(j+1)*c.ice_nx+i]
                neighbours.append(((a[3],a[2]),(b[0],b[1]),(0.,1.,0.),dx))
            for left,right,n,edge_length in neighbours:
                for na,nb in zip(left,right):
                    for z in (-c.ice_thickness/math.sqrt(12),c.ice_thickness/math.sqrt(12)):
                        pairs.append((int(na)+offset,int(nb)+offset,z))
                        normal.append(n);weights.append(edge_length*c.ice_thickness/4);seams.append(sid)
                sid+=1
    ids=[];B=[]
    for a,b,z in pairs:
        ids.append([6*a+k for k in range(6)]+[6*b+k for k in range(6)])
        T=torch.zeros((3,6),dtype=D);T[:,:3]=torch.eye(3,dtype=D)
        T[0,4]=z;T[1,3]=-z
        B.append(torch.cat((-T,T),dim=1))
    return torch.tensor(ids),torch.stack(B),torch.tensor(normal,dtype=D),torch.tensor(weights,dtype=D),torch.tensor(seams)


def cohesive_response(q,ids,B,normals,areas,law,history):
    """Equal-strength effective-opening law; compression stays undamaged.

    Horizontal opening and both shear components share one synthetic fracture
    energy. The update is work conjugate and irreversible, including unloading.
    Two endpoints and two thickness points represent each vertical ice seam.
    """
    jump=torch.einsum('pij,pj->pi',B,q[ids])
    gn=(jump*normals).sum(1)
    tensile=jump-torch.minimum(gn,torch.zeros_like(gn))[:,None]*normals
    effective=torch.linalg.vector_norm(tensile,dim=1)
    response=law.update(effective,CohesiveState(history))
    secant=law.stiffness*(1-response.damage)
    compression=torch.minimum(gn,torch.zeros_like(gn))
    traction=secant[:,None]*tensile+law.stiffness*compression[:,None]*normals
    local=torch.einsum('pij,pi->pj',B,traction*areas[:,None])
    force=torch.zeros_like(q);force.index_add_(0,ids.reshape(-1),local.reshape(-1))
    stored=.5*torch.sum(areas*(secant*effective**2+law.stiffness*compression**2))
    maximum=response.state.maximum_opening
    # Integral under bilinear envelope minus stored secant energy at maximum.
    dissipation=law.peak_traction*law.failure_opening/(2*(law.failure_opening-law.onset_opening))*torch.clamp(
        maximum-law.onset_opening,min=0,max=law.failure_opening-law.onset_opening)
    return force,stored,torch.sum(areas*dissipation),response.damage,maximum


def contact_response(q,ids,weights,gaps,penalties):
    """Vertical hull-node / ice-facet contact with exact nodal action-reaction."""
    penetration=torch.clamp((weights*q[ids]).sum(1)-gaps,min=0)
    magnitudes=penalties*penetration
    force=torch.zeros_like(q)
    force.index_add_(0,ids.reshape(-1),(weights*magnitudes[:,None]).reshape(-1))
    return force,.5*torch.dot(magnitudes,penetration),magnitudes,penetration


def build_impact_system(geometry,c=IceImpactConfig()):
    c.validate()
    if geometry.get('schema')!='tensorfem.suboff-outer-geometry/1':
        raise ValueError('unsupported SUBOFF geometry schema')
    outer=torch.tensor(geometry['nodes'],dtype=D)
    hull_cells=torch.tensor(geometry['elements'],dtype=torch.long)
    audit=audit_closed_quads(outer,hull_cells)
    if not audit['passed']:
        raise ValueError('hull must be a closed consistently oriented Q4 manifold')
    hull,normals=inward_midsurface(outer,hull_cells,c.hull_thickness)
    ice,ice_cells,grid=_ice_grid(geometry['length_m'],c)
    if float(outer[:,1].abs().max())>=c.ice_width/2:
        raise ValueError('ice sheet must cover the hull footprint')
    ice[:,2]=float(outer[:,2].max())+c.initial_gap+c.ice_thickness/2
    nh=len(hull);ice_cells_global=ice_cells+nh
    nodes=torch.cat((hull,ice));nd=6*len(nodes)
    mass=torch.zeros((len(nodes),6),dtype=D)
    areas=[];ke=[]
    elements=torch.cat((hull_cells,ice_cells_global))
    hcount=len(hull_cells)
    for ei,el in enumerate(elements):
        E,nu,t,rho=(c.hull_young,c.hull_poisson,c.hull_thickness,c.hull_density) if ei<hcount else (
            c.ice_young,c.ice_poisson,c.ice_thickness,c.ice_density)
        a=_area(nodes[el]);areas.append(a)
        ke.append(projected_shell4_stiffness(nodes[el],E,nu,t))
        m=rho*t*a/4
        mass[el,:3]+=m
        # Physical through-thickness rotary inertia; drilling inertia is explicit.
        mass[el,3:]+=m*t*t/12
    mass=mass.reshape(-1);ids=_ids(elements);ke=torch.stack(ke)
    shell_mass=float(mass[2:6*nh:6].sum())
    if c.hull_total_mass<shell_mass:
        raise ValueError('total hull mass must include at least the physical steel shell mass')
    # Declared equipment/ballast inertia follows the nodal shell-area weights.
    # Rotary inertia through thickness remains the actual steel section inertia.
    for component in range(3):
        mass[component:6*nh:6]*=c.hull_total_mass/shell_mass
    shell=_sparse([ids],[ke],nd)
    bid,B,bnormal,barea,seams=_bonds(ice_cells,nh,grid[2],grid[3],c)
    law=BilinearCohesiveLaw(c.cohesive_factor*c.ice_young/min(grid[2:]),
                          c.ice_strength,c.ice_fracture_energy)
    bk=torch.einsum('pij,pik->pjk',B,B)*(barea*law.stiffness)[:,None,None]
    # Ice outside boundary is clamped, each independent copy included.
    x0,y0,dx,dy=grid;x1=x0+c.ice_nx*dx;y1=y0+c.ice_ny*dy
    edge=(torch.isclose(ice[:,0],ice.new_tensor(x0))|torch.isclose(ice[:,0],ice.new_tensor(x1))|
          torch.isclose(ice[:,1],ice.new_tensor(y0))|torch.isclose(ice[:,1],ice.new_tensor(y1)))
    fixed_nodes=torch.nonzero(edge).flatten()+nh
    fixed=(6*fixed_nodes[:,None]+torch.arange(6)).reshape(-1)
    free=torch.ones(nd,dtype=torch.bool);free[fixed]=False
    # Fixed horizontal registration, appropriate only for predominantly vertical
    # initial impact. No contact with a hidden rigid plane after ice fracture.
    candidates=torch.nonzero(normals[:,2]>.05).flatten()
    cid=[];cw=[];gaps=[];cp=[]
    nodal_area=torch.zeros(nh,dtype=D)
    for k in range(4):
        nodal_area.index_add_(0,hull_cells[:,k],torch.tensor(areas[:hcount],dtype=D)/4)
    for node in candidates.tolist():
        x,y,z=outer[node].tolist()
        i=min(c.ice_nx-1,max(0,int((x-x0)/dx)))
        j=min(c.ice_ny-1,max(0,int((y-y0)/dy)))
        a=(x-(x0+i*dx))/dx;b=(y-(y0+j*dy))/dy
        N=[(1-a)*(1-b),a*(1-b),a*b,(1-a)*b]
        ice_el=ice_cells_global[j*c.ice_nx+i]
        cid.append([6*node+2]+[6*int(v)+2 for v in ice_el])
        cw.append([1.]+[-v for v in N])
        gaps.append(float(ice[0,2])-c.ice_thickness/2-z)
        cp.append(c.contact_factor*c.ice_young/c.ice_thickness*float(nodal_area[node]))
    cid=torch.tensor(cid);cw=torch.tensor(cw,dtype=D)
    cp=torch.tensor(cp,dtype=D);gaps=torch.tensor(gaps,dtype=D)
    ck=cp[:,None,None]*cw[:,:,None]*cw[:,None,:]
    softening_bound=max(1.,law.onset_opening/(law.failure_opening-law.onset_opening))
    # Sum entrywise absolute block operators. Coalescing signed blocks first
    # could hide cancellation when contacts or cohesive damage change.
    upper=(_sparse([ids],[ke.abs()],nd)+_sparse([bid],[bk.abs()*softening_bound],nd)
           +_sparse([cid],[ck.abs()],nd))
    bound=conservative_step_bound(upper,mass,free)
    # Rigid translations must not encounter spurious internal shell forces.
    rigid_error=[]
    for k in range(3):
        rigid=torch.zeros(nd,dtype=D);rigid[k:6*nh:6]=1
        action=torch.sparse.mm(shell,rigid[:,None])[:,0]
        scale=float(shell.values().abs().max())
        rigid_error.append(float(action[:6*nh].abs().max())/scale)
    if max(rigid_error)>1e-10:
        raise ValueError('hull shell fails rigid translation gate')
    stress_operators=[]
    for ei,el in enumerate(elements):
        xy,basis=_basis(nodes[el])
        E,nu,t=(c.hull_young,c.hull_poisson,c.hull_thickness) if ei<hcount else (
            c.ice_young,c.ice_poisson,c.ice_thickness)
        C=E/(1-nu**2)*xy.new_tensor([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])
        operators=[]
        for xi,eta in ((-3**-.5,-3**-.5),(3**-.5,-3**-.5),
                       (3**-.5,3**-.5),(-3**-.5,3**-.5)):
            _,dn=_shape(xy.new_tensor(xi),xy.new_tensor(eta))
            grad=dn@torch.linalg.inv(xy.T@dn)
            bm=torch.zeros((3,24),dtype=D);bb=torch.zeros_like(bm)
            for a in range(4):
                gx,gy=grad[a]
                bm[0,6*a:6*a+3]=gx*basis[0]
                bm[1,6*a:6*a+3]=gy*basis[1]
                bm[2,6*a:6*a+3]=gy*basis[0]+gx*basis[1]
                bb[0,6*a+3:6*a+6]=gx*basis[1]
                bb[1,6*a+3:6*a+6]=-gy*basis[0]
                bb[2,6*a+3:6*a+6]=gy*basis[1]-gx*basis[0]
            operators.append(torch.stack((C@(bm+t/2*bb),C@(bm-t/2*bb))))
        stress_operators.append(torch.stack(operators))
    return dict(nodes=nodes,outer=outer,elements=elements,hull_cells=hull_cells,
                ice_cells=ice_cells_global,nh=nh,hcount=hcount,mass=mass,shell=shell.to_sparse_csr(),
                ids=ids,ke=ke,areas=torch.tensor(areas,dtype=D),fixed=fixed,free=free,
                bid=bid,B=B,bnormal=bnormal,barea=barea,seams=seams,law=law,
                cid=cid,cw=cw,gaps=gaps,cp=cp,step_bound=bound,geometry_audit=audit,
                rigid_translation_errors=rigid_error,config=c,
                stress_operator=torch.stack(stress_operators),steel_shell_mass=shell_mass)


def _stress(system,q):
    stress=torch.einsum('egscd,ed->egsc',system['stress_operator'],q[system['ids']])
    vm=torch.sqrt(stress[:,:,:,0]**2-stress[:,:,:,0]*stress[:,:,:,1]+
                  stress[:,:,:,1]**2+3*stress[:,:,:,2]**2)
    return stress,vm


def simulate_impact(geometry,c=IceImpactConfig(),*,dt=None):
    """Velocity Verlet integration with independently audited energy/momentum."""
    s=build_impact_system(geometry,c);mass=s['mass'];nh=s['nh'];nd=len(mass)
    stable=s['step_bound']
    requested=c.time_safety*stable if dt is None else dt
    if not math.isfinite(requested) or requested<=0 or requested>c.time_safety*stable*(1+1e-12):
        raise ValueError('time step exceeds conservative stability limit')
    steps=math.ceil(c.duration/requested);dt=c.duration/steps
    q=torch.zeros(nd,dtype=D);v=torch.zeros_like(q);v[2:6*nh:6]=c.upward_speed
    maximum=torch.zeros(len(s['bid']),dtype=D)
    initial_energy=float(.5*torch.dot(mass,v*v))
    initial_momentum=float(torch.sum(mass[2::6]*v[2::6]))
    external_impulse=0.;max_momentum_error=0.;max_energy_error=0.
    history=[];snapshots=[];stride=max(1,steps//c.sample_count)
    max_penetration=0.;peak_force=0.;peak_q=q.clone();peak_time=0.
    ever_contact=False;max_rotation=0.;ice_w=0.;hull_vm=0.

    def evaluate(position,old_maximum):
        internal=torch.mv(s['shell'],position)
        shell_energy=.5*torch.dot(position,internal)
        bond,be,diss,damage,new_max=cohesive_response(position,s['bid'],s['B'],s['bnormal'],s['barea'],s['law'],old_maximum)
        contact,ce,forces,pen=contact_response(position,s['cid'],s['cw'],s['gaps'],s['cp'])
        total=internal+bond+contact
        return total,shell_energy,be,diss,damage,new_max,ce,forces,pen

    state=evaluate(q,maximum);maximum=state[5]
    acc=-state[0]/mass;acc[s['fixed']]=0
    for step in range(steps+1):
        internal,se,be,diss,damage,_,ce,forces,pen=state
        kinetic=.5*torch.dot(mass,v*v)
        total=float(kinetic+se+be+ce+diss)
        energy_error=abs(total/initial_energy-1)
        momentum=float(torch.sum(mass[2::6]*v[2::6]))
        momentum_error=abs(momentum-initial_momentum-external_impulse)/abs(initial_momentum)
        max_energy_error=max(max_energy_error,energy_error)
        max_momentum_error=max(max_momentum_error,momentum_error)
        force=float(forces.sum());max_penetration=max(max_penetration,float(pen.max()))
        ever_contact|=force>0
        max_rotation=max(max_rotation,float(q.reshape(-1,6)[:,3:].abs().max()))
        ice_w=max(ice_w,float(q.reshape(-1,6)[nh:,2].abs().max()))
        # Check all time steps and all in-plane Gauss points, both shell skins.
        stress,vm=_stress(s,q)
        hull_vm=max(hull_vm,float(vm[:s['hcount']].max()))
        if force>peak_force:
            peak_force=force;peak_q=q.clone();peak_time=step*dt
        if step%stride==0 or step==steps:
            history.append({'time_s':step*dt,'contact_force_N':force,'kinetic_J':float(kinetic),
                            'shell_strain_J':float(se),'cohesive_stored_J':float(be),
                            'contact_stored_J':float(ce),'fracture_dissipation_J':float(diss),
                            'energy_relative_error':energy_error,'momentum_relative_error':momentum_error,
                            'vertical_momentum_kg_m_s':momentum,'support_vertical_impulse_N_s':external_impulse,
                            'maximum_penetration_m':float(pen.max()),
                            'hull_vertical_velocity_m_s':float(torch.dot(mass[2:6*nh:6],v[2:6*nh:6])/mass[2:6*nh:6].sum()),
                            'maximum_damage':float(damage.max()),'failed_points':int((damage>=.999).sum())})
        if step in {0,steps//4,steps//2,3*steps//4,steps}:
            snapshots.append({'time_s':step*dt,'dofs':q.reshape(-1,6).tolist(),
                              'cohesive_damage':damage.tolist()})
        if step==steps:
            break
        half=v+.5*dt*acc
        q+=dt*half;q[s['fixed']]=0
        next_state=evaluate(q,maximum);maximum=next_state[5]
        next_acc=-next_state[0]/mass;next_acc[s['fixed']]=0
        v=half+.5*dt*next_acc;v[s['fixed']]=0
        # Supports supply the internal force required to keep constrained DOFs zero.
        fixed_z=s['fixed'][s['fixed']%6==2]
        external_impulse+=.5*dt*float(state[0][fixed_z].sum()+next_state[0][fixed_z].sum())
        state=next_state;acc=next_acc
        if not bool(torch.isfinite(q).all()) or not math.isfinite(float(torch.dot(v,v))):
            raise RuntimeError('nonfinite impact state')

    final_stress,final_vm=_stress(s,q);peak_stress,peak_vm=_stress(s,peak_q)
    final_damage=state[4]
    seam_damage=torch.stack([final_damage[s['seams']==i].max() for i in range(int(s['seams'].max())+1)])
    translation=q.reshape(-1,6)[:nh,:3]
    com=(translation*mass.reshape(-1,6)[:nh,:3]).sum(0)/mass.reshape(-1,6)[:nh,:3].sum(0)
    deformation=float(torch.linalg.vector_norm(translation-com,dim=1).max())
    checks={'geometry_closed':s['geometry_audit']['passed'],'contact_occurred':ever_contact,
            'energy_balance':max_energy_error<.03,'vertical_momentum_balance':max_momentum_error<1e-8,
            'penetration_small':max_penetration<c.ice_thickness*.03,
            'cohesive_fracture_occurred':bool((final_damage>=.999).any()),
            'small_rotation':max_rotation<.1,'small_ice_displacement':ice_w<c.ice_thickness*.5,
            'elastic_hull_gauss_stress':hull_vm<c.hull_yield_reference}
    metrics={'peak_contact_force_N':peak_force,'peak_contact_time_s':peak_time,
             'maximum_penetration_m':max_penetration,'maximum_energy_relative_error':max_energy_error,
             'maximum_vertical_momentum_relative_error':max_momentum_error,
             'fracture_dissipation_J':float(state[3]),'fully_failed_cohesive_points':int((final_damage>=.999).sum()),
             'damaged_seams':int((seam_damage>0).sum()),'maximum_ice_vertical_displacement_m':ice_w,
             'hull_deformation_without_translation_m':deformation,'maximum_rotation_rad':max_rotation,
             'maximum_hull_gauss_von_mises_Pa':hull_vm,
             'hull_mass_kg':float(mass[2:6*nh:6].sum()),'ice_mass_kg':float(mass[6*nh+2::6].sum()),
             'steel_shell_mass_kg':s['steel_shell_mass'],
             'attached_equipment_and_ballast_mass_kg':c.hull_total_mass-s['steel_shell_mass'],
             'initial_kinetic_energy_J':initial_energy,'initial_vertical_momentum_kg_m_s':initial_momentum,
             'support_vertical_impulse_N_s':external_impulse}
    return {'schema':'tensorfem.suboff-ice-impact/1','status':'demonstration-audited' if all(checks.values()) else 'blocked',
            'physical_accuracy_qualified':False,'checks':checks,'metrics':metrics,'config':asdict(c),
            'solver':{'method':'velocity Verlet, lumped physical mass, no mass scaling',
                      'time_step_s':dt,'steps':steps,'conservative_time_step_bound_s':stable,
                      'drilling_rotary_inertia':'isotropic rho*t^3/12 lumped on each rotational DOF'},
            'scope':'dry initial upward collision; elastic small-strain hull and ice facets; artificial equal-strength mixed-opening cohesive seams on prescribed grid; fixed horizontal contact registration; clamped ice perimeter; no buoyancy, water, crushing plasticity or fragment self-contact',
            'stress_scope':'four in-plane Gauss points, top/bottom skins, every integration step; local [sigma_x,sigma_y,tau_xy], transverse shear excluded',
            'geometry_source':geometry['source'],'geometry_audit':s['geometry_audit'],
            'rigid_translation_errors':s['rigid_translation_errors'],
            'nodes':s['nodes'].tolist(),'outer_hull_nodes':s['outer'].tolist(),
            'hull_elements':s['hull_cells'].tolist(),'ice_elements':s['ice_cells'].tolist(),
            'mass_diagonal':mass.tolist(),'fixed_dofs':s['fixed'].tolist(),
            'cohesive_node_pairs':(s['bid'][:,[0,6]]//6).tolist(),
            'cohesive_area_weights':s['barea'].tolist(),'cohesive_seam_ids':s['seams'].tolist(),
            'cohesive_damage':final_damage.tolist(),'cohesive_maximum_opening_m':maximum.tolist(),
            'seam_damage':seam_damage.tolist(),'history':history,'snapshots':snapshots,
            'final_dofs':q.reshape(-1,6).tolist(),'final_velocity':v.reshape(-1,6).tolist(),
            'peak_contact_dofs':peak_q.reshape(-1,6).tolist(),
            'final_stress_top_bottom_local_Pa':final_stress.tolist(),
            'final_von_mises_top_bottom_Pa':final_vm.tolist(),
            'peak_contact_stress_top_bottom_local_Pa':peak_stress.tolist(),
            'peak_contact_von_mises_top_bottom_Pa':peak_vm.tolist()}


def save_result(path,result,geometry_path=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if geometry_path is not None:
        result['geometry_file_sha256']=hashlib.sha256(Path(geometry_path).read_bytes()).hexdigest()
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(result,separators=(',',':'),allow_nan=False)+'\n')
    temporary.replace(path)
