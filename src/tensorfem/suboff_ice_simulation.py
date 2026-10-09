"""Appended SUBOFF and configurable ice impact, with conservative bookkeeping.

Initial small-strain impact only. Appendages are welded equivalent sheet shells.
Contact has fixed horizontal registration. Hydrostatic immersion and diagonal
drag are explicit reduced fluid models; finite-rotation fragment recontact and
sea-ice crushing are outside this solver's applicability.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import torch

from .cohesive import BilinearCohesiveLaw
from .ice_sheet import IceSimulationConfig,build_ice_sheet,buoyancy_response,ice_fragments
from .plate import _shape
from .spherical_shell import projected_shell4_stiffness
from .suboff_appendages import audit_appended_mesh
from .suboff_geometry import audit_closed_quads,inward_midsurface
from .suboff_ice_collision import _area,_basis,_ids,_sparse,_stress,cohesive_response,contact_response,conservative_step_bound

D=torch.float64


def _operators(nodes,elements,materials):
    out=[]
    for el,(E,nu,t,_) in zip(elements,materials):
        xy,basis=_basis(nodes[el]);C=E/(1-nu**2)*xy.new_tensor([[1.,nu,0.],[nu,1.,0.],[0.,0.,(1-nu)/2]])
        points=[]
        for xi,eta in ((-3**-.5,-3**-.5),(3**-.5,-3**-.5),(3**-.5,3**-.5),(-3**-.5,3**-.5)):
            _,dn=_shape(xy.new_tensor(xi),xy.new_tensor(eta));grad=dn@torch.linalg.inv(xy.T@dn)
            bm=torch.zeros((3,24),dtype=D);bb=torch.zeros_like(bm)
            for a,(gx,gy) in enumerate(grad):
                bm[0,6*a:6*a+3]=gx*basis[0];bm[1,6*a:6*a+3]=gy*basis[1]
                bm[2,6*a:6*a+3]=gy*basis[0]+gx*basis[1]
                bb[0,6*a+3:6*a+6]=gx*basis[1];bb[1,6*a+3:6*a+6]=-gy*basis[0]
                bb[2,6*a+3:6*a+6]=gy*basis[1]-gx*basis[0]
            points.append(torch.stack((C@(bm+t/2*bb),C@(bm-t/2*bb))))
        out.append(torch.stack(points))
    return torch.stack(out)


def build_simulation(geometry,c=IceSimulationConfig()):
    c.validate();schema=geometry.get('schema')
    if schema not in ('tensorfem.suboff-outer-geometry/1','tensorfem.suboff-appended-shell/1'):
        raise ValueError('unsupported geometry schema')
    outer=torch.tensor(geometry['nodes'],dtype=D);struct_cells=torch.tensor(geometry['elements'],dtype=torch.long)
    appended=schema=='tensorfem.suboff-appended-shell/1'
    bn=geometry['body_node_count'] if appended else len(outer)
    be=geometry['body_element_count'] if appended else len(struct_cells)
    components=geometry['components'] if appended else [{'id':0,'name':'bare_hull','root_nodes':[]}]
    labels=torch.tensor(geometry['element_component_ids'] if appended else [0]*be,dtype=torch.long)
    audit=(audit_appended_mesh(outer,struct_cells,bn,be,components) if appended else
           audit_closed_quads(outer,struct_cells))
    if not audit['passed']:raise ValueError('invalid connected structural geometry')
    body,body_normal=inward_midsurface(outer[:bn],struct_cells[:be],c.hull_thickness)
    structural=outer.clone();structural[:bn]=body;ns=len(structural);ne=len(struct_cells)
    hmap=(torch.tensor(c.ice_thickness_map,dtype=D).reshape(-1) if c.ice_thickness_map else
          torch.full((c.ice_nx*c.ice_ny,),c.ice_thickness,dtype=D))
    if c.open_water_cells:hmap[list(c.open_water_cells)]=0
    hmax=float(hmap.max()) if bool((hmap>0).any()) else c.ice_thickness
    sheet=build_ice_sheet(geometry['length_m'],float(outer[:,2].max())+c.initial_gap+hmax,c)
    ice_cells=sheet['elements']+ns
    nodes=torch.cat((structural,sheet['nodes']));elements=torch.cat((struct_cells,ice_cells));nd=6*len(nodes)
    mass=torch.zeros((len(nodes),6),dtype=D);ke=[];areas=[];materials=[]
    for i,el in enumerate(elements):
        if i<ne:
            t=c.hull_thickness if labels[i]==0 else c.appendage_thickness
            E,nu,rho=c.hull_young,c.hull_poisson,c.hull_density
        else:t=float(sheet['thickness'][i-ne]);E,nu,rho=c.ice_young,c.ice_poisson,c.ice_density
        area=_area(nodes[el]);areas.append(area);materials.append((E,nu,t,rho))
        ke.append(projected_shell4_stiffness(nodes[el],E,nu,t))
        m=rho*t*area/4;mass[el,:3]+=m;mass[el,3:]+=m*t*t/12
    physical_ice_mass=mass[ns:,2].clone()
    steel_mass=float(mass[:ns,2].sum())
    if c.hull_total_mass<steel_mass:raise ValueError('total vehicle mass is below steel-shell mass')
    mass[:ns,:3]*=c.hull_total_mass/steel_mass
    mass[ns:,:3]+=c.water_added_mass_per_area*sheet['node_area'][:,None]
    mass=mass.reshape(-1);ids=_ids(elements);ke=torch.stack(ke);shell=_sparse([ids],[ke],nd)
    bid=sheet['bond_ids']+6*ns;B=sheet['bond_B'];barea=sheet['bond_areas']
    law=BilinearCohesiveLaw(c.cohesive_factor*c.ice_young/min(sheet['grid'][2:]),c.ice_strength,c.ice_fracture_energy)
    bk=torch.einsum('pij,pik->pjk',B,B)*(law.stiffness*barea)[:,None,None]
    fixed=sheet['fixed_dofs']+6*ns;free=torch.ones(nd,dtype=torch.bool);free[fixed]=False
    nodal_area=torch.zeros(ns,dtype=D)
    for k in range(4):nodal_area.index_add_(0,struct_cells[:,k],torch.tensor(areas[:ne],dtype=D)/4)
    node_component=torch.zeros(ns,dtype=torch.long)
    for comp in components[1:]:
        el=struct_cells[labels==comp['id']];nonroot=el.unique();nonroot=nonroot[nonroot>=bn]
        node_component[nonroot]=comp['id']
    candidate=set(torch.nonzero(body_normal[:,2]>.05).flatten().tolist())
    if appended:
        candidate.update(geometry['components'][1]['top_nodes'])
        for comp in components[2:]:
            if comp['name']=='fin_upper':
                candidate.update(struct_cells[labels==comp['id']].unique().tolist())
        for node,area in geometry['sail_roof_contact_area_m2'].items():nodal_area[int(node)]=area
    x0,y0,dx,dy=sheet['grid'];cid=[];weights=[];gap=[];penalty=[];contact_nodes=[];contact_parts=[]
    for node in sorted(candidate):
        x,y,z=outer[node].tolist()
        i=int(math.floor((x-x0)/dx));j=int(math.floor((y-y0)/dy))
        if i<0 or i>=c.ice_nx or j<0 or j>=c.ice_ny:continue
        grid_id=j*c.ice_nx+i
        if grid_id not in sheet['cell_lookup'] or nodal_area[node]<=0:continue
        cell=sheet['cell_lookup'][grid_id];el=ice_cells[cell];h=float(sheet['thickness'][cell])
        a=(x-x0-i*dx)/dx;b=(y-y0-j*dy)/dy;N=[(1-a)*(1-b),a*(1-b),a*b,(1-a)*b]
        cid.append([6*node+2]+[6*int(v)+2 for v in el]);weights.append([1.]+[-v for v in N])
        gap.append(float(nodes[el[0],2])-h/2-z)
        penalty.append(c.contact_factor*c.ice_young/h*float(nodal_area[node]))
        contact_nodes.append(node);contact_parts.append(int(node_component[node]))
    cid=torch.tensor(cid,dtype=torch.long).reshape(-1,5);cw=torch.tensor(weights,dtype=D).reshape(-1,5)
    cp=torch.tensor(penalty,dtype=D);gaps=torch.tensor(gap,dtype=D)
    if bool((gaps< -1e-12).any()):raise ValueError('ice initially overlaps vehicle surface')
    ck=cp[:,None,None]*cw[:,:,None]*cw[:,None,:]
    softening=max(1.,law.onset_opening/(law.failure_opening-law.onset_opening))
    if c.ice_model=='elastic':softening=1.
    upper=_sparse([ids],[ke.abs()],nd)
    if len(bid):upper=upper+_sparse([bid],[bk.abs()*softening],nd)
    if len(cid):upper=upper+_sparse([cid],[ck.abs()],nd)
    water_diagonal=torch.zeros(nd,dtype=D)
    if c.hydrostatic_buoyancy:water_diagonal[6*ns+2::6]=c.water_density*c.gravity*sheet['node_area']
    wi=torch.nonzero(water_diagonal).flatten()
    if len(wi):
        upper=upper+torch.sparse_coo_tensor(torch.stack((wi,wi)),water_diagonal[wi],(nd,nd),check_invariants=True)
    bound=conservative_step_bound(upper,mass,free)
    rigid=[]
    for k in range(3):
        u=torch.zeros(nd,dtype=D);u[k:6*ns:6]=1
        f=torch.sparse.mm(shell,u[:,None])[:,0]
        rigid.append(float(f.abs().max())/float(shell.values().abs().max()))
    if max(rigid)>1e-10:raise ValueError('vehicle rigid-translation gate failed')
    damping=torch.zeros(nd,dtype=D)
    for k in range(3):damping[6*ns+k::6]=c.water_drag_per_area*sheet['node_area']
    return dict(nodes=nodes,outer=outer,structural_cells=struct_cells,ice_cells=ice_cells,
                elements=elements,ns=ns,ne=ne,bn=bn,body_element_count=be,components=components,
                labels=labels,areas=torch.tensor(areas,dtype=D),materials=materials,
                mass=mass,physical_ice_mass=physical_ice_mass,steel_mass=steel_mass,
                shell=shell.to_sparse_csr(),ids=ids,ke=ke,bid=bid,B=B,barea=barea,
                law=law,sheet=sheet,fixed=fixed,free=free,cid=cid,cw=cw,cp=cp,gaps=gaps,
                contact_components=torch.tensor(contact_parts,dtype=torch.long),contact_nodes=contact_nodes,
                time_bound=bound,water_diagonal=water_diagonal,damping=damping,
                geometry_audit=audit,rigid_translation_errors=rigid,
                stress_operator=_operators(nodes,elements,materials),config=c)


def evaluate_state(s,q,maximum):
    c=s['config'];internal=torch.mv(s['shell'],q);se=.5*torch.dot(q,internal)
    if not len(s['bid']):
        bond=torch.zeros_like(q);be=q.new_zeros(());diss=be.clone();damage=maximum.clone();new=maximum.clone()
    elif c.ice_model=='cohesive':
        bond,be,diss,damage,new=cohesive_response(q,s['bid'],s['B'],s['sheet']['bond_normals'],s['barea'],s['law'],maximum)
    else:
        jump=torch.einsum('pij,pj->pi',s['B'],q[s['bid']]);traction=s['law'].stiffness*jump
        f=torch.einsum('pij,pi->pj',s['B'],traction*s['barea'][:,None])
        bond=torch.zeros_like(q);bond.index_add_(0,s['bid'].reshape(-1),f.reshape(-1))
        be=.5*torch.sum(s['barea'][:,None]*traction*jump);diss=q.new_zeros(())
        damage=torch.zeros_like(maximum);new=maximum.clone()
    contact,ce,forces,penetration=contact_response(q,s['cid'],s['cw'],s['gaps'],s['cp'])
    ice_w=q.reshape(-1,6)[s['ns']:,2]
    water,water_energy=buoyancy_response(ice_w,s['sheet']['node_area'],s['sheet']['node_thickness'],c)
    total=internal+bond+contact;total[6*s['ns']+2::6]+=water
    return dict(force=total,shell_energy=se,bond_energy=be,dissipation=diss,damage=damage,
                maximum=new,contact_energy=ce,contact_forces=forces,penetration=penetration,
                water_force=water,water_energy=water_energy)


def simulate(sgeometry,c=IceSimulationConfig(),*,dt=None):
    s=build_simulation(sgeometry,c);mass=s['mass'];ns=s['ns'];nd=len(mass)
    request=c.time_safety*s['time_bound'] if dt is None else dt
    if not math.isfinite(request) or request<=0 or request>c.time_safety*s['time_bound']*(1+1e-12):
        raise ValueError('time step exceeds conservative bound')
    steps=math.ceil(c.duration/request);dt=c.duration/steps
    q=torch.zeros(nd,dtype=D);v=q.clone();v[2:6*ns:6]=c.upward_speed
    maximum=torch.zeros(len(s['bid']),dtype=D)
    maximum[s['sheet']['precracked_points']]=s['law'].failure_opening
    state=evaluate_state(s,q,maximum);preexisting_diss=float(state['dissipation'])
    initial_energy=float(.5*torch.dot(mass,v*v));p0=float(mass[2::6]@v[2::6])
    acc=-state['force']/mass;acc[s['fixed']]=0
    damping_factor=torch.exp(-.5*dt*s['damping']/mass);damping_factor[s['fixed']]=1
    drag_diss=0.;external_impulse=0.;emax=0.;pmax=0.;penetration=0.;peak=0.;peak_time=0.;peak_q=q.clone()
    max_rotation=0.;max_ice_ratio=0.;peak_vm=0.;part_stress=torch.zeros(len(s['components']),dtype=D)
    part_force=torch.zeros_like(part_stress);history=[];snapshots=[];stride=max(1,steps//c.sample_count)
    def drag(velocity):
        after=velocity*damping_factor
        loss=float(.5*torch.dot(mass,velocity**2-after**2))
        impulse=float(mass[2::6]@(after[2::6]-velocity[2::6]))
        return after,loss,impulse
    def external_force(st):
        fixed_z=s['fixed'][s['fixed']%6==2]
        return float(st['force'][fixed_z].sum()-st['water_force'].sum())
    for step in range(steps+1):
        kinetic=.5*torch.dot(mass,v*v)
        diss=float(state['dissipation'])-preexisting_diss
        total=float(kinetic+state['shell_energy']+state['bond_energy']+state['contact_energy']+state['water_energy'])+diss+drag_diss
        ee=abs(total/initial_energy-1);momentum=float(mass[2::6]@v[2::6]);pe=abs(momentum-p0-external_impulse)/abs(p0)
        emax=max(emax,ee);pmax=max(pmax,pe)
        f=state['contact_forces'];force=float(f.sum())
        if len(f):penetration=max(penetration,float(state['penetration'].max()))
        if force>peak:peak=force;peak_time=step*dt;peak_q=q.clone()
        for cid in range(len(part_force)):
            part_force[cid]=max(float(part_force[cid]),float(f[s['contact_components']==cid].sum()))
        max_rotation=max(max_rotation,float(q.reshape(-1,6)[:,3:].abs().max()))
        if len(s['sheet']['nodes']):
            max_ice_ratio=max(max_ice_ratio,float((q.reshape(-1,6)[ns:,2].abs()/s['sheet']['node_thickness']).max()))
        stress,vm=_stress(s,q)
        for cid in range(len(part_stress)):
            part_stress[cid]=max(float(part_stress[cid]),float(vm[:s['ne']][s['labels']==cid].max()))
        peak_vm=max(peak_vm,float(vm[:s['ne']].max()))
        if step%stride==0 or step==steps:
            topology=ice_fragments(s['sheet'],state['damage'],s['physical_ice_mass'],
                                  v.reshape(-1,6)[ns:,:3],s['nodes'][ns:]+q.reshape(-1,6)[ns:,:3])
            history.append({'time_s':step*dt,'contact_force_N':force,'kinetic_J':float(kinetic),
                            'shell_strain_J':float(state['shell_energy']),'cohesive_stored_J':float(state['bond_energy']),
                            'contact_stored_J':float(state['contact_energy']),'hydrostatic_potential_J':float(state['water_energy']),
                            'new_fracture_dissipation_J':diss,'drag_dissipation_J':drag_diss,
                            'energy_relative_error':ee,'vertical_momentum_kg_m_s':momentum,
                            'external_vertical_impulse_N_s':external_impulse,'momentum_relative_error':pe,
                            'maximum_damage':float(state['damage'].max()) if len(maximum) else 0.,
                            'fully_failed_points':int((state['damage']>=.999).sum()),
                            'ice_component_count':topology['component_count'],
                            'fully_severed_seam_count':len(topology['fully_severed_seams']),
                            'retained_physical_ice_mass_kg':topology['total_retained_mass_kg'],
                            'vehicle_vertical_velocity_m_s':float(mass[2:6*ns:6]@v[2:6*ns:6]/c.hull_total_mass)})
        if step in {0,steps//4,steps//2,3*steps//4,steps}:
            snapshots.append({'time_s':step*dt,'dofs':q.reshape(-1,6).tolist(),'damage':state['damage'].tolist()})
        if step==steps:break
        v,loss,impulse=drag(v);drag_diss+=loss;external_impulse+=impulse
        half=v+.5*dt*acc;q+=dt*half;q[s['fixed']]=0
        next_state=evaluate_state(s,q,state['maximum']);maximum=next_state['maximum']
        next_acc=-next_state['force']/mass;next_acc[s['fixed']]=0
        v=half+.5*dt*next_acc;v[s['fixed']]=0
        v,loss,impulse=drag(v);drag_diss+=loss;external_impulse+=impulse
        external_impulse+=.5*dt*(external_force(state)+external_force(next_state))
        state=next_state;acc=next_acc
        if not bool(torch.isfinite(q).all() and torch.isfinite(v).all()):raise RuntimeError('nonfinite dynamic state')
    final_stress,final_vm=_stress(s,q);pstress,pvm=_stress(s,peak_q)
    fragments=ice_fragments(s['sheet'],state['damage'],s['physical_ice_mass'],
                           v.reshape(-1,6)[ns:,:3],s['nodes'][ns:]+q.reshape(-1,6)[ns:,:3])
    active=len(s['sheet']['elements']);ice_mass=float(s['physical_ice_mass'].sum())
    contact_ok=(peak>0 if c.contact_expectation=='required' else peak==0 if c.contact_expectation=='absent' else True)
    checks={'connected_vehicle':s['geometry_audit']['passed'],'rigid_translation':max(s['rigid_translation_errors'])<1e-10,
            'energy_balance':emax<.03,'vertical_momentum_balance':pmax<1e-8,
            'contact_expectation':contact_ok,'small_penetration':penetration<c.ice_thickness*.03,
            'small_rotation':max_rotation<.1,'small_ice_deflection':max_ice_ratio<.5,
            'elastic_steel_gauss_stress':peak_vm<c.hull_yield_reference,
            'ice_mass_retained':abs(fragments['total_retained_mass_kg']-ice_mass)<1e-9*max(ice_mass,1)}
    return {'schema':'tensorfem.suboff-ice-simulation/2','status':'demonstration-audited' if all(checks.values()) else 'blocked',
            'physical_accuracy_qualified':False,'config':asdict(c),'geometry_source':sgeometry['source'],
            'geometry_audit':s['geometry_audit'],'checks':checks,
            'scope':'equivalent welded appendage sheets; small-strain initial collision; fixed horizontal contact registration; optional clipped hydrostatic immersion, declared added inertia and drag; prescribed ice crack grid; no sea-ice crushing or general fragment self-contact',
            'solver':{'method':'velocity Verlet with exact diagonal-drag half steps','time_step_s':dt,
                      'conservative_time_step_bound_s':s['time_bound'],'steps':steps,'mass_scaling':False},
            'metrics':{'peak_contact_force_N':peak,'peak_contact_time_s':peak_time,
                       'maximum_energy_relative_error':emax,'maximum_vertical_momentum_relative_error':pmax,
                       'maximum_penetration_m':penetration,'maximum_rotation_rad':max_rotation,
                       'maximum_ice_deflection_over_local_thickness':max_ice_ratio,
                       'maximum_vehicle_gauss_von_mises_Pa':peak_vm,
                       'component_peak_contact_force_N':dict(zip((p['name'] for p in s['components']),part_force.tolist())),
                       'component_maximum_von_mises_Pa':dict(zip((p['name'] for p in s['components']),part_stress.tolist())),
                       'vehicle_total_mass_kg':c.hull_total_mass,'steel_mass_kg':s['steel_mass'],
                       'physical_ice_mass_kg':ice_mass,'water_added_mass_kg':float(c.water_added_mass_per_area*s['sheet']['node_area'].sum()),
                       'initial_kinetic_energy_J':initial_energy,'initial_vertical_momentum_kg_m_s':p0,
                       'preexisting_crack_energy_J':preexisting_diss,'new_fracture_dissipation_J':diss,
                       'drag_dissipation_J':drag_diss,'active_ice_tiles':active,
                       'initial_failed_points':int(s['sheet']['precracked_points'].sum()),
                       'final_failed_points':int((state['damage']>=.999).sum()),
                       'final_ice_component_count':fragments['component_count']},
            'nodes':s['nodes'].tolist(),'outer_vehicle_nodes':s['outer'].tolist(),
            'vehicle_elements':s['structural_cells'].tolist(),'element_component_ids':s['labels'].tolist(),
            'components':s['components'],'ice_elements':s['ice_cells'].tolist(),
            'ice_grid_ids':s['sheet']['active_grid_ids'].tolist(),'ice_cell_thickness_m':s['sheet']['thickness'].tolist(),
            'mass_diagonal':mass.tolist(),'fixed_dofs':s['fixed'].tolist(),
            'bond_node_pairs':(s['bid'][:,[0,6]]//6).tolist(),'bond_area_m2':s['barea'].tolist(),
            'cohesive_seam_ids':s['sheet']['seam_ids'].tolist(),'cohesive_maximum_opening_m':state['maximum'].tolist(),
            'cohesive_damage':state['damage'].tolist(),'fragments':fragments,
            'final_dofs':q.reshape(-1,6).tolist(),'final_velocity':v.reshape(-1,6).tolist(),
            'final_stress_top_bottom_local_Pa':final_stress.tolist(),'final_von_mises_top_bottom_Pa':final_vm.tolist(),
            'peak_contact_dofs':peak_q.reshape(-1,6).tolist(),'peak_contact_von_mises_top_bottom_Pa':pvm.tolist(),
            'history':history,'snapshots':snapshots}
