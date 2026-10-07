"""Safe v1 arbitrary-mesh project loader and static solver dispatcher."""
from pathlib import Path
import hashlib,json,time,traceback,torch
from .model import TrussModel
from .solvers import solve_linear_static
from .solid3d import SolidModel,solve_solid
from .thermal import ThermalModel,assemble_thermal,solve_steady_thermal
SCHEMA="tensorfem.mesh-project.v1"; TYPES={"truss2d":(2,2),"tet4_linear":(3,4),"hex8_linear":(3,8),"thermal_line2":(1,2),"thermal_q4":(2,4)}
def _exact(x,keys,label):
 if set(x)!=set(keys):raise ValueError(f"unknown or missing {label} fields")
def _safe_file(root,name):
 if not isinstance(name,str) or Path(name).is_absolute():raise ValueError("mesh file must be a relative path")
 p=(root/name).resolve()
 if not p.is_relative_to(root.resolve()):raise ValueError("mesh file escapes project root")
 if p.suffix.lower()!='.json':raise ValueError("only JSON mesh files are allowed")
 return p
def load_mesh_project(path):
 path=Path(path).resolve();raw=json.loads(path.read_text());top={'schema','name','units','model','mesh','sets','materials','sections','step','loads','constraints','outputs'}
 _exact(raw,top,'project');
 if raw['schema']!=SCHEMA:raise ValueError("unsupported mesh project schema")
 _exact(raw['model'],{'name','element_type'},'model');etype=raw['model']['element_type']
 if etype not in TYPES:raise ValueError("element type is not allow-listed")
 thermal=etype.startswith('thermal_')
 _exact(raw['units'],{'length','temperature','heat_rate','conductivity'} if thermal else {'length','force','stress','displacement'},'units')
 mesh=raw['mesh'];
 if set(mesh)=={'file'}:mesh=json.loads(_safe_file(path.parent,mesh['file']).read_text())
 _exact(mesh,{'nodes','elements'},'mesh')
 for n in mesh['nodes']:_exact(n,{'id','coordinates'},'node')
 for e in mesh['elements']:_exact(e,{'id','type','connectivity'},'element')
 raw['mesh']=mesh
 _exact(raw['sets'],{'node','element','surface'},'sets')
 for m in raw['materials']:_exact(m,{'name','model','properties'},'material')
 for s in raw['sections']:_exact(s,{'name','element_set','material','properties'},'section')
 _exact(raw['step'],{'name','procedure'},'step')
 if raw['step']['procedure']!='linear_static':raise ValueError("only linear_static is allowed")
 for x in raw['loads']:_exact(x,{'name','type','target','dof','value'},'load')
 for x in raw['constraints']:_exact(x,{'name','type','target','dofs','value'},'constraint')
 for x in raw['outputs']:_exact(x,{'name','fields'},'output')
 _validate(raw);return raw
def _validate(p):
 dim,nen=TYPES[p['model']['element_type']];nodes=p['mesh']['nodes'];elements=p['mesh']['elements']
 nids=[n['id'] for n in nodes];eids=[e['id'] for e in elements]
 if len(set(nids))!=len(nids) or len(set(eids))!=len(eids):raise ValueError("duplicate node or element ID")
 ns=set(nids);es=set(eids)
 for n in nodes:
  if len(n['coordinates'])!=dim:raise ValueError("node coordinate dimension mismatch")
 for e in elements:
  if e['type']!=p['model']['element_type'] or len(e['connectivity'])!=nen or not set(e['connectivity'])<=ns:raise ValueError("invalid element topology")
 for name,ids in p['sets']['node'].items():
  if not set(ids)<=ns:raise ValueError("node set references missing node")
 for name,ids in p['sets']['element'].items():
  if not set(ids)<=es:raise ValueError("element set references missing element")
 for name,faces in p['sets']['surface'].items():
  for f in faces:
   _exact(f,{'element','face'},'surface');
   if f['element'] not in es or not 1<=f['face']<=nen:raise ValueError("invalid surface reference")
 mats={m['name']:m for m in p['materials']};elsets=p['sets']['element'];covered=set()
 for sec in p['sections']:
  if sec['element_set'] not in elsets or sec['material'] not in mats:raise ValueError("section has unknown reference")
  overlap=covered&set(elsets[sec['element_set']])
  if overlap:raise ValueError("element has multiple sections")
  covered|=set(elsets[sec['element_set']])
 if covered!=es:raise ValueError("every element needs exactly one section")
 thermal=p['model']['element_type'].startswith('thermal_');ndof=1 if thermal else dim
 for bc in p['constraints']:
  expected_bc='temperature' if thermal else 'displacement'
  if bc['type']!=expected_bc or bc['target'] not in p['sets']['node'] or any(d<1 or d>ndof for d in bc['dofs']) or (not thermal and bc['value']!=0):raise ValueError("unsupported constraint")
 for load in p['loads']:
  expected_load='heat' if thermal else 'nodal'
  if load['type']!=expected_load or load['target'] not in p['sets']['node'] or not 1<=load['dof']<=ndof:raise ValueError("unsupported load")
 allowed={'TEMP','FLUX'} if thermal else {'U','RF','S','E'}
 if any(not set(o['fields'])<=allowed for o in p['outputs']):raise ValueError("unsupported output field")
 expected='steady_thermal' if thermal else 'linear_elastic';
 if any(m['model']!=expected for m in p['materials']):raise ValueError("only linear_elastic material is allowed")
def _solve(p):
 dim,nen=TYPES[p['model']['element_type']];nodes=sorted(p['mesh']['nodes'],key=lambda x:x['id']);idx={n['id']:i for i,n in enumerate(nodes)}
 elements=sorted(p['mesh']['elements'],key=lambda x:x['id']);conn=torch.tensor([[idx[n] for n in e['connectivity']] for e in elements]);eid_index={e['id']:i for i,e in enumerate(elements)}
 mats={m['name']:m for m in p['materials']};sec_by={}
 for s in p['sections']:
  for eid in p['sets']['element'][s['element_set']]:sec_by[eid]=s
 thermal=p['model']['element_type'].startswith('thermal_');dof_per_node=1 if thermal else dim
 force=torch.zeros(dof_per_node*len(nodes),dtype=torch.float64);fixed=[]
 for l in p['loads']:
  for nid in p['sets']['node'][l['target']]:force[dof_per_node*idx[nid]+l['dof']-1]+=l['value']
 for b in p['constraints']:
  for nid in p['sets']['node'][b['target']]:
   for d in b['dofs']:fixed.append(dof_per_node*idx[nid]+d-1)
 xyz=torch.tensor([n['coordinates'] for n in nodes],dtype=torch.float64);fixed=torch.tensor(sorted(set(fixed)),dtype=torch.long)
 if p['model']['element_type']=='truss2d':
  E=[];A=[]
  for e in elements:
   s=sec_by[e['id']];prop=mats[s['material']]['properties'];
   if set(prop)!={'young'} or set(s['properties'])!={'area'}:raise ValueError("truss requires young and area")
   E.append(prop['young']);A.append(s['properties']['area'])
  r=solve_linear_static(TrussModel(xyz,conn,torch.tensor(E),torch.tensor(A),force,fixed))
  return r.displacement,r.reaction,{"stress":r.axial_stress.tolist(),"strain":r.axial_strain.tolist()}
 if p['model']['element_type'].startswith('thermal_'):
  conductivity=[];density=[];specific=[];thickness=[]
  for e in elements:
   s=sec_by[e['id']];prop=mats[s['material']]['properties']
   if set(prop)!={'conductivity','density','specific_heat'}:raise ValueError("thermal material requires conductivity, density and specific_heat")
   if set(s['properties'])!={'thickness'}:raise ValueError("thermal section requires thickness")
   conductivity.append(prop['conductivity']);density.append(prop['density']);specific.append(prop['specific_heat']);thickness.append(s['properties']['thickness'])
  fixed_nodes=[];fixed_temp=[]
  for b in p['constraints']:
   for nid in p['sets']['node'][b['target']]:fixed_nodes.append(idx[nid]);fixed_temp.append(b['value'])
  tm=ThermalModel(xyz,conn,torch.tensor(conductivity),torch.tensor(density),torch.tensor(specific),torch.tensor(fixed_nodes),torch.tensor(fixed_temp,dtype=torch.float64),nodal_heat=force,thickness=torch.tensor(thickness),element_type='line2' if p['model']['element_type']=='thermal_line2' else 'q4')
  temp=solve_steady_thermal(tm);flux=assemble_thermal(tm).conductivity@temp-assemble_thermal(tm).heat
  return temp,flux,{"temperature":temp.tolist(),"flux":flux.tolist()}
 E=[];nu=[]
 for e in elements:
  prop=mats[sec_by[e['id']]['material']]['properties']
  if set(prop)!={'young','poisson'}:raise ValueError("tet4 requires young and poisson")
  E.append(prop['young']);nu.append(prop['poisson'])
 solid_type='tet4' if p['model']['element_type']=='tet4_linear' else 'hex8'
 r=solve_solid(SolidModel(xyz,conn,torch.tensor(E),torch.tensor(nu),force,fixed,solid_type))
 return r.displacement,r.reaction,{"stress":r.stress.tolist(),"strain":r.strain.tolist()}
def run_mesh_project(path,run_root,replay=True):
 p=load_mesh_project(path);canonical=json.dumps(p,sort_keys=True,separators=(',',':'));jid=hashlib.sha256(canonical.encode()).hexdigest()[:20];d=Path(run_root)/jid;d.mkdir(parents=True,exist_ok=True);jp=d/'job.json';rp=d/'result.json';(d/'project.json').write_text(json.dumps(p,sort_keys=True,indent=2));start=time.time()
 try:
  if replay and jp.exists() and rp.exists():
   j=json.loads(jp.read_text());data=rp.read_bytes()
   if j.get('status')=='completed':
    if hashlib.sha256(data).hexdigest()!=j['result_sha256']:raise RuntimeError("result checksum mismatch")
    out=json.loads(data);out['metadata']['replayed']=True;return out
  jp.write_text(json.dumps({'status':'running','job_id':jid}));u,r,fields=_solve(p);out={'job_id':jid,'displacement':u.tolist(),'reaction':r.tolist(),'fields':fields,'metadata':{'schema':SCHEMA,'units':p['units'],'element_type':p['model']['element_type'],'elapsed_seconds':time.time()-start,'replayed':False}};data=json.dumps(out,sort_keys=True,indent=2).encode();rp.write_bytes(data);jp.write_text(json.dumps({'status':'completed','job_id':jid,'result_sha256':hashlib.sha256(data).hexdigest()}));return out
 except Exception as e:jp.write_text(json.dumps({'status':'failed','job_id':jid,'error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()}));raise
