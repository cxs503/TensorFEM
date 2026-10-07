"""Versioned, allow-listed engineering project workflow schema."""
from dataclasses import asdict,dataclass,field
from pathlib import Path
import hashlib,json,time,traceback
from .spherical_shell import hemisphere_with_hole
from .pinched_cylinder import solve_pinched_cylinder_linear
from .large_rotation_shell_benchmark import pure_bending_shell
SCHEMA="tensorfem.project.v1"
@dataclass(frozen=True)
class Material: name:str;model:str;properties:dict[str,float]
@dataclass(frozen=True)
class Section: name:str;kind:str;material:str;properties:dict[str,float]
@dataclass(frozen=True)
class Load: name:str;kind:str;case_definition:str
@dataclass(frozen=True)
class Constraint: name:str;kind:str;case_definition:str
@dataclass(frozen=True)
class OutputRequest: name:str;fields:tuple[str,...]
@dataclass(frozen=True)
class ProjectModel: name:str;case:str;parameters:dict[str,float|int];section:str
@dataclass(frozen=True)
class ProjectStep: name:str;procedure:str;loads:tuple[str,...];constraints:tuple[str,...];outputs:tuple[str,...]
@dataclass(frozen=True)
class Project:
 name:str;model:ProjectModel;materials:tuple[Material,...];sections:tuple[Section,...]
 steps:tuple[ProjectStep,...];loads:tuple[Load,...];constraints:tuple[Constraint,...];outputs:tuple[OutputRequest,...]
 units:dict[str,str];schema:str=SCHEMA
def _canon(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True)
def project_id(p):return hashlib.sha256(_canon(asdict(p)).encode()).hexdigest()[:20]
def _unique(items,label):
 names=[x.name for x in items]
 if len(names)!=len(set(names)):raise ValueError(f"duplicate {label} name")
 return set(names)
def validate_project(p):
 if p.schema!=SCHEMA:raise ValueError("unsupported project schema")
 if p.model.case not in {"hemisphere_18deg","pinched_cylinder","large_rotation_shell"}:raise ValueError("case is not allow-listed")
 mats=_unique(p.materials,"material");secs=_unique(p.sections,"section");loads=_unique(p.loads,"load");cons=_unique(p.constraints,"constraint");outs=_unique(p.outputs,"output")
 if p.model.section not in secs:raise ValueError("model references unknown section")
 for s in p.sections:
  if s.material not in mats:raise ValueError("section references unknown material")
 if len(p.steps)!=1:raise ValueError("exactly one step is currently supported")
 st=p.steps[0]
 if not set(st.loads)<=loads or not set(st.constraints)<=cons or not set(st.outputs)<=outs:raise ValueError("step has an unknown reference")
 expected="nonlinear_static" if p.model.case=="large_rotation_shell" else "linear_static"
 if st.procedure!=expected:raise ValueError("procedure incompatible with allow-listed case")
 if set(p.units)!={"length","force","displacement","rotation","stress"}:raise ValueError("explicit engineering units are required")
 allowed={"hemisphere_18deg":{"nphi","ntheta","drilling_factor"},"pinched_cylinder":{"n"},"large_rotation_shell":{"n","angle"}}[p.model.case]
 if set(p.model.parameters)!=allowed:raise ValueError("unknown or missing case parameters")
def _dispatch(p):
 c=p.model.case;q=p.model.parameters
 if c=="hemisphere_18deg":
  r=hemisphere_with_hole(int(q['nphi']),int(q['ntheta']),drilling_factor=float(q['drilling_factor']))
  return r.solution.tolist(),r.reaction.tolist(),{"probe":r.displacement,"relative_error":r.relative_error,"kernel":"hemisphere-with-hole.v1"}
 if c=="pinched_cylinder":
  r=solve_pinched_cylinder_linear(int(q['n']))
  return [r.displacement],[],{"probe":r.displacement,"relative_error":r.relative_error,"kernel":"pinched-cylinder-linear.v1"}
 r=pure_bending_shell(int(q['n']),angle=float(q['angle']))
 return r.result.dofs.tolist(),r.result.reaction.tolist(),{"tip":r.tip.tolist(),"tip_error":r.tip_error,"moment_error":r.moment_error,"kernel":"large-rotation-shell.v1"}
def run_project(p,root,replay=True):
 jid=project_id(p);d=Path(root)/jid;d.mkdir(parents=True,exist_ok=True);manifest=asdict(p);mp=d/'project.json';jp=d/'job.json';rp=d/'result.json'
 if mp.exists() and _canon(json.loads(mp.read_text()))!=_canon(manifest):raise RuntimeError("project id collision")
 mp.write_text(json.dumps(manifest,sort_keys=True,indent=2));start=time.time()
 try:
  validate_project(p)
  if replay and jp.exists() and rp.exists():
   j=json.loads(jp.read_text());raw=rp.read_bytes()
   if j.get('status')=='completed':
    if hashlib.sha256(raw).hexdigest()!=j['result_sha256']:raise RuntimeError("result checksum mismatch")
    out=json.loads(raw);out['metadata']['replayed']=True;return out
  jp.write_text(json.dumps({'schema':SCHEMA,'job_id':jid,'status':'running','started_unix':start},indent=2))
  u,r,meta=_dispatch(p);meta.update({'schema':SCHEMA,'units':p.units,'case':p.model.case,'elapsed_seconds':time.time()-start,'replayed':False})
  out={'job_id':jid,'displacement':u,'reaction':r,'metadata':meta};raw=json.dumps(out,sort_keys=True,indent=2).encode();rp.write_bytes(raw)
  jp.write_text(json.dumps({'schema':SCHEMA,'job_id':jid,'status':'completed','elapsed_seconds':time.time()-start,'result_sha256':hashlib.sha256(raw).hexdigest()},indent=2));return out
 except Exception as e:
  jp.write_text(json.dumps({'schema':SCHEMA,'job_id':jid,'status':'failed','error_type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()},indent=2));raise
def project_from_json(path):
 raw=json.loads(Path(path).read_text());expected={'schema','name','model','materials','sections','steps','loads','constraints','outputs','units'}
 if set(raw)!=expected:raise ValueError("unknown or missing project fields")
 def exact(obj,keys,label):
  if set(obj)!=set(keys):raise ValueError(f"unknown or missing {label} fields")
 exact(raw['model'],('name','case','parameters','section'),'model')
 maps=(("materials",Material,('name','model','properties')),("sections",Section,('name','kind','material','properties')),("steps",ProjectStep,('name','procedure','loads','constraints','outputs')),("loads",Load,('name','kind','case_definition')),("constraints",Constraint,('name','kind','case_definition')),("outputs",OutputRequest,('name','fields')))
 built={}
 for key,cls,keys in maps:
  for x in raw[key]:exact(x,keys,key)
  built[key]=tuple(cls(**x) for x in raw[key])
 p=Project(raw['name'],ProjectModel(**raw['model']),built['materials'],built['sections'],built['steps'],built['loads'],built['constraints'],built['outputs'],raw['units'],raw['schema']);validate_project(p);return p
