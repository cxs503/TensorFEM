"""Minimal Model--Step--Job--Result workflow over existing solver kernels."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from pathlib import Path
from typing import Any
import hashlib,json,time,traceback
import torch
from .spherical_shell import hemisphere_with_hole
SCHEMA="tensorfem.workflow.v1";KERNEL="hemisphere-with-hole.v1"
def _canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True)
def _write(path,payload):path.write_text(json.dumps(payload,sort_keys=True,indent=2),encoding="utf-8")
@dataclass(frozen=True)
class ModelSpec:
 kind:str="hemisphere_18deg_hole";nphi:int=8;ntheta:int=8;drilling_factor:float=1e-6
 material:dict[str,float]=field(default_factory=lambda:{"young":6.825e7,"poisson":.3,"thickness":.04})
 constraints:str="quarter_symmetry_and_vertical_gauge";load:str="equator_opposed_unit_point_loads"
@dataclass(frozen=True)
class StepSpec:
 name:str="linear_static";procedure:str="linear_static"
@dataclass(frozen=True)
class WorkflowSpec:
 model:ModelSpec=field(default_factory=ModelSpec);step:StepSpec=field(default_factory=StepSpec)
 units:dict[str,str]=field(default_factory=lambda:{"length":"consistent","force":"consistent","displacement":"consistent"})
@dataclass(frozen=True)
class WorkflowResult:
 job_id:str;displacement:list[float];reaction:list[float];metadata:dict[str,Any]
def deterministic_job_id(spec:WorkflowSpec):
 payload={"schema":SCHEMA,"model":asdict(spec.model),"step":asdict(spec.step),"units":spec.units}
 return hashlib.sha256(_canonical(payload).encode()).hexdigest()[:20]
def _validate(spec):
 m=spec.model
 if m.kind!="hemisphere_18deg_hole":raise ValueError("unsupported model kind")
 if m.nphi<2 or m.ntheta<2:raise ValueError("mesh counts must be at least two")
 if m.material!={"young":6.825e7,"poisson":.3,"thickness":.04}:raise ValueError("kernel material is fixed; unsupported override")
 if m.constraints!="quarter_symmetry_and_vertical_gauge" or m.load!="equator_opposed_unit_point_loads":raise ValueError("unsupported constraint/load definition")
 if spec.step.procedure!="linear_static":raise ValueError("only linear_static is supported")
 if set(spec.units)!={"length","force","displacement"}:raise ValueError("units must define length, force and displacement")
def run_job(spec:WorkflowSpec,root:str|Path,*,replay=True)->WorkflowResult:
 """Execute or replay a deterministic job; all terminal failures are recorded."""
 jid=deterministic_job_id(spec);directory=Path(root)/jid;directory.mkdir(parents=True,exist_ok=True)
 manifest={"schema":SCHEMA,"job_id":jid,"model":asdict(spec.model),"step":asdict(spec.step),"units":spec.units}
 manifest_path=directory/"manifest.json";job_path=directory/"job.json";result_path=directory/"result.json"
 if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:raise RuntimeError("job-id collision or manifest changed")
 _write(manifest_path,manifest);started=time.time()
 try:
  _validate(spec)
  if replay and job_path.exists() and result_path.exists():
   status=json.loads(job_path.read_text());raw=result_path.read_bytes()
   if status.get("status")=="completed":
    if hashlib.sha256(raw).hexdigest()!=status.get("result_sha256"):raise RuntimeError("completed result checksum mismatch")
    payload=json.loads(raw);payload["metadata"]["replayed"]=True;return WorkflowResult(**payload)
  _write(job_path,{"schema":SCHEMA,"job_id":jid,"status":"running","started_unix":started})
  m=spec.model;solved=hemisphere_with_hole(m.nphi,m.ntheta,drilling_factor=m.drilling_factor)
  elapsed=time.time()-started
  metadata={"schema":SCHEMA,"kernel_version":KERNEL,"procedure":spec.step.procedure,"units":spec.units,
   "model_kind":m.kind,"nnode":len(solved.nodes),"nelement":len(solved.elements),"probe_displacement":solved.displacement,
   "reference_displacement":solved.reference,"relative_error":solved.relative_error,"elapsed_seconds":elapsed,"replayed":False}
  result=WorkflowResult(jid,solved.solution.tolist(),solved.reaction.tolist(),metadata);payload=asdict(result)
  encoded=json.dumps(payload,sort_keys=True,indent=2).encode();result_path.write_bytes(encoded)
  _write(job_path,{"schema":SCHEMA,"job_id":jid,"status":"completed","started_unix":started,"elapsed_seconds":elapsed,
                   "result_sha256":hashlib.sha256(encoded).hexdigest()})
  return result
 except Exception as exc:
  _write(job_path,{"schema":SCHEMA,"job_id":jid,"status":"failed","started_unix":started,"elapsed_seconds":time.time()-started,
                   "error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
  raise
def spec_from_json(path:str|Path):
 raw=json.loads(Path(path).read_text(encoding="utf-8"))
 if raw.get("schema")!=SCHEMA:raise ValueError("unsupported workflow schema")
 allowed={"schema","model","step","units"}
 if set(raw)-allowed:raise ValueError("unknown manifest fields")
 return WorkflowSpec(ModelSpec(**raw["model"]),StepSpec(**raw["step"]),raw["units"])
