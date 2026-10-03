"""Deterministic multi-step execution over qualified TensorFEM kernels."""
from __future__ import annotations
from dataclasses import dataclass,field
import dataclasses
import hashlib
import json
from pathlib import Path
import time
from typing import Callable,Mapping
import torch


@dataclass(frozen=True)
class StepSpec:
    name: str
    kind: str
    inputs: Mapping[str,object]
    dependencies: Mapping[str,str]=field(default_factory=dict)


@dataclass(frozen=True)
class StepResult:
    name: str
    kind: str
    fields: Mapping[str,object]
    elapsed_seconds: float
    success: bool
    diagnostic: str=""


@dataclass
class StepContext:
    results: dict[str,StepResult]=field(default_factory=dict)

    def field(self,reference):
        try:step,name=reference.split(".",1);result=self.results[step]
        except (ValueError,KeyError) as exc:raise KeyError(f"unknown step dependency {reference}") from exc
        if not result.success or name not in result.fields:raise KeyError(f"unknown step dependency {reference}")
        return result.fields[name]


@dataclass(frozen=True)
class ExecutionResult:
    steps: tuple[StepResult,...]
    completed: bool
    resumed_steps: int


def _linear(inputs):
    from .solvers import solve_linear_static
    r=solve_linear_static(inputs["model"])
    return {k:getattr(r,k) for k in ("displacement","reaction","axial_strain","axial_stress","axial_force","strain_energy")}


def _thermal(inputs):
    from .thermal import solve_steady_thermal
    return {"temperature":solve_steady_thermal(inputs["model"])}


def _modal(inputs):
    from .modal import solve_modes
    r=solve_modes(inputs["stiffness"],inputs["mass"],inputs["constrained_dofs"],inputs.get("modes",6))
    return {"angular_frequencies":r.angular_frequencies,"frequencies_hz":r.frequencies_hz,"modes":r.modes}


def _thermoelastic(inputs):
    from .thermal import solve_thermoelastic_bar
    allowed={"nodes","elements","young_modulus","area","expansion_coefficient","temperature",
             "reference_temperature","fixed_nodes"}
    extra=set(inputs)-allowed
    if extra:raise ValueError(f"unsupported thermoelastic inputs {sorted(extra)}")
    r=solve_thermoelastic_bar(**inputs)
    return {"displacement":r.displacement,"reaction":r.reaction,"axial_force":r.axial_force}


KERNELS={"linear_static":_linear,"thermal_steady":_thermal,"modal":_modal,
         "thermoelastic_bar":_thermoelastic}


def _fingerprint(value):
    if isinstance(value,torch.Tensor):
        v=value.detach().cpu().contiguous();raw=bytes(v.reshape(-1).view(torch.uint8).tolist());return {"tensor_sha256":hashlib.sha256(raw).hexdigest(),
            "dtype":str(v.dtype),"shape":list(v.shape)}
    if dataclasses.is_dataclass(value):return {f.name:_fingerprint(getattr(value,f.name)) for f in dataclasses.fields(value)}
    if isinstance(value,Mapping):return {str(k):_fingerprint(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    if isinstance(value,(tuple,list)):return [_fingerprint(x) for x in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return {"type":f"{type(value).__module__}.{type(value).__qualname__}"}


def _plan_hash(steps):
    plan=[{"name":s.name,"kind":s.kind,"inputs":_fingerprint(s.inputs),
           "dependencies":dict(sorted(s.dependencies.items()))} for s in steps]
    return hashlib.sha256(json.dumps(plan,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def _encode(value):
    if isinstance(value,torch.Tensor):return {"tensor":value.detach().cpu().tolist(),"dtype":str(value.dtype).split(".")[-1]}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if isinstance(value,dict):return {k:_encode(v) for k,v in value.items()}
    raise TypeError(f"checkpoint cannot encode {type(value).__name__}")


def _decode(value):
    if isinstance(value,dict) and "tensor" in value:return torch.tensor(value["tensor"],dtype=getattr(torch,value["dtype"]))
    if isinstance(value,dict):return {k:_decode(v) for k,v in value.items()}
    return value


def _checkpoint(path,plan_hash,results):
    payload={"schema":"tensorfem.steps.v1","plan_hash":plan_hash,"steps":[{"name":r.name,"kind":r.kind,
        "fields":_encode(dict(r.fields)),"elapsed_seconds":r.elapsed_seconds,"success":r.success,
        "diagnostic":r.diagnostic} for r in results]}
    Path(path).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")


class StepExecutor:
    def __init__(self,kernels=None):self.kernels=dict(KERNELS if kernels is None else kernels)

    def execute(self,steps,*,checkpoint=None,resume=False,
                progress: Callable[[dict],None]|None=None):
        steps=tuple(steps);names=[s.name for s in steps]
        if len(names)!=len(set(names)) or any(not x for x in names):raise ValueError("step names must be unique and nonempty")
        signature=_plan_hash(steps);ctx=StepContext();resumed=0
        if resume:
            if checkpoint is None or not Path(checkpoint).exists():raise ValueError("resume checkpoint does not exist")
            saved=json.loads(Path(checkpoint).read_text(encoding="utf-8"))
            if saved.get("schema")!="tensorfem.steps.v1" or saved.get("plan_hash")!=signature:
                raise ValueError("checkpoint plan mismatch")
            for position,raw in enumerate(saved["steps"]):
                if position>=len(steps) or raw["name"]!=steps[position].name or raw["kind"]!=steps[position].kind:
                    raise ValueError("checkpoint step sequence mismatch")
                result=StepResult(raw["name"],raw["kind"],_decode(raw["fields"]),raw["elapsed_seconds"],raw["success"],raw["diagnostic"])
                if not result.success:break
                ctx.results[result.name]=result;resumed+=1
        results=list(ctx.results.values())
        for index,spec in enumerate(steps):
            if spec.name in ctx.results:continue
            if spec.kind not in self.kernels:
                result=StepResult(spec.name,spec.kind,{},0.,False,f"unsupported step kind {spec.kind}")
                results.append(result);return ExecutionResult(tuple(results),False,resumed)
            inputs=dict(spec.inputs)
            start=None
            try:
                for target,source in spec.dependencies.items():
                    if target in inputs:raise ValueError(f"dependency overwrites input {target}")
                    inputs[target]=ctx.field(source)
                if progress:progress({"event":"start","step":spec.name,"index":index,"total":len(steps)})
                start=time.perf_counter();fields=self.kernels[spec.kind](inputs);elapsed=time.perf_counter()-start
                result=StepResult(spec.name,spec.kind,fields,elapsed,True)
            except Exception as exc:
                elapsed=time.perf_counter()-start if start is not None else 0.
                result=StepResult(spec.name,spec.kind,{},elapsed,False,f"{type(exc).__name__}: {exc}")
                results.append(result)
                if progress:progress({"event":"failed","step":spec.name,"diagnostic":result.diagnostic})
                return ExecutionResult(tuple(results),False,resumed)
            ctx.results[spec.name]=result;results.append(result)
            if checkpoint:_checkpoint(checkpoint,signature,results)
            if progress:progress({"event":"complete","step":spec.name,"elapsed_seconds":elapsed,
                                  "completed":len(results),"total":len(steps)})
        return ExecutionResult(tuple(results),True,resumed)


def execution_result_db(execution):
    """Create a compact ResultDB manifest from multi-step scalar/vector outputs."""
    from .result_db import ResultDB
    histories={};shapes={}
    for step in execution.steps:
        for name,value in step.fields.items():
            if isinstance(value,torch.Tensor):
                shapes[f"{step.name}.{name}"]=list(value.shape)
                if value.ndim<=1:histories[f"{step.name}.{name}"]=value.reshape(-1)
    timings=torch.tensor([s.elapsed_seconds for s in execution.steps],dtype=torch.float64)
    histories["step_elapsed_seconds"]=timings
    return ResultDB(torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long),
        torch.empty((0,0),dtype=torch.long),histories=histories,
        job_metadata={"completed":execution.completed,"steps":[{"name":s.name,"kind":s.kind,
            "success":s.success,"diagnostic":s.diagnostic} for s in execution.steps],"field_shapes":shapes},
        units={"step_elapsed_seconds":"s"})
