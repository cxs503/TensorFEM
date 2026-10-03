"""Multi-step/frame ResultDB v2 with atomic optional HDF5 storage."""
from __future__ import annotations
from dataclasses import asdict,dataclass,field
from pathlib import Path
from typing import Mapping
import hashlib,json,math,os
import torch

SCHEMA="tensorfem.result-db.v2"
LOCATIONS=("node","element","integration_point")


def _hash_tensor(value):
    x=value.detach().cpu().contiguous();h=hashlib.sha256();h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes());return h.hexdigest()
def _canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),allow_nan=False)


@dataclass(frozen=True)
class FieldSpec:
    location: str
    unit: str
    components: tuple[str,...]


@dataclass(frozen=True)
class ResultFrame:
    index: int
    time: float
    load_factor: float
    fields: Mapping[str,Mapping[str,torch.Tensor]]=field(default_factory=dict)


@dataclass(frozen=True)
class ResultStep:
    name: str
    frames: tuple[ResultFrame,...]


@dataclass(frozen=True)
class HistorySeries:
    step: str
    time: torch.Tensor
    load_factor: torch.Tensor
    values: torch.Tensor
    unit: str
    components: tuple[str,...]


@dataclass(frozen=True)
class ResultDBv2:
    node_ids: torch.Tensor
    element_ids: torch.Tensor
    connectivity: torch.Tensor
    field_specs: Mapping[str,FieldSpec]
    steps: tuple[ResultStep,...]
    histories: Mapping[str,HistorySeries]=field(default_factory=dict)
    model_metadata: Mapping[str,object]=field(default_factory=dict)
    job_metadata: Mapping[str,object]=field(default_factory=dict)
    coordinate_system: str="global Cartesian"

    def validate(self):
        if self.node_ids.ndim!=1 or self.element_ids.ndim!=1 or self.connectivity.ndim!=2:raise ValueError("invalid v2 topology")
        if len(torch.unique(self.node_ids))!=len(self.node_ids) or len(torch.unique(self.element_ids))!=len(self.element_ids):raise ValueError("duplicate IDs")
        if len(self.connectivity)!=len(self.element_ids):raise ValueError("element/connectivity mismatch")
        if not set(self.connectivity.flatten().tolist())<=set(self.node_ids.tolist()):raise ValueError("unknown connectivity node")
        names=set()
        for key,spec in self.field_specs.items():
            if not key or spec.location not in LOCATIONS or not spec.unit or not spec.components:raise ValueError("invalid field specification")
        previous_steps=set()
        for step in self.steps:
            if not step.name or step.name in previous_steps or not step.frames:raise ValueError("invalid or duplicate step")
            previous_steps.add(step.name);last_index=-1;last_time=-math.inf
            for frame in step.frames:
                if frame.index<=last_index or frame.time<last_time or not math.isfinite(frame.time) or not math.isfinite(frame.load_factor):raise ValueError("frames must increase in index/time")
                last_index,last_time=frame.index,frame.time
                for location,fields in frame.fields.items():
                    if location not in LOCATIONS:raise ValueError("unknown field location")
                    expected=len(self.node_ids) if location=="node" else len(self.element_ids)
                    for name,value in fields.items():
                        spec=self.field_specs.get(name)
                        if spec is None or spec.location!=location or value.ndim<1 or len(value)!=expected:raise ValueError(f"invalid frame field {name}")
                        ncomp=1 if value.ndim==1 else value.shape[-1]
                        if ncomp!=len(spec.components) or not bool(torch.all(torch.isfinite(value))):raise ValueError(f"invalid components in {name}")
        for name,h in self.histories.items():
            if h.step not in previous_steps or h.time.ndim!=1 or h.load_factor.shape!=h.time.shape or len(h.values)!=len(h.time):raise ValueError(f"invalid history {name}")
            ncomp=1 if h.values.ndim==1 else h.values.shape[-1]
            if not name or not h.unit or ncomp!=len(h.components) or not bool(torch.all(torch.isfinite(h.values))):raise ValueError(f"invalid history {name}")
        return self

    def tensors(self):
        out={"topology/node_ids":self.node_ids,"topology/element_ids":self.element_ids,"topology/connectivity":self.connectivity}
        for step in self.steps:
            for frame in step.frames:
                for location,fields in frame.fields.items():
                    for name,value in fields.items():out[f"steps/{step.name}/frames/{frame.index:06d}/{location}/{name}"]=value
        for name,h in self.histories.items():
            out[f"histories/{name}/time"]=h.time;out[f"histories/{name}/load_factor"]=h.load_factor;out[f"histories/{name}/values"]=h.values
        return out

    def checksums(self):
        self.validate();hashes={k:_hash_tensor(v) for k,v in sorted(self.tensors().items())}
        meta={"schema":SCHEMA,"field_specs":{k:asdict(v) for k,v in sorted(self.field_specs.items())},
              "steps":[{"name":s.name,"frames":[{"index":f.index,"time":f.time,"load_factor":f.load_factor} for f in s.frames]} for s in self.steps],
              "histories":{k:{"step":v.step,"unit":v.unit,"components":v.components} for k,v in sorted(self.histories.items())},
              "model_metadata":self.model_metadata,"job_metadata":self.job_metadata,"coordinate_system":self.coordinate_system}
        hashes["metadata"]=hashlib.sha256(_canonical(meta).encode()).hexdigest();hashes["database"]=hashlib.sha256(_canonical(hashes).encode()).hexdigest();return hashes


def _inventory(db):
    checks=db.checksums();out={}
    for path,tensor in sorted(db.tensors().items()):
        item={"path":path,"shape":list(tensor.shape),"dtype":str(tensor.dtype).split(".")[-1],"checksum":checks[path]}
        parts=path.split("/")
        if parts[0]=="steps":
            spec=db.field_specs[parts[-1]];item.update({"unit":spec.unit,"components":list(spec.components),"location":spec.location})
        elif parts[0]=="histories" and parts[-1]=="values":
            h=db.histories[parts[1]];item.update({"unit":h.unit,"components":list(h.components),"location":"history"})
        out[path]=item
    return out


def write_result_db_v2(path,db:ResultDBv2,*,chunk_rows=1024):
    """Atomically publish JSON last; incomplete temporary bodies are ignored."""
    if chunk_rows<1:raise ValueError("chunk_rows must be positive")
    db.validate();base=Path(path);summary=base.with_suffix(".json");checks=db.checksums();tag=checks["database"][:16]
    body=base.with_name(base.name+f".{tag}.h5");temp=body.with_suffix(body.suffix+".tmp")
    try:import h5py
    except ImportError:h5py=None
    available=h5py is not None
    if available:
        try:
            with h5py.File(temp,"w") as h5:
                h5.attrs["schema"]=SCHEMA;h5.attrs["complete"]=False;h5.attrs["database_checksum"]=checks["database"]
                for key,value in db.tensors().items():
                    x=value.detach().cpu().numpy();chunks=None if x.ndim==0 or len(x)==0 else (min(chunk_rows,len(x)),*x.shape[1:])
                    h5.create_dataset(key,data=x,chunks=chunks)
                h5.flush();h5.attrs.modify("complete",True);h5.flush()
            os.replace(temp,body)
        finally:
            if temp.exists():temp.unlink()
    payload={"schema":SCHEMA,"complete":True,"body":{"available":available,"format":"hdf5","file":body.name if available else None},
        "counts":{"nodes":len(db.node_ids),"elements":len(db.element_ids),"steps":len(db.steps),"frames":sum(len(s.frames) for s in db.steps)},
        "field_specs":{k:asdict(v) for k,v in sorted(db.field_specs.items())},
        "steps":[{"name":s.name,"frames":[{"index":f.index,"time":f.time,"load_factor":f.load_factor} for f in s.frames]} for s in db.steps],
        "histories":{k:{"step":v.step,"unit":v.unit,"components":list(v.components)} for k,v in sorted(db.histories.items())},
        "inventory":_inventory(db),"checksums":checks,"model_metadata":db.model_metadata,"job_metadata":db.job_metadata,
        "coordinate_system":db.coordinate_system}
    tmp_summary=summary.with_suffix(summary.suffix+".tmp")
    tmp_summary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8");os.replace(tmp_summary,summary)
    return available


def _summary(path):
    summary=json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"))
    if summary.get("schema")!=SCHEMA or summary.get("complete") is not True:raise ValueError("incomplete or invalid ResultDB v2 summary")
    return summary


def _h5(path,summary):
    if not summary["body"]["available"]:raise RuntimeError("ResultDB v2 HDF5 body unavailable; install optional h5py")
    try:import h5py
    except ImportError as exc:raise RuntimeError("reading ResultDB v2 requires optional h5py") from exc
    body=Path(path).with_suffix(".json").parent/summary["body"]["file"]
    h5=h5py.File(body,"r")
    if h5.attrs.get("schema")!=SCHEMA or not bool(h5.attrs.get("complete",False)):
        h5.close();raise RuntimeError("ResultDB v2 body is incomplete")
    return h5


def read_result_db_v2(path,*,verify=True):
    s=_summary(path);h5=_h5(path,s)
    try:
        get=lambda key:torch.from_numpy(h5[key][...])
        specs={k:FieldSpec(**v) for k,v in s["field_specs"].items()};steps=[]
        for st in s["steps"]:
            frames=[]
            for fr in st["frames"]:
                fields={loc:{} for loc in LOCATIONS};prefix=f"steps/{st['name']}/frames/{fr['index']:06d}"
                for name,spec in specs.items():
                    key=f"{prefix}/{spec.location}/{name}"
                    if key in h5:fields[spec.location][name]=get(key)
                frames.append(ResultFrame(fr["index"],fr["time"],fr["load_factor"],{k:v for k,v in fields.items() if v}))
            steps.append(ResultStep(st["name"],tuple(frames)))
        histories={}
        for name,meta in s["histories"].items():
            histories[name]=HistorySeries(meta["step"],get(f"histories/{name}/time"),get(f"histories/{name}/load_factor"),
                                          get(f"histories/{name}/values"),meta["unit"],tuple(meta["components"]))
        db=ResultDBv2(get("topology/node_ids"),get("topology/element_ids"),get("topology/connectivity"),specs,tuple(steps),histories,
                      s["model_metadata"],s["job_metadata"],s["coordinate_system"])
    finally:h5.close()
    if verify and db.checksums()!=s["checksums"]:raise RuntimeError("ResultDB v2 checksum mismatch")
    return db


def query_frame_field(path,step,frame,location,field,*,ids=None,components=None):
    """Lazy HDF5 subset query without loading other frames or fields."""
    s=_summary(path);spec=s["field_specs"].get(field)
    if spec is None or spec["location"]!=location:raise KeyError(field)
    frames=next((x["frames"] for x in s["steps"] if x["name"]==step),None)
    if frames is None or not any(x["index"]==frame for x in frames):raise KeyError((step,frame))
    h5=_h5(path,s)
    try:
        dataset=h5[f"steps/{step}/frames/{frame:06d}/{location}/{field}"]
        idpath="topology/node_ids" if location=="node" else "topology/element_ids";all_ids=h5[idpath][...]
        positions=list(range(len(all_ids))) if ids is None else [dict((int(v),i) for i,v in enumerate(all_ids.tolist()))[int(v)] for v in ids]
        unique=sorted(set(positions));values=dataset[unique];reverse={p:i for i,p in enumerate(unique)};values=values[[reverse[p] for p in positions]]
        names=spec["components"]
        if components is not None:
            columns=[names.index(c) for c in components];values=values[...,columns]
        return torch.tensor([all_ids[p] for p in positions],dtype=torch.from_numpy(all_ids).dtype),torch.from_numpy(values)
    finally:h5.close()


def query_history_series(path,name,*,start=None,stop=None):
    s=_summary(path);h5=_h5(path,s)
    try:
        sl=slice(start,stop);prefix=f"histories/{name}"
        return tuple(torch.from_numpy(h5[f"{prefix}/{key}"][sl]) for key in ("time","load_factor","values"))
    finally:h5.close()


def migrate_v1(v1):
    """Explicitly map an in-memory ResultDB v1 to one Legacy step/frame."""
    from .result_db import ResultDB
    if not isinstance(v1,ResultDB):raise TypeError("migrate_v1 expects ResultDB v1")
    v1.validate();specs={}
    for location,fields in (("node",v1.node_fields),("element",v1.element_fields)):
        for name,value in fields.items():
            ncomp=1 if value.ndim==1 else value.shape[-1]
            components=(name,) if ncomp==1 else tuple(f"c{i}" for i in range(ncomp))
            specs[name]=FieldSpec(location,v1.units.get(name,"dimensionless"),components)
    frame=ResultFrame(0,0.,1.,{"node":dict(v1.node_fields),"element":dict(v1.element_fields)})
    histories={}
    for name,value in v1.histories.items():
        n=len(value);values=value[:,None];histories[name]=HistorySeries("Legacy",torch.arange(n,dtype=torch.float64),torch.ones(n,dtype=torch.float64),
            values,v1.units.get(name,"dimensionless"),(name,))
    return ResultDBv2(v1.node_ids,v1.element_ids,v1.connectivity,specs,(ResultStep("Legacy",(frame,)),),histories,
                      v1.model_metadata,v1.job_metadata,v1.coordinate_system).validate()


def read_result_db_compatible(path,*,migrate=False,verify=True):
    """Read v1 with its original reader or v2; optionally migrate v1 in memory."""
    summary=json.loads(Path(path).with_suffix(".json").read_text(encoding="utf-8"));schema=summary.get("schema")
    if schema==SCHEMA:return read_result_db_v2(path,verify=verify)
    if schema=="tensorfem.result-db.v1":
        from .result_db import read_result_db
        old=read_result_db(path,verify=verify)
        return migrate_v1(old) if migrate else old
    raise ValueError("unsupported ResultDB schema")
