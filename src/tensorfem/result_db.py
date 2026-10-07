"""Versioned result database with JSON index and optional HDF5 field body."""
from __future__ import annotations
from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path
from typing import Mapping
import torch

SCHEMA="tensorfem.result-db.v1"


def _tensor_hash(value):
    data=value.detach().cpu().contiguous()
    h=hashlib.sha256();h.update(str(data.dtype).encode());h.update(str(tuple(data.shape)).encode())
    # ResultDB JSON summaries remain usable without the optional NumPy/HDF5
    # stack.  Hash the tensor storage through a byte view instead of crossing
    # the NumPy bridge, which PyTorch may not have been built/configured with.
    raw=data.reshape(-1).view(torch.uint8).tolist()
    h.update(bytes(raw));return h.hexdigest()


def _canonical(value):return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False)


@dataclass(frozen=True)
class ResultDB:
    node_ids: torch.Tensor
    element_ids: torch.Tensor
    connectivity: torch.Tensor
    node_fields: Mapping[str,torch.Tensor]=field(default_factory=dict)
    element_fields: Mapping[str,torch.Tensor]=field(default_factory=dict)
    histories: Mapping[str,torch.Tensor]=field(default_factory=dict)
    probes: Mapping[str,dict]=field(default_factory=dict)
    model_metadata: Mapping[str,object]=field(default_factory=dict)
    job_metadata: Mapping[str,object]=field(default_factory=dict)
    units: Mapping[str,str]=field(default_factory=dict)
    coordinate_system: str="global Cartesian"

    def validate(self):
        if self.node_ids.ndim!=1 or self.element_ids.ndim!=1 or self.connectivity.ndim!=2:
            raise ValueError("invalid ResultDB topology shapes")
        if len(torch.unique(self.node_ids))!=len(self.node_ids) or len(torch.unique(self.element_ids))!=len(self.element_ids):
            raise ValueError("duplicate model identifiers")
        if len(self.connectivity)!=len(self.element_ids):raise ValueError("element/connectivity size mismatch")
        known=set(self.node_ids.tolist())
        if any(int(i) not in known for i in self.connectivity.reshape(-1).tolist()):
            raise ValueError("connectivity references unknown node")
        for name,value in self.node_fields.items():
            if not name or value.ndim<1 or len(value)!=len(self.node_ids):raise ValueError(f"invalid node field {name}")
            if not bool(torch.all(torch.isfinite(value))):raise ValueError(f"non-finite node field {name}")
        for name,value in self.element_fields.items():
            if not name or value.ndim<1 or len(value)!=len(self.element_ids):raise ValueError(f"invalid element field {name}")
        for name,value in self.histories.items():
            if not name or value.ndim!=1 or not bool(torch.all(torch.isfinite(value))):raise ValueError(f"invalid history {name}")

    def query_nodes(self,field,node_ids=None):
        self.validate();value=self.node_fields[field]
        if node_ids is None:return self.node_ids.clone(),value.clone()
        lookup={int(n):i for i,n in enumerate(self.node_ids.tolist())}
        try:index=torch.tensor([lookup[int(n)] for n in node_ids],dtype=torch.long,device=value.device)
        except KeyError as exc:raise KeyError(f"unknown node {exc.args[0]}") from exc
        return self.node_ids[index.cpu()],value[index]

    def checksums(self):
        self.validate();groups={"topology/node_ids":self.node_ids,"topology/element_ids":self.element_ids,
            "topology/connectivity":self.connectivity}
        groups.update({f"node_fields/{k}":v for k,v in self.node_fields.items()})
        groups.update({f"element_fields/{k}":v for k,v in self.element_fields.items()})
        groups.update({f"histories/{k}":v for k,v in self.histories.items()})
        hashes={k:_tensor_hash(v) for k,v in sorted(groups.items())}
        meta={"schema":SCHEMA,"model_metadata":self.model_metadata,"job_metadata":self.job_metadata,
              "probes":self.probes,"units":self.units,"coordinate_system":self.coordinate_system}
        hashes["metadata"]=hashlib.sha256(_canonical(meta).encode()).hexdigest()
        hashes["database"]=hashlib.sha256(_canonical(hashes).encode()).hexdigest()
        return hashes


def _inventory(fields):
    return {k:{"shape":list(v.shape),"dtype":str(v.dtype).split(".")[-1]} for k,v in sorted(fields.items())}


def write_result_db(path,result: ResultDB,*,chunk_rows=1024):
    """Write summary and optional HDF5 body; return whether the body was written."""
    if chunk_rows<1:raise ValueError("chunk_rows must be positive")
    result.validate();base=Path(path);summary=base.with_suffix(".json");body=base.with_suffix(".h5")
    try:import h5py
    except ImportError:h5py=None
    available=h5py is not None
    if available:
        with h5py.File(body,"w") as h5:
            def put(group,name,tensor):
                a=tensor.detach().cpu().numpy();chunks=None if a.ndim==0 or len(a)==0 else (min(chunk_rows,len(a)),*a.shape[1:])
                group.create_dataset(name,data=a,chunks=chunks)
            top=h5.create_group("topology");put(top,"node_ids",result.node_ids);put(top,"element_ids",result.element_ids);put(top,"connectivity",result.connectivity)
            for group_name,fields in (("node_fields",result.node_fields),("element_fields",result.element_fields),("histories",result.histories)):
                group=h5.create_group(group_name)
                for name,value in fields.items():put(group,name,value)
            h5.attrs["schema"]=SCHEMA
    payload={"schema":SCHEMA,"body":{"file":body.name,"format":"hdf5","available":available},
        "model_metadata":result.model_metadata,"job_metadata":result.job_metadata,"probes":result.probes,
        "units":result.units,"coordinate_system":result.coordinate_system,
        "counts":{"nodes":len(result.node_ids),"elements":len(result.element_ids)},
        "fields":{"node":_inventory(result.node_fields),"element":_inventory(result.element_fields),
                  "history":_inventory(result.histories)},"checksums":result.checksums()}
    summary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return available


def read_result_db(path,*,verify=True):
    base=Path(path);summary=json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    if summary.get("schema")!=SCHEMA:raise ValueError("invalid ResultDB schema")
    if not summary["body"]["available"]:raise RuntimeError("ResultDB HDF5 body unavailable; install optional h5py")
    try:import h5py
    except ImportError as exc:raise RuntimeError("reading ResultDB requires optional h5py") from exc
    body=base.with_suffix(".json").parent/summary["body"]["file"]
    with h5py.File(body,"r") as h5:
        get=lambda name:torch.from_numpy(h5[name][...])
        result=ResultDB(get("topology/node_ids"),get("topology/element_ids"),get("topology/connectivity"),
            {k:get(f"node_fields/{k}") for k in h5["node_fields"]},
            {k:get(f"element_fields/{k}") for k in h5["element_fields"]},
            {k:get(f"histories/{k}") for k in h5["histories"]},summary["probes"],
            summary["model_metadata"],summary["job_metadata"],summary["units"],summary["coordinate_system"])
    if verify and result.checksums()!=summary["checksums"]:raise RuntimeError("ResultDB checksum mismatch")
    return result


def query_result_nodes(path,field,node_ids=None):
    """Read only a node-field subset from an HDF5 body, preserving ID order."""
    base=Path(path);summary=json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    if summary.get("schema")!=SCHEMA:raise ValueError("invalid ResultDB schema")
    try:import h5py
    except ImportError as exc:raise RuntimeError("node subset query requires optional h5py") from exc
    body=base.with_suffix(".json").parent/summary["body"]["file"]
    with h5py.File(body,"r") as h5:
        ids=h5["topology/node_ids"][...];dataset=h5[f"node_fields/{field}"]
        if node_ids is None:return torch.from_numpy(ids),torch.from_numpy(dataset[...])
        lookup={int(n):i for i,n in enumerate(ids.tolist())}
        try:positions=[lookup[int(n)] for n in node_ids]
        except KeyError as exc:raise KeyError(f"unknown node {exc.args[0]}") from exc
        # h5py fancy indices must increase; unique sorted read plus inverse restores order/repeats.
        unique=sorted(set(positions));values=dataset[unique];reverse={p:i for i,p in enumerate(unique)}
        selected=values[[reverse[p] for p in positions]]
        return torch.tensor([ids[p] for p in positions],dtype=torch.from_numpy(ids).dtype),torch.from_numpy(selected)


def iter_result_node_chunks(path,field,*,chunk_rows=1024):
    """Yield bounded CPU chunks `(node_ids, values)` from an HDF5 node field."""
    if chunk_rows<1:raise ValueError("chunk_rows must be positive")
    base=Path(path);summary=json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    try:import h5py
    except ImportError as exc:raise RuntimeError("chunked ResultDB reading requires optional h5py") from exc
    body=base.with_suffix(".json").parent/summary["body"]["file"]
    with h5py.File(body,"r") as h5:
        ids=h5["topology/node_ids"];values=h5[f"node_fields/{field}"]
        for start in range(0,len(ids),chunk_rows):
            stop=min(start+chunk_rows,len(ids))
            yield torch.from_numpy(ids[start:stop]),torch.from_numpy(values[start:stop])


def hemisphere_result_db(post):
    """Adapt qualified hemisphere post-processing fields without recomputation."""
    n=len(post.nodes);e=len(post.elements);v=post.validation
    return ResultDB(torch.arange(n,dtype=torch.long),torch.arange(e,dtype=torch.long),post.elements.clone(),
        {"coordinates":post.nodes,"displacement":post.displacement,"rotation":post.rotation,
         "deformed_coordinates":post.deformed,"reaction_force":post.reaction_force,
         "reaction_moment":post.reaction_moment,"applied_force":post.applied_force}, {},
        {k:torch.tensor([float(v[k])],dtype=post.nodes.dtype) for k in
         ("computed_displacement","relative_error","free_residual_norm","normalized_force_balance_error","normalized_moment_balance_error")},
        {post.probe["name"]:post.probe},post.metadata,{"validation":v},
        {"coordinates":post.metadata["units"]["length"],"displacement":post.metadata["units"]["length"],
         "reaction_force":post.metadata["units"]["force"],"rotation":"radian"},post.metadata["coordinate_system"])
