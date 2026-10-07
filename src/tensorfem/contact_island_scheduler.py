"""Deterministic worker scheduling primitives for independent contact islands."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import dataclass
import json
import multiprocessing as mp
from typing import Any,Callable
import torch


@dataclass(frozen=True)
class IslandJob:
    island_id: int
    body_indices: torch.Tensor
    contact_indices: torch.Tensor
    cost: float
    payload: Any=None


@dataclass(frozen=True)
class IslandExecution:
    island_id: int
    worker_id: int
    value: Any


class IslandWorkerError(RuntimeError):
    """A worker failure annotated with its deterministic island identifier."""


def colored_island_jobs(data) -> tuple[IslandJob,...]:
    """Convert colored-contact connectivity to disjoint, ordered island jobs."""
    jobs=[];covered=[]
    for island_id,bodies_tuple in enumerate(data.islands):
        bodies=torch.tensor(bodies_tuple,dtype=torch.long,device=data.body_a.device)
        mask=torch.zeros(len(data.keys),dtype=torch.bool,device=data.body_a.device)
        for body in bodies_tuple:
            mask|=(data.body_a==body)|(data.body_b==body)
        contacts=torch.nonzero(mask).flatten();covered.extend(contacts.tolist())
        jobs.append(IslandJob(island_id,bodies,contacts,float(max(len(contacts),1))))
    if sorted(covered)!=list(range(len(data.keys))):
        raise ValueError("contact island partition loses or duplicates constraints")
    return tuple(jobs)


def deterministic_partitions(jobs,workers: int) -> tuple[tuple[IslandJob,...],...]:
    """Stable longest-processing-time partition with deterministic tie breaks."""
    if workers<1:raise ValueError("workers must be positive")
    ids=[j.island_id for j in jobs]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate island id")
    bins=[[] for _ in range(workers)];loads=[0.]*workers
    for job in sorted(jobs,key=lambda j:(-j.cost,j.island_id)):
        target=min(range(workers),key=lambda i:(loads[i],i))
        bins[target].append(job);loads[target]+=job.cost
    return tuple(tuple(sorted(group,key=lambda j:j.island_id)) for group in bins)


def _run_batch(worker_id,jobs,fn):
    out=[]
    for job in jobs:
        try:value=fn(job)
        except Exception as exc:
            raise IslandWorkerError(f"island {job.island_id} failed: {exc}") from exc
        out.append(IslandExecution(job.island_id,worker_id,value))
    return out


def execute_island_jobs(jobs,worker: Callable[[IslandJob],Any],*,workers=1,
                        backend="serial") -> tuple[IslandExecution,...]:
    """Execute independent islands and merge strictly by island id.

    ``process`` is an optional single-host CPU backend.  CUDA tensors are
    rejected rather than silently copied between processes.
    """
    jobs=tuple(jobs);partitions=deterministic_partitions(jobs,workers)
    if backend not in ("serial","process"):raise ValueError("backend must be serial or process")
    if backend=="process" and any(any(t.device.type!="cpu" for t in
            (j.body_indices,j.contact_indices)) for j in jobs):
        raise ValueError("process backend accepts CPU island metadata only")
    if backend=="serial" or workers==1:
        result=[]
        for wid,part in enumerate(partitions):result.extend(_run_batch(wid,part,worker))
    else:
        result=[]
        context=mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers,mp_context=context) as pool:
            futures=[pool.submit(_run_batch,wid,part,worker) for wid,part in enumerate(partitions) if part]
            try:
                for future in as_completed(futures):result.extend(future.result())
            except Exception:
                for future in futures:future.cancel()
                raise
    result.sort(key=lambda x:x.island_id)
    if [x.island_id for x in result]!=sorted(j.island_id for j in jobs):
        raise RuntimeError("island result merge loses or duplicates results")
    return tuple(result)


def _encode(value):
    if isinstance(value,torch.Tensor):
        return {"__tensor__":value.detach().cpu().tolist(),"dtype":str(value.dtype).split(".")[-1]}
    if isinstance(value,tuple):return {"__tuple__":[_encode(x) for x in value]}
    if isinstance(value,list):return [_encode(x) for x in value]
    if isinstance(value,dict):return {str(k):_encode(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError(f"unsupported island payload type: {type(value).__name__}")


def _decode(value):
    if isinstance(value,dict) and "__tensor__" in value:
        dtype=getattr(torch,value["dtype"],None)
        if dtype is None:raise ValueError("invalid tensor dtype")
        return torch.tensor(value["__tensor__"],dtype=dtype)
    if isinstance(value,dict) and "__tuple__" in value:return tuple(_decode(x) for x in value["__tuple__"])
    if isinstance(value,list):return [_decode(x) for x in value]
    if isinstance(value,dict):return {k:_decode(v) for k,v in value.items()}
    return value


def serialize_jobs(jobs) -> bytes:
    """Serialize scheduling input as a versioned, non-executable JSON archive."""
    records=[{"island_id":j.island_id,"body_indices":_encode(j.body_indices),
              "contact_indices":_encode(j.contact_indices),"cost":j.cost,
              "payload":_encode(j.payload)} for j in jobs]
    return json.dumps({"schema":"tensorfem.contact-islands.v1","jobs":records},
                      sort_keys=True,separators=(",",":")).encode()


def deserialize_jobs(blob: bytes) -> tuple[IslandJob,...]:
    payload=json.loads(blob)
    if payload.get("schema")!="tensorfem.contact-islands.v1":raise ValueError("invalid island archive")
    return tuple(IslandJob(int(x["island_id"]),_decode(x["body_indices"]).long(),
        _decode(x["contact_indices"]).long(),float(x["cost"]),_decode(x.get("payload")))
        for x in payload["jobs"])


def serialize_executions(results) -> bytes:
    records=[{"island_id":x.island_id,"worker_id":x.worker_id,"value":_encode(x.value)}
             for x in results]
    return json.dumps({"schema":"tensorfem.contact-results.v1","results":records},
                      sort_keys=True,separators=(",",":")).encode()


def deserialize_executions(blob: bytes) -> tuple[IslandExecution,...]:
    payload=json.loads(blob)
    if payload.get("schema")!="tensorfem.contact-results.v1":raise ValueError("invalid result archive")
    return tuple(IslandExecution(int(x["island_id"]),int(x["worker_id"]),_decode(x["value"]))
                 for x in payload["results"])


def available_execution_devices(requested: int) -> tuple[str,...]:
    """Deterministic CUDA assignment, with an explicit CPU fallback."""
    if requested<1:raise ValueError("requested devices must be positive")
    count=torch.cuda.device_count() if torch.cuda.is_available() else 0
    return tuple(f"cuda:{i}" for i in range(min(requested,count))) or ("cpu",)
