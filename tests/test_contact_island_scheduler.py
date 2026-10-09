import time
import pytest
import torch
from tensorfem.colored_contact_graph import prepare_colored_contacts
from tensorfem.contact_island_scheduler import (
    IslandJob,IslandWorkerError,available_execution_devices,colored_island_jobs,
    deserialize_executions,deserialize_jobs,deterministic_partitions,execute_island_jobs,
    serialize_executions,serialize_jobs,
)
from tensorfem.rigid_contact_graph import RigidContact

D=torch.float64


def checksum(job):
    return int(job.payload.sum())+job.island_id


def fail_on_seven(job):
    if job.island_id==7:raise ArithmeticError("deliberate")
    return job.island_id


def jobs(n):
    return tuple(IslandJob(i,torch.tensor([i]),torch.tensor([i]),float(i%7+1),
                           torch.tensor([i,2*i])) for i in range(n))


def test_one_two_four_process_workers_are_bitwise_identical():
    work=jobs(28)
    baseline=execute_island_jobs(work,checksum,workers=1)
    expected=[x.value for x in baseline]
    for count in (2,4):
        actual=execute_island_jobs(work,checksum,workers=count,backend="process")
        assert [x.island_id for x in actual]==list(range(len(work)))
        assert [x.value for x in actual]==expected


def test_partition_is_complete_balanced_and_deterministic():
    work=jobs(1003);start=time.perf_counter()
    a=deterministic_partitions(work,4);b=deterministic_partitions(tuple(reversed(work)),4)
    elapsed=time.perf_counter()-start
    assert [[x.island_id for x in p] for p in a]==[[x.island_id for x in p] for p in b]
    ids=[x.island_id for p in a for x in p]
    assert sorted(ids)==list(range(1003)) and len(ids)==len(set(ids))
    loads=[sum(x.cost for x in p) for p in a]
    assert max(loads)-min(loads)<=max(x.cost for x in work)
    assert elapsed<1.


def test_colored_islands_and_archive_round_trip_without_loss():
    contacts=tuple(RigidContact(("ground",i),i,-1,torch.tensor([float(i),0.,0.],dtype=D),
        torch.tensor([0.,0.,1.],dtype=D)) for i in range(1001))
    data=prepare_colored_contacts(1001,contacts,dtype=D)
    work=colored_island_jobs(data);again=deserialize_jobs(serialize_jobs(work))
    assert len(work)==1001
    assert [x.island_id for x in again]==list(range(1001))
    assert torch.equal(torch.cat([x.contact_indices for x in again]),torch.arange(1001))
    results=execute_island_jobs(jobs(8),checksum)
    decoded=deserialize_executions(serialize_executions(results))
    assert [x.value for x in decoded]==[x.value for x in results]


def test_worker_failure_is_propagated_with_island_identity():
    with pytest.raises(IslandWorkerError,match="island 7 failed"):
        execute_island_jobs(jobs(12),fail_on_seven,workers=2,backend="process")


def test_no_gpu_has_explicit_cpu_fallback():
    devices=available_execution_devices(4)
    assert devices and (devices==("cpu",) or all(x.startswith("cuda:") for x in devices))
