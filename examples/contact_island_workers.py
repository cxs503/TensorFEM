"""Deterministic single-host process scheduling demonstration."""
import torch
from tensorfem.contact_island_scheduler import IslandJob,execute_island_jobs


def checksum(job):
    return int(job.payload.sum())


if __name__=="__main__":
    jobs=tuple(IslandJob(i,torch.tensor([i]),torch.tensor([i]),float(i%3+1),
                         torch.tensor([i,2*i])) for i in range(12))
    result=execute_island_jobs(jobs,checksum,workers=2,backend="process")
    print([(x.island_id,x.worker_id,x.value) for x in result])
