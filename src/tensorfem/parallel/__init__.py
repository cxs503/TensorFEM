"""Distributed-memory (MPI-style) FEM parallelism for TensorFEM.

Submodules
----------
comm        communication framework (SerialComm / ProcessComm / MPIComm)
partition   RCB geometric + graph partitioning with quality metrics
poisson     distributed Q1 Poisson kernel: ghosted vectors, halo exchange,
            matrix-free distributed CG
scaling     strong/weak scaling harness (predeclared protocol)
"""
from .comm import (Comm, MPIComm, ProcessComm, SerialComm, backend_available,
                   spawn)
from .partition import (PartitionStats, element_adjacency, graph_partition,
                        partition_stats, rcb_partition)

__all__ = [
    "Comm", "SerialComm", "ProcessComm", "MPIComm", "backend_available", "spawn",
    "rcb_partition", "graph_partition", "element_adjacency", "partition_stats",
    "PartitionStats",
]