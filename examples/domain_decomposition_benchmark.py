"""Opt-in 100k/1M-DOF partition and sparse matvec performance smoke test.

This script intentionally does not factor large dense local matrices. It
measures production-relevant O(n) topology construction and sparse operator
application; the exact Schwarz validation remains in the lightweight tests.
"""
from __future__ import annotations

import argparse
import time
import torch

from tensorfem.domain_decomposition import build_subdomains, contiguous_owners
from tensorfem.sparse_core import sparse_mv


def chain(n: int):
    i = torch.arange(n); off = torch.arange(n-1)
    rows = torch.cat((i, off, off+1)); cols = torch.cat((i, off+1, off))
    vals = torch.cat((torch.full((n,), 2., dtype=torch.float64),
                      torch.full((2*(n-1),), -1., dtype=torch.float64)))
    with torch.sparse.check_sparse_tensor_invariants(enable=False):
        return torch.sparse_coo_tensor(torch.stack((rows, cols)), vals, (n,n),
                                       check_invariants=False).coalesce()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ndof", type=int, choices=(100_000, 1_000_000), default=100_000)
    parser.add_argument("--parts", type=int, default=16)
    args = parser.parse_args(); A = chain(args.ndof)
    start = time.perf_counter()
    domains = build_subdomains(A, contiguous_owners(args.ndof, args.parts), overlap=1)
    topology = time.perf_counter()-start
    x = torch.ones(args.ndof, dtype=torch.float64); start = time.perf_counter()
    y = sparse_mv(A, x); matvec = time.perf_counter()-start
    assert len(domains) == args.parts and torch.isfinite(y).all()
    print({"ndof": args.ndof, "parts": args.parts,
           "topology_seconds": topology, "matvec_seconds": matvec})


if __name__ == "__main__": main()
