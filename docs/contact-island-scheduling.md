# Contact-island worker scheduling

Independent contact islands can be scheduled with
`contact_island_scheduler`.  `colored_island_jobs` converts the deterministic
islands already produced by `prepare_colored_contacts` into disjoint body and
constraint index sets.  Longest-processing-time assignment uses stable island
identifiers as tie breakers, so input order does not affect placement.

The `serial` backend is the reference.  The optional `process` backend uses
spawned, single-host CPU workers and merges outputs in island-id order.  A
worker exception identifies the failed island and cancels outstanding work.
Torch's tensor-aware archive provides a versioned CPU input format for process
or external orchestration boundaries.

Tests require exact equality between one, two, and four workers, prove that
1,003 islands are neither lost nor duplicated, check deterministic load
balance and sub-second scheduler overhead, archive round-trip, and failure
propagation.

`available_execution_devices` reports deterministic CUDA labels when CUDA is
available and returns `("cpu",)` otherwise.  This is placement metadata only:
the process backend deliberately rejects CUDA metadata rather than silently
copying it.  No real multi-GPU execution has been qualified in the current
CPU-only environment, and no `torch.distributed` claim is made.

```bash
PYTHONWARNINGS=error pytest -q tests/test_contact_island_scheduler.py
python examples/contact_island_workers.py
```
