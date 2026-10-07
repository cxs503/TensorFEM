# Versioned result database

`ResultDB` is the solver-neutral result contract.  It stores model/job
metadata, integer node and element identifiers, connectivity, nodal and
element fields, scalar histories, probes, units and coordinate-system
metadata.  Every tensor and the metadata document has a SHA-256 checksum; a
database checksum covers the complete manifest.

`write_result_db("case", db)` always writes a compact `case.json` index.  With
the optional `h5py` package it also writes chunked datasets to `case.h5`.
`read_result_db` verifies all checksums by default.  If `h5py` is absent, the
index records `body.available=false`, writing returns `False`, and reading
raises an actionable error rather than silently producing an incomplete
result.  No dependency is installed by this feature.

In-memory `query_nodes(field, ids)` and lazy `query_result_nodes` preserve
requested ID order (including repeats) and reject unknown identifiers.
`iter_result_node_chunks` bounds read memory for large fields. HDF5 round-trip tests run automatically wherever the
optional dependency exists; the normal dependency-minimal CI instead verifies
the complete JSON inventory/checksum and explicit unavailable-body behavior.

The 16x16 hemisphere adapter retains all post-processing fields and histories.
Tests prove that displacement, probe, extrema, published-reference error, free
residual, and force/moment balance metrics do not drift during adaptation.
The fixture remains small (289 nodes and 256 elements), avoiding repository or
CI artifacts.

```bash
PYTHONWARNINGS=error pytest -q tests/test_result_db.py
python examples/hemisphere_result_db.py
```
