# Minimal industrial Model--Step--Job--Result workflow

The workflow references the existing MacNeal--Harder hemisphere kernel; it does
not duplicate shell assembly or solution code. `ModelSpec` fixes the 18-degree
hole geometry, material, quarter symmetry/gauge and opposed equator loads while
exposing mesh counts and drilling factor. `StepSpec` currently qualifies only
linear static analysis.

A SHA-256 digest of the canonical model, step, units and schema gives a stable
20-character job ID. Each run directory contains `manifest.json`, `job.json`
and `result.json`. Job state progresses through running to completed or failed;
failure type, message and traceback are retained. Completed replay requires an
exact result checksum. A stale running job is restarted deterministically, and
tampering or manifest collision fails closed.

Results contain full displacement and reaction vectors plus schema/kernel
versions, units, mesh size, probe response, reference error and timing. JSON is
portable and intentionally verbose; binary/HDF5 result backends can later be
added without changing the manifest contract.
