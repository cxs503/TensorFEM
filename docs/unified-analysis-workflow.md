# Unified analysis workflow

`analysis_workflow` is the first common vertical slice from an immutable
`ModelDB` through a versioned `AnalysisPlan`, registered analysis kernels,
deterministic job execution and `ResultDB v2` publication. It replaces no
solver formulation; it coordinates existing qualified kernels without loading
Python code from project input.

The v2 plan schema has strict fields and an explicit v1 migration. ModelDB
external integer IDs survive JSON round trips. Job IDs are derived from the
canonical plan, checkpoints and job state are atomically replaced, failed
steps are persisted, and resume reuses only successfully committed steps.
Unknown schemas, fields, forward dependencies and kernel kinds fail closed.

Current built-in registry coverage is intentionally narrow:

- 2-D T2D2 linear static analysis;
- LINE2 steady thermal analysis; and
- multistep publication to ResultDB v2.

This qualifies the orchestration contract, not universal ModelDB solver
coverage. Mixed element topology in one ResultDB, remote workers, concurrent
queue ownership, cancellation, resource limits, database locking and all
other analysis procedures remain future work.
