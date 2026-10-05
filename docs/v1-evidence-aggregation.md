# TensorFEM 1.0 evidence aggregation

`scripts/aggregate_v1_readiness.py` is the only bridge from persisted release
artifacts to `evaluate_v1_readiness`. It does not accept command-line `passed`
flags. Each supplied artifact is parsed, schema checked and SHA-256 verified
before its measured fields are mapped into the existing fail-closed contract.

Sources are:

- the 10k+ Shell4 sparse qualification report, validated by
  `verify_sparse_scaling_report`;
- a hashed `tensorfem.release-quality-evidence/1.0` record containing actual
  full-regression, offline-install and public-API audit outcomes;
- a hashed curved/frictional contact qualification report;
- an independently executed, hashed layered-J2 Shell4 robustness report that
  compares the same difficult step with fixed and backtracking globalization,
  checks against a strict reference path, and verifies rejected/caller state
  rollback;
- generation-scheduled 4x4, 8x8 and 12x12 panel directories, each verified by
  `panel_generation_readiness`, including status, event chain and checkpoint.

Missing paths are not errors that can accidentally stop reporting; they become
`missing_evidence` blockers. An existing artifact with a bad schema or hash is
an error and aborts aggregation. A curved frictionless subset remains out of
scope because v1 requires `general_surface_to_surface`. Panel prefixes remain
blocked until all three meshes have confirmed peaks and descending branches
and pass the existing convergence evaluator.

Example using the currently available real artifacts:

```bash
PYTHONPATH=src python scripts/aggregate_v1_readiness.py \
  --sparse-report .qualification/v1-shell-sparse-release/report.json \
  --nonlinear-shell-report .qualification/v1-readiness/nonlinear-shell.json \
  --panel-8 .qualification/v100-panel-8x8-backtracking-final \
  --output .qualification/v1-readiness/aggregate.json
```

Generate the nonlinear evidence by executing the real solver case (the runner
writes atomically):

```bash
PYTHONPATH=src python scripts/run_nonlinear_shell_robustness.py \
  --output .qualification/v1-readiness/nonlinear-shell.json
```

Panel generation counters and prefixes never qualify nonlinear robustness;
missing independent evidence leaves that gate blocked. A valid nonlinear
report qualifies only the bounded fixed-versus-backtracking layered-J2 case.
Likewise, a post-peak panel path alone does not qualify mesh convergence. Each
panel cache must also contain a hashed `peak-step-sensitivity.json` generated
from a local restart at a pre-peak checkpoint. The restart uses a smaller
`solver_maximum_step` (preferably `5e-5`, or a documented normalized-arc
equivalent), traverses only the peak window, and must change the peak reaction
by less than 3 percent. Missing, malformed, cross-mesh, or failing sensitivity
evidence is a hard blocker; the long path must not be rerun merely to satisfy
this gate.

`supervise_panel_generation.py` launches that bounded refinement after a
verified post-peak state. For supervisors which were already running when the
gate was introduced, `supervise_panel_peak_refinement.py` can be attached as a
non-invasive watcher. It copies a pre-peak immutable generation into a
separate control identity and never edits the source checkpoint. The resulting
evidence binds the source manifest and checkpoint hashes, refined manifest
hash, exact restart state, complete energy ledger, and both peak reactions.
The aggregate and embedded
readiness reports have independent hashes; tampering with either is detected
by `validate_v1_evidence_aggregate` and `validate_v1_readiness`.

Release-quality evidence is created only after actually executing the complete
regression, API audit and offline build/install smoke test:

```bash
PYTHONPATH=src python scripts/run_release_quality_evidence.py \
  --output .qualification/v1-readiness/release-quality.json
```

Each execution atomically writes its complete log and records the relative log
path, command, return code, duration, readable output tail and output SHA-256.
The aggregator reads each log, verifies its hash, recomputes each status from
the execution records, and rejects missing or inconsistent provenance as well
as an invalid outer evidence hash. It never
translates an ordinary hand-written JSON boolean into a passing gate.
