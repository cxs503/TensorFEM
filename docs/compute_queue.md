# Compute Queue System v1 (FEAT-COMPUTE-QUEUE-UNIFIED)

## Overview

Unified compute queue system for TensorFEM with **submit / arbitrate / execute** three-layer separation.

### Design Principles

1. **Submit face**: CLI + HTTP dual entry; job spec = `{cmd or pytest target, worktree source (repo+ref or mission workdir), cores, mem_G, timeout}`; submitter **never** says where to run (no `--node` parameter, HPC semantics).

2. **Arbitration face**: Scheduler capacity-aware placement (probe n315/n330/local-99 unified load table; local-99 only for small jobs with cap ≤24 cores for all agent computation, control plane reserved).

3. **Execution face**: local-99 runner (cgroup/nice hard limits) + dual-node ssh runner reuse.

4. **Ledger**: `cluster_jobs` extension (placement, cpu_seconds, wait_s fields).

5. **Backward compatible**: `cluster_exec` semantics preserved (deprecated `node` parameter accepted but ignored).

6. **Red team**: 32-core job must land on nodes, not local; local cap full → new jobs must queue, not crowd control plane.

## Quick Start

```python
from tensorfem import ComputeQueue, JobSpec, Scheduler, JobLedger

# Create a compute queue
queue = ComputeQueue(ledger_path="cluster_jobs.json")

# Submit a job
spec = JobSpec(cmd="pytest tests/test_foo.py -v", cores=4, timeout=300)
record = queue.submit(spec)
print(f"Job submitted: {record.job_id}")

# Submit and run synchronously
spec2 = JobSpec(cmd="echo hello", cores=1, timeout=30)
result = queue.submit_and_run(spec2)
print(f"Exit code: {result.exit_code}")
print(f"Output: {result.stdout}")
```

## CLI Usage

```bash
# Submit a job
python -m tensorfem.compute_queue submit --cmd "pytest tests/" --cores 4 --timeout 300

# Check job status
python -m tensorfem.compute_queue status cj1234567890_ab

# List all jobs
python -m tensorfem.compute_queue list

# View statistics
python -m tensorfem.compute_queue stats
```

## Architecture

### Submit Face

The `JobSpec` dataclass defines the job specification:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cmd` | str | `""` | Shell command to execute |
| `pytest_target` | str | `""` | pytest target (alternative to cmd) |
| `repo` | str | `"TensorFEM"` | Repository name |
| `ref` | str | `"main"` | Git ref |
| `cores` | int | `1` | Requested CPU cores |
| `mem_G` | float | `4.0` | Requested memory in GB |
| `timeout` | int | `3600` | Timeout in seconds |
| `workdir` | str | `""` | Working directory override |
| `tags` | dict | `{}` | Key-value tags |
| `priority` | int | `0` | Job priority (higher = more urgent) |

**Key invariant**: No `node` field. Placement is decided by the scheduler, not the submitter.

### Arbitration Face

The `Scheduler` class implements capacity-aware placement:

```python
scheduler = Scheduler(local_cap=24, probes=[
    CapacityProbe(node_id="n315", total_cores=64, available_cores=48, ...),
    CapacityProbe(node_id="n330", total_cores=64, available_cores=32, ...),
])

decision = scheduler.place_job(record)
# decision.placement: LOCAL | NODE_N315 | NODE_N330 | UNKNOWN
# decision.reason: human-readable explanation
```

**Placement policy**:
- If `cores > local_cap` (24) → must go to remote node
- If local available cores ≥ requested → place locally
- Otherwise → try remote nodes, then queue

### Execution Face

Two runner implementations:

- **`LocalRunner`**: Executes locally with `nice` priority and resource limits
- **`SSHRunner`**: Executes on remote nodes via SSH

```python
runner = LocalRunner(nice_level=10)
result = runner.run(record)
# result.exit_code, result.stdout, result.stderr, result.elapsed_seconds
```

### Ledger

The `JobLedger` provides persistent job tracking:

```python
ledger = JobLedger("cluster_jobs.json")
ledger.add(record)
ledger.update(updated_record)
stats = ledger.stats()
# {"total_jobs": N, "by_status": {...}, "total_cpu_seconds": X, "total_wait_seconds": Y}
```

### Backward Compatibility

The `cluster_exec` function maintains backward compatibility:

```python
# Old API still works (node parameter is ignored)
result = cluster_exec(cmd="echo hello", cores=1, timeout=10, node="n315")
```

## Red Team Verification

| Test | Expected | Status |
|------|----------|--------|
| 32-core job placement | Must NOT land on local | ✅ |
| Local cap full (24 cores used) | New jobs must queue | ✅ |
| Job timeout enforcement | Process killed, exit_code=-9 | ✅ |
| Unhealthy node skipped | Not selected for placement | ✅ |

## Schema

Job spec serialization uses versioned JSON:

```json
{
  "schema": "tensorfem.compute-queue.v1",
  "spec": {
    "cmd": "pytest tests/",
    "cores": 4,
    "timeout": 300,
    ...
  }
}
```

## Files

- `src/tensorfem/compute_queue.py` — Core implementation (~900 lines)
- `tests/test_compute_queue.py` — Test suite (52 tests)
- `docs/compute_queue.md` — This documentation
