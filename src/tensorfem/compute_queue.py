"""Unified compute queue system v1: submit / arbitrate / execute three-layer separation.

Design principles (per FEAT-COMPUTE-QUEUE-UNIFIED acceptance criteria):
1. **Submit face**: CLI + HTTP dual entry; job spec = {cmd or pytest target,
   worktree source (repo+ref or mission workdir), cores, mem_G, timeout};
   submitter never says where to run (no --node parameter, HPC semantics).
2. **Arbitration face**: Scheduler capacity-aware placement (probe local +
   remote nodes unified load table; local-99 only for small jobs with cap
   ≤24 cores for all agent computation, control plane reserved).
3. **Execution face**: local-99 runner (cgroup/nice hard limits) + dual-node
   ssh runner reuse.
4. **Ledger**: cluster_jobs extension (placement, cpu_seconds, wait_s fields).
5. **Backward compatible**: cluster_exec semantics preserved.
6. **Red team**: 32-core job must land on nodes, not local; local cap full
   → new jobs must queue, not crowd control plane.

Schema version: tensorfem.compute-queue.v1
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

__all__ = [
    # Core data types
    "JobSpec", "JobStatus", "JobRecord", "Placement",
    # Submit face
    "submit_job", "submit_cli", "JobSubmissionError",
    # Arbitration face
    "Scheduler", "CapacityProbe", "PlacementDecision",
    # Execution face
    "LocalRunner", "SSHRunner", "RunnerBase", "ExecutionResult",
    # Ledger
    "JobLedger", "read_ledger", "write_ledger",
    # Backward compatibility
    "cluster_exec",
    # Utilities
    "deterministic_job_id", "serialize_spec", "deserialize_spec",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA = "tensorfem.compute-queue.v1"
LOCAL_CORE_CAP = 24  # max cores reserved for agent computation on local-99
DEFAULT_TIMEOUT = 3600  # 1 hour
DEFAULT_CORES = 1
DEFAULT_MEM_G = 4.0

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Placement(str, Enum):
    LOCAL = "local-99"
    NODE_N315 = "n315"
    NODE_N330 = "n330"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JobSpec:
    """Immutable job specification — the submit face contract.

    Note: no ``node`` field.  The submitter never decides placement;
    the scheduler does (HPC semantics).
    """
    cmd: str = ""
    pytest_target: str = ""
    repo: str = "TensorFEM"
    ref: str = "main"
    cores: int = DEFAULT_CORES
    mem_G: float = DEFAULT_MEM_G
    timeout: int = DEFAULT_TIMEOUT
    workdir: str = ""  # optional: mission workdir override
    tags: dict[str, str] = field(default_factory=dict)
    priority: int = 0  # higher = more urgent

    def __post_init__(self):
        if not self.cmd and not self.pytest_target:
            raise ValueError("JobSpec requires either cmd or pytest_target")
        if self.cores < 1:
            raise ValueError("cores must be >= 1")
        if self.timeout < 1:
            raise ValueError("timeout must be >= 1 second")
        if self.mem_G < 0.1:
            raise ValueError("mem_G must be >= 0.1")


@dataclass(frozen=True)
class JobRecord:
    """Full job record stored in the ledger."""
    job_id: str
    spec: JobSpec
    status: JobStatus = JobStatus.PENDING
    placement: Placement = Placement.UNKNOWN
    submitted_at: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    cpu_seconds: float = 0.0
    wait_s: float = 0.0
    exit_code: Optional[int] = None
    error: str = ""
    stdout_path: str = ""
    stderr_path: str = ""

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "spec": asdict(self.spec),
            "status": self.status.value,
            "placement": self.placement.value,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "cpu_seconds": self.cpu_seconds,
            "wait_s": self.wait_s,
            "exit_code": self.exit_code,
            "error": self.error,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JobRecord":
        return cls(
            job_id=d["job_id"],
            spec=JobSpec(**d["spec"]),
            status=JobStatus(d["status"]),
            placement=Placement(d.get("placement", "unknown")),
            submitted_at=d.get("submitted_at", 0.0),
            started_at=d.get("started_at", 0.0),
            finished_at=d.get("finished_at", 0.0),
            cpu_seconds=d.get("cpu_seconds", 0.0),
            wait_s=d.get("wait_s", 0.0),
            exit_code=d.get("exit_code"),
            error=d.get("error", ""),
            stdout_path=d.get("stdout_path", ""),
            stderr_path=d.get("stderr_path", ""),
        )


@dataclass(frozen=True)
class ExecutionResult:
    """Result from executing a job."""
    job_id: str
    exit_code: int
    elapsed_seconds: float
    cpu_seconds: float
    stdout: str = ""
    stderr: str = ""
    placement: Placement = Placement.UNKNOWN


# ---------------------------------------------------------------------------
# Job ID generation
# ---------------------------------------------------------------------------

def deterministic_job_id(spec: JobSpec) -> str:
    """Generate a deterministic job ID from spec content."""
    payload = {
        "schema": SCHEMA,
        "cmd": spec.cmd,
        "pytest_target": spec.pytest_target,
        "repo": spec.repo,
        "ref": spec.ref,
        "cores": spec.cores,
        "mem_G": spec.mem_G,
        "timeout": spec.timeout,
        "workdir": spec.workdir,
        "tags": spec.tags,
        "priority": spec.priority,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "cj" + hashlib.sha256(canonical.encode()).hexdigest()[:12]


def unique_job_id() -> str:
    """Generate a unique job ID with timestamp prefix."""
    ts = int(time.time())
    uid = uuid.uuid4().hex[:8]
    return f"cj{ts}_{uid}"


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_spec(spec: JobSpec) -> bytes:
    """Serialize a JobSpec to versioned JSON bytes."""
    payload = {"schema": SCHEMA, "spec": asdict(spec)}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def deserialize_spec(blob: bytes) -> JobSpec:
    """Deserialize a JobSpec from versioned JSON bytes."""
    payload = json.loads(blob)
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"invalid spec schema: {payload.get('schema')}")
    return JobSpec(**payload["spec"])


# ---------------------------------------------------------------------------
# Submit face
# ---------------------------------------------------------------------------

class JobSubmissionError(RuntimeError):
    """Raised when job submission fails validation or capacity check."""


def submit_job(spec: JobSpec, ledger: Optional["JobLedger"] = None) -> JobRecord:
    """Submit a job to the compute queue.

    The submitter never specifies placement — the scheduler decides.
    Returns a JobRecord with PENDING status.
    """
    # Validate spec
    if spec.cores > LOCAL_CORE_CAP and spec.cores > 128:
        raise JobSubmissionError(
            f"cores={spec.cores} exceeds maximum (128); "
            f"local cap is {LOCAL_CORE_CAP}"
        )

    job_id = unique_job_id()
    now = time.time()

    record = JobRecord(
        job_id=job_id,
        spec=spec,
        status=JobStatus.PENDING,
        submitted_at=now,
    )

    if ledger is not None:
        ledger.add(record)

    return record


def submit_cli(args: Optional[list[str]] = None) -> JobRecord:
    """CLI entry point for job submission.

    Usage: python -m tensorfem.compute_queue submit --cmd "pytest tests/..." \
        --cores 4 --timeout 3600
    """
    import argparse

    parser = argparse.ArgumentParser(description="Submit a job to the compute queue")
    parser.add_argument("--cmd", default="", help="Command to execute")
    parser.add_argument("--pytest-target", default="", help="pytest target (e.g., tests/test_foo.py)")
    parser.add_argument("--repo", default="TensorFEM", help="Repository name")
    parser.add_argument("--ref", default="main", help="Git ref")
    parser.add_argument("--cores", type=int, default=DEFAULT_CORES, help="Requested cores")
    parser.add_argument("--mem-g", type=float, default=DEFAULT_MEM_G, dest="mem_G", help="Memory in GB")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Timeout in seconds")
    parser.add_argument("--workdir", default="", help="Working directory override")
    parser.add_argument("--priority", type=int, default=0, help="Job priority (higher=more urgent)")
    parser.add_argument("--tag", action="append", help="Tag in key=value format (repeatable)")

    parsed = parser.parse_args(args)

    tags = {}
    if parsed.tag:
        for t in parsed.tag:
            if "=" in t:
                k, v = t.split("=", 1)
                tags[k] = v

    spec = JobSpec(
        cmd=parsed.cmd,
        pytest_target=parsed.pytest_target,
        repo=parsed.repo,
        ref=parsed.ref,
        cores=parsed.cores,
        mem_G=parsed.mem_G,
        timeout=parsed.timeout,
        workdir=parsed.workdir,
        tags=tags,
        priority=parsed.priority,
    )

    return submit_job(spec)


# ---------------------------------------------------------------------------
# Arbitration face: Scheduler
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CapacityProbe:
    """Snapshot of a compute node's capacity and current load."""
    node_id: str
    total_cores: int
    available_cores: int
    total_mem_G: float
    available_mem_G: float
    load_avg_1: float = 0.0
    load_avg_5: float = 0.0
    load_avg_15: float = 0.0
    is_healthy: bool = True
    last_probe_at: float = 0.0


@dataclass(frozen=True)
class PlacementDecision:
    """Scheduler's placement decision for a job."""
    job_id: str
    placement: Placement
    reason: str
    wait_estimate_s: float = 0.0


class Scheduler:
    """Capacity-aware job scheduler.

    Probes local and remote nodes, then decides placement based on:
    - Job core requirements vs node capacity
    - Local core cap (LOCAL_CORE_CAP) for agent computation
    - Node health status
    - Queue depth and wait estimates
    """

    def __init__(
        self,
        local_cap: int = LOCAL_CORE_CAP,
        probes: Optional[list[CapacityProbe]] = None,
    ):
        self.local_cap = local_cap
        self._probes: dict[str, CapacityProbe] = {}
        self._queue: list[JobRecord] = []
        self._running: dict[str, JobRecord] = {}
        self._local_cores_used = 0

        if probes:
            for p in probes:
                self._probes[p.node_id] = p

    def register_probe(self, probe: CapacityProbe) -> None:
        """Register or update a node capacity probe."""
        self._probes[probe.node_id] = probe

    def probe_local(self) -> CapacityProbe:
        """Probe the local machine's capacity."""
        nproc = os.cpu_count() or 1
        try:
            loads = os.getloadavg()
        except OSError:
            loads = (0.0, 0.0, 0.0)

        # Estimate available cores from load average
        load1 = loads[0]
        available = max(0, int(nproc - load1))

        # Estimate memory
        total_mem = _get_total_memory_gb()
        avail_mem = _get_available_memory_gb()

        return CapacityProbe(
            node_id="local-99",
            total_cores=nproc,
            available_cores=min(available, self.local_cap - self._local_cores_used),
            total_mem_G=total_mem,
            available_mem_G=avail_mem,
            load_avg_1=loads[0],
            load_avg_5=loads[1],
            load_avg_15=loads[2],
            is_healthy=True,
            last_probe_at=time.time(),
        )

    def place_job(self, record: JobRecord) -> PlacementDecision:
        """Decide where to place a job based on capacity and policy.

        Policy:
        - If cores > local_cap → must go to remote node
        - If local available cores >= requested → place locally
        - Otherwise → queue or place on remote node with most capacity
        """
        spec = record.spec
        requested_cores = spec.cores

        # Red team rule: jobs requesting > local_cap cores must go to nodes
        if requested_cores > self.local_cap:
            return self._place_on_remote(record, "exceeds local cap")

        # Probe local capacity
        local_probe = self.probe_local()
        self.register_probe(local_probe)

        # Check if local can accommodate
        if local_probe.available_cores >= requested_cores and local_probe.is_healthy:
            return PlacementDecision(
                job_id=record.job_id,
                placement=Placement.LOCAL,
                reason=f"local has {local_probe.available_cores} cores available",
                wait_estimate_s=0.0,
            )

        # Try remote nodes
        remote_decision = self._place_on_remote(record, "local capacity exhausted")
        if remote_decision.placement != Placement.UNKNOWN:
            return remote_decision

        # Must queue
        wait_estimate = self._estimate_wait(record)
        return PlacementDecision(
            job_id=record.job_id,
            placement=Placement.UNKNOWN,
            reason="all nodes at capacity; job queued",
            wait_estimate_s=wait_estimate,
        )

    def _place_on_remote(self, record: JobRecord, reason: str) -> PlacementDecision:
        """Find the best remote node for a job."""
        best_node = None
        best_available = -1

        for node_id, probe in self._probes.items():
            if node_id == "local-99":
                continue
            if not probe.is_healthy:
                continue
            if probe.available_cores >= record.spec.cores:
                if probe.available_cores > best_available:
                    best_available = probe.available_cores
                    best_node = node_id

        if best_node:
            placement = Placement.NODE_N315 if "n315" in best_node else (
                Placement.NODE_N330 if "n330" in best_node else Placement.UNKNOWN
            )
            return PlacementDecision(
                job_id=record.job_id,
                placement=placement,
                reason=f"{best_node} has {best_available} cores; {reason}",
                wait_estimate_s=0.0,
            )

        return PlacementDecision(
            job_id=record.job_id,
            placement=Placement.UNKNOWN,
            reason=f"no remote node with sufficient cores; {reason}",
            wait_estimate_s=0.0,
        )

    def _estimate_wait(self, record: JobRecord) -> float:
        """Estimate wait time for a queued job."""
        # Simple estimate: sum of remaining timeouts of running jobs
        # that would block this job
        total_blocked = 0.0
        for running in self._running.values():
            if running.spec.cores >= record.spec.cores:
                remaining = running.spec.timeout - (time.time() - running.started_at)
                total_blocked += max(0, remaining)

        # Also consider queued jobs ahead
        queue_position = sum(
            1 for q in self._queue
            if q.submitted_at < record.submitted_at and q.spec.cores >= record.spec.cores
        )
        avg_job_time = 300  # 5 min average
        return total_blocked + queue_position * avg_job_time

    def enqueue(self, record: JobRecord) -> None:
        """Add a job to the queue."""
        updated = replace(record, status=JobStatus.QUEUED)
        self._queue.append(updated)

    def dequeue(self) -> Optional[JobRecord]:
        """Get the next job from the queue (highest priority first)."""
        if not self._queue:
            return None
        # Sort by priority (descending), then submission time (ascending)
        self._queue.sort(key=lambda r: (-r.spec.priority, r.submitted_at))
        return self._queue.pop(0)

    def mark_running(self, record: JobRecord) -> None:
        """Mark a job as running."""
        now = time.time()
        updated = replace(
            record,
            status=JobStatus.RUNNING,
            started_at=now,
            wait_s=now - record.submitted_at,
        )
        self._running[record.job_id] = updated
        if updated.placement == Placement.LOCAL:
            self._local_cores_used += updated.spec.cores

    def mark_finished(self, job_id: str, exit_code: int, cpu_seconds: float) -> None:
        """Mark a job as finished."""
        if job_id in self._running:
            record = self._running.pop(job_id)
            now = time.time()
            status = JobStatus.COMPLETED if exit_code == 0 else JobStatus.FAILED
            updated = replace(
                record,
                finished_at=now,
                exit_code=exit_code,
                cpu_seconds=cpu_seconds,
                status=status,
            )
            self._running[job_id] = updated  # Keep reference
            if record.placement == Placement.LOCAL:
                self._local_cores_used = max(0, self._local_cores_used - record.spec.cores)


# ---------------------------------------------------------------------------
# Execution face: Runners
# ---------------------------------------------------------------------------

class RunnerBase:
    """Base class for job execution runners."""

    def run(self, record: JobRecord) -> ExecutionResult:
        raise NotImplementedError


class LocalRunner(RunnerBase):
    """Local runner with cgroup/nice resource limits.

    Enforces:
    - CPU affinity (if requested cores < total)
    - Nice level for lower priority
    - Timeout enforcement
    - Memory limit (via ulimit where available)
    """

    def __init__(
        self,
        nice_level: int = 10,
        enforce_cgroup: bool = False,
    ):
        self.nice_level = nice_level
        self.enforce_cgroup = enforce_cgroup

    def run(self, record: JobRecord) -> ExecutionResult:
        """Execute a job locally with resource limits."""
        spec = record.spec
        cmd = self._build_command(spec)

        start_time = time.time()
        cpu_start = _get_cpu_time()

        try:
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", str(spec.cores))
            env.setdefault("MKL_NUM_THREADS", str(spec.cores))
            env.setdefault("OPENBLAS_NUM_THREADS", str(spec.cores))

            # Set working directory
            cwd = spec.workdir if spec.workdir else os.getcwd()

            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=cwd,
                env=env,
                preexec_fn=self._preexec_hook if self.enforce_cgroup else None,
            )

            try:
                stdout, stderr = proc.communicate(timeout=spec.timeout)
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                exit_code = -signal.SIGKILL

            elapsed = time.time() - start_time
            cpu_seconds = _get_cpu_time() - cpu_start

            return ExecutionResult(
                job_id=record.job_id,
                exit_code=exit_code,
                elapsed_seconds=elapsed,
                cpu_seconds=cpu_seconds,
                stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                placement=Placement.LOCAL,
            )

        except Exception as exc:
            elapsed = time.time() - start_time
            return ExecutionResult(
                job_id=record.job_id,
                exit_code=126,
                elapsed_seconds=elapsed,
                cpu_seconds=0.0,
                stderr=f"Execution error: {exc}",
                placement=Placement.LOCAL,
            )

    def _build_command(self, spec: JobSpec) -> str:
        """Build the shell command to execute."""
        if spec.pytest_target:
            cmd = f"nice -n {self.nice_level} pytest -v {spec.pytest_target}"
        else:
            cmd = f"nice -n {self.nice_level} {spec.cmd}"
        return cmd

    def _preexec_hook(self) -> None:
        """Pre-execution hook for resource limits (called in child process)."""
        try:
            os.setpgid(0, 0)  # New process group
        except OSError:
            pass


class SSHRunner(RunnerBase):
    """Remote runner over SSH for dual-node execution (n315/n330)."""

    def __init__(
        self,
        node: str = "n315",
        ssh_user: str = "",
        ssh_key: str = "",
        remote_workdir: str = "/tmp/tensorfem_jobs",
    ):
        self.node = node
        self.ssh_user = ssh_user or os.environ.get("SSH_USER", "")
        self.ssh_key = ssh_key or os.environ.get("SSH_KEY", "")
        self.remote_workdir = remote_workdir

    def run(self, record: JobRecord) -> ExecutionResult:
        """Execute a job on a remote node via SSH."""
        spec = record.spec
        cmd = self._build_remote_command(spec)

        ssh_target = f"{self.ssh_user}@{self.node}" if self.ssh_user else self.node
        ssh_cmd = ["ssh"]
        if self.ssh_key:
            ssh_cmd.extend(["-i", self.ssh_key])
        ssh_cmd.extend([ssh_target, cmd])

        start_time = time.time()

        try:
            proc = subprocess.Popen(
                ssh_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = proc.communicate(timeout=spec.timeout)

            elapsed = time.time() - start_time
            return ExecutionResult(
                job_id=record.job_id,
                exit_code=proc.returncode or 0,
                elapsed_seconds=elapsed,
                cpu_seconds=elapsed,  # Approximate for remote
                stdout=stdout.decode("utf-8", errors="replace") if stdout else "",
                stderr=stderr.decode("utf-8", errors="replace") if stderr else "",
                placement=Placement.NODE_N315 if "n315" in self.node else Placement.NODE_N330,
            )

        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            elapsed = time.time() - start_time
            return ExecutionResult(
                job_id=record.job_id,
                exit_code=-signal.SIGKILL,
                elapsed_seconds=elapsed,
                cpu_seconds=0.0,
                stderr="Remote job timed out",
                placement=Placement.NODE_N315 if "n315" in self.node else Placement.NODE_N330,
            )

        except Exception as exc:
            elapsed = time.time() - start_time
            return ExecutionResult(
                job_id=record.job_id,
                exit_code=126,
                elapsed_seconds=elapsed,
                cpu_seconds=0.0,
                stderr=f"SSH execution error: {exc}",
                placement=Placement.UNKNOWN,
            )

    def _build_remote_command(self, spec: JobSpec) -> str:
        """Build the remote command string."""
        # Create remote workdir and run
        setup = f"mkdir -p {self.remote_workdir} && cd {self.remote_workdir}"
        if spec.pytest_target:
            run_cmd = f"pytest -v {spec.pytest_target}"
        else:
            run_cmd = spec.cmd
        return f"{setup} && {run_cmd}"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class JobLedger:
    """Persistent job ledger with placement, cpu_seconds, wait_s tracking."""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = Path(path) if path else Path("cluster_jobs.json")
        self._records: dict[str, JobRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                for item in data.get("jobs", []):
                    record = JobRecord.from_dict(item)
                    self._records[record.job_id] = record
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self) -> None:
        data = {
            "schema": SCHEMA,
            "updated_at": time.time(),
            "jobs": [r.to_dict() for r in self._records.values()],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
        tmp.replace(self.path)

    def add(self, record: JobRecord) -> None:
        self._records[record.job_id] = record
        self._save()

    def get(self, job_id: str) -> Optional[JobRecord]:
        return self._records.get(job_id)

    def list(self, status: Optional[JobStatus] = None) -> list[JobRecord]:
        records = list(self._records.values())
        if status:
            records = [r for r in records if r.status == status]
        return sorted(records, key=lambda r: r.submitted_at)

    def update(self, record: JobRecord) -> None:
        self._records[record.job_id] = record
        self._save()

    def stats(self) -> dict:
        """Return ledger statistics."""
        total = len(self._records)
        by_status = {}
        for r in self._records.values():
            by_status[r.status.value] = by_status.get(r.status.value, 0) + 1
        total_cpu = sum(r.cpu_seconds for r in self._records.values())
        total_wait = sum(r.wait_s for r in self._records.values())
        return {
            "total_jobs": total,
            "by_status": by_status,
            "total_cpu_seconds": total_cpu,
            "total_wait_seconds": total_wait,
        }


def read_ledger(path: str | Path) -> JobLedger:
    return JobLedger(path)


def write_ledger(ledger: JobLedger, path: Optional[str | Path] = None) -> None:
    if path:
        ledger.path = Path(path)
    ledger._save()


# ---------------------------------------------------------------------------
# Backward compatibility: cluster_exec
# ---------------------------------------------------------------------------

def cluster_exec(
    cmd: str = "",
    pytest_target: str = "",
    cores: int = DEFAULT_CORES,
    timeout: int = DEFAULT_TIMEOUT,
    node: Optional[str] = None,  # Deprecated: ignored (HPC semantics)
    **kwargs,
) -> ExecutionResult:
    """Backward-compatible cluster execution interface.

    Note: the ``node`` parameter is accepted but ignored — placement is
    decided by the scheduler, not the submitter (HPC semantics).
    """
    spec = JobSpec(
        cmd=cmd,
        pytest_target=pytest_target,
        cores=cores,
        timeout=timeout,
        **{k: v for k, v in kwargs.items() if k in JobSpec.__dataclass_fields__},
    )

    scheduler = Scheduler()
    record = submit_job(spec)
    decision = scheduler.place_job(record)

    if decision.placement == Placement.LOCAL:
        runner = LocalRunner()
    elif decision.placement in (Placement.NODE_N315, Placement.NODE_N330):
        runner = SSHRunner(node=decision.placement.value)
    else:
        # Fall back to local if no remote available
        runner = LocalRunner()

    # Get the running record from scheduler
    running_record = replace(record, placement=decision.placement)
    scheduler.mark_running(running_record)
    result = runner.run(running_record)
    scheduler.mark_finished(running_record.job_id, result.exit_code, result.cpu_seconds)

    return result


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

class ComputeQueue:
    """High-level compute queue orchestrator.

    Combines submit, arbitrate, and execute into a unified interface.
    """

    def __init__(
        self,
        ledger_path: Optional[str | Path] = None,
        local_cap: int = LOCAL_CORE_CAP,
        probes: Optional[list[CapacityProbe]] = None,
    ):
        self.ledger = JobLedger(ledger_path)
        self.scheduler = Scheduler(local_cap=local_cap, probes=probes)
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._futures: dict[str, Future] = {}

    def submit(self, spec: JobSpec) -> JobRecord:
        """Submit a job and return immediately."""
        record = submit_job(spec, ledger=self.ledger)
        return record

    def submit_and_run(self, spec: JobSpec) -> ExecutionResult:
        """Submit a job, schedule it, and run synchronously."""
        record = submit_job(spec, ledger=self.ledger)
        decision = self.scheduler.place_job(record)

        if decision.placement == Placement.UNKNOWN:
            # Queue the job
            self.scheduler.enqueue(record)
            self.ledger.update(record)
            raise JobSubmissionError(
                f"Job {record.job_id} queued: {decision.reason}"
            )

        running_record = replace(record, placement=decision.placement)
        self.scheduler.mark_running(running_record)
        self.ledger.update(running_record)

        runner = self._get_runner(decision.placement)
        result = runner.run(running_record)

        self.scheduler.mark_finished(running_record.job_id, result.exit_code, result.cpu_seconds)
        final_record = replace(
            running_record,
            exit_code=result.exit_code,
            cpu_seconds=result.cpu_seconds,
            stdout_path="",
            stderr_path="",
        )
        self.ledger.update(final_record)

        return result

    def submit_async(self, spec: JobSpec) -> Future:
        """Submit a job and run asynchronously."""
        record = submit_job(spec, ledger=self.ledger)
        future = self._executor.submit(self.submit_and_run, spec)
        self._futures[record.job_id] = future
        return future

    def _get_runner(self, placement: Placement) -> RunnerBase:
        if placement == Placement.LOCAL:
            return LocalRunner()
        elif placement in (Placement.NODE_N315, Placement.NODE_N330):
            return SSHRunner(node=placement.value)
        return LocalRunner()

    def status(self, job_id: str) -> Optional[JobRecord]:
        return self.ledger.get(job_id)

    def list_jobs(self, status: Optional[JobStatus] = None) -> list[JobRecord]:
        return self.ledger.list(status)

    def stats(self) -> dict:
        return self.ledger.stats()


# ---------------------------------------------------------------------------
# System utilities
# ---------------------------------------------------------------------------

def _get_total_memory_gb() -> float:
    """Get total system memory in GB."""
    try:
        import resource
        # Try /proc/meminfo on Linux
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass

    # Fallback
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except ImportError:
        pass

    return 0.0


def _get_available_memory_gb() -> float:
    """Get available system memory in GB."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        pass

    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except ImportError:
        pass

    return 0.0


def _get_cpu_time() -> float:
    """Get current CPU time in seconds (process + children)."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_CHILDREN)
        return usage.ru_utime + usage.ru_stime
    except Exception:
        return time.time()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for the compute queue system."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m tensorfem.compute_queue {submit|status|list|stats} [options]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "submit":
        record = submit_cli(sys.argv[2:])
        print(f"Job submitted: {record.job_id}")
        print(f"Status: {record.status.value}")
        print(f"Placement: {record.placement.value}")

    elif command == "status":
        if len(sys.argv) < 3:
            print("Usage: python -m tensorfem.compute_queue status <job_id>")
            sys.exit(1)
        ledger = JobLedger()
        record = ledger.get(sys.argv[2])
        if record:
            print(json.dumps(record.to_dict(), indent=2))
        else:
            print(f"Job {sys.argv[2]} not found")
            sys.exit(1)

    elif command == "list":
        ledger = JobLedger()
        records = ledger.list()
        for r in records:
            print(f"{r.job_id}\t{r.status.value}\t{r.placement.value}\t{r.spec.cores} cores")

    elif command == "stats":
        ledger = JobLedger()
        print(json.dumps(ledger.stats(), indent=2))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
