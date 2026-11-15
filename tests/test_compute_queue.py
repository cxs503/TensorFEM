"""Tests for the unified compute queue system (FEAT-COMPUTE-QUEUE-UNIFIED).

Coverage:
- Submit face: JobSpec validation, submit_job, submit_cli, deterministic_job_id
- Arbitration face: Scheduler capacity-aware placement, local cap enforcement
- Execution face: LocalRunner timeout, SSHRunner (mocked), resource limits
- Ledger: persistence, stats, update
- Backward compatibility: cluster_exec with deprecated node param
- Red team: 32-core job must not land on local; local cap full → queue
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dataclasses import replace

from tensorfem.compute_queue import (
    JobSpec,
    JobStatus,
    JobRecord,
    Placement,
    CapacityProbe,
    PlacementDecision,
    Scheduler,
    LocalRunner,
    SSHRunner,
    JobLedger,
    ComputeQueue,
    JobSubmissionError,
    submit_job,
    submit_cli,
    cluster_exec,
    deterministic_job_id,
    serialize_spec,
    deserialize_spec,
    LOCAL_CORE_CAP,
    SCHEMA,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_ledger_path(tmp_path):
    return str(tmp_path / "cluster_jobs.json")


@pytest.fixture
def basic_spec():
    return JobSpec(cmd="echo hello", cores=2, timeout=60)


@pytest.fixture
def pytest_spec():
    return JobSpec(pytest_target="tests/test_compute_queue.py", cores=1, timeout=120)


@pytest.fixture
def scheduler_with_probes():
    probes = [
        CapacityProbe(
            node_id="n315",
            total_cores=64,
            available_cores=48,
            total_mem_G=256.0,
            available_mem_G=200.0,
            is_healthy=True,
        ),
        CapacityProbe(
            node_id="n330",
            total_cores=64,
            available_cores=32,
            total_mem_G=256.0,
            available_mem_G=180.0,
            is_healthy=True,
        ),
    ]
    return Scheduler(local_cap=24, probes=probes)


# ---------------------------------------------------------------------------
# Submit face tests
# ---------------------------------------------------------------------------

class TestJobSpec:
    def test_valid_cmd_spec(self):
        spec = JobSpec(cmd="pytest tests/", cores=4, timeout=300)
        assert spec.cmd == "pytest tests/"
        assert spec.cores == 4
        assert spec.timeout == 300

    def test_valid_pytest_spec(self):
        spec = JobSpec(pytest_target="tests/test_foo.py", cores=2)
        assert spec.pytest_target == "tests/test_foo.py"

    def test_requires_cmd_or_pytest(self):
        with pytest.raises(ValueError, match="requires either cmd or pytest_target"):
            JobSpec()

    def test_cores_must_be_positive(self):
        with pytest.raises(ValueError, match="cores must be >= 1"):
            JobSpec(cmd="echo", cores=0)

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValueError, match="timeout must be >= 1"):
            JobSpec(cmd="echo", timeout=0)

    def test_mem_g_minimum(self):
        with pytest.raises(ValueError, match="mem_G must be >= 0.1"):
            JobSpec(cmd="echo", mem_G=0.01)

    def test_default_values(self):
        spec = JobSpec(cmd="echo")
        assert spec.cores == 1
        assert spec.mem_G == 4.0
        assert spec.timeout == 3600
        assert spec.repo == "TensorFEM"
        assert spec.ref == "main"


class TestSubmitJob:
    def test_submit_creates_pending_record(self, basic_spec, tmp_ledger_path):
        ledger = JobLedger(tmp_ledger_path)
        record = submit_job(basic_spec, ledger=ledger)
        assert record.status == JobStatus.PENDING
        assert record.job_id.startswith("cj")
        assert record.submitted_at > 0
        assert record.placement == Placement.UNKNOWN

    def test_submit_too_many_cores_raises(self):
        spec = JobSpec(cmd="echo", cores=200)
        with pytest.raises(JobSubmissionError, match="exceeds maximum"):
            submit_job(spec)

    def test_submit_32_cores_allowed_but_not_local(self):
        """Red team: 32-core job is valid but must not land on local."""
        spec = JobSpec(cmd="echo", cores=32)
        record = submit_job(spec)
        assert record.status == JobStatus.PENDING
        # Placement decision happens in scheduler, not submit


class TestSubmitCLI:
    def test_cli_with_cmd(self):
        record = submit_cli(["--cmd", "echo hello", "--cores", "2"])
        assert record.spec.cmd == "echo hello"
        assert record.spec.cores == 2

    def test_cli_with_pytest_target(self):
        record = submit_cli(["--pytest-target", "tests/test_foo.py"])
        assert record.spec.pytest_target == "tests/test_foo.py"

    def test_cli_with_tags(self):
        record = submit_cli(["--cmd", "echo", "--tag", "env=dev", "--tag", "priority=high"])
        assert record.spec.tags == {"env": "dev", "priority": "high"}

    def test_cli_priority(self):
        record = submit_cli(["--cmd", "echo", "--priority", "5"])
        assert record.spec.priority == 5


class TestDeterministicJobId:
    def test_same_spec_same_id(self):
        spec1 = JobSpec(cmd="echo", cores=2)
        spec2 = JobSpec(cmd="echo", cores=2)
        id1 = deterministic_job_id(spec1)
        id2 = deterministic_job_id(spec2)
        assert id1 == id2

    def test_different_spec_different_id(self):
        spec1 = JobSpec(cmd="echo", cores=2)
        spec2 = JobSpec(cmd="echo", cores=4)
        assert deterministic_job_id(spec1) != deterministic_job_id(spec2)

    def test_id_format(self):
        spec = JobSpec(cmd="echo")
        job_id = deterministic_job_id(spec)
        assert job_id.startswith("cj")
        assert len(job_id) == 14  # "cj" + 12 hex chars


class TestSerialization:
    def test_round_trip(self):
        spec = JobSpec(cmd="pytest tests/", cores=4, timeout=300, tags={"env": "test"})
        blob = serialize_spec(spec)
        restored = deserialize_spec(blob)
        assert restored.cmd == spec.cmd
        assert restored.cores == spec.cores
        assert restored.timeout == spec.timeout
        assert restored.tags == spec.tags

    def test_invalid_schema_raises(self):
        blob = json.dumps({"schema": "wrong", "spec": {}}).encode()
        with pytest.raises(ValueError, match="invalid spec schema"):
            deserialize_spec(blob)


# ---------------------------------------------------------------------------
# Arbitration face tests
# ---------------------------------------------------------------------------

class TestScheduler:
    def test_place_small_job_locally(self, scheduler_with_probes):
        """Jobs within local cap should be placed locally if capacity available."""
        spec = JobSpec(cmd="echo", cores=4)
        record = JobRecord(job_id="cj001", spec=spec, submitted_at=time.time())

        # Mock local probe to have enough capacity
        with patch.object(scheduler_with_probes, "probe_local") as mock_probe:
            mock_probe.return_value = CapacityProbe(
                node_id="local-99",
                total_cores=128,
                available_cores=20,
                total_mem_G=512.0,
                available_mem_G=400.0,
                is_healthy=True,
            )
            decision = scheduler_with_probes.place_job(record)
            assert decision.placement == Placement.LOCAL

    def test_place_large_job_on_remote(self, scheduler_with_probes):
        """Jobs exceeding local cap must go to remote nodes."""
        spec = JobSpec(cmd="echo", cores=32)
        record = JobRecord(job_id="cj002", spec=spec, submitted_at=time.time())

        decision = scheduler_with_probes.place_job(record)
        assert decision.placement in (Placement.NODE_N315, Placement.NODE_N330)
        assert "exceeds local cap" in decision.reason

    def test_place_on_remote_when_local_full(self, scheduler_with_probes):
        """When local is at capacity, jobs go to remote."""
        spec = JobSpec(cmd="echo", cores=8)
        record = JobRecord(job_id="cj003", spec=spec, submitted_at=time.time())

        with patch.object(scheduler_with_probes, "probe_local") as mock_probe:
            mock_probe.return_value = CapacityProbe(
                node_id="local-99",
                total_cores=128,
                available_cores=0,  # Local is full
                total_mem_G=512.0,
                available_mem_G=100.0,
                is_healthy=True,
            )
            decision = scheduler_with_probes.place_job(record)
            assert decision.placement in (Placement.NODE_N315, Placement.NODE_N330)

    def test_queue_when_all_nodes_full(self, scheduler_with_probes):
        """When all nodes are at capacity, job is queued."""
        spec = JobSpec(cmd="echo", cores=8)
        record = JobRecord(job_id="cj004", spec=spec, submitted_at=time.time())

        # Make all nodes unavailable
        scheduler_with_probes._probes["n315"] = CapacityProbe(
            node_id="n315", total_cores=64, available_cores=0,
            total_mem_G=256.0, available_mem_G=50.0, is_healthy=True,
        )
        scheduler_with_probes._probes["n330"] = CapacityProbe(
            node_id="n330", total_cores=64, available_cores=0,
            total_mem_G=256.0, available_mem_G=50.0, is_healthy=True,
        )

        with patch.object(scheduler_with_probes, "probe_local") as mock_probe:
            mock_probe.return_value = CapacityProbe(
                node_id="local-99",
                total_cores=128,
                available_cores=0,
                total_mem_G=512.0,
                available_mem_G=50.0,
                is_healthy=True,
            )
            decision = scheduler_with_probes.place_job(record)
            assert decision.placement == Placement.UNKNOWN
            assert "queued" in decision.reason

    def test_enqueue_and_dequeue_priority(self, scheduler_with_probes):
        """Higher priority jobs should be dequeued first."""
        spec1 = JobSpec(cmd="echo low", cores=2, priority=1)
        spec2 = JobSpec(cmd="echo high", cores=2, priority=10)
        spec3 = JobSpec(cmd="echo medium", cores=2, priority=5)

        r1 = JobRecord(job_id="cj_low", spec=spec1, submitted_at=100.0)
        r2 = JobRecord(job_id="cj_high", spec=spec2, submitted_at=101.0)
        r3 = JobRecord(job_id="cj_med", spec=spec3, submitted_at=102.0)

        scheduler_with_probes.enqueue(r1)
        scheduler_with_probes.enqueue(r2)
        scheduler_with_probes.enqueue(r3)

        # Dequeue should return highest priority first
        first = scheduler_with_probes.dequeue()
        assert first.job_id == "cj_high"
        second = scheduler_with_probes.dequeue()
        assert second.job_id == "cj_med"
        third = scheduler_with_probes.dequeue()
        assert third.job_id == "cj_low"

    def test_mark_running_and_finished(self, scheduler_with_probes):
        spec = JobSpec(cmd="echo", cores=4)
        record = JobRecord(job_id="cj005", spec=spec, submitted_at=time.time())

        scheduler_with_probes.mark_running(record)
        # Scheduler stores updated record internally
        running = scheduler_with_probes._running.get("cj005")
        assert running is not None
        assert running.status == JobStatus.RUNNING
        assert running.started_at > 0
        assert running.wait_s >= 0

        scheduler_with_probes.mark_finished(record.job_id, 0, 2.5)
        finished = scheduler_with_probes._running.get("cj005")
        assert finished is not None
        assert finished.status == JobStatus.COMPLETED
        assert finished.cpu_seconds == 2.5
        assert finished.finished_at > 0

    def test_mark_failed(self, scheduler_with_probes):
        spec = JobSpec(cmd="false", cores=1)
        record = JobRecord(job_id="cj006", spec=spec, submitted_at=time.time())

        scheduler_with_probes.mark_running(record)
        scheduler_with_probes.mark_finished(record.job_id, 1, 0.5)
        finished = scheduler_with_probes._running.get("cj006")
        assert finished is not None
        assert finished.status == JobStatus.FAILED


# ---------------------------------------------------------------------------
# Red team tests
# ---------------------------------------------------------------------------

class TestRedTeam:
    def test_32_core_job_must_not_land_on_local(self, scheduler_with_probes):
        """Red team: submitting a 32-core job must fall on nodes, not local."""
        spec = JobSpec(cmd="echo", cores=32)
        record = JobRecord(job_id="cj_red1", spec=spec, submitted_at=time.time())

        # Even if local has plenty of cores, 32 > local_cap (24)
        with patch.object(scheduler_with_probes, "probe_local") as mock_probe:
            mock_probe.return_value = CapacityProbe(
                node_id="local-99",
                total_cores=128,
                available_cores=100,  # Plenty available
                total_mem_G=512.0,
                available_mem_G=400.0,
                is_healthy=True,
            )
            decision = scheduler_with_probes.place_job(record)
            assert decision.placement != Placement.LOCAL
            assert decision.placement in (Placement.NODE_N315, Placement.NODE_N330)

    def test_local_cap_full_must_queue(self):
        """Red team: when local cap is full, new jobs must queue, not crowd control plane."""
        scheduler = Scheduler(local_cap=24)
        scheduler._local_cores_used = 24  # Cap is full

        spec = JobSpec(cmd="echo", cores=4)
        record = JobRecord(job_id="cj_red2", spec=spec, submitted_at=time.time())

        # Make remote unavailable too
        scheduler._probes["n315"] = CapacityProbe(
            node_id="n315", total_cores=64, available_cores=0,
            total_mem_G=256.0, available_mem_G=50.0, is_healthy=True,
        )

        with patch.object(scheduler, "probe_local") as mock_probe:
            mock_probe.return_value = CapacityProbe(
                node_id="local-99",
                total_cores=128,
                available_cores=0,  # Local cap exhausted
                total_mem_G=512.0,
                available_mem_G=100.0,
                is_healthy=True,
            )
            decision = scheduler.place_job(record)
            assert decision.placement == Placement.UNKNOWN
            assert "queued" in decision.reason


# ---------------------------------------------------------------------------
# Execution face tests
# ---------------------------------------------------------------------------

class TestLocalRunner:
    def test_run_simple_command(self):
        runner = LocalRunner()
        spec = JobSpec(cmd="echo hello", cores=1, timeout=10)
        record = JobRecord(job_id="cj_exec1", spec=spec, submitted_at=time.time())

        result = runner.run(record)
        assert result.exit_code == 0
        assert "hello" in result.stdout
        assert result.placement == Placement.LOCAL
        assert result.elapsed_seconds > 0

    def test_run_failing_command(self):
        runner = LocalRunner()
        spec = JobSpec(cmd="false", cores=1, timeout=10)
        record = JobRecord(job_id="cj_exec2", spec=spec, submitted_at=time.time())

        result = runner.run(record)
        assert result.exit_code == 1
        assert result.placement == Placement.LOCAL

    def test_run_timeout(self):
        runner = LocalRunner()
        spec = JobSpec(cmd="sleep 10", cores=1, timeout=1)
        record = JobRecord(job_id="cj_exec3", spec=spec, submitted_at=time.time())

        result = runner.run(record)
        assert result.exit_code == -9  # SIGKILL
        assert "timed out" in result.stderr.lower() or result.exit_code < 0

    def test_run_pytest_target(self):
        runner = LocalRunner()
        spec = JobSpec(pytest_target="--version", cores=1, timeout=30)
        record = JobRecord(job_id="cj_exec4", spec=spec, submitted_at=time.time())

        result = runner.run(record)
        # pytest --version should succeed
        assert result.exit_code == 0
        assert "pytest" in result.stdout.lower() or "pytest" in result.stderr.lower()


class TestSSHRunner:
    def test_ssh_runner_builds_command(self):
        runner = SSHRunner(node="n315", remote_workdir="/tmp/test_jobs")
        spec = JobSpec(cmd="echo hello", cores=2, timeout=60)
        cmd = runner._build_remote_command(spec)
        assert "mkdir -p /tmp/test_jobs" in cmd
        assert "echo hello" in cmd

    def test_ssh_runner_pytest_command(self):
        runner = SSHRunner(node="n330")
        spec = JobSpec(pytest_target="tests/test_foo.py", cores=4, timeout=120)
        cmd = runner._build_remote_command(spec)
        assert "pytest -v tests/test_foo.py" in cmd


# ---------------------------------------------------------------------------
# Ledger tests
# ---------------------------------------------------------------------------

class TestJobLedger:
    def test_add_and_get(self, tmp_ledger_path):
        ledger = JobLedger(tmp_ledger_path)
        spec = JobSpec(cmd="echo", cores=2)
        record = JobRecord(job_id="cj_led1", spec=spec, submitted_at=time.time())

        ledger.add(record)
        retrieved = ledger.get("cj_led1")
        assert retrieved is not None
        assert retrieved.job_id == "cj_led1"

    def test_persistence(self, tmp_ledger_path):
        ledger1 = JobLedger(tmp_ledger_path)
        spec = JobSpec(cmd="echo", cores=2)
        record = JobRecord(job_id="cj_led2", spec=spec, submitted_at=time.time())
        ledger1.add(record)

        # Load in new ledger instance
        ledger2 = JobLedger(tmp_ledger_path)
        retrieved = ledger2.get("cj_led2")
        assert retrieved is not None
        assert retrieved.job_id == "cj_led2"

    def test_list_by_status(self, tmp_ledger_path):
        ledger = JobLedger(tmp_ledger_path)

        spec1 = JobSpec(cmd="echo", cores=1)
        spec2 = JobSpec(cmd="echo", cores=2)

        r1 = JobRecord(job_id="cj_led3", spec=spec1, status=JobStatus.COMPLETED, submitted_at=1.0)
        r2 = JobRecord(job_id="cj_led4", spec=spec2, status=JobStatus.PENDING, submitted_at=2.0)

        ledger.add(r1)
        ledger.add(r2)

        pending = ledger.list(status=JobStatus.PENDING)
        assert len(pending) == 1
        assert pending[0].job_id == "cj_led4"

        all_jobs = ledger.list()
        assert len(all_jobs) == 2

    def test_stats(self, tmp_ledger_path):
        ledger = JobLedger(tmp_ledger_path)

        spec = JobSpec(cmd="echo", cores=1)
        r1 = JobRecord(job_id="cj_led5", spec=spec, status=JobStatus.COMPLETED,
                       submitted_at=1.0, cpu_seconds=5.0, wait_s=2.0)
        r2 = JobRecord(job_id="cj_led6", spec=spec, status=JobStatus.FAILED,
                       submitted_at=2.0, cpu_seconds=1.0, wait_s=0.5)

        ledger.add(r1)
        ledger.add(r2)

        stats = ledger.stats()
        assert stats["total_jobs"] == 2
        assert stats["by_status"]["completed"] == 1
        assert stats["by_status"]["failed"] == 1
        assert stats["total_cpu_seconds"] == 6.0
        assert stats["total_wait_seconds"] == 2.5

    def test_update(self, tmp_ledger_path):
        ledger = JobLedger(tmp_ledger_path)
        spec = JobSpec(cmd="echo", cores=1)
        record = JobRecord(job_id="cj_led7", spec=spec, status=JobStatus.PENDING, submitted_at=time.time())

        ledger.add(record)
        updated = replace(record, status=JobStatus.RUNNING)
        ledger.update(updated)

        retrieved = ledger.get("cj_led7")
        assert retrieved.status == JobStatus.RUNNING


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestClusterExec:
    def test_cluster_exec_accepts_node_but_ignores(self):
        """cluster_exec accepts deprecated node parameter but ignores it."""
        # This should not raise even with node specified
        result = cluster_exec(cmd="echo hello", cores=1, timeout=10, node="n315")
        assert result.exit_code == 0

    def test_cluster_exec_with_pytest_target(self):
        result = cluster_exec(pytest_target="--version", cores=1, timeout=30)
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# ComputeQueue integration tests
# ---------------------------------------------------------------------------

class TestComputeQueue:
    def test_submit_and_run(self, tmp_ledger_path):
        queue = ComputeQueue(ledger_path=tmp_ledger_path)
        spec = JobSpec(cmd="echo integration_test", cores=1, timeout=30)

        result = queue.submit_and_run(spec)
        assert result.exit_code == 0
        assert "integration_test" in result.stdout

    def test_submit_returns_record(self, tmp_ledger_path):
        queue = ComputeQueue(ledger_path=tmp_ledger_path)
        spec = JobSpec(cmd="echo", cores=1, timeout=10)

        record = queue.submit(spec)
        assert record.status == JobStatus.PENDING
        assert queue.ledger.get(record.job_id) is not None

    def test_list_jobs(self, tmp_ledger_path):
        queue = ComputeQueue(ledger_path=tmp_ledger_path)
        spec = JobSpec(cmd="echo", cores=1, timeout=10)

        queue.submit(spec)
        jobs = queue.list_jobs()
        assert len(jobs) >= 1

    def test_stats(self, tmp_ledger_path):
        queue = ComputeQueue(ledger_path=tmp_ledger_path)
        stats = queue.stats()
        assert "total_jobs" in stats


# ---------------------------------------------------------------------------
# JobRecord tests
# ---------------------------------------------------------------------------

class TestJobRecord:
    def test_to_dict_and_from_dict(self):
        spec = JobSpec(cmd="echo", cores=2, timeout=60, tags={"env": "test"})
        record = JobRecord(
            job_id="cj_rec1",
            spec=spec,
            status=JobStatus.COMPLETED,
            placement=Placement.LOCAL,
            submitted_at=1000.0,
            started_at=1001.0,
            finished_at=1005.0,
            cpu_seconds=3.5,
            wait_s=1.0,
            exit_code=0,
        )

        d = record.to_dict()
        restored = JobRecord.from_dict(d)

        assert restored.job_id == record.job_id
        assert restored.status == record.status
        assert restored.placement == record.placement
        assert restored.cpu_seconds == record.cpu_seconds
        assert restored.wait_s == record.wait_s
        assert restored.spec.cmd == record.spec.cmd
        assert restored.spec.tags == record.spec.tags


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_cmd_and_pytest_raises(self):
        with pytest.raises(ValueError):
            JobSpec()

    def test_very_long_timeout(self):
        spec = JobSpec(cmd="echo", timeout=7200)
        assert spec.timeout == 7200

    def test_single_core_job(self):
        spec = JobSpec(cmd="echo", cores=1)
        assert spec.cores == 1

    def test_max_cores_within_limit(self):
        spec = JobSpec(cmd="echo", cores=LOCAL_CORE_CAP)
        record = submit_job(spec)
        assert record.status == JobStatus.PENDING

    def test_scheduler_with_no_probes(self):
        scheduler = Scheduler()
        spec = JobSpec(cmd="echo", cores=4)
        record = JobRecord(job_id="cj_edge1", spec=spec, submitted_at=time.time())

        with patch.object(scheduler, "probe_local") as mock_probe:
            mock_probe.return_value = CapacityProbe(
                node_id="local-99",
                total_cores=128,
                available_cores=20,
                total_mem_G=512.0,
                available_mem_G=400.0,
                is_healthy=True,
            )
            decision = scheduler.place_job(record)
            assert decision.placement == Placement.LOCAL

    def test_unhealthy_node_skipped(self):
        scheduler = Scheduler(local_cap=24)
        scheduler._probes["n315"] = CapacityProbe(
            node_id="n315", total_cores=64, available_cores=48,
            total_mem_G=256.0, available_mem_G=200.0, is_healthy=False,
        )

        spec = JobSpec(cmd="echo", cores=32)
        record = JobRecord(job_id="cj_edge2", spec=spec, submitted_at=time.time())

        decision = scheduler.place_job(record)
        # n315 is unhealthy, so should try n330 or queue
        assert decision.placement != Placement.NODE_N315 or "unhealthy" in decision.reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
