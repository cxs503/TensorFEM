"""Verified evidence aggregation for the TensorFEM 1.0 readiness contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .nonlinear_shell_robustness import validate_nonlinear_shell_robustness
from .panel_generation_readiness import panel_generation_readiness
from .shell_sparse_scaling import verify_sparse_scaling_report
from .v1_readiness import evaluate_v1_readiness, validate_v1_readiness


SCHEMA = "tensorfem.v1-evidence-aggregate/1.0"
RELEASE_SCHEMA = "tensorfem.release-quality-evidence/1.0"
CONTACT_SCHEMAS = {
    "tensorfem.curved-surface-contact3d-qualification/1.0",
    "tensorfem.frictional-surface-path-qualification/1.0",
    "tensorfem.frictional-surface-mesh-qualification/1.0",
    "tensorfem.curved-finite-strain-friction-qualification/1.0",
    "tensorfem.general-contact3d-composite/1.0",
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _read_hashed(path: str | Path, schemas: set[str]) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"evidence cannot be read: {path}: {error}") from error
    if payload.get("schema") not in schemas:
        raise ValueError(f"unsupported evidence schema: {payload.get('schema')}")
    digest = payload.pop("evidence_sha256", None)
    if digest != hashlib.sha256(_canonical(payload).encode()).hexdigest():
        raise ValueError("evidence SHA-256 mismatch")
    return {**payload, "evidence_sha256": digest}


def _read_contact_hashed(path: str | Path) -> dict[str, object]:
    """Verify contact physics hashes; runners may append wall-clock metadata."""
    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"evidence cannot be read: {path}: {error}") from error
    if payload.get("schema") not in CONTACT_SCHEMAS:
        raise ValueError(f"unsupported evidence schema: {payload.get('schema')}")
    digest = payload.get("evidence_sha256")
    body = {key: value for key, value in payload.items()
            if key != "evidence_sha256"}
    candidates = [body]
    if "wall_time_seconds" in body:
        candidates.append({key: value for key, value in body.items()
                           if key != "wall_time_seconds"})
    if digest not in {hashlib.sha256(_canonical(item).encode()).hexdigest()
                      for item in candidates}:
        raise ValueError("evidence SHA-256 mismatch")
    return payload


def _contact_scope_and_error(report: Mapping[str, object]) -> tuple[object, object]:
    schema = report["schema"]
    if schema == "tensorfem.general-contact3d-composite/1.0":
        return report.get("scope"), report.get("maximum_relative_error")
    scope = report.get("general_surface_to_surface")
    errors: list[float] = []
    rows = report.get("mesh_sequence")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, Mapping):
                errors.extend(float(row[name]) for name in (
                    "relative_error", "normal_relative_error",
                    "coulomb_relative_error") if row.get(name) is not None)
    interchange = report.get("master_slave_interchange_relative_error")
    if interchange is not None:
        errors.append(float(interchange))
    measured = max(errors) if errors else report.get(
        "maximum_relative_error",
        report.get("complete_newton_role_exchange_relative_error"))
    if schema == "tensorfem.curved-finite-strain-friction-qualification/1.0":
        scope = "qualified_curved_nonmatching_finite_strain_friction_subset"
    return scope, measured


def _validate_release_evidence(report: Mapping[str, object], source: Path) -> None:
    executions = report.get("executions")
    if not isinstance(executions, Mapping):
        raise ValueError("release evidence has no executable gate records")
    for name in ("full_regression", "api_audit", "offline_install"):
        execution = executions.get(name)
        if not isinstance(execution, Mapping):
            raise ValueError(f"release evidence has no {name} execution")
        digest = execution.get("output_sha256")
        command = execution.get("command")
        log_name = execution.get("log_path")
        if (not isinstance(digest, str) or len(digest) != 64
                or not isinstance(command, list) or not command
                or not isinstance(log_name, str)):
            raise ValueError(f"invalid {name} execution provenance")
        log_path = Path(log_name)
        if log_path.is_absolute() or ".." in log_path.parts:
            raise ValueError(f"unsafe {name} execution log path")
        try:
            log_bytes = (source.parent/log_path).read_bytes()
        except OSError as error:
            raise ValueError(f"missing {name} execution log") from error
        if hashlib.sha256(log_bytes).hexdigest() != digest:
            raise ValueError(f"{name} execution log hash mismatch")
        executed_pass = execution.get("returncode") == 0 and bool(
            execution.get("passed"))
        if bool(report.get(name)) != executed_pass:
            raise ValueError(f"release {name} status is inconsistent")
    derived = all(bool(report.get(name)) for name in (
        "full_regression", "api_audit", "offline_install"))
    if bool(report.get("passed")) != derived:
        raise ValueError("release aggregate status is inconsistent")


def aggregate_v1_evidence(
    *, sparse_report: str | Path | None,
    release_report: str | Path | None,
    contact_report: str | Path | None,
    nonlinear_shell_report: str | Path | None = None,
    panel_directories: Mapping[int, str | Path],
) -> dict[str, object]:
    """Map only hash/schema-verified artifacts into ``evaluate_v1_readiness``."""
    capabilities: dict[str, dict[str, object]] = {}
    sources: dict[str, dict[str, object]] = {}

    if sparse_report is not None and Path(sparse_report).exists():
        sparse = verify_sparse_scaling_report(sparse_report)
        metrics = sparse.get("qualified_metrics") or {}
        capabilities["shell_sparse_scalability"] = {
            "passed": bool(sparse.get("passed")),
            "maximum_relative_error": metrics.get("solution_relative_error"),
            "restart_or_rollback": True,
            "qualified_dofs": metrics.get("active_dofs", 0),
            "storage_reduction": metrics.get("memory_reduction_vs_superlu", 0.),
        }
        sources["shell_sparse_scalability"] = {
            "path": str(sparse_report), "schema": sparse["schema"],
            "sha256": sparse["evidence_sha256"]}

    if release_report is not None and Path(release_report).exists():
        release = _read_hashed(release_report, {RELEASE_SCHEMA})
        _validate_release_evidence(release, Path(release_report))
        full_regression = bool(release.get("full_regression"))
        offline_install = bool(release.get("offline_install"))
        api_audit = bool(release.get("api_audit"))
        capabilities["release_quality"] = {
            # Never trust a hand-authored aggregate ``passed`` flag. Derive
            # qualification solely from the three persisted workflow results.
            "passed": full_regression and offline_install and api_audit,
            "maximum_relative_error": 0., "restart_or_rollback": True,
            "full_regression": full_regression,
            "offline_install": offline_install,
            "api_audit": api_audit,
        }
        sources["release_quality"] = {
            "path": str(release_report), "schema": release["schema"],
            "sha256": release["evidence_sha256"]}

    if contact_report is not None and Path(contact_report).exists():
        contact = _read_contact_hashed(contact_report)
        scope, maximum_error = _contact_scope_and_error(contact)
        capabilities["general_double_deformable_contact_3d"] = {
            "passed": bool(contact.get("passed")),
            "maximum_relative_error": maximum_error,
            "restart_or_rollback": bool(contact.get("rollback_exact")),
            "scope": scope,
        }
        sources["general_double_deformable_contact_3d"] = {
            "path": str(contact_report), "schema": contact["schema"],
            "sha256": contact["evidence_sha256"]}

    if (nonlinear_shell_report is not None
            and Path(nonlinear_shell_report).exists()):
        try:
            nonlinear = json.loads(Path(nonlinear_shell_report).read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("nonlinear Shell4 evidence cannot be read: "
                             f"{nonlinear_shell_report}: {error}") from error
        validate_nonlinear_shell_robustness(nonlinear)
        capabilities["nonlinear_shell_robustness"] = {
            "passed": bool(nonlinear["passed"]),
            "maximum_relative_error": nonlinear["maximum_relative_error"],
            "restart_or_rollback": nonlinear[
                "caller_and_rejected_state_rollback_exact"],
            "difficult_step_improved": nonlinear["difficult_step_improved"],
        }
        sources["nonlinear_shell_robustness"] = {
            "path": str(nonlinear_shell_report),
            "schema": nonlinear["schema"],
            "sha256": nonlinear["evidence_sha256"],
        }

    panel_manifests = []
    for divisions in (4, 8, 12):
        directory = panel_directories.get(divisions)
        if directory is None or not Path(directory).exists():
            continue
        readiness = panel_generation_readiness(directory)
        clean = {key: value for key, value in readiness.items()
                 if key != "evidence_sha256"}
        if readiness.get("schema") != "tensorfem.panel-generation-readiness/1.0":
            raise ValueError("invalid panel readiness schema")
        if readiness.get("evidence_sha256") != hashlib.sha256(
                _canonical(clean).encode()).hexdigest():
            raise ValueError("panel readiness SHA-256 mismatch")
        panel_manifests.append({
            "divisions": divisions,
            "peak_force_n": readiness.get("peak_force_n"),
            "peak_confirmed": readiness.get("peak_confirmed", False),
            "post_peak_observed": readiness.get("post_peak_observed", False),
        })

    readiness_report = evaluate_v1_readiness(capabilities, panel_manifests)
    validate_v1_readiness(readiness_report)
    clean = {"schema": SCHEMA, "sources": sources,
             "readiness": readiness_report}
    return {**clean, "evidence_sha256": hashlib.sha256(
        _canonical(clean).encode()).hexdigest()}


def validate_v1_evidence_aggregate(report: Mapping[str, object]) -> Mapping[str, object]:
    if report.get("schema") != SCHEMA:
        raise ValueError("invalid v1 evidence aggregate schema")
    clean = {key: value for key, value in report.items() if key != "evidence_sha256"}
    if report.get("evidence_sha256") != hashlib.sha256(_canonical(clean).encode()).hexdigest():
        raise ValueError("v1 evidence aggregate hash mismatch")
    validate_v1_readiness(report["readiness"])
    return report
