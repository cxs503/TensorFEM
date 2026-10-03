"""TensorFEM-only qualification view for ship and offshore structures."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

from .benchmark_registry import run_registered_benchmarks
from .shell_benchmark_suite import run_classical_shell_suite
from .marine_plate_buckling import run_local_plate_buckling_qualification
from .hull_girder_ultimate import run_hull_girder_ultimate_benchmark
from .marine_fatigue_qualification import run_fatigue_qualification


SCHEMA = "tensorfem.marine-structural-qualification/1.0"
STRUCTURAL_IDS = (
    "truss.bar", "frame.cantilever", "beam.timoshenko", "buckling.euler",
    "modal.cantilever", "solid.hex8", "solid.tet10_patch", "plate.mindlin",
    "shell.patch_energy", "nonlinear.finite_bar", "plasticity.j2",
    "plasticity.tet4_j2", "marine.hull_girder.deflection",
    "marine.hull_girder.moment", "marine.stiffened_panel.navier",
)


def _digest(payload: dict[str, object]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def run_marine_structural_qualification() -> dict[str, object]:
    """Assemble structural evidence without executing or depending on TensorLBM."""
    registry = {item.id: item for item in run_registered_benchmarks()}
    missing = set(STRUCTURAL_IDS) - set(registry)
    if missing:
        raise RuntimeError(f"missing structural evidence: {sorted(missing)}")
    scalar = [asdict(registry[item_id]) for item_id in STRUCTURAL_IDS]
    shells = run_classical_shell_suite(tier="quick")
    local_buckling = run_local_plate_buckling_qualification()
    hull_ultimate = run_hull_girder_ultimate_benchmark()
    fatigue = run_fatigue_qualification()
    categories = {
        "global_hull_and_beam": {"status": "qualified", "evidence": 6},
        "plate_shell_linear_and_large_rotation": {"status": "qualified", "evidence": 7},
        "modal": {"status": "qualified", "evidence": 1},
        "linear_column_buckling": {"status": "qualified", "evidence": 1},
        "material_and_global_plasticity": {"status": "qualified_prototype", "evidence": 3},
        "local_plate_buckling": {"status": "qualified_prototype", "evidence": 3},
        "ultimate_hull_girder_strength": {"status": "qualified_section_prototype", "evidence": 2},
        "fatigue_damage_primitives": {"status": "qualified_prototype", "evidence": 4},
        "fatigue_and_fracture_life": {"status": "not_qualified", "evidence": 0},
    }
    clean: dict[str, object] = {
        "schema": SCHEMA,
        "solver_scope": "TensorFEM only; no TensorLBM or CFD execution",
        "passed": (all(item["passed"] for item in scalar) and bool(shells["passed"])
                   and bool(local_buckling["passed"]) and bool(hull_ultimate["passed"])
                   and bool(fatigue["passed"])),
        "scalar_evidence": scalar,
        "classical_shell_suite": shells,
        "local_plate_buckling": local_buckling,
        "hull_girder_ultimate": hull_ultimate,
        "fatigue_primitives": fatigue,
        "categories": categories,
    }
    return {**clean, "report_hash": _digest(clean)}


def validate_marine_structural_qualification(report: dict[str, object]) -> dict[str, object]:
    if report.get("schema") != SCHEMA or report.get("solver_scope") != "TensorFEM only; no TensorLBM or CFD execution":
        raise ValueError("invalid marine structural qualification schema or scope")
    clean = {key: value for key, value in report.items() if key != "report_hash"}
    if report.get("report_hash") != _digest(clean):
        raise ValueError("marine structural qualification hash mismatch")
    evidence = report.get("scalar_evidence")
    if not isinstance(evidence, list) or {row.get("id") for row in evidence} != set(STRUCTURAL_IDS):
        raise ValueError("marine structural qualification evidence mismatch")
    if report.get("passed") is not True or not all(row.get("passed") is True for row in evidence):
        raise ValueError("marine structural qualification did not pass")
    return report
