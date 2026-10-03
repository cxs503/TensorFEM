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
from .marine_plate_postbuckling import run_plate_postbuckling_qualification
from .hull_girder_progressive import run_progressive_hull_girder_benchmark
from .marine_fatigue_advanced import run_advanced_fatigue_qualification
from .shell4_buckling import run_shell4_buckling_qualification
from .marine_imperfect_strip import run_imperfect_strip_qualification
from .marine_hotspot_fracture import run_hotspot_fracture_qualification


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
    postbuckling = run_plate_postbuckling_qualification()
    progressive = run_progressive_hull_girder_benchmark()
    advanced_fatigue = run_advanced_fatigue_qualification()
    shell4_buckling = run_shell4_buckling_qualification()
    imperfect_strip = run_imperfect_strip_qualification()
    imperfect_strip = {**imperfect_strip, "final": asdict(imperfect_strip["final"])}
    hotspot_fracture = run_hotspot_fracture_qualification()
    categories = {
        "global_hull_and_beam": {"status": "qualified", "evidence": 6},
        "plate_shell_linear_and_large_rotation": {"status": "qualified", "evidence": 7},
        "modal": {"status": "qualified", "evidence": 1},
        "linear_column_buckling": {"status": "qualified", "evidence": 1},
        "material_and_global_plasticity": {"status": "qualified_prototype", "evidence": 3},
        "local_plate_buckling": {"status": "qualified_prototype", "evidence": 3},
        "ultimate_hull_girder_strength": {"status": "qualified_section_prototype", "evidence": 2},
        "fatigue_damage_primitives": {"status": "qualified_prototype", "evidence": 4},
        "imperfect_plate_postbuckling": {"status": "qualified_reduced_order", "evidence": 3},
        "multi_component_progressive_yielding": {"status": "qualified_section_prototype", "evidence": 1},
        "spectrum_fatigue_and_crack_growth": {"status": "qualified_primitives", "evidence": 4},
        "full_shell_progressive_collapse": {"status": "not_qualified", "evidence": 0},
        "shell4_initial_stress_buckling": {"status": "qualified", "evidence": 1},
        "imperfection_residual_stress_plasticity": {"status": "qualified_reduced_order", "evidence": 3},
        "resultdb_hotspot_and_lefm": {"status": "qualified_primitives", "evidence": 4},
        "shell_arc_length_postbuckling": {"status": "not_qualified", "evidence": 0},
    }
    clean: dict[str, object] = {
        "schema": SCHEMA,
        "solver_scope": "TensorFEM only; no TensorLBM or CFD execution",
        "passed": (all(item["passed"] for item in scalar) and bool(shells["passed"])
                   and bool(local_buckling["passed"]) and bool(hull_ultimate["passed"])
                   and bool(fatigue["passed"]) and bool(postbuckling["passed"])
                   and bool(progressive["passed"]) and bool(advanced_fatigue["passed"])
                   and bool(shell4_buckling["passed"]) and bool(imperfect_strip["passed"])
                   and bool(hotspot_fracture["passed"])),
        "scalar_evidence": scalar,
        "classical_shell_suite": shells,
        "local_plate_buckling": local_buckling,
        "hull_girder_ultimate": hull_ultimate,
        "fatigue_primitives": fatigue,
        "plate_postbuckling": postbuckling,
        "hull_girder_progressive": progressive,
        "advanced_fatigue": advanced_fatigue,
        "shell4_buckling": shell4_buckling,
        "imperfect_strip": imperfect_strip,
        "hotspot_fracture": hotspot_fracture,
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
