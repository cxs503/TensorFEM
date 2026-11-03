"""Independent Q8 reference audit of the production Cook-membrane stress field.

Writes ``benchmarks/cook_membrane_stress_reference.json``: the independent quadratic
(Q8) reference solution, its Richardson/GCI uncertainty, the convergence of the
published displacement probe, the loaded-edge profile comparison (where the traction
discontinuity makes pointwise values non-convergent) and the pointwise interior stress
comparison against ``tensorfem.additional_benchmark_fields.cook_fields``.

Run: ``OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/audit_cook_membrane_stress.py``
"""

from __future__ import annotations

import json
import platform as host
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from tensorfem import __version__
from tensorfem import cook_stress_reference as reference
from tensorfem.additional_benchmark_fields import cook_fields

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "cook_membrane_stress_reference.json"
REFERENCE_LEVELS = (12, 24, 48)
PLATFORM_LEVELS = (8, 16, 32, 64)


def loaded_edge_profile(nodes: np.ndarray, displacement: np.ndarray) -> dict:
    """Vertical displacement along the loaded edge x = 48, ordered by y."""
    solution = np.asarray(displacement).reshape(-1, 2)
    selected = np.isclose(nodes[:, 0], 48.0)
    y = nodes[selected, 1]
    v = solution[selected, 1]
    order = np.argsort(y)
    return {"y": y[order].tolist(), "v": v[order].tolist()}


def main() -> dict:
    reference_solution = reference.build_reference(REFERENCE_LEVELS)
    solutions = reference_solution.pop("solutions")

    platform_cases = []
    for level in PLATFORM_LEVELS:
        case = cook_fields(level)
        platform_cases.append(case)

    comparison = reference.compare_platform(platform_cases, reference_solution)
    cross_check = reference.cross_check_pointwise_recovery(platform_cases[0])

    # Independent verification of the recovery algebra itself: a consistent B matrix
    # reproduces any constant strain state on a distorted element, the Newton inverse
    # mapping must round-trip, and the audit's Q4 recovery must reproduce the stored
    # element-centre field before pointwise comparisons can be trusted.
    strain_matrix_verification = {
        "q4_patch_test": reference.q4_patch_check(reference.distorted_q4_patch()),
        "q8_patch_test": reference.q8_patch_check(reference.distorted_q8_patch()),
        "inverse_map": reference.inverse_map_error(),
        "q4_recovery_vs_stored": [
            reference.strain_recovery_agreement(level) for level in (8, 16)
        ],
    }
    strain_ok = (
        strain_matrix_verification["q4_patch_test"]["max_abs_error"] <= 1e-12
        and strain_matrix_verification["q8_patch_test"]["max_abs_error"] <= 1e-12
        and strain_matrix_verification["inverse_map"]["max_parent_error"] <= 1e-10
        and all(entry["passed"] for entry in strain_matrix_verification["q4_recovery_vs_stored"])
    )
    strain_matrix_verification["passed"] = strain_ok

    profiles = {
        "reference_levels": [
            {"divisions": solution["divisions"], **loaded_edge_profile(solution["nodes"], solution["displacement"])}
            for solution in solutions
        ],
        "platform_levels": [
            {"divisions": case["mesh"], **loaded_edge_profile(np.asarray(case["nodes"]), case["displacement"])}
            for case in platform_cases
        ],
    }

    evidence = {
        "case": "Cook membrane stress field",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"tensorfem": __version__, "python": host.python_version(), "numpy": np.__version__},
        "problem": {
            "corners": reference.CORNERS.tolist(),
            "young": reference.YOUNG,
            "poisson": reference.POISSON,
            "thickness": reference.THICKNESS,
            "loaded_edge_resultant": reference.RESULTANT,
            "loaded_edge_length": reference.LOADED_EDGE_LENGTH,
            "published_probe": reference.PUBLISHED_RESPONSE,
        },
        "reference": {
            "formulation": reference_solution["formulation"],
            "levels": reference_solution["levels"],
            "probe": reference_solution["probe"],
            "points": reference_solution["points"],
            "strain_energy": reference_solution["strain_energy"],
            "solver": [
                {
                    "divisions": solution["divisions"],
                    "nodes": solution["nodes_count"],
                    "elements": solution["elements_count"],
                    "cg_iterations": solution["iterations"],
                    "relative_residual": solution["relative_residual"],
                    "strain_energy": solution["strain_energy"],
                    "load_resultant": solution["load_resultant"],
                }
                for solution in solutions
            ],
        },
        "platform": {
            "levels": PLATFORM_LEVELS,
            "response": [
                {
                    "divisions": case["mesh"],
                    "probe": case["verification"]["response"],
                    "relative_error": case["verification"]["response_relative_error"],
                }
                for case in platform_cases
            ],
        },
        "comparison": comparison,
        "recovery_cross_check": cross_check,
        "strain_matrix_verification": strain_matrix_verification,
        "loaded_edge_profiles": profiles,
        "status": (
            "withheld"
            if not (comparison["qualified"] and cross_check["passed"])
            else "qualified"
        ),
        "scope": (
            "Pointwise interior stress components and von Mises at five fixed material points, "
            "against an independently coded Q8 plane-stress solution with Richardson extrapolation. "
            "Element-centre/nodal peak values on the loaded-edge traction discontinuity and at the "
            "clamped corner are excluded: those are singular and do not converge pointwise."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n")

    probe = reference_solution["probe"][-1]
    print(f"reference probe h={REFERENCE_LEVELS[-1]}: {probe['value']:.6f} "
          f"(published {probe['reference']}, error {100 * probe['relative_error']:.3f}%)")
    print(f"reference GCI(finest) {100 * comparison['reference_gci_relative']:.3f}%")
    for entry in comparison["cases"]:
        print(f"Q4 {entry['divisions']:>3}: max von Mises error "
              f"{100 * entry['max_von_mises_error']:.3f}% passed={entry['passed']}")
    print("monotone", comparison["monotone"], "qualified", comparison["qualified"])
    print("strain-matrix verification passed", strain_ok)
    print("recovery cross-check max relative difference "
          f"{100 * cross_check['max_relative_difference']:.3f}% passed={cross_check['passed']}")
    print("written", OUTPUT.relative_to(ROOT))
    return evidence


if __name__ == "__main__":
    main()