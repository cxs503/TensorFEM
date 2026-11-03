"""Cook membrane stress audit: independent Q8 reference against the production Q4 field.

Writes ``benchmarks/cook_membrane_stress_reference.json``: the independent quadratic
(Q8) reference solution, its Richardson/GCI uncertainty, the convergence of the
published displacement probe, the loaded-edge profile comparison, the pointwise interior
stress comparison against ``tensorfem.additional_benchmark_fields.cook_fields``, and the
accuracy of the *stored* element-centre field with the mesh a 3 % gate would need.

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
    finest = solutions[-1]

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

    # Where the accuracy question actually lives: the stored element-centre field is the
    # artefact production writes, so it is scored directly against the Q8 reference.  The
    # five fixed interior sample points above stay as the arbitrary-point criterion.
    centre_accuracy = [
        reference.element_centre_accuracy(case, finest) for case in platform_cases[2:]
    ]
    convergence = reference.convergence_analysis(
        [case["divisions"] for case in comparison["cases"]],
        [case["max_von_mises_error"] for case in comparison["cases"]],
    )
    centre_convergence = reference.convergence_analysis(
        [entry["divisions"] for entry in centre_accuracy],
        [entry["all_elements"]["p95_relative_error"] for entry in centre_accuracy],
    )
    edge_agreement = reference.loaded_edge_agreement(finest, platform_cases[-1])

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

    if cross_check["passed"] and strain_ok:
        status = "qualified" if comparison["qualified"] else "measured-not-qualified"
    else:
        status = "withheld"

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
        "loaded_edge_agreement": edge_agreement,
        "element_centre_accuracy": centre_accuracy,
        "convergence": {
            "arbitrary_point_max_von_mises": convergence,
            "element_centre_p95": centre_convergence,
        },
        "status": status,
        "scope": (
            "Pointwise interior stress components and von Mises at five fixed material points "
            "(arbitrary-point criterion) and the stored element-centre von Mises field "
            "(production artefact criterion), both against an independently coded Q8 plane-stress "
            "solution with Richardson extrapolation. Element-centre/nodal peak values on the "
            "loaded-edge traction discontinuity and at the clamped corner are excluded from the "
            "pointwise criteria: those are singular and do not converge pointwise. The reported "
            "element-centre measures are a scoped measurement, not a change of the production "
            "``stress_accuracy_qualified`` flag."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n")

    probe = reference_solution["probe"][-1]
    print(f"reference probe h={REFERENCE_LEVELS[-1]}: {probe['value']:.6f} "
          f"(published {probe['reference']}, error {100 * probe['relative_error']:.3f}%)")
    print(f"reference GCI(finest) {100 * comparison['reference_gci_relative']:.3f}%")
    for entry in comparison["cases"]:
        print(f"Q4 {entry['divisions']:>3}: arbitrary-point max von Mises error "
              f"{100 * entry['max_von_mises_error']:.3f}% passed={entry['passed']}")
    print("monotone", comparison["monotone"], "qualified", comparison["qualified"])
    print("strain-matrix verification passed", strain_ok)
    print("recovery cross-check max relative difference "
          f"{100 * cross_check['max_relative_difference']:.3f}% passed={cross_check['passed']}")
    for entry in centre_accuracy:
        stats = entry["all_elements"]
        print(f"stored element-centre field n={entry['divisions']:>3}: "
              f"median {100 * stats['median_relative_error']:.3f}% "
              f"p95 {100 * stats['p95_relative_error']:.3f}% "
              f"scale-normalised max {100 * stats['max_scale_normalized_error']:.3f}% "
              f"passed={entry['passed']}")
    print("loaded-edge agreement: max |dv| "
          f"{edge_agreement['max_absolute_difference']:.4f} "
          f"({100 * edge_agreement['max_relative_to_probe']:.3f}% of probe) "
          f"passed={edge_agreement['passed']}")
    print(f"arbitrary-point order p={convergence['observed_order']:.3f}, "
          f"divisions for 3% = {convergence['divisions_for_tolerance']:.0f}; "
          f"element-centre p95 order p={centre_convergence['observed_order']:.3f}")
    print("status", status)
    print("written", OUTPUT.relative_to(ROOT))
    return evidence


if __name__ == "__main__":
    main()