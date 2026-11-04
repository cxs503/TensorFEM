"""Cook membrane stress audit: independent Q8 reference against the production Q4 field.

Writes ``benchmarks/cook_membrane_stress_reference.json``: the independent quadratic
(Q8) reference solution, its Richardson/GCI uncertainty, the convergence of the
published displacement probe, the loaded-edge profile comparison, the pointwise
interior stress comparison against ``tensorfem.additional_benchmark_fields.cook_fields``
(strict interior ladder plus the legacy interface-coincident points with their
two-sided bracket), and the accuracy of the *stored* element-centre field with the
mesh a 3 % gate would need.

The pointwise comparison runs twice:

* ``comparison`` - the *strict* ladder on :data:`tensorfem.cook_stress_reference.INTERIOR_POINTS`,
  whose parent coordinates keep a documented margin from every element interface on all
  meshes, so the sampled Q4 stress is single-valued and the error sequence is a genuine
  discretisation error;
* ``comparison_legacy_points`` - the original five points, which coincide with element
  interfaces on every dyadic mesh.  Their two-sided von Mises bracket is reported so the
  ambiguity of the one-sided sample is measured instead of being hidden.

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
PLATFORM_LEVELS = (8, 16, 32, 64, 128)
LEGACY_POINTS = tuple(reference.SAMPLE_POINTS)
INTERIOR_POINTS = tuple(reference.INTERIOR_POINTS)
#: Margin (in cells) that a "strict interior" sample point must keep from an interface.
REQUIRED_MARGIN_CELLS = 0.1


def loaded_edge_profile(nodes: np.ndarray, displacement: np.ndarray) -> dict:
    """Vertical displacement along the loaded edge x = 48, ordered by y."""
    solution = np.asarray(displacement).reshape(-1, 2)
    selected = np.isclose(nodes[:, 0], 48.0)
    y = nodes[selected, 1]
    v = solution[selected, 1]
    order = np.argsort(y)
    return {"y": y[order].tolist(), "v": v[order].tolist()}


def main() -> dict:
    union = LEGACY_POINTS + INTERIOR_POINTS
    reference_solution = reference.build_reference(REFERENCE_LEVELS, union)
    solutions = reference_solution.pop("solutions")
    finest = solutions[-1]

    platform_cases = [cook_fields(level) for level in PLATFORM_LEVELS]

    comparison = reference.compare_platform(
        platform_cases, reference_solution, sample=INTERIOR_POINTS
    )
    legacy_comparison = reference.compare_platform(
        platform_cases, reference_solution, sample=LEGACY_POINTS
    )
    margins = {level: reference.sample_margins(union, level) for level in PLATFORM_LEVELS}
    interior_margin = min(
        reference.sample_margins(INTERIOR_POINTS, level)["min_margin_cells"]
        for level in PLATFORM_LEVELS
    )
    legacy_margin = max(
        reference.sample_margins(LEGACY_POINTS, level)["min_margin_cells"]
        for level in PLATFORM_LEVELS
    )

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
            reference.strain_recovery_agreement(level) for level in (8, 16, 64)
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
    # strict interior ladder above stays as the arbitrary-point criterion.  Scope kept
    # identical to the merged closure pass (32, 64): the negative control at n = 32 and
    # the scoped pass at n = 64 are the committed evidence contract.
    centre_accuracy = [
        reference.element_centre_accuracy(case, finest)
        for case in platform_cases if case["mesh"] in (32, 64)
    ]
    centre_convergence = reference.convergence_analysis(
        [entry["divisions"] for entry in centre_accuracy],
        [entry["all_elements"]["p95_relative_error"] for entry in centre_accuracy],
    )
    bracket_convergence = reference.convergence_analysis(
        [case["divisions"] for case in comparison["cases"]],
        [case["max_bracket_error"] for case in comparison["cases"]],
    )
    convergence = reference.convergence_prediction(
        list(PLATFORM_LEVELS),
        [entry["max_bracket_error"] for entry in comparison["cases"]],
    )
    edge_agreement = reference.loaded_edge_agreement(finest, platform_cases[-1])

    profiles = {
        "reference_levels": [
            {"divisions": solution["divisions"],
             **loaded_edge_profile(solution["nodes"], solution["displacement"])}
            for solution in solutions
        ],
        "platform_levels": [
            {"divisions": case["mesh"],
             **loaded_edge_profile(np.asarray(case["nodes"]), case["displacement"])}
            for case in platform_cases
        ],
    }
    profiles["agreement_finest"] = reference.profile_agreement(
        profiles["platform_levels"][-1], profiles["reference_levels"][-1]
    )

    interface_ambiguity = {
        "legacy_points": legacy_comparison["ambiguity_finest"],
        "legacy_max_margin_cells": float(legacy_margin),
        "interior_min_margin_cells": float(interior_margin),
        "required_margin_cells": REQUIRED_MARGIN_CELLS,
        "interior_points_clear_interfaces": bool(interior_margin >= REQUIRED_MARGIN_CELLS),
        "legacy_points_on_interface": bool(legacy_margin <= reference.INTERFACE_TOLERANCE),
        "margins_by_level": {str(level): entry for level, entry in margins.items()},
    }

    qualified = bool(
        comparison["qualified"]
        and cross_check["passed"]
        and strain_ok
        and interface_ambiguity["interior_points_clear_interfaces"]
    )
    # Three-state vocabulary of the 2026-11 closure pass (main/m2814, already merged):
    # "withheld" = tool/method untrusted, "measured-not-qualified" = tool trusted, the
    # measured accuracy misses the 3 % gate, "qualified" = certified.  The m2773 strict
    # interior ladder refines the compound ``qualified`` condition; the fail-closed
    # direction is unchanged.
    if not (cross_check["passed"] and strain_ok):
        status = "withheld"
    elif qualified:
        status = "qualified"
    else:
        status = "measured-not-qualified"

    evidence = {
        "case": "Cook membrane stress field",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "environment": {"tensorfem": __version__, "python": host.python_version(),
                        "numpy": np.__version__},
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
            # ``points`` keeps the merged contract: exactly the five legacy sample
            # points, which ``tensorfem.cook_probe_protocol.canonical_points`` and its
            # guard tests consume.  The strict interior points of the 2026-11-03
            # verification-debt closure live under their own key.
            "points": [
                point for point in reference_solution["points"]
                if (round(float(point["xi"]), 12), round(float(point["eta"]), 12))
                in {(round(xi, 12), round(eta, 12)) for xi, eta in LEGACY_POINTS}
            ],
            "interior_points": [
                point for point in reference_solution["points"]
                if (round(float(point["xi"]), 12), round(float(point["eta"]), 12))
                in {(round(xi, 12), round(eta, 12)) for xi, eta in INTERIOR_POINTS}
            ],
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
            "levels": list(PLATFORM_LEVELS),
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
        "comparison_legacy_points": legacy_comparison,
        "interface_ambiguity": interface_ambiguity,
        "convergence": {
            "arbitrary_point_max_von_mises": bracket_convergence,
            "element_centre_p95": centre_convergence,
            **convergence,
        },
        "recovery_cross_check": cross_check,
        "strain_matrix_verification": strain_matrix_verification,
        "loaded_edge_profiles": profiles,
        "loaded_edge_agreement": edge_agreement,
        "element_centre_accuracy": centre_accuracy,
        "status": status,
        "scope": (
            "Pointwise interior stress components and von Mises at fixed material points "
            "(strict interior criterion on points that keep a documented margin from every "
            "element interface, plus the legacy interface-coincident points with their "
            "two-sided bracket) and the stored element-centre von Mises field (production "
            "artefact criterion), both against an independently coded Q8 plane-stress "
            "solution with Richardson extrapolation. Element-centre/nodal peak values on "
            "the loaded-edge traction discontinuity and at the clamped corner are excluded "
            "from the pointwise criteria: those are singular and do not converge "
            "pointwise. The reported element-centre measures are a scoped measurement, not "
            "a change of the production ``stress_accuracy_qualified`` flag. The strict "
            "ladder uses parent points that stay at least "
            f"{REQUIRED_MARGIN_CELLS} cell inside an element on every mesh of the ladder."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(evidence, indent=2) + "\n")

    probe = reference_solution["probe"][-1]
    print(f"reference probe h={REFERENCE_LEVELS[-1]}: {probe['value']:.6f} "
          f"(published {probe['reference']}, error {100 * probe['relative_error']:.3f}%)")
    print(f"reference GCI(finest) {100 * comparison['reference_gci_relative']:.3f}%")
    print(f"interior margin {interior_margin:.3f} cells, legacy margin {legacy_margin:.2e} cells")
    for entry in comparison["cases"]:
        print(f"Q4 {entry['divisions']:>3}: max von Mises bracket error "
              f"{100 * entry['max_bracket_error']:.3f}% passed={entry['passed']}")
    print(f"order {convergence['observed_order']:.3f}, predicted divisions for "
          f"{100 * convergence['target']:.0f}% = {convergence['predicted_divisions']}")
    print("monotone", comparison["monotone"], "qualified", comparison["qualified"])
    print("strain-matrix verification passed", strain_ok)
    print("recovery cross-check max relative difference "
          f"{cross_check['max_relative_difference']:.3e} passed={cross_check['passed']}")
    print("loaded-edge profile agreement (Q4 finest vs Q8 finest) "
          f"{100 * profiles['agreement_finest']['max_relative_difference']:.4f}%")
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
    print(f"arbitrary-point bracket order p={bracket_convergence['observed_order']:.3f}, "
          f"divisions for 3% = {bracket_convergence['divisions_for_tolerance']:.0f}; "
          f"element-centre p95 order p={centre_convergence['observed_order']:.3f}")
    print("status", status)
    print("written", OUTPUT.relative_to(ROOT))
    return evidence


if __name__ == "__main__":
    main()
