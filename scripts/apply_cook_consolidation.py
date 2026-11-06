"""One-off patcher for the m4718 Cook consolidation (kept as the audit trail).

Applies exact string replacements; every replacement is counted so a silent miss is
impossible.  Run once from the repository root:

    python3 scripts/apply_cook_consolidation.py
"""

from __future__ import annotations

import pathlib
import sys

REPLACEMENTS = {
    "tests/test_cook_oracle_operator.py": [
        (
            'def test_q8_reference_reproduces_published_loaded_edge_midpoint():\n'
            '    """The independent Q8 path must solve the same boundary-value problem as Q4."""\n'
            "    solution = reference.solve_q8(12)",
            'def test_q8_reference_reproduces_published_loaded_edge_midpoint(cook_q8):\n'
            '    """The independent Q8 path must solve the same boundary-value problem as Q4."""\n'
            "    solution = cook_q8(12)",
        ),
        (
            "def test_loaded_edge_profiles_of_the_two_discretisations_agree():\n"
            '    """Q4 and Q8 vertical deflections along the loaded edge must agree to 3 %."""\n'
            "    from tensorfem.additional_benchmark_fields import cook_fields\n"
            "\n"
            "    fields = cook_fields(32)",
            "def test_loaded_edge_profiles_of_the_two_discretisations_agree(cook_q8, cook_case):\n"
            '    """Q4 and Q8 vertical deflections along the loaded edge must agree to 3 %."""\n'
            "    fields = cook_case(32)",
        ),
        (
            "    solution = reference.solve_q8(12)\n"
            "    reference_nodes = solution[\"nodes\"]",
            "    solution = cook_q8(12)\n"
            "    reference_nodes = solution[\"nodes\"]",
        ),
        (
            "def test_platform_q4_pointwise_stress_stays_uncertified():\n"
            '    """Honest negative result: the element-centre Q4 stress is not yet a 3 % match.',
            "def test_platform_q4_pointwise_stress_stays_uncertified(cook_case):\n"
            '    """Honest negative result: the element-centre Q4 stress is not yet a 3 % match.',
        ),
        (
            "    from tensorfem.additional_benchmark_fields import cook_fields\n"
            "\n"
            "    fields = cook_fields(16)\n"
            '    assert fields["stress_accuracy_qualified"] is False',
            '    fields = cook_case(16)\n'
            '    assert fields["stress_accuracy_qualified"] is False',
        ),
        (
            "def test_pointwise_recovery_reproduces_the_stored_q4_centre_stress():\n"
            '    """The ``inf`` cross-check debt is closed: recovery and stored field now agree."""\n'
            "    from tensorfem.additional_benchmark_fields import cook_fields\n"
            "\n"
            "    checked = reference.cross_check_pointwise_recovery(cook_fields(8))\n"
            '    assert checked["passed"] is True\n'
            '    assert checked["max_relative_difference"] < 1e-10\n'
            '    assert math.isfinite(checked["max_relative_difference"])\n'
            '    assert checked["samples"] == 16\n',
            "def test_pointwise_recovery_reproduces_the_stored_q4_centre_stress(cook_case):\n"
            '    """The ``inf`` cross-check debt is closed: recovery and stored field now agree.\n'
            "\n"
            "    Consolidated (m4718): this test evaluates ``cross_check_pointwise_recovery``\n"
            "    on ``cook_fields(8)`` exactly once and now also carries the assertions of\n"
            "    ``test_cook_stress_recovery_verification.py::test_cross_check_of_pointwise\n"
            "_recovery_passes``, which recomputed the same cross-check and asserted a\n"
            "    subset of what is asserted here (see\n"
            "    ``artifacts/cook_assertion_consolidation.json``).\n"
            '    """\n'
            "    check = reference.cross_check_pointwise_recovery(cook_case(8))\n"
            '    assert check["passed"] is True\n'
            '    assert check["max_relative_difference"] < 1e-10\n'
            '    assert math.isfinite(check["max_relative_difference"])\n'
            '    assert check["samples"] == 16\n'
            '    assert check["max_relative_difference"] <= check["tolerance"]\n',
        ),
        (
            "def test_pointwise_recovery_refuses_points_outside_the_material_square():\n"
            '    """A physical point outside the parent square must yield ``None``, never a wrong element."""\n'
            "    from tensorfem.additional_benchmark_fields import cook_fields\n"
            "\n"
            "    fields = cook_fields(8)",
            "def test_pointwise_recovery_refuses_points_outside_the_material_square(cook_case):\n"
            '    """A physical point outside the parent square must yield ``None``, never a wrong element."""\n'
            "    fields = cook_case(8)",
        ),
    ],
    "tests/test_cook_stress_audit_closure.py": [
        (
            "from tensorfem import cook_stress_reference as reference\n"
            "from tensorfem.additional_benchmark_fields import cook_fields\n",
            "from tensorfem import cook_stress_reference as reference\n",
        ),
        (
            "def test_field_stress_at_point_reproduces_the_parent_coordinate_evaluation():\n"
            "    solution = reference.solve_q8(12)",
            "def test_field_stress_at_point_reproduces_the_parent_coordinate_evaluation(cook_q8):\n"
            "    solution = cook_q8(12)",
        ),
        (
            "def test_stored_element_centre_field_converges_under_refinement():\n"
            "    fine_reference = reference.solve_q8(24)\n"
            '    coarse = reference.element_centre_accuracy(cook_fields(16), fine_reference)\n'
            '    fine = reference.element_centre_accuracy(cook_fields(32), fine_reference)',
            "def test_stored_element_centre_field_converges_under_refinement(cook_q8, cook_case):\n"
            "    fine_reference = cook_q8(24)\n"
            '    coarse = reference.element_centre_accuracy(cook_case(16), fine_reference)\n'
            '    fine = reference.element_centre_accuracy(cook_case(32), fine_reference)',
        ),
        (
            "def test_element_centre_gate_fails_on_a_coarse_mesh():\n"
            '    accuracy = reference.element_centre_accuracy(cook_fields(32), reference.solve_q8(24))',
            "def test_element_centre_gate_fails_on_a_coarse_mesh(cook_q8, cook_case):\n"
            '    accuracy = reference.element_centre_accuracy(cook_case(32), cook_q8(24))',
        ),
        (
            "def test_loaded_edge_profiles_of_both_discretisations_agree():\n"
            "    agreement = reference.loaded_edge_agreement(reference.solve_q8(12), cook_fields(16))",
            "def test_loaded_edge_profiles_of_both_discretisations_agree(cook_q8, cook_case):\n"
            "    agreement = reference.loaded_edge_agreement(cook_q8(12), cook_case(16))",
        ),
        (
            "def test_committed_evidence_matches_the_current_recovery_code():\n"
            "    if not EVIDENCE.exists():",
            "def test_committed_evidence_matches_the_current_recovery_code(cook_case):\n"
            "    if not EVIDENCE.exists():",
        ),
        (
            '    live = reference.cross_check_pointwise_recovery(cook_fields(8))',
            '    live = reference.cross_check_pointwise_recovery(cook_case(8))',
        ),
    ],
    "tests/test_cook_stress_reference.py": [
        (
            "def test_q8_solution_carries_the_full_traction_and_converges():\n"
            "    solution = reference.solve_q8(8)",
            "def test_q8_solution_carries_the_full_traction_and_converges(cook_q8):\n"
            "    solution = cook_q8(8)",
        ),
        (
            "def test_q8_reference_is_self_converged_at_the_interior_sample_points():\n"
            "    coarse = reference.solve_q8(12)\n"
            "    fine = reference.solve_q8(24)",
            "def test_q8_reference_is_self_converged_at_the_interior_sample_points(cook_q8):\n"
            "    coarse = cook_q8(12)\n"
            "    fine = cook_q8(24)",
        ),
        (
            "def test_published_deflection_is_confirmed_by_both_discretisations():\n"
            "    from tensorfem.additional_benchmark_fields import cook_fields\n",
            "def test_published_deflection_is_confirmed_by_both_discretisations(cook_q8, cook_case):\n",
        ),
        (
            "    platform = cook_fields(16)",
            "    platform = cook_case(16)",
        ),
        (
            "    independent = reference.probe_displacement(reference.solve_q8(24))",
            "    independent = reference.probe_displacement(cook_q8(24))",
        ),
        (
            "def test_two_sided_bracket_reproduces_the_stored_field_at_element_centres():\n"
            "    from tensorfem.additional_benchmark_fields import cook_fields\n"
            "\n"
            "    case = cook_fields(8)",
            "def test_two_sided_bracket_reproduces_the_stored_field_at_element_centres(cook_case):\n"
            "    case = cook_case(8)",
        ),
        (
            "def test_interface_point_returns_two_sides_with_a_finite_bracket():\n"
            "    from tensorfem.additional_benchmark_fields import cook_fields\n"
            "\n"
            "    case = cook_fields(8)",
            "def test_interface_point_returns_two_sides_with_a_finite_bracket(cook_case):\n"
            "    case = cook_case(8)",
        ),
    ],
    "tests/test_cook_stress_recovery_verification.py": [
        (
            "from tensorfem.additional_benchmark_fields import cook_fields\n",
            "",
        ),
        (
            "def test_cross_check_of_pointwise_recovery_passes():\n"
            "    check = reference.cross_check_pointwise_recovery(cook_fields(8))\n"
            '    assert check["passed"], check\n'
            '    assert check["max_relative_difference"] <= check["tolerance"]\n'
            "\n"
            "\n",
            "",
        ),
        (
            "def test_reference_loaded_edge_profile_matches_production_q4():\n",
            "def test_reference_loaded_edge_profile_matches_production_q4(cook_q8, cook_case):\n",
        ),
        (
            "    solution = reference.solve_q8(8)\n"
            "    case = cook_fields(16)",
            "    solution = cook_q8(8)\n"
            "    case = cook_case(16)",
        ),
    ],
}


def main() -> int:
    failures = 0
    for name, pairs in REPLACEMENTS.items():
        path = pathlib.Path(name)
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            count = text.count(old)
            if count != 1:
                print(f"FAIL {name}: {count} matches for {old[:60]!r}", file=sys.stderr)
                failures += 1
                continue
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
        print(f"{name}: {len(pairs)} replacement(s) applied")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())