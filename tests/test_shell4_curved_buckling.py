"""Audit tests for curved-geometry Shell4 geometric stiffness.

The audit verdict is negative, so the decisive test is *fail-closed*: it pins the
fact that the faceted-cylinder buckling factor does not converge to the classical
ring load and would break if someone silently changed that physics.
"""

import pytest
import torch

from tensorfem.shell4_curved_buckling import (
    assemble_shell4_buckling_system,
    cylinder_pressure_buckling,
    cylindrical_shell4_mesh,
    dominant_circumferential_wave,
    generalized_buckling_load_factor,
    long_shell_pressure_society_form,
    long_tube_buckling_pressure,
    ring_pressure_wave_factors,
    run_shell4_curved_buckling_qualification,
    simply_supported_plate_cross_check,
)

YOUNG, NU, R, T = 210e9, 0.3, 1.0, 0.01


def test_long_tube_reference_matches_classification_society_form():
    ring = long_tube_buckling_pressure(R, T, YOUNG, NU)
    society = long_shell_pressure_society_form(YOUNG, NU, T, 2 * R)
    assert abs(society - ring) / ring < 1e-14
    # (n^2 - 1) scaling: p(3)/p(2) = 8/3, p(4)/p(2) = 5
    factors = ring_pressure_wave_factors((2, 3, 4))
    assert abs(factors[2] - 1.0) < 1e-15
    assert abs(factors[3] - 8.0 / 3.0) < 1e-15
    assert abs(factors[4] - 5.0) < 1e-15
    for wave, factor in factors.items():
        ratio = long_tube_buckling_pressure(R, T, YOUNG, NU, wave) / ring
        assert abs(ratio - factor) < 1e-12


def test_reference_inputs_are_validated():
    with pytest.raises(ValueError):
        long_tube_buckling_pressure(-1.0, T, YOUNG, NU)
    with pytest.raises(ValueError):
        long_tube_buckling_pressure(R, T, YOUNG, NU, wave=1)
    with pytest.raises(ValueError):
        cylindrical_shell4_mesh(R, 1.0, n_circ=4)


def test_generic_solver_reproduces_the_qualified_flat_plate_path():
    """The curved results must rest on an eigensolver checked on known ground."""
    report = simply_supported_plate_cross_check(nx=6, ny=6)
    assert report["generic_vs_flat_relative"] < 1e-8
    assert abs(report["production_flat_path"] - report["navier"]) / report["navier"] < 0.03


def test_curved_mesh_is_planar_gap_free_and_annihilates_rigid_rotation():
    nodes, elements = cylindrical_shell4_mesh(R, 2.0, n_circ=12, n_axial=2)
    assert len(nodes) == 12 * 3 and len(elements) == 24
    # Gap-free: every non-boundary edge is shared by exactly two facets.
    edges = {}
    for conn in elements.tolist():
        for a, b in zip(conn, conn[1:] + conn[:1]):
            edges[tuple(sorted((a, b)))] = edges.get(tuple(sorted((a, b))), 0) + 1
    interior = {(a, b): count for (a, b), count in edges.items()
                if nodes[a, 2] > 0 and nodes[a, 2] < 2.0
                and nodes[b, 2] > 0 and nodes[b, 2] < 2.0}
    assert interior and all(count == 2 for count in interior.values())
    K, Kg = assemble_shell4_buckling_system(nodes, elements, YOUNG, NU, T, hoop=1.0)
    # Rigid rotation about the cylinder axis: zero elastic and zero geometric energy.
    rotation = torch.zeros(6 * len(nodes), dtype=torch.float64)
    for node, (x, y, _z) in enumerate(nodes.tolist()):
        rotation[6 * node + 0] = -y
        rotation[6 * node + 1] = x
        rotation[6 * node + 5] = 1.0        # rigid spin about the axis
    scale = float(torch.linalg.vector_norm(rotation))
    elastic = float(rotation @ K @ rotation) / scale**2
    geometric = float(rotation @ Kg @ rotation) / scale**2
    stiffest = float(torch.linalg.matrix_norm(K, ord=2))
    assert abs(elastic) / stiffest < 1e-10
    assert abs(geometric) < 1e-10 * max(1.0, stiffest)


def test_hoop_prestress_operator_is_positive_semidefinite_and_tension_is_rejected():
    nodes, elements = cylindrical_shell4_mesh(R, 2.0, n_circ=12, n_axial=2)
    _K, compression = assemble_shell4_buckling_system(
        nodes, elements, YOUNG, NU, T, hoop=1.0)
    eigenvalues = torch.linalg.eigvalsh(compression)
    assert float(eigenvalues[0]) > -1e-10 * float(eigenvalues[-1])
    with pytest.raises(ValueError):
        cylinder_pressure_buckling(R, 2.0, YOUNG, NU, T, n_circ=12, pressure=-1.0)
    with pytest.raises(RuntimeError):
        _, tension = assemble_shell4_buckling_system(
            nodes, elements, YOUNG, NU, T, hoop=-1.0)
        free = torch.arange(6 * len(nodes), dtype=torch.long)
        generalized_buckling_load_factor(_K, tension, free)


def test_faceted_cylinder_lowest_mode_is_facet_scale_and_degrades_with_refinement():
    """Fail-closed: the open curved-geometric-stiffness debt stays open."""
    exact = long_tube_buckling_pressure(R, T, YOUNG, NU)
    coarse = cylinder_pressure_buckling(R, 4.0, YOUNG, NU, T, n_circ=16, n_axial=2)
    fine = cylinder_pressure_buckling(R, 4.0, YOUNG, NU, T, n_circ=24, n_axial=2)
    for result, n_circ in ((coarse, 16), (fine, 24)):
        assert abs(result.load_factor - exact) / exact > 0.5
        assert dominant_circumferential_wave(result) == n_circ // 2
        # The mode carries almost no elastic energy: a facet-scale near-mechanism.
        assert result.diagnostics["elastic_fraction"] < 1e-3
    # Refinement must not look like convergence.
    assert coarse.load_factor / fine.load_factor > 1.0


def test_qualification_verdict_is_locked_fail_closed():
    report = run_shell4_curved_buckling_qualification()
    assert report["flat_plate_solver_cross_check"]["generic_vs_flat_relative"] < 1e-8
    assert report["society_form_relative_gap"] < 1e-12
    assert report["passed"] is False
    assert report["verdict"] == "curved-shell-geometric-stiffness-unqualified"
    assert report["worst_absolute_relative_error"] > report["tolerance"]
    assert report["monotone_degradation"] is True
    assert all(row["circumferential_wave"] == row["nyquist_wave"] for row in report["meshes"])
    assert all(row["elastic_fraction"] < 1e-3 for row in report["meshes"])

def test_documented_tables_are_machine_generated_from_the_evidence_json():
    """No hand-transcribed numbers: the doc tables must match the committed run."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    evidence = json.loads(
        (root / "docs/assets/shell4-curved-buckling/evidence.json").read_text())
    doc = (root / "docs/shell4-curved-geometric-stiffness.md").read_text()
    qualification = evidence["qualification"]

    assert qualification["passed"] is False
    assert qualification["verdict"] == "curved-shell-geometric-stiffness-unqualified"
    for row in qualification["meshes"]:
        expected = (f"| {row['n_circ']} | {row['dof']} | {row['pressure']:.4g} | "
                    f"{100 * row['relative_error']:+.2f} % | {row['circumferential_wave']} "
                    f"(Nyquist {row['nyquist_wave']}) | {row['elastic_fraction']:.2e} |")
        assert expected in doc, expected
    # Reference pressure and the classification-society identity are quoted too.
    assert f"{qualification['reference_pressure']:.6g}" in doc
    assert f"{qualification['society_form_relative_gap']:.1e}" in doc
    # The failure studies quoted in the doc must also come from the same run.
    for study in ("ends_study", "axial_subdivision_study", "length_study",
                  "thickness_scaling"):
        for row in qualification[study]:
            assert f"{row['pressure']:.4g}" in doc, (study, row)
