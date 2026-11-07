"""Tests for the three-case structural verification face.

The suite is deliberately cheap: it exercises the *contract* (gate constants,
ladder bookkeeping, load audits, published evidence) and the closed forms, not the
certified ladders -- the full ladders live in
``benchmarks/structural_verification_face.py`` and are reproduced by the
committed evidence file, which :func:`test_published_evidence_satisfies_contract`
re-checks against the code's own gate constants.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from tensorfem import structural_verification_face as face

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks" / "structural_verification_face_evidence.json"

YOUNG = 210.0e9
POISSON = 0.3


# --------------------------------------------------------------------------- #
# closed forms
# --------------------------------------------------------------------------- #
def test_euler_bernoulli_matches_hand_computation():
    inertia = 0.05**3 / 12.0
    value = face.euler_bernoulli_tip_deflection(1000.0, 1.0, YOUNG, inertia)
    assert value == pytest.approx(1000.0 / (3.0 * YOUNG * inertia), rel=1e-15)
    assert value == pytest.approx(1.5238095238095238e-4, rel=1e-12)


def test_timoshenko_exceeds_euler_bernoulli_by_the_shear_term():
    inertia = 0.05**3 / 12.0
    area = 0.05
    bernoulli = face.euler_bernoulli_tip_deflection(1000.0, 1.0, YOUNG, inertia)
    shear = face.timoshenko_tip_deflection(1000.0, 1.0, YOUNG, inertia, area, POISSON)
    shear_modulus = YOUNG / (2.0 * (1.0 + POISSON))
    expected = bernoulli + 1000.0 / (5.0 / 6.0 * shear_modulus * area)
    assert shear == pytest.approx(expected, rel=1e-15)
    assert shear > bernoulli
    # the shear gap must follow the classical ~0.78 (h/L)^2 law
    gap = shear / bernoulli - 1.0
    assert gap == pytest.approx(0.78 * (area / 1.0) ** 2, rel=0.02)


def test_beam_bending_stress_puts_the_drooping_cantilever_in_tension_on_top():
    inertia = 0.05**3 / 12.0
    assert face.beam_bending_stress(1000.0, +0.025, inertia) > 0.0
    assert face.beam_bending_stress(1000.0, -0.025, inertia) < 0.0


def test_ritz_reference_reproduces_the_tabulated_clamped_alpha():
    reference = face.ritz_reference(
        side=1.0, young=YOUNG, poisson=POISSON, thickness=0.01, pressure=1.0e3
    )
    # Timoshenko & Woinowsky-Krieger table: 0.00126 for the clamped square
    assert float(reference["alpha"]) == pytest.approx(0.00126, rel=0.01)
    # and the published independent value of this tree (docs/plate-uniform-load-verification.md)
    assert float(reference["alpha"]) == pytest.approx(0.00126551, rel=2e-3)
    # a reference that moves with the basis size would not be an oracle
    assert float(reference["mode_spread"]) < 2e-3


def test_flexural_rigidity_is_the_kirchhoff_definition():
    rigidity = face.flexural_rigidity(YOUNG, 0.01, POISSON)
    assert rigidity == pytest.approx(YOUNG * 0.01**3 / (12.0 * (1.0 - POISSON**2)), rel=1e-15)
    with pytest.raises(ValueError):
        face.flexural_rigidity(YOUNG, -0.01, POISSON)


# --------------------------------------------------------------------------- #
# load integration
# --------------------------------------------------------------------------- #
def test_edge_traction_on_a_straight_edge_conserves_the_resultant():
    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], dtype=face.DTYPE)
    forces = face.q8_edge_traction(points, lambda position: torch.tensor([0.0, -3.0]))
    total = forces.sum(dim=0)
    assert float(total[1]) == pytest.approx(-6.0, rel=1e-13)
    assert float(total[0]) == 0.0
    # the 1/6 - 2/3 - 1/6 optimal-lumping fingerprint of a quadratic edge
    assert float(forces[0, 1]) == pytest.approx(-1.0, rel=1e-13)
    assert float(forces[1, 1]) == pytest.approx(-4.0, rel=1e-13)


def test_edge_traction_on_an_arc_gives_the_exact_arc_resultant():
    radius = 0.75
    pressure = 2.0e7
    span = math.pi / 5.0
    angles = (-0.5 * span, 0.0, 0.5 * span)
    points = torch.tensor(
        [[radius * math.cos(a), radius * math.sin(a)] for a in angles], dtype=face.DTYPE
    )
    forces = face.q8_edge_traction(
        points, lambda position: pressure * position / float(torch.linalg.norm(position))
    )
    summed = forces.sum(dim=0)
    expected = 2.0 * pressure * radius * math.sin(0.5 * span)
    # The arc *measure* is exact (|dx/dxi| of the quadratic mapping) but the
    # traction direction is sampled on the quadratic interpolation of the circle,
    # so the resultant carries an O(span^2) direction error: 1.6e-4 here at
    # span = 36 deg, which is the same order as the per-edge audit of the coarse
    # cylindrical mesh (7.7e-5 at 30 deg elements).
    assert float(torch.linalg.norm(summed)) == pytest.approx(expected, rel=1e-3)
    assert float(summed[0]) == pytest.approx(expected, rel=1e-3)
    # the resultant still points along the arc bisector, i.e. along +x here
    assert abs(float(summed[1])) < 1e-6 * expected


# --------------------------------------------------------------------------- #
# cantilever arm
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def coarse_cantilever():
    return face.solve_cantilever(
        10, 2, length=1.0, height=0.05, young=YOUNG, poisson=POISSON, force=1000.0
    )


def test_cantilever_coarse_level_already_meets_the_one_percent_gate(coarse_cantilever):
    level = coarse_cantilever
    assert level["error_vs_euler_bernoulli"] < face.GATE_CANTILEVER
    assert level["bending_stress_error"] < 0.03
    # the applied shear must reproduce the requested tip force exactly
    assert level["load_resultant"] == pytest.approx(1000.0, rel=1e-12)
    # and the solution must be converged to well below the gate
    assert level["relative_residual"] < 1e-8


def test_cantilever_rejects_an_odd_through_depth_mesh():
    with pytest.raises(ValueError):
        face.solve_cantilever(
            10, 3, length=1.0, height=0.05, young=YOUNG, poisson=POISSON, force=1000.0
        )


# --------------------------------------------------------------------------- #
# thick cylinder arm
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def coarse_cylinder():
    return face.solve_lame_annulus(
        12, 4, inner=0.5, outer=0.8, pressure=100.0e6, young=YOUNG, poisson=POISSON
    )


def test_cylinder_coarse_level_meets_the_two_percent_hoop_gate(coarse_cylinder):
    assert coarse_cylinder["hoop_error"] < face.GATE_CYLINDER
    assert coarse_cylinder["radial_error"] < face.GATE_CYLINDER
    assert coarse_cylinder["bore_displacement_error"] < face.GATE_CYLINDER


def test_cylinder_load_is_the_exact_arc_traction(coarse_cylinder):
    # net resultant of a closed ring
    assert coarse_cylinder["load_resultant_ratio"] < 1e-12
    # every loaded edge reproduces 2 p r sin(dtheta/2) along its bisector
    assert coarse_cylinder["load_edge_resultant_error"] < 1e-4
    assert coarse_cylinder["load_edge_direction_error"] < 1e-6
    # and no tangential load leaked in
    assert coarse_cylinder["load_radial_leak"] < 1e-12


def test_cylinder_response_is_axisymmetric(coarse_cylinder):
    # the exact field has u_theta = 0 everywhere, so u_r cannot vary around the bore
    assert coarse_cylinder["bore_circumferential_spread"] < 1e-3


def test_cylinder_error_falls_under_refinement():
    coarse = face.solve_lame_annulus(
        12, 4, inner=0.5, outer=0.8, pressure=100.0e6, young=YOUNG, poisson=POISSON
    )
    fine = face.solve_lame_annulus(
        24, 8, inner=0.5, outer=0.8, pressure=100.0e6, young=YOUNG, poisson=POISSON
    )
    assert fine["hoop_error"] < 0.5 * coarse["hoop_error"]


# --------------------------------------------------------------------------- #
# plate arm
# --------------------------------------------------------------------------- #
def test_clamped_plate_coarse_mesh_meets_the_three_percent_gate():
    reference = face.ritz_reference(
        side=1.0, young=YOUNG, poisson=POISSON, thickness=0.01, pressure=1.0e3
    )
    level = face.solve_clamped_plate(
        8,
        side=1.0,
        young=YOUNG,
        poisson=POISSON,
        thickness=0.01,
        pressure=1.0e3,
        reference=reference,
    )
    assert level["error_vs_ritz"] < face.GATE_PLATE
    # the SRI Mindlin plate is slightly flexible on a coarse mesh, never stiff
    assert level["centre_deflection"] < level["ritz_deflection"]


def test_clamped_plate_converges_between_two_meshes():
    reference = face.ritz_reference(
        side=1.0, young=YOUNG, poisson=POISSON, thickness=0.01, pressure=1.0e3
    )
    coarse = face.solve_clamped_plate(
        8, side=1.0, young=YOUNG, poisson=POISSON, thickness=0.01,
        pressure=1.0e3, reference=reference,
    )
    fine = face.solve_clamped_plate(
        16, side=1.0, young=YOUNG, poisson=POISSON, thickness=0.01,
        pressure=1.0e3, reference=reference,
    )
    assert abs(fine["error_vs_ritz"]) < 0.25 * abs(coarse["error_vs_ritz"])


# --------------------------------------------------------------------------- #
# ladder bookkeeping
# --------------------------------------------------------------------------- #
def test_observed_order_requires_a_constant_refinement_ratio():
    with pytest.raises(ValueError):
        face.observed_order([1.0, 1.1, 1.11], [0.4, 0.2, 0.15])


def test_observed_order_rejects_a_non_monotone_sequence():
    with pytest.raises(ValueError):
        face.observed_order([1.0, 1.2, 1.1], [0.4, 0.2, 0.1])


def test_observed_order_recovers_a_planted_second_order_rate():
    # value = 2 - h^2 * 3 has an exact second-order rate
    sizes = [0.4, 0.2, 0.1]
    values = [2.0 - 3.0 * size**2 for size in sizes]
    assert face.observed_order(values, sizes) == pytest.approx(2.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# the published evidence is a contract, not a decoration
# --------------------------------------------------------------------------- #
def test_published_evidence_satisfies_contract():
    if not EVIDENCE.exists():  # pragma: no cover - artifact missing from a stale checkout
        pytest.skip("evidence file not present")
    payload = json.loads(EVIDENCE.read_text())
    assert payload["schema"] == "tensorfem-structural-verification-face/1"
    assert payload["all_passed"] is True
    # the gates in the artifact must be the gates in the code
    assert payload["gates"]["cantilever_error_vs_euler_bernoulli"] == pytest.approx(
        face.GATE_CANTILEVER
    )
    assert payload["gates"]["cylinder_hoop_error"] == pytest.approx(face.GATE_CYLINDER)
    assert payload["gates"]["plate_error_vs_ritz"] == pytest.approx(face.GATE_PLATE)
    assert payload["gates"]["min_observed_order"] == pytest.approx(face.MIN_ORDER)

    assert len(payload["cases"]) == 3
    for case in payload["cases"]:
        assert case["passed"] is True, case["case"]
        assert len(case["levels"]) == 3
        assert case["metrics"]["observed_order"] >= face.MIN_ORDER
        assert case["metrics"]["gate_value"] < case["gate"]
        # the ladder must be a genuine h-refinement with a constant ratio
        sizes = [level["h"] for level in case["levels"]]
        assert sizes == sorted(sizes, reverse=True)
        ratios = [sizes[i] / sizes[i + 1] for i in range(len(sizes) - 1)]
        assert ratios[0] == pytest.approx(ratios[1], rel=1e-9)
        # plotting payload must not have leaked into the committed evidence
        assert all("field" not in level for level in case["levels"])


def test_evidence_documents_the_kirchhoff_validity_boundary():
    if not EVIDENCE.exists():  # pragma: no cover
        pytest.skip("evidence file not present")
    payload = json.loads(EVIDENCE.read_text())
    plate = payload["cases"][2]
    ungated = [row for row in plate["metrics"]["aspect_sweep"] if not row["gated"]]
    gated = [row for row in plate["metrics"]["aspect_sweep"] if row["gated"]]
    # the sweep must show the yardstick breaking down on thick plates and holding
    # on thin ones -- that is the whole point of running it
    assert ungated and gated
    assert max(row["error_vs_ritz"] for row in ungated) > face.GATE_PLATE
    assert max(row["error_vs_ritz"] for row in gated) < face.GATE_PLATE