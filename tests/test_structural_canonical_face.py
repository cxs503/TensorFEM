"""Verification of :mod:`tensorfem.structural_canonical_face`.

Every threshold is an independent closed form (Euler--Bernoulli, the Navier
rectangular series, Timoshenko table 8, the Kirsch field) or an exact algebraic
identity of the element formulation.  No reference number comes from this
module's own output, except in the artifact-consistency test, which compares the
committed JSON evidence against the module and is labelled as such.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

import tensorfem.structural_canonical_face as face
from tensorfem.modal import cantilever_beam_modes
from tensorfem.structural_verification import kirchhoff_ss_plate_uniform_load

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks" / "structural_canonical_face.json"


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #
def test_cantilever_closed_form_satisfies_its_own_boundary_conditions():
    EI, P, L = 2000.0, 10.0, 1.0
    assert face.cantilever_deflection_field(0.0, length=L, bending_stiffness=EI, load=P) == 0.0
    assert face.cantilever_slope_field(0.0, length=L, bending_stiffness=EI, load=P) == 0.0
    tip = face.cantilever_deflection_field(L, length=L, bending_stiffness=EI, load=P)
    assert tip == pytest.approx(P * L**3 / (3.0 * EI), rel=1e-15)
    slope = face.cantilever_slope_field(L, length=L, bending_stiffness=EI, load=P)
    assert slope == pytest.approx(P * L**2 / (2.0 * EI), rel=1e-15)
    # curvature identity w''(x) = P (L - x)/EI by second-order finite differences
    h = 1e-4
    xs = np.array([0.3, 0.5, 0.7])
    second = (
        face.cantilever_deflection_field(xs + h, length=L, bending_stiffness=EI, load=P)
        - 2.0 * face.cantilever_deflection_field(xs, length=L, bending_stiffness=EI, load=P)
        + face.cantilever_deflection_field(xs - h, length=L, bending_stiffness=EI, load=P)
    ) / h**2
    assert np.allclose(second, P * (L - xs) / EI, rtol=1e-6)


def test_cantilever_reference_guards():
    with pytest.raises(ValueError):
        face.cantilever_tip_deflection(-1.0, 1.0, 1.0)
    with pytest.raises(TypeError):
        face.cantilever_tip_rotation(1.0, "2000", 1.0)
    with pytest.raises(ValueError):
        face.cantilever_deflection_field(np.array([0.5]), length=1.0,
                                         bending_stiffness=0.0, load=1.0)


def test_navier_series_reproduces_the_timoshenko_table():
    """Table 8 quotes three digits; the exact series must agree to that rounding."""
    for ratio, tabulated in face.TIMOSHENKO_ALPHA_TABLE.items():
        if not math.isfinite(ratio):
            continue
        series = face.rectangular_plate_alpha(ratio)
        assert abs(series - tabulated) / series < 6e-3, (ratio, series, tabulated)


def test_navier_series_tends_to_the_cylindrical_bending_limit():
    strip = face.rectangular_plate_alpha(60.0)
    assert strip == pytest.approx(face.STRIP_ALPHA, rel=1e-4)
    # monotone in the aspect ratio, bounded by the strip value
    values = [face.rectangular_plate_alpha(r) for r in (1.0, 1.5, 2.0, 3.0, 5.0)]
    assert values == sorted(values)
    assert values[-1] < face.STRIP_ALPHA


def test_navier_series_guards():
    with pytest.raises(ValueError):
        face.rectangular_plate_alpha(0.5)
    with pytest.raises(ValueError):
        face.rectangular_plate_navier(np.array([[0.1, 0.1]]), terms=80)
    with pytest.raises(ValueError):
        face.rectangular_plate_navier(np.array([[0.1, 0.1]]), poisson=0.5)


# --------------------------------------------------------------------------- #
# C1: tip-loaded cantilever
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("elements", [1, 2, 4, 8])
def test_tip_loaded_beam_is_nodally_exact(elements):
    """A nodal load makes the Hermite beam solution exact, not merely convergent."""
    solution = face.solve_cantilever_point_load(elements=elements)
    assert solution.tip_deflection_error < 1e-12
    assert solution.tip_rotation_error < 1e-12
    assert np.max(np.abs(solution.transverse - solution.exact_transverse())) < 1e-12


def test_tip_loaded_beam_reactions_balance_the_load():
    solution = face.solve_cantilever_point_load(elements=4, load=12.0, length=2.0)
    assert solution.base_reaction == pytest.approx(12.0, rel=1e-12)
    assert solution.base_moment == pytest.approx(12.0 * 2.0, rel=1e-12)


def test_tip_loaded_beam_energy_matches_clapeyron():
    solution = face.solve_cantilever_point_load(elements=4)
    assert solution.energy_relative_error < 1e-12
    assert solution.exact_strain_energy == pytest.approx(
        solution.load**2 * solution.length**3 / (6.0 * solution.bending_stiffness), rel=1e-14
    )


# --------------------------------------------------------------------------- #
# C2: rectangular simply supported Kirchhoff plate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 3.0])
def test_rectangular_plate_centre_deflection_within_two_percent(ratio):
    solution = face.solve_rectangular_plate_kirchhoff(
        length=ratio, width=1.0, elements_x=int(12 * ratio), elements_y=12
    )
    assert solution.centre_error() < face.GATE_PLATE_CENTRE
    # conforming element: the FE plate is stiffer than the exact one, so the
    # centre deflection converges from below
    assert solution.centre_deflection < float(
        solution.navier(np.array([[0.5 * ratio, 0.5]]))[0]
    )


def test_rectangular_plate_is_symmetric_and_restrained_on_the_edges():
    solution = face.solve_rectangular_plate_kirchhoff(
        length=2.0, width=1.0, elements_x=8, elements_y=4
    )
    w = solution.deflection
    assert np.allclose(w, w[::-1, :], atol=1e-18)
    assert np.allclose(w, w[:, ::-1], atol=1e-18)
    assert np.allclose(w[0, :], 0.0, atol=1e-18)
    assert np.allclose(w[-1, :], 0.0, atol=1e-18)
    assert np.allclose(w[:, 0], 0.0, atol=1e-18)
    assert np.allclose(w[:, -1], 0.0, atol=1e-18)
    X, Y, W = solution.grid()
    assert W.shape == (5, 9)
    assert X.max() == pytest.approx(2.0) and Y.max() == pytest.approx(1.0)


def test_rectangular_plate_alpha_approaches_the_strip_limit_in_the_long_plate():
    solution = face.solve_rectangular_plate_kirchhoff(
        length=4.0, width=1.0, elements_x=32, elements_y=8
    )
    assert solution.alpha < face.STRIP_ALPHA
    assert solution.alpha > 0.9 * face.STRIP_ALPHA


def test_rectangular_plate_guards():
    with pytest.raises(ValueError):
        face.solve_rectangular_plate_kirchhoff(elements_x=7, elements_y=8)
    with pytest.raises(ValueError):
        face.solve_rectangular_plate_kirchhoff(length=-1.0)
    with pytest.raises(ValueError):
        face.solve_rectangular_plate_kirchhoff(poisson=0.5)


# --------------------------------------------------------------------------- #
# C3 / C4: modal arms
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("elements", [4, 8])
def test_cantilever_first_frequency_within_three_percent(elements):
    result = cantilever_beam_modes(1.0, elements, 2000.0, 8.0, 1)
    computed = float(result.frequencies_hz[0])
    exact = face.cantilever_first_frequency(length=1.0, bending_stiffness=2000.0,
                                            mass_per_length=8.0)
    assert abs(computed - exact) / exact < face.GATE_BEAM_FREQUENCY
    # consistent-mass beam elements bracket the exact frequency from above
    assert computed >= exact


def test_rectangular_plate_first_frequency_within_three_percent():
    result = face.solve_plate_first_frequency()
    assert result["relative_error"] < face.GATE_PLATE_FREQUENCY
    expected = math.pi**2 * math.sqrt(
        (200.0e9 * 0.01**3 / (12.0 * (1 - 0.3**2))) / (7850.0 * 0.01)
    ) * ((1.0 / 2.0) ** 2 + (1.0 / 1.0) ** 2) / (2.0 * math.pi)
    assert result["exact_hz"] == pytest.approx(expected, rel=1e-12)


# --------------------------------------------------------------------------- #
# C5: Q8 Kirsch stress concentration
# --------------------------------------------------------------------------- #
def test_kirsch_q8_factor_within_the_absolute_gate():
    result = face.solve_kirsch_q8_concentration(n_theta=16, n_radial=4)
    assert result.absolute_error < face.GATE_STRESS_CONCENTRATION
    assert 2.9 < result.factor < 3.1
    # the prescribed boundary data are exact, so only discretisation error remains
    assert result.displacement_relative_l2 < 5e-3
    assert result.cg_relative_residual < 1e-10


def test_kirsch_q8_refinement_reduces_the_error():
    coarse = face.solve_kirsch_q8_concentration(n_theta=16, n_radial=2)
    fine = face.solve_kirsch_q8_concentration(n_theta=16, n_radial=4)
    assert abs(fine.factor - 3.0) < abs(coarse.factor - 3.0)
    assert fine.displacement_relative_l2 < coarse.displacement_relative_l2


def test_kirsch_q8_hole_edge_hoop_stress_limits():
    field = face.kirsch_q8_hole_edge_hoop_stress(n_theta=16, n_radial=4)
    theta = field["theta"]
    # theta = pi/2 -> 3 (the classical concentration), theta = 0 -> -1
    at_top = int(np.argmin(np.abs(theta - 0.5 * math.pi)))
    at_side = int(np.argmin(np.abs(theta)))
    assert field["computed"][at_top] == pytest.approx(3.0, abs=0.1)
    assert field["computed"][at_side] == pytest.approx(-1.0, abs=0.1)
    assert np.max(np.abs(field["computed"] - field["exact"])) < 0.1


def test_kirsch_q8_guards():
    with pytest.raises(ValueError):
        face.solve_kirsch_q8_concentration(n_theta=15)
    with pytest.raises(ValueError):
        face.solve_kirsch_q8_concentration(inner_radius=5.0, outer_radius=4.0)


# --------------------------------------------------------------------------- #
# C6: linear-element convergence order
# --------------------------------------------------------------------------- #
def test_bilinear_plate_l2_order_exceeds_the_campaign_gate():
    rows = face.linear_element_order_study((4, 8, 12))
    order = face.convergence_order(rows)
    assert order >= face.GATE_LINEAR_ORDER, (order, [r.relative_l2_error for r in rows])
    for row in rows[1:]:
        assert row.relative_l2_error < rows[rows.index(row) - 1].relative_l2_error
    for pair in face.pairwise_orders(rows):
        assert pair["observed_order"] > 1.85


def test_bilinear_plate_centre_deflection_agrees_with_navier():
    """The same linear element must agree pointwise, not only in the L2 norm."""
    solution = face.solve_rectangular_plate_mindlin(
        length=2.0, width=1.0, elements_x=24, elements_y=12
    )
    nodes = solution.nodes
    centre = int(np.argmin(np.hypot(nodes[:, 0] - 1.0, nodes[:, 1] - 0.5)))
    exact = float(
        face.rectangular_plate_navier(
            np.array([[1.0, 0.5]]), length=2.0, width=1.0,
            young=2.1e11, poisson=0.3, thickness=0.01, pressure=1.0,
        )[0]
    )
    assert abs(solution.deflection[centre] - exact) / exact < 0.05


# --------------------------------------------------------------------------- #
# the assembled campaign face
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def campaign_face():
    return face.run_canonical_face(
        plate_elements=(12, 12),
        convergence_levels=(4, 8, 12),
        kirsch_n_theta=16,
        kirsch_n_radial=4,
    )


def test_campaign_face_reports_every_gate(campaign_face):
    assert set(campaign_face["gates"]) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert campaign_face["all_pass"], campaign_face["gates"]
    assert campaign_face["arms"]["C2"]["pass"]
    assert campaign_face["arms"]["C2"]["strip_relative_error"] < 1e-3


# --------------------------------------------------------------------------- #
# committed evidence and the square-plate helper the face builds on
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not ARTIFACT.exists(), reason="audit artifact not generated")
def test_committed_artifact_is_consistent_with_the_module():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert set(payload["gates"]) == {
        "C1_beam_tip_deflection",
        "C2_plate_centre_deflection",
        "C3_beam_first_frequency",
        "C4_plate_first_frequency",
        "C5_stress_concentration",
        "C6_linear_element_order",
    }
    assert payload["all_pass"] is True
    # aspect-ratio table recorded in the artifact must still match the module
    for row in payload["arms"]["C2"]["rows"]:
        series = face.rectangular_plate_alpha(row["aspect_ratio"])
        assert row["alpha_navier"] == pytest.approx(series, rel=1e-12)
        assert row["relative_error"] < face.GATE_PLATE_CENTRE
    assert payload["gate_linear_order"] == face.GATE_LINEAR_ORDER


def test_square_plate_helper_grid_spans_the_modelled_plate():
    """Regression: ``n_y`` refines the mesh, it does not stretch the plate."""
    solution = kirchhoff_ss_plate_uniform_load(n_x=8, n_y=16)
    X, Y, W = solution.grid()
    assert W.shape == (17, 9)
    assert X.max() == pytest.approx(solution.side)
    assert Y.max() == pytest.approx(solution.side)
    assert W[8, 4] == pytest.approx(solution.centre_deflection)
