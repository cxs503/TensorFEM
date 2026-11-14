"""Gates for CASE-FEM-HEAT-L2: steady conduction L2 order >= 2 (MMS).

The thresholds asserted here are the ones predeclared in
``tensorfem.heat_l2_convergence`` (module docstring); the test bodies only
re-run a shortened ladder (nx = 8, 16, 32) so the suite stays inside the
pytest timeout.  The full ladder (to nx = 64), the figures and the six-section
report are produced by ``benchmarks/heat_l2_convergence.py``.
"""
import math

import pytest
import torch

from tensorfem.heat_l2_convergence import (
    GATE_CONTROL_ABS,
    GATE_L2_FINEST,
    GATE_L2_PAIR,
    GATE_QUADRATURE,
    field_cases,
    gauss_legendre_1d,
    mesh_dims,
    q4_shape,
    richardson_residual,
    run_all,
    solve_case,
    target_l2_order_from_manual,
)

D = torch.float64
SHORT_LEVELS = (8, 16, 32)


def test_target_order_is_parsed_from_the_repo_manual():
    """The target p = 2 must come from the in-repo a-priori table, not a literal."""
    target = target_l2_order_from_manual()
    assert target["order"] == 2.0
    assert target["file"] == "adaptive_poisson_2d.py"
    assert target["line"] > 0
    assert "O(h^2)" in target["row"]
    assert "L2 norm" in target["row"]


def test_gauss_legendre_integrates_polynomials_exactly():
    """A 3-point rule must be exact up to degree 5 (2n-1)."""
    x, w = gauss_legendre_1d(3)
    for degree in range(6):
        exact = 2.0 / (degree + 1) if degree % 2 == 0 else 0.0
        got = float((w * x ** degree).sum())
        assert abs(got - exact) < 1.0e-14, (degree, got, exact)


def test_q4_shape_functions_are_a_partition_of_unity():
    for xi, eta in ((0.0, 0.0), (-0.7, 0.3), (1.0, -1.0)):
        shape = q4_shape(xi, eta)
        assert abs(float(shape.sum()) - 1.0) < 1.0e-15


def test_element_mean_source_matches_closed_form():
    """The 8x8 element integral must reproduce the analytic element mean of q."""
    case = field_cases()["sine_generation"]
    x0, x1, y0, y1 = 0.25, 0.5, 0.0, 0.25
    nodes = torch.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=D)
    elements = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    hx, hy = x1 - x0, y1 - y0
    closed = (2.0 * (math.cos(math.pi * x0) - math.cos(math.pi * x1))
              * (math.cos(math.pi * y0) - math.cos(math.pi * y1)) / (hx * hy))
    got = float(case.element_mean_source(nodes, elements)[0])
    assert abs(got - closed) < 1.0e-12 * abs(closed)


def test_mesh_dims_are_two_to_one_for_the_aspect_case():
    assert mesh_dims(field_cases()["sine_aspect2"], 32) == (32, 16)
    assert mesh_dims(field_cases()["sine_generation"], 32) == (32, 32)


def test_l2_order_reaches_two_on_the_source_driven_mms():
    """Primary criterion: the L2 ladder of the heat kernel is >= 2nd order."""
    case = field_cases()["sine_generation"]
    study = solve_case  # keep the import explicit for readability
    runs = [study(case, level) for level in SHORT_LEVELS]
    hs = [run["h"] for run in runs]
    errors = [run["errors"]["l2_relative"] for run in runs]
    orders = [math.log(a / b) / math.log(c / d)
              for a, b, c, d in zip(errors[:-1], errors[1:], hs[:-1], hs[1:])]
    assert min(orders) >= GATE_L2_PAIR, orders
    assert GATE_L2_FINEST[0] <= orders[-1] <= GATE_L2_FINEST[1], orders[-1]


def test_l2_order_reaches_two_on_the_two_to_one_domain():
    case = field_cases()["sine_aspect2"]
    runs = [solve_case(case, level) for level in SHORT_LEVELS]
    hs = [run["h"] for run in runs]
    errors = [run["errors"]["l2_relative"] for run in runs]
    orders = [math.log(a / b) / math.log(c / d)
              for a, b, c, d in zip(errors[:-1], errors[1:], hs[:-1], hs[1:])]
    assert min(orders) >= GATE_L2_PAIR, orders


def test_harmonic_cubic_is_nodally_exact_and_adds_no_interpolation_error():
    """Uniform Q4 of a rectangle is the 5-point Laplacian: exact for cubics."""
    case = field_cases()["cubic_harmonic"]
    run = solve_case(case, 16)
    assert run["errors"]["max_abs_nodal"] < GATE_CONTROL_ABS
    # the L2 error is then pure interpolation error, so the ratio must be one
    assert abs(run["errors"]["l2_relative"]
               / run["interpolation_errors"]["l2_relative"] - 1.0) < 1.0e-9


def test_bilinear_control_is_solved_exactly():
    """The representable control has no signal: exact to machine precision."""
    case = field_cases()["bilinear_control"]
    for level in (8, 16):
        run = solve_case(case, level)
        assert run["errors"]["l2_absolute"] < GATE_CONTROL_ABS
        assert run["errors"]["max_abs_nodal"] < GATE_CONTROL_ABS


def test_quadratic_harmonic_is_not_representable():
    """Regression for the specification correction recorded in the module.

    ``span{1, x, y, x*y}`` does not contain ``x^2 - y^2``: the case is nodally
    exact yet has a genuine O(h^2) L2 error, so it must *not* be used as the
    "exact control" (that was the first, failed version of this study).
    """
    case = field_cases()["quadratic_harmonic"]
    run = solve_case(case, 16)
    assert run["errors"]["max_abs_nodal"] < GATE_CONTROL_ABS
    assert run["errors"]["l2_absolute"] > 1.0e-6


def test_error_norms_are_independent_of_the_post_processing_rule():
    case = field_cases()["sine_generation"]
    base = solve_case(case, 16, quad_order=4)["errors"]
    for order in (6, 8):
        other = solve_case(case, 16, quad_order=order)["errors"]
        for key, value in other.items():
            scale = abs(base[key]) if base[key] != 0.0 else 1.0
            assert abs(value - base[key]) / scale <= GATE_QUADRATURE, (order, key)


def test_richardson_residual_separates_power_laws():
    """A clean O(h^2) ladder extrapolates to zero; an O(h^3) one does not."""
    hs = [1.0, 0.5, 0.25]
    clean = [0.04, 0.01, 0.0025]
    assert richardson_residual(clean, hs, 2.0) < 1.0e-12
    cubic = [0.008, 0.001, 0.000125]
    assert richardson_residual(cubic, hs, 2.0) > 1.0e-2


@pytest.fixture(scope="module")
def short_ladder_evidence():
    """One shortened-ladder run shared by the end-to-end gate tests."""
    return run_all(levels=SHORT_LEVELS)


@pytest.mark.parametrize("case_key", ["sine_generation", "sine_aspect2",
                                      "cubic_harmonic", "bilinear_control"])
def test_predeclared_gates_all_pass_on_the_short_ladder(
        case_key, short_ladder_evidence):
    """End-to-end: every predeclared gate of the case must pass."""
    evidence = short_ladder_evidence
    assert evidence["verdict"]["checks"], "no gate ran"
    failing = [name for name in evidence["verdict"]["failed"]
               if name.startswith(case_key)]
    assert not failing, evidence["verdict"]


def test_short_ladder_verdict_is_green(short_ladder_evidence):
    """The whole predeclared gate set must pass on the shortened ladder."""
    verdict = short_ladder_evidence["verdict"]
    assert verdict["passed"], verdict["failed"]