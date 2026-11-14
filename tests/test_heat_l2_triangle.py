"""pytest gates for the P1-triangle conduction study (CASE-FEM-HEAT-L2, round 2).

The tests are deliberately cheap (levels up to 32) and assert *invariants*, not
just that code ran: exactness of the representable field, symmetry and
row-sum properties of the assembled conductivity matrix, agreement with an
independent scikit-fem solve, the measured L2 rate, and the flux balance of the
mixed Dirichlet/Neumann case.
"""
import math

import pytest
import torch

from tensorfem import heat_l2_triangle as hlt

LEVELS = (4, 8, 16, 32)


@pytest.fixture(scope="module")
def cases():
    return hlt.triangle_field_cases()


def test_target_order_is_read_from_the_manual():
    target = hlt.target_l2_order_from_manual()
    assert target["order"] == 2.0
    assert "O(h^2)" in target["verbatim"]
    assert target["line"] > 0


def test_quadrature_rule_integrates_degree5_exactly():
    bary, weights = hlt.triangle_gauss_rule()
    assert abs(float(weights.sum()) - 1.0) < 4e-15
    # weights are normalised to the triangle area, so the rule returns
    # 2 * int_{unit right triangle} x^p y^q = 2 * p! q! / (p+q+2)!
    for p, q in ((0, 0), (2, 0), (2, 2), (4, 1), (5, 0)):
        approx = float((weights * bary[:, 1] ** p * bary[:, 2] ** q).sum())
        exact = 2.0 * math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
        assert abs(approx - exact) / exact < 1e-13


def test_meshes_are_ccw_and_shape_regular():
    for kind in ("structured-a", "structured-b", "distorted"):
        mesh = hlt.build_mesh(kind, 8, 8)
        assert bool((hlt.signed_areas(mesh.nodes, mesh.elements) > 0).all())
        assert mesh.n_elements == 2 * 64
        assert mesh.h == pytest.approx(math.sqrt(1.0 / 128.0))
        quality = hlt.min_quality(mesh.nodes, mesh.elements)
        assert 0.1 < quality <= 1.0


def test_the_two_diagonal_splits_are_different_meshes(cases):
    """The asymmetry detector is only meaningful if the meshes differ."""
    a = hlt.solve_case(cases["sine_generation"], 8, kind="structured-a")
    b = hlt.solve_case(cases["sine_generation"], 8, kind="structured-b")
    delta = float((a["temperature"] - b["temperature"]).abs().max())
    assert delta > 1e-8
    l2_gap = abs(a["errors"]["l2_relative"] - b["errors"]["l2_relative"])
    assert l2_gap / a["errors"]["l2_relative"] < hlt.GATE_DIAGONAL


def test_conductivity_matrix_is_symmetric_with_zero_row_sums():
    mesh = hlt.build_mesh("structured-a", 4, 4)
    k = hlt.assemble_conductivity(mesh.nodes, mesh.elements)
    assert float((k - k.T).abs().max()) < 1e-14
    assert float(k.sum(dim=1).abs().max()) < 1e-12


def test_representable_field_is_exact(cases):
    """Control: a linear field is in the P1 space, so the solve adds nothing."""
    for kind in ("structured-a", "structured-b", "distorted"):
        ladder = hlt.study_case(cases["linear_control"], (4, 8, 16), kind=kind)
        assert max(ladder["max_abs_nodal"]) < hlt.GATE_CONTROL_ABS


def test_l2_rate_of_the_p1_kernel(cases):
    ladder = hlt.study_case(cases["sine_generation"], LEVELS, kind="structured-a")
    pairs = hlt.pair_orders(ladder["l2_orders"], ladder["levels"], hlt.PAIR_MIN_COARSE)
    assert min(pairs) >= hlt.GATE_L2_PAIR
    assert hlt.GATE_L2_FINEST[0] <= ladder["l2_orders"][-1] <= hlt.GATE_L2_FINEST[1]
    assert hlt.richardson_residual(ladder["l2_relative"], ladder["h"], 2.0) <= hlt.GATE_POWER_LAW


def test_distorted_family_keeps_the_rate(cases):
    ladder = hlt.study_case(cases["cubic_harmonic"], (8, 16, 32), kind="distorted")
    assert min(ladder["l2_orders"]) >= 1.85
    assert min(ladder["min_quality"]) > 0.1


def test_mixed_bc_flux_balance(cases):
    for level in (8, 16, 32):
        run = hlt.solve_case(cases["cosh_cos_neumann"], level)
        assert run["flux_balance"] <= hlt.GATE_FLUX_BALANCE
        assert run["free_dof_residual"] <= hlt.GATE_FLUX_BALANCE
        # a mixed-BC solve must carry a non-zero reaction and a non-zero load
        assert run["reaction_sum"] != 0.0
        assert run["flux_imposed"] != 0.0


def test_neumann_side_is_excluded_from_dirichlet(cases):
    run = hlt.solve_case(cases["cosh_cos_neumann"], 8)
    mesh = hlt.build_mesh("structured-a", 8, 8)
    neumann = torch.unique(hlt.boundary_edges(mesh, "x_max").reshape(-1))
    assert run["n_dof_dirichlet"] == int(mesh.boundary_nodes.numel()) - int(neumann.numel())


def test_subdivision_preserves_the_p1_function(cases):
    run = hlt.solve_case(cases["sine_generation"], 8)
    nodes, elements, values = run["nodes"], run["elements"], run["temperature"]
    coarse = hlt.evaluate_errors(nodes, elements, values, cases["sine_generation"])["l2_absolute"]
    nodes_f, elements_f, edge_map = hlt.subdivide(nodes, elements)
    fine = hlt.evaluate_errors(nodes_f, elements_f,
                               hlt.lift_to_subdivided(values, edge_map),
                               cases["sine_generation"])["l2_absolute"]
    assert abs(fine - coarse) / fine < 1e-4


def test_skfem_cross_validation(cases):
    if not hlt.skfem_available():
        pytest.skip("scikit-fem not importable")
    for case_key in ("sine_generation", "cosh_cos_neumann"):
        xval = hlt.skfem_crosscheck(cases[case_key], 8)
        assert xval["nodal_rel_diff"] <= hlt.GATE_XVAL
        assert xval["matrix_rel_diff"] <= hlt.GATE_XVAL


def test_run_all_gates_pass_on_a_short_ladder(monkeypatch):
    """The full evidence bundle, on the cheap ladder, must pass every gate."""
    evidence = hlt.run_all(levels=(4, 8, 16, 32, 64),
                           sensitivity_levels=(8, 16, 32), xval_level=16,
                           quadrature_level=32)
    assert evidence["not_run_gates"] == []
    assert evidence["passed"], {k: (v["value"], v["threshold"])
                                for k, v in evidence["gates"].items() if not v["passed"]}
    for gate in evidence["gates"].values():
        assert gate["passed"], gate