import math

import pytest
import torch

from tensorfem.marine_panel_ultimate_fe import (
    build_panel_case, classical_panel_references, panel_contract,
    run_panel_ultimate_qualification,
)


@pytest.mark.parametrize("n", (4, 8, 12))
def test_panel_is_real_multi_element_mesh_with_normalized_loading(n):
    case = build_panel_case(n)
    assert len(case.model.elements) == n*n
    assert len(case.model.nodes) == (n+1)**2
    assert case.model.n_dofs == 6*(n+1)**2
    assert float(-case.reference_load[0::6].sum()) == pytest.approx(1.0)
    assert torch.count_nonzero(case.reference_load[1::6]) == 0
    assert len(torch.unique(case.model.elements)) == (n+1)**2


def test_imperfection_is_stress_free_reference_sine_mode():
    case = build_panel_case(8)
    delta = case.model.nodes-case.perfect_nodes
    assert float(delta[:, :2].abs().max()) == 0.0
    assert float(delta[:, 2].max()) == pytest.approx(case.model.thickness/10)
    boundary = ((case.perfect_nodes[:, 0] == 0) | (case.perfect_nodes[:, 0] == 1)
                | (case.perfect_nodes[:, 1] == 0) | (case.perfect_nodes[:, 1] == 1))
    assert float(delta[boundary, 2].abs().max()) < 1e-15


@pytest.mark.parametrize("n", (4, 8, 12))
def test_residual_stress_is_symmetric_and_self_balanced(n):
    case = build_panel_case(n)
    residual = case.model.residual_stress[..., 0]
    # Equal structured cells/Gauss weights: this is the actual discrete resultant.
    assert abs(float(residual.mean())) < 5e-9*case.model.yield_stress
    assert float(residual.abs().max()) < case.model.yield_stress
    assert torch.equal(case.model.residual_stress[..., 1:],
                       torch.zeros_like(case.model.residual_stress[..., 1:]))


def test_classical_references_are_independent_plate_bounds():
    case = build_panel_case(4, residual_ratio=0)
    refs = classical_panel_references(case)
    d = case.model.young*case.model.thickness**3/(12*(1-case.model.poisson**2))
    assert refs["classical_elastic_buckling_force"] == pytest.approx(4*math.pi**2*d)
    assert refs["gross_section_squash_force"] == pytest.approx(2.5e6)
    assert refs["slenderness"] > 1


def test_contract_has_three_meshes_two_controls_and_fails_closed():
    contract = panel_contract()
    assert [r["elements"] for r in contract["mesh_matrix"]] == [16, 64, 144]
    assert len(contract["arc_length_controls"]) == 2
    assert "energy_residual" in contract["required_outputs"]
    result = run_panel_ultimate_qualification()
    assert result["status"] == "blocked" and result["passed"] is False
    assert "No peak or post-peak claim" in result["boundary"]
    assert "4/8/12 meshes" in result["blocking_capability"]


def test_invalid_panel_inputs_fail_closed():
    with pytest.raises(ValueError, match="even"):
        build_panel_case(3)
    with pytest.raises(ValueError, match="initial-field"):
        build_panel_case(4, residual_ratio=1.0)
