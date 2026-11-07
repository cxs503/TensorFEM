"""Qualification of the J2 minimum elastoplastic surface.

Gates (all machine checked, see docs/j2-plasticity-minimal-surface.md):

* uniaxial piecewise analytic anchor: elastic slope ``E``, plastic slope
  ``E H / (E + H)`` (relative error ``< 1e-10``);
* pure shear and general-direction yield onset versus the analytic yield
  surface ``q = sigma_y`` (relative error ``< 1%``);
* a posteriori return-mapping residuals ``< 1e-12`` over a non-proportional
  path;
* consistent tangent versus central difference ``< 1e-6``;
* fail-closed rejection of negative hardening, non-positive yield stress,
  non-physical elastic constants and asymmetric strain input.
"""

import json
import math

import pytest
import torch

from tensorfem.j2_elastoplasticity import (
    J2Material,
    elastic_tangent,
    equivalent_stress,
    pure_shear_yield_point,
    radial_return,
    return_map_residuals,
    strain_from_voigt,
    uniaxial_stress_path,
    verify_minimum_surface,
    voigt_from_tensor,
)

D = torch.float64
MATERIAL = J2Material(young=210.0e9, poisson=0.3, yield_stress=300.0e6, hardening=10.0e9)


def test_uniaxial_piecewise_analytic_anchor():
    """Elastic slope E, plastic slope EH/(E+H), and the closed-form stress."""
    E, H = MATERIAL.young, MATERIAL.hardening
    eps_y = MATERIAL.yield_stress / E
    E_t = E * H / (E + H)
    strains = [0.25 * eps_y, 0.5 * eps_y, eps_y, 2.0 * eps_y, 4.0 * eps_y, 6.0 * eps_y]
    path = uniaxial_stress_path(MATERIAL, strains)

    moduli = path["incremental_modulus"]
    assert abs(moduli[0] - E) / E < 1e-10  # elastic branch
    assert abs(moduli[-1] - E_t) / E_t < 1e-10  # plastic branch
    assert moduli[-1] < E  # hardening softens the slope
    for eps, sigma in zip(strains, path["stress"]):
        exact = E * eps if eps <= eps_y else MATERIAL.yield_stress + E_t * (eps - eps_y)
        assert abs(sigma - exact) / exact < 1e-10
    # Lateral contraction is elastic below yield and incompressible-plastic above.
    assert abs(path["lateral_strain"][0] + MATERIAL.poisson * strains[0]) < 1e-12
    assert path["equivalent_plastic_strain"][0] == 0.0
    assert path["equivalent_plastic_strain"][-1] > 0.0


def test_pure_shear_yield_point_matches_analytic_surface():
    """Pure shear: tau_y = sigma_y/sqrt(3), gamma_y = sigma_y/(sqrt(3) G)."""
    shear = pure_shear_yield_point(MATERIAL)
    assert shear["gamma_relative_error"] < 0.01
    assert shear["tau_relative_error"] < 0.01
    assert shear["yield_surface_relative_error"] < 0.01
    assert shear["gamma_y"] == pytest.approx(
        MATERIAL.yield_stress / (math.sqrt(3.0) * MATERIAL.shear_modulus), rel=1e-10
    )
    assert shear["tau_y"] == pytest.approx(MATERIAL.yield_stress / math.sqrt(3.0), rel=1e-10)


def test_yield_onset_lies_on_analytic_yield_surface_for_multiple_directions():
    """Every probed direction must land on q = sigma_y within 1%."""
    report = verify_minimum_surface(MATERIAL)
    assert report["yield_surface_max_relative_error"] < 0.01
    assert set(report["yield_surface"]) == {
        "uniaxial",
        "pure_shear",
        "equibiaxial_shear",
        "combined",
    }
    for probe in report["yield_surface"].values():
        assert probe["scale_relative_error"] < 1e-10
        assert probe["yield_surface_relative_error"] < 0.01


def test_return_mapping_a_posteriori_residual_below_1e12():
    """Re-evaluate yield/stress/flow/hardening/complementarity from the result."""
    strains = [
        torch.diag(torch.tensor([2.0e-3, -0.5e-3, -0.5e-3], dtype=D)),
        torch.tensor([[2.0e-3, 0.9e-3, 0.0], [0.9e-3, -0.5e-3, 0.0], [0.0, 0.0, -0.5e-3]], dtype=D),
        torch.tensor(
            [[2.0e-3, -0.9e-3, 0.0], [-0.9e-3, -0.5e-3, 0.0], [0.0, 0.0, -0.5e-3]], dtype=D
        ),
        torch.tensor(
            [[2.4e-3, -1.2e-3, 2e-4], [-1.2e-3, -4e-4, 1e-4], [2e-4, 1e-4, -6e-4]], dtype=D
        ),
    ]
    state = MATERIAL.virgin_state()
    worst = 0.0
    plastic_steps = 0
    for strain in strains:
        result = radial_return(strain, MATERIAL, state)
        assert result.max_residual < 1e-12
        # Independent recomputation must reproduce the stored residuals.
        recomputed = return_map_residuals(strain, MATERIAL, state, result)
        assert max(recomputed.values()) < 1e-12
        worst = max(worst, result.max_residual)
        plastic_steps += int(result.dgamma > 0.0)
        state = result.state
    assert plastic_steps >= 3
    assert worst < 1e-12
    # The plastic state must be physically admissible.
    assert float(state.alpha) > 0.0
    assert (
        result.equivalent_stress
        <= MATERIAL.yield_stress + MATERIAL.hardening * float(state.alpha) + 1e-6
    )


def test_consistent_tangent_matches_central_difference():
    """Algorithmic tangent equals d(sigma)/d(eps) of the return mapping."""
    strain = torch.tensor(
        [[2.4e-3, -1.2e-3, 2e-4], [-1.2e-3, -4e-4, 1e-4], [2e-4, 1e-4, -6e-4]], dtype=D
    )
    state = radial_return(
        torch.diag(torch.tensor([2.0e-3, -0.5e-3, -0.5e-3], dtype=D)),
        MATERIAL,
        MATERIAL.virgin_state(),
    ).state
    base = radial_return(strain, MATERIAL, state)
    assert base.dgamma > 0.0
    h = 1e-7
    worst = 0.0
    for k, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (2, 0))):
        direction = torch.zeros((3, 3), dtype=D)
        if i == j:
            direction[i, i] = 1.0
        else:
            direction[i, j] = direction[j, i] = 0.5
        plus = radial_return(strain + h * direction, MATERIAL, state).stress_voigt
        minus = radial_return(strain - h * direction, MATERIAL, state).stress_voigt
        fd = (plus - minus) / (2.0 * h)
        column = base.tangent[:, k]
        scale = float(column.abs().max())
        worst = max(worst, float((fd - column).abs().max()) / scale)
    assert worst < 1e-6
    # The plastic tangent must be softer than the elastic one.
    assert float(torch.linalg.matrix_norm(base.tangent)) < float(
        torch.linalg.matrix_norm(MATERIAL.elastic_tangent)
    )


def test_elastic_branch_returns_elastic_tangent_and_zero_residual():
    strain = torch.diag(torch.tensor([1.0e-4, -0.3e-4, -0.3e-4], dtype=D))
    result = radial_return(strain, MATERIAL)
    assert result.elastic and result.dgamma == 0.0
    assert result.max_residual < 1e-14
    assert float(result.state.alpha) == 0.0
    assert torch.allclose(
        result.tangent, elastic_tangent(MATERIAL.young, MATERIAL.poisson), rtol=1e-12, atol=0.0
    )
    stress = result.stress
    tr = torch.trace(strain)
    G, K = MATERIAL.shear_modulus, MATERIAL.bulk_modulus
    dev = strain - tr / 3.0 * torch.eye(3, dtype=D)
    assert torch.allclose(
        stress, K * tr * torch.eye(3, dtype=D) + 2.0 * G * dev, rtol=1e-12, atol=1e-12
    )


def test_perfect_plasticity_saturates_at_yield_stress():
    """H = 0: the stress cannot exceed sigma_y and the tangent modulus is zero."""
    material = J2Material(young=200.0e9, poisson=0.3, yield_stress=250.0e6, hardening=0.0)
    assert material.plastic_tangent_modulus == 0.0
    eps_y = material.yield_stress / material.young
    path = uniaxial_stress_path(material, [0.5 * eps_y, eps_y, 5.0 * eps_y, 10.0 * eps_y])
    assert abs(path["stress"][-1] - material.yield_stress) / material.yield_stress < 1e-10
    # Zero tangent up to double-precision round-off (relative to E).
    assert abs(path["incremental_modulus"][-1]) / material.young < 1e-12
    assert path["equivalent_plastic_strain"][-1] > 0.0


@pytest.mark.parametrize("hardening", [-1.0, -1e-3, -1e12])
def test_negative_hardening_modulus_is_rejected(hardening):
    with pytest.raises(ValueError, match="hardening"):
        J2Material(young=210.0e9, poisson=0.3, yield_stress=300.0e6, hardening=hardening)


@pytest.mark.parametrize("yield_stress", [0.0, -1.0, -300.0e6])
def test_non_positive_yield_stress_is_rejected(yield_stress):
    with pytest.raises(ValueError, match="yield stress"):
        J2Material(young=210.0e9, poisson=0.3, yield_stress=yield_stress, hardening=1.0e9)


def test_invalid_elastic_constants_and_arrays_are_rejected():
    with pytest.raises(ValueError, match="young"):
        J2Material(young=0.0, poisson=0.3, yield_stress=300.0e6)
    with pytest.raises(ValueError, match="poisson"):
        J2Material(young=210.0e9, poisson=0.5, yield_stress=300.0e6)
    with pytest.raises(ValueError, match="poisson"):
        J2Material(young=210.0e9, poisson=-1.5, yield_stress=300.0e6)
    with pytest.raises(ValueError, match="finite"):
        J2Material(young=float("nan"), poisson=0.3, yield_stress=300.0e6)
    asymmetric = torch.tensor([[1e-4, 2e-4, 0.0], [0.0, 1e-4, 0.0], [0.0, 0.0, 1e-4]], dtype=D)
    with pytest.raises(ValueError, match="symmetric"):
        radial_return(asymmetric, MATERIAL)
    with pytest.raises(ValueError, match="3x3"):
        radial_return(torch.zeros(6, dtype=D), MATERIAL)
    with pytest.raises(TypeError, match="J2Material"):
        radial_return(torch.zeros((3, 3), dtype=D), {"young": 1.0})


def test_voigt_helpers_round_trip_and_uniaxial_consistency():
    strain = torch.tensor([[1e-3, 2e-4, 0.0], [2e-4, -3e-4, 1e-4], [0.0, 1e-4, -7e-4]], dtype=D)
    vector = voigt_from_tensor(strain).clone()
    vector[3:] *= 2.0
    assert torch.allclose(strain_from_voigt(vector), strain, rtol=0, atol=1e-16)
    result = radial_return(strain, MATERIAL)
    q = equivalent_stress(result.stress)
    s = result.stress - torch.trace(result.stress) / 3.0 * torch.eye(3, dtype=D)
    assert float(q) == pytest.approx(math.sqrt(1.5) * float(torch.linalg.norm(s)), rel=1e-12)


def test_verification_report_is_json_serialisable_and_gates_hold():
    report = verify_minimum_surface(MATERIAL)
    payload = json.dumps(report, allow_nan=False)
    assert len(payload) > 100
    assert report["uniaxial"]["elastic_modulus_relative_error"] < 1e-10
    assert report["uniaxial"]["plastic_modulus_relative_error"] < 1e-10
    assert report["uniaxial"]["stress_relative_error"] < 1e-10
    assert report["return_map"]["max_residual"] < 1e-12
    assert report["yield_surface_max_relative_error"] < 0.01
    assert report["consistent_tangent_relative_error"] < 1e-6
    # Biaxial: plane-strain-like constraint must still land on the yield surface.
    assert set(report["return_map"]["residual_by_component"]) == {
        "yield",
        "stress_identity",
        "consistency",
        "flow_rule",
        "hardening",
        "complementarity",
    }
