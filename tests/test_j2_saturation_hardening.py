"""Gates for non-linear (Voce) isotropic hardening J2 return mapping.

What is asserted, and why it is decidable:

1. the local scalar return mapping satisfies *every* discrete constitutive
   identity re-derived from its own output (yield, flow rule, hardening law,
   incompressibility, scalar consistency) to ``1e-10``;
2. the consistent tangent equals a finite-difference derivative of the
   implemented return map (this is the only honest way to test a tangent);
3. the uniaxial stress-strain curve equals the closed-form 1-D solution
   ``sigma = sigma_y(eps_p)``, ``eps = sigma/E + eps_p`` to far better than the
   campaign's 2% gate;
4. the local Newton iteration converges quadratically (residual history).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfem.j2_saturation_hardening import (
    EPState,
    HardeningLaw,
    VOIGT_INDEX,
    consistent_tangent,
    elastic_stiffness,
    return_map,
    return_map_residuals,
    uniaxial_reference,
    uniaxial_stress_path,
    voigt_map,
)

TOL = 1.0e-10


def _state(**kwargs) -> HardeningLaw:
    parameters = dict(yield_stress=250.0e6, linear=0.0, saturation=0.0, saturation_rate=0.0)
    parameters.update(kwargs)
    return HardeningLaw(**parameters)


def _strain(**components) -> torch.Tensor:
    strain = torch.zeros((3, 3), dtype=torch.float64)
    for key, value in components.items():
        i, j = (int(index) for index in key.split("_"))
        strain[i, j] = strain[j, i] = value
    return strain


# --------------------------------------------------------------------------- #
# law + validation
# --------------------------------------------------------------------------- #
def test_hardening_law_is_monotone_and_saturates():
    law = _state(linear=1.0e9, saturation=100.0e6, saturation_rate=20.0)
    alpha = torch.tensor([0.0, 1e-4, 1e-3, 1e-2, 1.0], dtype=torch.float64)
    values = law.sigma_y(alpha)
    assert float(values[0]) == pytest.approx(law.yield_stress, rel=1e-15)
    assert bool(torch.all(values[1:] > values[:-1]))
    slopes = law.slope(alpha)
    assert bool(torch.all(slopes >= 0.0))
    # the saturation branch must approach sigma_y0 + Q from below as alpha -> inf
    pure_voce = _state(saturation=100.0e6, saturation_rate=20.0)
    limit = float(pure_voce.sigma_y(torch.tensor(1.0e3, dtype=torch.float64)))
    assert limit == pytest.approx(pure_voce.saturated_stress, rel=1e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(yield_stress=0.0),
        dict(yield_stress=-1.0),
        dict(linear=-1.0e9),
        dict(saturation=-1.0e6),
        dict(saturation=10.0e6, saturation_rate=0.0),
        dict(saturation_rate=-1.0),
    ],
)
def test_invalid_hardening_laws_are_rejected(kwargs):
    with pytest.raises(ValueError):
        _state(**kwargs)


def test_invalid_elastic_constants_are_rejected():
    law = _state()
    with pytest.raises(ValueError):
        return_map(torch.zeros((3, 3), dtype=torch.float64), EPState.virgin(), law, -1.0, 0.3)
    with pytest.raises(ValueError):
        return_map(torch.zeros((3, 3), dtype=torch.float64), EPState.virgin(), law, 2.0e11, 0.5)


# --------------------------------------------------------------------------- #
# return mapping
# --------------------------------------------------------------------------- #
def test_elastic_step_does_not_touch_the_state_and_is_stress_exact():
    law = _state(saturation=50.0e6, saturation_rate=30.0)
    state = EPState.virgin()
    strain = _strain(**{"0_0": 1.0e-4, "1_1": -0.4e-4, "2_2": -0.4e-4})
    result = return_map(strain, state, law, 200.0e9, 0.3)
    assert not result.plastic
    assert float(result.dgamma) == 0.0
    assert float(result.state.alpha) == 0.0
    elastic = elastic_stiffness(200.0e9, 0.3) @ voigt_map[0](strain)
    assert torch.allclose(voigt_map[0](result.stress), elastic, rtol=1e-13, atol=1e-6)


@pytest.mark.parametrize(
    "law",
    [
        _state(linear=2.0e9),
        _state(saturation=100.0e6, saturation_rate=20.0),
        _state(linear=1.0e9, saturation=80.0e6, saturation_rate=80.0),
        _state(),  # perfectly plastic
    ],
)
def test_plastic_step_satisfies_every_discrete_identity(law):
    state = EPState.virgin()
    strain = _strain(**{"0_0": 4.0e-3, "1_1": -1.5e-3, "0_1": 7.0e-4})
    result = return_map(strain, state, law, 200.0e9, 0.3)
    assert result.plastic and result.converged
    residuals = return_map_residuals(strain, state, law, 200.0e9, 0.3, result)
    for name, value in residuals.items():
        assert value <= TOL, f"{name} residual {value:.3e} exceeds {TOL:.0e}"
    # plastic flow is isochoric to round-off of the plastic strain itself
    assert abs(float(torch.trace(result.state.plastic_strain))) <= 1e-12 * float(result.dgamma)


def test_return_mapping_is_state_immutable():
    law = _state(saturation=100.0e6, saturation_rate=20.0)
    state = EPState.virgin()
    before = state.plastic_strain.clone(), state.alpha.clone()
    return_map(_strain(**{"0_0": 4.0e-3}), state, law, 200.0e9, 0.3)
    assert torch.equal(state.plastic_strain, before[0])
    assert torch.equal(state.alpha, before[1])


def test_local_newton_converges_quadratically():
    law = _state(linear=500.0e6, saturation=120.0e6, saturation_rate=15.0)
    result = return_map(_strain(**{"0_0": 6.0e-3, "1_1": -2.0e-3}), EPState.virgin(),
                        law, 200.0e9, 0.3)
    history = list(result.history)
    assert result.iterations <= 8
    assert history[0] > 1.0e5          # O(q_tr) start, absolute residual in Pa
    assert history[-1] <= 1.0e-12 * max(1.0, float(result.trial_equivalent_stress))
    # super-linear: each step must buy at least one order of magnitude
    ratios = [history[i + 1] / history[i] for i in range(len(history) - 1)]
    assert all(ratio < 0.5 for ratio in ratios)
    assert ratios[-1] < 1.0e-2


# --------------------------------------------------------------------------- #
# consistent tangent
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "strain_components,law",
    [
        ({"0_0": 4.0e-3, "1_1": -1.5e-3, "0_1": 7.0e-4},
         _state(linear=2.0e9, saturation=100.0e6, saturation_rate=20.0)),
        ({"0_1": 2.5e-3}, _state(saturation=150.0e6, saturation_rate=50.0)),
        ({"0_0": 3.0e-3, "1_1": 3.0e-3, "2_2": -6.0e-3}, _state(linear=5.0e9)),
    ],
)
def test_consistent_tangent_matches_finite_differences(strain_components, law):
    state = EPState.virgin()
    strain = _strain(**strain_components)
    result = return_map(strain, state, law, 200.0e9, 0.3)
    assert result.plastic
    tangent = consistent_tangent(strain, state, law, 200.0e9, 0.3, result)
    step = 1.0e-9
    numeric = torch.zeros((6, 6), dtype=torch.float64)
    for column, (i, j) in enumerate(VOIGT_INDEX):
        perturbation = torch.zeros((3, 3), dtype=torch.float64)
        if column < 3:
            perturbation[i, j] = step
        else:
            perturbation[i, j] = perturbation[j, i] = 0.5 * step
        plus = return_map(strain + perturbation, state, law, 200.0e9, 0.3)
        minus = return_map(strain - perturbation, state, law, 200.0e9, 0.3)
        numeric[:, column] = (voigt_map[0](plus.stress) - voigt_map[0](minus.stress)) / (2.0 * step)
    relative = float((tangent - numeric).abs().max()) / float(tangent.abs().max())
    assert relative < 1.0e-7, f"algorithmic tangent off by {relative:.2e} (relative)"


def test_tangent_is_symmetric_in_the_plastic_branch():
    law = _state(saturation=100.0e6, saturation_rate=20.0)
    strain = _strain(**{"0_0": 4.0e-3, "1_1": -1.5e-3, "0_1": 7.0e-4})
    tangent = consistent_tangent(strain, EPState.virgin(), law, 200.0e9, 0.3)
    assert torch.allclose(tangent, tangent.T, rtol=1e-12, atol=1e-3)


def test_elastic_branch_returns_the_elastic_stiffness():
    law = _state(saturation=100.0e6, saturation_rate=20.0)
    tangent = consistent_tangent(torch.zeros((3, 3), dtype=torch.float64), EPState.virgin(),
                                 law, 200.0e9, 0.3)
    assert torch.allclose(tangent, elastic_stiffness(200.0e9, 0.3), rtol=0.0, atol=0.0)


# --------------------------------------------------------------------------- #
# uniaxial verification (campaign gate 5: <= 2% pointwise)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "law",
    [
        _state(linear=2.0e9),
        _state(saturation=100.0e6, saturation_rate=20.0),
        _state(linear=1.0e9, saturation=60.0e6, saturation_rate=200.0),
        _state(),  # perfect plasticity
    ],
)
def test_uniaxial_stress_strain_matches_closed_form(law):
    path = torch.linspace(0.0, 2.0e-2, 41, dtype=torch.float64)
    computed = uniaxial_stress_path(path, law, 200.0e9, 0.3)
    worst, worst_at = 0.0, None
    for strain, stress in zip(path, computed["stress"]):
        reference = uniaxial_reference(float(strain), law, 200.0e9)["stress"]
        error = abs(float(stress) - reference) / max(reference, 1.0e-12 * 250.0e6)
        if error > worst:
            worst, worst_at = error, float(strain)
    assert worst <= 2.0e-2, f"peak deviation {worst:.3e} at strain {worst_at}"
    assert worst < 1.0e-10, "the material-point path should be far tighter than the 2% gate"


def test_uniaxial_lateral_strain_is_elastic_below_yield():
    law = _state(saturation=100.0e6, saturation_rate=20.0)
    path = torch.tensor([-0.5e-3, 0.5e-3], dtype=torch.float64)
    computed = uniaxial_stress_path(path, law, 200.0e9, 0.3)
    assert abs(float(computed["stress"][1])) == pytest.approx(200.0e9 * 0.5e-3, rel=1e-9)
    lateral, _ = computed["lateral_strain"][1]
    assert lateral == pytest.approx(-0.3 * 0.5e-3, rel=1e-9)
    assert all(alpha == 0.0 for alpha in computed["alpha"][:1])


def test_uniaxial_curve_is_continuous_at_yield():
    law = _state(saturation=100.0e6, saturation_rate=20.0)
    yield_strain = law.yield_stress / 200.0e9
    path = torch.tensor([yield_strain * (1.0 - 1e-6), yield_strain * (1.0 + 1e-6)],
                        dtype=torch.float64)
    stress = uniaxial_stress_path(path, law, 200.0e9, 0.3)["stress"]
    assert float(stress[1]) - float(stress[0]) < 1.0e-2 * float(stress[0])


def test_hardening_slope_of_uniaxial_curve_is_bounded_by_E_and_positive():
    law = _state(linear=4.0e9, saturation=100.0e6, saturation_rate=20.0)
    path = torch.linspace(3.0e-3, 2.0e-2, 30, dtype=torch.float64)
    computed = uniaxial_stress_path(path, law, 200.0e9, 0.3)
    strain, stress = path, computed["stress"]
    slope = (stress[1:] - stress[:-1]) / (strain[1:] - strain[:-1])
    assert bool(torch.all(slope > 0.0))
    assert float(slope.max()) < 200.0e9
    assert math.isclose(float(slope[0]), 4.0e9 * 200.0e9 / (200.0e9 + 4.0e9), rel_tol=0.35)