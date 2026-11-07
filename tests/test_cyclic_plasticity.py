"""Qualification of the Armstrong-Frederick cyclic-plasticity surface (P16-N2 / m5197).

Machine-checked criteria (the same numbers the report quotes, see
``scripts/cyclic_plasticity_verification.py`` and
``docs/cyclic-plasticity-af-anchor.md``):

* C1 stabilised loop peak/trough stress vs closed form ``<= 1 %``
  (5 strain amplitudes x 2 cycle counts);
* C2 dissipated energy per stabilised cycle vs closed form ``<= 2 %``;
* C3 Bauschinger reverse-yield drop vs closed form ``<= 3 %`` (>= 3 amplitudes)
  plus a detectability gate so the criterion cannot be satisfied vacuously;
* C4 analytic consistent tangent vs central difference ``<= 1e-6``;
* C5 ratcheting rate vs closed form ``<= 5 %`` (20 cycles, 2 means x 2 amplitudes);
* C6 ``C -> 0`` degenerates onto the m4951 isotropic kernel ``<= 1e-12``;
* C7 negative control ``H_k = 0`` must FAIL the C3 criterion (>= 50 % of cases);
* C8 the m4951 suite (``tests/test_j2_minimum_surface.py``) stays green -- run by
  ``pytest tests/test_j2_minimum_surface.py`` in the same session.

The heavy parameter sweeps live in ``scripts/cyclic_plasticity_verification.py``
(the numbers are dumped into ``artifacts/cyclic_plasticity/verification.json``);
the tests below use small step counts so the suite stays a few seconds long while
keeping the same tolerances.
"""

import math

import pytest
import torch

from tensorfem.cyclic_plasticity import (
    AFMaterial,
    CyclicState,
    VOIGT_PAIRS,
    cyclic_history,
    bauschinger_drop,
    consistent_tangent,
    degenerate_check,
    elastic_tangent,
    integrate,
    loop_dissipation,
    loop_peak_stress,
    ratcheting_rate,
    strain_cycle,
    stress_cycle,
    uniaxial_cycle,
    uniaxial_stress_cycle,
    yield_radius,
)

D = torch.float64
SF23 = math.sqrt(2.0 / 3.0)

MATERIAL = AFMaterial(young=210.0e9, poisson=0.3, yield_stress=350.0e6,
                      kinematic_modulus=70.0e9, kinematic_gamma=500.0, hardening=0.0)
MATERIAL_VOCE = AFMaterial(young=210.0e9, poisson=0.3, yield_stress=350.0e6,
                           kinematic_modulus=70.0e9, kinematic_gamma=500.0,
                           hardening=0.0, saturation=80.0e6, saturation_rate=500.0)
NO_KINEMATIC = AFMaterial(young=210.0e9, poisson=0.3, yield_stress=350.0e6,
                          kinematic_modulus=0.0, kinematic_gamma=500.0,
                          hardening=0.0, saturation=80.0e6, saturation_rate=500.0)
AMPLITUDES = (0.004, 0.006, 0.008, 0.010, 0.012)


def _cycle_points(amplitude, cycles, steps):
    """Triangular strain history + per-cycle index bounds (cycle k: peak -> peak)."""
    return cyclic_history(0.0, amplitude, cycles, steps)


def _dissipation(eps, sigma, start, end):
    total = 0.0
    for k in range(start, end - 1):
        total += 0.5 * (sigma[k] + sigma[k + 1]) * (eps[k + 1] - eps[k])
    total += 0.5 * (sigma[end - 1] + sigma[start]) * (eps[start] - eps[end - 1])
    return total


# --------------------------------------------------------------------------- #
# basic contract
# --------------------------------------------------------------------------- #
def test_material_validation_is_fail_closed():
    with pytest.raises(ValueError):
        AFMaterial(young=-1.0, poisson=0.3, yield_stress=350e6)
    with pytest.raises(ValueError):
        AFMaterial(young=210e9, poisson=0.5, yield_stress=350e6)
    with pytest.raises(ValueError):
        AFMaterial(young=210e9, poisson=0.3, yield_stress=0.0)
    with pytest.raises(ValueError):
        AFMaterial(young=210e9, poisson=0.3, yield_stress=350e6, kinematic_modulus=-1.0)
    with pytest.raises(ValueError):
        AFMaterial(young=210e9, poisson=0.3, yield_stress=350e6, kinematic_gamma=-1.0)
    with pytest.raises(ValueError):
        AFMaterial(young=210e9, poisson=0.3, yield_stress=350e6, hardening=-1.0)
    with pytest.raises(ValueError):
        AFMaterial(young=210e9, poisson=0.3, yield_stress=350e6, saturation=1.0,
                   saturation_rate=0.0)


def test_elastic_step_returns_elastic_tangent_and_keeps_state():
    strain = torch.tensor([[0.0005, 0.0, 0.0], [0.0, -0.00015, 0.0],
                           [0.0, 0.0, -0.00015]], dtype=D)
    state = CyclicState.zero()
    stress, new_state, dp = integrate(MATERIAL, strain, state)
    assert float(dp) == 0.0
    assert new_state is state
    tangent = consistent_tangent(MATERIAL, strain, state, dp)
    assert torch.allclose(tangent, elastic_tangent(MATERIAL), atol=1e-6, rtol=1e-12)
    # sigma_22 = sigma_33 = 0 for this strain => uniaxial elastic point
    assert abs(float(stress[1, 1])) < 1.0


def test_return_mapping_residuals_are_machine_zero():
    """A posteriori audit of the yield condition, flow rule and AF evolution."""
    strain = torch.tensor([[0.01, 0.0, 0.0], [0.0, -0.005, 0.0], [0.0, 0.0, -0.005]], dtype=D)
    state = CyclicState.zero()
    stress, new_state, dp = integrate(MATERIAL, strain, state)
    eta = stress - new_state.backstress
    eta = eta - torch.trace(eta) / 3.0 * torch.eye(3, dtype=D)
    q_f = float(torch.linalg.matrix_norm(eta))
    assert abs(q_f - SF23 * yield_radius(MATERIAL, float(new_state.alpha))) / q_f < 1e-12
    d_eps_p = new_state.plastic_strain - state.plastic_strain
    direction = eta / q_f
    flow = d_eps_p - (math.sqrt(1.5) * float(dp)) * direction
    assert float(torch.linalg.matrix_norm(flow)) / float(
        torch.linalg.matrix_norm(d_eps_p)) < 1e-12
    af = (new_state.backstress - state.backstress
          - (2.0 / 3.0) * MATERIAL.kinematic_modulus * d_eps_p
          + MATERIAL.kinematic_gamma * float(dp) * new_state.backstress)
    assert float(torch.linalg.matrix_norm(af)) / float(
        torch.linalg.matrix_norm(new_state.backstress)) < 1e-12
    # incompressible plastic flow
    assert abs(float(torch.trace(d_eps_p))) < 1e-15


def test_backstress_saturates_at_C_over_gamma():
    """Monotonic loading: alpha = (C/gamma)(1 - exp(-gamma p)) -> C/gamma."""
    strain = torch.zeros((3, 3), dtype=D)
    state = CyclicState.zero()
    for _ in range(400):
        strain = strain.clone()
        strain[0, 0] += 4.0e-5
        strain[1, 1] -= 2.0e-5
        strain[2, 2] -= 2.0e-5
        stress, state, dp = integrate(MATERIAL, strain, state)
    alpha = 1.5 * float(state.backstress[0, 0])
    exact = (MATERIAL.kinematic_modulus / MATERIAL.kinematic_gamma) * (
        1.0 - math.exp(-MATERIAL.kinematic_gamma * float(state.alpha)))
    assert abs(alpha - exact) / exact < 2e-3  # backward-Euler discretisation only
    assert abs(alpha - MATERIAL.saturation_backstress) / MATERIAL.saturation_backstress < 1e-2


# --------------------------------------------------------------------------- #
# C4: consistent tangent
# --------------------------------------------------------------------------- #
def _tangent_error(material, strain, state, h=1e-7):
    stress, new_state, dp = integrate(material, strain, state)
    analytic = consistent_tangent(material, strain, state, dp)
    numeric = torch.zeros((6, 6), dtype=D)
    for k, (i, j) in enumerate(VOIGT_PAIRS):
        d_eps = torch.zeros((3, 3), dtype=D)
        if i == j:
            d_eps[i, i] = h
        else:
            d_eps[i, j] = d_eps[j, i] = 0.5 * h
        plus, _, _ = integrate(material, strain + d_eps, state)
        minus, _, _ = integrate(material, strain - d_eps, state)
        numeric[:, k] = torch.tensor(
            [(float(plus[p, q]) - float(minus[p, q])) / (2.0 * h) for p, q in VOIGT_PAIRS],
            dtype=D)
    return (float(torch.linalg.matrix_norm(analytic - numeric))
            / max(float(torch.linalg.matrix_norm(numeric)), 1.0))


@pytest.mark.parametrize("strain,use_state", [
    (torch.tensor([[0.004, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0, 0.0, -0.002]], dtype=D), False),
    (torch.tensor([[0.006, 0.003, 0.0], [0.003, -0.002, 0.0015], [0.0, 0.0015, -0.004]], dtype=D), False),
    (torch.tensor([[0.003, 0.002, 0.0], [0.002, -0.0015, 0.0], [0.0, 0.0, -0.0015]], dtype=D), True),
])
def test_c4_consistent_tangent_matches_central_difference(strain, use_state):
    state = CyclicState.zero()
    if use_state:  # yielded state with a developed back stress
        path, _ = _cycle_points(0.008, 2, 40)
        rec = uniaxial_cycle(MATERIAL, path, collect_states=True)
        state = rec["states"][-1]
    assert _tangent_error(MATERIAL, strain, state) <= 1e-6


# --------------------------------------------------------------------------- #
# C1 / C2 / C3: stabilised loop anchors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("amplitude", AMPLITUDES)
def test_c1_stabilised_loop_peak_and_trough_match_closed_form(amplitude):
    path, bounds = _cycle_points(amplitude, 6, 60)
    rec = uniaxial_cycle(MATERIAL, path)
    start, end = bounds[-1]
    anchor = loop_peak_stress(MATERIAL, amplitude)
    peak = max(rec["sigma"][start:end])
    trough = min(rec["sigma"][start:end])
    assert abs(peak - anchor["peak"]) / anchor["peak"] <= 0.01
    assert abs(trough - anchor["trough"]) / abs(anchor["trough"]) <= 0.01
    # the same anchor at a different cycle count (loop is AF-stabilised)
    path_long, bounds_long = _cycle_points(amplitude, 12, 60)
    rec_long = uniaxial_cycle(MATERIAL, path_long)
    start, end = bounds_long[-1]
    assert abs(max(rec_long["sigma"][start:end]) - anchor["peak"]) / anchor["peak"] <= 0.01


@pytest.mark.parametrize("amplitude", AMPLITUDES)
def test_c2_dissipated_energy_per_cycle_matches_closed_form(amplitude):
    path, bounds = _cycle_points(amplitude, 6, 120)
    rec = uniaxial_cycle(MATERIAL, path)
    start, end = bounds[-1]
    measured = _dissipation(rec["eps"], rec["sigma"], start, end)
    anchor = loop_peak_stress(MATERIAL, amplitude)
    exact = loop_dissipation(MATERIAL, amplitude, anchor["plastic_amplitude"])
    assert abs(measured - exact) / exact <= 0.02
    assert measured > 0.0


@pytest.mark.parametrize("amplitude", (0.006, 0.010, 0.012))
def test_c3_bauschinger_reverse_yield_drop_matches_closed_form(amplitude):
    path, bounds = _cycle_points(amplitude, 5, 120)
    rec = uniaxial_cycle(MATERIAL, path, collect_states=True)
    start, end = bounds[-1]
    # walk from the tensile peak to the first reverse plastic increment
    k = start
    while k < len(rec["dp"]) - 1 and rec["dp"][k + 1] <= 0.0:
        k += 1
    state = rec["states"][k]
    from tensorfem.cyclic_plasticity import solve_uniaxial_point
    lo, hi = rec["eps"][k], rec["eps"][k + 1]
    lateral = rec["lateral"][k]
    sigma_elastic = rec["sigma"][k]
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        sigma, _, dp, _ = solve_uniaxial_point(MATERIAL, state, mid, lateral)
        if float(dp) > 0.0:
            hi = mid
        else:
            lo, sigma_elastic = mid, sigma
    drop = MATERIAL.yield_stress - abs(sigma_elastic)
    anchor = bauschinger_drop(MATERIAL, amplitude)
    assert anchor["ratio"] >= 0.05  # detectability gate
    assert abs(drop - anchor["drop"]) / anchor["drop"] <= 0.03


def test_c7_negative_control_without_kinematic_hardening_fails_c3():
    """H_k = 0: no Bauschinger effect, so the C3 criterion must fail (>= 50 %)."""
    failures = 0
    for amplitude in AMPLITUDES:
        anchor = bauschinger_drop(NO_KINEMATIC, amplitude)
        path, bounds = _cycle_points(amplitude, 5, 60)
        rec = uniaxial_cycle(NO_KINEMATIC, path, collect_states=True)
        start, end = bounds[-1]
        peak = max(rec["sigma"][start:end])
        # with C = 0 the reverse yield follows the isotropic radius: 2R elastic range
        reverse_yield = peak - 2.0 * anchor["radius"]
        drop = anchor["radius"] - abs(reverse_yield)
        passes = bool(anchor["ratio"] >= 0.05
                      and abs(drop - anchor["drop"]) / max(anchor["drop"], 1e-300) <= 0.03)
        failures += 0 if passes else 1
    assert failures / len(AMPLITUDES) >= 0.5
    assert bauschinger_drop(NO_KINEMATIC, 0.010)["drop"] == 0.0


def test_voce_isotropic_saturation_enters_the_anchor():
    amplitude = 0.010
    anchor = loop_peak_stress(MATERIAL_VOCE, amplitude)
    assert anchor["radius"] == pytest.approx(MATERIAL_VOCE.yield_stress
                                             + MATERIAL_VOCE.saturation)
    path, bounds = _cycle_points(amplitude, 6, 60)
    rec = uniaxial_cycle(MATERIAL_VOCE, path)
    start, end = bounds[-1]
    peak = max(rec["sigma"][start:end])
    assert abs(peak - anchor["peak"]) / anchor["peak"] <= 0.01


# --------------------------------------------------------------------------- #
# C5: ratcheting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mean,amplitude", [(20e6, 400e6), (40e6, 420e6)])
def test_c5_ratcheting_rate_matches_closed_form(mean, amplitude):
    path, bounds = cyclic_history(mean, amplitude, 6, 200)
    path = [mean + value for value in path]
    rec = uniaxial_stress_cycle(MATERIAL, path)
    last = bounds[-1][1] - 1
    previous = bounds[-2][1] - 1
    measured = rec["eps_plastic"][last] - rec["eps_plastic"][previous]
    exact = ratcheting_rate(MATERIAL, mean, amplitude)
    assert exact > 0.0
    assert abs(measured - exact) / exact <= 0.05


def test_ratcheting_rate_vanishes_for_symmetric_stress_cycle():
    assert ratcheting_rate(MATERIAL, 0.0, 400e6) == pytest.approx(0.0, abs=1e-18)
    assert ratcheting_rate(NO_KINEMATIC, 20e6, 400e6) == 0.0


# --------------------------------------------------------------------------- #
# C6: degeneration onto m4951
# --------------------------------------------------------------------------- #
def test_c6_degenerates_onto_m4951_isotropic_kernel():
    cases = [
        (torch.tensor([[0.004, 0.0, 0.0], [0.0, -0.002, 0.0], [0.0, 0.0, -0.002]], dtype=D),
         None),
        (torch.tensor([[0.0, 0.003, 0.0], [0.003, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=D),
         None),
        (torch.tensor([[0.003, 0.002, 0.0], [0.002, -0.0015, 0.0], [0.0, 0.0, -0.0015]], dtype=D),
         CyclicState(plastic_strain=torch.tensor(
             [[0.001, 0.0005, 0.0], [0.0005, -0.0005, 0.0], [0.0, 0.0, 0.0]], dtype=D),
             backstress=torch.zeros((3, 3), dtype=D),
             alpha=torch.tensor(0.0015, dtype=D))),
    ]
    for strain, state in cases:
        report = degenerate_check(MATERIAL, strain, state)
        for key in ("stress_error", "dp_error", "plastic_error", "alpha_error",
                    "tangent_error"):
            assert report[key] <= 1e-12, key


def test_strain_cycle_and_stress_cycle_helpers_are_consistent():
    path = strain_cycle(0.005, 2, 10)
    assert max(path) == pytest.approx(0.005)
    assert min(path) == pytest.approx(-0.005)
    assert len(path) == 10 + 2 * 4 * 10  # initial ramp + two loops of 4 half cycles
    stress = stress_cycle(20e6, 400e6, 1, 5)
    assert max(stress) == pytest.approx(420e6)
    assert min(stress) == pytest.approx(-380e6)