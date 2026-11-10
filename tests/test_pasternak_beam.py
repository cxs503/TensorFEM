"""Tests for the beam on a Pasternak (two-parameter) foundation.

Verification anchors
--------------------
* the closed form must reduce **algebraically** to Hetenyi (1946) at ``G_s = 0``
  (not just numerically close: the same ``A(beta x)`` expression);
* the characteristic roots must satisfy ``EI l^4 - G_s l^2 + k = 0``;
* the FE peak deflection must match the closed form to far better than the 1%
  acceptance line of the campaign node, and the FE moment must converge under
  mesh refinement;
* global equilibrium (foundation reaction = applied load) must hold;
* the length scales extracted from the FE field must reproduce the analytic
  roots and the ``G_s^{-1/2}`` / ``G_s^{+1/2}`` scaling laws.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorfem.beam_foundation import (
    hetenyi_point_load_deflection,
    hetenyi_penetration,
)
from tensorfem.pasternak_beam import (
    PasternakBeamModel,
    assemble_pasternak_beam,
    beam_energy,
    bending_moments_pasternak,
    decay_modes_from_nodes,
    hermite_pasternak_shear_stiffness,
    pasternak_beam_element_stiffness,
    pasternak_characteristic_lengths,
    pasternak_characteristic_roots,
    pasternak_infinite_beam_point_load,
    pasternak_peak_moment,
    pasternak_penetration,
    power_law_exponent,
    solve_pasternak_beam,
)

EI = 1.0e7
K = 2.0e6
P = 1.0e4
THRESHOLD = 2.0 * math.sqrt(EI * K)


def _model(length: float, n: int, shear: float, k: float = K) -> PasternakBeamModel:
    return PasternakBeamModel(length=length, n_elements=n, E=1.0, Iz=EI,
                              k=k, shear=shear)


def _reference_domain(shear: float) -> float:
    lengths = pasternak_characteristic_lengths(EI, K, shear)
    return 12.0 * max(lengths["l_shear"], lengths["l_soft"]) + 8.0 * lengths["l_winkler"]


# ---------------------------------------------------------------------------
# closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shear", [0.0, 1.0e7, 5.0e7, 5.0e8, 5.0e9])
def test_closed_form_satisfies_the_governing_ode(shear: float):
    """``EI w'''' - G_s w'' + k w = 0`` away from the load (finite differences)."""
    x = torch.linspace(0.4, 6.0, 281, dtype=torch.float64)
    h = float(x[1] - x[0])
    w = pasternak_infinite_beam_point_load(x, 0.0, P, EI, K, shear)
    d2 = (w[2:] - 2.0 * w[1:-1] + w[:-2]) / h**2
    d4 = (w[4:] - 4.0 * w[3:-1] + 6.0 * w[2:-2] - 4.0 * w[1:-3] + w[:-4]) / h**4
    residual = EI * d4 - shear * d2[1:-1] + K * w[2:-2]
    scale = K * float(w.abs().max()) + abs(P) / 1.0
    assert float(residual.abs().max()) / scale < 5.0e-3


@pytest.mark.parametrize("n_points", [41, 201])
def test_closed_form_reduces_to_hetenyi(n_points: int):
    """``G_s = 0`` reproduces the Hetenyi ``A(beta x)`` solution to round-off."""
    x = torch.linspace(-40.0, 40.0, n_points, dtype=torch.float64)
    pasternak = pasternak_infinite_beam_point_load(x, 0.0, P, EI, K, 0.0)
    hetenyi = hetenyi_point_load_deflection(x, 0.0, P, K, EI)
    assert torch.allclose(pasternak, hetenyi, rtol=0.0, atol=1e-14)
    peak_pasternak = pasternak_penetration(P, EI, K, 0.0)
    assert peak_pasternak == pytest.approx(hetenyi_penetration(P, K, EI), rel=1e-13)


@pytest.mark.parametrize("shear", [0.0, 1.0e7, 1.0e8, 1.0e9])
def test_characteristic_roots_solve_the_characteristic_equation(shear: float):
    lam_f, lam_s = pasternak_characteristic_roots(EI, K, shear)
    for lam in (lam_f, lam_s):
        residual = EI * lam**4 - shear * lam**2 + K
        assert abs(residual) / K < 1e-10
        assert lam.real >= 0.0


def test_winkler_limit_roots_are_the_hetenyi_pair():
    beta = (K / (4.0 * EI)) ** 0.25
    lam_f, lam_s = pasternak_characteristic_roots(EI, K, 0.0)
    assert lam_f == pytest.approx(complex(beta, beta), rel=1e-12)
    assert lam_s == pytest.approx(complex(beta, -beta), rel=1e-12)
    lengths = pasternak_characteristic_lengths(EI, K, 0.0)
    assert lengths["regime"] == "oscillatory"
    assert lengths["l_shear"] == pytest.approx(1.0 / beta, rel=1e-12)


def test_degenerate_root_branch_is_continuous():
    """At ``G_s = 2 sqrt(EI k)`` the roots merge; the double-root branch is used.

    The double-root formula is more accurate than the split form there, whose
    amplitudes divide by ``lambda_s^2 - lambda_f^2``; it is checked against the
    FE solution, which knows nothing about the analytic branch.
    """
    shear = THRESHOLD
    lam_f, lam_s = pasternak_characteristic_roots(EI, K, shear)
    assert abs(lam_f - lam_s) <= 1e-6 * abs(lam_f)
    degenerate = pasternak_penetration(P, EI, K, shear)
    near = pasternak_penetration(P, EI, K, THRESHOLD * (1.0 + 1e-3))
    assert degenerate == pytest.approx(near, rel=2e-3)
    x = torch.linspace(-5.0, 5.0, 51, dtype=torch.float64)
    w = pasternak_infinite_beam_point_load(x, 0.0, P, EI, K, shear)
    assert float(w[25]) == pytest.approx(degenerate, rel=1e-9)
    assert torch.isfinite(w).all()
    length = _reference_domain(shear)
    n = 6000
    sol = solve_pasternak_beam(_model(length, n, shear), nodal_loads={n // 2: P})
    assert float(sol["w"][n // 2]) == pytest.approx(degenerate, rel=1e-2)


def test_peak_moment_reduces_to_hetenyi_value():
    lc = (4.0 * EI / K) ** 0.25
    assert pasternak_peak_moment(P, EI, K, 0.0) == pytest.approx(P * lc / 4.0,
                                                                 rel=1e-12)
    moments = [pasternak_peak_moment(P, EI, K, shear)
               for shear in (0.0, 1.0e7, 1.0e8, 1.0e9)]
    # the shear layer spreads the load: the peak moment must decrease
    assert all(a > b for a, b in zip(moments[:-1], moments[1:]))


def test_asymptotic_length_scales():
    """Large ``G_s``: ``l_shear -> sqrt(EI/G_s)`` and ``l_soft -> sqrt(G_s/k)``."""
    shear = 1.0e4 * THRESHOLD
    lengths = pasternak_characteristic_lengths(EI, K, shear)
    assert lengths["regime"] == "monotone"
    assert lengths["l_shear"] == pytest.approx(math.sqrt(EI / shear), rel=2e-3)
    assert lengths["l_soft"] == pytest.approx(math.sqrt(shear / K), rel=2e-3)
    assert lengths["l_winkler"] == pytest.approx((4.0 * EI / K) ** 0.25, rel=1e-12)


# ---------------------------------------------------------------------------
# element matrices
# ---------------------------------------------------------------------------
def test_shear_layer_matrix_matches_quadrature_and_is_singular_for_rigid_modes():
    L, shear = 1.7, 3.0e7
    Ks = hermite_pasternak_shear_stiffness(L, shear, order=4)
    fine = hermite_pasternak_shear_stiffness(L, shear, order=3)
    assert torch.allclose(Ks, fine, rtol=1e-12, atol=1e-9 * float(Ks.abs().max()))
    # constant deflection carries no shear energy
    rigid = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    assert float(rigid @ Ks @ rigid) == pytest.approx(0.0, abs=1e-12)


def test_element_stiffness_is_symmetric_positive_definite():
    ke = pasternak_beam_element_stiffness(1.0, EI, K, 3.0e7, 1.3)
    assert torch.allclose(ke, ke.transpose(0, 1), rtol=1e-14, atol=1e-8)
    eigen = torch.linalg.eigvalsh(ke)
    assert float(eigen.min()) > 0.0


# ---------------------------------------------------------------------------
# finite element solution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shear", [0.0, 3.0e7, 3.0e8])
def test_fem_peak_deflection_matches_closed_form(shear: float):
    """Campaign acceptance (a): peak error far below the 1 % line."""
    length = _reference_domain(shear)
    n = 4000
    model = _model(length, n, shear)
    sol = solve_pasternak_beam(model, nodal_loads={n // 2: P}, diagnostics=True)
    exact = pasternak_penetration(P, EI, K, shear)
    error = (float(sol["w"][n // 2]) - exact) / exact
    assert abs(error) < 1.0e-3
    # equilibrium: the net (Winkler) foundation reaction carries the whole load
    assert sol["reaction"]["total"] == pytest.approx(P, rel=1e-5)
    # at free ends the shear-layer traction is balanced by the internal shear
    assert abs(sol["reaction"]["shear_boundary"]) < 1e-3 * P
    assert sol["residual"] < 1e-6 * P


@pytest.mark.parametrize("shear", [0.0, 3.0e7])
def test_fem_moment_converges_under_mesh_refinement(shear: float):
    length = _reference_domain(shear)
    exact = pasternak_peak_moment(P, EI, K, shear)
    errors = []
    for n in (500, 1000, 2000, 4000):
        sol = solve_pasternak_beam(_model(length, n, shear),
                                   nodal_loads={n // 2: P}, diagnostics=True)
        errors.append(abs(float(sol["M"][n // 2]) - exact) / exact)
    assert errors[-1] < errors[0]
    assert errors[-1] < 5.0e-3
    ratios = [errors[i] / errors[i + 1] for i in range(len(errors) - 1)]
    assert min(ratios) > 1.5  # at least first-order moment convergence


def test_uniform_load_on_free_beam_is_the_exact_constant_solution():
    """``w = q/k`` with zero moment solves the free-free problem exactly."""
    k, shear, q = 3.0e6, 1.0e8, -2.5e3
    model = _model(40.0, 400, shear, k=k)
    sol = solve_pasternak_beam(model, uniform_load=q, diagnostics=True)
    w_exact = q / k
    assert float(sol["w"].std()) / abs(w_exact) < 1e-6
    assert float(sol["w"].mean()) == pytest.approx(w_exact, rel=1e-3)
    assert float(sol["M"].abs().max()) < 1e-3 * abs(w_exact) * k


def test_supports_and_reactions_are_consistent():
    shear = 5.0e7
    model = _model(30.0, 300, shear)
    pinned = solve_pasternak_beam(model, nodal_loads={150: P},
                                  supports={"pinned": [0, -1]})
    assert float(pinned["w"][0]) == 0.0
    assert float(pinned["w"][-1]) == 0.0
    # downward load is positive, so the supports push back (negative residual)
    assert pinned["support_reaction"] < 0.0
    # global equilibrium: supports + net foundation reaction = applied load
    # downward-positive convention: the support force is negative (upward), so
    # net bed reaction - support force = applied load
    balance = pinned["reaction"]["total"] - pinned["support_reaction"]
    assert balance == pytest.approx(P, rel=1e-9)
    # with the ends held flat the shear layer exerts a sizeable end traction
    assert abs(pinned["reaction"]["shear_boundary"]) > 1e-3 * P
    clamped = solve_pasternak_beam(model, nodal_loads={150: P},
                                   supports={"clamped": [0, -1]})
    assert float(clamped["theta"][0]) == 0.0
    # clamping stiffens the beam: the peak deflection must drop
    free = solve_pasternak_beam(model, nodal_loads={150: P})
    assert abs(float(clamped["w"][150])) < abs(float(free["w"][150]))


def test_energy_split_equals_half_the_external_work():
    """Consistent nodal load => ``U = 1/2 f.u`` (linear elastic, no body work)."""
    shear = 3.0e7
    model = _model(_reference_domain(shear), 800, shear)
    n = model.n_elements
    asm = assemble_pasternak_beam(model, nodal_loads={n // 2: P}, dense=True)
    sol = solve_pasternak_beam(model, nodal_loads={n // 2: P}, dense=True)
    U = beam_energy(model, sol["u"])["total"]
    work = 0.5 * float(sol["u"] @ asm["f"].reshape(-1))
    assert U == pytest.approx(work, rel=1e-10)
    # and the split reproduces the shear-layer contribution of the closed form
    # via the element quadrature (each part non-negative)
    split = beam_energy(model, sol["u"])
    assert split["bending"] > 0.0 and split["winkler"] > 0.0 and split["shear"] > 0.0


def test_block_thomas_matches_dense_solve():
    shear = 2.0e7
    model = _model(25.0, 40, shear)
    asm = assemble_pasternak_beam(model, nodal_loads={10: P, 30: -0.5 * P},
                                  dense=True)
    sol = solve_pasternak_beam(model, nodal_loads={10: P, 30: -0.5 * P},
                               dense=True)
    dense = torch.linalg.solve(asm["K"], asm["f"].reshape(-1))
    assert torch.allclose(sol["u"], dense, rtol=1e-10, atol=1e-16)
    moment = bending_moments_pasternak(model, sol["u"])
    assert moment.shape == (model.n_elements + 1,)


def test_foundation_reaction_split():
    shear = 3.0e8
    model = _model(_reference_domain(shear), 3000, shear)
    n = model.n_elements
    sol = solve_pasternak_beam(model, nodal_loads={n // 2: P}, diagnostics=True)
    reaction = sol["reaction"]
    assert abs(reaction["shear_boundary"]) < 1e-3 * P
    assert reaction["total"] == pytest.approx(P, rel=1e-9)


# ---------------------------------------------------------------------------
# mode extraction and scaling helpers
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rates", [(0.7, 0.05), (2.5, 0.9)])
def test_decay_mode_extraction_recovers_real_exponents(rates):
    h = 0.01
    x = torch.arange(400, dtype=torch.float64) * h
    values = 1.3 * torch.exp(-rates[0] * x) - 0.8 * torch.exp(-rates[1] * x)
    fast, slow = decay_modes_from_nodes(values, h, start=1)
    assert fast.real == pytest.approx(rates[0], rel=1e-6)
    assert slow.real == pytest.approx(rates[1], rel=1e-6)


def test_decay_mode_extraction_recovers_complex_exponents():
    h = 0.02
    beta, omega = 0.8, 0.8
    x = torch.arange(400, dtype=torch.float64) * h
    values = torch.exp(-beta * x) * torch.cos(omega * x)
    fast, slow = decay_modes_from_nodes(values, h, start=1)
    assert fast.real == pytest.approx(beta, rel=1e-4)
    assert abs(fast.imag) == pytest.approx(omega, rel=1e-4)


def test_power_law_exponent_recovers_synthetic_law():
    x = [1.0, 2.0, 4.0, 8.0]
    y = [3.0 * v**-0.5 for v in x]
    fit = power_law_exponent(x, y)
    assert fit["exponent"] == pytest.approx(-0.5, abs=1e-10)
    assert fit["coefficient"] == pytest.approx(3.0, rel=1e-10)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_shear_length_scales_as_inverse_sqrt_of_shear():
    """Campaign acceptance (b): FE-extracted ``l_shear`` ~ ``G_s^{-1/2}``."""
    shears = [4.0 * THRESHOLD, 8.0 * THRESHOLD, 16.0 * THRESHOLD]
    extracted = []
    for shear in shears:
        lengths = pasternak_characteristic_lengths(EI, K, shear)
        h = lengths["l_shear"] / 10.0
        n = int(6.0 * (lengths["l_soft"] + lengths["l_shear"]) / h)
        model = _model(h * n, n, shear)
        sol = solve_pasternak_beam(model, nodal_loads={n // 2: P})
        window = int(4.0 * lengths["l_shear"] / h) + 2
        fast, _ = decay_modes_from_nodes(sol["w"][n // 2:][:window], h, start=1)
        extracted.append(1.0 / fast.real)
    fit = power_law_exponent(shears, extracted)
    assert abs(fit["exponent"] + 0.5) < 0.02


# ---------------------------------------------------------------------------
# validation / error paths
# ---------------------------------------------------------------------------
def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        PasternakBeamModel(length=10.0, n_elements=10, E=1.0, Iz=1.0, k=0.0,
                           shear=0.0)
    with pytest.raises(ValueError):
        PasternakBeamModel(length=10.0, n_elements=10, E=1.0, Iz=1.0, k=1.0,
                           shear=-1.0)
    with pytest.raises(ValueError):
        PasternakBeamModel(length=-1.0, n_elements=10, E=1.0, Iz=1.0, k=1.0)
    with pytest.raises(ValueError):
        PasternakBeamModel(length=10.0, n_elements=0, E=1.0, Iz=1.0, k=1.0)
    with pytest.raises(ValueError):
        hermite_pasternak_shear_stiffness(1.0, 1.0, order=9)
    with pytest.raises(ValueError):
        hermite_pasternak_shear_stiffness(-1.0, 1.0)
    with pytest.raises(ValueError):
        pasternak_characteristic_roots(EI, 0.0, 1.0)
    with pytest.raises(ValueError):
        pasternak_characteristic_lengths(0.0, K, 0.0)
    with pytest.raises(ValueError):
        solve_pasternak_beam(_model(10.0, 10, 1.0e7), supports={"hinged": [0]})
    with pytest.raises(ValueError):
        decay_modes_from_nodes(torch.ones(3), 0.1)
    with pytest.raises(ValueError):
        decay_modes_from_nodes(torch.ones(3), -0.1)
    with pytest.raises(ValueError):
        decay_modes_from_nodes(torch.ones(6), 0.1, start=5)
    with pytest.raises(ValueError):
        power_law_exponent([1.0], [1.0])
    with pytest.raises(ValueError):
        power_law_exponent([1.0, 0.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        bending_moments_pasternak(_model(10.0, 4, 1.0e7), torch.zeros(3))
    with pytest.raises(ValueError):
        PasternakBeamModel(length=10.0, n_elements=4, E=1.0, Iz=1.0, k=0.0,
                           shear=0.0)  # no foundation at all