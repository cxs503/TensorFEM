"""Qualification tests for the plane-stress J2 membrane surface.

The assertions are invariants, not smoke checks:

* the batched 3-D kernel is *the same operator* as ``plasticity.update_j2`` /
  ``j2_consistent_tangent`` (bitwise-level agreement on random states);
* the plane-stress return map satisfies ``sigma_zz = 0`` and the consistency
  condition ``q = sigma_y + H alpha`` at the returned state;
* the uniaxial response reproduces the closed form, including the plastic slope
  ``E H / (E + H)`` (not ``H`` -- the elastic strain keeps growing);
* the dissipated work ``int (sigma_y + H alpha) dalpha`` equals the external work
  minus the stored elastic energy, both at a material point (closed strain cycle)
  and on the assembled finite-element path;
* energy dissipation is monotone (second law at every Gauss point).
"""

import math

import pytest
import torch

from tensorfem.j2_plane_stress import (
    J2PlaneStressMaterial,
    MembraneState,
    cook_membrane_model,
    dent_specimen_model,
    elastic_solve,
    gauss_response,
    mises_plane_stress,
    plane_stress_return,
    q4_kinematics,
    run_path,
    uniaxial_plane_stress,
)
from tensorfem import j2_plane_stress as kernel
from tensorfem.plasticity import J2State, j2_consistent_tangent, update_j2

D = torch.float64


def _material(**kwargs):
    base = dict(young=210e3, poisson=0.3, yield_stress=250.0, hardening=2000.0)
    base.update(kwargs)
    return J2PlaneStressMaterial(**base)


def _random_strain(n, scale=0.02, seed=3):
    generator = torch.Generator().manual_seed(seed)
    return (torch.rand(n, 3, generator=generator, dtype=D) - 0.5) * scale


def test_material_card_is_fail_closed():
    bad = [
        dict(young=-1.0, poisson=0.3, yield_stress=250.0),
        dict(young=210e3, poisson=0.5, yield_stress=250.0),
        dict(young=210e3, poisson=-1.0, yield_stress=250.0),
        dict(young=210e3, poisson=0.3, yield_stress=0.0),
        dict(young=210e3, poisson=0.3, yield_stress=250.0, hardening=-10.0),
    ]
    for card in bad:
        with pytest.raises(ValueError):
            J2PlaneStressMaterial(**card)


def test_batched_kernel_is_the_oracle_kernel():
    """The vectorised kernel must equal ``plasticity.update_j2`` point by point."""
    material = _material()
    strain = _random_strain(64)
    state = MembraneState.virgin(64)
    stress, new_state, _, _, diagnostics = plane_stress_return(material, strain, state)
    for i in range(strain.shape[0]):
        strain3 = torch.zeros(3, 3, dtype=D)
        strain3[0, 0], strain3[1, 1], strain3[2, 2] = strain[i, 0], strain[i, 1], new_state.eps_zz[i]
        strain3[0, 1] = strain3[1, 0] = 0.5 * strain[i, 2]
        reference, ref_state, _ = update_j2(strain3, material.young, material.poisson,
                                            material.yield_stress, material.hardening,
                                            J2State(torch.zeros(3, 3, dtype=D),
                                                    torch.zeros((), dtype=D)))
        got = stress[i]
        want = torch.stack([reference[0, 0], reference[1, 1], reference[0, 1]])
        assert float((got - want).abs().max()) < 1e-10 * material.yield_stress
        assert abs(float(new_state.alpha[i]) - float(ref_state.alpha)) < 1e-11


def test_batched_tangent_is_the_oracle_tangent():
    material = _material()
    strain = _random_strain(32, seed=11)
    state = MembraneState.virgin(32)
    _, new_state, _, tangent, _ = plane_stress_return(material, strain, state)
    for i in range(strain.shape[0]):
        strain3 = torch.zeros(3, 3, dtype=D)
        strain3[0, 0], strain3[1, 1], strain3[2, 2] = strain[i, 0], strain[i, 1], new_state.eps_zz[i]
        strain3[0, 1] = strain3[1, 0] = 0.5 * strain[i, 2]
        _, _, dgamma = update_j2(strain3, material.young, material.poisson,
                                 material.yield_stress, material.hardening,
                                 J2State(torch.zeros(3, 3, dtype=D), torch.zeros((), dtype=D)))
        full = j2_consistent_tangent(strain3, material.young, material.poisson,
                                     material.hardening,
                                     J2State(torch.zeros(3, 3, dtype=D), torch.zeros((), dtype=D)),
                                     dgamma)
        in_plane = torch.tensor([0, 1, 3])
        condensed = (full[in_plane][:, in_plane]
                     - torch.outer(full[in_plane, 2], full[2, in_plane]) / full[2, 2])
        scale = float(full.abs().max())
        assert float((condensed - tangent[i]).abs().max()) < 1e-9 * scale


def test_plane_stress_constraint_and_consistency():
    material = _material()
    strain = torch.cat([_random_strain(100, scale=0.002, seed=7),
                        _random_strain(100, scale=0.03, seed=8)])
    state = MembraneState.virgin(200)
    stress, new_state, dWp, _, diagnostics = plane_stress_return(material, strain, state)
    assert diagnostics["sigma_zz_max_abs"] < 1e-9 * material.yield_stress
    surface = material.yield_stress + material.hardening * new_state.alpha
    mises = mises_plane_stress(stress)
    elastic = new_state.alpha <= 0.0
    # elastic points stay inside the surface, plastic points sit exactly on it
    assert bool(elastic.any()) and bool((~elastic).any())
    assert float((mises[elastic] - surface[elastic]).max()) <= 1e-9 * material.yield_stress
    assert float((mises[~elastic] - surface[~elastic]).abs().max()) < 1e-8 * material.yield_stress
    assert float(dWp.min()) >= 0.0


def test_uniaxial_closed_form_including_plastic_slope():
    material = _material()
    strain = torch.linspace(0.0, 0.03, 61, dtype=D)
    sigma, lateral, state, diagnostics = uniaxial_plane_stress(material, strain)
    reference = material.uniaxial_stress(strain)
    assert float(((sigma - reference).abs() / reference.clamp(min=1.0)).max()) < 1e-11
    assert float(diagnostics["sigma_zz_max_abs"]) < 1e-9 * material.yield_stress
    # the transverse stress vanishes on the closed-form lateral strain
    strain_plane = torch.stack([strain, lateral, torch.zeros_like(strain)], dim=1)
    plane_stress, _, _, _, _ = plane_stress_return(material, strain_plane,
                                                   MembraneState.virgin(strain.numel()))
    assert float(plane_stress[:, 1].abs().max()) < 1e-9 * material.yield_stress
    # the plastic slope is E H /(E+H); with H ~ E that is clearly not H
    stiff = _material(hardening=0.5 * 210e3)
    strain_long = torch.linspace(0.0, 0.03, 61, dtype=D)
    sigma_long = uniaxial_plane_stress(stiff, strain_long)[0]
    slope = float((sigma_long[-1] - sigma_long[40]) / (strain_long[-1] - strain_long[40]))
    expected = stiff.young * stiff.hardening / (stiff.young + stiff.hardening)
    assert abs(slope - expected) / expected < 1e-6
    assert abs(slope - stiff.hardening) / stiff.hardening > 0.3
    # the hardening parameter also matches the closed form
    assert float((state.alpha - material.uniaxial_hardening_parameter(strain)).abs().max()) < 1e-12


def test_material_point_work_equals_stored_plus_dissipated():
    """Monotone load: ``int sigma d eps = U_elastic + int (sigma_y + H a) da``.

    Then a *partial* unload: the hardening parameter must freeze (elastic) and the
    unloading slope must be Young's modulus.  (A full unload to zero strain is not
    elastic for this card: the residual plastic strain ``alpha_max = 0.0186`` is far
    above ``sigma_y/E``, so the specimen would yield in reverse compression -- which
    is physics, not a solver defect, and is why the cycle check is partial.)
    """
    material = _material()
    peak = 0.02
    path = torch.linspace(0.0, peak, 401, dtype=D)
    state = MembraneState.virgin(1)
    sigma_prev, eps_prev, work = 0.0, 0.0, 0.0
    for value in path:
        eps = float(value)
        alpha = float(material.uniaxial_hardening_parameter(value.reshape(1)))
        sigma_axial = float(material.uniaxial_stress(value.reshape(1)))
        lateral = -material.poisson * sigma_axial / material.young - 0.5 * alpha
        stress, state, _, _, _ = plane_stress_return(
            material, torch.tensor([[eps, lateral, 0.0]], dtype=D), state)
        sigma = float(stress[0, 0])
        work += 0.5 * (sigma + sigma_prev) * (eps - eps_prev)
        sigma_prev, eps_prev = sigma, eps
    alpha_max = float(state.alpha[0])
    elastic_energy = 0.5 * float(stress[0, 0]) * (eps_prev - alpha_max)
    dissipated = float(material.plastic_work_closed_form(torch.tensor([alpha_max], dtype=D)))
    assert dissipated > 0.0
    assert abs(work - elastic_energy - dissipated) / work < 1e-4
    # elastic unloading: alpha frozen, slope E
    delta = 0.5 * float(material.uniaxial_stress(torch.tensor([peak], dtype=D))) / material.young
    eps_unload = peak - delta
    sigma_peak = float(material.uniaxial_stress(torch.tensor([peak], dtype=D)))
    sigma_unload = sigma_peak - material.young * delta
    lateral = -material.poisson * sigma_unload / material.young - 0.5 * alpha_max
    stress, after, _, _, _ = plane_stress_return(
        material, torch.tensor([[eps_unload, lateral, 0.0]], dtype=D), state)
    assert abs(float(after.alpha[0]) - alpha_max) / alpha_max < 1e-12
    slope = (sigma_peak - float(stress[0, 0])) / delta
    assert abs(slope - material.young) / material.young < 1e-6
    assert abs(float(stress[0, 1])) < 1e-9 * material.yield_stress


def test_element_patch_uniform_strain():
    """A single Q4 element under a uniform strain reproduces the point operator."""
    material = _material()
    nodes = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=D)
    elements = torch.tensor([[0, 1, 2, 3]])
    from tensorfem.j2_plane_stress import MembraneProblem
    problem = MembraneProblem(nodes, elements, 1.0, material, torch.zeros(0, dtype=torch.long))
    strain = torch.tensor([1.5e-3, -4.0e-4, 7.0e-4], dtype=D)
    u = torch.zeros(8, dtype=D)
    for node in range(4):
        x, y = float(nodes[node, 0]), float(nodes[node, 1])
        u[2 * node] = strain[0] * x + 0.5 * strain[2] * y
        u[2 * node + 1] = strain[1] * y + 0.5 * strain[2] * x
    response = gauss_response(problem, u, MembraneState.virgin(4))
    reference, _, _, _, _ = plane_stress_return(material, strain.reshape(1, 3),
                                                MembraneState.virgin(1))
    assert float((response.stress.reshape(-1, 3) - reference).abs().max()) < 1e-9 * material.yield_stress
    # the nodal internal forces of a uniform-strain element sum to zero
    assert float(response.fint.reshape(4, 2).sum(0).abs().max()) < 1e-9 * material.yield_stress


def test_cook_membrane_elastic_literature_anchor():
    """Simo & Rifai (1990) reference 23.96 at the loaded-edge midpoint (Q4 n=32)."""
    material = J2PlaneStressMaterial(young=1.0, poisson=1.0 / 3.0, yield_stress=1e9)
    model = cook_membrane_model(32, material)
    solution = elastic_solve(model.problem, fext=model.load(1.0))
    value = float(solution["u"][model.response_dof])
    assert abs(value - 23.96) / 23.96 < 0.01


def test_fe_energy_balance_and_monotone_dissipation():
    material = _material()
    model = cook_membrane_model(16, material)
    elastic = elastic_solve(model.problem, fext=model.load(1.0))
    yield_load = material.yield_stress / float(mises_plane_stress(elastic["stress"]).max())
    targets = [{"fext": model.load(2.0 * yield_load * k / 20.0), "label": str(k)}
               for k in range(1, 21)]
    result = run_path(model.problem, targets, response_dofs=[model.response_dof])
    history = result["history"]
    assert all(entry["converged"] for entry in history)
    final = history[-1]
    closure = abs(final["external_work"] - final["elastic_energy"] - final["plastic_work"])
    assert closure / abs(final["external_work"]) < 1e-3
    assert final["plastic_work"] > 0.0
    increments = [history[0]["plastic_work"]] + [
        b["plastic_work"] - a["plastic_work"] for a, b in zip(history, history[1:])]
    assert min(increments) >= 0.0
    # incremental accounting (sum of per-Gauss-point dWp) equals the closed form
    assert abs(final["plastic_work_incremental"] - final["plastic_work"]) / final["plastic_work"] < 1e-10


def test_dent_limit_load_is_inside_the_limit_analysis_bracket():
    """Plane-stress net-section bracket: sigma_y t w <= P <= 1.1547 sigma_y t w."""
    material = J2PlaneStressMaterial(young=200e3, poisson=0.3, yield_stress=300.0,
                                     hardening=300.0)
    specimen = dent_specimen_model(16, 14, width=100.0, height=120.0, notch_length=25.0,
                                   notch_height=40.0, thickness=1.0, material=material)
    top = specimen.top_edge
    displacement = 0.5
    steps = 8
    targets = [{"prescribed": (top, torch.full((top.numel(),),
                                               displacement * k / steps, dtype=D)),
                "label": str(k)} for k in range(1, steps + 1)]
    result = run_path(specimen.problem, targets, tol=1e-9, max_iter=40)
    history = result["history"]
    assert all(entry["converged"] for entry in history)
    peak = max(entry["reaction_sum"] for entry in history)
    lower = specimen.limit_load
    upper = 2.0 / math.sqrt(3.0) * specimen.limit_load
    assert lower * 0.98 <= peak <= upper * 1.0005