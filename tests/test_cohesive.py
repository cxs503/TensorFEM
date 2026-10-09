import torch

from tensorfem.cohesive import BilinearCohesiveLaw, two_node_interface_response


D = torch.float64


def law():
    # d0=0.01, df=0.1 and triangular area Gc=0.5.
    return BilinearCohesiveLaw(stiffness=1000.0, peak_traction=10.0, fracture_energy=0.5)


def test_peak_and_failure_openings_match_closed_form():
    model = law()
    peak = model.update(torch.tensor(model.onset_opening, dtype=D))
    failed = model.update(torch.tensor(model.failure_opening, dtype=D), peak.state)
    assert abs(float(peak.traction) - 10.0) / 10.0 < 1e-14
    assert abs(model.failure_opening - 0.1) / 0.1 < 1e-14
    assert abs(float(failed.traction)) < 1e-14
    assert abs(float(failed.damage) - 1.0) < 1e-14


def test_monotonic_envelope_integrates_to_fracture_energy():
    model = law()
    openings = torch.linspace(0.0, model.failure_opening, 10001, dtype=D)
    energy = torch.trapezoid(model.envelope_traction(openings), openings)
    relative_error = abs(float(energy) - model.fracture_energy) / model.fracture_energy
    assert relative_error < 1e-8


def test_unloading_is_secant_and_damage_is_irreversible():
    model = law()
    loaded = model.update(torch.tensor(0.055, dtype=D))
    unloaded = model.update(torch.tensor(0.022, dtype=D), loaded.state)
    reloaded = model.update(torch.tensor(0.044, dtype=D), unloaded.state)
    secant = float(loaded.traction) / 0.055
    assert abs(float(unloaded.traction) - secant * 0.022) < 1e-12
    assert abs(float(reloaded.traction) - secant * 0.044) < 1e-12
    assert float(unloaded.damage) == float(loaded.damage) == float(reloaded.damage)
    assert float(unloaded.state.maximum_opening) == 0.055


def test_compression_does_not_grow_tensile_damage():
    model = law()
    loaded = model.update(torch.tensor(0.055, dtype=D))
    compressed = model.update(torch.tensor(-0.003, dtype=D), loaded.state)
    assert abs(float(compressed.traction) + 3.0) < 1e-14
    assert float(compressed.state.maximum_opening) == float(loaded.state.maximum_opening)


def test_two_node_interface_equilibrium_and_tangent():
    model = law()
    u = torch.tensor([0.002, 0.012], dtype=D)
    force, tangent, _ = two_node_interface_response(u, 2.5, model)
    assert abs(float(force.sum())) < 1e-14
    assert torch.allclose(force, torch.tensor([-25.0, 25.0], dtype=D))
    assert torch.allclose(tangent, 2500.0 * torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=D))


def test_parameter_validation_rejects_nonphysical_softening():
    try:
        BilinearCohesiveLaw(1000.0, 10.0, 0.04)  # df=.008 < d0=.01
    except ValueError:
        pass
    else:
        raise AssertionError("nonphysical parameters must be rejected")
