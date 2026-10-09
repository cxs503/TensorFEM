import torch

from tensorfem.contact import hertz_sphere_on_halfspace, solve_rigid_plane_contact


DT = torch.float64


def _spring_problem(k=1000.0, force=20.0, clearance=0.01):
    return (torch.tensor([[k]], dtype=DT), torch.tensor([-force], dtype=DT),
            torch.tensor([[1.0]], dtype=DT), torch.tensor([clearance], dtype=DT))


def test_active_set_spring_against_plane_exact_complementarity():
    K, f, C, gap0 = _spring_problem()
    result = solve_rigid_plane_contact(K, f, C, gap0)
    # Independent solution: u=-g0 and lambda=P-k*g0=10 N.
    assert torch.allclose(result.displacement, torch.tensor([-0.01], dtype=DT))
    assert torch.allclose(result.contact_force, torch.tensor([10.0], dtype=DT))
    assert abs(result.gap.item()) < 1e-12
    assert result.contact_force.item() >= 0
    assert abs((result.gap * result.contact_force).item()) < 1e-12
    assert torch.allclose(result.structural_energy, torch.tensor(0.05, dtype=DT))


def test_penalty_spring_reaction_penetration_and_energy_below_three_percent():
    K, f, C, gap0 = _spring_problem()
    kp = 100.0 * K.item()
    result = solve_rigid_plane_contact(K, f, C, gap0, method="penalty", penalty=kp)
    penetration_ref = (20.0 - 1000.0 * 0.01) / (1000.0 + kp)
    reaction_ref = kp * penetration_ref
    energy_ref = 0.5 * kp * penetration_ref**2
    assert abs((-result.gap.item() - penetration_ref) / penetration_ref) < 0.03
    assert abs((result.contact_force.item() - reaction_ref) / reaction_ref) < 0.03
    assert abs((result.contact_energy.item() - energy_ref) / energy_ref) < 0.03
    # Physical rigid-contact reaction is 10 N; penalty regularization error < 3%.
    assert abs(result.contact_force.item() / 10.0 - 1.0) < 0.03


def test_open_contact_has_zero_force():
    K, _, C, gap0 = _spring_problem()
    result = solve_rigid_plane_contact(K, torch.tensor([1.0], dtype=DT), C, gap0)
    assert not result.active.item()
    assert result.gap.item() > 0
    assert result.contact_force.item() == 0


def test_hertz_analytical_reference_internal_identities():
    x = lambda v: torch.tensor(v, dtype=DT)
    ref = hertz_sphere_on_halfspace(x(1000), x(0.01), x(210e9), x(0.3), x(70e9), x(0.33))
    force_from_pressure = 2 * torch.pi * ref.maximum_pressure * ref.contact_radius**2 / 3
    assert torch.allclose(force_from_pressure, x(1000), rtol=1e-12)
    assert torch.allclose(ref.indentation, ref.contact_radius**2 / x(0.01), rtol=1e-12)
