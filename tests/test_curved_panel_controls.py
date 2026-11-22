import math

import pytest

from tensorfem.curved_panel_contact import CurvedPanelContact


def panel(young, count):
    return CurvedPanelContact(
        9, count, side=1.0, radius=3.0, thickness=0.02, young=young, poisson=0.3, stiffness=1.0e9
    )


def test_rigid_paraboloid_distributed_load_limit():
    depth = 0.004
    step = panel(1e17, 160).solve(depth)
    force = math.pi * 1.0e9 * 3 * depth**2
    assert step.force_N == pytest.approx(force, rel=0.001)
    assert step.contact_area_m2 == pytest.approx(2 * math.pi * 3 * depth, rel=0.015)
    assert step.contact_energy_J == pytest.approx(force * depth / 3, rel=0.001)


def test_reaction_and_work_at_finite_panel_compliance():
    model = panel(2.1e11, 64)
    d, h = 0.004, 1e-7
    step = model.solve(d)
    assert step.relative_residual < 1e-9
    assert step.support_N == pytest.approx(step.force_N, rel=1e-8)

    def energy(depth):
        state = model.solve(depth)
        return state.elastic_energy_J + state.contact_energy_J

    assert (energy(d + h) - energy(d - h)) / (2 * h) == pytest.approx(step.force_N, rel=1e-6)
    unloaded = model.solve(-d)
    assert unloaded.force_N == unloaded.contact_area_m2 == unloaded.elastic_energy_J == 0.0
