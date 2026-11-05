"""Low-order surface contact and analytical Hertz traction integration."""

import torch
from tensorfem.surface_contact3d import (
    hertz_sphere_halfspace_reference,
    initial_surface_contact_state,
    integrate_hertz_pressure,
    update_surface_contact,
)

D = torch.float64
slave = torch.tensor(
    [[0.0, 0.0, 0.01], [1.0, 0.0, 0.01], [1.0, 1.0, 0.01], [0.0, 1.0, 0.01]], dtype=D
)
faces = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
master = torch.tensor(
    [[-1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [2.0, 2.0, 0.0], [-1.0, 2.0, 0.0]], dtype=D
)
state = initial_surface_contact_state(slave, faces, master, faces)
slave[:, 2] = -0.002
result = update_surface_contact(
    slave,
    faces,
    master,
    faces,
    state,
    normal_penalty=2e5,
    tangential_penalty=1e4,
    friction=0.0,
    relative_increments=torch.zeros_like(slave),
)
print("surface resultant:", result.slave_forces.sum(0))
print("force imbalance:", result.slave_forces.sum(0) + result.master_forces.sum(0))

load = 1000.0
ref = hertz_sphere_halfspace_reference(load, 0.05, 210e9, 0.3, 210e9, 0.3)
integrated = integrate_hertz_pressure(ref, radial_cells=64)
print("Hertz radius:", ref.contact_radius)
print("integrated/reference load:", integrated, load)
print("quadrature relative error:", abs(integrated - load) / load)
