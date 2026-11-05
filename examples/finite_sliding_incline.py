"""Displacement-driven inclined-plane finite-sliding contact example."""

import math

import torch

from tensorfem.finite_sliding_contact import initial_friction_state, update_node_polyline_contact

angle = math.radians(30.0)
t = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
n = torch.tensor([-math.sin(angle), math.cos(angle)], dtype=torch.float64)
surface = torch.stack((-2 * t, 2 * t))
point = 0.2 * t + 0.001 * n
state = initial_friction_state(point, surface)
point = 0.2 * t - 0.002 * n
for increment in (0.001, 0.004, 0.010):
    result = update_node_polyline_contact(
        point,
        surface,
        state,
        normal_penalty=2e5,
        tangential_penalty=4e4,
        friction=0.25,
        relative_tangential_increment=increment,
    )
    state = result.state
    print(
        result.normal_traction.item(),
        result.tangential_traction.item(),
        "stick" if state.sticking else "slip",
    )
