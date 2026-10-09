"""Displacement-driven inclined-plane finite-sliding contact example."""
import math

import torch

from tensorfem.finite_sliding_contact import initial_friction_state, update_node_polyline_contact

angle = math.radians(30.)
t = torch.tensor([math.cos(angle), math.sin(angle)], dtype=torch.float64)
n = torch.tensor([-math.sin(angle), math.cos(angle)], dtype=torch.float64)
surface = torch.stack((-2*t, 2*t))
point = .2*t + .001*n
state = initial_friction_state(point, surface)
point = .2*t - .002*n
for increment in (.001, .004, .010):
    result = update_node_polyline_contact(
        point, surface, state, normal_penalty=2e5,
        tangential_penalty=4e4, friction=.25,
        relative_tangential_increment=increment,
    )
    state = result.state
    print(result.normal_traction.item(), result.tangential_traction.item(),
          "stick" if state.sticking else "slip")
