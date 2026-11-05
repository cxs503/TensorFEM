"""Low-order deformable TRI3 contact in a global FE residual."""

import torch

from tensorfem.global_contact3d import (
    GlobalSurfaceContactModel,
    NodeTrianglePair,
    solve_global_surface_contact_path,
)


dtype = torch.float64
nodes = torch.tensor(
    [
        [-2.0, -2.0, 0.0],
        [3.0, -2.0, 0.0],
        [3.0, 3.0, 0.0],
        [-2.0, 3.0, 0.0],
        [0.2, 0.4, 0.05],
    ],
    dtype=dtype,
)
stiffness = torch.zeros((15, 15), dtype=dtype)
stiffness[-3:, -3:] = 1000.0 * torch.eye(3, dtype=dtype)
model = GlobalSurfaceContactModel(
    nodes,
    stiffness,
    (NodeTrianglePair(4, ((0, 1, 2), (0, 2, 3))),),
    tuple(range(12)),
    normal_penalty=1.0e4,
    tangential_penalty=1.0e3,
    friction=0.2,
)
loads = []
for horizontal in (0.0, 300.0, 800.0, 1400.0):
    force = torch.zeros(15, dtype=dtype)
    force[-3:] = torch.tensor([horizontal, 0.0, -100.0], dtype=dtype)
    loads.append(force)

steps = solve_global_surface_contact_path(model, loads)
last = steps[-1]
print(
    {
        "slave_displacement": last.displacement[-3:].tolist(),
        "active_master_face": last.state.histories[0].face,
        "sticking": last.state.histories[0].sticking,
        "dissipated_energy": float(last.state.histories[0].dissipated_energy),
        "residual_norm": last.residual_norm,
    }
)
