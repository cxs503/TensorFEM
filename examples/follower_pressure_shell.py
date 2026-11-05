"""Audit the configuration-dependent pressure operator on a curved Shell4."""

import torch
from tensorfem.follower_shell_load import follower_pressure_force, follower_pressure_force_tangent

torch.set_default_dtype(torch.float64)
X = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.1], [1.0, 1.0, 0.2], [0.0, 1.0, 0.1]])
q = torch.linspace(-0.02, 0.03, 24)
force, tangent = follower_pressure_force_tangent(X, q, 2.0)
resultant = force.reshape(4, 6)[:, :3].sum(0)
direction = torch.linspace(-1.0, 1.0, 24)
direction /= torch.linalg.vector_norm(direction)
h = 1e-6
fd = (
    follower_pressure_force(X, q + h * direction, 2.0)
    - follower_pressure_force(X, q - h * direction, 2.0)
) / (2 * h)
relative = float(
    torch.linalg.vector_norm(fd - tangent @ direction)
    / torch.linalg.vector_norm(tangent @ direction)
)
print({"resultant": resultant.tolist(), "directional_tangent_error": relative})
