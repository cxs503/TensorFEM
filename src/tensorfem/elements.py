"""Element formulations."""
import torch


def truss2d_stiffness(
    coordinates: torch.Tensor, young_modulus: torch.Tensor, area: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return global 4x4 stiffness, length and direction for 2D truss elements."""
    delta = coordinates[:, 1] - coordinates[:, 0]
    length = torch.linalg.vector_norm(delta, dim=1)
    if torch.any(length <= torch.finfo(coordinates.dtype).eps):
        raise ValueError("zero-length truss element")
    direction = delta / length[:, None]
    c, s = direction[:, 0], direction[:, 1]
    pattern = torch.stack(
        (
            c*c, c*s, -c*c, -c*s,
            c*s, s*s, -c*s, -s*s,
            -c*c, -c*s, c*c, c*s,
            -c*s, -s*s, c*s, s*s,
        ), dim=1
    ).reshape(-1, 4, 4)
    stiffness = (young_modulus * area / length)[:, None, None] * pattern
    return stiffness, length, direction


def truss2d_dofs(elements: torch.Tensor) -> torch.Tensor:
    n1, n2 = elements[:, 0], elements[:, 1]
    return torch.stack((2*n1, 2*n1+1, 2*n2, 2*n2+1), dim=1)
