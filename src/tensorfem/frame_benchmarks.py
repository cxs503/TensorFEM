"""Reference benchmarks for the 2-D frame implementation."""
import torch

from .frame2d import FrameModel, solve_frame_static


def cantilever_tip_load(dtype=torch.float64):
    """One-element cantilever: v=PL^3/(3EI), theta=PL^2/(2EI)."""
    E, A, I, L, P = 210e9, 0.02, 8e-6, 3.0, -10_000.0
    model = FrameModel(
        torch.tensor([[0., 0.], [L, 0.]], dtype=dtype),
        torch.tensor([[0, 1]]), torch.tensor(E, dtype=dtype),
        torch.tensor(A, dtype=dtype), torch.tensor(I, dtype=dtype),
        torch.tensor([0., 0., 0., 0., P, 0.], dtype=dtype), torch.tensor([0, 1, 2]),
    )
    result = solve_frame_static(model)
    expected = torch.tensor([P*L**3/(3*E*I), P*L**2/(2*E*I)], dtype=dtype)
    return result, expected


def cantilever_uniform_load(dtype=torch.float64):
    """One-element cantilever under full-span UDL: v=qL^4/(8EI)."""
    E, A, I, L, q = 200e9, 0.015, 5e-6, 2.5, -4_000.0
    model = FrameModel(
        torch.tensor([[0., 0.], [L, 0.]], dtype=dtype), torch.tensor([[0, 1]]),
        torch.tensor(E, dtype=dtype), torch.tensor(A, dtype=dtype), torch.tensor(I, dtype=dtype),
        torch.zeros(6, dtype=dtype), torch.tensor([0, 1, 2]), torch.tensor(q, dtype=dtype),
    )
    result = solve_frame_static(model)
    expected = torch.tensor([q*L**4/(8*E*I), q*L**3/(6*E*I)], dtype=dtype)
    return result, expected
