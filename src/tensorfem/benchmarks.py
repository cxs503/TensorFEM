"""Small deterministic verification cases with analytical references."""
from dataclasses import dataclass

import torch

from .model import TrussModel
from .solvers import StaticResult, solve_linear_static


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    computed: float
    reference: float
    relative_error: float
    passed: bool


def single_bar(dtype: torch.dtype = torch.float64) -> tuple[TrussModel, float]:
    length, young, area, load = 2.0, 210e9, 3e-4, 12000.0
    model = TrussModel(
        nodes=torch.tensor([[0., 0.], [length, 0.]], dtype=dtype),
        elements=torch.tensor([[0, 1]], dtype=torch.long),
        young_modulus=torch.tensor(young, dtype=dtype),
        area=torch.tensor(area, dtype=dtype),
        forces=torch.tensor([0., 0., load, 0.], dtype=dtype),
        fixed_dofs=torch.tensor([0, 1, 3], dtype=torch.long),
    )
    return model, load * length / (young * area)


def run_single_bar(tolerance: float = 1e-10) -> BenchmarkResult:
    model, reference = single_bar()
    result = solve_linear_static(model)
    computed = float(result.displacement[2])
    error = abs(computed-reference) / abs(reference)
    return BenchmarkResult("single_bar_tension", computed, reference, error, error <= tolerance)


def three_bar_truss(dtype: torch.dtype = torch.float64) -> TrussModel:
    return TrussModel(
        nodes=torch.tensor([[0., 0.], [2., 0.], [1., 1.]], dtype=dtype),
        elements=torch.tensor([[0, 2], [1, 2], [0, 1]], dtype=torch.long),
        young_modulus=torch.tensor(200e9, dtype=dtype),
        area=torch.tensor(5e-4, dtype=dtype),
        forces=torch.tensor([0., 0., 0., 0., 20000., -80000.], dtype=dtype),
        fixed_dofs=torch.tensor([0, 1, 2, 3], dtype=torch.long),
    )


def solve_three_bar() -> StaticResult:
    return solve_linear_static(three_bar_truss())
