import pytest
import torch

from tensorfem.model import TrussModel


def test_rejects_zero_length_element_during_solution():
    from tensorfem.solvers import solve_linear_static
    model = TrussModel(torch.zeros((2, 2), dtype=torch.float64), torch.tensor([[0, 1]]),
                       torch.tensor(1.), torch.tensor(1.), torch.zeros(4), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="zero-length"):
        solve_linear_static(model)
