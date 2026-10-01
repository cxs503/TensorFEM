"""TensorFEM public API."""
from .model import TrussModel
from .solvers import StaticResult, solve_linear_static

__version__ = "0.1.0"
__all__ = ["TrussModel", "StaticResult", "solve_linear_static"]
