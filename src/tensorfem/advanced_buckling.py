"""Classical Euler-column boundary-condition benchmarks."""
import math
from .buckling import BucklingResult, solve_linear_buckling, uniform_column_matrices

EFFECTIVE_LENGTH = {"pinned-pinned": 1.0, "fixed-free": 2.0,
                    "fixed-pinned": 0.6991556596428412, "fixed-fixed": 0.5}

def column_buckling(length: float, elements: int, EI: float, boundary: str) -> BucklingResult:
    if boundary not in EFFECTIVE_LENGTH:
        raise ValueError("unsupported boundary")
    K, Kg = uniform_column_matrices(length, elements, EI)
    end_w, end_r = 2*elements, 2*elements+1
    fixed = {"pinned-pinned": (0, end_w), "fixed-free": (0, 1),
             "fixed-pinned": (0, 1, end_w), "fixed-fixed": (0, 1, end_w, end_r)}[boundary]
    return solve_linear_buckling(K, Kg, fixed, 1)

def exact_critical_load(length: float, EI: float, boundary: str) -> float:
    if boundary not in EFFECTIVE_LENGTH:
        raise ValueError("unsupported boundary")
    return math.pi**2*EI/(EFFECTIVE_LENGTH[boundary]*length)**2
