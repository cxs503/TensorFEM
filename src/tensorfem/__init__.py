"""TensorFEM public API."""
from .model import TrussModel
from .solvers import StaticResult, solve_linear_static
from .frame2d import FrameModel, FrameResult, solve_frame_static
from .continuum import ContinuumModel, ContinuumResult, solve_continuum
from .modal import ModalResult, cantilever_beam_modes, solve_modes
from .buckling import BucklingResult, pinned_pinned_column_buckling, solve_linear_buckling
from .benchmark_registry import BenchmarkEvidence, run_registered_benchmarks, verification_report

__version__ = "0.3.0"
__all__ = [
    "TrussModel", "StaticResult", "solve_linear_static",
    "FrameModel", "FrameResult", "solve_frame_static",
    "ContinuumModel", "ContinuumResult", "solve_continuum",
    "ModalResult", "cantilever_beam_modes", "solve_modes",
    "BucklingResult", "pinned_pinned_column_buckling", "solve_linear_buckling",
    "BenchmarkEvidence", "run_registered_benchmarks", "verification_report",
]
