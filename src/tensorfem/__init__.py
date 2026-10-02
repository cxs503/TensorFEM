"""TensorFEM public API."""
from .model import TrussModel
from .solvers import StaticResult, solve_linear_static
from .frame2d import FrameModel, FrameResult, solve_frame_static
from .continuum import ContinuumModel, ContinuumResult, solve_continuum
from .modal import ModalResult, cantilever_beam_modes, solve_modes
from .buckling import BucklingResult, pinned_pinned_column_buckling, solve_linear_buckling
from .benchmark_registry import BenchmarkEvidence, run_registered_benchmarks, verification_report
from .sparse_core import (
    DofManager, MPC, IterativeResult, SparseStaticResult, assemble_coo,
    conjugate_gradient, solve_sparse_static,
)
from .modeldb import ModelDB, ElementBlock, Material, Section
from .nonlinear_step import (
    NonlinearConvergenceError, StepState, solve_adaptive,
    save_checkpoint, load_checkpoint,
)
from .thermal import (
    ThermalModel, ThermalTransientResult, ThermoelasticBarResult,
    solve_steady_thermal, solve_transient_thermal, solve_thermoelastic_bar,
)
from .explicit_dynamics import ExplicitResult, central_difference, critical_time_step
from .solid_plasticity import Tet4J2Model, PlasticSolidState, solve_load_steps
from .quadratic_solid import Tet10Model, solve_tet10
from .hex20 import Hex20Model, solve_hex20
from .finite_sliding_contact import (
    PolylineProjection, FrictionState, ContactUpdate,
    project_point_to_polyline, update_node_polyline_contact,
)
from .sparse_advanced import gmres, jacobi_inverse, solve_multiple_rhs
from .coupled_nonlinear import PlasticBarContact, coupled_bar_contact_problem
from .domain_decomposition import Subdomain, AdditiveSchwarz, solve_schwarz
from .contact3d import (
    FacetProjection, Contact3DState, Contact3DUpdate,
    project_point_to_facets, update_node_facet_contact, update_contact_nodes,
)
from .mesh_pipeline import MeshReport, diagnose_model, read_gmsh, convert_model
from .corotational_shell import (
    CorotationalState, axis_angle, corotational_cylindrical_shell4,
    cylindrical_nodes,
)
from .coupled_3d_demo import (
    Coupled3DModel, Coupled3DState, Coupled3DResponse,
    default_coupled_3d_model, coupled_3d_problem, coupled_3d_response,
    monotonic_reference_displacement,
)
from .structural_grid_benchmark import (
    GridBenchmarkResult, grid_spring_stiffness, analytical_mode,
    manufactured_solution, run_grid_benchmark,
)

__version__ = "0.6.0"
__all__ = [
    "TrussModel", "StaticResult", "solve_linear_static",
    "FrameModel", "FrameResult", "solve_frame_static",
    "ContinuumModel", "ContinuumResult", "solve_continuum",
    "ModalResult", "cantilever_beam_modes", "solve_modes",
    "BucklingResult", "pinned_pinned_column_buckling", "solve_linear_buckling",
    "BenchmarkEvidence", "run_registered_benchmarks", "verification_report",
    "DofManager", "MPC", "IterativeResult", "SparseStaticResult",
    "assemble_coo", "conjugate_gradient", "solve_sparse_static",
    "ModelDB", "ElementBlock", "Material", "Section",
    "NonlinearConvergenceError", "StepState", "solve_adaptive",
    "save_checkpoint", "load_checkpoint",
    "ThermalModel", "ThermalTransientResult", "ThermoelasticBarResult",
    "solve_steady_thermal", "solve_transient_thermal", "solve_thermoelastic_bar",
    "ExplicitResult", "central_difference", "critical_time_step",
    "Tet4J2Model", "PlasticSolidState", "solve_load_steps",
    "Tet10Model", "solve_tet10",
    "Hex20Model", "solve_hex20",
    "PolylineProjection", "FrictionState", "ContactUpdate",
    "project_point_to_polyline", "update_node_polyline_contact",
    "gmres", "jacobi_inverse", "solve_multiple_rhs",
    "PlasticBarContact", "coupled_bar_contact_problem",
    "Subdomain", "AdditiveSchwarz", "solve_schwarz",
    "FacetProjection", "Contact3DState", "Contact3DUpdate",
    "project_point_to_facets", "update_node_facet_contact", "update_contact_nodes",
    "MeshReport", "diagnose_model", "read_gmsh", "convert_model",
    "CorotationalState", "axis_angle", "corotational_cylindrical_shell4",
    "cylindrical_nodes",
    "Coupled3DModel", "Coupled3DState", "Coupled3DResponse",
    "default_coupled_3d_model", "coupled_3d_problem", "coupled_3d_response",
    "monotonic_reference_displacement",
    "GridBenchmarkResult", "grid_spring_stiffness", "analytical_mode",
    "manufactured_solution", "run_grid_benchmark",
]
