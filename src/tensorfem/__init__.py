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
from .global_plasticity import GlobalPlasticResult, solve_tet4_j2_path
from .surface_contact3d import (
    SurfaceContactState, SurfaceContactUpdate, HertzReference,
    tributary_areas, initial_surface_contact_state, update_surface_contact,
    hertz_sphere_halfspace_reference, integrate_hertz_pressure,
)
from .shell_consistent import (
    ShellTangentResult, rotation_matrix_from_vector, shell_energy,
    consistent_internal_force_tangent,
)
from .finite_strain_elasticity import (
    FiniteStrainTet4Model, FiniteElasticResult, neo_hookean_response,
    assemble_finite_tet4, solve_finite_elastic_path,
)
from .hertz_axisymmetric import (
    AxisymmetricHertzResult, hertz_reference, axisymmetric_mesh,
    solve_axisymmetric_hertz,
)
from .shell_nonlinear_step import (
    CylindricalShellMesh, ShellIncrement, ShellStepResult,
    assemble_shell, solve_shell_step, write_shell_checkpoint,
    read_shell_checkpoint,
)
from .finite_strain_plasticity import (
    MultiplicativeJ2State, FinitePlasticState, FinitePlasticResult,
    update_multiplicative_j2, assemble_finite_plastic,
    solve_finite_plastic_path,
)
from .mortar_contact3d import (
    MortarContactResult, integrate_mortar_contact, self_contact_candidates,
)
from .pinched_cylinder import (
    PinchedCylinderResult, pinched_cylinder_model,
    solve_pinched_cylinder_linear, solve_pinched_cylinder_nonlinear,
)
from .nonproportional_plasticity import (
    AdaptiveFiniteJ2State, AdaptiveUpdate, virgin_adaptive_state,
    integrate_adaptive, integrate_path, richardson_algorithmic_tangent,
    fixed_substep_reference,
)
from .frictional_mortar import (
    MortarPointState, FrictionalMortarState, FrictionalMortarResult,
    SelfContactResult, initial_frictional_mortar_state,
    update_frictional_mortar, update_self_contact,
)
from .spherical_shell import (
    HemisphereResult, projected_shell4_stiffness, hemisphere_with_hole,
)
from .adaptive_global_plasticity import (
    AdaptiveGlobalState, AdaptiveGlobalResult,
    assemble_adaptive_global, solve_adaptive_global_path,
)
from .self_contact_ccd import (
    CCDEvent, DynamicPairHistory, DynamicSelfContactState,
    DynamicSelfContactUpdate, vertex_triangle_ccd,
    swept_vertex_face_candidates, earliest_self_contact,
    update_dynamic_self_contact,
)
from .general_shell_nonlinear import (
    GeneralShellMesh, GeneralShellResult, general_shell_energy,
    general_shell_force_tangent, assemble_general_shell, solve_general_shell,
)
from .ad_plastic_tangent import ADMetrics, assemble_ad_global, benchmark_tangents
from .dynamic_contact_extended import (
    AmbiguousCoplanarContactError, EdgeEdgeEvent, UnifiedEvent, ImpactResult,
    edge_edge_ccd, spatial_hash_edge_pairs, earliest_edge_contacts,
    earliest_unified_contacts, coulomb_impact,
)
from .large_rotation_shell_benchmark import LargeRotationResult, pure_bending_shell
from .implicit_plastic_tangent import (
    smooth_spd_log, implicit_update, implicit_material_tangent,
    tet4_implicit_response,
)
from .multi_contact_impulse import (
    ContactConstraint, ContactImpulseHistory, MultiContactState,
    MultiContactResult, solve_multi_contact_impulses,
    build_vertex_face_manifold, constraints_from_unified_events,
)
from .arc_length import (
    ArcLengthPoint, ArcLengthResult, ArcLengthProblem,
    general_shell_arc_problem, solve_arc_length,
)
from .von_mises_arch import (
    ArchReference, arch_load, arch_limit_reference, trace_von_mises_arch,
)
from .hybrid_plastic_tangent import HybridMetrics, assemble_hybrid
from .rigid_contact_graph import (
    RigidContact, RigidImpulseHistory, RigidContactState, RigidContactResult,
    contact_islands, solve_rigid_contact_graph,
)
from .snapthrough_shell_qualification import (
    QualificationAudit, karatas_yuksel_ring_load_audit,
)
from .hybrid_global_solver import HybridGlobalResult, solve_hybrid_global_path
from .colored_contact_graph import (
    ColoredContactData, ColoredContactState, ColoredContactResult,
    prepare_colored_contacts, solve_colored_contacts,
)
from .follower_shell_load import (
    follower_pressure_force, follower_pressure_force_tangent,
    assemble_follower_pressure, general_shell_pressure_arc_problem,
)

__version__ = "0.15.0"
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
    "GlobalPlasticResult", "solve_tet4_j2_path",
    "SurfaceContactState", "SurfaceContactUpdate", "HertzReference",
    "tributary_areas", "initial_surface_contact_state", "update_surface_contact",
    "hertz_sphere_halfspace_reference", "integrate_hertz_pressure",
    "ShellTangentResult", "rotation_matrix_from_vector", "shell_energy",
    "consistent_internal_force_tangent",
    "FiniteStrainTet4Model", "FiniteElasticResult", "neo_hookean_response",
    "assemble_finite_tet4", "solve_finite_elastic_path",
    "AxisymmetricHertzResult", "hertz_reference", "axisymmetric_mesh",
    "solve_axisymmetric_hertz",
    "CylindricalShellMesh", "ShellIncrement", "ShellStepResult",
    "assemble_shell", "solve_shell_step", "write_shell_checkpoint",
    "read_shell_checkpoint",
    "MultiplicativeJ2State", "FinitePlasticState", "FinitePlasticResult",
    "update_multiplicative_j2", "assemble_finite_plastic",
    "solve_finite_plastic_path",
    "MortarContactResult", "integrate_mortar_contact",
    "self_contact_candidates",
    "PinchedCylinderResult", "pinched_cylinder_model",
    "solve_pinched_cylinder_linear", "solve_pinched_cylinder_nonlinear",
    "AdaptiveFiniteJ2State", "AdaptiveUpdate", "virgin_adaptive_state",
    "integrate_adaptive", "integrate_path", "richardson_algorithmic_tangent",
    "fixed_substep_reference",
    "MortarPointState", "FrictionalMortarState", "FrictionalMortarResult",
    "SelfContactResult", "initial_frictional_mortar_state",
    "update_frictional_mortar", "update_self_contact",
    "HemisphereResult", "projected_shell4_stiffness", "hemisphere_with_hole",
    "AdaptiveGlobalState", "AdaptiveGlobalResult",
    "assemble_adaptive_global", "solve_adaptive_global_path",
    "CCDEvent", "DynamicPairHistory", "DynamicSelfContactState",
    "DynamicSelfContactUpdate", "vertex_triangle_ccd",
    "swept_vertex_face_candidates", "earliest_self_contact",
    "update_dynamic_self_contact",
    "GeneralShellMesh", "GeneralShellResult", "general_shell_energy",
    "general_shell_force_tangent", "assemble_general_shell",
    "solve_general_shell",
    "ADMetrics", "assemble_ad_global", "benchmark_tangents",
    "AmbiguousCoplanarContactError", "EdgeEdgeEvent", "UnifiedEvent",
    "ImpactResult", "edge_edge_ccd", "spatial_hash_edge_pairs",
    "earliest_edge_contacts", "earliest_unified_contacts", "coulomb_impact",
    "LargeRotationResult", "pure_bending_shell",
    "smooth_spd_log", "implicit_update", "implicit_material_tangent",
    "tet4_implicit_response",
    "ContactConstraint", "ContactImpulseHistory", "MultiContactState",
    "MultiContactResult", "solve_multi_contact_impulses",
    "build_vertex_face_manifold", "constraints_from_unified_events",
    "ArcLengthPoint", "ArcLengthResult", "ArcLengthProblem",
    "general_shell_arc_problem", "solve_arc_length",
    "ArchReference", "arch_load", "arch_limit_reference",
    "trace_von_mises_arch",
    "HybridMetrics", "assemble_hybrid",
    "RigidContact", "RigidImpulseHistory", "RigidContactState",
    "RigidContactResult", "contact_islands", "solve_rigid_contact_graph",
    "QualificationAudit", "karatas_yuksel_ring_load_audit",
    "HybridGlobalResult", "solve_hybrid_global_path",
    "ColoredContactData", "ColoredContactState", "ColoredContactResult",
    "prepare_colored_contacts", "solve_colored_contacts",
    "follower_pressure_force", "follower_pressure_force_tangent",
    "assemble_follower_pressure", "general_shell_pressure_arc_problem",
]
