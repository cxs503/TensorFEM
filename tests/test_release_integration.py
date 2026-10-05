import json

import tensorfem
from tensorfem.cli import main


def test_v0340_public_industrial_api_is_importable():
    assert tensorfem.__version__ == "0.44.0"
    required = {
        "DofManager", "MPC", "assemble_coo", "solve_sparse_static",
        "ModelDB", "solve_adaptive", "save_checkpoint", "load_checkpoint",
        "ThermalModel", "solve_steady_thermal", "solve_transient_thermal",
        "central_difference", "critical_time_step", "Tet4J2Model",
        "solve_load_steps", "Tet10Model", "solve_tet10",
        "Hex20Model", "solve_hex20", "project_point_to_polyline",
        "update_node_polyline_contact",
        "gmres", "solve_multiple_rhs", "PlasticBarContact",
        "coupled_bar_contact_problem",
        "solve_schwarz", "update_node_facet_contact", "diagnose_model",
        "corotational_cylindrical_shell4", "coupled_3d_problem",
        "run_grid_benchmark",
        "solve_tet4_j2_path", "update_surface_contact",
        "consistent_internal_force_tangent",
        "solve_finite_elastic_path", "solve_axisymmetric_hertz",
        "solve_shell_step",
        "solve_finite_plastic_path", "integrate_mortar_contact",
        "solve_pinched_cylinder_linear",
        "integrate_adaptive", "update_frictional_mortar",
        "hemisphere_with_hole",
        "solve_adaptive_global_path", "vertex_triangle_ccd",
        "solve_general_shell",
        "assemble_ad_global", "edge_edge_ccd", "pure_bending_shell",
        "implicit_material_tangent", "solve_multi_contact_impulses",
        "solve_arc_length",
        "assemble_hybrid", "solve_rigid_contact_graph",
        "karatas_yuksel_ring_load_audit",
        "solve_hybrid_global_path", "solve_colored_contacts",
        "general_shell_pressure_arc_problem",
        "taped_material_tangent", "solve_colored_frictional_contacts",
        "execute_island_jobs",
        "run_job", "build_hemisphere_post", "run_case",
        "run_project", "write_result_db", "run_classical_shell_suite",
        "run_mesh_project", "StepExecutor", "write_result_db_v2",
        "MortarContactAssembly", "assemble_mortar_contact",
        "FrictionalMortarAssembly", "assemble_frictional_mortar",
        "FrictionalSurfaceState", "FrictionalSurfaceLoad", "FrictionalSurfaceStep",
        "initial_frictional_surface_state", "solve_frictional_surface_path",
        "run_frictional_surface_path_qualification",
    }
    assert required <= set(tensorfem.__all__)
    assert all(hasattr(tensorfem, name) for name in required)


def test_capabilities_cli_separates_stable_and_experimental(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["tensorfem", "capabilities"])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == "0.44.0"
    assert any("10k-active-DOF" in item for item in payload["stable"])
    assert any("generation-scheduled panel" in item for item in payload["stable"])
    assert any("general curved frictional" in item for item in payload["experimental"])
    assert any("sparse" in item for item in payload["stable"])
    assert any("sparse Shell4" in item for item in payload["stable"])
    assert any("ten-point" in item for item in payload["stable"])
    assert any("energy-conjugate" in item for item in payload["stable"])
    assert any("curved shell" in item for item in payload["experimental"])
    assert any("descending-branch" in item for item in payload["experimental"])
    assert any("4x4 cross-step elastic peak/post-peak and energy" in item
               for item in payload["experimental"])
