import json

import tensorfem
from tensorfem.cli import main


def test_v070_public_industrial_api_is_importable():
    assert tensorfem.__version__ == "0.7.0"
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
    }
    assert required <= set(tensorfem.__all__)
    assert all(hasattr(tensorfem, name) for name in required)


def test_capabilities_cli_separates_stable_and_experimental(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["tensorfem", "capabilities"])
    main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["version"] == "0.7.0"
    assert any("sparse" in item for item in payload["stable"])
    assert any("curved shell" in item for item in payload["experimental"])
