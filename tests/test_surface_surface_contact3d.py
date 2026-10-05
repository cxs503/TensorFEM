import os
import pytest
from tensorfem.surface_surface_contact3d import (
    build_surface_patch_model, initial_surface_patch_state,
    run_curved_surface_contact_qualification,
    run_curved_surface_contact_smoke,
    run_nonmatching_surface_contact_qualification,
    run_surface_surface_contact_qualification, solve_surface_patch_path)


def test_multi_point_surface_contact_qualification():
    r=run_surface_surface_contact_qualification()
    assert r["passed"] and len(r["mesh_sequence"])==3
    assert max(x["relative_error"] for x in r["mesh_sequence"])<.03
    errors=[x["relative_error"] for x in r["mesh_sequence"]]
    assert all(a>b for a,b in zip(errors,errors[1:]))
    assert max(x["force_imbalance"] for x in r["mesh_sequence"])<1e-9
    assert max(x["moment_imbalance"] for x in r["mesh_sequence"])<1e-9
    assert r["objectivity_relative_error"]<1e-10
    assert r["master_slave_exchange_relative_error"]<.03
    assert r["rollback_exact"] is True
    assert "not curved Hertz" in r["scope"]


def test_failed_surface_increment_does_not_mutate_state():
    m=build_surface_patch_model(cells=1); state=initial_surface_patch_state(m)
    with pytest.raises(RuntimeError,match="committed state unchanged"):
        solve_surface_patch_path(m,[180.],initial_state=state,max_iterations=1)
    assert state.slave_displacement.count_nonzero()==0


def test_nonmatching_complete_newton_paths_remain_fail_closed_general_scope():
    r=run_nonmatching_surface_contact_qualification()
    errors=[row["relative_error"] for row in r["mesh_sequence"]]
    assert errors[-1]<.03 and all(a>b for a,b in zip(errors,errors[1:]))
    assert r["complete_newton_role_exchange_relative_error"]<.03
    assert r["objectivity_relative_error"]<1e-10 and r["rollback_exact"]
    assert r["general_surface_to_surface"]=="blocked"


def test_curved_nonmatching_complete_newton_qualification():
    r=run_curved_surface_contact_qualification()
    errors=[row["relative_error"] for row in r["mesh_sequence"]]
    assert errors[-1]<.03 and errors[-1]<errors[0]
    assert r["objectivity_relative_error"]<1e-10
    assert r["complete_newton_role_exchange_relative_error"]<.03
    assert r["rollback_exact"]
    assert r["general_surface_to_surface"]=="qualified_curved_frictionless_subset"
    assert "not classical Hertz" in r["scope"]


test_curved_nonmatching_complete_newton_qualification = pytest.mark.skipif(
    os.environ.get("TENSORFEM_RUN_SLOW_CURVED_CONTACT") != "1",
    reason="set TENSORFEM_RUN_SLOW_CURVED_CONTACT=1 for the 4/5 Hessian qualification",
)(test_curved_nonmatching_complete_newton_qualification)


def test_curved_two_pass_default_smoke_gate():
    r=run_curved_surface_contact_smoke()
    assert r["passed"] and r["two_pass"]
    assert r["active_quadrature_points"]>=2
    assert r["force_imbalance"]<1e-9 and r["rollback_exact"]
