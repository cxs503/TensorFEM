import pytest
from tensorfem.surface_surface_contact3d import (
    build_surface_patch_model, initial_surface_patch_state,
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
