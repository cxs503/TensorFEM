import torch
from tensorfem.frictional_surface_path import run_frictional_surface_path_qualification
from tensorfem.surface_surface_contact3d import build_parabolic_surface_model


def test_nonmatching_curved_double_deformable_stick_slip_path():
    model=build_parabolic_surface_model(cells=2,master_cells=3,radius=4.,
        clearance=.005,foundation=2.e4,normal_penalty=1.e6)
    report=run_frictional_surface_path_qualification(model)
    assert report["passed"] and report["stick_history"]==[True,True,False]
    assert report["coulomb_relative_error"]<.03
    assert report["force_imbalance"]<1e-9
    assert report["equilibrium_moment_residual"]<1e-7
    assert report["rollback_exact"]
    assert report["general_surface_to_surface"]=="blocked"


def test_all_committed_path_history_tensors_are_detached():
    from tensorfem.frictional_surface_path import (
        FrictionalSurfaceLoad, solve_frictional_surface_path)
    model=build_parabolic_surface_model(cells=1,master_cells=2,radius=4.,
        clearance=.005,foundation=2.e4,normal_penalty=1.e6)
    steps=solve_frictional_surface_path(model,[FrictionalSurfaceLoad(250.,0.),
        FrictionalSurfaceLoad(250.,80.)],tangential_penalty=5e4,friction=.3)
    for step in steps:
        tensors=[step.state.slave_displacement,step.state.master_displacement]
        for point in step.state.contact.points:
            tensors.extend([point.elastic_slip,point.dissipated_energy_density,
                point.projection,point.slave_point,point.master_weights])
        assert all(value is not None and not value.requires_grad for value in tensors)
        assert all(value.grad_fn is None for value in tensors)
