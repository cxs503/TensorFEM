import os
import pytest
import torch
from tensorfem.finite_strain_contact3d import (
    balanced_face_load,build_curved_nonmatching_two_block_contact,
    run_curved_finite_strain_friction_qualification,
    solve_finite_strain_friction_adaptive_path,
    solve_finite_strain_friction_path)


def test_real_curved_nonmatching_tet4_friction_level_one_closure():
    model=build_curved_nonmatching_two_block_contact(master_cells=1,slave_cells=2)
    loads=[balanced_face_load(model,normal=p) for p in (5.,10.,15.,20.,25.)]
    loads.extend(balanced_face_load(model,normal=25.,shear=s) for s in (2.,8.,16.))
    steps=solve_finite_strain_friction_path(model,loads,tangential_penalty=2e4,
        friction=.3,tolerance=3e-9)
    result=steps[-1].contact.result
    assert steps[-1].residual_norm<1e-7
    assert torch.min(steps[-1].jacobian)>0
    assert float(result.dissipation_increment)>=0
    assert torch.linalg.vector_norm(result.slave_forces.sum(0)+result.master_forces.sum(0))<1e-9
    assert len(result.state.forward.points)!=len(result.state.reverse.points)
    expected=.5*(result.forward.dissipation_increment+result.reverse.dissipation_increment)
    assert torch.equal(result.dissipation_increment,expected)


def test_curved_nonmatching_generator_scales_both_independent_meshes():
    coarse=build_curved_nonmatching_two_block_contact(master_cells=1,slave_cells=2)
    fine=build_curved_nonmatching_two_block_contact(master_cells=2,slave_cells=3)
    assert fine.solid.n_dofs>coarse.solid.n_dofs
    assert fine.master_faces.shape[0]==4 and fine.slave_faces.shape[0]==9
    assert torch.min(fine.solid.reference_nodes[fine.slave_nodes,2])>0


def test_secant_predictor_crosses_one_sided_stick_slip_branch_to_requested_load():
    model=build_curved_nonmatching_two_block_contact(master_cells=1,slave_cells=2)
    loads=[balanced_face_load(model,normal=p) for p in (5.,10.,15.,20.,25.)]
    loads.extend(balanced_face_load(model,normal=25.,shear=s) for s in (2.,8.,16.))
    steps=solve_finite_strain_friction_adaptive_path(model,loads,
        tangential_penalty=2e4,friction=.3,tolerance=3e-9)
    assert torch.equal(steps[-1].load,loads[-1])
    assert steps[-1].residual_norm<1e-7
    assert torch.min(steps[-1].jacobian)>0
    result=steps[-1].contact.result
    ratio=torch.linalg.vector_norm(result.tangential_resultant)/(
        .3*torch.linalg.vector_norm(result.normal_resultant))
    assert abs(float(ratio)-1)<.03
    assert float(result.dissipation_increment)>0


@pytest.mark.skipif(os.environ.get("TENSORFEM_RUN_SLOW_FINITE_STRAIN_FRICTION")!="1",
    reason="set TENSORFEM_RUN_SLOW_FINITE_STRAIN_FRICTION=1 for the 2/3 path")
def test_two_level_curved_finite_strain_friction_qualification():
    report=run_curved_finite_strain_friction_qualification()
    assert report["passed"] and len(report["mesh_sequence"])==2
    assert max(row["coulomb_relative_error"] for row in report["mesh_sequence"])<.03
    assert report["master_slave_interchange_relative_error"]<.03
    assert len(report["evidence_sha256"])==64
