import math
import pytest
import torch

from tensorfem.shell4_buckling import (
    navier_plate_load, rectangular_shell_mesh, run_shell4_buckling_qualification,
    shell4_geometric_stiffness, simply_supported_plate_buckling,
)


def test_element_geometric_stiffness_is_symmetric_and_rigid_translation_free():
    xyz = torch.tensor([[0.,0.,0.],[2.,0.,0.],[2.,1.,0.],[0.,1.,0.]], dtype=torch.float64)
    kg = shell4_geometric_stiffness(xyz, torch.tensor([3., 1., .2], dtype=torch.float64))
    assert kg.shape == (24,24)
    assert torch.allclose(kg, kg.T, atol=1e-12)
    rigid = torch.zeros(24, dtype=torch.float64); rigid[2::6] = 1.
    assert torch.linalg.vector_norm(kg @ rigid) < 1e-12


def test_real_shell4_mesh_converges_to_navier_plate_load():
    values=[]
    kw=dict(a=1.,b=1.,young=210e9,poisson=.3,thickness=.01,ny_ratio=0.)
    for n in (4,8,16):
        values.append(float(simply_supported_plate_buckling(nx=n,ny=n,**kw).load_factors[0]))
    exact=navier_plate_load(**kw)
    assert max(abs(value-exact)/exact for value in values) < .03
    assert abs(values[-1]-values[-2])/values[-1] < .03


def test_biaxial_square_is_half_uniaxial_load():
    common=dict(a=1.,b=1.,nx=12,ny=12,young=70e9,poisson=.3,thickness=.008)
    uni=simply_supported_plate_buckling(**common,ny_ratio=0.).load_factors[0]
    bi=simply_supported_plate_buckling(**common,ny_ratio=1.).load_factors[0]
    assert float(abs(bi/uni-.5)) < .01


def test_qualification_report_passes_three_percent_gate():
    report=run_shell4_buckling_qualification()
    assert report["passed"] is True
    assert report["meshes"][-1]["relative_error"] < report["tolerance"]


@pytest.mark.parametrize("kwargs",[
    {"a":0.}, {"young":float("nan")}, {"poisson":.5}, {"thickness":-1.},
    {"nx":1}, {"ny_ratio":-1.}, {"modes":0},
])
def test_plate_solver_rejects_invalid_input(kwargs):
    base=dict(a=1.,b=1.,nx=4,ny=4,young=1e7,poisson=.3,thickness=.01)
    base.update(kwargs)
    with pytest.raises((ValueError,TypeError)):
        simply_supported_plate_buckling(**base)


def test_mesh_rejects_noninteger_resolution():
    with pytest.raises(ValueError): rectangular_shell_mesh(1.,1.,2.5,4)
