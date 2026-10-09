import math
import pytest
import torch

from tensorfem.marine_hotspot_fracture import (
    finite_width_edge_factor, hot_spot_extrapolate, mode_i_fracture_gate,
    sample_result_path, linearize_through_thickness,
    run_hotspot_fracture_qualification,
)
from tensorfem.result_db import ResultDB
from tensorfem.result_db_v2 import FieldSpec, ResultDBv2, ResultFrame, ResultStep

D = torch.float64


def line_db(*, element=False):
    x=torch.linspace(0.,.02,5,dtype=D)
    xyz=torch.stack((x,torch.zeros_like(x)),1)
    ids=torch.tensor([11,12,13,14,15])
    conn=torch.tensor([[11,12],[12,13],[13,14],[14,15]])
    # affine structural stress: exact under path interpolation and toe extrapolation
    stress=150e6-2e9*x
    return ResultDB(ids,torch.arange(4),conn,{"coordinates":xyz,"S":stress},
                    {"S":150e6-2e9*(x[:-1]+x[1:])/2} if element else {})


def test_node_path_sampling_and_hotspot_have_independent_affine_oracle():
    path=sample_result_path(line_db(),"S",torch.tensor([0.,0.],dtype=D),
                            torch.tensor([.02,0.],dtype=D),samples=21)
    oracle=150e6-2e9*path.distance_m
    assert torch.max(torch.abs(path.values-oracle))/150e6 < 1e-12
    assert abs(hot_spot_extrapolate(path,.01,method="linear")-150e6)/150e6 < .03
    assert abs(hot_spot_extrapolate(path,.01,method="quadratic")-150e6)/150e6 < .03


def test_integration_point_centroids_and_v2_frame_are_supported():
    db=line_db(element=True)
    path=sample_result_path(db,"S",torch.tensor([.0025,0.],dtype=D),
                            torch.tensor([.0175,0.],dtype=D),samples=7,location="integration_point")
    assert torch.allclose(path.values,150e6-2e9*path.coordinates_m[:,0])
    frame=ResultFrame(3,1.,1.,{"node":{"coordinates":db.node_fields["coordinates"],"S":db.node_fields["S"]}})
    specs={"coordinates":FieldSpec("node","m",("x","y")),"S":FieldSpec("node","Pa",("S11",))}
    v2=ResultDBv2(db.node_ids,db.element_ids,db.connectivity,specs,(ResultStep("Load",(frame,)),))
    p2=sample_result_path(v2,"S",torch.tensor([0.,0.],dtype=D),torch.tensor([.02,0.],dtype=D),
                          samples=5,step="Load",frame=3)
    assert torch.equal(p2.values,db.node_fields["S"])


def test_through_thickness_membrane_bending_linearization_oracle():
    z=torch.linspace(-.01,.01,17,dtype=D)
    stress=80e6+30e6*z/.01
    result=linearize_through_thickness(z,stress)
    assert abs(result.membrane_pa-80e6)/80e6 < .03
    assert abs(result.bending_surface_pa-30e6)/30e6 < .03
    assert abs(result.linearized_inner_pa-50e6)/50e6 < .03
    assert abs(result.linearized_outer_pa-110e6)/110e6 < .03


def test_mode_i_k_j_and_critical_toughness_gate_against_direct_oracle():
    sigma,a,w,e,nu=120e6,.01,.2,210e9,.3
    y=finite_width_edge_factor(a,w)
    result=mode_i_fracture_gate(sigma,a,y,e,20e6,poisson=nu,plane_strain=True)
    k_oracle=(1.12-.231*(a/w)+10.55*(a/w)**2-21.72*(a/w)**3+30.39*(a/w)**4)*sigma*math.sqrt(math.pi*a)
    j_oracle=k_oracle**2*(1-nu**2)/e
    assert abs(result.stress_intensity_pa_sqrt_m-k_oracle)/k_oracle < .03
    assert abs(result.j_integral_j_m2-j_oracle)/j_oracle < .03
    assert result.critical and result.utilization > 1
    safe=mode_i_fracture_gate(sigma,a,y,e,80e6)
    assert not safe.critical


def test_combined_hotspot_fracture_qualification_gate():
    report = run_hotspot_fracture_qualification()
    assert report["passed"] and len(report["evidence"]) == 4
    assert all(row["relative_error"] < row["tolerance"] <= 0.03 for row in report["evidence"])


@pytest.mark.parametrize("call,error",[
    (lambda: sample_result_path(line_db(),"S",torch.tensor([0.,0.]),torch.tensor([0.,0.]),samples=3),ValueError),
    (lambda: sample_result_path(line_db(),"S",torch.tensor([0.,0.]),torch.tensor([.02,.001]),samples=3),ValueError),
    (lambda: sample_result_path(line_db(),"S",torch.tensor([0.,0.]),torch.tensor([.02,0.]),samples=True),TypeError),
    (lambda: mode_i_fracture_gate(True,.1,1.,1.,1.),TypeError),
    (lambda: mode_i_fracture_gate(1.,.1,1.,1.,1.,poisson=.5),ValueError),
    (lambda: finite_width_edge_factor(.7,1.),ValueError),
])
def test_invalid_inputs_fail_closed(call,error):
    with pytest.raises(error):
        call()
