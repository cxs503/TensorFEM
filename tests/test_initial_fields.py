import pytest
import torch

from tensorfem.initial_fields import (StructuralInitialFields, import_initial_fields,
    imperfect_strip_inputs, initial_fields_result_db, solve_imported_elastic_strip,
    run_initial_field_transfer_qualification)
from tensorfem.result_db import ResultDB
from tensorfem.result_db_v2 import read_result_db_v2, write_result_db_v2

D=torch.float64

def topology():
    ids=torch.tensor([10,20,30,40]);eids=torch.tensor([7]);conn=torch.tensor([[10,20,30,40]])
    xyz=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.]],dtype=D)
    return ids,eids,conn,xyz

def test_v1_id_mapping_and_nodal_strip_transfer():
    ids,eids,conn,xyz=topology();order=torch.tensor([2,0,3,1])
    imp=torch.zeros((4,3),dtype=D);imp[:,2]=torch.tensor([0.,.1,0.,-.1])
    stress=torch.zeros((4,6),dtype=D);stress[:,0]=torch.tensor([2.,-2.,2.,-2.])
    db=ResultDB(ids[order],eids,conn,{
        "coordinates":xyz[order],"initial_imperfection":imp[order],"residual_stress":stress[order]},
        units={"coordinates":"m","initial_imperfection":"m","residual_stress":"MPa"})
    fields=import_initial_fields(db,target_node_ids=ids,target_element_ids=eids,
        target_connectivity=conn,target_coordinates=xyz)
    assert torch.equal(fields.imperfection,imp);assert torch.equal(fields.residual_stress,stress)
    assert torch.allclose(fields.assert_self_equilibrated(),torch.zeros(6,dtype=D))
    w,s=imperfect_strip_inputs(fields);assert torch.equal(w,imp[:,2]);assert torch.equal(s,stress[:,0])

def test_coordinate_fallback_and_missing_policies():
    ids,eids,conn,xyz=topology();source=ResultDB(ids+100,eids,conn+100,
        {"coordinates":xyz,"initial_imperfection":torch.ones((4,3),dtype=D)},units={"initial_imperfection":"m"})
    mapped=import_initial_fields(source,target_node_ids=ids,target_element_ids=eids,
        target_connectivity=conn,target_coordinates=xyz+1e-10,coordinate_tolerance=1e-8)
    assert torch.equal(mapped.imperfection,torch.ones((4,3),dtype=D))
    with pytest.raises(KeyError):
        import_initial_fields(source,target_node_ids=ids,target_element_ids=eids,
            target_connectivity=conn,target_coordinates=xyz+.1,coordinate_tolerance=1e-8)
    zero=import_initial_fields(source,target_node_ids=ids,target_element_ids=eids,
        target_connectivity=conn,target_coordinates=xyz+.1,missing="zero")
    assert not torch.any(zero.imperfection)

def test_v2_ip_roundtrip_and_component_unit_validation(tmp_path):
    ids,eids,conn,xyz=topology();stress=torch.tensor([[[1.,0,0,0,0,0],[-1.,0,0,0,0,0]]],dtype=D)
    fields=StructuralInitialFields(ids,eids,conn,torch.zeros((4,3),dtype=D),stress,
        "integration_point",{"initial_imperfection":"mm","residual_stress":"MPa"})
    db=initial_fields_result_db(fields,coordinates=xyz)
    in_memory=import_initial_fields(db,target_node_ids=ids,target_element_ids=eids,
        target_connectivity=conn,target_coordinates=xyz,step="Initial",frame=0)
    assert torch.equal(in_memory.residual_stress,stress)
    if not write_result_db_v2(tmp_path/"initial",db):pytest.skip("h5py unavailable")
    loaded=read_result_db_v2(tmp_path/"initial")
    mapped=import_initial_fields(loaded,target_node_ids=ids,target_element_ids=eids,
        target_connectivity=conn,target_coordinates=xyz,step="Initial",frame=0)
    assert mapped.residual_location=="integration_point"
    assert torch.equal(mapped.residual_stress,stress)
    assert torch.equal(mapped.imperfection,fields.imperfection)

def test_validation_and_self_balance_fail_closed():
    ids,eids,conn,xyz=topology();bad=torch.ones((4,6),dtype=D)
    fields=StructuralInitialFields(ids,eids,conn,torch.zeros((4,3),dtype=D),bad,"node")
    with pytest.raises(ValueError,match="not self-equilibrated"):fields.assert_self_equilibrated()
    db=ResultDB(ids,eids,conn,{"initial_imperfection":torch.zeros((4,3),dtype=D)},units={"initial_imperfection":"kelvin"})
    with pytest.raises(ValueError,match="unit"):
        import_initial_fields(db,target_node_ids=ids,target_element_ids=eids,target_connectivity=conn,target_coordinates=xyz)

def test_imported_distribution_enters_actual_strip_solve():
    n=16;ids=torch.arange(100,100+n);eids=torch.tensor([1]);conn=ids[None,:]
    x=torch.arange(n,dtype=D)/n;xyz=torch.stack((x,torch.zeros(n,dtype=D),torch.zeros(n,dtype=D)),1)
    stress=torch.zeros((n,6),dtype=D);stress[:,0]=.1*torch.cos(2*torch.pi*x)
    def solve(profile):
        imp=torch.zeros((n,3),dtype=D);imp[:,2]=profile
        db=ResultDB(ids.flip(0),eids,conn,{"coordinates":xyz.flip(0),
            "initial_imperfection":imp.flip(0),"residual_stress":stress.flip(0)},
            units={"initial_imperfection":"dimensionless","residual_stress":"dimensionless"})
        mapped=import_initial_fields(db,target_node_ids=ids,target_element_ids=eids,
            target_connectivity=conn,target_coordinates=xyz)
        # Mapping order is observable by the exact recovered asymmetric profile.
        assert torch.equal(mapped.imperfection[:,2],profile)
        return solve_imported_elastic_strip(mapped,shortening=.002,young=100.,bending_stiffness=200.)
    sine=solve(.02*torch.sin(2*torch.pi*x))
    harmonic=solve(.02*torch.sin(4*torch.pi*x))
    assert abs(sine.maximum_amplitude-harmonic.maximum_amplitude)>1e-5

def test_coordinate_mapping_rejects_ambiguity_and_reuse():
    ids,eids,conn,xyz=topology();source_xyz=xyz.clone();source_xyz[1]=source_xyz[0]
    db=ResultDB(ids+100,eids,conn+100,{"coordinates":source_xyz,
        "initial_imperfection":torch.ones((4,3),dtype=D)},units={"initial_imperfection":"m"})
    with pytest.raises(ValueError,match="ambiguous"):
        import_initial_fields(db,target_node_ids=ids,target_element_ids=eids,
            target_connectivity=conn,target_coordinates=xyz,missing="nearest")

def test_initial_field_transfer_qualification():
    report=run_initial_field_transfer_qualification()
    assert report["passed"] and report["mapping_max_error"] < 1e-14
    assert report["equilibrium_residual"] < 1e-10
    assert report["field_distribution_response_difference"] > 1e-5
