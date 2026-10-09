import importlib.util,json

import pytest
import torch

from tensorfem.plasticity import Plastic1DState,update_bilinear_1d
from tensorfem.result_db import ResultDB
from tensorfem.result_db_v2 import (
    FieldSpec,HistorySeries,ResultDBv2,ResultFrame,ResultStep,migrate_v1,
    query_frame_field,query_history_series,read_result_db_compatible,read_result_db_v2,write_result_db_v2,
)

D=torch.float64


def plastic_db():
    E,sy,H=200000.,250.,10000.;z=torch.tensor(0.,dtype=D);state=Plastic1DState(z,z)
    frames=[];stress=[];alpha=[]
    for i,strain in enumerate((0.,.003)):
        s,_,state=update_bilinear_1d(torch.tensor(strain,dtype=D),E,sy,H,state);stress.append(float(s));alpha.append(float(state.alpha))
        fields={"node":{"displacement":torch.tensor([[0.],[strain]],dtype=D)},
                "element":{"strain":torch.tensor([[strain]],dtype=D)},
                "integration_point":{"stress":s.reshape(1,1,1),"alpha":state.alpha.reshape(1,1,1)}}
        frames.append(ResultFrame(i,float(i),float(i),fields))
    load=ResultStep("Load",tuple(frames));strain=0.;s,_,state=update_bilinear_1d(torch.tensor(strain,dtype=D),E,sy,H,state)
    unload=ResultStep("Unload",(ResultFrame(0,2.,0.,{"node":{"displacement":torch.tensor([[0.],[0.]],dtype=D)},
        "element":{"strain":torch.tensor([[0.]],dtype=D)},"integration_point":{"stress":s.reshape(1,1,1),"alpha":state.alpha.reshape(1,1,1)}}),))
    specs={"displacement":FieldSpec("node","m",("U1",)),"strain":FieldSpec("element","1",("E11",)),
           "stress":FieldSpec("integration_point","Pa",("S11",)),"alpha":FieldSpec("integration_point","1",("PEEQ",))}
    histories={"load_stress":HistorySeries("Load",torch.tensor([0.,1.],dtype=D),torch.tensor([0.,1.],dtype=D),
                    torch.tensor([[stress[0]],[stress[1]]],dtype=D),"Pa",("S11",)),
               "load_alpha":HistorySeries("Load",torch.tensor([0.,1.],dtype=D),torch.tensor([0.,1.],dtype=D),
                    torch.tensor([[alpha[0]],[alpha[1]]],dtype=D),"1",("alpha",)),
               "unload_stress":HistorySeries("Unload",torch.tensor([2.],dtype=D),torch.tensor([0.],dtype=D),
                    torch.tensor([[float(s)]],dtype=D),"Pa",("S11",)),
               "unload_alpha":HistorySeries("Unload",torch.tensor([2.],dtype=D),torch.tensor([0.],dtype=D),
                    torch.tensor([[float(state.alpha)]],dtype=D),"1",("alpha",))}
    return ResultDBv2(torch.tensor([10,20]),torch.tensor([7]),torch.tensor([[10,20]]),specs,(load,unload),histories,
                      {"model":"one_bar"},{"analysis":"plastic load-unload"})


def test_multistep_plastic_history_roundtrip_and_lazy_subsets(tmp_path):
    db=plastic_db();base=tmp_path/"plastic";written=write_result_db_v2(base,db,chunk_rows=1)
    summary=json.loads((tmp_path/"plastic.json").read_text())
    assert summary["schema"]=="tensorfem.result-db.v2" and summary["complete"]
    assert summary["counts"]=={"nodes":2,"elements":1,"steps":2,"frames":3}
    assert "steps/Load/frames/000001/integration_point/stress" in summary["inventory"]
    assert written==(importlib.util.find_spec("h5py") is not None)
    if not written:
        with pytest.raises(RuntimeError,match="h5py"):read_result_db_v2(base)
        return
    loaded=read_result_db_v2(base);assert loaded.checksums()==db.checksums()
    ids,stress=query_frame_field(base,"Load",1,"integration_point","stress",ids=[7],components=["S11"])
    assert ids.tolist()==[7] and torch.equal(stress,db.steps[0].frames[1].fields["integration_point"]["stress"])
    time,load,values=query_history_series(base,"load_alpha",start=1)
    assert time.tolist()==[1.] and load.tolist()==[1.] and values.shape==(1,1)
    assert values[0,0]>0


def test_checksum_tamper_and_incomplete_publication_fail_closed(tmp_path):
    if importlib.util.find_spec("h5py") is None:return
    import h5py
    db=plastic_db();base=tmp_path/"case";write_result_db_v2(base,db)
    summary=json.loads((tmp_path/"case.json").read_text());body=tmp_path/summary["body"]["file"]
    with h5py.File(body,"r+") as h5:h5["histories/load_stress/values"][0,0]+=1
    with pytest.raises(RuntimeError,match="checksum"):read_result_db_v2(base)
    incomplete=tmp_path/"broken.json";summary["complete"]=False;incomplete.write_text(json.dumps(summary))
    with pytest.raises(ValueError,match="incomplete"):read_result_db_v2(tmp_path/"broken")


def test_explicit_v1_migration_preserves_fields_and_histories():
    old=ResultDB(torch.tensor([0,1]),torch.tensor([0]),torch.tensor([[0,1]]),
        {"u":torch.tensor([[0.],[.1]],dtype=D)},{"s":torch.tensor([[2.]],dtype=D)},
        {"energy":torch.tensor([1.,2.],dtype=D)},units={"u":"m","s":"Pa","energy":"J"})
    new=migrate_v1(old)
    assert new.steps[0].name=="Legacy" and len(new.steps[0].frames)==1
    assert torch.equal(new.steps[0].frames[0].fields["node"]["u"],old.node_fields["u"])
    assert torch.equal(new.histories["energy"].values[:,0],old.histories["energy"])
    assert new.field_specs["s"].unit=="Pa"


def test_compatible_reader_keeps_v1_readable_and_can_migrate(tmp_path):
    if importlib.util.find_spec("h5py") is None:return
    from tensorfem.result_db import write_result_db
    old=ResultDB(torch.tensor([0,1]),torch.tensor([0]),torch.tensor([[0,1]]),{"u":torch.zeros((2,1),dtype=D)})
    base=tmp_path/"legacy";write_result_db(base,old)
    assert isinstance(read_result_db_compatible(base),ResultDB)
    assert isinstance(read_result_db_compatible(base,migrate=True),ResultDBv2)


def test_invalid_frame_order_and_component_schema_fail_closed():
    db=plastic_db();bad=ResultDBv2(db.node_ids,db.element_ids,db.connectivity,db.field_specs,
        (ResultStep("bad",(ResultFrame(1,1.,1.,{}),ResultFrame(0,0.,0.,{}))),),{})
    with pytest.raises(ValueError,match="increase"):bad.validate()
    specs=dict(db.field_specs);specs["stress"]=FieldSpec("integration_point","Pa",("S11","S22"))
    bad=ResultDBv2(db.node_ids,db.element_ids,db.connectivity,specs,db.steps,db.histories)
    with pytest.raises(ValueError,match="components"):bad.validate()
