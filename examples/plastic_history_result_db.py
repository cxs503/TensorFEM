"""Write a multi-frame material-point load/unload ResultDB v2."""
import torch
from tensorfem.plasticity import Plastic1DState,update_bilinear_1d
from tensorfem.result_db_v2 import (
    FieldSpec,HistorySeries,ResultDBv2,ResultFrame,ResultStep,write_result_db_v2,
)
D=torch.float64;state=Plastic1DState(torch.tensor(0.,dtype=D),torch.tensor(0.,dtype=D))
frames=[];stresses=[];alphas=[]
for i,strain in enumerate((0.,.003,0.)):
    stress,_,state=update_bilinear_1d(torch.tensor(strain,dtype=D),200000.,250.,10000.,state)
    stresses.append(stress);alphas.append(state.alpha)
    fields={"node":{"displacement":torch.tensor([[0.],[strain]],dtype=D)},
            "integration_point":{"stress":stress.reshape(1,1,1),"alpha":state.alpha.reshape(1,1,1)}}
    frames.append(ResultFrame(i,float(i),float(strain/.003),fields))
time=torch.arange(3,dtype=D);factor=torch.tensor([0.,1.,0.],dtype=D)
histories={"stress":HistorySeries("Cycle",time,factor,torch.stack(stresses)[:,None],"Pa",("S11",)),
           "alpha":HistorySeries("Cycle",time,factor,torch.stack(alphas)[:,None],"1",("alpha",))}
db=ResultDBv2(torch.tensor([0,1]),torch.tensor([0]),torch.tensor([[0,1]]),
    {"displacement":FieldSpec("node","m",("U1",)),"stress":FieldSpec("integration_point","Pa",("S11",)),
     "alpha":FieldSpec("integration_point","1",("PEEQ",))},(ResultStep("Cycle",tuple(frames)),),histories)
available=write_result_db_v2("plastic-cycle",db,chunk_rows=1)
print("HDF5 available:",available)
print("database checksum:",db.checksums()["database"])
