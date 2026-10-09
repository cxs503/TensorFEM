"""Global spring/contact equilibrium while a node slides across segments."""
import torch

from tensorfem.global_contact2d import GlobalContactModel,NodePolylinePair,solve_global_contact_path

D=torch.float64
x=torch.tensor([[0.,0.],[.5,0.],[1.5,0.],[.2,.05]],dtype=D)
k=torch.zeros((8,8),dtype=D);k[6,6]=k[7,7]=1000.
m=GlobalContactModel(x,k,(NodePolylinePair(3,(0,1,2)),),tuple(range(6)),1e4,1e3,.2)
loads=[]
for fx in (0.,200.,500.,900.,1200.):
    f=torch.zeros(8,dtype=D);f[6:]=torch.tensor([fx,-100.],dtype=D);loads.append(f)
for step in solve_global_contact_path(m,loads):
    h=step.state.histories[0]
    print(float(step.external[6]),step.displacement[6:].tolist(),h.segment,h.sticking,
          float(h.dissipated_energy))
