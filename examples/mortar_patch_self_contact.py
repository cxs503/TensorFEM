"""Nonmatching mortar patch and folded-surface candidate search."""
import torch
from tensorfem.mortar_contact3d import integrate_mortar_contact,self_contact_candidates

D=torch.float64
slave=torch.tensor([[0.,0.,-.001],[1.,0.,-.001],[1.,1.,-.001],[0.,1.,-.001]],dtype=D)
sf=torch.tensor([[0,1,2,3]],dtype=torch.long)
master=torch.tensor([[-.2,-.2,0.],[.5,-.2,0.],[1.2,-.2,0.],
                     [-.2,1.2,0.],[.5,1.2,0.],[1.2,1.2,0.]],dtype=D)
mf=torch.tensor([[0,1,4,3],[1,2,5,4]],dtype=torch.long)
r=integrate_mortar_contact(slave,sf,master,mf,normal_penalty=2e5)
print("slave resultant:",r.slave_forces.sum(0))
print("balance:",r.slave_forces.sum(0)+r.master_forces.sum(0))

fold=torch.tensor([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.],
                   [.05,0.,.02],[.05,1.,.02]],dtype=D)
faces=torch.tensor([[0,2,1],[1,2,3],[2,4,3],[3,4,5]])
print("nonadjacent near pairs:",self_contact_candidates(fold,faces,search_distance=.08))

