"""Coulomb mortar sliding and folded-strip self-contact."""
import torch
from tensorfem.frictional_mortar import (
    initial_frictional_mortar_state,update_frictional_mortar,update_self_contact,
)
D=torch.float64
slave=torch.tensor([[0.,0.,-.001],[1.,0.,-.001],[1.,1.,-.001],[0.,1.,-.001]],dtype=D)
faces=torch.tensor([[0,1,2,3]])
master=slave.clone(); master[:,2]=0.
state=initial_frictional_mortar_state(slave,faces,master,faces)
inc=torch.zeros_like(slave); inc[:,0]=.01
r=update_frictional_mortar(slave,faces,master,faces,state,normal_penalty=1e5,
    tangential_penalty=1e4,friction=.3,relative_increments=inc)
print("normal/tangential:",r.normal_resultant,r.tangential_resultant)
print("dissipation:",r.dissipation_increment)

fold=torch.tensor([[0.,0.,0.],[0.,1.,0.],[1.,0.,0.],[1.,1.,0.],
                   [.05,0.,.02],[.05,1.,.02]],dtype=D)
ff=torch.tensor([[0,2,1],[1,2,3],[2,4,3],[3,4,5]])
s=update_self_contact(fold,ff,clearance=.08,normal_penalty=1e4)
print("active self-contact pairs:",s.active_pairs)
print("force/moment balance:",s.forces.sum(0),torch.linalg.cross(fold,s.forces).sum(0))

