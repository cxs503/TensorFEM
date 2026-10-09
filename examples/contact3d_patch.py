"""Two slave nodes contacting adjacent QUAD4 facets."""
import torch
from tensorfem.contact3d import initial_contact_state, update_contact_nodes

D=torch.float64
vertices=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.],[0.,1.,0.],
                       [2.,0.,0.],[2.,1.,0.]],dtype=D)
faces=torch.tensor([[0,1,2,3],[1,4,5,2]],dtype=torch.long)
open_points=torch.tensor([[.25,.5,.1],[1.5,.5,.1]],dtype=D)
states=[initial_contact_state(p,vertices,faces) for p in open_points]
points=open_points.clone(); points[:,2]=-.001
updates,slave_forces,master_forces=update_contact_nodes(
    points,vertices,faces,states,normal_penalty=1e5,
    tangential_penalty=2e4,friction=.3,
    relative_increment=torch.tensor([.01,0.,0.],dtype=D),
)
print("slave forces:",slave_forces)
print("force imbalance:",slave_forces.sum(0)+master_forces.sum(0))
print("dissipated energy:",sum(float(u.dissipation_increment) for u in updates))
