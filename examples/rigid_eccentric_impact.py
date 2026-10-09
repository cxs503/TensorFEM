"""Eccentric frictional impact of two rigid bodies."""
import torch
from tensorfem.rigid_contact_graph import RigidContact,RigidContactState,solve_rigid_contact_graph
D=torch.float64
com=torch.tensor([[-1.,.5,0.],[1.,-.5,0.]],dtype=D)
mass=torch.tensor([2.,3.],dtype=D)
inertia=torch.stack((torch.diag(torch.tensor([.4,.5,.6],dtype=D)),
                     torch.diag(torch.tensor([.7,.8,.9],dtype=D))))
linear=torch.tensor([[2.,1.,0.],[0.,0.,0.]],dtype=D); angular=torch.zeros((2,3),dtype=D)
contact=(RigidContact(("pair",0),0,1,torch.zeros(3,dtype=D),torch.tensor([-1.,0.,0.],dtype=D)),)
r=solve_rigid_contact_graph(com,mass,inertia,linear,angular,contact,RigidContactState(()),friction=.3)
print("linear velocity:",r.linear_velocity)
print("angular velocity:",r.angular_velocity)
print("normal/friction impulse:",r.state.histories[0])
print("complementarity/cone:",r.complementarity_residual,r.cone_residual)
print("energy:",r.kinetic_before,r.kinetic_after)

