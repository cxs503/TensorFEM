"""Vectorised 1024-constraint ground-contact color batch."""
import time,torch
from tensorfem.colored_contact_graph import prepare_colored_contacts,solve_colored_contacts
from tensorfem.rigid_contact_graph import RigidContact
D=torch.float64; n=1024
com=torch.zeros((n,3),dtype=D); com[:,0]=torch.arange(n,dtype=D)*2
mass=torch.ones(n,dtype=D); inertia=torch.eye(3,dtype=D).repeat(n,1,1)
velocity=torch.zeros((n,3),dtype=D); velocity[:,2]=-1.; angular=torch.zeros_like(velocity)
contacts=tuple(RigidContact(("ground",i),i,-1,com[i]-torch.tensor([0.,0.,1.],dtype=D),
                            torch.tensor([0.,0.,1.],dtype=D)) for i in range(n))
data=prepare_colored_contacts(n,contacts,dtype=D)
t=time.perf_counter(); result=solve_colored_contacts(com,mass,inertia,velocity,angular,data); elapsed=time.perf_counter()-t
print("constraints/colors/islands:",len(contacts),len(data.colors),len(data.islands))
print("iterations/residual:",result.iterations,result.complementarity_residual)
print("elapsed seconds:",elapsed)

