"""Four simultaneous contact points solved as one complementarity set."""
import torch
from tensorfem.multi_contact_impulse import ContactConstraint,MultiContactState,solve_multi_contact_impulses

D=torch.float64
empty=torch.empty(0,dtype=torch.long); ew=torch.empty(0,dtype=D)
constraints=tuple(ContactConstraint(("ground",i),torch.tensor([i]),torch.ones(1,dtype=D),
    empty,ew,torch.tensor([0.,0.,1.],dtype=D)) for i in range(4))
velocity=torch.tensor([[1.,0.,-1.]]*4,dtype=D)
result=solve_multi_contact_impulses(velocity,torch.ones(4,dtype=D),constraints,
                                    MultiContactState(()),friction=.25)
print("nodal impulses:",result.impulses)
print("complementarity/cone residual:",result.complementarity_residual,result.cone_residual)
print("kinetic energy:",result.kinetic_before,result.kinetic_after)
print("iterations:",result.iterations)

