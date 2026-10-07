"""High-speed edge crossing followed by a Coulomb impact projection."""
import torch
from tensorfem.dynamic_contact_extended import (
    UnifiedEvent,coulomb_impact,edge_edge_ccd,
)
D=torch.float64
a0=torch.tensor([[-1.,0.,0.],[1.,0.,0.]],dtype=D); a1=a0.clone()
b0=torch.tensor([[0.,-1.,1.],[0.,1.,1.]],dtype=D); b1=b0.clone(); b1[:,2]=-1.
hit=edge_edge_ccd(a0,a1,b0,b1)
print("edge-edge TOI:",hit.toi)
event=UnifiedEvent("edge-edge",hit.toi,torch.tensor([0,1]),torch.tensor([.5,.5],dtype=D),
    torch.tensor([2,3]),torch.tensor([.5,.5],dtype=D),hit.normal)
velocity=torch.tensor([[1.,0.,0.],[1.,0.,0.],[0.,0.,-2.],[0.,0.,-2.]],dtype=D)
impact=coulomb_impact(event,velocity,torch.ones(4,dtype=D),friction=.3)
print("impulse balance:",impact.impulses.sum(0))
print("normal/friction impulse:",impact.normal_impulse,impact.friction_impulse)
print("kinetic energy before/after:",impact.kinetic_before,impact.kinetic_after)

