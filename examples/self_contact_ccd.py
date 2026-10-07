"""High-speed vertex/triangle impact clipped at its time of impact."""
import torch
from tensorfem.self_contact_ccd import DynamicSelfContactState,update_dynamic_self_contact

D=torch.float64
start=torch.tensor([[.25,.25,1.],[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]],dtype=D)
end=start.clone(); end[0,2]=-1.
faces=torch.tensor([[1,2,3]])
result=update_dynamic_self_contact(start,end,faces,DynamicSelfContactState(()),dt=.01)
print("TOI:",result.accepted_fraction)
print("impact point:",result.impact_positions[0])
print("impulse balance:",result.impulses.sum(0))
print("moment balance:",torch.linalg.cross(result.impact_positions,result.impulses).sum(0))
print("trial pair history:",result.state.pairs)

