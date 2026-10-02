import torch,time
from tensorfem.finite_strain_plasticity import MultiplicativeJ2State,update_multiplicative_j2
from tensorfem.implicit_plastic_tangent import implicit_material_tangent
D=torch.float64;M=(1000.,.3,35.,60.);state=MultiplicativeJ2State(torch.eye(3,dtype=D),torch.zeros((),dtype=D))
F0=torch.tensor([[1.04,.12,.01],[.02,.98,.03],[0.,.01,1.]],dtype=D);_,state,_,_=update_multiplicative_j2(F0,*M,state)
F=F0+.1*(torch.tensor([[1.08,.16,.02],[.03,.95,.04],[.01,.02,1.01]],dtype=D)-F0)
P,C,state,diss,seconds=implicit_material_tangent(F,*M,state)
print({'seconds':seconds,'finite':bool(torch.isfinite(C).all()),'detFp':float(torch.linalg.det(state.plastic_gradient)),'dissipation':float(diss)})
