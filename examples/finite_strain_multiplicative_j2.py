import math,torch
from tensorfem.finite_strain_plasticity import MultiplicativeJ2State,update_multiplicative_j2
D=torch.float64; e=.18; F=torch.diag(torch.tensor([math.exp(e),math.exp(-e/2),math.exp(-e/2)],dtype=D))
state=MultiplicativeJ2State(torch.eye(3,dtype=D),torch.zeros((),dtype=D))
P,state,diss,y=update_multiplicative_j2(F,1000.,.3,40.,80.,state)
print({"detFp":float(torch.linalg.det(state.plastic_gradient)),"alpha":float(state.alpha),
       "dissipation":float(diss),"yield_residual":float(y),"P11":float(P[0,0])})
