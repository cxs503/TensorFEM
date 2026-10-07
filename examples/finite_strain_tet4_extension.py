"""Finite-strain elastic material response at 50% extension and large rotation."""
import math,torch
from tensorfem.finite_strain_elasticity import neo_hookean_response
D=torch.float64
F=torch.diag(torch.tensor([1.5,1.,1.],dtype=D)); energy,P,sigma=neo_hookean_response(F,1000.,.3)
a=1.2; R=torch.tensor([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]],dtype=D)
rotated=neo_hookean_response(R@F,1000.,.3)
print({"J":float(torch.linalg.det(F)),"energy":float(energy),"P11":float(P[0,0]),
       "objectivity_energy_error":abs(float(rotated[0]-energy))})
