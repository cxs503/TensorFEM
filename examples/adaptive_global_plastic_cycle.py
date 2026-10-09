import torch
from tensorfem.adaptive_global_plasticity import solve_adaptive_global_path
from tensorfem.finite_strain_elasticity import FiniteStrainTet4Model
D=torch.float64; x=torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
m=FiniteStrainTet4Model(x,torch.tensor([[0,1,2,3]]),1000.,.3,torch.tensor([0,1,2,4,5,6,7,8,9,10,11]))
f=torch.zeros(12,dtype=D); f[3]=16.
for r in solve_adaptive_global_path(m,f,(.7,1.,-.35,.55,0.),yield_stress=35.,hardening=60.,material_rtol=3e-5):
 print({"factor":r.load_factor,"u":float(r.displacement[3]),"dissipation":float(r.increment_dissipation.sum()),"tangent_evaluations":r.tangent_material_evaluations})
