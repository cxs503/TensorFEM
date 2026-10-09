import math,torch
from tensorfem.nonproportional_plasticity import virgin_adaptive_state,integrate_path
D=torch.float64; shear=torch.tensor([[1.,.28,0.],[0.,1.,0.],[0.,0.,1.]],dtype=D)
a=.63; R=torch.tensor([[math.cos(a),-math.sin(a),0.],[math.sin(a),math.cos(a),0.],[0.,0.,1.]],dtype=D)
stretch=R@torch.diag(torch.tensor([1.22,1/1.22,1.],dtype=D))@R.T
for r in integrate_path((shear,stretch@shear,torch.eye(3,dtype=D)),virgin_adaptive_state(),1000.,.3,35.,60.,rtol=2e-6):
 print({"substeps":r.accepted_substeps,"error":r.estimated_error,"detFp":float(torch.linalg.det(r.state.plastic.plastic_gradient)),"dissipation":float(r.dissipation)})
