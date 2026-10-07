"""Report finite-rotation objectivity evidence for one cylindrical patch."""
import json, math
import torch
from tensorfem.corotational_shell import axis_angle, corotational_cylindrical_shell4, cylindrical_nodes

p=torch.tensor([[0.,-.2],[2.,-.2],[2.,.2],[0.,.2]],dtype=torch.float64)
X=cylindrical_nodes(p,5.); rows=[]
for degrees in (5,30,90,150,179):
    R=axis_angle(torch.tensor([.3,-.7,.2],dtype=torch.float64),math.radians(degrees))
    state=corotational_cylindrical_shell4(p,X@R.T+torch.tensor([13.,-8.,4.]),
                                           R.expand(4,3,3).clone(),5.,70e9,.25,.03)
    rows.append({"rotation_degrees":degrees,"strain_energy":float(state.energy),
                 "deformation_norm":float(torch.linalg.vector_norm(state.generalized_deformation))})
print(json.dumps(rows,indent=2))
