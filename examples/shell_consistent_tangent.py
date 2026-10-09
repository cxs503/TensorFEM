"""Audit the corotational shell residual/tangent and Roof convergence."""
import json, torch
from tensorfem.shell_consistent import consistent_internal_force_tangent
from tensorfem.cylindrical_shell_benchmarks import scordelis_lo_cylindrical

p=torch.tensor([[0.,-.2],[2.,-.2],[2.,.2],[0.,.2]],dtype=torch.float64)
q=torch.zeros(24,dtype=torch.float64); q[6+2]=.001; q[12+2]=.001
r=consistent_internal_force_tangent(p,q,5.,70e9,.25,.03)
roof=[]
for n in (6,8,12):
    value=scordelis_lo_cylindrical(n,n).probe_displacement
    roof.append({"mesh":f"{n}x{n}","displacement":value,
                 "relative_error":abs(value/-0.3024-1.)})
print(json.dumps({"tangent_symmetry":float(torch.linalg.matrix_norm(r.tangent-r.tangent.T)/torch.linalg.matrix_norm(r.tangent)),
                  "roof":roof},indent=2))
