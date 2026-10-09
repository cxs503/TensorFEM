"""Run the auditable two-element cylindrical strip analysis."""
import json, torch
from tensorfem.shell_nonlinear_step import CylindricalShellMesh, solve_shell_step

p=torch.tensor([(x,t) for x in (0.,1.,2.) for t in (-.1,.1)],dtype=torch.float64)
mesh=CylindricalShellMesh(p,torch.tensor([[0,2,3,1],[2,4,5,3]]),5.,2e8,0.,.02)
loads=torch.zeros(36,dtype=torch.float64); loads[24]=loads[30]=500.
fixed={6*n+c:0. for n in range(6) for c in range(1,6)}; fixed.update({0:0.,6:0.})
r=solve_shell_step(mesh,loads,fixed,initial_increment=.5)
exact=1000.*2./(2e8*.02*5.*.2)
print(json.dumps({"converged":r.converged,"increments":len(r.increments),
                  "tip_displacement":float(r.dofs.reshape(-1,6)[4:,0].mean()),
                  "reference":exact,"relative_error":abs(float(r.dofs.reshape(-1,6)[4:,0].mean())/exact-1.)},indent=2))
