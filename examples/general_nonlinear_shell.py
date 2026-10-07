"""Run a two-facet doubly curved shell load path."""
import json,torch
from tensorfem.general_shell_nonlinear import GeneralShellMesh,solve_general_shell
phi=torch.tensor([.45,.75],dtype=torch.float64);theta=torch.tensor([0.,.35,.7],dtype=torch.float64);p,t=torch.meshgrid(phi,theta,indexing="ij")
x=5*torch.stack((torch.sin(p)*torch.cos(t),torch.sin(p)*torch.sin(t),torch.cos(p)),-1).reshape(-1,3)
mesh=GeneralShellMesh(x,torch.tensor([[0,3,4,1],[1,4,5,2]]),2e7,.25,.03)
f=torch.zeros(36,dtype=torch.float64);f[24:27]=-.05*x[4]/5
fixed={6*n+k:0. for n in (0,1,2) for k in range(6)};fixed.update({23:0.,29:0.,35:0.})
r=solve_general_shell(mesh,f,fixed,probe_dof=24,increment=.25,tolerance=1e-6)
print(json.dumps({"converged":r.converged,"path":r.path},indent=2))
