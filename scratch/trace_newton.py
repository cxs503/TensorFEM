import numpy as np, torch
from tensorfem.elastica_cantilever import _beam_energy
torch.set_num_threads(1)
L=10.0; E=1.2e6; b=1.0; t=0.1
EI=E*b*t**3/12; EA=E*b*t
alpha=1.0; n=8
xs=np.linspace(0,L,n+1)
nodes0=torch.tensor(np.stack([xs,np.zeros_like(xs)],axis=1),dtype=torch.float64)
lengths=torch.full((n,),L/n,dtype=torch.float64)
dofs=torch.zeros(3*(n+1),dtype=torch.float64); free=torch.arange(3,3*(n+1))
load=alpha*EI/L**2
force=torch.zeros_like(dofs); force[3*n+1]=-load
def gradient(qq):
    q=qq.detach().clone().requires_grad_(True)
    return torch.autograd.grad(_beam_energy(nodes0,q.reshape(-1,3),EA,EI,lengths),q)[0]
def tangent(q,h=1e-6):
    base=q.detach().clone(); cols=[]
    for j in range(base.numel()):
        s=torch.zeros_like(base); s[j]=h
        cols.append((gradient(base+s)-gradient(base-s))/(2*h))
    K=torch.stack(cols,dim=1); return 0.5*(K+K.T)
q=dofs.clone()
for it in range(1,16):
    res=gradient(q)-0.1*force
    r=float(torch.linalg.vector_norm(res[free]))
    print(f"it{it:2d} ||r_free||={r:.6e} ||grad_free||={float(torch.linalg.vector_norm(gradient(q)[free])):.3e}")
    if r<=1e-10: print("converged"); break
    K=tangent(q)[free[:,None],free]
    print("    cond(K_free)=%.3e  lam_min=%.3e"%(np.linalg.cond(K.numpy()),np.linalg.eigvalsh(K.numpy())[0]))
    d=torch.linalg.solve(K,-res[free])
    print("    ||d||=%.3e  max|d|=%.3e"%(float(torch.linalg.vector_norm(d)),float(torch.max(torch.abs(d)))))
    q=q.detach().clone(); q[free]+=d
