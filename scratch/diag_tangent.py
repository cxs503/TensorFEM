import numpy as np, torch
from tensorfem.elastica_cantilever import _beam_energy, corotational_beam_cantilever

torch.set_num_threads(1)
L=10.0; E=1.2e6; b=1.0; t=0.1
I=b*t**3/12; EI=E*I; EA=E*b*t

def build(n):
    xs=np.linspace(0,L,n+1)
    nodes0=torch.tensor(np.stack([xs,np.zeros_like(xs)],axis=1),dtype=torch.float64)
    lengths=torch.full((n,),L/n,dtype=torch.float64)
    return nodes0,lengths

def grad(nodes0,lengths,q):
    q=q.detach().clone().requires_grad_(True)
    return torch.autograd.grad(_beam_energy(nodes0,q.reshape(-1,3),EA,EI,lengths),q)[0]

def ad_hess(nodes0,lengths,q):
    n=nodes0.shape[0]
    H=torch.autograd.functional.hessian(
        lambda qq:_beam_energy(nodes0,qq.reshape(-1,3),EA,EI,lengths), q.detach().clone())
    return H

def fd_hess(nodes0,lengths,q,h=1e-6):
    n=q.numel(); cols=[]
    for j in range(n):
        s=torch.zeros_like(q); s[j]=h
        cols.append((grad(nodes0,lengths,q+s)-grad(nodes0,lengths,q-s))/(2*h))
    K=torch.stack(cols,dim=1); return 0.5*(K+K.T)

for n in (2,4,8,16):
    nodes0,lengths=build(n)
    q=torch.zeros(3*(n+1),dtype=torch.float64)
    # slightly deformed state
    q2=q.clone(); q2[1::3]=torch.tensor(np.linspace(0,0.5,n+1)**2*0.02); q2[2::3]=torch.tensor(np.linspace(0,0.5,n+1)*0.01)
    for name,qq in (("straight",q),("bent",q2)):
        HA=ad_hess(nodes0,lengths,qq).numpy(); HF=fd_hess(nodes0,lengths,qq).numpy()
        eA=np.linalg.eigvalsh(HA); eF=np.linalg.eigvalsh(HF)
        scale=np.max(np.abs(HA))
        err=np.max(np.abs(HA-HF))/scale
        print(f"n={n:2d} {name:8s} ||K||={scale:.3e} cond={eA[-1]/max(abs(eA[0]),1e-300):.3e} "
              f"minEig_AD={eA[0]:.3e} minEig_FD={eF[0]:.3e} relDiff(AD vs FD)={err:.3e}")
