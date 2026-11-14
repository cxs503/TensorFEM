import numpy as np, torch
from tensorfem.elastica_cantilever import _beam_energy
torch.set_num_threads(1)
L=10.0; E=1.2e6; b=1.0; t=0.1
I=b*t**3/12; EI=E*I; EA=E*b*t
for n in (2,8):
    xs=np.linspace(0,L,n+1)
    nodes0=torch.tensor(np.stack([xs,np.zeros_like(xs)],axis=1),dtype=torch.float64)
    lengths=torch.full((n,),L/n,dtype=torch.float64)
    q=torch.zeros(3*(n+1),dtype=torch.float64)
    H=torch.autograd.functional.hessian(lambda qq:_beam_energy(nodes0,qq.reshape(-1,3),EA,EI,lengths),q).numpy()
    free=np.arange(3,3*(n+1))
    Hf=H[np.ix_(free,free)]
    w,V=np.linalg.eigh(Hf)
    print(f"n={n} free dofs={len(free)} lam_max={w[-1]:.3e} lam_min={w[0]:.3e} ratio={w[0]/w[-1]:.2e}")
    print("  5 smallest:", np.array2string(w[:5],precision=3))
    v=V[:,0].reshape(-1,3)
    print("  null mode (per node u,w,theta):")
    for k in range(v.shape[0]):
        print(f"    node{k+1}: {v[k,0]: .3e} {v[k,1]: .3e} {v[k,2]: .3e}")
    # energy along that mode
    for s in (1e-2,1e-1,1.0):
        qq=q.clone(); qq[free]+=torch.tensor(s*V[:,0])
        e=float(_beam_energy(nodes0,qq.reshape(-1,3),EA,EI,lengths))
        print(f"   E(s={s})={e:.6e}  E/s^2={e/s**2:.6e}")
