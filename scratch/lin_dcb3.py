import sys, numpy as np, scipy.sparse.linalg as spla, time
sys.path.insert(0,'src')
from tensorfem.cohesive3d_dcb import DCBGeometry, build_dcb_model, beam_compliance
def run(dx, nz, L=80e-3, a0=50e-3, b=3e-3):
    t=time.time(); nx=int(round(L/dx))
    g=DCBGeometry(length=L,width=b,arm_thickness=3e-3,initial_crack=a0,nx=nx,ny=1,nz=nz)
    m=build_dcb_model(g); K=m.solid_stiffness(); u=np.zeros(m.n_dofs)
    _,Kc,_,_=m.interface_response(u,None); free=m.free_dofs
    p=np.zeros(m.n_dofs); p[m.upper_load_dofs]=0.5/m.upper_load_dofs.size; p[m.lower_load_dofs]=-0.5/m.lower_load_dofs.size
    u[free]=spla.spsolve((K+Kc)[free][:,free].tocsc(),p[free])
    d=u[m.upper_load_dofs].mean()-u[m.lower_load_dofs].mean()
    c0=beam_compliance(a0,g,root_rotation=False); c1=beam_compliance(a0,g,root_rotation=True)
    deff=(d/c0)**(1/3)*a0-a0
    print(f"dx={dx*1e3:.2f} nz={nz} dofs={m.n_dofs} C={d:.5e} noroot {100*(d/c0-1):+.2f}% root {100*(d/c1-1):+.2f}% Delta_eff/h={deff/3e-3:.3f} t={time.time()-t:.1f}s")
for dx,nz in ((0.5e-3,3),(0.25e-3,3),(0.5e-3,6),(0.25e-3,6)):
    run(dx,nz)
