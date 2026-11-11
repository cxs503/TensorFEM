import sys, numpy as np, scipy.sparse.linalg as spla
sys.path.insert(0,'src')
from tensorfem.cohesive3d_dcb import DCBGeometry, build_dcb_model, beam_compliance
def run(L, a0=50e-3, nz=3, b=3e-3, dx=0.5e-3):
    nx=int(round(L/dx))
    g=DCBGeometry(length=L,width=b,arm_thickness=3e-3,initial_crack=a0,nx=nx,ny=1,nz=nz)
    m=build_dcb_model(g); K=m.solid_stiffness(); u=np.zeros(m.n_dofs)
    _,Kc,_,_=m.interface_response(u,None); free=m.free_dofs
    p=np.zeros(m.n_dofs); p[m.upper_load_dofs]=0.5/m.upper_load_dofs.size; p[m.lower_load_dofs]=-0.5/m.lower_load_dofs.size
    u[free]=spla.spsolve((K+Kc)[free][:,free].tocsc(),p[free])
    d=u[m.upper_load_dofs].mean()-u[m.lower_load_dofs].mean()
    c0=beam_compliance(a0,g,root_rotation=False); c1=beam_compliance(a0,g,root_rotation=True)
    print(f"L={L*1e3:.0f} bonded={(L-a0)*1e3:.0f} nz={nz} C={d:.4e} vs noroot {100*(d/c0-1):+.2f}% vs root {100*(d/c1-1):+.2f}%")
for L in (80e-3,120e-3,160e-3):
    run(L)
for nz in (4,6):
    run(120e-3,nz=nz)
