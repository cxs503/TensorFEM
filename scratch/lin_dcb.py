import sys, numpy as np, scipy.sparse.linalg as spla
sys.path.insert(0,'src')
from tensorfem.cohesive3d_dcb import (DCBGeometry, build_dcb_model, beam_compliance, williams_root_rotation)
def run(nz, nx=160, ny=1, b=3e-3, a0=50e-3, L=80e-3, nzr=None):
    g = DCBGeometry(length=L, width=b, arm_thickness=3e-3, initial_crack=a0, nx=nx, ny=ny, nz=nz)
    m = build_dcb_model(g)
    K = m.solid_stiffness(); u=np.zeros(m.n_dofs)
    f0,Kc,_,_ = m.interface_response(u,None)
    free=m.free_dofs
    p=np.zeros(m.n_dofs)
    p[m.upper_load_dofs]=0.5/m.upper_load_dofs.size
    p[m.lower_load_dofs]=-0.5/m.lower_load_dofs.size
    u[free]=spla.spsolve((K+Kc)[free][:,free].tocsc(), p[free])
    delta=u[m.upper_load_dofs].mean()-u[m.lower_load_dofs].mean()
    c0=beam_compliance(a0,g,root_rotation=False); c1=beam_compliance(a0,g,root_rotation=True)
    print(f"nz={nz} ny={ny} nx={nx} b={b*1e3:.0f}mm a0={a0*1e3:.0f} C_fem={delta:.4e} C_noroot={c0:.4e} ({100*(delta/c0-1):+.2f}%) C_root={c1:.4e} ({100*(delta/c1-1):+.2f}%)")
for nz in (3,4,6):
    run(nz)
