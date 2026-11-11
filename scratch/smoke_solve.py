import sys, time, numpy as np
sys.path.insert(0,'src')
from tensorfem.cohesive3d import BKCohesive3DLaw
from tensorfem.cohesive3d_dcb import DCBGeometry, build_dcb_model, solve_dcb, beam_peak_load, beam_compliance
g=DCBGeometry(length=60e-3,width=3e-3,arm_thickness=3e-3,initial_crack=30e-3,nx=120,ny=1,nz=3)
law=BKCohesive3DLaw(2.4e14,1.0e14,120e6,80e6,300.0,800.0,2.2)
m=build_dcb_model(g,law)
print("dofs",m.n_dofs,"iface",m.n_iface,"P_ref",beam_peak_load(g.initial_crack,300.0,g),"C",beam_compliance(g.initial_crack,g))
t=time.time()
r=solve_dcb(m, delta_max=8e-6, n_steps=20, verbose=True, tol=1e-8)
print("time",time.time()-t, r.stats)
for c in r.curve[::4]: print({k:(round(v,10) if isinstance(v,float) else v) for k,v in c.items()})
