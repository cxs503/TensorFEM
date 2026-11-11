import sys, time, numpy as np, scipy.sparse.linalg as spla
sys.path.insert(0,'src')
from tensorfem.cohesive3d_dcb import (DCBGeometry, build_dcb_model, beam_compliance,
                                      beam_peak_load, williams_root_rotation, solve_dcb)
g = DCBGeometry(length=20e-3, width=3e-3, arm_thickness=3e-3, initial_crack=12e-3,
                nx=40, ny=1, nz=3)
m = build_dcb_model(g)
print("dofs", m.n_dofs, "iface", m.n_iface, "bonded cols", g.n_bonded, "Delta", williams_root_rotation(g.arm_thickness, g.young_modulus, g.shear_modulus))
# linear compliance check
K = m.solid_stiffness()
u = np.zeros(m.n_dofs)
fake = np.zeros(m.n_dofs); fake[m.upper_load_dofs]=0.5; fake[m.lower_load_dofs]=-1.0/ (m.lower_load_dofs.size) *0.5*2
free = m.free_dofs
# assemble cohesive at u=0 (linear)
f0,Kc,hist,diag = m.interface_response(u, None)
print("interface force at u=0 max", np.abs(f0).max())
Kt = (K+Kc)
del_ = 1e-9*np.ones(1)
rhs = np.zeros(m.n_dofs)
p = np.zeros(m.n_dofs)
p[m.upper_load_dofs]=0.5/m.upper_load_dofs.size
p[m.lower_load_dofs]=-0.5/m.lower_load_dofs.size
u[free] = spla.spsolve(Kt[free][:,free].tocsc(), p[free])
delta = u[m.upper_load_dofs].mean() - u[m.lower_load_dofs].mean()
print("delta/P", delta, "beam", beam_compliance(g.initial_crack, g), "rel", (delta/beam_compliance(g.initial_crack,g)-1)*100, "%")
