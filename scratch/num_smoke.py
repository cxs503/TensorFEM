import sys, time, numpy as np, scipy.sparse.linalg as spla, torch
sys.path.insert(0,'src'); torch.set_num_threads(1)
from tensorfem.cohesive3d import BKCohesive3DLaw
from tensorfem.cohesive3d_dcb import *
law=lambda: BKCohesive3DLaw(2.4e14,1.0e14,120e6,80e6,300.0,800.0,2.2)
print("== linear compliance bracket (a0=60mm, L=120mm, b=h=3mm) ==", flush=True)
for nz in (2,3):
    g=DCBGeometry(length=120e-3,width=3e-3,arm_thickness=3e-3,initial_crack=60e-3,nx=200,ny=1,nz=nz)
    m=build_dcb_model(g,law()); K=m.solid_stiffness(); u=np.zeros(m.n_dofs)
    _,Kc,_,_=m.interface_response(u,None); free=m.free_dofs
    p=np.zeros(m.n_dofs); p[m.upper_load_dofs]=0.5/m.upper_load_dofs.size; p[m.lower_load_dofs]=-0.5/m.lower_load_dofs.size
    u[free]=spla.spsolve((K+Kc)[free][:,free].tocsc(),p[free])
    c=u[m.upper_load_dofs].mean()-u[m.lower_load_dofs].mean()
    cp=beam_compliance(g.initial_crack,g,root_rotation=False); cr=beam_compliance(g.initial_crack,g,root_rotation=True)
    print(f"nz={nz} C_fem={c:.6e} vs plain {100*(c/cp-1):+.2f}% vs root {100*(c/cr-1):+.2f}%  (Delta={1000*williams_root_rotation(g.arm_thickness,g.young_modulus,g.shear_modulus):.2f}mm)", flush=True)
print("== propagation (L=40mm, a0=20mm, nx=80, nz=2) ==", flush=True)
g=DCBGeometry(length=40e-3,width=3e-3,arm_thickness=3e-3,initial_crack=20e-3,nx=80,ny=1,nz=2)
m=build_dcb_model(g,law()); t=time.time()
r=solve_dcb(m,delta_max=3.0e-5,n_steps=60,tol=1e-8)
pr=beam_peak_load(g.initial_crack,300.0,g); pp=beam_peak_load(g.initial_crack,300.0,g,root_rotation=False)
print(f"steps={r.stats['steps']} conv={r.stats['converged_all']} t={time.time()-t:.1f}s peak={r.peak['load']:.3f}N LEFM(root)={pr:.3f} err={100*(r.peak['load']/pr-1):+.2f}% LEFM(plain)={pp:.3f} err={100*(r.peak['load']/pp-1):+.2f}%", flush=True)
print(f"energy_balance_rel={100*r.stats['energy_balance_rel']:.3f}%  W_coh/failed_area={r.stats['interface_work_per_area']:.1f} J/m2 (G_Ic=300) failed_area={r.stats['final_failed_area']:.6f} m2 Dmax={r.damage.max():.3f}", flush=True)
rc=[(c['crack_front_m'],c['load']) for c in r.curve if c['crack_front_m']>g.initial_crack+1e-12]
if rc:
    errs=[abs(l/beam_peak_load(a,300.0,g)-1) for a,l in rc]
    print(f"R-curve steps={len(rc)} max|P/P_LEFM(front)-1|={100*max(errs):.2f}%", flush=True)
print("== DONE ==", flush=True)
