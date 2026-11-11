import sys, time, numpy as np, torch, scipy.sparse.linalg as spla
sys.path.insert(0,'src')
torch.set_num_threads(1)
from tensorfem.cohesive3d import BKCohesive3DLaw, cohesive_hex8_response
from tensorfem.cohesive3d_dcb import *
g=DCBGeometry(length=60e-3,width=3e-3,arm_thickness=3e-3,initial_crack=30e-3,nx=120,ny=1,nz=3)
law=BKCohesive3DLaw(2.4e14,1.0e14,120e6,80e6,300.0,800.0,2.2)
m=build_dcb_model(g,law)
print("dofs",m.n_dofs,"iface",m.n_iface,"P_ref",round(beam_peak_load(g.initial_crack,300.0,g),3))
# equivalence check on 3 elements with random displacement
u=np.zeros(m.n_dofs); rng=np.random.default_rng(0)
u=rng.normal(scale=1e-7,size=m.n_dofs)
f,K,hn,diag=m.interface_response(u,None)
err_f=0.; err_k=0.
for e in range(min(3,m.n_iface)):
    conn=m.iface[e]; fe,ke,_,_=cohesive_hex8_response(m.nodes[conn], torch.as_tensor(u,dtype=torch.float64).reshape(m.n_nodes,3)[conn], law)
    dofs=(3*conn.numpy()[:,None]+np.arange(3)).reshape(-1)
    err_f=max(err_f,np.abs(f[dofs]-fe.numpy().reshape(-1)).max()/max(1e-30,np.abs(fe.numpy()).max()))
    err_k=max(err_k,np.abs(K[dofs][:,dofs].toarray()-ke.numpy()).max()/np.abs(ke.numpy()).max())
print("vectorized==element rel err force",err_f,"tangent",err_k)
t=time.time(); r=solve_dcb(m,delta_max=2.5e-5,n_steps=40,verbose=False,tol=1e-8); print("solve time",round(time.time()-t,1))
print("peak",round(r.peak["load"],3),"at disp",r.peak["disp"],"P_Lefm",round(beam_peak_load(g.initial_crack,300.0,g),3),
      "err%",round(100*(r.peak["load"]/beam_peak_load(g.initial_crack,300.0,g)-1),2))
print({k:(round(v,8) if isinstance(v,float) else v) for k,v in r.stats.items()})
print("Dmax",r.damage.max(),"failed area",r.stats["final_failed_area"],"W/A",r.stats["interface_work_per_area"])
