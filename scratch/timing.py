import time, numpy as np, sys
sys.path.insert(0,'src')
from tensorfem.composite_laminate_bending import *
mat = PAGANO_MATERIAL
lam = Laminate([0,90,90,0], 0.5/4, mat)
for nx,nzp in ((4,2),(6,3),(8,4)):
    t0=time.time(); r=layered_hex20_fem(lam, a=1.0, q0=1.0, nx=nx, nz_per_ply=nzp)
    print("hex (%d,%d) %.1fs dofs=%d w=%.6e" % (nx,nzp,time.time()-t0,r["n_dofs"],r["w_center"]), flush=True)
lam_thin = Laminate([0,90,90,0], 0.01/4, mat)
t0=time.time(); r=layered_hex20_fem(lam_thin, a=1.0, q0=1.0, nx=4, nz_per_ply=2)
print("thin hex (4,2) %.1fs dofs=%d w=%.6e" % (time.time()-t0,r["n_dofs"],r["w_center"]), flush=True)
t0=time.time(); rp=plate_fem(Laminate(symmetric_layup([0,45,-45,90]),0.5/8,mat), 1.0, 1.0, n=32)
print("plate n=32 %.1fs w=%.6e"%(time.time()-t0, rp["w_center"]), flush=True)
