import math, numpy as np
from tensorfem.wave_dispersion import *
E,nu,rho,h=210e9,0.3,7850,0.01
cL=longitudinal_speed(E,nu,rho); cT=shear_speed(E,nu,rho)
cp=plate_speed(E,nu,rho); cR=rayleigh_speed(E,nu,rho)
print(f"cL={cL:.4f} cT={cT:.4f} cL/cT={cL/cT:.6f} cp={cp:.4f} cp/cT={cp/cT:.6f} cR={cR:.4f} cR/cT={cR/cT:.6f}")
for kh in (0.05,0.1,0.2,0.5,1.0,2.0,10.0,20.0):
    k=kh/h
    r=rayleigh_lamb_frequencies(k,h,E,nu,rho,n_roots=2)
    a0=r['antisymmetric'][0] if r['antisymmetric'] else float('nan')
    s0=r['symmetric'][0] if r['symmetric'] else float('nan')
    print(f"kh={kh:5.2f} A0 omega={a0:12.4f} c/cT={a0/k/cT:8.4f}   S0 omega={s0:12.4f} c/cT={s0/k/cT:8.4f}  flex_pred={flexural_phase_velocity(k,E,nu,rho,h)/cT:7.4f} cp/cT={cp/cT:6.4f}")
