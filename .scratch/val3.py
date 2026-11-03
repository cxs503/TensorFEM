import sys, math; sys.path.insert(0,'src')
import torch as T
from tensorfem.hetenyi_reference import (finite_beam_bvp, finite_beam_series,
    infinite_beam_point_load, foundation_reaction_integral)
D=T.float64
EI=604.8; k_b=4.0e5; lam=(k_b/(4*EI))**0.25; Lc=1/lam; P=1000.0

# SS with point load vs Fourier series
L=1.0*Lc
x=T.linspace(1e-4, L-1e-4, 81, dtype=D)
s=finite_beam_bvp(x,beam_length=L,lam=lam,k_b=k_b,EI=EI,point_loads=[(L/2,P)],end_a="simply_supported",end_b="simply_supported")
ser=finite_beam_series(x,L,lam,k_b,EI,[(L/2,P)],boundary="simply_supported",n_terms=400)
print("SS vs series: max rel", float(((s.deflection-ser)/ser).abs().max()), "eq", foundation_reaction_integral(s)/P)

# free-free residual check on fine grid
for nL in [1.0, 2.0, 4.0, 6.0]:
    L=nL*Lc
    x=T.linspace(0,L,2001,dtype=D)
    s=finite_beam_bvp(x,beam_length=L,lam=lam,k_b=k_b,EI=EI,point_loads=[(L/2,P)])
    w=s.deflection
    h=float(x[1]-x[0])
    # 4th derivative central diff (2nd order accurate)
    d4=(w[4:]-4*w[3:-1]+6*w[2:-2]-4*w[1:-3]+w[:-4])/h**4
    resid=(EI*d4 + k_b*w[2:-2])
    i0=len(x)//2
    # away from load: residual should be 0
    mask=T.ones(len(resid),dtype=T.bool); mask[i0-6:i0+6]=False
    print(f"lamL={nL}: eq={foundation_reaction_integral(s)/P:.6f} resid/rms(k w)={float((resid[mask]).abs().max()/ (k_b*w.abs().max())):.3e} wmid/inf={float(w[i0]/ (P*lam/(2*k_b))):.5f}")
