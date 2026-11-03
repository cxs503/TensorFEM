import sys, math; sys.path.insert(0,'src')
import torch as T
from tensorfem.hetenyi_reference import finite_beam_bvp, foundation_reaction_integral
from tensorfem.modal import beam_bending_stiffness
D=T.float64
EI=604.8; k_b=4.0e5; lam=(k_b/(4*EI))**0.25; Lc=1/lam
P=1000.0
for nL in [0.5, 1.0, 2.0]:
    L=nL*Lc
    x=T.linspace(0,L,201,dtype=D)
    s=finite_beam_bvp(x,beam_length=L,lam=lam,k_b=k_b,EI=EI,point_loads=[(L/2,P)],end_a="free",end_b="free")
    tot=foundation_reaction_integral(s)
    print(f"lamL={nL}: w_mid={float(s.deflection.max()):.6e} eq={tot/P:.6f} V0={float(s.shear[0]):.3e} M0={float(s.moment[0]):.3e} slope_mid={float(s.slope[100]):.3e}")

# FE free-free with consistent Winkler matrix
def fe_free(L, nel, EI, k_b, P):
    le=L/nel; ndof=2*(nel+1)
    K=T.zeros((ndof,ndof),dtype=D)
    kew=T.tensor([[156.,22*le,54.,-13*le],[22*le,4*le**2,13*le,-3*le**2],[54.,13*le,156.,-22*le],[-13*le,-3*le**2,-22*le,4*le**2]],dtype=D)*k_b*le/420.
    f=T.zeros(ndof,dtype=D)
    for e in range(nel):
        K[2*e:2*e+4,2*e:2*e+4]+=beam_bending_stiffness(EI,le)+kew
    mid=nel//2
    f[2*mid]=P
    u=T.linalg.solve(K,f)
    return u[2*mid]
L=1.0*Lc
for nel in [8,16,32,64]:
    print(nel, fe_free(L,nel,EI,k_b,P))
x=T.linspace(0,L,201,dtype=D)
s=finite_beam_bvp(x,beam_length=L,lam=lam,k_b=k_b,EI=EI,point_loads=[(L/2,P)])
print("exact", float(s.deflection[100]))
